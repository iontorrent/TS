/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <cassert>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "RegionWellsReader.h"
#include "Utils.h"
#include "IonErr.h"
#include "dbgmem.h"
#include "RawWells.h"


RegionWellsReader::RegionWellsReader()
{
  wells = NULL;
  sizeX = 0;
  sizeY = 0;
  numRegionsX = 0;
  numRegionsY = 0;
  sizeRegionX = 0;
  sizeRegionY = 0;
  nextRegionX = 0;
  nextRegionY = 0;
  numFlows = 0;

  pthread_mutex_t tmp_mutex = PTHREAD_MUTEX_INITIALIZER;
  read_mutex = new pthread_mutex_t(tmp_mutex);
}

RegionWellsReader::~RegionWellsReader()
{
  Close();
  delete read_mutex;
}

bool RegionWellsReader::OpenForRead(RawWells *_wells, int _sizeX, int _sizeY, int _numRegionsX, int _numRegionsY)
{
  Close();

  wells = _wells;
  sizeX = _sizeX;
  sizeY = _sizeY;
  numRegionsX = _numRegionsX;
  numRegionsY = _numRegionsY;
  sizeRegionX = (sizeX + numRegionsX - 1) / numRegionsX;
  sizeRegionY = (sizeY + numRegionsY - 1) / numRegionsY;
  nextRegionX = 0;
  nextRegionY = 0;
  numFlows = wells->NumFlows();
  return true;
}

bool RegionWellsReader::OpenForRead2(RawWells *_wells, int _sizeX, int _sizeY, int _sizeRegionX, int _sizeRegionY)
{
  Close();

  wells = _wells;
  sizeX = _sizeX;
  sizeY = _sizeY;
  sizeRegionX = _sizeRegionX;
  sizeRegionY = _sizeRegionY;
  numRegionsX = (sizeX + sizeRegionX - 1) / sizeRegionX;
  numRegionsY = (sizeY + sizeRegionY - 1) / sizeRegionY;
  nextRegionX = 0;
  nextRegionY = 0;
  numFlows = wells->NumFlows();
  return true;
}


void RegionWellsReader::Close()
{
}



bool RegionWellsReader::loadNextRegion(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements, int &iRegion)
{
  wellX.clear();
  wellY.clear();
  wellMeasurements.clear();

  pthread_mutex_lock(read_mutex);
  int Y = std::min(nextRegionY*sizeRegionY,(int)wells->NumRows());
  int X = std::min(nextRegionX*sizeRegionX,(int)wells->NumCols());
  int nY = ((nextRegionY+1)*sizeRegionY);
  int nX = ((nextRegionX+1)*sizeRegionX);
  wells->SetChunk(Y, std::min(nY-Y,(int)wells->NumRows() - Y),
                  X, std::min(nX-X,(int)wells->NumCols() - X),
                  0, wells->NumFlows());
  wells->ReadWells();
  if (nextRegionX >= numRegionsX) {
    pthread_mutex_unlock(read_mutex);
    return false; // All regions read already
  }
  iRegion = nextRegionY + numRegionsY * nextRegionX;
  for (int nextY = nextRegionY * sizeRegionY; (nextY < (nextRegionY + 1) * sizeRegionY) && (nextY < sizeY); nextY++) {
   for (int nextX = nextRegionX * sizeRegionX; (nextX < (nextRegionX + 1) * sizeRegionX) && (nextX < sizeX); nextX++) {
      wellX.push_back(nextX);
      wellY.push_back(nextY);
      wellMeasurements.push_back(std::vector<float>());
      wellMeasurements.back().resize(numFlows);
      const WellData *w = wells->ReadXY(nextX, nextY);
      copy(w->flowValues, w->flowValues + numFlows, wellMeasurements.back().begin());
    }
  }

  nextRegionY++;
  if (nextRegionY == numRegionsY) {
    nextRegionY = 0;
    nextRegionX++;
  }

  pthread_mutex_unlock(read_mutex);

  return true; // Region reading successful

}





bool RegionWellsReader::loadRegion(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements,
    int regionX, int regionY, Mask *mask)
{
  if ((regionX >= numRegionsX) || (regionY >= numRegionsY))
    return false;

  wellX.clear();
  wellY.clear();
  wellMeasurements.clear();
  pthread_mutex_lock(read_mutex);
  int Y = std::min(regionY*sizeRegionY,(int)wells->NumRows());
  int X = std::min(regionX*sizeRegionX,(int)wells->NumCols());
  int nY = ((regionY+1)*sizeRegionY);
  int nX = ((regionX+1)*sizeRegionX);
  wells->SetChunk(Y, std::min(nY-Y,(int)wells->NumRows() - Y),
                  X, std::min(nX-X,(int)wells->NumCols() - X),
                  0, wells->NumFlows());
  wells->ReadWells();
  for (int nextY = regionY * sizeRegionY; (nextY < (regionY + 1) * sizeRegionY) && (nextY < sizeY); nextY++) {
    for (int nextX = regionX * sizeRegionX; (nextX < (regionX + 1) * sizeRegionX) && (nextX < sizeX); nextX++) {

      if (!mask->Match(nextX, nextY, MaskTF) && !mask->Match(nextX, nextY, MaskLib))
        continue;

      wellX.push_back(nextX);
      wellY.push_back(nextY);
      wellMeasurements.push_back(std::vector<float>());
      wellMeasurements.back().resize(numFlows);


      const WellData *w = wells->ReadXY(nextX, nextY);
      copy(w->flowValues, w->flowValues + numFlows, wellMeasurements.back().begin());

    }
  }
  pthread_mutex_unlock(read_mutex);
  return true; // Region reading successful

}



