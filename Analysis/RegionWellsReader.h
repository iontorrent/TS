/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONWELLSREADER_H
#define REGIONWELLSREADER_H

#include <deque>
#include <vector>
#include <string>
#include <cstdio>
#include "Mask.h"


// RegionWellsReader - a simple, thread-safe reader for .wells file that fetches one region at a time
class RawWells;

class RegionWellsReader {
public:
  RegionWellsReader();
  ~RegionWellsReader();

  bool OpenForRead(RawWells *_wells, int _sizeX, int _sizeY, int _numRegionsX, int _numRegionsY);

  bool OpenForRead2(RawWells *_wells, int _sizeX, int _sizeY, int _sizeRegionX, int _sizeRegionY);

  bool loadNextRegion(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements, int &iRegion);

  bool loadRegion(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements,
      int regionX, int regionY, Mask *mask);

  void Close();


private:

  int sizeX;
  int sizeY;
public:
  int numRegionsX;
  int numRegionsY;
private:
  int sizeRegionX;
  int sizeRegionY;

  unsigned short numFlows;

  int nextRegionX;
  int nextRegionY;

  RawWells *wells;
  pthread_mutex_t *read_mutex;

};

#endif // REGIONWELLSREADER_H 
