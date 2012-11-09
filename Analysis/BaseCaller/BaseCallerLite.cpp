/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <vector>

#include <stdio.h>

#include "Mask.h"
#include "RawWells.h"
#include "OrderedRegionSFFWriter.h"
#include "IonErr.h"
#include "DPTreephaser.h"
#include "OptArgs.h"
#include "Utils.h"

using namespace std;



struct BaseCallerLite {

  RawWells                *wellsPtr;
  Mask                    *maskPtr;
  string                  runId;
  vector<int>             libKeyFlows;
  int                     libNumKeyFlows;
  double                  CF, IE;
  int                     regionXSize, regionYSize;
  int                     rows, cols;
  int                     numRegions;
  int                     numRegionsX, numRegionsY;
  int                     numFlows;
  ion::FlowOrder          flowOrder;
  pthread_mutex_t         wellsAccessMutex;
  int                     nextRegionX;
  int                     nextRegionY;
  unsigned int            numWellsCalled;
  OrderedRegionSFFWriter  libSFF;

  void BasecallerWorker();
};

static void *BasecallerWorkerWrapper(void *input);




int main (int argc, const char *argv[])
{

  if (argc == 1) {
    printf ("BaseCallerLite - Bare bone basecaller\n");
    printf ("\n");
    printf ("Usage:\n");
    printf ("BaseCallerLite [options]\n");
    printf ("\tOptions:\n");
    printf ("\t\tComing soon\n");
    printf ("\n");
    return 1;
  }

  string libKey = "TCAG";
  string inputDirectory = ".";
  string outputDirectory = ".";
  bool singleCoreCafie = false;

  BaseCallerLite basecaller;
  basecaller.regionXSize = 50;
  basecaller.regionYSize = 50;
  basecaller.runId = "BCLTE";
  basecaller.CF = 0.0;
  basecaller.IE = 0.0;
  basecaller.numWellsCalled = 0;
  basecaller.nextRegionX = 0;
  basecaller.nextRegionY = 0;


  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(basecaller.CF, "0.0", '-',  "cf");
  opts.GetOption(basecaller.IE, "0.0", '-',  "ie");
  opts.GetOption(inputDirectory, ".", '-',  "input-dir");
  opts.GetOption(outputDirectory, ".", '-',  "output-dir");
  opts.GetOption(singleCoreCafie, "false", '-',  "singlecorecafie");

  int numWorkers = 2*numCores();
  if (singleCoreCafie)
    numWorkers = 1;


  Mask mask (1, 1);
  if (mask.SetMask ((inputDirectory + "/bfmask.bin").c_str()))
    exit (EXIT_FAILURE);
  RawWells wells (inputDirectory.c_str(),"1.wells");
  //SetWellsToLiveBeadsOnly(wells,&mask);
  wells.OpenForIncrementalRead();

  basecaller.maskPtr = &mask;
  basecaller.wellsPtr = &wells;
  basecaller.rows = mask.H();
  basecaller.cols = mask.W();
  basecaller.flowOrder.SetFlowOrder(wells.FlowOrder(), wells.NumFlows());
  basecaller.numFlows = wells.NumFlows();


  basecaller.numRegionsX = (basecaller.cols +  basecaller.regionXSize - 1) / basecaller.regionXSize;
  basecaller.numRegionsY = (basecaller.rows +  basecaller.regionYSize - 1) / basecaller.regionYSize;
  basecaller.numRegions = basecaller.numRegionsX * basecaller.numRegionsY;

  basecaller.libKeyFlows.assign(basecaller.numFlows,0);
  basecaller.libNumKeyFlows = basecaller.flowOrder.BasesToFlows(libKey, &basecaller.libKeyFlows[0], basecaller.numFlows);

  basecaller.libSFF.Open(outputDirectory+"/rawlib.sff", basecaller.numRegions,
      basecaller.flowOrder, libKey);


  time_t startBasecall;
  time(&startBasecall);

  pthread_mutex_init(&basecaller.wellsAccessMutex, NULL);

  pthread_t worker_id[numWorkers];
  for (int iWorker = 0; iWorker < numWorkers; iWorker++)
    if (pthread_create(&worker_id[iWorker], NULL, BasecallerWorkerWrapper, &basecaller)) {
      printf("*Error* - problem starting thread\n");
      return 1;
    }

  for (int iWorker = 0; iWorker < numWorkers; iWorker++)
    pthread_join(worker_id[iWorker], NULL);

  pthread_mutex_destroy(&basecaller.wellsAccessMutex);

  time_t endBasecall;
  time(&endBasecall);

  basecaller.libSFF.Close();

  printf("\nBASECALLING: called %d of %d wells in %1.1f seconds with %d threads\n",
      basecaller.numWellsCalled, basecaller.rows*basecaller.cols, difftime(endBasecall,startBasecall), numWorkers);
  printf("Generated library SFF with %d reads\n", basecaller.libSFF.num_reads());

  return 0;
}



void *BasecallerWorkerWrapper(void *input)
{
  static_cast<BaseCallerLite*>(input)->BasecallerWorker();
  return NULL;
}






/*

//
//  VERSION WITH LEAN AND MULTITHREADED WELLS ACCESS
//

void BaseCallerLite::BasecallerWorker()
{
  vector<float>   wellBuffer(regionXSize * regionYSize * numFlows);

  RawWells wells2 (".","1.wells");
  wells2.OpenForIncrementalRead();

  BasecallerRead currentRead;
  DPTreephaser dpTreephaser(flowOrder.c_str(), numFlows, 8);

  while (true) {

    pthread_mutex_lock(&wellsAccessMutex);

    if (nextRegionY >= numRegionsY) {
      pthread_mutex_unlock(&wellsAccessMutex);
      return;
    }

    int currentRegion = nextRegionX + numRegionsX * nextRegionY;
    int beginY = nextRegionY * regionYSize;
    int beginX = nextRegionX * regionXSize;
    int endY = min((nextRegionY+1) * regionYSize, rows);
    int endX = min((nextRegionX+1) * regionXSize, cols);

    int numUsableWells = 0;
    for (int y = beginY; y < endY; y++)
      for (int x = beginX; x < endX; x++)
        if (maskPtr->Match(x, y, (MaskType)(MaskLib|MaskKeypass), MATCH_ALL))
          numUsableWells++;

    if (nextRegionX == 0)
      printf("% 5d/% 5d: ", nextRegionY*regionYSize, rows);
    if (numUsableWells == 0)
      printf("  ");
    else if (numUsableWells < 750)
      printf(". ");
    else if (numUsableWells < 1500)
      printf("o ");
    else if (numUsableWells < 2250)
      printf("# ");
    else
      printf("$ ");

    nextRegionX++;
    if (nextRegionX == numRegionsX) {
      nextRegionX = 0;
      nextRegionY++;
      printf("\n");
    }
    fflush(NULL);

    numWellsCalled += numUsableWells;

    pthread_mutex_unlock(&wellsAccessMutex);


    // Process the data
    deque<SFFWriterWell> libReads;

    if (numUsableWells == 0) {
      libSFF.WriteRegion(currentRegion,libReads);
      continue;
    }

    wells2.SetChunk(beginY, endY-beginY, beginX, endX-beginX, 0, numFlows);
    wells2.ReadWells();

    dpTreephaser.SetModelParameters(CF, IE, 0);

    int wellIndex = 0;
    for (int y = beginY; y < endY; y++) {
      for (int x = beginX; x < endX; x++, wellIndex++) {
        if (!maskPtr->Match(x, y, (MaskType)(MaskLib|MaskKeypass), MATCH_ALL))
          continue;

        libReads.push_back(SFFWriterWell());
        SFFWriterWell& readResults = libReads.back();
        stringstream wellNameStream;
        wellNameStream << runId << ":" << y << ":" << x;
        readResults.name = wellNameStream.str();
        readResults.clipQualLeft = 4; // TODO
        readResults.clipQualRight = 0;
        readResults.clipAdapterLeft = 0;
        readResults.clipAdapterRight = 0;
        readResults.flowIonogram.resize(numFlows);

        int minReadLength = 8; // TODO

        const WellData *w = wells2.ReadXY(x, y);
        currentRead.SetDataAndKeyNormalize(w->flowValues, numFlows, &libKeyFlows[0], libNumKeyFlows - 1);

        dpTreephaser.NormalizeAndSolve5(currentRead, numFlows); // sliding window adaptive normalization

        readResults.numBases = 0;
        for (int iFlow = 0; iFlow < numFlows; iFlow++) {
          readResults.flowIonogram[iFlow] = 100 * currentRead.solution[iFlow];
          readResults.numBases += currentRead.solution[iFlow];
        }

        if(readResults.numBases < minReadLength) {
          libReads.pop_back();
          continue;
        }

        bool isFailKeypass = false;
        for (int iFlow = 0; iFlow < (libNumKeyFlows-1); iFlow++)
          if (libKeyFlows[iFlow] != currentRead.solution[iFlow])
            isFailKeypass = true;

        if(isFailKeypass) {
          libReads.pop_back();
          continue;
        }

        readResults.baseFlowIndex.reserve(readResults.numBases);
        readResults.baseCalls.reserve(readResults.numBases);
        readResults.baseQVs.reserve(readResults.numBases);

        unsigned int prev_used_flow = 0;
        for (int iFlow = 0; iFlow < numFlows; iFlow++) {
          for (hpLen_t hp = 0; hp < currentRead.solution[iFlow]; hp++) {
            readResults.baseFlowIndex.push_back(1 + iFlow - prev_used_flow);
            readResults.baseCalls.push_back(flowOrder[iFlow % flowOrder.length()]);
            readResults.baseQVs.push_back(20); // BaseCallerLite is stripped of QV generator
            prev_used_flow = iFlow + 1;
          }
        }
      }
    }

    libSFF.WriteRegion(currentRegion,libReads);
  }
}

*/





//
//  SUPER LEGACY VERSION
//

void BaseCallerLite::BasecallerWorker()
{

  while (true) {

    deque<int> wellX;
    deque<int> wellY;
    deque<vector<float> > wellMeasurements;

    pthread_mutex_lock(&wellsAccessMutex);

    if (nextRegionY >= numRegionsY) {
      pthread_mutex_unlock(&wellsAccessMutex);
      return;
    }

    int currentRegionX = nextRegionX;
    int currentRegionY = nextRegionY;
    int currentRegion = currentRegionX + numRegionsX * currentRegionY;


    int beginY = currentRegionY * regionYSize;
    int beginX = currentRegionX * regionXSize;
    int endY = min((currentRegionY+1) * regionYSize,rows);
    int endX = min((currentRegionX+1) * regionXSize,cols);
    wellsPtr->SetChunk(beginY, endY-beginY, beginX, endX-beginX, 0, numFlows);
    wellsPtr->ReadWells();
    for (int y = beginY; y < endY; y++) {
      for (int x = beginX; x < endX; x++) {
        if (!maskPtr->Match(x, y, MaskLib))
          continue;

        wellX.push_back(x);
        wellY.push_back(y);
        wellMeasurements.push_back(vector<float>());
        wellMeasurements.back().resize(numFlows);

        const WellData *w = wellsPtr->ReadXY(x, y);
        copy(w->flowValues, w->flowValues + numFlows, wellMeasurements.back().begin());
      }
    }

    if (currentRegionX == 0)
      printf("% 5d/% 5d: ", currentRegionY*regionYSize, rows);
    if (wellX.size() == 0)
      printf("  ");
    else if (wellX.size() < 750)
      printf(". ");
    else if (wellX.size() < 1500)
      printf("o ");
    else if (wellX.size() < 2250)
      printf("# ");
    else
      printf("$ ");

    nextRegionX++;
    if (nextRegionX == numRegionsX) {
      nextRegionX = 0;
      nextRegionY++;
      printf("\n");
    }
    fflush(NULL);

    pthread_mutex_unlock(&wellsAccessMutex);


    BasecallerRead currentRead;
    DPTreephaser dpTreephaser(flowOrder);
    dpTreephaser.SetModelParameters(CF, IE, 0);

    // Process the data
    deque<SFFEntry> libReads;

    deque<int>::iterator x = wellX.begin();
    deque<int>::iterator y = wellY.begin();
    deque<std::vector<float> >::iterator measurements = wellMeasurements.begin();

    for (; x != wellX.end() ; x++, y++, measurements++) {

      if (!maskPtr->Match(*x, *y, (MaskType)(MaskLib|MaskKeypass), MATCH_ALL))
        continue;

      libReads.push_back(SFFEntry());
      SFFEntry& readResults = libReads.back();
      stringstream wellNameStream;
      wellNameStream << runId << ":" << (*y) << ":" << (*x);
      readResults.name = wellNameStream.str();
      readResults.clip_qual_left = 4; // TODO
      readResults.clip_qual_right = 0;
      readResults.clip_adapter_left = 0;
      readResults.clip_adapter_right = 0;
      readResults.flowgram.resize(numFlows);

      int minReadLength = 8; // TODO

      currentRead.SetDataAndKeyNormalize(&(measurements->at(0)), numFlows, &libKeyFlows[0], libNumKeyFlows - 1);

      dpTreephaser.NormalizeAndSolve5(currentRead, numFlows); // sliding window adaptive normalization

      readResults.n_bases = 0;
      for (int iFlow = 0; iFlow < numFlows; iFlow++) {
        readResults.flowgram[iFlow] = 100 * currentRead.solution[iFlow];
        readResults.n_bases += currentRead.solution[iFlow];
      }

      if(readResults.n_bases < minReadLength) {
        libReads.pop_back();
        continue;
      }

      bool isFailKeypass = false;
      for (int iFlow = 0; iFlow < (libNumKeyFlows-1); iFlow++)
        if (libKeyFlows[iFlow] != currentRead.solution[iFlow])
          isFailKeypass = true;

      if(isFailKeypass) {
        libReads.pop_back();
        continue;
      }

      readResults.flow_index.reserve(readResults.n_bases);
      readResults.bases.reserve(readResults.n_bases);
      readResults.quality.reserve(readResults.n_bases);

      unsigned int prev_used_flow = 0;
      for (int iFlow = 0; iFlow < numFlows; iFlow++) {
        for (hpLen_t hp = 0; hp < currentRead.solution[iFlow]; hp++) {
          readResults.flow_index.push_back(1 + iFlow - prev_used_flow);
          readResults.bases.push_back(flowOrder[iFlow]);
          readResults.quality.push_back(20); // BaseCallerLite is stripped of QV generator
          prev_used_flow = iFlow + 1;
        }
      }

    }

    libSFF.WriteRegion(currentRegion,libReads);
  }
}

