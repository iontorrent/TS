/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PHASEESTIMATOR_H
#define PHASEESTIMATOR_H


#include <unistd.h>
#include <math.h>
#include <vector>
#include <string>
#include <cassert>
#include "json/json.h"

#include "RawWells.h"
#include "Mask.h"
#include "DPTreephaser.h"
#include "OptArgs.h"
#include "SeqList.h"
#include "BaseCallerUtils.h"

using namespace std;


class PhaseEstimator {
public:
  PhaseEstimator();

  void InitializeFromOptArgs(OptArgs& opts, const string& _flowOrder, int _numFlows, const vector<KeySequence>& _keys);

  void DoPhaseEstimation(RawWells *wellsPtr, Mask *maskPtr, int _regionXSize, int _regionYSize, bool singleCoreCafie);


  float getCF(int x, int y);
  float getIE(int x, int y);
  float getDR(int x, int y);

  void ExportResultsToJson(Json::Value &phasing);


private:
  void SpatialRefiner(RawWells *wells, Mask *mask,int _regionSizeX, int _regionSizeY, int numWorkers);

  string phaseEstimator;
  float overrideCF, overrideIE, overrideDR;
  int cfiedrRegionsX, cfiedrRegionsY;

  vector<float>       cf;
  vector<float>       ie;
  vector<float>       dr;


  string              flowOrder;
  int                 numFlows;
  int                 numEstimatorFlows;

  vector<KeySequence> keys;

  int regionSizeX, regionSizeY;
  int numRegionsX, numRegionsY, numRegions;
  int numLevels;
  int numWellsX, numWellsY;

  vector<int> regionDensityMap;

  int RegionByXY(int x, int y) { return x + y * numRegionsX; }


  struct Subblock {
    float         CF, IE, DR;
    int           beginX, endX, beginY, endY, level;
    vector<int>   searchOrder;
    Subblock*     subblocks[4];
    int           posX, posY;
    Subblock*     superblock;
  };
  vector<Subblock> subblocks;

  vector<vector<BasecallerRead> > regionReads;
  size_t loadRegion(int region, RawWells *wells, Mask *mask);

  void NelderMeadOptimization (Subblock& s, int numUsefulReads, DPTreephaser& treephaser,
      float *parameters, int numEvaluations, int numParameters);

  float evaluateParameters(Subblock& s, int numUsefulReads, DPTreephaser& treephaser, float *parameters);

  struct CompareDensity {
  public:
    vector<int> *density;
    CompareDensity(vector<int> *_density) : density(_density) {}
    bool operator() (int i, int j) { return fabs(1500-density->at(i)) < fabs(1500-density->at(j)); }
  };


  pthread_mutex_t regionLoaderMutex;
  pthread_mutex_t jobQueueMutex;
  pthread_cond_t  jobQueueCond;

  static void *EstimatorWorkerWrapper(void *arg);
  void EstimatorWorker();

  vector<int> actionMap;
  RawWells *sharedWells;
  Mask *sharedMask;
  int numJobsSubmitted;
  int numJobsCompleted;
  int nextJob;
  Subblock *jobQueue[2*4*4*4];
};




#endif // PHASEESTIMATOR_H
