/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <algorithm>
#include <vector>
#include <string>
#include <cassert>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "PhaseEstimator.h"
#include "RawWells.h"
#include "Mask.h"
#include "IonErr.h"
#include "DPTreephaser.h"
#include "Utils.h"
#include "RegionAnalysis.h"



PhaseEstimator::PhaseEstimator()
{
  numFlows = 0;
  numEstimatorFlows = 0;
  regionSizeX = regionSizeY = 0;
  numRegionsX = numRegionsY = numRegions = 0;
  numWellsX = numWellsY = 0;
  numLevels = 0;
  sharedWells = NULL;
  sharedMask = NULL;
  numJobsSubmitted = 0;
  numJobsCompleted = 0;
  nextJob = 0;

  overrideCF = 0.0;
  overrideIE = 0.0;
  overrideDR = 0.0;
  cfiedrRegionsX = 13;
  cfiedrRegionsY = 12;

  jobQueue[0] = NULL;
}

void PhaseEstimator::InitializeFromOptArgs(OptArgs& opts, const string& _flowOrder, int _numFlows, const vector<KeySequence>& _keys)
{
  // ***** Phase parameter estimation options

  // "nel-mead-treephaser"; // "nel-mead-adaptive-treephaser";
  phaseEstimator          = opts.GetFirstString('-', "phasing-estimator", "spatial-refiner");

  string argLibCFIEDR     = opts.GetFirstString('-', "libcf-ie-dr", "");
  if (!argLibCFIEDR.empty()) {
    int stat = sscanf (argLibCFIEDR.c_str(), "%f,%f,%f", &overrideCF, &overrideIE, &overrideDR);
    if (stat != 3)
    {
      fprintf (stderr, "Option Error: libcf-ie-dr %s\n", argLibCFIEDR.c_str());
      exit (EXIT_FAILURE);
    }
    phaseEstimator = "override";
  }

  string argCfiedrRegions = opts.GetFirstString('R', "phasing-regions", "");
  if (!argCfiedrRegions.empty()) {
    int stat = sscanf (argCfiedrRegions.c_str(), "%dx%d", &cfiedrRegionsX, &cfiedrRegionsY);
    if (stat != 2)
    {
      fprintf (stderr, "Option Error: cfiedr-regions %s\n", argCfiedrRegions.c_str());
      exit (EXIT_FAILURE);
    }
  }

  flowOrder = _flowOrder;
  numFlows = _numFlows;
  numEstimatorFlows = min(numFlows, 120);
  keys = _keys;
}

void PhaseEstimator::DoPhaseEstimation(RawWells *wellsPtr, Mask *maskPtr,
    int _regionXSize, int _regionYSize, bool singleCoreCafie)
{
  numWellsX = maskPtr->W();
  numWellsY = maskPtr->H();


  printf("Phase estimation mode = %s\n", phaseEstimator.c_str());

  if (phaseEstimator == "override") {
    cfiedrRegionsX = 1;
    cfiedrRegionsY = 1;
    cf.assign(1,overrideCF);
    ie.assign(1,overrideIE);
    dr.assign(1,overrideDR);
    return;
  }

  if ((phaseEstimator == "nel-mead-treephaser") || (phaseEstimator == "nel-mead-adaptive-treephaser")) {

    int numWorkers = max(numCores(), 2);
    if (singleCoreCafie)
      numWorkers = 1;

    cf.assign(cfiedrRegionsX*cfiedrRegionsY,0.0);
    ie.assign(cfiedrRegionsX*cfiedrRegionsY,0.0);
    dr.assign(cfiedrRegionsX*cfiedrRegionsY,0.0);
    RegionAnalysis regionAnalysis;
    regionAnalysis.analyze(&cf, &ie, &dr, wellsPtr, maskPtr, keys, flowOrder, numFlows, numWorkers,
        cfiedrRegionsX, cfiedrRegionsY, phaseEstimator);
    return;
  }

  if (phaseEstimator == "spatial-refiner") {

    int numWorkers = max(numCores(), 2);
    if (singleCoreCafie)
      numWorkers = 1;

    wellsPtr->Close();
    wellsPtr->OpenForIncrementalRead();
    SpatialRefiner(wellsPtr, maskPtr, _regionXSize, _regionYSize, numWorkers);

    return;
  }

  ION_ABORT("Requested phase estimator is not recognized");

}


void PhaseEstimator::ExportResultsToJson(Json::Value &phasing)
{
  // Save phase estimates to BaseCaller.json

  float CFmean = 0;
  float IEmean = 0;
  float DRmean = 0;
  int count = 0;

  for (int r = 0; r < cfiedrRegionsY*cfiedrRegionsX; r++) {
    phasing["CFbyRegion"][r] = cf[r];
    phasing["IEbyRegion"][r] = ie[r];
    phasing["DRbyRegion"][r] = dr[r];
    if (cf[r] || ie[r] || dr[r]) {
      CFmean += cf[r];
      IEmean += ie[r];
      DRmean += dr[r];
      count++;
    }
  }
  phasing["RegionRows"] = cfiedrRegionsY;
  phasing["RegionCols"] = cfiedrRegionsX;

  phasing["CF"] = count ? (CFmean/count) : 0;
  phasing["IE"] = count ? (IEmean/count) : 0;
  phasing["DR"] = count ? (DRmean/count) : 0;

}




float PhaseEstimator::getCF(int x, int y)
{
  int cafieYinc = ceil(numWellsY / (double) cfiedrRegionsY);
  int cafieXinc = ceil(numWellsX / (double) cfiedrRegionsX);
  int iRegion = (y / cafieYinc) + (x / cafieXinc) * cfiedrRegionsY;
  return cf[iRegion];
}

float PhaseEstimator::getIE(int x, int y)
{
  int cafieYinc = ceil(numWellsY / (double) cfiedrRegionsY);
  int cafieXinc = ceil(numWellsX / (double) cfiedrRegionsX);
  int iRegion = (y / cafieYinc) + (x / cafieXinc) * cfiedrRegionsY;
  return ie[iRegion];
}

float PhaseEstimator::getDR(int x, int y)
{
  int cafieYinc = ceil(numWellsY / (double) cfiedrRegionsY);
  int cafieXinc = ceil(numWellsX / (double) cfiedrRegionsX);
  int iRegion = (y / cafieYinc) + (x / cafieXinc) * cfiedrRegionsY;
  return dr[iRegion];
}





void PhaseEstimator::SpatialRefiner(RawWells *wells, Mask *mask,int _regionSizeX, int _regionSizeY, int numWorkers)
{
  printf("PhaseEstimator::analyze start\n");

  regionSizeX = _regionSizeX;
  regionSizeY = _regionSizeY;

  numWellsX = mask->W();
  numWellsY = mask->H();
  numRegionsX = (numWellsX+regionSizeX-1) / regionSizeX;
  numRegionsY = (numWellsY+regionSizeY-1) / regionSizeY;
  numRegions = numRegionsX * numRegionsY;

  numLevels = 1;
  if (numRegionsX >= 2 and numRegionsY >= 2)
    numLevels = 2;
  if (numRegionsX >= 4 and numRegionsY >= 4)
    numLevels = 3;
  if (numRegionsX >= 8 and numRegionsY >= 8)
    numLevels = 4;

  printf("Using numEstimatorFlows %d, chip is %d x %d, region is %d x %d, numRegions is %d x %d, numLevels %d\n",
      numEstimatorFlows, numWellsX, numWellsY, regionSizeX, regionSizeY, numRegionsX, numRegionsY, numLevels);

  // Step 1. Use mask to build region density map

  regionDensityMap.assign(numRegions, 0);
  for (int x = 0; x < mask->W(); x++)
    for (int y = 0; y < mask->H(); y++)
      if (mask->Match(x,y,(MaskType)(MaskTF|MaskLib)))
        regionDensityMap[RegionByXY(x/regionSizeX,y/regionSizeY)]++;

  // Step 2. Build the tree of estimation subblocks.

  int maxSubblocks = 2*4*4*4;
  subblocks.reserve(maxSubblocks);
  subblocks.push_back(Subblock());
  subblocks.back().CF = 0.0;
  subblocks.back().IE = 0.0;
  subblocks.back().DR = 0.0;
  subblocks.back().beginX = 0;
  subblocks.back().endX = numRegionsX;
  subblocks.back().beginY = 0;
  subblocks.back().endY = numRegionsY;
  subblocks.back().level = 1;
  subblocks.back().posX = 0;
  subblocks.back().posY = 0;
  subblocks.back().superblock = NULL;

  for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
    Subblock &s = subblocks[idx];
    if (s.level == numLevels) {
      s.subblocks[0] = NULL;
      s.subblocks[1] = NULL;
      s.subblocks[2] = NULL;
      s.subblocks[3] = NULL;
      continue;
    }

    int cutX = (s.beginX + s.endX) / 2;
    int cutY = (s.beginY + s.endY) / 2;

    subblocks.push_back(s);
    subblocks.back().CF = -1.0;
    subblocks.back().IE = -1.0;
    subblocks.back().DR = -1.0;
    subblocks.back().endX = cutX;
    subblocks.back().endY = cutY;
    subblocks.back().level++;
    subblocks.back().posX = (s.posX << 1);
    subblocks.back().posY = (s.posY << 1);
    subblocks.back().superblock = &s;
    s.subblocks[0] = &subblocks.back();

    subblocks.push_back(s);
    subblocks.back().CF = -1.0;
    subblocks.back().IE = -1.0;
    subblocks.back().DR = -1.0;
    subblocks.back().beginX = cutX;
    subblocks.back().endY = cutY;
    subblocks.back().level++;
    subblocks.back().posX = (s.posX << 1) + 1;
    subblocks.back().posY = (s.posY << 1);
    subblocks.back().superblock = &s;
    s.subblocks[1] = &subblocks.back();

    subblocks.push_back(s);
    subblocks.back().CF = -1.0;
    subblocks.back().IE = -1.0;
    subblocks.back().DR = -1.0;
    subblocks.back().endX = cutX;
    subblocks.back().beginY = cutY;
    subblocks.back().level++;
    subblocks.back().posX = (s.posX << 1);
    subblocks.back().posY = (s.posY << 1) + 1;
    subblocks.back().superblock = &s;
    s.subblocks[2] = &subblocks.back();

    subblocks.push_back(s);
    subblocks.back().CF = -1.0;
    subblocks.back().IE = -1.0;
    subblocks.back().DR = -1.0;
    subblocks.back().beginX = cutX;
    subblocks.back().beginY = cutY;
    subblocks.back().level++;
    subblocks.back().posX = (s.posX << 1) + 1;
    subblocks.back().posY = (s.posY << 1) + 1;
    subblocks.back().superblock = &s;
    s.subblocks[3] = &subblocks.back();
  }

  // Step 3. Populate region searchOrder in lowermost subblocks

  vector<int> subblockMap(numRegions,0);

  for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
    Subblock &s = subblocks[idx];
    if (s.level != numLevels)
      continue;

    int numSubblockRegions = (s.endX - s.beginX) * (s.endY - s.beginY);
    assert(numSubblockRegions > 0);
    s.searchOrder.reserve(numSubblockRegions);
    for (int regionX = s.beginX; regionX < s.endX; regionX++) {
      for (int regionY = s.beginY; regionY < s.endY; regionY++) {
        s.searchOrder.push_back(RegionByXY(regionX,regionY));
        subblockMap[RegionByXY(regionX,regionY)] = ((s.posX ^ s.posY) & 1);
      }
    }

    sort(s.searchOrder.begin(), s.searchOrder.end(), CompareDensity(&regionDensityMap));

  }

  // Step 4. Populate region searchOrder in remaining subblocks

  for (int currentLevel = numLevels-1; currentLevel >= 1; currentLevel--) {
    for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
      Subblock &s = subblocks[idx];
      if (s.level != currentLevel)
        continue;

      assert(s.subblocks[0] != NULL);
      assert(s.subblocks[1] != NULL);
      assert(s.subblocks[2] != NULL);
      assert(s.subblocks[3] != NULL);
      unsigned int sumRegions = s.subblocks[0]->searchOrder.size()
          + s.subblocks[1]->searchOrder.size()
          + s.subblocks[2]->searchOrder.size()
          + s.subblocks[3]->searchOrder.size();
      s.searchOrder.reserve(sumRegions);
      vector<int>::iterator V0 = s.subblocks[0]->searchOrder.begin();
      vector<int>::iterator V1 = s.subblocks[1]->searchOrder.begin();
      vector<int>::iterator V2 = s.subblocks[2]->searchOrder.begin();
      vector<int>::iterator V3 = s.subblocks[3]->searchOrder.begin();
      while (s.searchOrder.size() < sumRegions) {
        if (V0 != s.subblocks[0]->searchOrder.end())
          s.searchOrder.push_back(*V0++);
        if (V2 != s.subblocks[2]->searchOrder.end())
          s.searchOrder.push_back(*V2++);
        if (V1 != s.subblocks[1]->searchOrder.end())
          s.searchOrder.push_back(*V1++);
        if (V3 != s.subblocks[3]->searchOrder.end())
          s.searchOrder.push_back(*V3++);
      }
    }
  }


  // Step 5. Show time. Spawn multiple worker threads to do phasing estimation

  regionReads.clear();
  regionReads.resize(numRegions);
  actionMap.assign(numRegions,0);

  pthread_mutex_init(&regionLoaderMutex, NULL);
  pthread_mutex_init(&jobQueueMutex, NULL);
  pthread_cond_init(&jobQueueCond, NULL);

  sharedWells = wells;
  sharedMask = mask;

  jobQueue[0] = &subblocks[0];
  numJobsSubmitted = 1;
  numJobsCompleted = 0;
  nextJob = 0;

  pthread_t worker_id[numWorkers];

  for (int iWorker = 0; iWorker < numWorkers; iWorker++)
    if (pthread_create(&worker_id[iWorker], NULL, EstimatorWorkerWrapper, this))
      ION_ABORT("*Error* - problem starting thread");

  for (int iWorker = 0; iWorker < numWorkers; iWorker++)
    pthread_join(worker_id[iWorker], NULL);

  pthread_cond_destroy(&jobQueueCond);
  pthread_mutex_destroy(&jobQueueMutex);
  pthread_mutex_destroy(&regionLoaderMutex);



  // Print a silly action map

  for (int regionY = 0; regionY < numRegionsY; regionY++) {
    for (int regionX = 0; regionX < numRegionsX; regionX++) {

      int r = RegionByXY(regionX,regionY);
      if (regionDensityMap[r] == 0) {
        printf("  ");
        continue;
      }

      if (actionMap[r] == 0)
        printf(" ");
      else
        printf("%d",actionMap[r]);
      if (subblockMap[r] == 0)
        printf("#");
      else
        printf(" ");
    }
    printf("\n");
  }

  // Crunching complete. Retrieve phasing estimates

  cfiedrRegionsX = 1 << (numLevels-1);
  cfiedrRegionsY = 1 << (numLevels-1);
  cf.assign(cfiedrRegionsX*cfiedrRegionsY,0.0);
  ie.assign(cfiedrRegionsX*cfiedrRegionsY,0.0);
  dr.assign(cfiedrRegionsX*cfiedrRegionsY,0.0);

  for (unsigned int subIdx = 0; subIdx < subblocks.size(); subIdx++) {
    Subblock *current = &subblocks[subIdx];
    if (current->level != numLevels)
      continue;
    while (current) {
      if (current->CF >= 0) {
        cf[subblocks[subIdx].posY + cfiedrRegionsY * subblocks[subIdx].posX] = current->CF;
        ie[subblocks[subIdx].posY + cfiedrRegionsY * subblocks[subIdx].posX] = current->IE;
        dr[subblocks[subIdx].posY + cfiedrRegionsY * subblocks[subIdx].posX] = current->DR;
        break;
      }
      current = current->superblock;
    }
  }

  printf("PhaseEstimator::analyze end\n");
}






size_t PhaseEstimator::loadRegion(int region, RawWells *wells, Mask *mask)
{
  if (regionDensityMap[region] == 0) // Nothing to load ?
    return 0;
  if (regionReads[region].size() > 0) // Region already loaded?
    return 0;

  ClockTimer timer;
  timer.StartTimer();

  regionReads[region].reserve(regionDensityMap[region]);

  int regionY = region / numRegionsX;
  int regionX = region - numRegionsX*regionY;

  int beginY = regionY * regionSizeY;
  int beginX = regionX * regionSizeX;
  int endY = min((regionY+1) * regionSizeY, (int)wells->NumRows());
  int endX = min((regionX+1) * regionSizeX, (int)wells->NumCols());


  //printf("  - Region % 4d: Preparing to read %d %d %d %d 0 %d\n",
  //      region, beginY, endY-beginY, beginX, endX-beginX, numEstimatorFlows);

  // Mutex needed for wells access, but not needed for regionReads access
  // TODO: Investigate possibility of each thread having its own RawWells class.
  pthread_mutex_lock(&regionLoaderMutex);

  wells->SetChunk(beginY, endY-beginY, beginX, endX-beginX, 0, numEstimatorFlows);
  wells->ReadWells();

  vector<float> wellBuffer(numEstimatorFlows);

  for (int y = beginY; y < endY; y++) {
    for (int x = beginX; x < endX; x++) {

      if (!mask->Match(x, y, MaskLive))
        continue;
      if (!mask->Match(x, y, MaskBead))
        continue;

      int cls = 0;
      if (!mask->Match(x, y, MaskLib)) {  // Not a library bead?
        cls = 1;
        if (!mask->Match(x, y, MaskTF))   // Not a tf bead?
          continue;
      }

      for (int iFlow = 0; iFlow < numEstimatorFlows; iFlow++)
        wellBuffer[iFlow] = wells->At(y,x,iFlow);

      // Sanity check. If there are NaNs in this read, print warning
      vector<int> nanflow;
      for (int flow = 0; flow < numEstimatorFlows; ++flow) {
        if (!isnan(wellBuffer[flow]))
          continue;
        wellBuffer[flow] = 0;
        nanflow.push_back(flow);
      }
      if(nanflow.size() > 0) {
        fprintf(stderr, "ERROR: BaseCaller read NaNs from wells file, x=%d y=%d flow=%d", x, y, nanflow[0]);
        for(unsigned int iFlow=1; iFlow < nanflow.size(); iFlow++) {
          fprintf(stderr, ",%d", nanflow[iFlow]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
      }

      regionReads[region].push_back(BasecallerRead());
      regionReads[region].back().SetDataAndKeyNormalize(&wellBuffer[0], numEstimatorFlows, keys[cls].flows(), keys[cls].flows_length()-1);

      bool keypass = true;
      for (int iFlow = 0; iFlow < (keys[cls].flows_length() - 1); iFlow++) {
        if ((int) (regionReads[region].back().measurements[iFlow] + 0.5) != keys[cls][iFlow])
          keypass = false;
        if (isnan(regionReads[region].back().measurements[iFlow]))
          keypass = false;
      }

      if (!keypass) {
        regionReads[region].pop_back();
        continue;
      }

    }
  }

  pthread_mutex_unlock(&regionLoaderMutex);

//  printf("  - Region % 4d: Expected %d, read %lu, time %luus\n",
//      region, regionDensityMap[region], regionReads[region].size(), timer.GetMicroSec());

  regionDensityMap[region] = (int)regionReads[region].size();

  return timer.GetMicroSec();
}



void *PhaseEstimator::EstimatorWorkerWrapper(void *arg)
{
  static_cast<PhaseEstimator*>(arg)->EstimatorWorker();
  return NULL;
}


void PhaseEstimator::EstimatorWorker()
{

  DPTreephaser dpTreephaser(flowOrder.c_str(), numEstimatorFlows, 8);

  while (true) {

    int currentJob = -1;
    pthread_mutex_lock(&jobQueueMutex);
    while (true) {
      if (numJobsSubmitted == numJobsCompleted) {   // No more work
        pthread_mutex_unlock(&jobQueueMutex);
        return;
      }
      if (nextJob < numJobsSubmitted) { // Job is available. Get on it
        currentJob = nextJob++;
        pthread_mutex_unlock(&jobQueueMutex);
        break;
      }
      // No jobs available now, but more may come, so stick around
      pthread_cond_wait(&jobQueueCond, &jobQueueMutex);
    }

    Subblock &s = *jobQueue[currentJob];

    // Processing

    //  - Get first region at lvl 0
    //  - Keep getting more regions at level 0:
    //    - Stop when have 5000 eligible reads or run out of regions
    //  - Generate estimates
    //  - If lvl 0 is last, stop (and unload regions at lvl 0)
    //    - Else spawn lvl 1 jobs



    int numGlobalIterations = 1;
    int desiredNumReads = 5000;
    if (s.level == 1)
      numGlobalIterations += 2;
    int numUsefulReads = 0;

    for (int iGlobalIteration = 0; iGlobalIteration < numGlobalIterations; iGlobalIteration++) {

      ClockTimer timer;
      timer.StartTimer();
      size_t iotimer = 0;

      dpTreephaser.SetModelParameters(s.CF, s.IE, s.DR);

      numUsefulReads = 0;

      for (vector<int>::iterator region = s.searchOrder.begin(); region != s.searchOrder.end(); region++) {

        if (actionMap[*region] == 0)
          actionMap[*region] = s.level;

        iotimer += loadRegion(*region, sharedWells, sharedMask);
        // Ensure region loaded.
        // Grab reads, filter
        // Enough reads? Stop.

        // Filter. Mark filtered out reads with -1e20 in normalizedMeasurements[0]
        for (vector<BasecallerRead>::iterator R = regionReads[*region].begin(); R != regionReads[*region].end(); R++) {

          for (int iFlow = 0; iFlow < numEstimatorFlows; iFlow++)
            R->normalizedMeasurements[iFlow] = R->measurements[iFlow];

          dpTreephaser.Solve(*R, min(100, numEstimatorFlows));
          R->Normalize(11, min(80, numEstimatorFlows));
          dpTreephaser.Solve(*R, min(120, numEstimatorFlows));
          R->Normalize(11, min(100, numEstimatorFlows));
          dpTreephaser.Solve(*R, min(120, numEstimatorFlows));

          float metric = 0;
          for (int iFlow = 20; (iFlow < 100) && (iFlow < numEstimatorFlows); iFlow++) {
            if (R->normalizedMeasurements[iFlow] > 1.2)
              continue;
            float delta = R->normalizedMeasurements[iFlow] - R->prediction[iFlow];
            if (!isnan(delta))
              metric += delta * delta;
            else
              metric += 1e10;
          }

          if (metric > 1)
            R->normalizedMeasurements[0] = -1e20; // Ignore me
          else
            numUsefulReads++;
        }

        if (numUsefulReads >= desiredNumReads)
          break;
      }

      if (s.level > 1 and numUsefulReads < 1000) // Not enough reads to even try
        break;

      // Do estimation with reads collected, update estimates
      float parameters[3];
      parameters[0] = s.CF;
      parameters[1] = s.IE;
      parameters[2] = s.DR;
      NelderMeadOptimization(s, numUsefulReads, dpTreephaser, parameters, 50, 3);
      s.CF = parameters[0];
      s.IE = parameters[1];
      s.DR = parameters[2];

      printf("Completed stage %2d/%2d :(%2d-%2d)x(%2d-%2d), total time %5.2lf sec, i/o time %5.2lf sec, %d reads, CF=%1.2f%% IE=%1.2f%% DR=%1.2f%%\n",
          currentJob+1, numJobsSubmitted, s.beginX,s.endX,s.beginY,s.endY,
          (double)timer.GetMicroSec()/1000000.0, (double)iotimer/1000000.0, numUsefulReads,
          100.0*s.CF, 100.0*s.IE, 100.0*s.DR);
    }


    if (s.level == numLevels or numUsefulReads < 4000) {
      // Do not subdivide this block
      for (vector<int>::iterator region = s.searchOrder.begin(); region != s.searchOrder.end(); region++)
        regionReads[*region].clear();

      pthread_mutex_lock(&jobQueueMutex);
      numJobsCompleted++;
      if (numJobsSubmitted == numJobsCompleted)  // No more work, let everyone know
        pthread_cond_broadcast(&jobQueueCond);
      pthread_mutex_unlock(&jobQueueMutex);

    } else {
      // Subdivide. Spawn new jobs:

      pthread_mutex_lock(&jobQueueMutex);
      numJobsCompleted++;
      for (int subjob = 0; subjob < 4; subjob++) {
        jobQueue[numJobsSubmitted] = s.subblocks[subjob];
        jobQueue[numJobsSubmitted]->CF = s.CF;
        jobQueue[numJobsSubmitted]->IE = s.IE;
        jobQueue[numJobsSubmitted]->DR = s.DR;
        numJobsSubmitted++;
      }
      pthread_cond_broadcast(&jobQueueCond);  // More work, let everyone know
      pthread_mutex_unlock(&jobQueueMutex);

    }
  }

}




float PhaseEstimator::evaluateParameters(Subblock& s, int numUsefulReads, DPTreephaser& treephaser, float *parameters)
{
  float metric = 0;

  if (parameters[0] < 0) // cf
    metric = 1e10;
  if (parameters[1] < 0) // ie
    metric = 1e10;
  if (parameters[2] < 0) // dr
    metric = 1e10;

  if (parameters[0] > 0.04) // cf
    metric = 1e10;
  if (parameters[1] > 0.04) // ie
    metric = 1e10;
  if (parameters[2] > 0.01) // dr
    metric = 1e10;

  if (metric == 0) {

    treephaser.SetModelParameters(parameters[0], parameters[1], parameters[2]);

    int counter = 0;
    for (vector<int>::iterator region = s.searchOrder.begin(); region != s.searchOrder.end() and (counter < numUsefulReads); region++) {
      for (vector<BasecallerRead>::iterator R = regionReads[*region].begin(); R != regionReads[*region].end() and (counter < numUsefulReads); R++) {
        if (R->normalizedMeasurements[0] < -1e10)
          continue;

        treephaser.Simulate3(*R, 120);
        R->Normalize(20, 100);

        for (int iFlow = 20; iFlow < std::min(100, numEstimatorFlows); iFlow++) {
          if (R->measurements[iFlow] > 1.2)
            continue;
          float delta = R->measurements[iFlow] - R->prediction[iFlow] * R->miscNormalizer;
          metric += delta * delta;
        }
        counter++;
      }
    }
  }

  if (isnan(metric))
    metric = 1e10;

  return metric;
}





#define NMalpha   1.0
#define NMgamma   2.0
#define NMrho   0.5
#define NMsigma   0.5


void PhaseEstimator::NelderMeadOptimization (Subblock& s, int numUsefulReads, DPTreephaser& treephaser,
    float *parameters, int numEvaluations, int numParameters)
{

  int iEvaluation = 0;

  //
  // Step 1. Pick initial vertices, evaluate the function at vertices, and sort the vertices
  //

  float   vertex[numParameters+1][numParameters];
  float   value[numParameters+1];
  int     order[numParameters+1];

  for (int iVertex = 0; iVertex <= numParameters; iVertex++) {

    for (int iParam = 0; iParam < numParameters; iParam++)
      vertex[iVertex][iParam] = parameters[iParam];

        switch (iVertex) {
        case 0:                 // First vertex just matches the provided starting values
            break;
        case 1:                 // Second vertex has higher CF
            vertex[iVertex][0] += 0.004;
            break;
        case 2:                 // Third vertex has higher IE
            vertex[iVertex][1] += 0.004;
            break;
        case 3:                 // Fourth vertex has higher DR
            vertex[iVertex][2] += 0.001;
            break;
        default:                // Default for future parameters
            vertex[iVertex][iVertex-1] *= 1.5;
            break;
        }

    value[iVertex] = evaluateParameters(s, numUsefulReads, treephaser, vertex[iVertex]);
    iEvaluation++;

    order[iVertex] = iVertex;

    for (int xVertex = iVertex; xVertex > 0; xVertex--) {
      if (value[order[xVertex]] < value[order[xVertex-1]]) {
        int x = order[xVertex];
        order[xVertex] = order[xVertex-1];
        order[xVertex-1] = x;
      }
    }
  }

  // Main optimization loop

  while (iEvaluation < numEvaluations) {

    //
    // Step 2. Attempt reflection (and possibly expansion)
    //

    float center[numParameters];
    float reflection[numParameters];

    int worst = order[numParameters];
    int secondWorst = order[numParameters-1];
    int best = order[0];

    for (int iParam = 0; iParam < numParameters; iParam++) {
      center[iParam] = 0;
      for (int iVertex = 0; iVertex <= numParameters; iVertex++)
        if (iVertex != worst)
          center[iParam] += vertex[iVertex][iParam];
      center[iParam] /= numParameters ;
      reflection[iParam] = center[iParam] + NMalpha * (center[iParam] - vertex[worst][iParam]);
    }

    float reflectionValue = evaluateParameters(s, numUsefulReads, treephaser, reflection);
    iEvaluation++;

    if (reflectionValue < value[best]) {    // Consider expansion:

      float expansion[numParameters];
      for (int iParam = 0; iParam < numParameters; iParam++)
        expansion[iParam] = center[iParam] + NMgamma * (center[iParam] - vertex[worst][iParam]);
      float expansionValue = evaluateParameters(s, numUsefulReads, treephaser, expansion);
      iEvaluation++;

      if (expansionValue < reflectionValue) {   // Expansion indeed better than reflection
        for (int iParam = 0; iParam < numParameters; iParam++)
          reflection[iParam] = expansion[iParam];
        reflectionValue = expansionValue;
      }
    }

    if (reflectionValue < value[secondWorst]) { // Either reflection or expansion was successful

      for (int iParam = 0; iParam < numParameters; iParam++)
        vertex[worst][iParam] = reflection[iParam];
      value[worst] = reflectionValue;

      for (int xVertex = numParameters; xVertex > 0; xVertex--) {
        if (value[order[xVertex]] < value[order[xVertex-1]]) {
          int x = order[xVertex];
          order[xVertex] = order[xVertex-1];
          order[xVertex-1] = x;
        }
      }
      continue;
    }


    //
    // Step 3. Attempt contraction (reflection was unsuccessful)
    //

    float contraction[numParameters];
    for (int iParam = 0; iParam < numParameters; iParam++)
      contraction[iParam] = vertex[worst][iParam] + NMrho * (center[iParam] - vertex[worst][iParam]);
    float contractionValue = evaluateParameters(s, numUsefulReads, treephaser, contraction);
    iEvaluation++;

    if (contractionValue < value[worst]) {  // Contraction was successful

      for (int iParam = 0; iParam < numParameters; iParam++)
        vertex[worst][iParam] = contraction[iParam];
      value[worst] = contractionValue;

      for (int xVertex = numParameters; xVertex > 0; xVertex--) {
        if (value[order[xVertex]] < value[order[xVertex-1]]) {
          int x = order[xVertex];
          order[xVertex] = order[xVertex-1];
          order[xVertex-1] = x;
        }
      }
      continue;
    }


    //
    // Step 4. Perform reduction (contraction was unsuccessful)
    //

    for (int iVertex = 1; iVertex <= numParameters; iVertex++) {

      for (int iParam = 0; iParam < numParameters; iParam++)
        vertex[order[iVertex]][iParam] = vertex[best][iParam] + NMsigma * (vertex[order[iVertex]][iParam] - vertex[best][iParam]);

      value[order[iVertex]] = evaluateParameters(s, numUsefulReads, treephaser, vertex[order[iVertex]]);
      iEvaluation++;

      for (int xVertex = iVertex; xVertex > 0; xVertex--) {
        if (value[order[xVertex]] < value[order[xVertex-1]]) {
          int x = order[xVertex];
          order[xVertex] = order[xVertex-1];
          order[xVertex-1] = x;
        }
      }
    }
  }

  for (int iParam = 0; iParam < numParameters; iParam++)
    parameters[iParam] = vertex[order[0]][iParam];
}




