/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BASECALLER_H
#define BASECALLER_H

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "Mask.h"
#include "RawWells.h"
#include "LinuxCompat.h"
#include "CommandLineOpts.h"
#include "mixed.h"
#include "Stats.h"
#include "TrackProgress.h"
#include "RegionAnalysis.h"
#include "ReservoirSample.h"
#include "CafieSolver.h"
#include "IonErr.h"
#include "DPTreephaser.h"
#include "PerBaseQual.h"
#include "RegionWellsReader.h"
#include "OrderedRegionSFFWriter.h"

#include "json/json.h"


using namespace std;



//#define MAX_MER 12
#define MAX_KEY_FLOWS     64

// The max number of flows to be evaluated for "percent positive flows" (ppf) metric
#define PERCENT_POSITIVE_FLOWS_N 60

// The max number of flows to be evaluated for Cafie residual metrics
#define CAFIE_RESIDUAL_FLOWS_N 60

typedef uint32_t     well_index_t;  // To hold well indexes.  32-bit ints will allow us up to 4.3G wells
typedef uint8_t      read_class_t;  // To hold "read classes" - TF species & library keys.  8-bit ints will allow us up to 255 classes.
typedef uint16_t     read_region_t; // To hold spatial region indices.  16-bit ints will allow us up to 66K regions.



void GenerateBasesFromWells(CommandLineOpts &clo, RawWells &rawWells, Mask *maskPtr, SequenceItem *seqList, int rows, int cols,
                            char *experimentName, TrackProgress &my_progress);



/* Class to handle multi-threaded basecalling on a full wells file */

class BaseCaller {
public:
  BaseCaller(CommandLineOpts *_clo, RawWells *_rawWells, const char *_flowOrder, Mask *_maskPtr, int _rows, int _cols, FILE *fpLog);
  virtual ~BaseCaller();

  void FindClonalPopulation(char *experimentName, const std::vector<int>& keyIonogram);
  void DoPhaseEstimation(SequenceItem *seqList);
  void DoThreadedBasecalling(char *resultsDir, char *sffLIBFileName, char *sffTFFileName);

  // Methods in BaseCallerLogFiles.cpp
  void generateCafieRegionsFile(const char *basecaller_output_directory);

  void saveBaseCallerJson(const char *basecaller_output_directory);

protected:

  void OpenWellStatFile();
  void WriteWellStatFileEntry(MaskType bfType, int *keyFlow, int keyFlowLen, int x, int y,
      vector<weight_t> &keyNormSig, int numCalls, double cf, double ie, double dr, double multiplier,
      double ppf, bool clonal, double medianAbsCafieResidual);
  void writePrettyText(std::ostream &out);
  void writeTSV(char *filename);
  bool LoadRegion(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements, int &currentRegion, std::string &msg);
  void PrintStatus(std::deque<int> &wellX, std::deque<int> &wellY, int &currentRegion);

  friend void *doBasecall(void *input);
  void BasecallerWorker(std::deque<int> &wellX, std::deque<int> &wellY, std::deque<std::vector<float> > &wellMeasurements, PerBaseQual &pbq, int currentRegion);

  // Important outside entities accessed by BaseCaller
  CommandLineOpts         *clo;
  RawWells                *rawWells;
  Mask                    *maskPtr;

  // General run parameters
  int                     rows;
  int                     cols;
  int                     numRegions;
  int                     numFlows;
  string                  flowOrder;
  ChipIdEnum              chipID;

  // Information about read classes
  const static int        numClasses = 2;
  string                  className[numClasses];
  int                     classKeyFlows[numClasses][MAX_KEY_FLOWS];
  int                     classKeyFlowsLength[numClasses];
  string                  classKeyBases[numClasses];
  int                     classKeyBasesLength[numClasses];

  // Parameters that are estimated before basecalling can begin
  clonal_filter           clonalPopulation;
  vector<float>           cf;
  vector<float>           ie;
  vector<float>           droop;

  // Thread management
  unsigned int            numWorkers;
  pthread_mutex_t         *commonOutputMutex;

  // Wells reading
  RegionWellsReader       wellsReader;
  int                     nextRegionX;
  int                     nextRegionY;

  // Basecalling results saved here
  RawWells                *phaseResid;
  FILE                    *wellStatFileFP;
  unsigned int            numWellsCalled;
  ofstream                filterStatus;
  OrderedRegionSFFWriter  libSFF;
  OrderedRegionSFFWriter  tfSFF;
  set<well_index_t>       randomLibSet;
  OrderedRegionSFFWriter  randomLibSFF;

  // Filtering flags and stats
  bool                    classFilterPolyclonal[numClasses];
  bool                    classFilterHighResidual[numClasses];
  well_index_t            classCountPolyclonal[numClasses];
  well_index_t            classCountHighPPF[numClasses];
  well_index_t            classCountZeroBases[numClasses];
  well_index_t            classCountTooShort[numClasses];
  well_index_t            classCountFailKeypass[numClasses];
  well_index_t            classCountHighResidual[numClasses];
  well_index_t            classCountValid[numClasses];
  well_index_t            classCountTotal[numClasses];

  Json::Value             basecallerJson;

};

void *doBasecall(void *input);


double getPPF(hpLen_vec_t &predictedExtension, unsigned int nFlowsToAssess);
double getMedianAbsoluteCafieResidual(std::vector<weight_t> &residual, unsigned int nFlowsToAssess);



#endif // BASECALLER_H
