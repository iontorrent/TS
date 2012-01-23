/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PROCESSIMAGETOWELL_H
#define PROCESSIMAGETOWELL_H


#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <limits.h>
#include <signal.h>
#include <vector>
#include <set>
#include <algorithm>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>

#include "cudaWrapper.h"
#include "Flow.h"
#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Separator.h"
#include "BkgModel.h"
#include "GaussianExponentialFit.h"
#include "WorkerInfoQueue.h"
#include "Stats.h"
#include "SampleStats.h"
#include "ReservoirSample.h"
#include "SampleQuantiles.h"
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "ImageLoader.h"
#include "WellFileManipulation.h"
#include "DifferentialSeparator.h"
#include "TrackProgress.h"
#include "SpecialDataTypes.h"
#include "RegionTimingCalc.h"

typedef std::pair<int, int> beadRegion;
typedef std::vector<beadRegion> regionProcessOrderVector;
bool sortregionProcessOrderVector(const beadRegion& r1, const beadRegion& r2);

void SetUpWholeChip(Region &wholeChip,int rows, int cols);
void FixCroppedRegions(CommandLineOpts &clo, ImageSpecClass &my_image_spec);

void NNSmoothT0Estimate(Mask *mask,int imgRows,int imgCols,std::vector<float> &sep_t0_est,std::vector<float> &output_t0_est);


struct BkgModelWorkInfo
{
  int type;
  BkgModel *bkgObj;
  int flow;
  Image *img;
  bool last;
  bool learning;
};

struct SeparatorWorkInfo
{
  int type;
  char label[10];
  Separator *separator;
  Image *img;
  Region *region;
  Mask *mask;
};

struct ImageInitBkgWorkInfo
{
  int type;
  //AvgKeyIncorporation *kic;
  // possible replacement for kic
  int t_mid_nuc;
  int t_sigma;
  int numRegions;
  Region *regions;
  int r;
  char *experimentName;
  Mask *localMask;
  Mask *maskPtr;
  RawWells *rawWells;
  int rows;
  int cols;
  int maxFrames;
  int uncompFrames;
  int *timestamps;
  RawWells *bkgDbg1;
  RawWells *bkgDbg2;
  RawWells *bkgDebugKmult;
  CommandLineOpts *clo;
  SequenceItem *seqList;
  int numSeqListItems;
  std::vector<float> *sep_t0_estimate;
  PoissonCDFApproxMemo *math_poiss;
  BkgModel **BkgModelFitters;
  std::set<int> *sample;
};


// allocate compute resources for our analysis
struct ComputationPlanner
{
  int numBkgWorkers;
  int numDynamicBkgWorkers;
  int numBkgWorkers_gpu;
  bool use_gpu_acceleration;
  bool dynamic_gpu_balance;
  float gpu_work_load;
  bool use_all_gpus;
  std::vector<int> valid_devices;
  regionProcessOrderVector region_order;
  // dummy
  int lastRegionToProcess;
};

struct ProcessorQueue
{
  // Create array to hold gpu_info
  std::vector<BkgFitWorkerGpuInfo> gpu_info;
  WorkerInfoQueue *threadWorkQ;
  WorkerInfoQueue *threadWorkQ_gpu;
  DynamicWorkQueueGpuCpu* dynamicthreadWorkQ;
  // our one item instance
  WorkerInfoQueueItem item;
};



//lightweight class to handle debugging wells: this & bubbles need to be rethought sometime
class dbgWellTracker
{
public:
  RawWells *bkgDbg1;
  RawWells *bkgDbg2;
  RawWells *bkgDebugKmult;

  dbgWellTracker();
  void Init(char *experimentName, int rows, int cols, int numFlows, Flow *flw);
  void Close();
  ~dbgWellTracker();
};


void ExportSubRegionSpecsToImage(CommandLineOpts &clo);
void UpdateBeadFindOutcomes(Mask *maskPtr, Region &wholeChip, char *experimentName, CommandLineOpts &clo, int update_stats);
void SetExcludeMask(CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols);
void MakeSymbolicLinkToOldDirectory(CommandLineOpts &clo, char *experimentName);
void LoadBeadMaskFromFile(CommandLineOpts &clo, Mask *maskPtr, int &rows, int &cols);
void CopyFilesForReportGeneration(CommandLineOpts &clo, char *experimentName, SequenceItem *seqList);

extern void *DynamicBkgFitWorker(void *arg, bool use_gpu);
void* BkgFitWorkerCpu(void *arg);
void* DynamicBkgFitWorkerCpu(void *arg);
extern void *BkgFitWorker(void *arg, bool use_gpu);
bool CheckBkgDbgRegion(Region *r,CommandLineOpts *clo);

void PlanComputation(ComputationPlanner &my_compute_plan, CommandLineOpts &clo);

void SetRegionProcessOrder(int numRegions, BkgModel** fitters, ComputationPlanner &analysis_compute_plan);
void AllocateProcessorQueue(ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions);
void AssignQueueForItem(ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan, CommandLineOpts &clo, int r);

void WaitForRegionsToFinishProcessing(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan, CommandLineOpts &clo, int flow);
void SpinDownCPUthreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
void SpinUpGPUAndDynamicThreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan, CommandLineOpts &clo);
void UnSpinGpuThreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);

void SetUpImageLoaderInfo(ImageLoadWorkInfo &glinfo, CommandLineOpts &clo, Mask &localMask, ImageTracker &my_img_set, ImageSpecClass &my_image_spec, bubbleTracker &my_bubble_tracker, int numFlows);


void SetBkgModelGlobalDefaults(CommandLineOpts &clo, char *chipType,char *experimentName);
void DoThreadedBackgroundModel(RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, Flow *flw, char *experimentName, int numFlows, char *chipType,
                               ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, RegionTiming *region_timing, SequenceItem* seqlist,int numSeqListItems);
                               
  void IncrementalWriteWells(RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int numFlows);
  
void DumpBkgModelRegionInfo(char *experimentName,BkgModel *BkgModelFitters[],int numRegions,int flow,bool last_flow);
void DumpBkgModelBeadInfo(char *experimentName, BkgModel *BkgModelFitters[], int numRegions, int flow, bool last_flow);

void SetUpRegions(Region *regions, int rows, int cols, int xinc, int yinc);
void SetUpRegionDivisions(CommandLineOpts &clo, int rows, int cols);

void DoDiffSeparatorFromCLO(DifferentialSeparator *diffSeparator, CommandLineOpts &clo, Mask *maskPtr, string &analysisLocation,
                            Flow *flw, SequenceItem *seqList, int numSeqListItems);

void GetFromImagesToWells(RawWells &rawWells, Mask *maskPtr,
                          CommandLineOpts &clo,
                          char *experimentName, string &analysisLocation,
                          Flow *flw, int numFlows,
                          SequenceItem *seqList, int numSeqListItems,
                          TrackProgress &my_progress, Region &wholeChip,
                          int &well_rows, int &well_cols);

#endif // PROCESSIMAGETOWELL_H
