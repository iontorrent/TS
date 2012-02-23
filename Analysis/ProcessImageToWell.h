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
#include "SeqList.h"
#include "RegionTimingCalc.h"
#include "DataCube.h"
#include "H5File.h"
#include "BkgMagicDefines.h"

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
  float t_mid_nuc;
  float t_sigma;
  int numRegions;
  Region *regions;
  int r;
  char *experimentName;
  Mask *localMask;     // shared among threads
  Mask *maskPtr;
  short *emptyInFlow;  // shared among threads
  RawWells *rawWells;
  DataCube<float> *resError;
  DataCube<float> *kMult;
  DataCube<float> *beadOnceParam;
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
  void Init(char *experimentName, int rows, int cols, int numFlows, char *flowOrder);
  void Close();
  ~dbgWellTracker();
};

class BkgFitterTracker{
  public:
  BkgModel **BkgModelFitters;
  int numFitters;
   // queue object for fitters
    BkgModelWorkInfo *bkinfo;

  // shared math cache
    PoissonCDFApproxMemo poiss_cache; // math routines the bkg model needs to do a lot

  // how we're going to fit
    ProcessorQueue analysis_queue;
    ComputationPlanner analysis_compute_plan;
 
  // debugging file
  dbgWellTracker my_bkg_dbg_wells;
  void ThreadedInitialization (RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, Mask &localMask, short *emptyInFlow, char *experimentName,
                                                 ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing,SeqListClass &my_keys, DataCube<float> *bgResidualError, DataCube<float> *kMult, DataCube<float> *beadOnceParam);
  void ExecuteFitForFlow(int flow, ImageTracker &my_img_set, bool last);
  void PlanComputation(BkgModelControlOpts &bkg_control);
  void SpinUp();
  void UnSpinGpuThreads();
  void SetRegionProcessOrder();
  BkgFitterTracker(int numRegions);
  void DeleteFitters();
  ~BkgFitterTracker();
};

class BkgParamH5 {
 public:

  BkgParamH5() {
    kRateDS = NULL;
    resErrorDS = NULL;
    beadOnceParamDS = NULL;
    regionParamDS = NULL;
	regionParamDSExtra = NULL;
  }

  ~BkgParamH5() { Close(); }

  void DumpBkgModelRegionInfoH5(BkgModel *BkgModelFitters[], int numRegions, int flow, bool last_flow)
  {
    if ((flow+1) % NUMFB ==0 || last_flow) {
      struct reg_params rp;
      int numParam = sizeof(reg_params)/sizeof(float); // +1 naot necessary
	  //cout << "DumpBkgModelRegionInfoH5... numParam=" << numParam << ", sizeof(reg_params) " << sizeof(reg_params) << " / size(float)" << sizeof(float) << endl;

      for (int r = 0; r < numRegions; r++) {
        BkgModelFitters[r]->GetRegParams(&rp);
        float *rpp = (float *) &rp;
        
        for (int i = 0; i < numParam; i++) {
          regionalParams.at(r, i) = rpp[i];
        }
	  regionalParamsExtra.at(r, 0) = GetModifiedMidNucTime(&rp.nuc_shape,TNUCINDEX);
	  regionalParamsExtra.at(r, 1) = GetModifiedMidNucTime(&rp.nuc_shape,ANUCINDEX);
	  regionalParamsExtra.at(r, 2) = GetModifiedMidNucTime(&rp.nuc_shape,CNUCINDEX);
	  regionalParamsExtra.at(r, 3) = GetModifiedMidNucTime(&rp.nuc_shape,GNUCINDEX);
	  regionalParamsExtra.at(r, 4) = GetModifiedSigma(&rp.nuc_shape,TNUCINDEX);
	  regionalParamsExtra.at(r, 5) = GetModifiedSigma(&rp.nuc_shape,ANUCINDEX);
	  regionalParamsExtra.at(r, 6) = GetModifiedSigma(&rp.nuc_shape,CNUCINDEX);
	  regionalParamsExtra.at(r, 7) = GetModifiedSigma(&rp.nuc_shape,GNUCINDEX);
	  regionalParamsExtra.at(r, 8) = rp.nuc_shape.sigma;
	  regionalParamsExtra.at(r, 9) = 1234; // use properties instead
      }
      size_t starts[2];
      size_t ends[2];
	  //note: regionParam only has to allocate enough regionalParam.n_rows
      starts[0] = flow/NUMFB;
      starts[0] *= numRegions;
      ends[0] = starts[0] + numRegions;
	  //cout << "DumpBkgModelRegionInfoH5... flow= " << flow << endl;
      if (last_flow && (flow+1) % NUMFB != 0) {
		ends[0] = starts[0] + (flow+1) % NUMFB;
        //starts[0] = ceil((float) flow  / NUMFB);
		cout << "DumpBkgModelRegionInfoH5... lastflow= " << flow << endl;
		cout << "changing starts[0] from " << (flow/NUMFB)* numRegions << " to " << starts[0] << " at flow " << flow << endl;
      }
      starts[1] = 0;
	  ends[1] = regionalParams.n_cols; // numParam, not numParam+1
	  //cout << "DumpBkgModelRegionInfoH5... regionalParam.n_rows,n_cols=" << regionalParam.n_rows << "," << regionalParam.n_cols << " flow=" << flow << ", starts[0]=" << starts[0] << endl;
	  //cout << "DumpBkgModelRegionInfoH5... starts,end[0]=(" << starts[0] << "," << ends[0] << "), starts,end[1]=(" << starts[1] << "," << ends[1] << ")" << endl;
	  
	  ION_ASSERT(ends[0] <= regionalParams.n_rows, "ends[0] > regionalParam.n_rows");
	  ION_ASSERT(ends[1] <= regionalParams.n_cols, "ends[1] > regionalParam.n_cols");
	  //size_t size = regionalParam.n_rows * regionalParam.n_cols;
	  //size_t size = (ends[0]-starts[0]) * (ends[0]-starts[0]); // size not used at all
      arma::Mat<float> m = arma::trans(regionalParams);
      regionParamDS->WriteRangeData(starts, ends, m.memptr());

	  ends[1] = regionalParamsExtra.n_cols;
	  arma::Mat<float> m1 = arma::trans(regionalParamsExtra);
	  regionParamDSExtra->WriteRangeData(starts, ends, m1.memptr());
	  }
  }

  void IncrementalWriteParam(DataCube<float> &cube, H5DataSet *set, int flow, int saveWellsFrequency,int numFlows)
  {
    int testWellFrequency = saveWellsFrequency*NUMFB; // block size                                                                                  
    if (((flow+1) % (saveWellsFrequency*NUMFB) == 0 && (flow != 0))  || (flow+1) >= numFlows) {
      fprintf(stdout, "Writing incremental wells at flow: %d\n", flow);
      MemUsage("BeforeWrite");
      size_t starts[3];
      size_t ends[3];
      cube.SetStartsEnds(starts, ends);
      set->WriteRangeData(starts, ends, cube.GetMemPtr());
      cube.SetRange(0, cube.GetNumX(), 0, cube.GetNumY(), flow+1, flow + 1 + min(testWellFrequency,numFlows-(flow+1)));
      MemUsage("AfterWrite");
    }
  }
  
  void Init(CommandLineOpts &clo, int numFlows) {
    std::string hgBgDbgFile = ToStr(clo.sys_context.basecaller_output_directory) + "/bg_param.h5";
    h5BgDbg.Init();
    h5BgDbg.SetFile(hgBgDbgFile);
    h5BgDbg.Open(true);
    
    int blocksOfFlow = NUMFB;
    kRateMultiplier.Init(clo.loc_context.cols, clo.loc_context.rows, numFlows);
    kRateMultiplier.SetRange(0,clo.loc_context.cols, 0, clo.loc_context.rows, 0, blocksOfFlow);
    kRateMultiplier.AllocateBuffer();
    bgResidualError.Init(clo.loc_context.cols, clo.loc_context.rows, numFlows);
    bgResidualError.SetRange(0,clo.loc_context.cols, 0, clo.loc_context.rows, 0, blocksOfFlow);
    bgResidualError.AllocateBuffer();
    beadOnceParam.Init(clo.loc_context.cols, clo.loc_context.rows, 4);
    beadOnceParam.SetRange(0,clo.loc_context.cols, 0, clo.loc_context.rows, 0, beadOnceParam.GetNumZ());
    beadOnceParam.AllocateBuffer();
    
    kRateDS = h5BgDbg.CreateDataSet("/bkg/k_rate_multiplier", kRateMultiplier, 3);
    resErrorDS = h5BgDbg.CreateDataSet("/bkg/res_error", bgResidualError, 3);
    beadOnceParamDS = h5BgDbg.CreateDataSet("/bkg/bead_init_param", beadOnceParam, 3);
	hsize_t rp_dims[2], rp_chunks[2];
	int numParam = sizeof(reg_params)/sizeof(float); // +1 naot necessary
    rp_dims[1] = rp_chunks[1] = numParam;
    rp_dims[0] = clo.loc_context.numRegions * ceil(float(numFlows)/NUMFB);
    rp_chunks[0] = clo.loc_context.numRegions;
	//cout << "Init...regionalParams.n_rows,n_cols=" << regionalParams.n_rows << "," << regionalParams.n_cols << endl;
    //regionalParams.set_size(clo.loc_context.numRegions, sizeof(struct reg_params)/sizeof(float) + 1);
	//regionalParams.set_size(clo.loc_context.numRegions, sizeof(struct reg_params)/sizeof(float));
	regionalParams.set_size(rp_dims[0], rp_dims[1]);
	regionalParamsExtra.set_size(rp_dims[0], 10);
	
    regionParamDS = h5BgDbg.CreateDataSet("/bkg/region_param", 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(regionalParams.at(0,0)));
	rp_dims[1] = rp_chunks[1] = regionalParamsExtra.n_cols;
	regionParamDSExtra = h5BgDbg.CreateDataSet("/bkg/region_param_extra", 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(regionalParamsExtra.at(0,0)));
  }

  void IncrementalWrite(BkgFitterTracker &GlobalFitter, CommandLineOpts &clo, int flow, int numFlows) {
    if (regionParamDS != NULL) {
      DumpBkgModelRegionInfoH5(GlobalFitter.BkgModelFitters, clo.loc_context.numRegions, flow, flow == (numFlows -1));
      IncrementalWriteParam(kRateMultiplier,kRateDS,flow,1,numFlows);
      IncrementalWriteParam(bgResidualError,resErrorDS,flow,1,numFlows);
      if (flow+1 == NUMFB) {
        beadOnceParamDS->WriteDataCube(beadOnceParam);
      }
    }
  }

  void Close() {
    if (kRateDS != NULL) {
      kRateDS->Close();
      kRateDS = NULL;
      resErrorDS->Close();
      resErrorDS = NULL;
      beadOnceParamDS->Close();
      beadOnceParamDS = NULL;
      regionParamDS->Close();
      regionParamDS = NULL;
      regionParamDSExtra->Close();
      regionParamDSExtra = NULL;
    }
  }

  //DataCube<float> amplMultiplier; // variables A1-Ann in BkgModelBeadData.0020.txt
  DataCube<float> kRateMultiplier; // variables M1-Mnn in BkgModelBeadData.0020.txt
  DataCube<float> bgResidualError;
  DataCube<float> beadOnceParam;
  arma::Mat<float> regionalParams;
  arma::Mat<float> regionalParamsExtra;
  H5DataSet *kRateDS;
  H5DataSet *resErrorDS;
  H5DataSet *beadOnceParamDS;
  H5DataSet *regionParamDS;
  H5DataSet *regionParamDSExtra;
  H5File h5BgDbg;
};


void ExportSubRegionSpecsToImage(CommandLineOpts &clo);
void UpdateBeadFindOutcomes(Mask *maskPtr, Region &wholeChip, char *experimentName, CommandLineOpts &clo, int update_stats);
void SetExcludeMask(CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols);
void MakeSymbolicLinkToOldDirectory(CommandLineOpts &clo, char *experimentName);
void LoadBeadMaskFromFile(CommandLineOpts &clo, Mask *maskPtr, int &rows, int &cols);
void CopyFilesForReportGeneration(CommandLineOpts &clo, char *experimentName, SeqListClass &my_keys);

extern void *DynamicBkgFitWorker(void *arg, bool use_gpu);
void* BkgFitWorkerCpu(void *arg);
void* DynamicBkgFitWorkerCpu(void *arg);
extern void *BkgFitWorker(void *arg, bool use_gpu);
bool CheckBkgDbgRegion(Region *r,CommandLineOpts *clo);

void PlanMyComputation(ComputationPlanner &my_compute_plan, BkgModelControlOpts &bkg_control);

void SetRegionProcessOrder(int numRegions, BkgModel** fitters, ComputationPlanner &analysis_compute_plan);
void AllocateProcessorQueue(ProcessorQueue &my_queue,ComputationPlanner &analysis_compute_plan, int numRegions);
void AssignQueueForItem(ProcessorQueue &analysis_queue,ComputationPlanner &analysis_compute_plan, int numRegions, int r);

void WaitForRegionsToFinishProcessing(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan, int flow);
void SpinDownCPUthreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);
void SpinUpGPUAndDynamicThreads(ProcessorQueue &analysis_queue, ComputationPlanner &analysis_compute_plan);


void SetUpImageLoaderInfo(ImageLoadWorkInfo &glinfo, CommandLineOpts &clo, Mask &localMask, ImageTracker &my_img_set, ImageSpecClass &my_image_spec, bubbleTracker &my_bubble_tracker, int numFlows);


void SetBkgModelGlobalDefaults(CommandLineOpts &clo, char *chipType,char *experimentName);
void DoThreadedBackgroundModel(RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, char *experimentName, int numFlows, char *chipType,
                               ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing, SeqListClass &my_keys);
                               
void IncrementalWriteWells(RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int numFlows);
void ApplyClonalFilter(const char* experimentName, BkgModel *BkgModelFitters[], int numRegions, bool doClonalFilter, int flow);
void ApplyClonalFilter(BkgModel *BkgModelFitters[], int numRegions, const deque<float>& ppf, const deque<float>& ssq);
void GetFilterTrainingSample(deque<float>& ppf, deque<float>& ssq, BkgModel *BkgModelFitters[], int numRegions);

void DumpBkgModelRegionInfo(char *experimentName,BkgModel *BkgModelFitters[],int numRegions,int flow,bool last_flow);
void DumpBkgModelBeadInfo(char *experimentName, BkgModel *BkgModelFitters[], int numRegions, int flow, bool last_flow, bool debug_bead_only);
void DumpFilterInfo(const char* experimentName, const deque<float>& ppf, const deque<float>& ssq);

void SetUpRegions(Region *regions, int rows, int cols, int xinc, int yinc);
void SetUpRegionDivisions(CommandLineOpts &clo, int rows, int cols);

void DoDiffSeparatorFromCLO(DifferentialSeparator *diffSeparator, CommandLineOpts &clo, Mask *maskPtr, string &analysisLocation,
                            SequenceItem *seqList, int numSeqListItems);

void GetFromImagesToWells(RawWells &rawWells, Mask *maskPtr,
                          CommandLineOpts &clo,
                          char *experimentName, string &analysisLocation,
                          int numFlows,
                          SeqListClass &my_keys,
                          TrackProgress &my_progress, Region &wholeChip,
                          int &well_rows, int &well_cols);

#endif // PROCESSIMAGETOWELL_H
