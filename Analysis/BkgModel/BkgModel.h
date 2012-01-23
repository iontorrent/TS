/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODEL_H
#define BKGMODEL_H

#include <stdio.h>
#include <set>
#include <vector>

#include "Image.h"
#include "BkgFitMatrixPacker.h"
#include "Region.h"
#include "FlowBuffer.h"
//#include "ExpBkgWGPFit.h"
#include "DiffEqModel.h"
#include "MultiFlowModel.h"
#include "Utils.h"
#include "BkgMagicDefines.h"
#include "BeadParams.h"
#include "BeadTracker.h"
#include "RegionParams.h"
#include "RegionTracker.h"
#include "EmphasisVector.h"
#include "GlobalDefaultsForBkgModel.h"
#include "CrossTalkSpec.h"
#include "TimeCompression.h"
#include "BkgFitOptim.h"
#include "FitControl.h"
#include "StandAlone.h"
#include "SingleFlowFit.h"
#include "BkgTrace.h"
#include "LevMarState.h"
#include "BkgSearchAmplitude.h"

class RawWells;
//#define FIT_DOUBLE_TAP_DATA

// Initialize lev mar sparse matrices
void InitializeLevMarSparseMatrices(int *my_nuc_block);

// SHOULD be called after all BkgModel objects are deleted to free up memory
void CleanupLevMarSparseMatrices(void);


struct debug_collection{
      // wells' file debug output
    RawWells    *BkgDbg1;
    RawWells    *BkgDbg2;
    RawWells    *BkgDebugKmult; // debug rate changes for softtware

    // debug output files
    FILE        *data_dbg_file;
    FILE        *trace_dbg_file;
    FILE        *grid_dbg_file;
    FILE        *iter_dbg_file;
    FILE        *region_trace_file;
    debug_collection(){
    BkgDbg1 = NULL;
    BkgDbg2 = NULL;
    BkgDebugKmult = NULL;

    trace_dbg_file = NULL;
    data_dbg_file  = NULL;
    grid_dbg_file  = NULL;
    iter_dbg_file  = NULL;
    region_trace_file = NULL;
  };
};



// forward declaration to make function calls happy
class BkgModelCuda;
class MultiFlowLevMar;

class BkgModel
{
public:
  // cache math that all bkgmodel objects need to execute
    PoissonCDFApproxMemo *math_poiss;

    // Forward declaration of the CUDA model which needs private access to the CPU model
    friend class BkgModelCuda;
    friend class MultiFlowLevMar;

    // constructor used by Analysis pipeline
 
   
    
    BkgModel(char *_experimentName, Mask *_mask, Mask *_pinnedMask, RawWells *_rawWells, Region *_region, std::set<int>& sample,
             std::vector<float> *sep_t0_est,bool debug_trace_enable,
             int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps, PoissonCDFApproxMemo *math_poiss,
             
             float sigma_guess=2.5,float t0_guess=35,float dntp_uM=50.0,
	     bool enable_xtalk_correction=true,bool enable_clonal_filter=false,SequenceItem* seqList=NULL,int numSeqListItems=2);	     

    // constructor used for testing outside of Analysis pipeline (doesn't require mask, region, or RawWells obects)
    BkgModel(int numLBeads, int numFrames, float sigma_guess=2.5,float t0_guess=35,float dntp_uM=50.0,
             bool enable_xtalk_correction=true,bool enable_clonal_filter=false);

    virtual ~BkgModel();

    // image process entry point used by Analysis pipeline
    bool  ProcessImage(Image *img, int flow, bool last, bool learning, bool use_gpu=false);

    // image process entry point for testing outside of Analysis pipeline
    // (doesn't require Image object, takes well data and background data separately)
    bool  ProcessImage(short *img, short *bkg, int flow, bool last, bool learning, bool use_gpu=false);

    void SetImageParams(int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps)
    {
        my_trace.imgRows=_rows;
        my_trace.imgCols=_cols;
        my_trace.imgFrames=_uncompFrames;
        my_trace.compFrames=_frames;
        my_trace.timestamps=_timestamps;
//            rawImgFrames=_frames;
    }
    void    SetParameterDebug(RawWells *_BkgDbg1,RawWells *_BkgDbg2, RawWells *_BkgDebugKmult) {
        my_debug.BkgDbg1 = _BkgDbg1;
        my_debug.BkgDbg2 = _BkgDbg2;
        my_debug.BkgDebugKmult = _BkgDebugKmult;
    }
    void doDebug(char *name, float diff, float *output);
    void    GetParams(int iWell,struct bead_params *pOut)
    {
        if ((iWell >= 0) && (iWell < my_beads.numLBeads) & (pOut != NULL))
        {
            memcpy(pOut,&my_beads.params_nn[iWell],sizeof(struct bead_params));
        }
    }
    void DumpExemplarBead(FILE *my_fp, bool debug_only)
    {
      if (region!=NULL)
        my_beads.DumpBeads(my_fp,debug_only,region->col, region->row);
    }
    void DumpDarkMatterTitle(FILE *my_fp)
    {
      my_regions.DumpDarkMatterTitle(my_fp);
    }
    void DumpDarkMatter(FILE *my_fp)
    {
      if (region!=NULL)
        my_regions.DumpDarkMatter(my_fp,region->col,region->row);
    }
    void DumpTimeAndEmphasisByRegion(FILE *my_fp);
    void    GetRegParams(struct reg_params *pOut)
    {
        if (pOut != NULL)
        {
            memcpy(pOut,&my_regions.rp,sizeof(struct reg_params));
        }
    }
    Region *GetRegion(void) {
        return region;
    }

    // evaluates the model function for a set of well and region parameters using the background
    // data already stored in the BkgModel object from previous calls to ProcessImage
    int GetModelEvaluation(int iWell,struct bead_params *p,struct reg_params *rp,
                           float **fg,float **bg,float **feval,float **isig,float **pf);
    void    SetAmplLowerLimit(float limit_val) {
        AmplLowerLimit = limit_val;
    }
    void    SetKrateConstraintType(int relaxed) {
        relax_krate_constraint = relaxed;    // what relaxation level should we be at?
    }
    void SetXtalkName(char *my_name) {
      char xtalk_name[512];
        strcpy(xtalk_name,my_name);
        xtalk_spec.ReadCrossTalkFromFile(xtalk_name); //happens after initialization
    };

    int GetNumLiveBeads() const {
        return my_beads.numLBeads;
    }
    void performVectorization(bool perform) {
        use_vectorization = perform;
    }
private:

    void  BkgModelInit(char *_experimentName, bool debug_trace_enable,float sigma_guess,
                         float t0_guess,float dntp_uM,std::vector<float> *sep_t0_est,std::set<int>& sample);
			 
    // various scratch spaces
    void  AllocFitBuffers();

// time functions
    void    SetupTimeCompression(void);

    // when we're ready to do something to a block of Flows
    void    FitModelForBlockOfFlows(int flow, bool last, bool learning, bool use_gpu);
    void    WriteAnswersToWells(int iFlowBuffer);

    // big optimization
    //int     MultiFlowSpecializedLevMarFitParameters(int max_iter, int max_reg_iter, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit, float lambda_start, int clonal_restriction = 0);
    // the heart of the routine - compute our traces
    void    MultiFlowComputeTotalSignalTrace(float *fval,struct bead_params *p,struct reg_params *reg_p,float *sbg=NULL);


    // compute one of our important functions of the data
    void    CalculateDarkMatter(int max_fnum);

    // track error as we optimize

    void ComputeAverageErrorAndRescaleErrorByFlow();

    // emphasis vector stuff
    /*  void    GenerateEmphasis(float amult,float width, float ampl);*/
    void    ReadEmphasisVectorFromFile(void);

    // amplitude functions

    void    FitAmplitudePerFlow(void);
    void    FitAmplitudePerBeadPerFlow(int ibd, float *nucRise, int *i_start, float *sbkg);


    void CorrectBkgBead(float *block_signal_corrected, bead_params *p, float *sbg);

    // classify beads
    void    DetectWashoutsAndWriteToMask(void);
 
    // do cross-talk
    void NewXtalkFlux(int ibd,float *my_xtflux);

    // changing parameters as we loop


    // debugging functions
    char   *findName(float *ptr);
    void    DebugFileOpen(void);
    void    DebugFileClose(void);
    void    DumpRegionTrace(FILE *my_fp);
    void    DumpRegionParameters();
    void    DebugIterations();
    void    DebugBeadIteration(bead_params &eval_params, reg_params &eval_rp, int iter, int ibd);

    
#ifdef ION_COMPILE_CUDA
    void GPUFitModelForBlockOfFlows(int flow, bool last, bool learning);
    void GPUFitInitialFlowBlockModel(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUFitLaterBlockOfFlows(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPURefineAmplitudeEstimates(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUFitAmplitudeAndDarkMatter(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUPostKeyFit(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUBootUpModel(BkgModelCuda* bkg_model_cuda,double &elapsed_time,Timer &fit_timer);
    void GPUGuessCrudeAmplitude(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUFitTimeVaryingRegion(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
 #endif
    void CPUFitModelForBlockOfFlows(int flow, bool last, bool learning);
    void FitInitialFlowBlockModel(double &elapsed_time, Timer &fit_timer);
    void FitLaterBlockOfFlows(double &elapsed_time, Timer &fit_timer);
    void RefineAmplitudeEstimates(double &elapsed_time, Timer &fit_timer);
    void FitAmplitudeAndDarkMatter(double &elapsed_time, Timer &fit_timer);
    void PostKeyFit(double &elapsed_time, Timer &fit_timer);
    void BootUpModel(double &elapsed_time,Timer &fit_timer);
    void GuessCrudeAmplitude(double &elapsed_time, Timer &fit_timer);
    void FitTimeVaryingRegion(double &elapsed_time, Timer &fit_timer);
    
    // Parameters under here!

    // the subregion this object operates on
    Region *region;

    flow_buffer_info my_flow;
    BkgTrace my_trace;
    TimeCompression time_c;

 
// local region cross-talk parameters - may vary across the chip by region
    CrossTalkSpecification xtalk_spec;

    BeadTracker my_beads;
    RegionTracker my_regions;
    
// local region emphasis vectors - may vary across chip by trace
    EmphasisClass emphasis_data;

    // space for processing current bead in optimization (multiflow levmar)
    // recycled for use in other optimizers
    BeadScratchSpace my_scratch;
    
    // optimizers for generating fits to data
    SearchAmplitude my_search;
    single_flow_optimizer my_single_fit;

    // setup stuff for lev-mar control
    FitControl_t fit_control;
    MultiFlowLevMar *lev_mar_fit;


    // masks that indicate where beads/pinned wells are located
    Mask        *bfmask;  // global information on bead type and layout
    Mask        *pinnedmask; // which beads are pinned in this flow

    // the actual output of this whole process
    RawWells    *rawWells;

    // name out output directory
    char     *dirName;
    
   // control of optimization
    // whether to perform CPU vectorization or not
    bool use_vectorization;
    int    relax_krate_constraint;
    float   AmplLowerLimit;  // sadly ignored at the moment

    stand_alone_buffer SA;
    debug_collection my_debug;
    
    
    // initial guesses for nuc rise parameters
    float   sigma_start;
    float   t_mid_nuc_start;

    // enzyme defaults
    float   dntp_concentration_in_uM;
    
    // whether to skip processing for mixed reads:
    bool do_clonal_filter;

    // Isolating the grab-bag of global defaults
    // shared as static variables across all instances of this class
    // this may or may not be sensible for all variables
    // but this isolates the madness in a clearly defined way
    GlobalDefaultsForBkgModel global_defaults;
    
    //Key Sequence Data is used for initialization
    SequenceItem *seqList;
    int numSeqListItems;
};




#endif // BKGMODEL_H
