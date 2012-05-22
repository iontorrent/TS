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

#include "BkgTrace.h"
#include "EmptyTraceTracker.h"
#include "LevMarState.h"
#include "BkgSearchAmplitude.h"
#include "DataCube.h"
#include "XtalkCurry.h"
#include "BkgModelReplay.h"

//#include "BkgDataPointers.h"
class BkgDataPointers; // forward declaration to avoid including <armadillo> which is in BkgDataPointers.h


class RawWells;
//#define FIT_DOUBLE_TAP_DATA

// Initialize lev mar sparse matrices
void InitializeLevMarSparseMatrices (int *my_nuc_block);

// SHOULD be called after all BkgModel objects are deleted to free up memory
void CleanupLevMarSparseMatrices (void);



struct debug_collection
{
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
  FILE        *region_1mer_trace_file;
  FILE        *region_0mer_trace_file;
  debug_collection()
  {
    BkgDbg1 = NULL;
    BkgDbg2 = NULL;
    BkgDebugKmult = NULL;

    trace_dbg_file = NULL;
    data_dbg_file  = NULL;
    grid_dbg_file  = NULL;
    iter_dbg_file  = NULL;
    region_trace_file = NULL;
    region_1mer_trace_file = NULL;
    region_0mer_trace_file = NULL;
  };
};

// things referred to globally
struct extern_links{
    // mask that indicate where beads/pinned wells are located
    // shared across threads, but pinning is only up to flow 0
    Mask        *bfmask;  // global information on bead type and layout

    // array to keep track of flows for which empty wells are unpinned
    // shared across threads
    PinnedInFlow *pinnedInFlow;  // not regional, set when creating this object

    // the actual output of this whole process
    RawWells    *rawWells;

    // name out output directory
    char     *dirName;
};

// forward declaration to make function calls happy
class BkgModelCuda;
class MultiFlowLevMar;
class Axion;
class RefineFit;
class TraceCorrector;

class BkgModel
{
  public:
    // cache math that all bkgmodel objects need to execute
    PoissonCDFApproxMemo *math_poiss;

    // Forward declaration of the CUDA model which needs private access to the CPU model
    // too many friends!  Need to clean up the relationships between all these objects
    // i.e. >data< vs >control flow< vs >fitting<
    friend class BkgModelCuda;
    friend class MultiFlowLevMar;
    friend class Axion;
    friend class RefineFit;
    friend class TraceCorrector;

    
    friend class BkgModelReplay;
    friend class BkgModelReplayReader;
    friend class BkgModelReplayRecorder;

    // constructor used by Analysis pipeline



    BkgModel (char *_experimentName, Mask *_mask, PinnedInFlow *_pinnedInFlow, RawWells *_rawWells, Region *_region, std::set<int>& sample,
              std::vector<float> *sep_t0_est,bool debug_trace_enable,
              int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps, PoissonCDFApproxMemo *math_poiss, EmptyTraceTracker *emptyTraceTracker,
        BkgModelReplay *_replay,
              float sigma_guess=2.5,float t0_guess=35,float dntp_uM=50.0,
              bool enable_xtalk_correction=true,bool enable_clonal_filter=false,SequenceItem* seqList=NULL,int numSeqListItems=2);


    BkgModel (char *_experimentName, Mask *_mask, PinnedInFlow *_pinnedInFlow, RawWells *_rawWells, Region *_region, std::set<int>& sample,
              std::vector<float> *sep_t0_est, std::vector<float> *tauB, std::vector<float> *tauE,bool debug_trace_enable,
              int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps, PoissonCDFApproxMemo *math_poiss, EmptyTraceTracker *emptyTraceTracker,
              float sigma_guess=2.5,float t0_guess=35,float dntp_uM=50.0,
              bool enable_xtalk_correction=true,bool enable_clonal_filter=false,SequenceItem* seqList=NULL,int numSeqListItems=2);

    // constructor used for testing outside of Analysis pipeline (doesn't require mask, region, or RawWells obects)
    BkgModel (int numLBeads, int numFrames,
              float sigma_guess=2.5,float t0_guess=35,float dntp_uM=50.0,
              bool enable_xtalk_correction=true,bool enable_clonal_filter=false);

    virtual ~BkgModel();

    // image process entry point used by Analysis pipeline
    bool  ProcessImage (Image *img, int flow, bool last, bool learning, bool use_gpu=false);

    // image process entry point for testing outside of Analysis pipeline
    // (doesn't require Image object, takes well data and background data separately)
    bool  ProcessImage (short *img, short *bkg, int flow, bool last, bool learning, bool use_gpu=false);

    void SetImageParams (int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps)
    {
      my_trace.SetImageParams(_rows,_cols,_frames,_uncompFrames,_timestamps);
    }
    void    SetParameterDebug (RawWells *_BkgDbg1,RawWells *_BkgDbg2, RawWells *_BkgDebugKmult)
    {
      my_debug.BkgDbg1 = _BkgDbg1;
      my_debug.BkgDbg2 = _BkgDbg2;
      my_debug.BkgDebugKmult = _BkgDebugKmult;
    }
    void doDebug (char *name, float diff, float *output);
    bead_params& GetParams (int iWell)
    {
      return my_beads.params_nn[iWell];
    }
    void GetParams (int iWell,struct bead_params *pOut)
    {
      if ( (iWell >= 0) && (iWell < my_beads.numLBeads) & (pOut != NULL))
      {
        memcpy (pOut,&my_beads.params_nn[iWell],sizeof (struct bead_params));
      }
    }
    void DumpInitValues (FILE *my_fp)
    {
      if (region!=NULL)
        fprintf (my_fp, "%d\t%d\t%f\t%f\n",region->col,region->row, t_mid_nuc_start, sigma_start);
    }
    void DumpExemplarBead (FILE *my_fp, bool debug_only)
    {
      if (region!=NULL)
        my_beads.DumpBeads (my_fp,debug_only, region->col, region->row);
    }
    void DumpDarkMatterTitle (FILE *my_fp)
    {
      my_regions->missing_mass.DumpDarkMatterTitle (my_fp);
    }
    void DumpDarkMatter (FILE *my_fp)
    {
      if (region!=NULL)
        my_regions->missing_mass.DumpDarkMatter (my_fp,region->col,region->row,my_regions->rp.darkness[0]);
    }
    void DumpEmptyTrace (FILE *my_fp);
    void DumpExemplarBeadDcOffset (FILE *my_fp, bool debug_only)
    {
      if (region!=NULL)
        my_trace.DumpBeadDcOffset (my_fp, debug_only, my_beads.DEBUG_BEAD, region->col,region->row,my_beads);
    }
    void DumpTimeAndEmphasisByRegion (FILE *my_fp);
    void DumpTimeAndEmphasisByRegionH5 (int r);
    void DumpBkgModelBeadFblkInfo (int r);

    void GetRegParams (struct reg_params *pOut)
    {
      // only for backward compatibility of Obsolete/bkgFit.cpp
      if (pOut != NULL)
      {
  memcpy (pOut,&(my_regions->rp),sizeof (struct reg_params));
      }
    }

    void GetRegParams (struct reg_params &pOut)
    {
      //ION_ASSERT(sizeof(pOut) == sizeof(reg_params), "sizeof(pOut) != sizeof(reg_params)");
      if (&pOut != NULL)
      {
          // avoid the variable sizeof(reg_params) problem, if there's one here, due to the align(16) statement in the struc
          if (sizeof(pOut) == sizeof(reg_params))
      memcpy (&pOut,&(my_regions->rp),sizeof (struct reg_params));
          else { // if size diff due to align(16) in reg_params
             std::cout << "Warning in BkgModel::GetRegParams(): sizeof(pOut)=" << sizeof(pOut) << " != sizeof(reg_params)=" << sizeof(reg_params) << std::endl;
             pOut = my_regions->rp;
          }
      }
    }

    Region *GetRegion (void)
    {
      return region;
    }

    // evaluates the model function for a set of well and region parameters using the background
    // data already stored in the BkgModel object from previous calls to ProcessImage
    int GetModelEvaluation (int iWell,struct bead_params *p,struct reg_params *rp,
                            float **fg,float **bg,float **feval,float **isig,float **pf);
    void    SetAmplLowerLimit (float limit_val)
    {
      AmplLowerLimit = limit_val;
    }

    void SetXtalkName (char *my_name)
    {
      char xtalk_name[512];
      strcpy (xtalk_name,my_name);
      xtalk_spec.ReadCrossTalkFromFile (xtalk_name); //happens after initialization
    };

    int GetNumLiveBeads() const
    {
      return my_beads.numLBeads;
    }

    int GetNumHighPPF() const
    {
      return my_beads.NumHighPPF();
    }

    int GetNumPolyclonal() const
    {
      return my_beads.NumPolyclonal();
    }

    int GetNumBadKey() const
    {
      return my_beads.NumBadKey();
    }

    void performVectorization (bool perform)
    {
      use_vectorization = perform;
    }
    bool doingVectorization(){return(use_vectorization);};

// I hate all these initializers for controls
// I especially hate the static global-defaults structure.
    void SetSingleFlowFitProjectionSearch(bool enable) { global_defaults.projection_search_enable = enable; }

    int get_emptytrace_imgFrames();
    float *get_emptytrace_bg_buffers();
    float *get_emptytrace_bg_dc_offset();

  private:

    bool LoadOneFlow(Image *img, int flow);
    bool TriggerBlock(bool last);
    void ExecuteBlock(int flow, bool last, bool learning, bool use_gpu);


    void NothingInit();
    void  BkgModelInit (char *_experimentName, bool debug_trace_enable,float sigma_guess,
                        float t0_guess,float dntp_uM,std::vector<float> *sep_t0_est,std::set<int>& sample,SequenceItem* _seqList,int _numSeqListItems);

    void  BkgModelInit (char *_experimentName, bool debug_trace_enable,float sigma_guess,
                        float t0_guess,float dntp_uM,std::vector<float> *sep_t0_est,std::set<int>& sample,SequenceItem* _seqList,int _numSeqListItems,
                        std::vector<float> *tauB, std::vector<float> *tauE);

    void InitLevMar();

    void InitXtalk();
    
    // various scratch spaces
    void  AllocFitBuffers();
    void AllocTraceBuffers();

    void AddOneFlowToBuffer(int flow);
    void UpdateTracesFromImage(Image *img, int flow);
    
    // when we're ready to do something to a block of Flows
    void    FitModelForBlockOfFlows (int flow, bool last, bool learning, bool use_gpu);
    
    // export data of various types per flow
    void    FlushDataToWells(bool last);
    void    WriteAnswersToWells (int iFlowBuffer);
    void    WriteBeadParameterstoDataCubes(int iFlowBuffer, bool last);
    void    WriteDebugWells(int iFlowBuffer);

    // big optimization
    //int     MultiFlowSpecializedLevMarFitParameters(int max_iter, int max_reg_iter, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit, float lambda_start, int clonal_restriction = 0);
    // the heart of the routine - compute our traces
    void    MultiFlowComputeTotalSignalTrace (float *fval,struct bead_params *p,struct reg_params *reg_p,float *sbg=NULL);


    // emphasis vector stuff
    /*  void    GenerateEmphasis(float amult,float width, float ampl);*/
    void    ReadEmphasisVectorFromFile (void);
    void    SetTimeAndEmphasis();

    // Rezero traces
    void RezeroTraces(float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum);
    void RezeroTracesAllFlows(float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty);

    // amplitude functions


    // classify beads
    void    UpdateBeadStatusAfterFit (int flow);


#ifdef ION_COMPILE_CUDA
    void GPUFitModelForBlockOfFlows (int flow, bool last, bool learning);
    void GPUFitInitialFlowBlockModel (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUFitLaterBlockOfFlows (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPURefineAmplitudeEstimates (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUFitAmplitudeAndDarkMatter (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUPostKeyFit (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUBootUpModel (BkgModelCuda* bkg_model_cuda,double &elapsed_time,Timer &fit_timer);
    void GPUGuessCrudeAmplitude (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
    void GPUFitTimeVaryingRegion (BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer);
#endif
    void CPUFitModelForBlockOfFlows (int flow, bool last, bool learning);
    void FitInitialFlowBlockModel (double &elapsed_time, Timer &fit_timer);
    void FitLaterBlockOfFlows (int flow, double &elapsed_time, Timer &fit_timer);
    void RefineAmplitudeEstimates (double &elapsed_time, Timer &fit_timer);
    void ApproximateDarkMatter();
    void FitAmplitudeAndDarkMatter (double &elapsed_time, Timer &fit_timer);
    void PostKeyFit (double &elapsed_time, Timer &fit_timer);
    void FitWellParametersConditionalOnRegion(double &elapsed_time, Timer &fit_timer);
    void BootUpModel (double &elapsed_time,Timer &fit_timer);
    
    void PickRepresentativeHighQualityWells();
    void GuessCrudeAmplitude (double &elapsed_time, Timer &fit_timer);
    void FitTimeVaryingRegion (double &elapsed_time, Timer &fit_timer);

    void SendErrorVectorToHDF5(bead_params *p, error_track &err_t);
        // debugging functions
    char   *findName (float *ptr);
    void    DebugFileOpen (void);
    void    DebugFileClose (void);
    void    DumpRegionTrace (FILE *my_fp);
    void    DumpRegionParameters();
    void    DebugIterations();
    void    DebugBeadIteration (bead_params &eval_params, reg_params &eval_rp, int iter, int ibd);

    void JGVFitModelForBlockOfFlows(int flow);
    void JGVTraceLogger(int flow);
    void JGVAmplitudeLogger(int flow);

    // end debugging functions

    // Parameters under here!

    // the subregion this object operates on
    Region *region;

    flow_buffer_info my_flow;

    TimeCompression time_c;

    BkgTrace my_trace;  // initialized and populated by this object
    // EmptyTrace populated in ImageLoader
    EmptyTraceTracker *emptyTraceTracker; //why is this here?
    EmptyTrace *emptytrace;

// local region emphasis vectors - may vary across chip by trace
    EmphasisClass emphasis_data;


// local region cross-talk parameters - may vary across the chip by region
    CrossTalkSpecification xtalk_spec;
    XtalkCurry xtalk_execute;

    TraceCorrector *trace_bkg_adj;

    // The things we're applying optimizers to fit:
    BeadTracker my_beads;
    RegionTracker *my_regions;

    // space for processing current bead in optimization (multiflow levmar)
    // recycled for use in other optimizers
    // possibly should have more than one instance
    BeadScratchSpace my_scratch;

    // optimizers for generating fits to data
    SearchAmplitude my_search; // crude/quick guess at amplitude

    // setup stuff for lev-mar control
    FitControl_t fit_control;
    MultiFlowLevMar *lev_mar_fit;
    
    // this controls refining the fit for a collection of beads
    RefineFit *refine_fit;
    
    // specialized residual fitter
    Axion *axion_fit;


    // initial guesses for nuc rise parameters
    float   sigma_start;
    float   t_mid_nuc_start;

    // whether to skip processing for mixed reads:
    bool do_clonal_filter;
    bool use_vectorization;
    float   AmplLowerLimit;  // sadly ignored at the moment

    
    // Isolating the grab-bag of global defaults
    // shared as static variables across all instances of this class
    // this may or may not be sensible for all variables
    // but this isolates the madness in a clearly defined way
    GlobalDefaultsForBkgModel global_defaults;

    // talking to external world
    extern_links global_state;
    stand_alone_buffer SA;
    debug_collection my_debug;

    BkgModelReplay *replay;


public:
    // functions to access the private members of BkgModel
    int get_trace_imgFrames() { return my_trace.imgFrames; }
    //@TODO:  Please do not access buffers directly(!)
    //@TODO: I've gone to a lot of trouble to get my_trace separate from the main bkg model
    //@TODO: especially do not use flowBufferReadPos here as a hidden variable
    // float * get_trace_bg_buffer() { return emptytrace->get_bg_buffers(my_flow.flowBufferReadPos); }
    // please do not access buffers.
    float * get_region_darkMatter(int nuc) { return my_regions->missing_mass.dark_nuc_comp[nuc]; }
    int get_tim_c_npts() { return time_c.npts; }
    float get_t_mid_nuc_start() { return t_mid_nuc_start; }
    float get_sigma_start() { return sigma_start; }
    // why are we not accessing the region directly here?
    int get_region_col() { return region->col; }
    int get_region_row() { return region->row; }

    void SetPointers(BkgDataPointers *ptrs) { mPtrs = ptrs; }

private:
    BkgDataPointers *mPtrs;

};




#endif // BKGMODEL_H
