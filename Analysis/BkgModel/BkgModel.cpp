/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <float.h>
#include <vector>
#include "LinuxCompat.h"
#include "BkgModel.h"
#include "RawWells.h"
#include "MathOptim.h"

#ifdef ION_COMPILE_CUDA
#include "BkgModelCuda.h"
#endif

#include "DNTPRiseModel.h"
#include "DiffEqModel.h"
#include "Vectorization.h"
#include "MultiLevMar.h"

#include "SampleQuantiles.h"

#define BKG_MODEL_DEBUG_DIR "/bkg_debug/"
#define EMPHASIS_AMPLITUDE 1.0

#define TUNE_INCORP_PARAMS_DEBUG


//#define FIT_ITERATION_DEBUG_TRACE

//#define READ_VECTOR_FILE

#define P_EMP_SCALE (0.61)


#ifndef n_to_uM_conv
#define n_to_uM_conv (0.000062f)
#endif

// #define LIVE_WELL_WEIGHT_BG



//
// Reads image file data and stores it in circular buffer
// This is >the< entry point for this functions
// Everything else is just setup
// this triggers the fit when enough data has accumulated
//
bool BkgModel::ProcessImage(Image *img, int flow, bool last, bool learning, bool use_gpu)
{
    if (my_beads.numLBeads == 0)
        return false;
    const RawImage *raw = img->GetImage();
    if (!raw) {
        fprintf(stderr, "ERROR: no image data\n");
        return true;
    }

    // keep track of which flow # is in which buffer
    // also keep track of which nucleotide is associated with each flow
    my_flow.buff_flow[my_flow.flowBufferWritePos] = flow;
    my_flow.flow_ndx_map[my_flow.flowBufferWritePos] = global_defaults.glob_flow_ndx_map[flow%global_defaults.flow_order_len];
    my_flow.dbl_tap_map[my_flow.flowBufferWritePos] = global_defaults.IsDoubleTap(flow);

    my_trace.GenerateAverageEmptyTrace(region,pinnedmask,bfmask,img, my_flow.flowBufferWritePos);
    my_trace.GenerateAllBeadTrace(region,my_beads,img, my_flow.flowBufferWritePos);
    my_trace.RezeroTraces(my_regions.rp.nuc_shape.t_mid_nuc,my_regions.rp.nuc_shape.sigma,MAGIC_OFFSET_FOR_EMPTY_TRACE, my_flow.flowBufferWritePos);

    // fill the bucket of buffers
    // Stores the last bits of the flow and iterates to the next buffer
    my_trace.PrecomputeBackgroundSlopeForDeriv(my_flow.flowBufferWritePos);

    // some parameters are not remembered from one flow to the next, set those back to
    // the appropriate default values
    my_beads.ResetFlowParams(my_flow.flowBufferWritePos,flow);

    my_flow.Increment();

    // When there are enough buffers filled,
    // do the computation and
    // pull out the results

    if ((my_flow.flowBufferCount < my_flow.numfb) && !last)
        return false;

    // do the fit
    FitModelForBlockOfFlows(flow,last,learning, use_gpu);

    //Write Wells data
    if (my_flow.flowBufferCount > 0) {
        while (my_flow.flowBufferCount > 0) {
            WriteAnswersToWells(my_flow.flowBufferReadPos);
            my_flow.Decrement();
        }

        if(do_clonal_filter and (flow+1)%20==0 and flow<80)
            my_beads.UpdateClonalFilter();

        if(do_clonal_filter and flow==79)
            my_beads.FindClonalReads();
    }

    // false: no error
    return false;
}




// constructor used by Analysis pipeline
BkgModel::BkgModel(char *_experimentName, Mask *_mask, Mask *_pinnedMask, RawWells *_rawWells, Region *_region, set<int>& sample,
                   vector<float> *sep_t0_est, bool debug_trace_enable,
                   int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps, PoissonCDFApproxMemo *_math_poiss,
                   float sigma_guess,float t_mid_nuc_guess,float dntp_uM,
		   bool enable_xtalk_correction,bool enable_clonal_filter,SequenceItem* _seqList,int _numSeqListItems)
{
    rawWells = _rawWells;
    region = _region;
    bfmask = _mask;
    pinnedmask = _pinnedMask;
    my_trace.imgRows=_rows;
    my_trace.imgCols=_cols;
    my_trace.imgFrames=_uncompFrames;
    my_trace.compFrames=_frames;
    my_trace.timestamps=_timestamps;
    use_vectorization = false;
    math_poiss = _math_poiss;
    xtalk_spec.do_xtalk_correction = enable_xtalk_correction;
    do_clonal_filter = enable_clonal_filter;
    seqList = _seqList;
    numSeqListItems = _numSeqListItems;
    BkgModelInit(_experimentName,debug_trace_enable,sigma_guess,t_mid_nuc_guess,dntp_uM,sep_t0_est,sample);
}

// constructor used for testing outside of Analysis pipeline (doesn't require mask, region, or RawWells obects)
BkgModel::BkgModel(int _numLBeads, int numFrames, float sigma_guess,float t_mid_nuc_guess,float dntp_uM,
                   bool enable_xtalk_correction, bool enable_clonal_filter)
{
    rawWells = NULL;
    region = NULL;
    bfmask = NULL;
    pinnedmask = NULL;

    my_beads.numLBeads = _numLBeads;

    my_trace.imgRows=0;
    my_trace.imgCols=0;
    my_trace.imgFrames=numFrames;
    my_trace.compFrames=numFrames;
    my_trace.timestamps=NULL;
    use_vectorization = false;
    // make me a local cache for math
    math_poiss = new PoissonCDFApproxMemo;
    math_poiss->Allocate(MAX_HPLEN,512,0.05);
    math_poiss->GenerateValues();
    xtalk_spec.do_xtalk_correction = enable_xtalk_correction;
    do_clonal_filter = enable_clonal_filter;
    seqList = NULL;
    numSeqListItems = 0; // old style key detection here, I guess on this testing branch for now?

    set<int> emptySample;
    BkgModelInit("",false,sigma_guess,t_mid_nuc_guess,dntp_uM,NULL,emptySample);
}








void BkgModel::BkgModelInit(char *_experimentName, bool debug_trace_enable,float sigma_guess,
                            float t_mid_nuc_guess,float dntp_uM,vector<float> *sep_t0_est, set<int>& sample)
{
  

    dirName = (char *) malloc(strlen(_experimentName) +1);
    strncpy(dirName, _experimentName, strlen(_experimentName) +1);
    sigma_start = sigma_guess;

    // ASSUMPTIONS:   15fps..  if that changes, this is hosed!!
    // use separator estimate
    t_mid_nuc_start = t_mid_nuc_guess;
    my_trace.t0_mean = t_mid_nuc_start - 1.5*sigma_guess; // in case we have no other good ideas, this should be no nucleotide
    my_trace.T0EstimateToMap(sep_t0_est,region,bfmask);

    dntp_concentration_in_uM = dntp_uM;

    //Image parameters
    my_flow.numfb = NUMFB;

    my_beads.InitBeadList(bfmask,region,seqList,numSeqListItems, sample);
    xtalk_spec.BootUpXtalkSpec((region!=NULL),global_defaults.chipType,global_defaults.xtalk_name);


    if ((my_beads.numLBeads > 0) && debug_trace_enable)
        DebugFileOpen();


    my_regions.InitRegionParams(t_mid_nuc_start,sigma_start,dntp_concentration_in_uM, global_defaults);
    
    AllocFitBuffers();
    // fix up multiflow lev_mar for operation
    lev_mar_fit = new MultiFlowLevMar(*this);
    lev_mar_fit->lm_state.SetNonIntegerPenalty(global_defaults.clonal_call_scale,global_defaults.clonal_call_penalty,MAGIC_MAX_CLONAL_HP_LEVEL);
    lev_mar_fit->lm_state.AllocateBeadFitState(my_beads.numLBeads);
    lev_mar_fit->lm_state.AssignBeadsToRegionGroups();

}

//Destructor
BkgModel::~BkgModel()
{
    DebugFileClose();
    if (lev_mar_fit!=NULL) delete lev_mar_fit;

    free(dirName);
}


void BkgModel::AllocFitBuffers()
{
    SetupTimeCompression();

    // scratch space will be used directly by fit-control derivatives
    // so we need to make sure these structures match
    my_scratch.Allocate(time_c.npts,fit_control.fitParams.NumSteps);
    my_trace.Allocate(my_flow.numfb,my_scratch.bead_flow_t,my_beads.numLBeads);
    my_trace.time_cp = &time_c; // point to the global time compression

    my_regions.AllocScratch(time_c.npts);
    // ReadEmphasisVector possibly over-ridden by set gopt?
    emphasis_data.SetDefaultValues(global_defaults.emp,global_defaults.emphasis_ampl_default, global_defaults.emphasis_width_default);
    emphasis_data.GenerateEmphasis(time_c.npts, my_regions.rp.nuc_shape.t_mid_nuc, time_c.frames_per_point,time_c.frameNumber, FINEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);

    // create matrix packing object(s)
    //Note:  scratch-space is used directly by the matrix packer objects to get the derivatives
    // so this object >must< persist in order to be used by the fit control object in the Lev_Mar fit.
    fit_control.AllocPackers(my_scratch.scratchSpace, global_defaults.no_RatioDrift_fit_first_20_flows, my_scratch.bead_flow_t, time_c.npts);
    my_single_fit.AllocLevMar(time_c,math_poiss,global_defaults.dampen_kmult);
}

//
//  Configures the BkgModel object to compress data points in time.  Points are
//  averaged together to compress them, but this is done differently for different
//  parts of the time history.
//

void BkgModel::SetupTimeCompression(void)
{
    const int LEFT_AVG = 5;
    const int START_DETAIL = -5;
    const int STOP_DETAIL = 16;
    time_c.SetUpTime(my_trace.imgFrames,t_mid_nuc_start,START_DETAIL, STOP_DETAIL, LEFT_AVG);
}

//
// Copies image data from input arrays and does the fitting (if enough data has been supplied)
//  This version of ProcessImage does not require Image objects or Mask objects to work properly
//
bool BkgModel::ProcessImage(short *img, short *bkg, int flow, bool last, bool learning, bool use_gpu)
{
    if (my_beads.numLBeads == 0)
        return false;

    // keep track of which flow # is in which buffer
    my_flow.buff_flow[my_flow.flowBufferWritePos] = flow;
    // also keep track of which nucleotide is associated with each flow
    my_flow.flow_ndx_map[my_flow.flowBufferWritePos] = global_defaults.glob_flow_ndx_map[flow%global_defaults.flow_order_len];

    my_trace.FillEmptyTraceFromBuffer(bkg,my_flow.flowBufferWritePos);
    my_trace.FillBeadTraceFromBuffer(img,my_flow.flowBufferWritePos);
    my_trace.PrecomputeBackgroundSlopeForDeriv(my_flow.flowBufferWritePos);

    // some parameters are not remembered from one flow to the next, set those back to
    // the appropriate default values
    my_beads.ResetFlowParams(my_flow.flowBufferWritePos,flow);

    my_flow.Increment();

    if ((my_flow.flowBufferCount >= my_flow.numfb) || last)
    {
        // do the fit
        FitModelForBlockOfFlows(flow,last,learning, use_gpu);
        while (my_flow.flowBufferCount > 0)
        {
          my_flow.Decrement();
        }
    }

    return false;
}

void BkgModel::BootUpModel(double &elapsed_time,Timer &fit_timer)
{

        fit_timer.restart();
        //fit_control.FitInitial 
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(2, 5, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
        elapsed_time += fit_timer.elapsed();

      float mean_copy_count = my_beads.KeyNormalizeReads();

        fit_timer.restart();
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
        elapsed_time += fit_timer.elapsed();

        mean_copy_count = my_beads.KeyNormalizeReads();

        fit_timer.restart();
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
        elapsed_time += fit_timer.elapsed();

        mean_copy_count = my_beads.KeyNormalizeReads();

        my_trace.RezeroTracesAllFlows(my_regions.rp.nuc_shape.t_mid_nuc, my_regions.rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);
        lev_mar_fit->lm_state.RestrictRegionFitToHighCopyBeads(my_beads, mean_copy_count);
}

void BkgModel::PostKeyFit(double &elapsed_time, Timer &fit_timer)
{
        fit_timer.restart();
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(1, 15, fit_control.FitPostKey, fit_control.FitRegionInit2, 1.0);
        elapsed_time += fit_timer.elapsed();

        my_trace.RezeroTracesAllFlows(my_regions.rp.nuc_shape.t_mid_nuc, my_regions.rp.nuc_shape.sigma, 4.0);
        // try to use the first non-key cycle to help normalize everything and
        // identify incorporation model parameters
        //if(do_clonal_filter)
        //    my_beads.FindClonalReads();

        fit_timer.restart();
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(1, 15, fit_control.FitPostKey, fit_control.FitRegionFull, 1.0, 5);
        elapsed_time += fit_timer.elapsed();

}

void BkgModel::FitAmplitudeAndDarkMatter(double &elapsed_time, Timer &fit_timer)
{
        // now figure out whatever remaining error there is in the fit, on average
        CalculateDarkMatter(FIRSTNFLOWSFORERRORCALC);

        my_beads.AssignEmphasisForAllBeads(emphasis_data.numEv-1);
        emphasis_data.GenerateEmphasis(time_c.npts,my_regions.rp.nuc_shape.t_mid_nuc,time_c.frames_per_point,time_c.frameNumber, CRUDEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);

        fit_timer.restart();
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(1, 15, fit_control.FitAmpl, fit_control.FitRegionSlimErr, 10.0);
        elapsed_time += fit_timer.elapsed();
}

void BkgModel::FitInitialFlowBlockModel(double &elapsed_time, Timer &fit_timer)
{
        BootUpModel(elapsed_time,fit_timer);

        PostKeyFit(elapsed_time, fit_timer);

        FitAmplitudeAndDarkMatter(elapsed_time, fit_timer);

#ifdef TUNE_INCORP_PARAMS_DEBUG
        DumpRegionParameters();
#endif

        my_regions.RestrictRatioDrift();
}

void BkgModel::GuessCrudeAmplitude(double &elapsed_time, Timer &fit_timer)
{
        emphasis_data.GenerateEmphasis(time_c.npts,my_regions.rp.nuc_shape.t_mid_nuc,time_c.frames_per_point,time_c.frameNumber, CRUDEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);

        fit_timer.restart();
        //@TODO: badly organized data still requiring too much stuff passed around
        // I blame MultiFlowModel wrapper functions that don't exist
        my_search.ParasitePointers(math_poiss, &my_trace,&my_scratch,&my_regions,&time_c,&my_flow,&emphasis_data);
        my_search.BinarySearchAmplitude(my_beads, 0.5,true); // apply search method to current bead list - wrong OO?  Method on bead list?
        elapsed_time += fit_timer.elapsed();

        my_beads.AssignEmphasisForAllBeads(emphasis_data.numEv-1);
}

void BkgModel::FitTimeVaryingRegion(double &elapsed_time, Timer &fit_timer)
{
        fit_timer.restart();
        lev_mar_fit->MultiFlowSpecializedLevMarFitParameters(0, 4, fit_control.DontFitWells, fit_control.FitRegionSlim, 1.0);
        elapsed_time += fit_timer.elapsed();
}

void BkgModel::FitLaterBlockOfFlows(double &elapsed_time, Timer &fit_timer)
{
      GuessCrudeAmplitude(elapsed_time,fit_timer);
      FitTimeVaryingRegion(elapsed_time,fit_timer);
}


void BkgModel::RefineAmplitudeEstimates(double &elapsed_time, Timer &fit_timer)
{
    // in either case, finish by fitting amplitude to finalized regional parameters
    emphasis_data.GenerateEmphasis(time_c.npts,my_regions.rp.nuc_shape.t_mid_nuc,time_c.frames_per_point,time_c.frameNumber, FINEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);
    fit_timer.restart();
    FitAmplitudePerFlow();
    elapsed_time += fit_timer.elapsed();

}

void BkgModel::CPUFitModelForBlockOfFlows(int flow, bool last, bool learning)
{
    Timer fit_timer;
    Timer total_timer;
    double elapsed_time = 0;


    if ((flow+1) <=my_flow.numfb)  // fit regions for this set of flows
    {
      FitInitialFlowBlockModel(elapsed_time,fit_timer);
    }
    else
    {
      FitLaterBlockOfFlows(elapsed_time, fit_timer);
    }

    RefineAmplitudeEstimates(elapsed_time,fit_timer);

    printf(".");
    fflush(stdout);

    DumpRegionTrace(my_debug.region_trace_file);
}

#ifdef ION_COMPILE_CUDA

void BkgModel::GPUBootUpModel(BkgModelCuda* bkg_model_cuda, double &elapsed_time,Timer &fit_timer)
{

        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(2, 5, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
        elapsed_time += fit_timer.elapsed();

        float mean_copy_count = my_beads.KeyNormalizeReads();

        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
        elapsed_time += fit_timer.elapsed();

        mean_copy_count = my_beads.KeyNormalizeReads();

        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(1, 2, fit_control.FitInitial, fit_control.FitRegionInit1, 0.1);
        elapsed_time += fit_timer.elapsed();

        mean_copy_count = my_beads.KeyNormalizeReads();

        my_trace.RezeroTracesAllFlows(my_regions.rp.nuc_shape.t_mid_nuc,  my_regions.rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);
        lev_mar_fit->lm_state.RestrictRegionFitToHighCopyBeads(my_beads,mean_copy_count);
}

void BkgModel::GPUPostKeyFit(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(1, 15, fit_control.FitPostKey, fit_control.FitRegionInit2, 1.0);
        elapsed_time += fit_timer.elapsed();

        my_trace.RezeroTracesAllFlows(my_regions.rp.nuc_shape.t_mid_nuc,  my_regions.rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);
        // try to use the first non-key cycle to help normalize everything and
        // identify incorporation model parameters
        //if(do_clonal_filter)
        //    my_beads.FindClonalReads();

        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(1, 15, fit_control.FitPostKey, fit_control.FitRegionFull, 1.0, 5);
        elapsed_time += fit_timer.elapsed();
}

void BkgModel::GPUFitAmplitudeAndDarkMatter(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
        // now figure out whatever remaining error there is in the fit, on average
        CalculateDarkMatter(FIRSTNFLOWSFORERRORCALC);

        my_beads.AssignEmphasisForAllBeads(emphasis_data.numEv-1);
        emphasis_data.GenerateEmphasis(time_c.npts,my_regions.rp.nuc_shape.t_mid_nuc,time_c.frames_per_point,time_c.frameNumber, CRUDEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);

        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(1, 15, fit_control.FitAmpl, fit_control.FitRegionSlimErr, 10.0);
        elapsed_time += fit_timer.elapsed();
}

void BkgModel::GPUFitInitialFlowBlockModel(BkgModelCuda* bkg_model_cuda, double &elapsed_time, Timer &fit_timer)
{
        GPUBootUpModel(bkg_model_cuda, elapsed_time,fit_timer);

        GPUPostKeyFit(bkg_model_cuda,elapsed_time, fit_timer);

        GPUFitAmplitudeAndDarkMatter(bkg_model_cuda,elapsed_time, fit_timer);

#ifdef TUNE_INCORP_PARAMS_DEBUG
        DumpRegionParameters();
#endif

        my_regions.RestrictRatioDrift();
}

void BkgModel::GPUGuessCrudeAmplitude(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
        emphasis_data.GenerateEmphasis(time_c.npts,my_regions.rp.nuc_shape.t_mid_nuc,time_c.frames_per_point,time_c.frameNumber, CRUDEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);

        fit_timer.restart();
        bkg_model_cuda->BinarySearchAmplitude(0.5,true);
        elapsed_time += fit_timer.elapsed();

        my_beads.AssignEmphasisForAllBeads(emphasis_data.numEv-1);
}

void BkgModel::GPUFitTimeVaryingRegion(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
        fit_timer.restart();
        bkg_model_cuda->MultiFlowSpecializedLevMarFitParameters(0, 4, fit_control.DontFitWells, fit_control.FitRegionSlim, 1.0);
        elapsed_time += fit_timer.elapsed();
}

void BkgModel::GPUFitLaterBlockOfFlows(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
      GPUGuessCrudeAmplitude(bkg_model_cuda,elapsed_time,fit_timer);
      GPUFitTimeVaryingRegion(bkg_model_cuda,elapsed_time,fit_timer);
}


void BkgModel::GPURefineAmplitudeEstimates(BkgModelCuda* bkg_model_cuda,double &elapsed_time, Timer &fit_timer)
{
    // in either case, finish by fitting amplitude to finalized regional parameters
    emphasis_data.GenerateEmphasis(time_c.npts,my_regions.rp.nuc_shape.t_mid_nuc,time_c.frames_per_point,time_c.frameNumber, FINEXEMPHASIS,emphasis_data.emphasis_width,emphasis_data.emphasis_ampl);
    fit_timer.restart();
    bkg_model_cuda->FitAmplitudePerFlow();
    elapsed_time += fit_timer.elapsed();

}
 void BkgModel::GPUFitModelForBlockOfFlows(int flow, bool last, bool learning)              
 {              
    Timer fit_timer;
    Timer total_timer;
    double elapsed_time = 0;

    //
    // Create the GPU object
    //
    // The timing of this creation is critical to a thread-safe implementation. It is a
    // requirement of CUDA that GPU memory must be allocated, accessed, and freed by the
    // same thread.  If this requirement is violated, you will see "invalid argument"
    // errors at memory management calls and computation kernels. A future work around will
    // involve using the CUDA context model that NVIDIA has proposed.
    //
    BkgModelCuda* bkg_model_cuda = NULL;

    bkg_model_cuda = new BkgModelCuda(*this, fit_control.fitParams.NumSteps, fit_control.fitParams.Steps);

    if ((flow+1) <=my_flow.numfb)  // fit regions for this set of flows
    {
      GPUFitInitialFlowBlockModel(bkg_model_cuda,elapsed_time,fit_timer);
    }
    else
    {
      GPUFitLaterBlockOfFlows(bkg_model_cuda,elapsed_time, fit_timer);
    }

    GPURefineAmplitudeEstimates(bkg_model_cuda, elapsed_time,fit_timer);

    printf(".");
    fflush(stdout);

    DumpRegionTrace(my_debug.region_trace_file);

    if (bkg_model_cuda != NULL)
        delete bkg_model_cuda;

        //printf("r(%d,%d) Livebeads: %d, GPU fit process complete in: %f (%f in MultiFlowSpecializedLevMarFitParameters)\n",
        //       region->col, region->row, my_beads.numLBeads, total_timer.elapsed(), elapsed_time);
           
 }    
 
 #endif 

               
               
// only call this when the buffers are full up and we're going to output something
// This is the big routine that fires off the optimizations we're doing
void BkgModel::FitModelForBlockOfFlows(int flow, bool last, bool learning, bool use_gpu)
{
    (void) learning;
    if (use_gpu) {}; // compiler happiness


#ifdef ION_COMPILE_CUDA
    if (use_gpu)
      GPUFitModelForBlockOfFlows(flow,last,learning);
    else
#endif
      CPUFitModelForBlockOfFlows(flow,last,learning);

    DetectWashoutsAndWriteToMask();
    // accumulate error
    // track the per-well average error
    my_beads.AllBeadAverageError();
    my_beads.LimitBeadEvolution((flow+1)<=NUMFB,MAXRCHANGE,MAXCOPYCHANGE);

}

// this puts our answers into the data structures where they belong
// should be the only point of contact with the external world, but isn't
void BkgModel::WriteAnswersToWells(int iFlowBuffer)
{
    // make absolutely sure we're upt to date
    my_regions.rp.copy_multiplier[iFlowBuffer] = pow(my_regions.rp.CopyDrift,my_flow.buff_flow[iFlowBuffer]);
    //Write one flow's data to 1.wells
    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
        float val = my_beads.params_nn[ibd].Ampl[iFlowBuffer] * my_beads.params_nn[ibd].Copies * my_regions.rp.copy_multiplier[iFlowBuffer];
        int x = my_beads.params_nn[ibd].x+region->col;
        int y = my_beads.params_nn[ibd].y+region->row;
        
        rawWells->WriteFlowgram(my_flow.buff_flow[iFlowBuffer], x, y, val);

        // debug parameter output, if necessary
        if (my_debug.BkgDbg1 != NULL || my_debug.BkgDbg2 !=NULL || my_debug.BkgDebugKmult!=NULL)
        {
            float etbR,tauB;
            int NucID = my_flow.flow_ndx_map[iFlowBuffer];
            //Everything uses the same functions to compute this so we're compatible
            etbR = AdjustEmptyToBeadRatioForFlow(my_beads.params_nn[ibd].R,&my_regions.rp,NucID,my_flow.buff_flow[iFlowBuffer]);
            tauB = ComputeTauBfromEmptyUsingRegionLinearModel(&my_regions.rp,etbR);
            if (my_debug.BkgDbg1!=NULL)
                my_debug.BkgDbg1->WriteFlowgram(my_flow.buff_flow[iFlowBuffer],x,y,tauB);
            if (my_debug.BkgDbg2!=NULL)
                my_debug.BkgDbg2->WriteFlowgram(my_flow.buff_flow[iFlowBuffer],x,y,etbR);
            if (my_debug.BkgDebugKmult!=NULL)
                my_debug.BkgDebugKmult->WriteFlowgram(my_flow.buff_flow[iFlowBuffer],x,y,my_beads.params_nn[ibd].kmult[iFlowBuffer]); // kmultiplier to go with amplitude in main
        }
    }
}


void BkgModel::CorrectBkgBead(float *block_signal_corrected, bead_params *p, float *sbg)
{

    float vb[my_scratch.bead_flow_t];
    float* vb_out[my_flow.numfb];
    float* sbgPtr[my_flow.numfb];
    float block_bkg_plus_xtalk[my_scratch.bead_flow_t]; // set up instead of shifted background
    memset(vb,0,sizeof(float[my_scratch.bead_flow_t]));

    // add cross-talk for this bead to the empty-trace
    CopyVector(block_bkg_plus_xtalk,sbg,my_scratch.bead_flow_t);
    AccumulateVector(block_bkg_plus_xtalk,my_scratch.cur_xtflux_block,my_scratch.bead_flow_t);

    // compute the zeromer
    // setup pointers into the arrays
    for (int fnum=0; fnum<my_flow.numfb; fnum++)
    {
        // remove zeromer background - just like oneFlowFit.
        // should include xtalk (0) so I can reuse this routine
        sbgPtr[fnum] = &block_bkg_plus_xtalk[fnum*time_c.npts];
        vb_out[fnum] = &vb[fnum*time_c.npts];
    }

    // do the actual calculation in parallel or not
    if (use_vectorization) {
        BlueSolveBackgroundTrace_Vec(my_flow.numfb, vb_out, sbgPtr, time_c.npts, time_c.deltaFrame,
                                     my_scratch.cur_buffer_block.tauB, my_scratch.cur_buffer_block.etbR);
    } else {
        for (int fnum=0; fnum<my_flow.numfb; fnum++)
            BlueSolveBackgroundTrace(vb_out[fnum],sbgPtr[fnum],time_c.npts,time_c.deltaFrame,
                                     my_scratch.cur_buffer_block.tauB[fnum],my_scratch.cur_buffer_block.etbR[fnum]);
    }

    MultiplyVectorByScalar(vb,p->gain,my_scratch.bead_flow_t);
    ApplyDarkMatter(vb,&my_regions.rp, my_regions.dark_matter_compensator,my_flow.flow_ndx_map,my_flow.buff_flow,time_c.npts);
    
    // zeromer computed, now remove from observed
    DiminishVector(block_signal_corrected,vb,my_scratch.bead_flow_t); // remove calculated background to produce corrected signal
}

void BkgModel::FitAmplitudePerBeadPerFlow(int ibd, float *nucRise, int *i_start, float *sbkg)
{
 
    bead_params *p = &my_beads.params_nn[ibd];
    reg_params *reg_p = &my_regions.rp;
    
    float block_signal_corrected[my_scratch.bead_flow_t];
    // Get my original trace across all flows
    FG_BUFFER_TYPE *pfg = &my_trace.fg_buffers[my_scratch.bead_flow_t*ibd];
    CopySignalForFits(block_signal_corrected, pfg,my_scratch.bead_flow_t); // temporarily place uncorrected signal here

    // calculate proton flux from neighbors
    my_scratch.ResetXtalkToZero();
    NewXtalkFlux(ibd,my_scratch.cur_xtflux_block);  

    // set up current bead parameters by flow
    FillBufferParamsBlockFlows(&my_scratch.cur_buffer_block,p,reg_p,my_flow.flow_ndx_map,my_flow.buff_flow);
    FillIncorporationParamsBlockFlows(&my_scratch.cur_bead_block, p,reg_p,my_flow.flow_ndx_map,my_flow.buff_flow);
    // make my corrected signal
    // subtract computed zeromer signal
    // uses parameters above
    CorrectBkgBead(block_signal_corrected,p, sbkg);

    for (int fnum=0;fnum < NUMFB;fnum++)
    {
        float evect[time_c.npts];
        emphasis_data.CustomEmphasis(evect, p->Ampl[fnum]);
        float *signal_corrected = &block_signal_corrected[fnum*time_c.npts];
        
        my_single_fit.FitOneFlow(fnum,evect,p,signal_corrected,nucRise, i_start,my_flow,time_c,emphasis_data,my_regions);
    }
    
}


// fits all wells one flow at a time, using a LevMarFitter derived class
// only the amplitude term is fit
void BkgModel::FitAmplitudePerFlow(void)
{
    float sbkg[my_scratch.bead_flow_t];

    my_regions.CalculateNucRiseFineStep(&my_regions.rp,time_c); // the same for the whole region because time-shift happens per well

    my_trace.GetShiftedBkg(my_regions.rp.tshift,sbkg);

    my_single_fit.FillDecisionThreshold(global_defaults.krate_adj_limit,my_flow.flow_ndx_map);
    
    for (int ibd = 0;ibd < my_beads.numLBeads;ibd++)
    {
        if (my_beads.params_nn[ibd].clonal_read or my_beads.params_nn[ibd].random_samp)
            FitAmplitudePerBeadPerFlow(ibd,my_regions.nuc_rise_fine_step,my_regions.i_start_fine_step, sbkg);
    }

    ComputeAverageErrorAndRescaleErrorByFlow();
//    printf("krate fit reduction cnt:%d amt:%f\n",krate_cnt,krate_red);
}

void BkgModel::ComputeAverageErrorAndRescaleErrorByFlow()
{
    float flow_res_mean[NUMFB] = {0.0};
      
    my_beads.MeanResidualByFlow(flow_res_mean);
    my_beads.RescaleResidualByMeanPerFlow(flow_res_mean);    
    lev_mar_fit->lm_state.ComputeAndSetAverageResidualFromMeanPerFlow(flow_res_mean);
}


void BkgModel::MultiFlowComputeTotalSignalTrace(float *fval,struct bead_params *p,struct reg_params *reg_p,float *sbg)
{
    float sbg_local[my_scratch.bead_flow_t];

    // allow the background to be passed in to save processing
    if (sbg == NULL)
    {
        my_trace.GetShiftedBkg(reg_p->tshift,sbg_local);
        sbg = sbg_local;
    }
    //@TODO possibly same for nuc_rise step
    MultiFlowComputeCumulativeIncorporationSignal(p,reg_p,my_scratch.ival,my_regions,my_scratch.cur_bead_block,time_c,my_flow,math_poiss);
    MultiFlowComputeIncorporationPlusBackground(fval,p,reg_p,my_scratch.ival,sbg,my_regions,my_scratch.cur_buffer_block,time_c,my_flow,use_vectorization, my_scratch.bead_flow_t);
}


void BkgModel::CalculateDarkMatter(int max_fnum)
{
    float sf[NUMNUC]={0.0};
    // reset unexplained background to zero
    my_regions.ResetDarkMatter();
//@TODO put nuc rise computation here
    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
        // get the current parameter values for this bead
        bead_params *ep = &my_beads.params_nn[ibd];

        if (lev_mar_fit->lm_state.well_region_fit[ibd] == false)
            continue;
        my_scratch.FillObserved(my_trace.fg_buffers, ibd);

        // evaluate the function
        MultiFlowComputeTotalSignalTrace(my_scratch.fval,ep,&my_regions.rp);

        // calculate error b/w current fit and actual data
        for (int fnum=0;fnum < max_fnum;fnum++)
        {
            // if our bead has enough signal in this flow to be interesting we include it
            // if our bead is close enough in mean-square distance we include it
            // exclude small amplitude but high mean-square beads
            bool include_me_flag = ((ep->Ampl[fnum]>=0.3) || (lev_mar_fit->lm_state.residual[ibd]<=(lev_mar_fit->lm_state.avg_resid*2)));

            if (include_me_flag)
            {                
                float eval;
                float *et = &my_regions.dark_matter_compensator[my_flow.flow_ndx_map[fnum]*time_c.npts];

                for (int i=0;i<time_c.npts;i++)
                {
                    int ti = i+fnum*time_c.npts;
                    eval = (my_scratch.observed[ti]-my_scratch.fval[ti]);
                    et[i] += eval;
                }
                sf[my_flow.flow_ndx_map[fnum]]++;
            }
        }
        
    }
    my_regions.NormalizeDarkMatter(sf,time_c.npts);
 }

// refactor to simplify
void BkgModel::NewXtalkFlux(int ibd,float *my_xtflux)
{

    if ((my_beads.ndx_map != NULL) & xtalk_spec.do_xtalk_correction)
    {
        int nn_ndx,cx,cy,ncx,ncy;

        cx = my_beads.params_nn[ibd].x;
        cy = my_beads.params_nn[ibd].y;

        // Iterate over the number of neighbors, accumulating hydrogen ions
        int nei_total = 0;
        for (int nei_idx=0; nei_idx<xtalk_spec.nei_affected; nei_idx++)
        {
            // phase for hex-packed
            if (!xtalk_spec.hex_packed)
                CrossTalkSpecification::NeighborByGridPhase(ncx,ncy,cx,cy,xtalk_spec.cx[nei_idx],xtalk_spec.cy[nei_idx], 0);
            else
                CrossTalkSpecification::NeighborByGridPhase(ncx,ncy,cx,cy,xtalk_spec.cx[nei_idx],xtalk_spec.cy[nei_idx], (region->row+cy+1) % 2); // maybe????

            if ((ncx>-1) && (ncx <region->w) && (ncy>-1) && (ncy<region->h)) // neighbor within region
            {
                if ((nn_ndx=my_beads.ndx_map[ncy*region->w+ncx])!=-1) // bead present
                {
                    // tau_top = how fast ions leave well
                    // tau_bulk = how slowly ions leave bulk over well - 'simulates' neighbors having different upstream profiles
                    // multiplier = how many ions get to this location as opposed to others
                    if (xtalk_spec.multiplier[nei_idx] > 0)
                        AccumulateSingleNeighborXtalkTrace(my_xtflux,&my_beads.params_nn[nn_ndx], &my_regions.rp, 
                                                           my_scratch, time_c, my_regions, my_flow, math_poiss, use_vectorization,
                                                           xtalk_spec.tau_top[nei_idx],xtalk_spec.tau_fluid[nei_idx],xtalk_spec.multiplier[nei_idx]);
                    nei_total++;
                }
            }
        }
    }
}


void BkgModel::DetectWashoutsAndWriteToMask()
{
  my_beads.DetectCorrupted();
  my_beads.WriteCorruptedToMask(region,bfmask);
  lev_mar_fit->lm_state.SuppressCorruptedWellFits(my_beads);
}


// debugging functions down here in the darkness
// so I don't have to read them every time I wander through the code
void BkgModel::DebugFileOpen(void)
{
    if (region == NULL)
        return;

    char *fname;
    int name_len = strlen(dirName) + strlen(BKG_MODEL_DEBUG_DIR) + 64;
    struct stat fstatus;
    int         status;

    fname = new char[name_len];

    snprintf(fname,name_len,"%s%s",dirName,BKG_MODEL_DEBUG_DIR);
    status = stat(fname,&fstatus);

    if (status != 0)
    {
        // directory does not exist yet, create it
        mkdir(fname,S_IRWXU | S_IRWXG | S_IRWXO);
    }

    snprintf(fname,name_len,"%s%sdatax%dy%d.txt",dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
    fopen_s(&my_debug.data_dbg_file,fname, "wt");

    snprintf(fname,name_len,"%s%stracex%dy%d.txt",dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
    fopen_s(&my_debug.trace_dbg_file,fname, "wt");
    fprintf(my_debug.trace_dbg_file,"Background Fit Object Created x = %d, y = %d\n",region->col,region->row);
    fflush(my_debug.trace_dbg_file);

#ifdef FIT_ITERATION_DEBUG_TRACE
    snprintf(fname,name_len,"%s%siterx%dy%d.txt",dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
    fopen_s(&my_debug.iter_dbg_file,fname,"wt");
#endif

    snprintf(fname,name_len,"%s/reg_tracex%dy%d.txt",dirName,region->col,region->row);
    fopen_s(&my_debug.region_trace_file,fname, "wt");

    delete [] fname;
}

void BkgModel::DebugFileClose(void)
{
#ifdef FIT_ITERATION_DEBUG_TRACE
    if (my_debug.iter_dbg_file != NULL)
        fclose(my_debug.iter_dbg_file);
#endif

    if (my_debug.trace_dbg_file != NULL)
        fclose(my_debug.trace_dbg_file);

    if (my_debug.data_dbg_file != NULL)
        fclose(my_debug.data_dbg_file);

    if (my_debug.region_trace_file != NULL)
        fclose(my_debug.region_trace_file);
}

void BkgModel::DebugIterations()
{
    DumpRegionParamsCSV(my_debug.iter_dbg_file,&my_regions.rp);
    my_beads.DumpAllBeadsCSV(my_debug.iter_dbg_file);
}

void BkgModel::DebugBeadIteration(bead_params &eval_params, reg_params &eval_rp, int iter, int ibd)
{
    fprintf(my_debug.trace_dbg_file,"iter:% 3d,(% 5.3f, % 5.3f,% 6.2f, % 2.1f, % 5.3f, % 5.3f, % 5.3f) ",
            iter,eval_params.gain,eval_params.Copies,lev_mar_fit->lm_state.residual[ibd],eval_rp.nuc_shape.sigma,eval_params.R,my_regions.rp.RatioDrift,my_regions.rp.CopyDrift);
    fprintf(my_debug.trace_dbg_file,"% 3.2f,% 3.2f,% 3.2f,% 3.2f,",
            eval_params.Ampl[0],eval_params.Ampl[1],eval_params.Ampl[2],eval_params.Ampl[3]);
    fprintf(my_debug.trace_dbg_file,"% 3.2f,% 3.2f,% 3.2f,% 3.2f,",
            eval_params.Ampl[4],eval_params.Ampl[5],eval_params.Ampl[6],eval_params.Ampl[7]);
    fprintf(my_debug.trace_dbg_file,"% 2.1f,% 2.1f,% 2.1f,% 2.1f,",
            eval_rp.nuc_shape.t_mid_nuc,eval_rp.nuc_shape.t_mid_nuc,eval_rp.nuc_shape.t_mid_nuc,eval_rp.nuc_shape.t_mid_nuc); // wrong! should be delayed

    fprintf(my_debug.trace_dbg_file,"(% 5.3f, % 5.3f, % 5.3f, % 5.3f) ",
            eval_rp.d[0],eval_rp.d[1],eval_rp.d[2],eval_rp.d[3]);
    fprintf(my_debug.trace_dbg_file,"(% 5.3f, % 5.3f, % 5.3f, % 5.3f, % 5.3f\n) ",
            eval_rp.krate[0],eval_rp.krate[1],eval_rp.krate[2],eval_rp.krate[3],eval_rp.sens);

    fflush(my_debug.trace_dbg_file);
}

void BkgModel::DumpRegionParameters()
{
    if (region != NULL)
        printf("r(x,y) d(T,A,C,G) k(T,A,C,G)=(%d,%d) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f) %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n",
               region->col,region->row,
               my_regions.rp.d[0],my_regions.rp.d[1],my_regions.rp.d[2],my_regions.rp.d[3],
               my_regions.rp.krate[0],my_regions.rp.krate[1],my_regions.rp.krate[2],my_regions.rp.krate[3],
               my_regions.rp.kmax[0],my_regions.rp.kmax[1],my_regions.rp.kmax[2],my_regions.rp.kmax[3],
               lev_mar_fit->lm_state.avg_resid,my_regions.rp.tshift,my_regions.rp.tau_R_m,my_regions.rp.tau_R_o,my_regions.rp.nuc_shape.sigma,my_regions.rp.nuc_shape.t_mid_nuc);
    if (region != NULL)
        printf("---(%d,%d) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f)\n",
               region->col,region->row,
               my_regions.rp.nuc_shape.t_mid_nuc_delay[0],my_regions.rp.nuc_shape.t_mid_nuc_delay[1],my_regions.rp.nuc_shape.t_mid_nuc_delay[2],my_regions.rp.nuc_shape.t_mid_nuc_delay[3],
               my_regions.rp.nuc_shape.sigma_mult[0],my_regions.rp.nuc_shape.sigma_mult[1],my_regions.rp.nuc_shape.sigma_mult[2],my_regions.rp.nuc_shape.sigma_mult[3]);
}

#define DUMP_N_VALUES(key,format,var,n) \
{\
    fprintf(my_fp,"%s",key);\
    for (int i=0;i<n;i++) fprintf(my_fp,format,var[i]);\
    fprintf(my_fp,"\n");\
}

// outputs a full trace of a region
// 1.) all region-wide parameters are output along w/ text identifiers so that they can be
//     parsed by downstream tools
// 2.) All pertinent data for a subset of live wells is output.  Per-well data is put in the trace file
//     in blocks of 20 flows (just as they are processed).  All parameters are output w/ text
//     identifiers so that they can be parsed later.  If there are more than 1000 live wells, some will
//     automatically be skipped in order to limit the debug output to ~1000 wells total.  Wells are skipped
//     in a sparse fashion in order to evenly represent the region
//
#define MAX_REGION_TRACE_WELLS  1000
void BkgModel::DumpRegionTrace(FILE *my_fp)
{
      if (my_fp){
    // if this is the first time through, dump all the region-wide parameters that don't change
    if (my_flow.buff_flow[0] == 0)
    {
        fprintf(my_fp,"reg_row:\t%d\n",region->row);
        fprintf(my_fp,"reg_col:\t%d\n",region->col);
        fprintf(my_fp,"npts:\t%d\n",time_c.npts);
        fprintf(my_fp,"tshift:\t%f\n",my_regions.rp.tshift);
        fprintf(my_fp,"tau_R_m:\t%f\n",my_regions.rp.tau_R_m);
        fprintf(my_fp,"tau_R_o:\t%f\n",my_regions.rp.tau_R_o);
        fprintf(my_fp,"sigma:\t%f\n",my_regions.rp.nuc_shape.sigma);
        DUMP_N_VALUES("krate:","\t%f",my_regions.rp.krate,NUMNUC);
        float tmp[NUMNUC];
        for (int i=0;i<NUMNUC;i++) tmp[i]=my_regions.rp.d[i];
        DUMP_N_VALUES("d:","\t%f",tmp,NUMNUC);
        DUMP_N_VALUES("kmax:","\t%f",my_regions.rp.kmax,NUMNUC);
        fprintf(my_fp,"sens:\t%f\n",my_regions.rp.sens);
        DUMP_N_VALUES("NucModifyRatio:","\t%f",my_regions.rp.NucModifyRatio,NUMNUC);
        DUMP_N_VALUES("ftimes:","\t%f",time_c.frameNumber,time_c.npts);
        DUMP_N_VALUES("error_term:","\t%f",my_regions.dark_matter_compensator,my_regions.nuc_flow_t);   // one time_c.npts-long term per nuc
        fprintf(my_fp,"end_section:\n");
        // we don't output t_mid_nuc, CopyDrift, or RatioDrift here, because those can change every block of 20 flows
    }
// TODO: dump computed parameters taht are functions of apparently "basic" parameters
// because the routines to compute them are "hidden" in the code
    // now dump parameters and data that can be unique for every block of 20 flows
    DUMP_N_VALUES("flows:","\t%d",my_flow.buff_flow,NUMFB);
    fprintf(my_fp,"CopyDrift:\t%f\n",my_regions.rp.CopyDrift);
    fprintf(my_fp,"RatioDrift:\t%f\n",my_regions.rp.RatioDrift);
    fprintf(my_fp,"t_mid_nuc:\t%f\n",my_regions.rp.nuc_shape.t_mid_nuc);
    DUMP_N_VALUES("nnum:","\t%d",my_flow.flow_ndx_map,NUMFB);
    fprintf(my_fp,"end_section:\n");

    float tmp[my_scratch.bead_flow_t];
    struct reg_params eval_rp = my_regions.rp;
    float my_xtflux[my_scratch.bead_flow_t];
    float sbg[my_scratch.bead_flow_t];

    my_trace.GetShiftedBkg(my_regions.rp.tshift,sbg);
    float skip_num = 1.0;
    if (my_beads.numLBeads > MAX_REGION_TRACE_WELLS)
    {
        skip_num = (float)(my_beads.numLBeads)/1000.0;
    }

    float skip_next = 0.0;
    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
        if ((float)ibd >= skip_next)
            skip_next += skip_num;
        else
            continue;

        struct bead_params *p = &my_beads.params_nn[ibd];
        fprintf(my_fp,"bead_row:%d\n",my_beads.params_nn[ibd].y);
        fprintf(my_fp,"bead_col:%d\n",my_beads.params_nn[ibd].x);
        float R_tmp[NUMFB],tau_tmp[NUMFB];
        for (int i=0;i < NUMFB;i++)
            R_tmp[i] = AdjustEmptyToBeadRatioForFlow(p->R,&my_regions.rp,my_flow.flow_ndx_map[i],my_flow.buff_flow[i]);
        DUMP_N_VALUES("R:","\t%f",R_tmp,NUMFB);
        for (int i=0;i < NUMFB;i++)
            tau_tmp[i] = ComputeTauBfromEmptyUsingRegionLinearModel(&my_regions.rp,R_tmp[i]);
        DUMP_N_VALUES("tau:","\t%f",tau_tmp,NUMFB);
        fprintf(my_fp,"P:%f\n",p->Copies);
        fprintf(my_fp,"gain:%f\n",p->gain);

        fprintf(my_fp,"dmult:%f\n",p->dmult);
//        fprintf(my_fp,"in_cnt:%d\n",in_cnt[my_beads.params_nn[ibd].y*region->w+my_beads.params_nn[ibd].x]);
        DUMP_N_VALUES("Ampl:","\t%f",p->Ampl,NUMFB);
        DUMP_N_VALUES("kmult:","\t%f",p->kmult,NUMFB);

        // run the model
        MultiFlowComputeTotalSignalTrace(my_scratch.fval,&my_beads.params_nn[ibd],&my_regions.rp,sbg);

        struct bead_params eval_params = my_beads.params_nn[ibd];
        memset(eval_params.Ampl,0,sizeof(eval_params.Ampl));

        // calculate the model with all 0-mers to get synthetic background by itself
        MultiFlowComputeTotalSignalTrace(tmp,&eval_params,&eval_rp,sbg);

        // calculate proton flux from neighbors
        //CalcXtalkFlux(ibd,my_xtflux);

        // output values
        DUMP_N_VALUES("raw_data:","\t%d",(&my_trace.fg_buffers[my_scratch.bead_flow_t*ibd]),my_scratch.bead_flow_t);
        DUMP_N_VALUES("fit_data:","\t%.1f",my_scratch.fval,my_scratch.bead_flow_t);
        DUMP_N_VALUES("avg_empty:","\t%.1f",sbg,my_scratch.bead_flow_t);
        DUMP_N_VALUES("background:","\t%.1f",tmp,my_scratch.bead_flow_t);
        DUMP_N_VALUES("xtalk:","\t%.1f",my_xtflux,my_scratch.bead_flow_t);
    }
    fprintf(my_fp,"end_section:\n");

    fflush(my_fp);
      } // end file exists
}

void BkgModel::doDebug(char *name, float diff, float *output)
{
#if 1
    (void) name;
    (void) diff;
    (void) output;
#else
    double avg=0;

//@TODO this is out of synch with the proper dealing with flows
    for (int i=0;i<my_scratch.bead_flow_t;i++)
        avg += *output++;

    avg /= my_scratch.bead_flow_t;
    printf("%s(%f) %.1lf ",name,diff,avg);
#endif
}

char *BkgModel::findName(float *ptr)
{
    int i;
    for (i=0;i<fit_control.fitParams.NumSteps;i++)
    {
        if (ptr >= fit_control.fitParams.Steps[i].ptr && ptr < (fit_control.fitParams.Steps[i].ptr + my_scratch.bead_flow_t))
            return (fit_control.fitParams.Steps[i].name);
    }
    return (NULL);
}

// interface to running in weird standalone mode from torrentR
int BkgModel::GetModelEvaluation(int iWell,struct bead_params *p,struct reg_params *rp,
                                 float **fg,float **bg,float **feval,float **isig,float **pf)
{
  SA.allocate(my_scratch.bead_flow_t);

    // make sure iWell is in the correct range
    if ((iWell < 0) || (iWell > my_beads.numLBeads))
    {
        return (-1);
    }

    *fg = SA.fg;
    *bg = SA.bg;
    *feval = SA.feval;
    *isig = SA.isig;
    *pf = SA.pf;

    my_trace.GetShiftedBkg(rp->tshift,SA.bg);
    // iterate over all data points doing the right thing

    //@TODO put nuc rise here
    MultiFlowComputeCumulativeIncorporationSignal(p,rp,SA.pf,my_regions,my_scratch.cur_bead_block,time_c,my_flow,math_poiss);
    MultiFlowComputeIncorporationPlusBackground(SA.feval,p,rp,SA.pf,SA.bg,my_regions,my_scratch.cur_buffer_block,time_c,my_flow, use_vectorization, my_scratch.bead_flow_t);

    // easiest way to get the incorporation signal w/ background subtracted off is to calculate
    // the function with Ampl=0 and subtract that from feval
    struct bead_params p_tmp = *p;
    memset(&p_tmp,0,sizeof(p_tmp.Ampl));
    MultiFlowComputeTotalSignalTrace(SA.isig,&p_tmp,rp,SA.bg);

    for (int fnum=0; fnum<my_flow.numfb; fnum++)
    {
        for (int i=0;i < time_c.npts;i++)
        {
            int tmp_index = i+fnum*time_c.npts;
            SA.isig[tmp_index] = SA.feval[tmp_index] - SA.isig[tmp_index];
        }
    }
    my_scratch.FillObserved(my_trace.fg_buffers,iWell);

    for (int fnum=0;fnum < my_flow.numfb;fnum++)
    {

        for (int i=0;i < time_c.npts;i++)
            SA.fg[i+fnum*time_c.npts] = my_scratch.observed[i+fnum*time_c.npts];
    }

    return (my_scratch.bead_flow_t);
}


void BkgModel::DumpTimeAndEmphasisByRegion(FILE *my_fp)
{
   // this will be a dumb file format
   // each line has x,y, type, hash, data per time point 
   // dump timing
   int max_pts = 40; // should be large enough
   int i = 0;
   fprintf(my_fp,"%d\t%d\t", region->col, region->row);
   fprintf(my_fp,"time\t%d\t",time_c.npts);
   
   for (i=0; i<time_c.npts; i++)
     fprintf(my_fp,"%f\t",time_c.frameNumber[i]);
   for (; i<max_pts; i++)
     fprintf(my_fp,"0.0\t");
   fprintf(my_fp,"\n");
   fprintf(my_fp,"%d\t%d\t", region->col, region->row);
   fprintf(my_fp,"frames_per_point\t%d\t",time_c.npts);
   for (i=0; i<time_c.npts; i++)
     fprintf(my_fp,"%d\t", time_c.frames_per_point[i]);
   for (; i<max_pts; i++)
     fprintf(my_fp,"0\t");
   fprintf(my_fp,"\n");
   // dump emphasis
   for (int el=0; el<emphasis_data.numEv; el++)
   {
    fprintf(my_fp,"%d\t%d\t", region->col, region->row);
    fprintf(my_fp,"em\t%d\t",el);
    for (i=0; i<time_c.npts; i++)
      fprintf(my_fp,"%f\t",emphasis_data.EmphasisVectorByHomopolymer[el][i]);
   for (; i<max_pts; i++)
     fprintf(my_fp,"0.0\t");
    fprintf(my_fp,"\n");
   }
}



