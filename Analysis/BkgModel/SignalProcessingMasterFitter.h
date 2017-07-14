/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SIGNALPROCESSINGMASTERFITTER_H
#define SIGNALPROCESSINGMASTERFITTER_H

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


#include "BkgTrace.h"
#include "EmptyTraceTracker.h"
#include "LevMarState.h"
#include "BkgSearchAmplitude.h"
#include "DataCube.h"
#include "XtalkCurry.h"
#include "SpatialCorrelator.h"

#include "DebugWriter.h"
#include "GlobalWriter.h"

#include "RegionalizedData.h"
#include "SlicedChipExtras.h"


// SHOULD be called after all SignalProcessingMasterFitter objects are deleted to free up memory
void CleanupLevMarSparseMatrices (void);


// forward declaration to make function calls happy
class BkgModelCuda;
class MultiFlowLevMar;
class Axion;
class RefineFit;
class SpatialCorrelator;
class RefineTime;
class ExpTailDecayFit;
class TraceCorrector;

/*
// declare save_construct_data
class SignalProcessingMasterFitter;
namespace boost {namespace serialization {
    template<class Archive>
      inline void save_construct_data(Archive& ar, const SignalProcessingMasterFitter * o, const unsigned int version);
    template<class Archive>
      inline void load_construct_data(Archive& ar, SignalProcessingMasterFitter * o, const unsigned int version);
  }}
*/

class SignalProcessingMasterFitter
{
  public:

    // Forward declaration of the CUDA model which needs private access to the CPU model
    // too many friends!  Need to clean up the relationships between all these objects
    // i.e. >data< vs >control flow< vs >fitting<
    friend class BkgModelCuda;
    friend class MultiFlowLevMar;
    friend class Axion;
    friend class RefineFit;
    friend class SpatialCorrelator;
    friend class RefineTime;
    friend class ExpTailDecayFit;
    friend class TraceCorrector;
    friend class debug_collection;

    // constructor used by Analysis pipeline

    SignalProcessingMasterFitter (RegionalizedData *local_patch, 
        const SlicedChipExtras & local_extras, GlobalDefaultsForBkgModel &_global_defaults,
        const char *_results_folder, Mask *_mask, PinnedInFlow *_pinnedInFlow, 
        class RawWells *_rawWells, 
        Region *_region, std::set<int>& sample,
        const std::vector<float>& sep_t0_est,bool debug_trace_enable,
        int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps,  
        EmptyTraceTracker *emptyTraceTracker,
        float sigma_guess,float t0_guess, float t0_frame_guess, 
				bool ignorekey/*=false*/,
				SequenceItem* seqList/*=NULL*/,int numSeqListItems/*=2*/,
        bool restart/*=false*/, int16_t *_washout_flow/*=NULL*/,
        const CommandLineOpts *_inception_state );

    void SetUpFitObjects();

    void SetComputeControlFlags (bool enable_xtalk_correction=true);

    void writeDebugFiles (bool debug) { write_debug_files = debug; }
    void   UpdateBeadBufferingFromExternalEstimates (std::vector<float> *tauB, std::vector<float> *tauE);

    virtual ~SignalProcessingMasterFitter();

    // image process entry point used by Analysis pipeline
    // trigger computation
    //bool TestAndExecuteBlock (int flow, bool last);
    //void FitUpstreamModel (int flow, bool last);
    void MultiFlowRegionalFitting ( int flow, bool last, int flow_key, int flow_block_size , master_fit_type_table *table, int flow_block_start );
    void FitAllBeadsForInitialFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );
    void RemainingFitStepsForInitialFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );
    void FitEmbarassinglyParallelRefineFit( int flow_block_size, int flow_block_start );
    void PreWellCorrectionFactors( bool ewscale_correct, int flow_block_size, int flow_block_start );
    void ExportAllAndReset(int flow, bool last, int flow_block_size, const PolyclonalFilterOpts & opts, int flow_block_id, int flow_block_start );
    bool TestAndTriggerComputation(bool last);
    //void ExecuteBlock (int flow, bool last);

    // Whatever initialization we need when the flow block size might change.
    void InitializeFlowBlock( int flow_block_size );
   
    // break apart image processing and computation
    bool ProcessImage (Image *img, int raw_flow, int flow_buffer_index, int flow_block_size);

    // ProcessImage had to be broken into two function, before and after GPUGenerateBeadTraces.
    bool InitProcessImageForGPU ( Image *img, int raw_flow, int flow_buffer_index );
    bool FinalizeProcessImageForGPU ( int flow_block_size );


    // allow GPU code to trigger PCA Dark Matter Calculation on CPU
    void CPU_DarkMatterPCA( int flow_block_size, int flow_block_start );

    // image process entry point for testing outside of Analysis pipeline
    // (doesn't require Image object, takes well data and background data separately)
    // bool  ProcessImage (short *img, short *bkg, int flow, bool last);

    void SetupTimeAndBuffers (float sigma_guess,
                              float t_mid_nuc_guess,
                              float t0_offset, int flow_block_size, int global_flow_block_size );

    void SetPointers (BkgDataPointers *ptrs)
    {
      global_state.SetHdf5Pointer( ptrs );
    }

    void SetImageParams (int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps)
    {
      region_data->my_trace.SetImageParams (_rows,_cols,_frames,_uncompFrames,_timestamps);
    }


    void GetRegParams (struct reg_params *pOut)
    {
      // only for backward compatibility of Obsolete/bkgFit.cpp
      if (pOut != NULL)
      {
        memcpy (pOut,& (region_data->my_regions.rp),sizeof (struct reg_params));
      }
    }

    void GetRegParams (struct reg_params &pOut)
    {
      //ION_ASSERT(sizeof(pOut) == sizeof(reg_params), "sizeof(pOut) != sizeof(reg_params)");
        // avoid the variable sizeof(reg_params) problem, if there's one here, due to the align(16) statement in the struc
        if (sizeof (pOut) == sizeof (reg_params))
          memcpy (&pOut,& (region_data->my_regions.rp),sizeof (struct reg_params));
        else   // if size diff due to align(16) in reg_params
        {
          std::cout << "Warning in SignalProcessingMasterFitter::GetRegParams(): sizeof(pOut)=" << sizeof (pOut) << " != sizeof(reg_params)=" << sizeof (reg_params) << std::endl;
          pOut = region_data->my_regions.rp;
        }
    }

    const Region *GetRegion (void)
    {
      return region_data->region;
    }

    // evaluates the model function for a set of well and region parameters using the background
    // data already stored in the SignalProcessingMasterFitter object from previous calls to ProcessImage
    int GetModelEvaluation (int iWell,BeadParams *p,struct reg_params *rp,
                            float **fg,float **bg,float **feval,float **isig,float **pf);

    /*
    // Is direct access needed? Current initialization is via BootUpXtalkSpec()
    // It could be moved here if we sitch to always-read-from-file mode for xtalk paramters initialization
    void SetXtalkName (char *my_name)
    {
      char xtalk_name[512];
      strcpy (xtalk_name,my_name);
      trace_xtalk_spec.ReadCrossTalkFromFile (xtalk_name); //happens after initialization
    };
    */

    void DumpExemplarBead (FILE *my_fp, bool debug_only, int flow_block_size)
    {
      if (region_data->region!=NULL)
        region_data->my_beads.DumpBeads (my_fp,debug_only, region_data->region->col, region_data->region->row, flow_block_size);
    }
    void DumpDarkMatterTitle (FILE *my_fp)
    {
      region_data->my_regions.missing_mass.DumpDarkMatterTitle (my_fp);
    }
    void DumpDarkMatter (FILE *my_fp)
    {
      if (region_data->region!=NULL)
        region_data->my_regions.missing_mass.DumpDarkMatter (my_fp,region_data->region->col,region_data->region->row,region_data->my_regions.rp.darkness[0]);
    }

    void DumpExemplarBeadDcOffset (FILE *my_fp, bool debug_only)
    {
      if (region_data->region!=NULL)
        region_data->my_trace.DumpBeadDcOffset (my_fp, debug_only, region_data->my_beads.DEBUG_BEAD, region_data->region->col,region_data->region->row,region_data->my_beads);
    }
    void DumpTimeAndEmphasisByRegion (FILE *my_fp)
    {
      region_data->DumpTimeAndEmphasisByRegion (my_fp);
    };
    void DumpTimeAndEmphasisByRegionH5 (int r, int max_frames)
    {
      global_state.DumpTimeAndEmphasisByRegionH5 (r,region_data->time_c,region_data->emphasis_data, max_frames);
    };
    void DumpBkgModelBeadFblkInfo (int r);

    
    void SetPoissonCache (PoissonCDFApproxMemo *poiss)
    {
      math_poiss = poiss;
    }

    // functions to access the private members of SignalProcessingMasterFitter
    int get_trace_imgFrames()
    {
      return region_data->my_trace.imgFrames;
    }

    int get_time_c_npts()
    {
      return region_data->time_c.npts();
    }
    float get_t_mid_nuc_start()
    {
      return region_data->t_mid_nuc_start;
    }
    float get_sigma_start()
    {
      return region_data->sigma_start;
    }
    // why are we not accessing the region directly here?
    int get_region_col()
    {
      return region_data->region->col;
    }
    int get_region_row()
    {
      return region_data->region->row;
    }

    GlobalDefaultsForBkgModel & getGlobalDefaultsForBkgModel()
    {
      return global_defaults;
    }

    XtalkCurry& getXtalkExecute()
    {
      return trace_xtalk_execute;
    }
    GlobalWriter &GetGlobalStage() { return global_state; }

    const WellXtalk & getWellXTalk() const {return well_xtalk_corrector.my_xtalk; }
    const TraceCrossTalkSpecification & getTraceXTalkSpecs() const {return trace_xtalk_spec; }

	void setWashoutThreshold(float threshold) { washout_threshold = threshold; }
	float getWashoutThreshold() { return washout_threshold; }
	void setWashoutFlowDetection(int detection) { washout_flow_detection = detection; }
	int getWashoutFlowDetection() { return washout_flow_detection; }

// making this public for temporary simplicity
    // Data and parameters here --------------
    RegionalizedData *region_data;
    SlicedChipExtras region_data_extras;

    void SetFittersIfNeeded(); // temporarily made public for new GPU pipeline ToDo: find beter way

  private:

    bool LoadOneFlow (Image *img, int flow);

    void AllocateRegionData();

    bool TriggerBlock (bool last);
    bool NeverProcessRegion();
    bool IsFirstBlock(int flow);
    
    void DoPreComputationFiltering( int flow_block_size);    
    void PostModelProtonCorrection( int flow_block_size, int flow_block_start );
    void CompensateAmplitudeForEmptyWellNormalization( int flow_block_size );

    void NothingInit();

    void BkgModelInit( bool debug_trace_enable,float sigma_guess,
                       float t0_guess, float t0_frame_guess, 
                       const std::vector<float>& sep_t0_est,std::set<int>& sample,bool nokey, 
                       SequenceItem* _seqList,int _numSeqListItems, bool restart );


    void NothingFitters();
    void DestroyFitObjects();

    void InitXtalk();

    // when we're ready to do something to a block of Flows
    // void FitModelForBlockOfFlows (int flow, bool last);

    // export data of various types per flow
    void ExportStatusToMask(int flow);
    void ExportDataToWells( int flow_block_start );
    void ExportDataToDataCubes (bool last, int last_flow, int flow_block_id, int flow_block_start );

    void UpdateClonalFilterData (int flow, const PolyclonalFilterOpts & opts, int flow_block_size, int flow_block_start );
    void ResetForNextBlockOfData();

    // emphasis vector stuff
    /*  void    GenerateEmphasis(float amult,float width, float ampl);*/
    void    ReadEmphasisVectorFromFile (void);
    void    SetTimeAndEmphasis (float t0_offset);

    /* Older function set */
    void CPUxEmbarassinglyParallelRefineFit( int flow_block_size, int flow_block_start );
    
    /* Relevant functions to integrate GPU multi flow fitting into signal processing pipeline*/
    void RefineAmplitudeEstimates (double &elapsed_time, Timer &fit_timer, int flow_block_size, int flow_block_start );
    void ApproximateDarkMatter( const LevMarBeadAssistant & post_key_state, bool isSampled, int flow_block_size, int flow_block_start);
    void FitAmplitudeAndDarkMatter (MultiFlowLevMar & lev_mar_fit, double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, int flow_block_start );
    void PostKeyFitWithRegionalSampling (MultiFlowLevMar &post_key_fit, double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, int flow_block_start);
    void PostKeyFitNoRegionalSampling (MultiFlowLevMar &post_key_fit, double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, int flow_block_start);
    void PostKeyFitAllWells(double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );
     void SetupAllWellsFromSample(int flow_block_size);
    void FitWellParametersConditionalOnRegion ( MultiFlowLevMar & lev_mar_fit, double &elapsed_time, Timer &fit_timer);
    void BootUpModel (double &elapsed_time,Timer &fit_timer, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start);
    void FirstPassSampledRegionParamFit( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );
    void FirstPassRegionParamFit( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );
    void PickRepresentativeHighQualityWells();
    void GuessCrudeAmplitude (double &elapsed_time, Timer &fit_timer, bool sampledOnly, int flow_block_size, int flow_block_start);
    void FitTimeVaryingRegion (double &elapsed_time, Timer &fit_timer, int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start);
    void RegionalFittingForInitialFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );
    void RegionalFittingForLaterFlowBlock( int flow_key, int flow_block_size, master_fit_type_table *table, int flow_block_start );

    void ChooseSampledForRegionParamFit( int flow_block_size );

    // debugging functions

    void    DumpRegionParameters();


    // end debugging functions


    // things that fit the model after here ----------------

    // pointer to the actual common defaults entity at top level
    GlobalDefaultsForBkgModel &global_defaults;
    const CommandLineOpts* inception_state;

    // talking to external world
    GlobalWriter global_state;
    
    debug_collection my_debug;
    DebugSaver debugSaver;
    
    // cache math that all bkgmodel objects need to execute
    PoissonCDFApproxMemo *math_poiss;

// local region cross-talk parameters - may vary across the chip by region
    TraceCrossTalkSpecification trace_xtalk_spec;
    XtalkCurry trace_xtalk_execute;
    SpatialCorrelator well_xtalk_corrector;

    // corrector for proton
    RefineTime *refine_time_fit;
    TraceCorrector *trace_bkg_adj;
    // buffering corrector for proton
    ExpTailDecayFit *refine_buffering;

    // optimizers for generating fits to data
    SearchAmplitude my_search; // crude/quick guess at amplitude

    // this controls refining the fit for a collection of beads
    RefineFit *refine_fit;

    // specialized residual fitter
    Axion *axion_fit;
   
    // flag controlling whether to write debug file pertaining to bead and region params
    bool write_debug_files; 

    float washout_threshold; 
    int washout_flow_detection; 

 private:
    SignalProcessingMasterFitter();
    /*
    SignalProcessingMasterFitter(GlobalDefaultsForBkgModel& obj)
      : global_defaults (obj) {
        math_poiss = NULL;
	lev_mar_fit = NULL;
	refine_fit = NULL;
	axion_fit = NULL;
	correct_spatial = NULL;
	refine_time_fit = NULL;
	region_data = NULL;
	trace_bkg_adj = NULL;
    }

    // Serialization section
    friend class boost::serialization::access;

    template<typename Archive>
      friend void boost::serialization::save_construct_data(Archive& ar, const SignalProcessingMasterFitter * o, const unsigned int version);
    template<class Archive>
      friend void boost::serialization::load_construct_data(Archive& ar, SignalProcessingMasterFitter * o, const unsigned int version);

    template<typename Archive>
      void serialize(Archive& ar, const unsigned int version) const {
      fprintf(stdout, "Serialize SignalProcessingMasterFitter\n");
        ar & 
	  const_cast<GlobalWriter &>(global_state) &
	  // my_debug &
	  // math_poiss &
	  const_cast<CrossTalkSpecification &>(xtalk_spec);
	// xtalk_execute &
	fprintf(stdout, "Serialization, SignalProcessingMasterFitter: need to rebuild my_debug (debug_collection), math_poiss (PoisonCDFApproxMemo), xtalk_execute (XtalkCurry)\n");
    }
    */
};

/*
namespace boost { namespace serialization {
    template<typename Archive>
      inline void save_construct_data(Archive& ar, const SignalProcessingMasterFitter * o, const unsigned int version){
      fprintf(stdout, "Serialization: save_construct_data SignalProcessingMasterFitter\n");
      GlobalDefaultsForBkgModel * gb = & o->global_defaults;
      ar << gb;
    }

    template<typename Archive>
      inline void load_construct_data(Archive& ar, SignalProcessingMasterFitter * o, const unsigned int version){
      // will this give us scoping issues???
      GlobalDefaultsForBkgModel *global_defaults_ref;
      ar >> global_defaults_ref;
      ::new(o) SignalProcessingMasterFitter(*global_defaults_ref);
    }
  }
}
*/

#endif // SIGNALPROCESSINGMASTERFITTER_H
