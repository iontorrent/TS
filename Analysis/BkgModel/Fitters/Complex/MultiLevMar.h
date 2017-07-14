/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MULTILEVMAR_H
#define MULTILEVMAR_H

#include "SignalProcessingMasterFitter.h"


// this will be a friend of bkgmodel for now
// and look just like the cuda code :-)
// ideally this should be passed a bead tracker & a region tracker object
// and fit the current bead list & region parameters for that bead list
class MultiFlowLevMar
{
public:
    SignalProcessingMasterFitter &bkg; // reference to source class for now
    LevMarBeadAssistant lm_state;
    BeadScratchSpace lev_mar_scratch;
    incorporation_params_block_flows lev_mar_cur_bead_block;
    buffer_params_block_flows lev_mar_cur_buffer_block;

    // setup stuff for lev-mar control
    FitControl_t fit_control;
    bool use_vectorization;

    // start caches for processing
    float tshift_cache;

    float *cache_sbg;
    float *cache_slope; // derivative calculation
    

    void InitTshiftCache();
    void FillTshiftCache ( float my_tshift, int flow_block_size );
    void InitRandomCache();
    void DeleteRandomCache();

    // cache for regional parameters for derivatives
    // we recompute these for each bead needlessly
    std::vector< reg_params> step_rp;
    std::vector< NucStep> step_nuc_cache;
    void SynchRefBead(int ibd);
    void FillDerivativeStepCache ( BeadParams &eval_params, reg_params &eval_rp, unsigned int PartialDeriv_mask, int flow_block_size );
    void ComputeCachedPartialDerivatives ( BeadParams &eval_params,   BeadParams *ref_p,
                                           int ref_span,unsigned int PartialDeriv_mask, int flow_key, int flow_block_size, int flow_block_start );

    MultiFlowLevMar ( SignalProcessingMasterFitter &, int flow_block_size,
                      master_fit_type_table * table );
    ~MultiFlowLevMar();

    int MultiFlowSpecializedLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction, int flow_key, int flow_block_size, int flow_block_start  );
    int MultiFlowSpecializedSampledLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit, float lambda_start, int clonal_restriction, int flow_key, int flow_block_size, int flow_block_start );
    void MultiFlowSpecializedLevMarFitAllWells (int bead_only_iterations, BkgFitMatrixPacker *well_fit, float lambda_start, int clonal_restriction /*=0*/, int flow_key, int flow_block_size,
                int flow_block_start);
    int  MultiFlowSpecializedLevMarFitParametersOnlyRegion ( int number_region_iterations_wanted,  BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction, int flow_key, int flow_block_size, int flow_block_start );
    float   CalculateCurrentResidualForTestBeads ( int flow_key, int flow_block_size, int flow_block_start ); // within loop

    int LevMarAccumulateRegionDerivsForSampledActiveBeadList (
        reg_params &eval_rp,
        BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
        int iter, int flow_key, int flow_block_size, int flow_block_start );

    int LevMarAccumulateRegionDerivsForActiveBeadList (
        reg_params &eval_rp,
        BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
        int iter, int flow_key, int flow_block_size, int flow_block_start );

    void LevMarFitRegion ( BkgFitMatrixPacker *reg_fit, int flow_key, int flow_block_size,
                            int flow_block_start );

    float    LevMarFitToActiveBeadList (
        bool well_only_fit,
        reg_params& eval_rp,
        BkgFitMatrixPacker* well_fit,  unsigned int PartialDeriv_mask,
        int bead_iterations,
	      bool isSample, int flow_key, int flow_block_size,
        int flow_block_start );

    int DoSampledRegionIteration ( BkgFitMatrixPacker *reg_fit,
                                    int iter, int flow_key, int flow_block_size,
                                    int flow_block_start );
    int DoRegionIteration( BkgFitMatrixPacker *reg_fit,
                             int iter, int flow_key, int flow_block_size,
                             int flow_block_start );
    bool DoSampledBeadIteration ( bool well_only_fit,
                                  BkgFitMatrixPacker *well_fit,
                                  int iter, int flow_key, int flow_block_size,
                                  int flow_block_start );
    float LevMarFitToRegionalActiveBeadList (
            bool well_only_fit, reg_params &eval_rp, BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
            int iter, int flow_key, int flow_block_size, int flow_block_start );

    void LevMarBuildMatrixForBead ( int ibd,
                                    bool well_only_fit,
                                    reg_params &eval_rp, NucStep &cache_step,
                                    BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
                                    int iter, int flow_key, int flow_block_size,
                                    int flow_block_start );

    int   LevMarFitOneBead ( int ibd,
                             reg_params &eval_rp,
                             BkgFitMatrixPacker *well_fit,bool well_only_fit, 
                             int flow_key, int flow_block_size, int flow_block_start );
    void AccumulateRegionDerivForOneBead (
        int ibd, int &reg_wells,
        BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
        int iter, int flow_key, int flow_block_size, int flow_block_start );

    // do steps for computing partial derivatives
    // it is entirely likely that these functions should be functions for the "scratchspace" object
    void    ComputePartialDerivatives ( BeadParams &eval_params,
                                        BeadParams *ref_p,
                                          int ref_span,
                                        reg_params &eval_rp,  NucStep &cache_step, unsigned int PartialDeriv_mask, int flow_key, int flow_block_size, int flow_block_start );
    void ComputeOnePartialDerivative ( float *output, CpuStep *StepP,
                                       BeadParams &eval_params,
                                       BeadParams *ref_p,
                                         int ref_span,
                                       reg_params &eval_rp,
                                       NucStep &cache_step, int flow_key, 
                                       int flow_block_size, int flow_block_start );
    void    MultiFlowComputePartialDerivOfTimeShift ( float *fval,struct BeadParams *p, struct reg_params *reg_p, float *neg_sbg, int flow_block_size, int flow_block_start );
    void    Dfdgain_Step ( float *output,struct BeadParams *eval_p, int flow_block_size );
    void    Dfderr_Step ( float *output,struct BeadParams *eval_p, int flow_block_size );
    void    Dfyerr_Step ( float *y_minus_f_emphasized, int flow_block_size );
    float   TryNewRegionalParameters ( reg_params *new_rp, int flow_key, int flow_block_size,
                int flow_block_start );
    // utility functions for optimization
    void    UpdateBeadParametersFromRegion ( reg_params *new_rp, int flow_block_size );
    void FillScratchForEval ( BeadParams *p, BeadParams *ref_p, int ref_span, reg_params *reg_p, NucStep &cache_step, int flow_key, int flow_block_size, int flow_block_start );
    void DynamicEmphasis ( BeadParams &p, int flow_block_size );
    void ChooseSkipBeads ( bool _skip_beads );
    bool SkipBeads();
    const char* findName ( float *ptr );
    void EnterTheOptimization(BkgFitMatrixPacker* arg1, BkgFitMatrixPacker* arg2, float arg3, int arg4);
    void CleanTerminateOptimization();
    void SetupAnyIteration(reg_params &eval_rp, int iter, int flow_block_size );
    bool DoAllBeadIteration( bool arg3, BkgFitMatrixPacker* arg4, int iter, 
                int bead_iterations /*=1*/, bool isSample /*=false*/, 
                int flow_key, int flow_block_size, int flow_block_start );

 private:
    bool ExcludeBead(int ibd);
    void FinishBead (int ibd);

    static void DoStepDiff ( int add, float *, CpuStep *step, BeadParams *p, reg_params *reg_p,
                             int flow_block_size );

};
// add a unique penalty to values
void    UpdateOneBeadFromRegion ( BeadParams *p, bound_params *hi, bound_params *lo, reg_params *new_rp, int *dbl_tap_map, float krate_adj_limit, int flow_block_size );

void IdentifyParameters ( BeadTracker &my_beads, RegionTracker &my_regions, FlowBufferInfo &my_flow, int flow_block_size,  unsigned int well_mask, unsigned int reg_mask, bool skip_beads );
void    IdentifyDmult ( BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads );

void    IdentifyDmult (BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads , int flow_block_size);

void IdentifyNucMultiplyRatio ( BeadTracker &my_beads, RegionTracker &my_regions );

void IdentifyParametersFromSample (BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask, const LevMarBeadAssistant &lm_state , int flow_block_size);
void    IdentifyDmultFromSample (BeadTracker &my_beads, RegionTracker &my_regions, const LevMarBeadAssistant &lm_state , int flow_block_size);
void IdentifyNucMultiplyRatioFromSample ( BeadTracker &my_beads, RegionTracker &my_regions );
void IdentifyKmult(BeadTracker &my_beads, RegionTracker &my_regions, FlowBufferInfo &my_flow, int flow_block_size, bool skip_beads );

#endif // MULTILEVMAR_H
