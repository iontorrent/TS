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
    // setup stuff for lev-mar control
    FitControl_t fit_control;
    bool use_vectorization;

    // start caches for processing
    float tshift_cache;

    float *cache_sbg;
    float *cache_slope; // derivative calculation
    

    void InitTshiftCache();
    void FillTshiftCache ( float my_tshift);
    void InitRandomCache();
    void DeleteRandomCache();

    // cache for regional parameters for derivatives
    // we recompute these for each bead needlessly
    std::vector< reg_params> step_rp;
    std::vector< NucStep> step_nuc_cache;

    void FillDerivativeStepCache ( bead_params &eval_params, reg_params &eval_rp, unsigned int PartialDeriv_mask );
    void ComputeCachedPartialDerivatives ( bead_params &eval_params,  unsigned int PartialDeriv_mask );

    MultiFlowLevMar ( SignalProcessingMasterFitter & );
    ~MultiFlowLevMar();

    int MultiFlowSpecializedLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction  );
    int MultiFlowSpecializedSampledLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit, float lambda_start, int clonal_restriction );
    void MultiFlowSpecializedLevMarFitAllWells (int bead_only_iterations, BkgFitMatrixPacker *well_fit, float lambda_start, int clonal_restriction=0 );
    int  MultiFlowSpecializedLevMarFitParametersOnlyRegion ( int number_region_iterations_wanted,  BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction );
    float   CalculateCurrentResidualForTestBeads (  ); // within loop

    int LevMarAccumulateRegionDerivsForSampledActiveBeadList (

        reg_params &eval_rp,
        BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
        int iter );

    int LevMarAccumulateRegionDerivsForActiveBeadList (

        reg_params &eval_rp,
        BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
        int iter );

    void  LevMarFitRegion (
                            BkgFitMatrixPacker *reg_fit );

    float    LevMarFitToActiveBeadList (
        bool well_only_fit,
        reg_params& eval_rp,
        BkgFitMatrixPacker* well_fit,  unsigned int PartialDeriv_mask,
        int bead_iterations,
	bool isSample=false);

    void DoSampledRegionIteration (  
                                    BkgFitMatrixPacker *reg_fit,
                                    int iter );
    void DoRegionIteration (  
                             BkgFitMatrixPacker *reg_fit,
                             int iter );
    bool DoSampledBeadIteration (  
                                  bool well_only_fit,
                                  BkgFitMatrixPacker *well_fit,
                                  int iter );
    float LevMarFitToRegionalActiveBeadList (
            bool well_only_fit, reg_params &eval_rp, BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
            int iter );

    void LevMarBuildMatrixForBead ( int ibd,
                                   
                                    bool well_only_fit,
                                    reg_params &eval_rp, NucStep &cache_step,
                                    BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
                                    int iter );

    int   LevMarFitOneBead ( int ibd,
                             reg_params &eval_rp,
                             BkgFitMatrixPacker *well_fit,bool well_only_fit );
    void AccumulateRegionDerivForOneBead (
        int ibd, int &reg_wells,

        BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
        int iter );
    // do steps for computing partial derivatives
    // it is entirely likely that these functions should be functions for the "scratchspace" object
    void    ComputePartialDerivatives ( bead_params &eval_params, reg_params &eval_rp,  NucStep &cache_step, unsigned int PartialDeriv_mask);
    void ComputeOnePartialDerivative ( float *output, CpuStep_t *StepP,
                                       bead_params &eval_params, reg_params &eval_rp, NucStep &cache_step);
    void    MultiFlowComputePartialDerivOfTimeShift ( float *fval,struct bead_params *p, struct reg_params *reg_p, float *neg_sbg );
    void    Dfdgain_Step ( float *output,struct bead_params *eval_p );
    void    Dfderr_Step ( float *output,struct bead_params *eval_p );
    void    Dfyerr_Step ( float *y_minus_f_emphasized );
    float   TryNewRegionalParameters ( reg_params *new_rp );
    // utility functions for optimization
    void    UpdateBeadParametersFromRegion ( reg_params *new_rp );
    void FillScratchForEval ( struct bead_params *p, struct reg_params *reg_p, NucStep &cache_step );
    void DynamicEmphasis ( bead_params &p );
    void ChooseSkipBeads ( bool _skip_beads );
    bool SkipBeads();
    char   *findName ( float *ptr );
    void EnterTheOptimization(BkgFitMatrixPacker* arg1, BkgFitMatrixPacker* arg2, float arg3, int arg4);
    void CleanTerminateOptimization();
    void SetupAnyIteration(reg_params &eval_rp, int iter);
    bool DoAllBeadIteration( bool arg3, BkgFitMatrixPacker* arg4, int iter, int bead_iterations=1, bool isSample=false );

 private:
    bool ExcludeBead(int ibd);
    void FinishBead (int ibd);


};
// add a unique penalty to values
void    UpdateOneBeadFromRegion ( bead_params *p, bound_params *hi, bound_params *lo, reg_params *new_rp, int *dbl_tap_map );
void IdentifyParameters ( BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask, bool skip_beads );
void    IdentifyDmult ( BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads );
void IdentifyNucMultiplyRatio ( BeadTracker &my_beads, RegionTracker &my_regions );
void DoStepDiff ( int add, float *, CpuStep_t *step, struct bead_params *p, struct reg_params *reg_p );

void IdentifyParametersFromSample ( BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask, bool skip_beads, const LevMarBeadAssistant &lm_state );
void    IdentifyDmultFromSample ( BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads, const LevMarBeadAssistant &lm_state );
void IdentifyNucMultiplyRatioFromSample ( BeadTracker &my_beads, RegionTracker &my_regions );

#endif // MULTILEVMAR_H
