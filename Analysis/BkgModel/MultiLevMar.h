/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MULTILEVMAR_H
#define MULTILEVMAR_H

#include "BkgModel.h"


// this will be a friend of bkgmodel for now
// and look just like the cuda code :-)
// ideally this should be passed a bead tracker & a region tracker object
// and fit the current bead list & region parameters for that bead list
class MultiFlowLevMar
{
  public:
    BkgModel &bkg; // reference to source class for now
    LevMarBeadAssistant lm_state;


    MultiFlowLevMar (BkgModel &);
    int MultiFlowSpecializedLevMarFitParameters (int max_iter, int max_reg_iter, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction = 0);
    float   CalculateCurrentResidualForTestBeads (float *sbg); // within loop
    int LevMarAccumulateRegionDerivsForActiveBeadList (
      float *ival, float *sbg,
      reg_params &eval_rp,
      BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
      int iter, float *fvdbg, float *BackgroundDebugTapA, float *BackgroundDebugTapB);
    void  LevMarFitRegion (float &tshift_cache,  float *sbg,
                           BkgFitMatrixPacker *reg_fit);
    float    LevMarFitToActiveBeadList (
      float* ival, float* sbg,
      bool well_only_fit,
      reg_params& eval_rp,
      BkgFitMatrixPacker* well_fit,  unsigned int PartialDeriv_mask,
      int iter, float *fvdbg, float* BackgroundDebugTapA, float* BackgroundDebugTapB);
    void LevMarBuildMatrixForBead (int ibd,
                                   float *ival, float *sbg,
                                   bool well_only_fit,
                                   reg_params &eval_rp,
                                   BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
                                   int iter, float *fvdbg, float *BackgroundDebugTapA, float *BackgroundDebugTapB);

    int   LevMarFitOneBead (int ibd, int &nbd,
                            float *sbg,
                            reg_params &eval_rp,
                            BkgFitMatrixPacker *well_fit,bool well_only_fit);
    // do steps for computing partial derivatives
    // it is entirely likely that these functions should be functions for the "scratchspace" object
    void    ComputePartialDerivatives (bead_params &eval_params, reg_params &eval_rp,  unsigned int PartialDeriv_mask, float *ival, float *sbg, bool debug_flag);
    void ComputeOnePartialDerivative (float *output, CpuStep_t *StepP,
                                      bead_params &eval_params, reg_params &eval_rp, float *ival, float *sbg);
    void    MultiFlowComputePartialDerivOfTimeShift (float *fval,struct bead_params *p, struct reg_params *reg_p, float *sbg);
    void    Dfdgain_Step (float *output,struct bead_params *eval_p);
    void    Dfderr_Step (float *output,struct bead_params *eval_p);
    void    Dfyerr_Step (float *y_minus_f_emphasized);
    float   TryNewRegionalParameters (reg_params *new_rp,  float *sbg);
    // utility functions for optimization
    void    UpdateBeadParametersFromRegion (reg_params *new_rp);
    void FillScratchForEval (struct bead_params *p, struct reg_params *reg_p, float *sbg);
    void DynamicEmphasis(bead_params &p);
};
// add a unique penalty to values
void    UpdateOneBeadFromRegion (bead_params *p, bound_params *hi, bound_params *lo, reg_params *new_rp, int *dbl_tap_map);
void IdentifyParameters (BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask);
void    IdentifyDmult (BeadTracker &my_beads, RegionTracker &my_regions);
void IdentifyNucMultiplyRatio (BeadTracker &my_beads, RegionTracker &my_regions);
void DoStepDiff (int add, float *, CpuStep_t *step, struct bead_params *p, struct reg_params *reg_p);

#endif // MULTILEVMAR_H
