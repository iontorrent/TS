/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LEVMARSTATE_H
#define LEVMARSTATE_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include "BeadParams.h"
#include "DiffEqModel.h"
#include "BeadTracker.h"
#include "BkgFitMatrixPacker.h"


// track beads for use in levmar fit
class LevMarBeadAssistant{
  public:
    int numLBeads; // size of indexes
    // utilities for optimization
    // regional subsets of beads
    int     *region_group;
    int     num_region_groups;
    int current_bead_region_group;
      // track active beads in fit
    bool    *well_completed;
    bool    *well_ignored;
    bool    *well_region_fit;
    int     *fit_indicies;
    int     ActiveBeads; // size of active set
    bool    advance_bd; // iteration check if we shuffle order to replace finished beads
    
    // data used in fitting algorithm by bead
    float   *lambda;
    float   *residual;
    float   avg_resid;
    float   lambda_max;
    float   lambda_escape;
    
    // I may regret this, but let's try lev-mar region state here in addition
        // data used in fitting algorithm by region
    float reg_lambda;
    float reg_error;
    float reg_lambda_min;
    float reg_lambda_max;
    
    // more lev-mar state
    float restrict_clonal;
    float non_integer_penalty[MAX_HPLEN];
    
    // current optimizations
    unsigned int well_mask;
    unsigned int reg_mask;
    
    LevMarBeadAssistant();
    ~LevMarBeadAssistant();
    void AssignBeadsToRegionGroups();
    void AllocateBeadFitState(int _numLBeads);
    void SetupActiveBeadList(float lambda_start);
    void FinishCurrentBead(int ibd,int nbd);
    void FinalComputeAndSetAverageResidual();
    void ComputeAndSetAverageResidualFromMeanPerFlow(float *flow_res_mean);
    void RestrictRegionFitToHighCopyBeads(BeadTracker &my_beads,float mean_copy_count);
    void SuppressCorruptedWellFits(BeadTracker &my_beads);
    void PhaseInClonalRestriction(int iter, int clonal_restriction);
    void InitializeLevMarFit(BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit);
    void SetNonIntegerPenalty(float *clonal_call_scale, float clonal_call_penalty, int len);
    void Delete();
    void ApplyClonalRestriction(float *fval, struct bead_params *p, int npts);
    void ReduceRegionStep();
    bool IncreaseRegionStep();
    void ReduceBeadLambda(int ibd);
    bool IncreaseBeadLambda(int ibd);
    void IncrementRegionGroup();
    bool ValidBeadGroup(int ibd);
};


#endif // LEVMARSTATE_H