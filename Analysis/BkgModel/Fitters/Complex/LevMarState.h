/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LEVMARSTATE_H
#define LEVMARSTATE_H

#include <vector>
#include "BkgMagicDefines.h"
#include "BeadParams.h"
#include "DiffEqModel.h"
#include "BeadTracker.h"
#include "BkgFitMatrixPacker.h"

// arithmetic regularizers to handle the unfortunate zero-derivative cases
#define LM_BEAD_REGULARIZER 0.00001f
#define LM_REG_REGULARIZER 0.00001f
#define MAX_LM_MESSAGE_LOG 100

// track beads for use in levmar fit
class LevMarBeadAssistant{
  public:
    int      numLBeads; // size of indexes
    // utilities for optimization
    // regional subsets of beads
    int     *region_group;
    int     num_region_groups;
    int     current_bead_region_group;

    //list of beads to be used for sampling
    std::vector<int> beadSampleList;
      // track active beads in fit
    bool    *well_completed;

    // data used in fitting algorithm by bead
    float   *lambda;
    float *regularizer;
    float   lambda_max;
    float   lambda_escape;
    
    // control parameter
    int min_bead_to_fit_region;
    float bead_failure_rate_to_abort;
    float min_amplitude_change;
    
    
    int num_errors_logged; // track number of error messages spammed to output log, stop when we hit max
    int region_success_step; // track number of regional optimization successes: most important when this is zero

    // residuals
    float *residual;
    float avg_resid;
    int   res_state;
    int   avg_resid_state;
    
    // I may regret this, but let's try lev-mar region state here in addition
    // data used in fitting algorithm by region
    float reg_lambda;
    float reg_regularizer;
    float reg_error;
    float reg_lambda_min;
    float reg_lambda_max;
    
    // more lev-mar state
    int nonclonal_call_penalty_enforcement;
    float restrict_clonal;
    float non_integer_penalty[MAX_POISSON_TABLE_COL];

    float ref_penalty_scale;
    float kmult_penalty_scale;

    bool  skip_beads; // skip individual wells when doing regional optimization, pick up well parameters later
    int derivative_direction; // take derivative in additive or negative direction - matters if we hit a parameter boundary
    
    // current optimizations
    unsigned int well_mask;
    unsigned int reg_mask;

    // cheap scratch space
    BeadParams ref_bead;
    int ref_span;
    
    LevMarBeadAssistant();
    ~LevMarBeadAssistant();
    void AssignBeadsToRegionGroups(void);
    void AllocateBeadFitState(int _numLBeads);
    void SetupActiveBeadList(float lambda_start);
    void FinishCurrentBead(int ibd);
    void FinalComputeAndSetAverageResidual(BeadTracker& my_beads);
    void ComputeAndSetAverageResidualFromMeanPerFlow(float *flow_res_mean);

    void PhaseInClonalRestriction(int iter, int clonal_restriction);
    void InitializeLevMarFit(BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit);
    void SetNonIntegerPenalty(float *clonal_call_scale, float clonal_call_penalty, int len);
    void Delete();
    void ApplyClonalRestriction(float *fval, struct BeadParams *p, int npts, int flow_key, int flow_block_size);

    void PenaltyForDeviationFromRef(float *fval, struct BeadParams *p, struct BeadParams *ref_ampl, int ref_span, int npts, int flow_block_size);
    void PenaltyForDeviationFromKmult(float *fval, BeadParams *p,  int npts, int flow_block_size);

    void ReduceRegionStep();
    bool IncreaseRegionStep();
    void IncreaseRegionRegularizer();
    void ReduceBeadLambda(int ibd);
    bool IncreaseBeadLambda(int ibd);
    void IncreaseRegularizer(int ibd);
    void IncrementRegionGroup();
    bool ValidBeadGroup(int ibd) const;
    bool WellBehavedBead(int ibd);
    int CountHappyBeads();
    void ReAssignBeadsToRegionGroups(BeadTracker &my_beads, int num_beads_per_group);
    bool LogMessage(){if (num_errors_logged<MAX_LM_MESSAGE_LOG){
            num_errors_logged++;
            return(true);} else {return(false);}};
};


#endif // LEVMARSTATE_H
