/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "LevMarState.h"


LevMarBeadAssistant::LevMarBeadAssistant()
{
    well_completed = NULL;
    lambda = NULL;
    lambda_max = 1E+10;
    lambda_escape = 1E+8; // smaller - abandon this well for now
    fit_indicies = NULL;
    residual = NULL;
    well_ignored = NULL;
    well_region_fit = NULL;
    avg_resid = 0.0;
    advance_bd=true;
    ActiveBeads = 0;
    current_bead_region_group = 0;
    num_region_groups = 1;
    region_group = NULL;
    numLBeads = 0;
    reg_lambda = 0.001;
    reg_error = 0;
    reg_lambda_min = FLT_MIN;
    reg_lambda_max = 1E+9;
    restrict_clonal = 0.0;
    well_mask = 0;
    reg_mask = 0;
    for (int i=0; i<MAX_HPLEN; i++)
        non_integer_penalty[i] = 0;
}

void LevMarBeadAssistant::AllocateBeadFitState(int _numLBeads)
{
    numLBeads = _numLBeads;
    // allocate arrays used by fitting algorithm
    well_completed = new bool[numLBeads];
    lambda = new float[numLBeads];
    fit_indicies = new int[numLBeads];
    residual = new float[numLBeads];

    well_ignored = new bool[numLBeads];
    well_region_fit = new bool[numLBeads];

    for (int i=0;i < numLBeads;i++)
    {
        well_region_fit[i] = true;
        well_ignored[i] = false;
    }
}


void LevMarBeadAssistant::SetupActiveBeadList(float lambda_start)
{
    // initialize fitting algorithm
    ActiveBeads=0;
    for (int i=0;i < numLBeads;i++)
    {
        lambda[i] = lambda_start;
        well_completed[i] = false;

        if (~well_ignored[i])
            fit_indicies[ActiveBeads++] = i;
    }
}

void LevMarBeadAssistant::FinishCurrentBead(int ibd, int nbd)
{
    well_completed[ibd] = true;
    // remove it from the list of indicies to fit
    if (ActiveBeads > 1)
    {
        fit_indicies[nbd] = fit_indicies[ActiveBeads-1];
        advance_bd = false;
    }
    // decrement count of wells we are fitting
    ActiveBeads--;
}

// set up beads
void LevMarBeadAssistant::AssignBeadsToRegionGroups()
{
    // assign all beads to one of a few subsets of 500 beads or less.  These groups
    // are used to limit the number of beads used for region-wide parameter fitting
    num_region_groups = (numLBeads / NUMBEADSPERGROUP) + 1;
    region_group = new int[numLBeads];

    int gnum = 0;
    for (int i=0;i <numLBeads;i++)
    {
        region_group[i] = gnum++;
        gnum = gnum % num_region_groups;
    }
}
void LevMarBeadAssistant::FinalComputeAndSetAverageResidual()
{
    avg_resid = 0.0;
    for (int i=0;i <numLBeads;i++)
        avg_resid += residual[i];

    avg_resid /= numLBeads;
}

void LevMarBeadAssistant::ComputeAndSetAverageResidualFromMeanPerFlow(float *flow_res_mean)
{
    avg_resid=0;
    for (int fnum=0; fnum<NUMFB; fnum++)
        avg_resid += flow_res_mean[fnum];

    avg_resid /= NUMFB;
}


void LevMarBeadAssistant::RestrictRegionFitToHighCopyBeads(BeadTracker &my_beads, float mean_copy_count)
{
    // only use the top amplitude signal beads for region-wide parameter fitting
    for (int ibd=0;ibd < my_beads.numLBeads;ibd++)
    {
        if (my_beads.params_nn[ibd].Copies > mean_copy_count)
            well_region_fit[ibd] = true;
        else
            well_region_fit[ibd] = false;
    }
}

void LevMarBeadAssistant::SuppressCorruptedWellFits(BeadTracker &my_beads)
{
    for (int ibd=0; ibd<my_beads.numLBeads; ibd++)
    {
        if (my_beads.params_nn[ibd].corrupt)
            well_region_fit[ibd] = false;
    }
}

void LevMarBeadAssistant::PhaseInClonalRestriction(int iter, int clonal_restriction)
{
    float hpmax = 1;
    if (clonal_restriction > 0)
    {
        hpmax = (int)(iter / 4) +2;

        if (hpmax > clonal_restriction)
            hpmax = clonal_restriction;

        restrict_clonal = hpmax-0.5;
    }
}



void LevMarBeadAssistant::InitializeLevMarFit(BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit)
{
    reg_mask = 0;
    well_mask = 0;

// initialize regional lev mar fitt
    current_bead_region_group =0;
    reg_lambda = 0.0001;
    restrict_clonal = 0.0; // we only restrict clonal within this routine

    if (well_fit != NULL)
        well_mask = well_fit->GetPartialDerivMask();
    if (reg_fit != NULL)
        reg_mask = reg_fit->GetPartialDerivMask();

    well_mask |= YERR | FVAL; // always fill in yerr and fval
    reg_mask |= YERR | FVAL; // always fill in yerr and fval
}

void LevMarBeadAssistant::Delete()
{
    delete [] well_region_fit;
    delete [] well_ignored;
    delete [] residual;
    delete [] fit_indicies;
    delete [] well_completed;
    delete [] lambda;
    delete [] region_group;
}

LevMarBeadAssistant::~LevMarBeadAssistant()
{
    Delete();
}

void LevMarBeadAssistant::SetNonIntegerPenalty(float *clonal_call_scale, float clonal_call_penalty, int len)
{
    int tlen = MAX_HPLEN-1;
    if (len<tlen)
        tlen = len;
    for (int i=0; i<=len; i++)
    {
        non_integer_penalty[i] = clonal_call_scale[i]*clonal_call_penalty;
    }
}

// global_defaults.clonal_call_scale, global_defaults.clonal_call_penalty, lm_state.clonal_restrict.level
void LevMarBeadAssistant::ApplyClonalRestriction(float *fval, bead_params *p, int npts)
{
    float clonal_error_term = 0.0;
    // ASSUMES fnum = flow number for first 20 flows!!!
    // ASSUMES 5-mers are most clonal penalty applies to
    // p->clonal_read so we short-circuit the loop
    if(p->clonal_read){
	    for (int fnum=KEY_LEN+1;fnum<NUMFB;fnum++)
	    {
	        float *vb_out;
	        vb_out = fval + fnum*npts;        // get ptr to start of the function evaluation for the current flow  // yet another place where bad indexing is annoying
	        // this is an ad-hoc modifier to make Lev-Mar move the right direction by modifying the error
	        // at the first MAXCLONALMODIFYPOINTSERROR points
	
	        int intcall = ((int)(p->Ampl[fnum]+0.5));
	        if (intcall < 0)
	            intcall = 0;
	        if (intcall > MAGIC_MAX_CLONAL_HP_LEVEL)
	            intcall = MAGIC_MAX_CLONAL_HP_LEVEL;
	
	        //@TODO - why shouldn't the key flows be used here - aren't they integers?
	        if ((p->Ampl[fnum] < restrict_clonal))
	            clonal_error_term = fabs(p->Ampl[fnum] - intcall) * non_integer_penalty[intcall];
	        else
	            clonal_error_term = 0.0;
	        for (int i=0; i<MAXCLONALMODIFYPOINTSERROR; i++)
	        {
	            vb_out[i] += clonal_error_term* ((float)(i&1) - 0.5);  // alternating error points
	        }
	    }
    }
}


void LevMarBeadAssistant::ReduceRegionStep()
{
    reg_lambda /= 10.0;
    if (reg_lambda < FLT_MIN)
        reg_lambda = FLT_MIN;

    if (reg_lambda < reg_lambda_min)
    {
        reg_lambda = reg_lambda_min;
    }
}

bool LevMarBeadAssistant::IncreaseRegionStep()
{
    bool cont_proc=false;
    reg_lambda *= 10.0;

    if (reg_lambda > reg_lambda_max)
        cont_proc = true;
    return(cont_proc);
}

void LevMarBeadAssistant::ReduceBeadLambda(int ibd)
{
    lambda[ibd] /= 10.0;
    if (lambda[ibd] < FLT_MIN)
        lambda[ibd] = FLT_MIN;
}

bool LevMarBeadAssistant::IncreaseBeadLambda(int ibd)
{
    lambda[ibd] *=10.0;
    return(false);  // do comparisons elsewhere due to tortured logic - @TODO fix this
}

void LevMarBeadAssistant::IncrementRegionGroup()
{
                current_bead_region_group++;
                current_bead_region_group = current_bead_region_group % num_region_groups;
}

bool LevMarBeadAssistant::ValidBeadGroup(int ibd)
{
  return(region_group[ibd] == current_bead_region_group) ;
}
