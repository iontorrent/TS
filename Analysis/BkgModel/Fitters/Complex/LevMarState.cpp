/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "LevMarState.h"

using namespace std;

LevMarBeadAssistant::LevMarBeadAssistant()
{
  well_completed = NULL;
  lambda = NULL;
  lambda_max = 1E+10f;
  lambda_escape = 1E+8f; // smaller - abandon this well for now
  regularizer = NULL;
  residual = NULL;
  
  num_errors_logged = 0;
  region_success_step = 0; // no successes

  avg_resid = 0.0f;

  current_bead_region_group = 0;
  num_region_groups = 1;
  region_group = NULL;
  min_bead_to_fit_region = 10;
  derivative_direction = 1;
  
  bead_failure_rate_to_abort = 0.85f;
  min_amplitude_change=0.001f;
  numLBeads = 0;
  reg_lambda = 0.001f;
  reg_error = 0.0f;
  reg_lambda_min = FLT_MIN;
  reg_lambda_max = 1E+9f;
  reg_regularizer = 0.0f;
  
  nonclonal_call_penalty_enforcement = 0;
  restrict_clonal = 0.0f;
  skip_beads = false;
  well_mask = 0;
  reg_mask = 0;
  for (int i=0; i<MAX_POISSON_TABLE_COL; i++)
    non_integer_penalty[i] = 0.0f;
  
  res_state = UNINITIALIZED;
  avg_resid_state = UNINITIALIZED;

  ref_span = 0;
  ref_penalty_scale = 0.0f;
  kmult_penalty_scale = 0.0f;
}

void LevMarBeadAssistant::AllocateBeadFitState (int _numLBeads)
{
  numLBeads = _numLBeads;
  // allocate arrays used by fitting algorithm
  well_completed = new bool[numLBeads];
  lambda = new float[numLBeads];
  regularizer = new float [numLBeads];
  memset(regularizer,0,sizeof(float[numLBeads]));
  residual = new float[numLBeads];
  memset(residual,0,sizeof(float[numLBeads]));
}

void LevMarBeadAssistant::SetupActiveBeadList (float lambda_start)
{
  // initialize fitting algorithm
  for (int i=0;i < numLBeads;i++)
  {
    lambda[i] = lambda_start;
    regularizer[i] = 0.0f; // null out regularizer to start
    well_completed[i] = false;
  }
}

void LevMarBeadAssistant::FinishCurrentBead (int ibd)
{
   // remove it from the list of indicies to fit
   // don't waste time shuffing indices around and complicating the loops
   // when iterating through beads becomes our worst problem, we can deal with it better.
  well_completed[ibd] = true;
}

// set up beads
// should these two do something other than 'even sampling'
// i.e. pseudo-random ordering?
void LevMarBeadAssistant::AssignBeadsToRegionGroups( )
{
  // assign all beads to one of a few subsets of 500 beads or less.  These groups
  // are used to limit the number of beads used for region-wide parameter fitting

  num_region_groups = (numLBeads / NUMBEADSPERGROUP) + 1;
  region_group = new int[numLBeads];

  int gnum = 0;
  for (int i=0;i <numLBeads;i++)
  {
      if( gnum == 0 )
          beadSampleList.push_back( i );
      region_group[i] = gnum++;
      gnum = gnum % num_region_groups;
  }
}

void LevMarBeadAssistant::ReAssignBeadsToRegionGroups(BeadTracker &my_beads, int num_beads_per_group){
  int high_quality_count = my_beads.NumHighQuality();
  num_region_groups = (high_quality_count / num_beads_per_group) + 1;
  // region_group already allocated

  int gnum=0;
  int bnum=0; // assign poor quality beads as well
  for (int i=0; i<numLBeads; i++){
    if(my_beads.high_quality[i]){
      region_group[i] = gnum++;
      gnum = gnum % num_region_groups;
    } else {
      region_group[i] = bnum++;
      bnum = bnum % num_region_groups;
    }
  }
  // good beads and bad beads are now allocated to groups containing enough good beads
  // assign current region group number to be valid
  current_bead_region_group = current_bead_region_group % num_region_groups;
}




bool LevMarBeadAssistant::WellBehavedBead(int ibd)
{
  if (avg_resid_state!=res_state)
    printf("Error: incomparable values\n");
  return(residual[ibd] < avg_resid*4.0);
}


void LevMarBeadAssistant::FinalComputeAndSetAverageResidual(BeadTracker& my_beads)
{
    avg_resid = 0.0;
    float beads_counted = 0.0001;
    for (int ibd=0; ibd < numLBeads; ibd++){
      // if this iteration is a region-wide parameter fit, then only process beads
      // in the sampled sub-group
      if ( my_beads.isSampled ) {
	// regional sampling enabled
	if (my_beads.Sampled(ibd) ) {// && my_beads.BeadIncluded(ibd,skip_beads) ) {
	  avg_resid += residual[ibd];
	  beads_counted += 1.0f;
	}
      }
      else {
	// rolling regional groups enabled
	if (ValidBeadGroup (ibd)) {
	  if (my_beads.BeadIncluded(ibd,skip_beads)) {
	    avg_resid += residual[ibd];
	    beads_counted += 1.0f;
	  }
	}
      }
    }
    avg_resid /= beads_counted;

    avg_resid_state = res_state;
}


void LevMarBeadAssistant::PhaseInClonalRestriction (int iter, int clonal_restriction)
{
  float hpmax = 1.0f;
  if (clonal_restriction > 0)
  {
    hpmax = (int) (iter / 4) +2;

    if (hpmax > clonal_restriction)
      hpmax = clonal_restriction;

    restrict_clonal = hpmax-0.5f;
  }
}



void LevMarBeadAssistant::InitializeLevMarFit (BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit)
{
  reg_mask = 0;
  well_mask = 0;

  region_success_step = 0; // no successes yet

// initialize regional lev mar fitt
  current_bead_region_group =0;
  reg_lambda = 0.0001f;
  reg_regularizer= 0.0f;
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
  delete [] residual; residual = NULL;
  delete [] well_completed; well_completed = NULL;
  delete [] lambda; lambda = NULL;
  delete [] regularizer; regularizer = NULL;
  delete [] region_group; region_group = NULL;
}

LevMarBeadAssistant::~LevMarBeadAssistant()
{
  Delete();
}

void LevMarBeadAssistant::SetNonIntegerPenalty (float *clonal_call_scale, float clonal_call_penalty, int len)
{
  int tlen = MAX_POISSON_TABLE_COL;
  if (len<tlen)
    tlen = len;
  for (int i=0; i<tlen; i++)
  {
    non_integer_penalty[i] = clonal_call_scale[i]*clonal_call_penalty;
  }
}

// this is of course part of the same horror that is our optimizer
void LevMarBeadAssistant::PenaltyForDeviationFromRef(float *fval, BeadParams *p, BeadParams *ref_ampl, int ref_span,  int npts, int flow_block_size)
{
  if (ref_span>0){
    // for every flow affected

    for (int fnum = 0; (fnum<flow_block_size) &(fnum<ref_span) ; fnum++){
      float *vb_out;
      vb_out = fval+fnum*npts;
      // "cheap logarithm" - make more efficient by encode ref-ampl with pre-division?
      float penalty_score = 1.0f-(p->Ampl[fnum]+0.5f)/(ref_ampl->Ampl[fnum]+0.5f);
      penalty_score = penalty_score*penalty_score; // squared error term
      float penalty_error_term = penalty_score * ref_penalty_scale;
      // modify function evaluation because our LevMar optimizer implementation is sub-optimal
      for (int i=0; i<MAXCLONALMODIFYPOINTSERROR; i++)
      {
        vb_out[i] += penalty_error_term* ( (float) (i&1) - 0.5);  // alternating error points
      }
    }
  }
}

// the horror continues
void LevMarBeadAssistant::PenaltyForDeviationFromKmult(float *fval, BeadParams *p,  int npts, int flow_block_size)
{
  if (ref_span>0){
    // for every flow affected

    for (int fnum = 0; (fnum<flow_block_size) ; fnum++){
      float *vb_out;
      vb_out = fval+fnum*npts;
      // "cheap logarithm"
      // "kmult for 0 ampl shouldn't move much because data doesn't demand it"
      float penalty_score = 1.0f-p->kmult[fnum];
      penalty_score = penalty_score*penalty_score; // squared error term
      float penalty_error_term = penalty_score * kmult_penalty_scale;
      // modify function evaluation because our LevMar optimizer implementation is sub-optimal
      for (int i=0; i<MAXCLONALMODIFYPOINTSERROR; i++)
      {
        vb_out[i] += penalty_error_term* ( (float) (i&1) - 0.5);  // alternating error points
      }
    }
  }
}


// global_defaults.clonal_call_scale, global_defaults.clonal_call_penalty, lm_state.clonal_restrict.level
void LevMarBeadAssistant::ApplyClonalRestriction (float *fval, BeadParams *p, int npts, int flow_key, int flow_block_size)
{
  float clonal_error_term = 0.0;
  // ASSUMES fnum = flow number for first 20 flows!!!
  // ASSUMES 5-mers are most clonal penalty applies to
  // p->clonal_read so we short-circuit the loop
  if (p->my_state->clonal_read)
  {
    for (int fnum=flow_key+1;fnum<flow_block_size;fnum++)
    {
      float *vb_out;
      vb_out = fval + fnum*npts;        // get ptr to start of the function evaluation for the current flow  // yet another place where bad indexing is annoying
      // this is an ad-hoc modifier to make Lev-Mar move the right direction by modifying the error
      // at the first MAXCLONALMODIFYPOINTSERROR points

      int intcall = ( (int) (p->Ampl[fnum]+0.5));
      if (intcall < 0)
        intcall = 0;
      if (intcall > MAGIC_MAX_CLONAL_HP_LEVEL)
        intcall = MAGIC_MAX_CLONAL_HP_LEVEL;

      //@TODO - why shouldn't the key flows be used here - aren't they integers?
      if ( (p->Ampl[fnum] < restrict_clonal))
        clonal_error_term = fabs (p->Ampl[fnum] - intcall) * non_integer_penalty[intcall];
      else
        clonal_error_term = 0.0;
      for (int i=0; i<MAXCLONALMODIFYPOINTSERROR; i++)
      {
        vb_out[i] += clonal_error_term* ( (float) (i&1) - 0.5);  // alternating error points
      }
    }
  }
}

#define LAMBDA_STEP  30.0

void LevMarBeadAssistant::ReduceRegionStep()
{
  if (reg_lambda>(LAMBDA_STEP*FLT_MIN))
    reg_lambda /= LAMBDA_STEP;

  if (reg_lambda < reg_lambda_min)
  {
    reg_lambda = reg_lambda_min;
  }
  reg_regularizer = 0.0f; // when we improve, assume all is well
}


bool LevMarBeadAssistant::IncreaseRegionStep()
{
  bool cont_proc=false;
  reg_lambda *= LAMBDA_STEP;

  if (reg_lambda > reg_lambda_max)
    cont_proc = true;
  return (cont_proc);
}

void LevMarBeadAssistant::IncreaseRegionRegularizer()
{
  reg_regularizer += LM_REG_REGULARIZER;
}

void LevMarBeadAssistant::ReduceBeadLambda (int ibd)
{
  if (lambda[ibd]>(LAMBDA_STEP*FLT_MIN))
    lambda[ibd] /= LAMBDA_STEP;
  regularizer[ibd] = 0.0f; // reset to nothing, as we're being successful
}

bool LevMarBeadAssistant::IncreaseBeadLambda (int ibd)
{
  lambda[ibd] *=LAMBDA_STEP;
  return (false); // do comparisons elsewhere due to tortured logic - @TODO fix this
}

void LevMarBeadAssistant::IncreaseRegularizer(int ibd)
{
  regularizer[ibd] += LM_BEAD_REGULARIZER;
}

void LevMarBeadAssistant::IncrementRegionGroup()
{
  current_bead_region_group++;
  current_bead_region_group = current_bead_region_group % num_region_groups;
}

bool LevMarBeadAssistant::ValidBeadGroup (int ibd) const
{
  return (region_group[ibd] == current_bead_region_group) ;
}
