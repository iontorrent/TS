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

  avg_resid = 0.0f;

  current_bead_region_group = 0;
  num_region_groups = 1;
  region_group = NULL;
  min_bead_to_fit_region = 10;

  
  bead_failure_rate_to_abort = 0.5f;
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
  for (int i=0; i<MAX_HPLEN; i++)
    non_integer_penalty[i] = 0.0f;
  
  shrink_factor = 0.0f;
  res_state = UNINITIALIZED;
  avg_resid_state = UNINITIALIZED;
}

void LevMarBeadAssistant::AllocateBeadFitState (int _numLBeads)
{
  numLBeads = _numLBeads;
  // allocate arrays used by fitting algorithm
  well_completed = new bool[numLBeads];
  lambda = new float[numLBeads];
  regularizer = new float [numLBeads];
  residual = new float[numLBeads];
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
      // in the selection sub-group
      if ( my_beads.isSampled ) {
	// regional sampling enabled
	if (!my_beads.Sampled (ibd) )
	  continue;
      }
      else {
	// rolling regional groups enabled
	if (!ValidBeadGroup (ibd))
	  continue;
      }

      if (my_beads.BeadIncluded(ibd,skip_beads))
      {
	avg_resid += residual[ibd];
	beads_counted += 1.0f;
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
  delete [] residual;
  delete [] well_completed;
  delete [] lambda;
  delete [] regularizer;
  delete [] region_group;
}

LevMarBeadAssistant::~LevMarBeadAssistant()
{
  Delete();
}

void LevMarBeadAssistant::SetNonIntegerPenalty (float *clonal_call_scale, float clonal_call_penalty, int len)
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
void LevMarBeadAssistant::ApplyClonalRestriction (float *fval, bead_params *p, int npts)
{
  float clonal_error_term = 0.0;
  // ASSUMES fnum = flow number for first 20 flows!!!
  // ASSUMES 5-mers are most clonal penalty applies to
  // p->clonal_read so we short-circuit the loop
  if (p->my_state->clonal_read)
  {
    for (int fnum=KEY_LEN+1;fnum<NUMFB;fnum++)
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

bool LevMarBeadAssistant::InValidBeadItem(int ibd, const vector<bool>& quality){
 return(!ValidBeadGroup ( ibd ) || ( quality[ibd]==false ) || well_completed[ibd]);
}


