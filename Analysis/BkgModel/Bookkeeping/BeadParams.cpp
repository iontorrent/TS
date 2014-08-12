/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BeadParams.h"

void BeadParams::ApplyUpperBound (bound_params *bound, int flow_block_size)
{
  for (int i=0;i<flow_block_size;i++)
  {
    MAX_BOUND_PAIR_CHECK (Ampl[i],Ampl);
    MAX_BOUND_PAIR_CHECK (kmult[i],kmult);
  }

  MAX_BOUND_CHECK (Copies);
  MAX_BOUND_CHECK (gain);

  MAX_BOUND_CHECK (R);
  MAX_BOUND_CHECK (dmult);
};

void BeadParams::ApplyLowerBound (bound_params *bound, int flow_block_size)
{
  for (int i=0;i<flow_block_size;i++)
  {
    MIN_BOUND_PAIR_CHECK (Ampl[i],Ampl);
    MIN_BOUND_PAIR_CHECK (kmult[i],kmult);
  }

  MIN_BOUND_CHECK (Copies);
  MIN_BOUND_CHECK (gain);

  MIN_BOUND_CHECK (R);
  MIN_BOUND_CHECK (dmult);
};


void BeadParams::ApplyAmplitudeZeros (const int *zero, int flow_block_size)
{
  for (int i=0; i<flow_block_size; i++)
    Ampl[i] *= zero[i];
}

void bound_params::SetBeadStandardHigh ()
{

    Ampl = LAST_POISSON_TABLE_COL; // max achievable value in the software
    // Ampl = MAXAMPL; // some lower value we want to restrain
    kmult = 1.0f;

  Copies = 30.0f;
  gain   = 1.1f;
  R      = 1.0f;
  dmult  = 1.8f;
}

void bound_params::SetBeadStandardLow (float AmplLowerLimit)
{

  Ampl  = AmplLowerLimit;
  kmult = 0.25f;

  Copies = 0.05f;
  gain   = 0.9f;
  R      = 0.001f;
  dmult  = 0.2f;
}

void state_Init (bead_state *my_state)
{
  if (my_state!=NULL)
  {
  my_state->avg_err = FLT_MAX; // don't filter anything, ever
  my_state->key_norm    = 0.0f;
  my_state->ppf         = 0.0f;
  my_state->ssq         = 0.0f;
  my_state->bad_read    = false;
  my_state->clonal_read = true;
  my_state->corrupt     = false;
  my_state->random_samp = false;
  }
 /* for (int j=0; j<MAX_NUM_FLOWS_IN_BLOCK_GPU; j++)
    my_state->hits_by_flow[j] = 0;*/
}

void BeadParams::SetStandardFlow()
{
  for (int j=0;j<MAX_NUM_FLOWS_IN_BLOCK_GPU;j++)
  {
    Ampl[j] = 0.5f;
    kmult[j] = 1.0f;
  }
}

void BeadParams::SetBeadStandardValue ()
{
  SetStandardFlow();


  Copies  = 2.0f;
  gain    = 1.0f;

  R           = 0.7f;
  dmult       = 1.0f;
  trace_ndx = -1;
  x = -1;
  y = -1;

  for (int i=0;i < NUM_DM_PCA;i++)
     pca_vals[i] = 0.0f;

  tau_adj = 1.0f;
  phi         = 0.6f; // -vm: average incorparation rate per flow

  state_Init ((my_state));
}

void BeadParams::SetBeadZeroValue()
{
  Copies = 0.0f;
  gain = 0.0f;
  R = 0.0f;
  dmult = 0.0f;
  trace_ndx = -1;
  x = -1;
  y = -1;
  my_state = NULL;

  for (int i=0;i < NUM_DM_PCA;i++)
     pca_vals[i] = 0.0f;

  tau_adj = 1.0f;
  phi         = 0.6f; // -vm: average incorparation rate per flow
}

void BeadParams::AccumulateBeadValue(const BeadParams *source)
{
  Copies += source->Copies;
  R += source->R;
  gain += source->gain;
  dmult += source->dmult;
}

void BeadParams::ScaleBeadValue(float multiplier)
{
  Copies *=multiplier;
  R *= multiplier;
  gain *= multiplier;
  dmult *=multiplier;
}

void BeadParams::SetAmplitude (const float *Ampl, int flow_block_size)
{
  for (int i=0; i<flow_block_size; i++)
    this->Ampl[i] = Ampl[i];
}

void BeadParams::LockKey (float *key, int keylen)
{
  for (int i=0; i<keylen; i++)
    Ampl[i] = key[i];
}

void BeadParams::UnLockKey (float limit_val, int keylen)
{
  for (int i=0; i<keylen; i++)
    Ampl[i] =limit_val;
}


float BeadParams::LargestAmplitudeCopiesChange ( const BeadParams *that, int flow_block_size ) const
{
  float achg = 0.0f;
  for (int i = 0 ; i < flow_block_size ; i++)
  {
    float chg = fabs ( this->Ampl[i]*this->Copies - that->Ampl[i]*that->Copies );
    if (chg > achg) achg = chg;
  }
  return (achg);
}

// detect unusual levels of variation in this bead
void BeadParams::DetectCorruption (const error_track &my_err, float threshold, int decision,
                                   int flow_block_size )
{
  int high_err_cnt = 0;

  // count from end the number of excessively high error flows
  float limit = my_state->avg_err;
  if (my_state->avg_err < FLT_MAX) {
    limit = my_state->avg_err*threshold;
  }
  else { // @TODO should this be happening?
    // fprintf(stdout, "DetectCorruption: Bad avg_error %f at localx,y %d,%d - never see this bead warning again\n", p->my_state->avg_err,p->x,p->y);
  } 
  for (int fnum = flow_block_size-1; (my_err.mean_residual_error[fnum] > limit) && fnum >= 0 ; fnum--)
    high_err_cnt++;

  // stop at the first non-excessive flow and check for too many in a row
  if (high_err_cnt > decision)
  {
    my_state->corrupt = true;
  }
}


// cumulative average error for this bead
void BeadParams::UpdateCumulativeAvgError (const error_track &my_err, 
        int last_flow_of_current_block,
        int flow_block_size)
{
  float total_error;

  // How many flows have been done before the current flow?
  float num_flows_previous_average = last_flow_of_current_block-flow_block_size; 
  const float effective_infinity = 10000000.0f;
  // as this is >zero< for the first block (flow=20), total_error = 0.0
  if (my_state->avg_err<=effective_infinity or num_flows_previous_average<0.5f)
  {
    total_error = my_state->avg_err * num_flows_previous_average;
    for (int i=0; i<flow_block_size; i++)
      total_error += my_err.mean_residual_error[i]; // add the errors for each flow to total error
    my_state->avg_err = total_error/last_flow_of_current_block; // "average over all flows"
    if (my_state->avg_err > effective_infinity)
    {
      fprintf(stdout, "UpdateCumulativeError: Bad avg_error %f at localx,y %d,%d last_flow_of_current_block %d- never see this bead warning again\n", 
        my_state->avg_err,x,y,last_flow_of_current_block);
      // alert analysis that something very bad has happened here
      //my_state->corrupt = true;
      //my_state->bad_read = true;
    }
  }
}

void BeadParams::UpdateCumulativeAvgIncorporation(int flow, int flow_block_size){
  // -vm: adjust phi (average incorparation rate) 
  // based on amplitudes from the last finished flow block
  // reconstruct sum of amplitudes before this flow block; phi=0.6 during first flow block
  float old_sumAmpl = phi * (flow + 1 - flow_block_size);

  float this_sumAmpl = 0;
  for (int iFlow=0; iFlow<flow_block_size; iFlow++){
    this_sumAmpl += Ampl[iFlow];
  }

  phi = (old_sumAmpl + this_sumAmpl)/(flow + 1);

}


void BeadParams::ComputeEmphasisOneBead(int *WhichEmphasis, float *Ampl, int max_emphasis, 
    int flow_block_size)
{
    for (int i=0;i < flow_block_size;i++)
    {
      int nev = (int) (Ampl[i]);
      if (nev> max_emphasis) nev=max_emphasis;
      WhichEmphasis[i] = nev;
    }
}

void BeadParams::DumpBeadTitle (FILE *my_fp, int flow_block_size)
{
  fprintf (my_fp,"x\ty\t");
  fprintf (my_fp,"Copies\tetbR\t");
  fprintf (my_fp,"gain\tdmult\t");
  for (int i=0; i<flow_block_size; i++)
    fprintf (my_fp,"A%d\t",i);
  for (int i=0; i<flow_block_size; i++)
    fprintf (my_fp,"M%d\t",i);
  fprintf (my_fp,"avg_err\t");
  fprintf (my_fp,"clonal\tcorrupt\tbad_read\t");
  for (int i=0; i<NUM_DM_PCA; i++)
    fprintf(my_fp,"PCA_DM%d\t",i);
  fprintf(my_fp,"tau_adj\t");
  fprintf(my_fp,"phi");
  fprintf (my_fp,"\n");
}

//@TODO replace by nice HDF5 object
void BeadParams::DumpBeadProfile (FILE* my_fp, int offset_col, int offset_row, int flow_block_size)
{
  fprintf (my_fp, "%d\t%d\t", x+offset_col,y+offset_row); // put back into absolute chip coordinates
  fprintf (my_fp,"%f\t%f\t", Copies,R);
  fprintf (my_fp,"%f\t%f\t",gain,dmult);
  for (int i=0; i<flow_block_size; i++)
    fprintf (my_fp,"%0.3f\t",Ampl[i]);
  for (int i=0; i<flow_block_size; i++)
    fprintf (my_fp,"%0.3f\t",kmult[i]);
  fprintf (my_fp, "%0.3f\t",my_state->avg_err);
  fprintf (my_fp,"%d\t%d\t%d\t", (my_state->clonal_read ? 1 : 0), (my_state->corrupt ? 1:0), (my_state->bad_read ? 1:0));
  //new variables added
  for (int i=0; i<NUM_DM_PCA; i++)
    fprintf(my_fp,"%f\t",pca_vals[i]);
  fprintf(my_fp,"%f\t",tau_adj);
  fprintf(my_fp,"%f",phi);

  fprintf (my_fp,"\n");
}

bool BeadParams::FitBeadLogic()
{
  // this may be an elaborate function of the state
  return ((my_state->random_samp or my_state->clonal_read) and 
          (not my_state->corrupt) and 
          (not my_state->pinned));
}
