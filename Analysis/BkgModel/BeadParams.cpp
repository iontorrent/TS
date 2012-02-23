/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BeadParams.h"

void params_ApplyUpperBound (bead_params *cur, bound_params *bound)
{
  for (int i=0;i<NUMFB;i++)
  {
    MAX_BOUND_PAIR_CHECK (Ampl[i],Ampl);
    MAX_BOUND_PAIR_CHECK (kmult[i],kmult);
  }

  MAX_BOUND_CHECK (Copies);
  MAX_BOUND_CHECK (gain);

  MAX_BOUND_CHECK (R);
  MAX_BOUND_CHECK (dmult);
};

void params_ApplyLowerBound (bead_params *cur, bound_params *bound)
{
  for (int i=0;i<NUMFB;i++)
  {
    MIN_BOUND_PAIR_CHECK (Ampl[i],Ampl);
    MIN_BOUND_PAIR_CHECK (kmult[i],kmult);
  }

  MIN_BOUND_CHECK (Copies);
  MIN_BOUND_CHECK (gain);

  MIN_BOUND_CHECK (R);
  MIN_BOUND_CHECK (dmult);
};

void params_ApplyAmplitudeZeros (bead_params *cur, int *zero)
{
  for (int i=0; i<NUMFB; i++)
    cur->Ampl[i] *= zero[i];
}

void params_SetBeadStandardHigh (bound_params *cur)
{

    cur->Ampl = MAX_HPLEN - 1;
    cur->kmult = 1.0;

  cur->Copies = 30.0;
  cur->gain   = 1.1;
  cur->R      = 1;
  cur->dmult  = 1.8;
}

void params_SetBeadStandardLow (bound_params *cur)
{

    cur->Ampl  = 0.001;
    cur->kmult = 0.25;

  cur->Copies = 0.05;
  cur->gain   = 0.9;
  cur->R      = 0.001;
  cur->dmult  = 0.2;
}

void state_Init (bead_state &my_state)
{
  my_state.avg_err = FLT_MAX; // don't filter anything, ever
  my_state.ppf         = 0.0;
  my_state.ssq         = 0.0;
  my_state.bad_read    = false;
  my_state.clonal_read = true;
  my_state.corrupt     = false;
  my_state.random_samp = false;
}

void params_SetBeadStandardValue (bead_params *cur)
{
  for (int j=0;j<NUMFB;j++)
  {
    cur->Ampl[j] = 0.5;
    cur->kmult[j] = 1.0;
    cur->WhichEmphasis[j] = 0;
  }


  cur->Copies  = 2.0;
  cur->gain    = 1.0;

  cur->R           = 0.7;
  cur->dmult       = 1.0;
  cur->trace_ndx = -1;
  cur->x = -1;
  cur->y = -1;
  state_Init (cur->my_state);
}

void params_SetAmplitude (bead_params *cur, float *Ampl)
{
  for (int i=0; i<NUMFB; i++)
    cur->Ampl[i] = Ampl[i];
}

void params_LockKey (bead_params *cur, float *key, int keylen)
{
  for (int i=0; i<keylen; i++)
    cur->Ampl[i] = key[i];
}

void params_UnLockKey (bead_params *cur, float limit_val, int keylen)
{
  for (int i=0; i<keylen; i++)
    cur->Ampl[i] =limit_val;
}


float CheckSignificantSignalChange (bead_params *start, bead_params *finish, int numfb)
{
  float achg = 0.0;
  for (int i=0;i <numfb;i++)
  {
    float chg = fabs (start->Ampl[i]*start->Copies
                      -finish->Ampl[i]*finish->Copies);
    if (chg > achg) achg = chg;
  }
  return (achg);
}

// detect unusual levels of variation in this bead
void DetectCorruption (bead_params *p, error_track &my_err, float threshold, int decision)
{
  int high_err_cnt = 0;

  // count from end the number of excessively high error flows
  for (int fnum = NUMFB-1; (fnum >= 0) && (my_err.rerr[fnum] >p->my_state.avg_err*threshold);fnum--)
    high_err_cnt++;

  // stop at the first non-excessive flow and check for too many in a row
  if (high_err_cnt > decision)
  {
    p->my_state.corrupt = true;
  }
}


// cumulative average error for this bead
void UpdateCumulativeAvgError (bead_params *p, error_track &my_err, int flow)
{
  float sum;
  sum = p->my_state.avg_err * (flow-NUMFB); // "un-average" = 0 for first block of 20
  for (int i=0; i<NUMFB; i++)
    sum += my_err.rerr[i];
  p->my_state.avg_err = sum/flow; // "average over flows"

}

void ComputeEmphasisOneBead(int *WhichEmphasis, float *Ampl, int max_emphasis)
{
    for (int i=0;i < NUMFB;i++)
    {
      int nev = (int) (Ampl[i]);
      if (nev> max_emphasis) nev=max_emphasis;
      WhichEmphasis[i] = nev;
    }
}

void DumpBeadTitle (FILE *my_fp)
{
  fprintf (my_fp,"x\ty\t");
  fprintf (my_fp,"Copies\tetbR\t");
  fprintf (my_fp,"gain\tdmult\t");
  for (int i=0; i<NUMFB; i++)
    fprintf (my_fp,"A%d\t",i);
  for (int i=0; i<NUMFB; i++)
    fprintf (my_fp,"M%d\t",i);
  fprintf (my_fp,"avg_err\t");
  fprintf (my_fp,"clonal\tcorrupt");
  fprintf (my_fp,"\n");
}

//@TODO replace by nice HDF5 object
void DumpBeadProfile (bead_params* cur, FILE* my_fp, int offset_col, int offset_row)
{
  fprintf (my_fp, "%d\t%d\t", cur->x+offset_col,cur->y+offset_row); // put back into absolute chip coordinates
  fprintf (my_fp,"%f\t%f\t", cur->Copies,cur->R);
  fprintf (my_fp,"%f\t%f\t",cur->gain,cur->dmult);
  for (int i=0; i<NUMFB; i++)
    fprintf (my_fp,"%0.3f\t",cur->Ampl[i]);
  for (int i=0; i<NUMFB; i++)
    fprintf (my_fp,"%0.3f\t",cur->kmult[i]);
  fprintf (my_fp, "%0.3f\t",cur->my_state.avg_err);
  fprintf (my_fp,"%d\t%d", (cur->my_state.clonal_read ? 1 : 0), (cur->my_state.corrupt ? 1:0));
  fprintf (my_fp,"\n");
}
