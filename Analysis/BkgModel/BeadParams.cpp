/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BeadParams.h"

void params_ApplyUpperBound(bead_params *cur, bead_params *bound)
{
  for (int i=0;i<NUMFB;i++)
  {
    MAX_BOUND_CHECK(Ampl[i]);
    MAX_BOUND_CHECK(kmult[i]);
  }

  MAX_BOUND_CHECK(Copies);
  MAX_BOUND_CHECK(gain);

  MAX_BOUND_CHECK(R);
  MAX_BOUND_CHECK(dmult);
};

void params_ApplyLowerBound(bead_params *cur, bead_params *bound)
{
  for (int i=0;i<NUMFB;i++)
  {
    MIN_BOUND_CHECK(Ampl[i]);
    MIN_BOUND_CHECK(kmult[i]);
  }

  MIN_BOUND_CHECK(Copies);
  MIN_BOUND_CHECK(gain);

  MIN_BOUND_CHECK(R);
  MIN_BOUND_CHECK(dmult);
};

void params_ApplyAmplitudeZeros(bead_params *cur, int *zero)
{
  for (int i=0; i<NUMFB; i++)
    cur->Ampl[i] *= zero[i];
}

void params_SetBeadStandardHigh(bead_params *cur)
{
    for (int j=0;j<NUMFB;j++)
    {
      cur->Ampl[j]    = MAX_HPLEN - 1;
      cur->kmult[j]      = 1.0;
    }

    cur->Copies        = 30.0;
    cur->gain     = 1.1;

    cur->R          = 1;
    cur->dmult         = 1.8;
}

void params_SetBeadStandardLow(bead_params *cur)
{
    for (int j=0;j<NUMFB;j++)
    {
      cur->Ampl[j] = 0.001;
      cur->kmult[j] = 0.25;
    }


    cur->Copies         = 0.05;
    cur->gain      = 0.9;

    cur->R         = 0.001;
    cur->dmult        = 0.2;
}

void params_SetBeadStandardValue(bead_params *cur)
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
    cur->clonal_read = true;
    cur->corrupt     = false;
    cur->random_samp = false;
    cur->avg_err     = FLT_MAX;
}

void params_SetAmplitude(bead_params *cur, float *Ampl)
{
  for (int i=0; i<NUMFB; i++)
    cur->Ampl[i] = Ampl[i];
}

void params_LockKey(bead_params *cur, float *key, int keylen)
{
  for (int i=0; i<keylen; i++)
    cur->Ampl[i] = key[i];
}

void params_UnLockKey(bead_params *cur, float limit_val, int keylen)
{
  for (int i=0; i<keylen; i++)
    cur->Ampl[i] =limit_val;
}

void RescaleRerr(bead_params *cur_p, int numfb)
{
    float mean_res = 0.0;
    for (int fnum=0; fnum<numfb; fnum++)
      mean_res += cur_p->rerr[fnum];
    mean_res /= numfb;
    for (int fnum=0;fnum < numfb;fnum++)
        cur_p->rerr[fnum] /= mean_res;
}

float CheckSignificantSignalChange(bead_params *start, bead_params *finish, int numfb)
{
  float achg = 0.0;
                for (int i=0;i <numfb;i++)
                {
                    float chg = fabs(start->Ampl[i]*start->Copies
                                     -finish->Ampl[i]*finish->Copies);
                    if (chg > achg) achg = chg;
                }
     return(achg);           
}

void DumpBeadTitle(FILE *my_fp)
{
  fprintf(my_fp,"x\ty\t");
  fprintf(my_fp,"Copies\tetbR\t");
  fprintf(my_fp,"gain\tdmult\t");
  for (int i=0; i<NUMFB; i++)
    fprintf(my_fp,"A%d\t",i);
  for (int i=0; i<NUMFB; i++)
    fprintf(my_fp,"M%d\t",i);
  fprintf(my_fp,"clonal\tcorrupt");
  fprintf(my_fp,"\n");
}

//@TODO replace by nice HDF5 object
void DumpBeadProfile(bead_params* cur, FILE* my_fp, int offset_col, int offset_row)
{
  fprintf(my_fp, "%d\t%d\t", cur->x+offset_col,cur->y+offset_row); // put back into absolute chip coordinates
  fprintf(my_fp,"%f\t%f\t", cur->Copies,cur->R);
  fprintf(my_fp,"%f\t%f\t",cur->gain,cur->dmult);
  for (int i=0; i<NUMFB; i++)
    fprintf(my_fp,"%0.3f\t",cur->Ampl[i]);
  for (int i=0; i<NUMFB; i++)
    fprintf(my_fp,"%0.3f\t",cur->kmult[i]);
  fprintf(my_fp,"%d\t%d",(cur->clonal_read ? 1 : 0), (cur->corrupt ? 1:0));
  fprintf(my_fp,"\n");
}
