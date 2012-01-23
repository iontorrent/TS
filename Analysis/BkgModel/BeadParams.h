/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADPARAMS_H
#define BEADPARAMS_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

// independent parameters for each well
struct bead_params
{
  float Copies __attribute__ ((aligned (16)));        // P is now copies per bead, normalized to 1E+6
  float Ampl[NUMFB]  __attribute__ ((aligned (16))); // homopolymer length mixture
  float kmult[NUMFB] __attribute__ ((aligned (16)));  // individual flow multiplier to rate of enzyme action


  float R __attribute__ ((aligned (16)));  // ratio of bead buffering to empty buffering - one component of many in this calculation
  float dmult __attribute__ ((aligned (16)));  // diffusion alteration of this well
  float gain __attribute__ ((aligned (16)));  // responsiveness of this well

  float rerr[NUMFB] __attribute__ ((aligned (16)));

  // put any non-float value after this comment.  Anything below this line can NOT be
  // a fitted parameter
  bool clonal_read;  // am I a high quality read that can be used for some parameter fitting purposes?
  bool corrupt;      // has this bead been lost for whatever reason
  bool random_samp;  // process this bead no matter how the above flags are set
  int  WhichEmphasis[NUMFB];  // which emphasis vector do I choose for each flow in the frame buffer

  float avg_err;  // keeps track of running average fit error for this bead
  int trace_ndx; // what trace do i correspond to
  int x,y;  // relative location within the region
};
void params_ApplyUpperBound(bead_params *cur, bead_params *bound);
void params_ApplyLowerBound(bead_params *cur, bead_params *bound);
void params_SetBeadStandardHigh(bead_params *cur);
void params_SetBeadStandardLow(bead_params *cur);
void params_SetBeadStandardValue(bead_params *cur);
void params_ApplyAmplitudeZeros(bead_params *cur, int *zero);
void params_SetAmplitude(bead_params *cur, float *Ampl);
void params_LockKey(bead_params *cur, float *key, int keylen);
void params_UnLockKey(bead_params *cur, float limit_val, int keylen);
void RescaleRerr(bead_params *cur_p, int numfb);
float CheckSignificantSignalChange(bead_params *start, bead_params *finish, int numfb);
//@TODO: swear that these will only be used on cropped data until my hdf5 file arrives
void DumpBeadProfile(bead_params *cur, FILE *my_fp, int offset_col, int offset_row);
void DumpBeadTitle(FILE *my_fp);
#endif // BEADPARAMS_H
