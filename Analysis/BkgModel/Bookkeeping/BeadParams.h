/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADPARAMS_H
#define BEADPARAMS_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

// track error activity through code
struct error_track{
  float mean_residual_error[NUMFB];
};

// track status of bead, not fitted parameters
struct bead_state{
  bool bad_read;     // bad key, infinite signal,
  bool clonal_read;  // am I a high quality read that can be used for some parameter fitting purposes?
  bool corrupt;      // has this bead been lost for whatever reason
  bool pinned;       // this bead got pinned in some flow
  bool random_samp;  // process this bead no matter how the above flags are set
  // track classification entities across blocks of flows
  float key_norm;
  float ppf;
  float ssq;
  // track cumulative average error looking to detect corruption
  float avg_err;
  // per bead loading: temporary
  //int hits_by_flow[NUMFB]; // temporary tracker

};
void state_Init(bead_state &my_state);

// which parameters have bounded ranges for optimization
struct bound_params
{
  float Copies;
  float Ampl; // note not vectors!!!!
  float kmult; // note not vectors!!!
  float R;
  float dmult;
  float gain;
};

// independent parameters for each well
struct bead_params
{
  float Copies __attribute__ ((aligned (16)));        // P is now copies per bead, normalized to 1E+6
  float Ampl[NUMFB]  __attribute__ ((aligned (16))); // homopolymer length mixture
  float kmult[NUMFB] __attribute__ ((aligned (16)));  // individual flow multiplier to rate of enzyme action


  float R __attribute__ ((aligned (16)));  // ratio of bead buffering to empty buffering - one component of many in this calculation
  float dmult __attribute__ ((aligned (16)));  // diffusion alteration of this well
  float gain __attribute__ ((aligned (16)));  // responsiveness of this well


  // put any non-float value after this comment.  Anything below this line can NOT be
  // a fitted parameter

  bead_state my_state;
  int trace_ndx; // what trace do i correspond to
  int x,y;  // relative location within the region
};
void params_SetStandardFlow(bead_params *cur);
void params_ApplyUpperBound(bead_params *cur, bound_params *bound);
void params_ApplyLowerBound(bead_params *cur, bound_params *bound);
void params_SetBeadStandardHigh(bound_params *cur);
void params_SetBeadStandardLow(bound_params *cur);
void params_SetBeadStandardValue(bead_params *cur);
void params_ApplyAmplitudeZeros(bead_params *cur, int *zero);
void params_SetAmplitude(bead_params *cur, float *Ampl);
void params_LockKey(bead_params *cur, float *key, int keylen);
void params_UnLockKey(bead_params *cur, float limit_val, int keylen);
float CheckSignificantSignalChange(bead_params *start, bead_params *finish, int numfb);
//@TODO: swear that these will only be used on cropped data until my hdf5 file arrives
void DumpBeadProfile(bead_params *cur, FILE *my_fp, int offset_col, int offset_row);
void DumpBeadTitle(FILE *my_fp);

void DetectCorruption(bead_params *p, error_track &my_err, float threshold, int decision);
void UpdateCumulativeAvgError(bead_params *p, error_track &my_err, int flow);
void ComputeEmphasisOneBead(int *WhichEmphasis, float *Ampl, int max_emphasis);

void params_IncrementHits(bead_params *cur);
void params_CopyHits(bead_params *tmp, bead_params *cur);

#endif // BEADPARAMS_H
