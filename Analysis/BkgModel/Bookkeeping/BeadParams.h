/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADPARAMS_H
#define BEADPARAMS_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "Serialization.h"

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

  private:
  // Boost serialization support:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
    // fprintf(stdout, "Serialize: bead_state ... ");
    ar & bad_read
      & clonal_read
      & corrupt
      & pinned
      & random_samp
      & key_norm
      & ppf
      & ssq
      & avg_err;
    // fprintf(stdout, "done\n");
  }

};
void state_Init(bead_state &my_state);

// which parameters have bounded ranges for optimization
struct bound_params
{
  float Copies;
  float R;
  float dmult;
  float gain;
  float Ampl; // note not vectors!!!!
  float kmult; // note not vectors!!!

  private:
  // Boost serialization support:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
    // fprintf(stdout, "Serialize: bound_params ... ");
    ar & Copies;
    ar & Ampl;
    ar & kmult;
    ar & R;
    ar & dmult;
    ar & gain;
    // fprintf(stdout, "done\n");
  }
};

// independent parameters for each well
struct bead_params
{
///////////////////////////////////////////////////////////////////
  // DO NOT CHANGE ORDER OF DATA FIELDS WITHIN  
  // OR ADD NEW ELELEMTSABOVE THIS SECTION !!!!
  // DOING SO WILL KILL GPU DATA ACCESS
  
  // removing aligned to 16 byte boundaries requiremnt. Should not be needed. Vectorized routines should allocate
  // 16 byte aligned buffers and copy the params. Needed to remove unnecessary bloating of the structure
  float Copies;        // P is now copies per bead, normalized to 1E+6
  float R;  // ratio of bead buffering to empty buffering - one component of many in this calculation
  float dmult;  // diffusion alteration of this well
  float gain;  // responsiveness of this well
  float Ampl[NUMFB]; // homopolymer length mixture
  float kmult[NUMFB];  // individual flow multiplier to rate of enzyme action

  // these parameters are not fit in the main lev mar fitter, but are bead-specific parameters
  float pca_vals[NUM_DM_PCA]; // dark matter compensator coefficients
  float tau_adj;              // exp-tail-fitting adjustment to tau

  // DO NOT CHANGE THE ORDER OF DATA FIELDS WITHIN  
  // IN THE SECTION ABOVE!!!! 
  // DOING SO WILL KILL GPU DATA ACCESS
  ///////////////////////////////////////////////////////////////

  // put any non-float value after this comment.  Anything below this line can NOT be
  // a fitted parameter

  bead_state *my_state; // pointer to reduce the size of this structure
  int trace_ndx; // what trace do i correspond to
  int x,y;  // relative location within the region

  private:
  // Boost serialization support:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
    // fprintf(stdout, "Serialize: bound_params ... ");
    ar & Copies;
    ar & R;
    ar & dmult;
    ar & gain;
    ar & Ampl;
    ar & kmult;
    ar & pca_vals;
    ar & tau_adj;
    ar & trace_ndx;
    ar & my_state;
    ar & x & y;
    // fprintf(stdout, "done\n");
  } 
};
void params_SetStandardFlow(bead_params *cur);
void params_ApplyUpperBound(bead_params *cur, bound_params *bound);
void params_ApplyLowerBound(bead_params *cur, bound_params *bound);
void params_SetBeadStandardHigh(bound_params *cur);
void params_SetBeadStandardLow(bound_params *cur,float AmplLowerLimit);
void params_SetBeadStandardValue(bead_params *cur);
void params_SetBeadZeroValue(bead_params *cur);
void params_ScaleBeadValue(bead_params *cur, float multiplier);
void params_AccumulateBeadValue(bead_params *sink, bead_params *source);
void params_ApplyAmplitudeZeros(bead_params *cur, int *zero);
void params_SetAmplitude(bead_params *cur, float *Ampl);
void params_LockKey(bead_params *cur, float *key, int keylen);
void params_UnLockKey(bead_params *cur, float limit_val, int keylen);
void params_ShrinkTowardsIntegers(bead_params *cur, float shrink);
float CheckSignificantSignalChange(bead_params *start, bead_params *finish, int numfb);
//@TODO: swear that these will only be used on cropped data until my hdf5 file arrives
void DumpBeadProfile(bead_params *cur, FILE *my_fp, int offset_col, int offset_row);
void DumpBeadTitle(FILE *my_fp);

void DetectCorruption(bead_params *p, error_track &my_err, float threshold, int decision);
void UpdateCumulativeAvgError(bead_params *p, error_track &my_err, int flow);
void ComputeEmphasisOneBead(int *WhichEmphasis, float *Ampl, int max_emphasis);

void params_IncrementHits(bead_params *cur);
void params_CopyHits(bead_params *tmp, bead_params *cur);
bool FitBeadLogic(bead_params *p);


#endif // BEADPARAMS_H
