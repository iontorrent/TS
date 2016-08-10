/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADPARAMS_H
#define BEADPARAMS_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "Serialization.h"

// As to the difference between POSTKEY and KEY_LEN, Earl writes (12/4/13):
//   I separated them because they were slightly different semantics - in particular,
//   with the default keys and flow orders, they are always the same. With non-default keys
//   and flow orders, these numbers should be different, and arguably different per bead.
//   (i.e. by key per bead).
//
//   (one is used in the offsets to the big static structures for optimization, one is used
//   for key normalization purposes).
//
// This is the one used for key normalization purposes, and the one that could, theoretically,
// be different per-bead.
#define POSTKEY 7

// track error activity through code
// CPU oriented:  fix up for GPU required to avoid too much data volume passed
struct error_track{
  float mean_residual_error[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float tauB[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output trace.h5
  float etbR[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output trace.h5
  float bkg_leakage[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output
  int fit_type[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output
  bool converged[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output
  float initA[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float initkmult[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float t_sigma_actual[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float t_mid_nuc_actual[MAX_NUM_FLOWS_IN_BLOCK_GPU];

    error_track() {
    // Initialize to 0.f, in case we skip flows for some reason.
    for( size_t i = 0 ; i < MAX_NUM_FLOWS_IN_BLOCK_GPU ; ++i ){
      mean_residual_error[i] = 0.f;
      tauB[i] = 6.0f;
      etbR[i] = 1.0f;
      fit_type[i] = 0;
      converged[i] = false;
      initA[i] = 1.0;
      initkmult[i] = 1.0;
      bkg_leakage[i] = 0.0f;
      t_sigma_actual[i] = 0.f;
      t_mid_nuc_actual[i] = 0.f;
    }
  }
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
  //int hits_by_flow[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // temporary tracker

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
public:
  void SetBeadStandardHigh();
  void SetBeadStandardLow(float AmplLowerLimit);
};

// independent parameters for each well
struct BeadParams
{
  ///////////////////////////////////////////////////////////////////
  // DO NOT CHANGE ORDER OF DATA FIELDS WITHIN  
  // OR ADD NEW ELEMENTS ABOVE THIS SECTION !!!!
  // DOING SO WILL KILL GPU DATA ACCESS AND CAUSE ASSERTIONS IN SingleFitStream.cu
  
  // removing aligned to 16 byte boundaries requiremnt. Should not be needed. Vectorized 
  // routines should allocate 16 byte aligned buffers and copy the params. Needed to remove 
  // unnecessary bloating of the structure

  float Copies;        // P is now copies per bead, normalized to 1E+6
  float R;  // ratio of bead buffering to empty buffering - one component of many in this calculation
  float dmult;  // diffusion alteration of this well
  float gain;  // responsiveness of this well
  float Ampl[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // homopolymer length mixture
  float kmult[MAX_NUM_FLOWS_IN_BLOCK_GPU];  // individual flow multiplier to rate of enzyme action

  // these parameters are not fit in the main lev mar fitter, but are bead-specific parameters
  float pca_vals[NUM_DM_PCA]; // dark matter compensator coefficients
  float tau_adj;              // exp-tail-fitting adjustment to tau
  float phi; //-vm: tracks average rate of incorporation per flow

  // DO NOT CHANGE THE ORDER OF DATA FIELDS WITHIN  
  // THE SECTION ABOVE!!!! 
  // DOING SO WILL KILL GPU DATA ACCESS AND CAUSE ASSERTIONS IN SingleFitStream.cu
  ///////////////////////////////////////////////////////////////

  // put any non-float value after this comment.  Anything below this line can NOT be
  // a fitted parameter

  bead_state *my_state; // pointer to reduce the size of this structure
  int trace_ndx; // what trace do i correspond to
  int x,y;  // relative location within the region

  // Here we have some access routines for particular parameters of interest.
  // This is the C++ way to access arbitrary members, defined at run-time. In C, we'd use offsets.
  float * AccessR()           { return &R; }
  float * AccessCopies()      { return &Copies; }
  float * AccessDmult()       { return &dmult; }
  float * AccessAmpl()        { return Ampl; }
  float * AccessAmplPostKey() { return Ampl + POSTKEY; }
  float * AccessKmult()       { return kmult; }

  // Look for the largest change in amplitude * copies.
  float LargestAmplitudeCopiesChange( const BeadParams *that, int flow_block_size ) const;

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
    ar & phi;
    ar & trace_ndx;
    ar & my_state;
    ar & x & y;


    // fprintf(stdout, "done\n");
  } 
  void LockKey(float *key, int keylen);
  void UnLockKey(float limit_val, int keylen);
  
public:
  //@TODO: swear that these will only be used on cropped data until my hdf5 file arrives
  void DumpBeadProfile(FILE *my_fp, int offset_col, int offset_row, int flow_block_size);
  static void DumpBeadTitle(FILE *my_fp, int flow_block_size);

  void SetStandardFlow();
  void SetBeadStandardValue();
  void ApplyUpperBound(bound_params *bound, int flow_block_size);
  void ApplyLowerBound(bound_params *bound, int flow_block_size);
  void ApplyAmplitudeZeros(const int *zero, int flow_block_size);
  void ApplyAmplitudeDrivenKmultLimit(int flow_block_size, float min_threshold);

  bool FitBeadLogic();
  static void ComputeEmphasisOneBead(int *WhichEmphasis, float *Ampl, int max_emphasis, int flow_block_size);

  void UpdateCumulativeAvgIncorporation(int flow, int flow_block_size);
  void DetectCorruption(const error_track &my_err, float threshold, int decision, int flow_block_size );
  void UpdateCumulativeAvgError(const error_track &my_err, int last_flow_of_current_block, 
                                int flow_block_size );
  
  void SetAmplitude(const float *Ampl, int flow_block_size);
  void SetBeadZeroValue();
  void AccumulateBeadValue(const BeadParams *source);    // "this" is the sink
  void ScaleBeadValue(float multiplier);
};


#endif // BEADPARAMS_H
