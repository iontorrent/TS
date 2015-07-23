/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONPARAMS_H
#define REGIONPARAMS_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <string.h>
#include "Serialization.h"

#ifdef ION_COMPILE_CUDA
  #include <host_defines.h>   // for __host__ and __device__
#endif

// Isolate the over-parameterized 
//
// Within this structure, there are a number of array parameters that are of size
// MAX_NUM_FLOWS_IN_BLOCK_GPU. In an ideal world, these could be dynamically sized.
// In this world, however, these structures are copied over to the GPU in raw form.
// If we allocated them dynamically, we'd need extra copies, which aren't worth the time.
// Should this balance change someday, it'd be reasonable to try and allocate these on the fly.
class nuc_rise_params
{
  // These chunks of internal data storage ought to be protected from unauthorized use.
  // Even if there's a full Access() routine, we want to be able to rearrange or reallocate them.

  // rise timing parameters
  float t_mid_nuc[MAX_NUM_FLOWS_IN_BLOCK_GPU];

public:
  // empirically-derived per-nuc modifiers for t_mid_nuc and sigma
  float t_mid_nuc_delay[NUMNUC];
  float sigma;
  float sigma_mult[NUMNUC];

  // refined t_mid_nuc
  float t_mid_nuc_shift_per_flow[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // note how this is redundant(!)
  
  // not actually modified,should be an input parameter(!)
  float C[NUMNUC]; // dntp in uM
  float valve_open; // timing parameter
  float nuc_flow_span; // timing of nuc flow
  float magic_divisor_for_timing; // timing parameter

  #ifdef ION_COMPILE_CUDA
  __host__ __device__
  #endif
  const float * AccessTMidNuc() const       { return t_mid_nuc; }
        float * AccessTMidNuc()             { return t_mid_nuc; }
        float * AccessSigma()               { return &sigma; }
        float * AccessTMidNucDelay()        { return t_mid_nuc_delay; }
        float * AccessSigmaMult()           { return sigma_mult; }
        float * AccessTMidNucShiftPerFlow() { return t_mid_nuc_shift_per_flow; }

  void ResetPerFlowTimeShift( int flow_block_size );

private:
  friend class boost::serialization::access;
  // Boost serialization support:
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    // fprintf(stdout, "Serialize nuc_rise_params ... ");
    ar
      & t_mid_nuc
      & t_mid_nuc_delay
      & sigma
      & sigma_mult
      & t_mid_nuc_shift_per_flow
      & C
      & valve_open
      & nuc_flow_span
      & magic_divisor_for_timing;
    // fprintf(stdout, "done nuc_rise_params\n");
  }

};


float GetModifiedMidNucTime(nuc_rise_params *cur,int NucID, int fnum);
float GetModifiedIncorporationEnd(nuc_rise_params *cur, int NucID, int fnum, float mer_guess);
float GetModifiedSigma(nuc_rise_params *cur,int NucID);

float GetTypicalMidNucTime(nuc_rise_params *cur);

// parameters shared across a whole region
//
// Within this structure, there are a number of array parameters that are of size
// MAX_NUM_FLOWS_IN_BLOCK_GPU. In an ideal world, these could be dynamically sized.
// In this world, however, these structures are copied over to the GPU in raw form.
// If we allocated them dynamically, we'd need extra copies, which aren't worth the time.
// Should this balance change someday, it'd be reasonable to try and allocate these on the fly.
struct reg_params
{
  // enzyme and nucleotide rates
  float krate[NUMNUC];// rate of incorporation
  float d[NUMNUC];    // dNTP diffusion rate
  float kmax[NUMNUC];  // saturation per nuc action rate

  // relative timing parameter to traces
  float tshift;
 
  // parameters controlling the evolution of the buffering for beads
  // model empty & bead buffering as falling on a particular line
  // this stabilizes the estimate across the region
  float reg_error; // regional error/residual during region_param fit, output to region_param.h5 to check for convergence with heatmap
  float tau_R_m;  // relationship of empty to bead slope
  float tau_R_o;  // relationship of empty to bead offset
  float tauE;
  float min_tauB;
  float mid_tauB;
  float max_tauB; // range of possible values
  float RatioDrift;      // change over time in buffering
  float NucModifyRatio[NUMNUC];  // buffering modifier per nuc

  // account for typical change in number of live copies per bead over time
  float CopyDrift;

  // strength of unexplained background "dark matter"
  float darkness[MAX_NUM_FLOWS_IN_BLOCK_GPU];

  // used during regional-fitting to allow the region-wide fit to adjust these parameters
  // across all reads.  This allows the fit to efficiently optimize across both region-wide
  // and per-well parameters
  
  // !!!do not store the following in the .h5 file, except sens and nuc_shape!!!
  // It might be better to create another structure to copy the things that need to be output to there and output
  float Ampl[MAX_NUM_FLOWS_IN_BLOCK_GPU];
  float R;
  float Copies;
  
  // cache for use in fitting
  // keep the current "copydrift" pattern
  // only recompute as copydrift changes
  float copy_multiplier[MAX_NUM_FLOWS_IN_BLOCK_GPU];

  // not fitted, inputs to the model
  float sens; // conversion b/w protons generated and signal - no reason why this should vary by nuc as hydrogens are hydrogens.
  float molecules_to_micromolar_conversion; // depends on volume of well
  // fitted differently
  nuc_rise_params nuc_shape;
  bool fit_taue;
  bool use_alternative_etbR_equation;
  bool use_log_taub;

  int hydrogenModelType;

  // Region level models dependent only on regional parameters
  float ComputeTauBfromEmptyUsingRegionLinearModel(float etbR) const;
  float AdjustEmptyToBeadRatioForFlow(float etbR_original, float Ampl, float Copy, float phi, int nuc_id, int flow) const;
  float CalculateCopyDrift(int absolute_flow) const;

  float * AccessD()                   { return d; }
  float * AccessAmpl()                { return Ampl; }
  float * AccessKrate()               { return krate; }
  float * AccessKmax()                { return kmax; }
  float * AccessNucModifyRatio()      { return NucModifyRatio; }
  float * AccessDarkness()            { return darkness; }
  float * AccessR()                   { return & R; }
  float * AccessRegError()            { return & reg_error; }
  float * AccessTauRM()               { return & tau_R_m; }
  float * AccessTauRO()               { return & tau_R_o; }
  float * AccessTauE()                { return & tauE; }
  float * AccessRatioDrift()          { return & RatioDrift; }
  float * AccessCopyDrift()           { return & CopyDrift; }
  float * AccessTShift()              { return & tshift; }
  float * AccessCopies()              { return & Copies; }
  float * AccessTMidNuc()             { return nuc_shape.AccessTMidNuc(); }
  float * AccessSigma()               { return nuc_shape.AccessSigma(); }
  float * AccessTMidNucDelay()        { return nuc_shape.AccessTMidNucDelay(); }
  float * AccessSigmaMult()           { return nuc_shape.AccessSigmaMult(); }  
  float * AccessTMidNucShiftPerFlow() { return nuc_shape.AccessTMidNucShiftPerFlow(); }

private:
  // Boost serialization support:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    // fprintf(stdout, "Serialize reg_params ... ");
    ar
      & krate
      & d
      & kmax
      & tshift
      & reg_error
      & tau_R_m
      & tau_R_o
      & tauE
      & min_tauB
      & mid_tauB
      & max_tauB
      & fit_taue
      & use_alternative_etbR_equation
      & use_log_taub
      & hydrogenModelType
      & RatioDrift
      & NucModifyRatio
      & CopyDrift
      & darkness
      & Ampl
      & R
      & Copies
      & copy_multiplier
      & sens
      & molecules_to_micromolar_conversion
      & nuc_shape;
    // fprintf(stdout, "done reg_params\n");
  }


public:
  void ApplyUpperBound(const reg_params *bound, int flow_block_size);
  void ApplyLowerBound(const reg_params *bound, int flow_block_size);

  // standard initialization values
  void SetStandardHigh( float t_mid_nuc_start, int flow_block_size );
  void SetStandardLow( float t_mid_nuc_start, int flow_block_size );
                           
  void SetStandardValue( float t_mid_nuc_start, float sigma_start, float *dntp_concentration_in_uM,
                         bool _fit_taue, bool _use_alternative_etbR_equation, bool _use_log_taub,
                         int _hydrogenModelType, int flow_block_size );
  void SetTshift(float _tshift);
  static void DumpRegionParamsTitle(FILE *my_fp, int flow_block_size);
  void DumpRegionParamsLine(FILE *my_fp,int my_row, int my_col, int flow_block_size);
};


struct reg_params_H5
{
  // enzyme and nucleotide rates
  float krate[NUMNUC];// rate of incorporation
  float d[NUMNUC];    // dNTP diffusion rate
  float kmax[NUMNUC];  // saturation per nuc action rate

  // relative timing parameter to traces
  float tshift;

  // parameters controlling the evolution of the buffering for beads
  // model empty & bead buffering as falling on a particular line
  // this stabilizes the estimate across the region
  float reg_error; // regional error/residual during region_param fit, output to region_param.h5 to check for convergence with heatmap
  float tau_R_m;  // relationship of empty to bead slope
  float tau_R_o;  // relationship of empty to bead offset
  float tauE;
  float min_tauB;
  float mid_tauB;
  float max_tauB; // range of possible values
  float RatioDrift;      // change over time in buffering
  float NucModifyRatio[NUMNUC];  // buffering modifier per nuc

  // account for typical change in number of live copies per bead over time
  float CopyDrift;

  // strength of unexplained background "dark matter"
  float darkness[MAX_NUM_FLOWS_IN_BLOCK_GPU];

  // not fitted, inputs to the model
  float sens; // conversion b/w protons generated and signal - no reason why this should vary by nuc as hydrogens are hydrogens.

  // fitted differently
  nuc_rise_params nuc_shape;
};

void reg_params_copyTo_reg_params_H5(reg_params &rp, reg_params_H5 &rph5);
void reg_params_H5_copyTo_reg_params(reg_params_H5 &rp5, reg_params &rp);


// the actual computations, so they can be exported to R
float xComputeTauBfromEmptyUsingRegionLinearModel(float tau_R_m,float tau_R_o, float etbR, float min_tauB, float max_tauB);
float xAdjustEmptyToBeadRatioForFlow(float etbR_original, float Ampl, float Copy, float phi, float NucModifyRatio, float RatioDrift, int flow, bool if_use_alternative_etbR_equation);
float xComputeTauBfromEmptyUsingRegionLinearModelWithAdjR(float tauE,float adjR, float min_tauB, float max_tauB);
float xAdjustEmptyToBeadRatioForFlowWithAdjR(float etbR_original, float NucModifyRatio, float RatioDrift, int flow);

// individual bits read from global defaults
void reg_params_setKrate(reg_params *cur, float *krate_default);
void reg_params_setDiffusion(reg_params *cur, float *d_default);
void reg_params_setKmax(reg_params *cur, float *kmax_default);
void reg_params_setSens(reg_params *cur, float sens_default);
void reg_params_setConversion(reg_params *cur, float _molecules_conversion);
void reg_params_setBuffModel(reg_params *cur, float tau_R_m_default, float tau_R_o_default);
void reg_params_setBuffModel(reg_params *cur, float tau_E_default);
void reg_params_setBuffRange(reg_params *cur, float min_tauB_default, float max_tauB_default, float mid_tauB_default);
void reg_params_setNoRatioDriftValues(reg_params *cur);
void reg_params_setSigmaMult(reg_params *cur, float *sigma_mult_default);
void reg_params_setT_mid_nuc_delay (reg_params *cur, float *t_mid_nuc_delay_default);
void SetAverageDiffusion(reg_params &rp);

void ResetRegionBeadShiftsToZero(reg_params *reg_p);

void DumpRegionParamsCSV(FILE *my_fp, reg_params *reg_p);
#endif // REGIONPARAMS_H
