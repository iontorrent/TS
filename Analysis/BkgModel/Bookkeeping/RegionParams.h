/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONPARAMS_H
#define REGIONPARAMS_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <string.h>

// Isolate the over-parameterized 
// First n pointers must be floats so they can be cast properly in the CpuStep structure
// which seems like a lot of work to avoid a switch statement
struct nuc_rise_params
{
  // rise timing parameters
  float t_mid_nuc[NUMFB];
  // empirically-derived per-nuc modifiers for t_mid_nuc and sigma
  float t_mid_nuc_delay[NUMNUC];
  float sigma;
  float sigma_mult[NUMNUC];

  // refined t_mid_nuc
  float t_mid_nuc_shift_per_flow[NUMFB]; // note how this is redundant(!)
  
  // not actually modified,should be an input parameter(!)
  float C; // dntp in uM
  float valve_open; // timing parameter
  float magic_divisor_for_timing; // timing parameter
};

float GetModifiedMidNucTime(nuc_rise_params *cur,int NucID, int fnum);
float GetModifiedSigma(nuc_rise_params *cur,int NucID);

float GetTypicalMidNucTime(nuc_rise_params *cur);
void ResetPerFlowTimeShift(nuc_rise_params *cur);

// parameters shared across a whole region
// first "n" parameters must be floats for the annoying "casting" of pointers to work properly
// in the CpuStep structure
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
  float tau_R_m;  // relationship of empty to bead slope
  float tau_R_o;  // relationship of empty to bead offset
  float RatioDrift;      // change over time in buffering
  float NucModifyRatio[NUMNUC];  // buffering modifier per nuc

  // account for typical change in number of live copies per bead over time
  float CopyDrift;

  // strength of unexplained background "dark matter"
  float darkness[NUMFB] __attribute__ ((aligned (16)));

  // used during regional-fitting to allow the region-wide fit to adjust these parameters
  // across all reads.  This allows the fit to efficiently optimize across both region-wide
  // and per-well parameters
  
  // !!!do not store the following in the .h5 file, except sens and nuc_shape!!!
  // It might be better to create another structure to copy the things that need to be output to there and output
  float Ampl[NUMFB] __attribute__ ((aligned (16)));
  float R;
  float Copies;
  
  // cache for use in fitting
  // keep the current "copydrift" pattern
  // only recompute as copydrift changes
  float copy_multiplier[NUMFB] __attribute__ ((aligned (16)));

  // not fitted, inputs to the model
  float sens; // conversion b/w protons generated and signal - no reason why this should vary by nuc as hydrogens are hydrogens.
  float molecules_to_micromolar_conversion; // depends on volume of well
  // fitted differently
  nuc_rise_params nuc_shape;

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
  float tau_R_m;  // relationship of empty to bead slope
  float tau_R_o;  // relationship of empty to bead offset
  float RatioDrift;      // change over time in buffering
  float NucModifyRatio[NUMNUC];  // buffering modifier per nuc

  // account for typical change in number of live copies per bead over time
  float CopyDrift;

  // strength of unexplained background "dark matter"
  float darkness[NUMFB];

  /*
  float darkness[NUMFB] __attribute__ ((aligned (16)));

  // used during regional-fitting to allow the region-wide fit to adjust these parameters
  // across all reads.  This allows the fit to efficiently optimize across both region-wide
  // and per-well parameters

  // !!!do not store the following in the .h5 file, except sens and nuc_shape!!!
  // It might be better to create another structure to copy the things that need to be output to there and output
  float Ampl[NUMFB] __attribute__ ((aligned (16)));
  float R;
  float Copies;

  // cache for use in fitting
  // keep the current "copydrift" pattern
  // only recompute as copydrift changes
  float copy_multiplier[NUMFB] __attribute__ ((aligned (16)));
  */

  // not fitted, inputs to the model
  float sens; // conversion b/w protons generated and signal - no reason why this should vary by nuc as hydrogens are hydrogens.

  // fitted differently
  nuc_rise_params nuc_shape;

};

void reg_params_copyTo_reg_params_H5(reg_params &rp, reg_params_H5 &rph5);
void reg_params_H5_copyTo_reg_params(reg_params_H5 &rp5, reg_params &rp);

void reg_params_ApplyUpperBound(reg_params *cur, reg_params *bound);
void reg_params_ApplyLowerBound(reg_params *cur, reg_params *bound);

// the actual computations, so they can be exported to R
float xComputeTauBfromEmptyUsingRegionLinearModel(float tau_R_m,float tau_R_o, float etbR);
float xAdjustEmptyToBeadRatioForFlow(float etbR_original, float NucModifyRatio, float RatioDrift, int flow);

// Region level models dependent only on regional parameters
float ComputeTauBfromEmptyUsingRegionLinearModel(reg_params *reg_p, float etbR);
float AdjustEmptyToBeadRatioForFlow(float etbR_original, reg_params *reg_p, int nuc_id, int flow);
float CalculateCopyDrift(reg_params &rp, int absolute_flow);

// standard initialization values
void reg_params_setStandardHigh(reg_params *cur, float t_mid_nuc_start);
void reg_params_setStandardLow(reg_params *cur, float t_mid_nuc_start);
                          
void reg_params_setStandardValue(reg_params *cur, float t_mid_nuc_start, float sigma_start, float dntp_concentration_in_uM);

// individual bits read from global defaults
void reg_params_setKrate(reg_params *cur, float *krate_default);
void reg_params_setDiffusion(reg_params *cur, float *d_default);
void reg_params_setKmax(reg_params *cur, float *kmax_default);
void reg_params_setSens(reg_params *cur, float sens_default);
void reg_params_setConversion(reg_params *cur, float _molecules_conversion);
void reg_params_setBuffModel(reg_params *cur, float tau_R_m_default, float tau_R_o_default);
void reg_params_setNoRatioDriftValues(reg_params *cur);
void reg_params_setSigmaMult(reg_params *cur, float *sigma_mult_default);
void reg_params_setT_mid_nuc_delay (reg_params *cur, float *t_mid_nuc_delay_default);
void SetAverageDiffusion(reg_params &rp);

void ResetRegionBeadShiftsToZero(reg_params *reg_p);

void DumpRegionParamsTitle(FILE *my_fp);
void DumpRegionParamsLine(FILE *my_fp,int my_row, int my_col, reg_params &rp);
void DumpRegionParamsCSV(FILE *my_fp, reg_params *reg_p);
#endif // REGIONPARAMS_H
