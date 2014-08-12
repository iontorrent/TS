/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RegionParams.h"
#include <math.h>

void reg_params::ApplyUpperBound(const reg_params *bound, int flow_block_size)
{
  
  for (int i=0;i<flow_block_size;i++)
  {
    // MAX_BOUND_CHECK(nuc_shape.t_mid_nuc_shift_per_flow);
    MAX_BOUND_CHECK(nuc_shape.AccessTMidNuc()[i]);
    MAX_BOUND_CHECK(darkness[i]);
  }

  for (int i=0;i<NUMNUC;i++)
  {
    MAX_BOUND_CHECK(krate[i]);
    MAX_BOUND_CHECK(d[i]);
    MAX_BOUND_CHECK(kmax[i]);
    MAX_BOUND_CHECK(NucModifyRatio[i]);
    MAX_BOUND_CHECK(nuc_shape.t_mid_nuc_delay[i]);
    MAX_BOUND_CHECK(nuc_shape.sigma_mult[i]);
  }

  MAX_BOUND_CHECK(sens);
  MAX_BOUND_CHECK(tshift);

  MAX_BOUND_CHECK(nuc_shape.sigma);
  MAX_BOUND_CHECK(RatioDrift);
  MAX_BOUND_CHECK(CopyDrift);
  MAX_BOUND_CHECK(tau_R_m);
  MAX_BOUND_CHECK(tau_R_o);
  MAX_BOUND_CHECK(tauE);
}

void reg_params::ApplyLowerBound(const reg_params *bound, int flow_block_size)
{
  for (int i=0;i<flow_block_size;i++)
  {
    // MIN_BOUND_CHECK(nuc_shape.t_mid_nuc_shift_per_flow);
    MIN_BOUND_CHECK(nuc_shape.AccessTMidNuc()[i]);
    MIN_BOUND_CHECK(darkness[i]);
  }

  for (int i=0;i<NUMNUC;i++)
  {
    MIN_BOUND_CHECK(krate[i]);
    MIN_BOUND_CHECK(d[i]);
    MIN_BOUND_CHECK(kmax[i]);
    MIN_BOUND_CHECK(NucModifyRatio[i]);
    MIN_BOUND_CHECK(nuc_shape.t_mid_nuc_delay[i]);
    MIN_BOUND_CHECK(nuc_shape.sigma_mult[i]);
  }
  MIN_BOUND_CHECK(sens);
  MIN_BOUND_CHECK(tshift);

  MIN_BOUND_CHECK(nuc_shape.sigma);
  MIN_BOUND_CHECK(RatioDrift);
  MIN_BOUND_CHECK(CopyDrift);
  MIN_BOUND_CHECK(tau_R_m);
  MIN_BOUND_CHECK(tau_R_o);
  MIN_BOUND_CHECK(tauE);
}

float xComputeTauBfromEmptyUsingRegionLinearModel(float tau_R_m,float tau_R_o, float etbR, float min_tauB, float max_tauB)
{
  float tauB = (tau_R_m*etbR+tau_R_o);
  if (tauB < min_tauB) tauB = min_tauB;
  if (tauB > max_tauB) tauB = max_tauB;
  return (tauB);
}

float xComputeTauBfromEmptyUsingRegionLinearModelWithAdjR(float tauE,float etbR, float min_tauB, float max_tauB)
{
  float tauB = min_tauB;
  if (etbR != 0)
    tauB = tauE/etbR;
  if (tauB < min_tauB) tauB = min_tauB;
  if (tauB > max_tauB) tauB = max_tauB;
  return (tauB);
}

float reg_params::ComputeTauBfromEmptyUsingRegionLinearModel(float etbR) const
{
  if (fit_taue)
    return(xComputeTauBfromEmptyUsingRegionLinearModelWithAdjR(tauE,etbR,min_tauB,max_tauB));
  else
    return(xComputeTauBfromEmptyUsingRegionLinearModel(tau_R_m,tau_R_o,etbR,min_tauB,max_tauB));
}

float xAdjustEmptyToBeadRatioForFlow(float etbR_original, float Ampl, float Copy, float phi, float NucModifyRatio, 
             float RatioDrift, int flow, bool if_use_alternative_etbR_equation)
{

  float ModifiedRatio, etbR;
  ModifiedRatio = etbR_original*NucModifyRatio;
  
  if (!if_use_alternative_etbR_equation)
    etbR = ModifiedRatio + (1.0f-ModifiedRatio) * RatioDrift * flow / SCALEOFBUFFERINGCHANGE; // smooth adjustment towards being as fast as an empty
  else
    // 6.0 constat: SCALEOFBUFFERINGCHANGE was 1,000, now it needs to be 6,000
    etbR = ModifiedRatio + RatioDrift * phi * Copy * flow / (6.0 * SCALEOFBUFFERINGCHANGE) ; 
  
  return etbR;
}

float xAdjustEmptyToBeadRatioForFlowWithAdjR(float etbR_original, float NucModifyRatio, float RatioDrift, int flow)
{
  float TimeAdjust, etbR;
  //float ModifiedRatio = etbR_original*NucModifyRatio;
  TimeAdjust = RatioDrift*flow/SCALEOFBUFFERINGCHANGE;
  //etbR = ModifiedRatio + (1.0f-ModifiedRatio) * TimeAdjust;   // smooth adjustment towards being as fast as an empty
  etbR = etbR_original;
  if (etbR_original != 0)
    //etbR = NucModifyRatio/(NucModifyRatio + exp(-1.0*TimeAdjust)*(1.0/etbR_original-1));
    etbR = NucModifyRatio/(NucModifyRatio + (1.0 - TimeAdjust)*(1.0/etbR_original-1));

  return (etbR);
}

float reg_params::AdjustEmptyToBeadRatioForFlow(float etbR_original, float Ampl, float Copy, float phi, int nuc_id, int flow) const
{

  if (fit_taue)
    return(xAdjustEmptyToBeadRatioForFlowWithAdjR(etbR_original,NucModifyRatio[nuc_id],RatioDrift,flow));
  else 
    return(xAdjustEmptyToBeadRatioForFlow(etbR_original, Ampl, Copy, phi, NucModifyRatio[nuc_id], RatioDrift, flow, use_alternative_etbR_equation));
}


void reg_params::SetStandardHigh( float t0_start, int flow_block_size)
{
  // per-region parameters
  for (int j=0;j<flow_block_size;j++)
  {
    darkness[j] = 2.0f;
    copy_multiplier[j] = 1.0f;
    nuc_shape.AccessTMidNuc()[j]     = t0_start+6.0f; // really????
    nuc_shape.t_mid_nuc_shift_per_flow[j] = 0.0f+3.0f;
  }


  fit_taue = false;
  use_alternative_etbR_equation = false; 

  tshift    = 3.5f;
  nuc_shape.sigma = 8.5f; // increase for super slow project

  if (use_alternative_etbR_equation)
    RatioDrift = 5.0f; 
  else
    RatioDrift = 10.0f; //-vm:

  CopyDrift = 1.0f;

  krate[TNUCINDEX] = 100.0f;
  krate[ANUCINDEX] = 100.0f;
  krate[CNUCINDEX] = 100.0f;
  krate[GNUCINDEX] = 100.0f;
  sens = 250.0f;       // counts per 10K protons generated

  d[TNUCINDEX] =  1000.0f; // decreased
  d[ANUCINDEX] =  1000.0f;
  d[CNUCINDEX] =  1000.0f;
  d[GNUCINDEX] =  1000.0f;

  kmax[TNUCINDEX] = 20000.0f;
  kmax[ANUCINDEX] = 20000.0f;
  kmax[CNUCINDEX] = 20000.0f;
  kmax[GNUCINDEX] = 20000.0f;
  tau_R_m = 100.0f;
  tau_R_o = 400.0f;
  tauE = 20.0f;
  min_tauB = 4.0f;
  max_tauB = 65.0f;

  NucModifyRatio[TNUCINDEX] = 1.1f;
  NucModifyRatio[ANUCINDEX] = 1.1f;
  NucModifyRatio[CNUCINDEX] = 1.1f;
  NucModifyRatio[GNUCINDEX] = 1.1f;
  
  nuc_shape.t_mid_nuc_delay[TNUCINDEX] = 3.1f;
  nuc_shape.t_mid_nuc_delay[ANUCINDEX] = 3.1f;
  nuc_shape.t_mid_nuc_delay[CNUCINDEX] = 3.1f;
  nuc_shape.t_mid_nuc_delay[GNUCINDEX] = 3.1f;
  nuc_shape.sigma_mult[TNUCINDEX] = 2.1f;
  nuc_shape.sigma_mult[ANUCINDEX] = 2.1f;
  nuc_shape.sigma_mult[CNUCINDEX] = 2.1f;
  nuc_shape.sigma_mult[GNUCINDEX] = 2.1f;
  
  for (int i_nuc=0; i_nuc<NUMNUC; i_nuc++)
    nuc_shape.C[i_nuc]    =  500.0f;
  
    nuc_shape.valve_open = 15.0f; // frames(!)
    nuc_shape.nuc_flow_span = 60.0f; // frames = 15.0f per second
    nuc_shape.magic_divisor_for_timing = 20.7; // frames(!)
}

void reg_params::SetStandardLow(float t0_start, int flow_block_size)
{
  // per-region parameters
  for (int j=0;j<flow_block_size;j++)
  {
    darkness[j] = 0.0f;
    copy_multiplier[j] = 0.0f; // can drift very low
    nuc_shape.AccessTMidNuc()[j]      = t0_start-6.0f; // really???
    nuc_shape.t_mid_nuc_shift_per_flow[j] = 0.0f-3.0f;
  }

  fit_taue = false;
  use_alternative_etbR_equation = false; 

  tshift    = -1.5f;
  nuc_shape.sigma  = 0.4f;
  RatioDrift    = 0.0f;
  CopyDrift    = 0.99f;

  krate[TNUCINDEX] = 0.01f;
  krate[ANUCINDEX] = 0.01f;
  krate[CNUCINDEX] = 0.01f;
  krate[GNUCINDEX] = 0.01f;
  sens =  0.5f;

  d[TNUCINDEX] =  0.1f;
  d[ANUCINDEX] =  0.1f;
  d[CNUCINDEX] =  0.1f;
  d[GNUCINDEX] =  0.1f;

  kmax[TNUCINDEX] = 5.0f;
  kmax[ANUCINDEX] = 5.0f;
  kmax[CNUCINDEX] = 5.0f;
  kmax[GNUCINDEX] = 5.0f;
  tau_R_m = -100.0f;
  tau_R_o = -100.0f;
  tauE = 1.0f;
  min_tauB = 4.0f;
  max_tauB = 65.0f;
  NucModifyRatio[TNUCINDEX] = 0.9f;
  NucModifyRatio[ANUCINDEX] = 0.9f;
  NucModifyRatio[CNUCINDEX] = 0.9f;
  NucModifyRatio[GNUCINDEX] = 0.9f;
  nuc_shape.t_mid_nuc_delay[TNUCINDEX] = -3.0f;
  nuc_shape.t_mid_nuc_delay[ANUCINDEX] = -3.0f;
  nuc_shape.t_mid_nuc_delay[CNUCINDEX] = -3.0f;
  nuc_shape.t_mid_nuc_delay[GNUCINDEX] = -3.0f;
  nuc_shape.sigma_mult[TNUCINDEX] = 0.5f;
  nuc_shape.sigma_mult[ANUCINDEX] = 0.5f;
  nuc_shape.sigma_mult[CNUCINDEX] = 0.5f;
  nuc_shape.sigma_mult[GNUCINDEX] = 0.5f;
  
  for (int i_nuc=0; i_nuc<NUMNUC; i_nuc++)
    nuc_shape.C[i_nuc]    = 1.0f;
  
    nuc_shape.valve_open = 15.0f; // frames(!)
    nuc_shape.nuc_flow_span = 15.0f; // frames = 15.0f per second
    nuc_shape.magic_divisor_for_timing = 20.7; // frames(!)
}

void reg_params_setKrate(reg_params *cur, float *krate_default)
{
  cur->krate[TNUCINDEX] = krate_default[TNUCINDEX];
  cur->krate[ANUCINDEX] = krate_default[ANUCINDEX];
  cur->krate[CNUCINDEX] = krate_default[CNUCINDEX];
  cur->krate[GNUCINDEX] = krate_default[GNUCINDEX];
}

void reg_params_setDiffusion(reg_params *cur, float *d_default)
{
  cur->d[TNUCINDEX] = d_default[TNUCINDEX];
  cur->d[ANUCINDEX] = d_default[ANUCINDEX];
  cur->d[CNUCINDEX] = d_default[CNUCINDEX];
  cur->d[GNUCINDEX] = d_default[GNUCINDEX];
}

void reg_params_setKmax(reg_params *cur, float *kmax_default)
{
  cur->kmax[TNUCINDEX] = kmax_default[TNUCINDEX];
  cur->kmax[ANUCINDEX] = kmax_default[ANUCINDEX];
  cur->kmax[CNUCINDEX] = kmax_default[CNUCINDEX];
  cur->kmax[GNUCINDEX] = kmax_default[GNUCINDEX];
}

void reg_params_setSigmaMult(reg_params *cur, float *sigma_mult_default)
{
  cur->nuc_shape.sigma_mult[TNUCINDEX] = sigma_mult_default[TNUCINDEX];
  cur->nuc_shape.sigma_mult[ANUCINDEX] = sigma_mult_default[ANUCINDEX];
  cur->nuc_shape.sigma_mult[CNUCINDEX] = sigma_mult_default[CNUCINDEX];
  cur->nuc_shape.sigma_mult[GNUCINDEX] = sigma_mult_default[GNUCINDEX];
}

void reg_params_setT_mid_nuc_delay (reg_params *cur, float *t_mid_nuc_delay_default)
{
  cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] = t_mid_nuc_delay_default[TNUCINDEX];
  cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] = t_mid_nuc_delay_default[ANUCINDEX];
  cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] = t_mid_nuc_delay_default[CNUCINDEX];
  cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] = t_mid_nuc_delay_default[GNUCINDEX];
}

void reg_params_setSens(reg_params *cur, float sens_default)
{
  cur->sens = sens_default;

}

void reg_params_setConversion(reg_params *cur, float _molecules_conversion)
{
  cur->molecules_to_micromolar_conversion = _molecules_conversion;

}

void reg_params_setBuffModel(reg_params *cur, float tau_R_m_default, float tau_R_o_default)
{
  cur->tau_R_m = tau_R_m_default;
  cur->tau_R_o = tau_R_o_default;
  cur->RatioDrift = 2.5f;
}

void reg_params_setBuffModel(reg_params *cur, float tau_E_default)
{
  cur->tauE = tau_E_default;
  cur->RatioDrift = 2.5f;
}

void reg_params_setNoRatioDriftValues(reg_params *cur)
{
    cur->RatioDrift    = 1.3f;

    // in the no-RatioDrift fit version, we added fitting these parameters
    // in this case, they have unbiased defaults
    cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] =  0.0f;
    cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] =  0.0f;
    cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] =  0.0f;
    cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] =  0.0f;

    cur->nuc_shape.sigma_mult[TNUCINDEX] = 1.0f;
    cur->nuc_shape.sigma_mult[ANUCINDEX] = 1.0f;
    cur->nuc_shape.sigma_mult[CNUCINDEX] = 1.0f;
    cur->nuc_shape.sigma_mult[GNUCINDEX] = 1.0f;
}

void reg_params::SetTshift(float _tshift){
  tshift = _tshift;
}

//@TODO: can this be exported to a sensible JSON file?
void reg_params::SetStandardValue(float t_mid_nuc_start, float sigma_start, 
        float *dntp_concentration_in_uM, bool _fit_taue, bool _use_alternative_etbR_equation,
        int _hydrogenModelType, int flow_block_size)
{
  // per-region parameters
  for (int j=0;j<flow_block_size;j++)
  {
    darkness[j] = 0.0f;
    copy_multiplier[j] = 1.0f;
    nuc_shape.AccessTMidNuc()[j]      = t_mid_nuc_start;
    nuc_shape.t_mid_nuc_shift_per_flow[j] = 0.0f;
  }
  min_tauB = 4.0f;
  max_tauB = 65.0f;

  molecules_to_micromolar_conversion = 0.000062; // 3 micron wells

  fit_taue = _fit_taue;
  use_alternative_etbR_equation = _use_alternative_etbR_equation;

  hydrogenModelType = _hydrogenModelType;

  tshift = 0.4f;
  nuc_shape.sigma  = sigma_start;
  // This is correct Nuc flow time. We need it logged in explog so that 
  // we use whatever timing is used for the experiment
  //nuc_shape.nuc_flow_span = 16.5f;
  nuc_shape.nuc_flow_span = 22.5f;
  CopyDrift    = 0.9987f;

  RatioDrift    = 2.0f;


  for (int i_nuc=0; i_nuc<NUMNUC; i_nuc++)
    nuc_shape.C[i_nuc]    =  dntp_concentration_in_uM[i_nuc];
  
    // defaults consistent w/ original v7 behavior
    nuc_shape.t_mid_nuc_delay[TNUCINDEX] =  0.69f;
    nuc_shape.t_mid_nuc_delay[ANUCINDEX] =  1.78f;
    nuc_shape.t_mid_nuc_delay[CNUCINDEX] =  0.0f;
    nuc_shape.t_mid_nuc_delay[GNUCINDEX] =  0.17f;

    nuc_shape.sigma_mult[TNUCINDEX] = 1.162f;
    nuc_shape.sigma_mult[ANUCINDEX] = 1.124f;
    nuc_shape.sigma_mult[CNUCINDEX] = 1.0f;
    nuc_shape.sigma_mult[GNUCINDEX] = 0.8533f;

    //@TODO: this is denominated in frames per second = 15
    //@TODO: please can we not do this operation at all
    nuc_shape.valve_open = 15.0f; // frames(!)
    nuc_shape.magic_divisor_for_timing = 20.7; // frames(!)

  NucModifyRatio[TNUCINDEX] = 1.0f;
  NucModifyRatio[ANUCINDEX] = 1.0f;
  NucModifyRatio[CNUCINDEX] = 1.0f;
  NucModifyRatio[GNUCINDEX] = 1.0f;
}

void reg_params::DumpRegionParamsTitle(FILE *my_fp, int flow_block_size)
{
  fprintf(my_fp,"row\tcol\td[0]\td[1]\td[2]\td[3]\tkr[0]\tkr[1]\tkr[2]\tkr[3]\tkmax[0]\tkmax[1]\tkmax[2]\tkmax[3]\tt_mid_nuc\tt_mid_nuc[0]\tt_mid_nuc[1]\tt_mid_nuc[2]\tt_mid_nuc[3]\tsigma\tsigma[0]\tsigma[1]\tsigma[2]\tsigma[3]\tNucModifyRatio[0]\tNucModifyRatio[1]\tNucModifyRatio[2]\tNucModifyRatio[3]\ttau_m\ttau_o\ttauE\trdr\tpdr\ttshift");
  for (int i=0; i<flow_block_size; i++)
    fprintf(my_fp,"\tt_mid_flow[%d]",i);
  fprintf(my_fp,"\n");
}



void reg_params::DumpRegionParamsLine(FILE *my_fp,int my_row, int my_col, int flow_block_size)
{
  // officially the wrong way to do this
  fprintf(my_fp,"%4d\t%4d\t", my_row,my_col);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",d[TNUCINDEX],d[ANUCINDEX],d[CNUCINDEX],d[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",krate[TNUCINDEX],krate[ANUCINDEX],krate[CNUCINDEX],krate[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",kmax[TNUCINDEX],kmax[ANUCINDEX],kmax[CNUCINDEX],kmax[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t",nuc_shape.AccessTMidNuc()[0]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",GetModifiedMidNucTime(&nuc_shape,TNUCINDEX,0),GetModifiedMidNucTime(&nuc_shape,ANUCINDEX,0),GetModifiedMidNucTime(&nuc_shape,CNUCINDEX,0),GetModifiedMidNucTime(&nuc_shape,GNUCINDEX,0));
  fprintf(my_fp,"%5.3f\t",nuc_shape.sigma);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",GetModifiedSigma(&nuc_shape,TNUCINDEX),GetModifiedSigma(&nuc_shape,ANUCINDEX),GetModifiedSigma(&nuc_shape,CNUCINDEX),GetModifiedSigma(&nuc_shape,GNUCINDEX));
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",NucModifyRatio[TNUCINDEX],NucModifyRatio[ANUCINDEX],NucModifyRatio[CNUCINDEX],NucModifyRatio[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t",tau_R_m,tau_R_o,tauE,RatioDrift,CopyDrift,tshift);
  for (int i=0; i<flow_block_size; i++)
    fprintf(my_fp,"%5.3f\t",nuc_shape.t_mid_nuc_shift_per_flow[i]);
  fprintf(my_fp,"\n");
}

float GetTypicalMidNucTime(nuc_rise_params *cur)
{
    return(cur->AccessTMidNuc()[0]);
}

void nuc_rise_params::ResetPerFlowTimeShift(int flow_block_size)
{
  for (int fnum=0; fnum<flow_block_size; fnum++)
    t_mid_nuc_shift_per_flow[fnum] = 0.0f;
}

float GetModifiedMidNucTime(nuc_rise_params *cur, int NucID, int fnum)
{
  float retval_time = cur->AccessTMidNuc()[0];
  retval_time +=  cur->t_mid_nuc_delay[NucID]* (cur->AccessTMidNuc()[0]-cur->valve_open) /(cur->magic_divisor_for_timing+SAFETYZERO);
  retval_time += cur->t_mid_nuc_shift_per_flow[fnum];
  return(retval_time);
}

float GetModifiedIncorporationEnd(nuc_rise_params *cur, int NucID, int fnum, float mer_guess){
  float my_t_mid_nuc = GetModifiedMidNucTime(cur,NucID,fnum);

  // estimated average end of incorporation
  // for PI chip

  float fi_end = my_t_mid_nuc + MIN_INCORPORATION_TIME_PI + MIN_INCORPORATION_TIME_PER_MER_PI_VERSION_ONE*mer_guess;
  return(fi_end);
}

float GetModifiedSigma(nuc_rise_params *cur, int NucID)
{
    return(cur->sigma*cur->sigma_mult[NucID]);  // to make sure we knwo that this is modified
}

float reg_params::CalculateCopyDrift(int absolute_flow) const
{
    return pow (CopyDrift,absolute_flow);
}

void SetAverageDiffusion(reg_params &rp)
{
  // equalize diffusion parameters across all Nucs
  // this is a hack to avoid having to change the lev-mar control
  // if we were clever, this would be a geometric average...
  float sum=0.0f;
  for (int idx=0; idx<NUMNUC; idx++)
    sum += rp.d[idx];
  sum /= NUMNUC;
  for (int idx=0; idx<NUMNUC; idx++)
    rp.d[idx] = sum;
}


void ResetRegionBeadShiftsToZero(reg_params *reg_p)
{
        memset(reg_p->Ampl,0,sizeof(reg_p->Ampl));
        reg_p->R = 0.0f;
        reg_p->Copies = 0.0f;
}

void DumpRegionParamsCSV(FILE *my_fp, reg_params *reg_p)
{
    fprintf(my_fp,"%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,",
            reg_p->nuc_shape.sigma,reg_p->RatioDrift,reg_p->CopyDrift,reg_p->nuc_shape.AccessTMidNuc()[0],reg_p->tshift,reg_p->krate[0],reg_p->krate[1],reg_p->krate[2],reg_p->krate[3],reg_p->d[0],reg_p->d[1],reg_p->d[2],reg_p->d[3]);
}



// adjust the parameter based on the well size we are analyzing
float ComputeMoleculesToMicromolarConversion(float well_height, float well_diameter)
{
    float avagadro = 6.02E23f;
    float micromole = 1E6f;
    float cubic_micron_to_cc = 1E-12f;
    float cc_to_mL = 1.0f;
    float mL_to_L = 1E-3f;
    float my_pi = 3.14159f;

    // well is approximately a cylinder
    float cubic_well = well_height * (well_diameter*well_diameter/4)*my_pi;
    // convert to micromolar
    float retval = (1.0f/avagadro)*micromole/(cubic_well*cubic_micron_to_cc*cc_to_mL*mL_to_L);
    return(retval);
}

void reg_params_copyTo_reg_params_H5 ( reg_params &rp, reg_params_H5 &rp5 )
{
  for ( int i=0; i<NUMNUC; i++ )
  {
    rp5.krate[i] = rp.krate[i];
    rp5.d[i] = rp.d[i];
    rp5.kmax[i] = rp.kmax[i];
    rp5.NucModifyRatio[i] = rp.NucModifyRatio[i];
  }
  for ( int i=0; i<MAX_NUM_FLOWS_IN_BLOCK_GPU; i++ )
  {
    rp5.darkness[i] = rp.darkness[i];
  }
  rp5.tshift = rp.tshift;
  rp5.tau_R_m = rp.tau_R_m;
  rp5.tau_R_o = rp.tau_R_o;
  rp5.RatioDrift = rp.RatioDrift;
  rp5.CopyDrift = rp.CopyDrift;
  rp5.sens = rp.sens;
  rp5.tauE = rp.tauE;
  rp5.nuc_shape = rp.nuc_shape;

}

void reg_params_H5_copyTo_reg_params ( reg_params_H5 &rp5, reg_params &rp )
{
  for ( int i=0; i<NUMNUC; i++ )
  {
    rp.krate[i] = rp5.krate[i];
    rp.d[i] = rp5.d[i];
    rp.kmax[i] = rp5.kmax[i];
    rp.NucModifyRatio[i] = rp5.NucModifyRatio[i];
  }
  for ( int i=0; i<MAX_NUM_FLOWS_IN_BLOCK_GPU; i++ )
  {
    rp.darkness[i] = rp5.darkness[i];
  }
  rp.tshift = rp5.tshift;
  rp.tau_R_m = rp5.tau_R_m;
  rp.tau_R_o = rp5.tau_R_o;
  rp.RatioDrift = rp5.RatioDrift;
  rp.CopyDrift = rp5.CopyDrift;
  rp.sens = rp5.sens;
  rp.tauE = rp5.tauE;
  rp.nuc_shape = rp5.nuc_shape;

}

