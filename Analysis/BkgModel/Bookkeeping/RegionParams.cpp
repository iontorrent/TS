/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RegionParams.h"
#include <math.h>

void reg_params_ApplyUpperBound(reg_params *cur, reg_params *bound)
{
  
  for (int i=0;i<NUMFB;i++)
  {
    // MAX_BOUND_CHECK(nuc_shape.t_mid_nuc_shift_per_flow);
    MAX_BOUND_CHECK(nuc_shape.t_mid_nuc[i]);
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

void reg_params_ApplyLowerBound(reg_params *cur, reg_params *bound)
{
  for (int i=0;i<NUMFB;i++)
  {
    // MIN_BOUND_CHECK(nuc_shape.t_mid_nuc_shift_per_flow);
    MIN_BOUND_CHECK(nuc_shape.t_mid_nuc[i]);
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

float xComputeTauBfromEmptyUsingRegionLinearModel(float tau_R_m,float tau_R_o, float etbR)
{
  float tauB = (tau_R_m*etbR+tau_R_o);
  if (tauB < MINTAUB) tauB = MINTAUB;
  if (tauB > MAXTAUB) tauB = MAXTAUB;
  return (tauB);
}

float xComputeTauBfromEmptyUsingRegionLinearModelWithAdjR(float tauE,float etbR)
{
  float tauB = MINTAUB;
  if (etbR != 0)
    tauB = tauE/etbR;
  if (tauB < MINTAUB) tauB = MINTAUB;
  if (tauB > MAXTAUB) tauB = MAXTAUB;
  return (tauB);
}

float ComputeTauBfromEmptyUsingRegionLinearModel(reg_params *reg_p, float etbR)
{
  if (reg_p->fit_taue)
    return(xComputeTauBfromEmptyUsingRegionLinearModelWithAdjR(reg_p->tauE,etbR));
  else
    return(xComputeTauBfromEmptyUsingRegionLinearModel(reg_p->tau_R_m,reg_p->tau_R_o,etbR));
}

float xAdjustEmptyToBeadRatioForFlow(float etbR_original, float NucModifyRatio, float RatioDrift, int flow)
{
  float ModifiedRatio, TimeAdjust, etbR;
  ModifiedRatio = etbR_original*NucModifyRatio;
  TimeAdjust = RatioDrift*flow/SCALEOFBUFFERINGCHANGE;
  etbR = ModifiedRatio + (1.0f-ModifiedRatio) * TimeAdjust;   // smooth adjustment towards being as fast as an empty
  return (etbR);
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

float AdjustEmptyToBeadRatioForFlow(float etbR_original, reg_params *reg_p, int nuc_id, int flow)
{

  if (reg_p->fit_taue)
    return(xAdjustEmptyToBeadRatioForFlowWithAdjR(etbR_original,reg_p->NucModifyRatio[nuc_id],reg_p->RatioDrift,flow));
  else
    return(xAdjustEmptyToBeadRatioForFlow(etbR_original,reg_p->NucModifyRatio[nuc_id],reg_p->RatioDrift,flow));
}


void reg_params_setStandardHigh(reg_params *cur, float t0_start)
{
  // per-region parameters
  for (int j=0;j<NUMFB;j++)
  {
    cur->darkness[j] = 2.0f;
    cur->copy_multiplier[j] = 1.0f;
    cur->nuc_shape.t_mid_nuc[j]     = t0_start+6.0f; // really????
    cur->nuc_shape.t_mid_nuc_shift_per_flow[j] = 0.0f+3.0f;
  }


  cur->fit_taue = false;

  cur->tshift    = 3.5f;
  cur->nuc_shape.sigma = 8.5f; // increase for super slow project
  cur->RatioDrift = 5.0f;
  cur->CopyDrift = 1.0f;

  cur->krate[TNUCINDEX] = 100.0f;
  cur->krate[ANUCINDEX] = 100.0f;
  cur->krate[CNUCINDEX] = 100.0f;
  cur->krate[GNUCINDEX] = 100.0f;
  cur->sens = 250.0f;       // counts per 10K protons generated

  cur->d[TNUCINDEX] =  1000.0f; // decreased
  cur->d[ANUCINDEX] =  1000.0f;
  cur->d[CNUCINDEX] =  1000.0f;
  cur->d[GNUCINDEX] =  1000.0f;

  cur->kmax[TNUCINDEX] = 20000.0f;
  cur->kmax[ANUCINDEX] = 20000.0f;
  cur->kmax[CNUCINDEX] = 20000.0f;
  cur->kmax[GNUCINDEX] = 20000.0f;
  cur->tau_R_m = 100.0f;
  cur->tau_R_o = 400.0f;
  cur->tauE = 20.0f;
  cur->NucModifyRatio[TNUCINDEX] = 1.1f;
  cur->NucModifyRatio[ANUCINDEX] = 1.1f;
  cur->NucModifyRatio[CNUCINDEX] = 1.1f;
  cur->NucModifyRatio[GNUCINDEX] = 1.1f;
  
  cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] = 3.1f;
  cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] = 3.1f;
  cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] = 3.1f;
  cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] = 3.1f;
  cur->nuc_shape.sigma_mult[TNUCINDEX] = 2.1f;
  cur->nuc_shape.sigma_mult[ANUCINDEX] = 2.1f;
  cur->nuc_shape.sigma_mult[CNUCINDEX] = 2.1f;
  cur->nuc_shape.sigma_mult[GNUCINDEX] = 2.1f;
  
  for (int i_nuc=0; i_nuc<NUMNUC; i_nuc++)
    cur->nuc_shape.C[i_nuc]    =  500.0f;
  
    cur->nuc_shape.valve_open = 15.0f; // frames(!)
    cur->nuc_shape.nuc_flow_span = 60.0f; // frames = 15.0f per second
    cur->nuc_shape.magic_divisor_for_timing = 20.7; // frames(!)
}

void reg_params_setStandardLow(reg_params *cur, float t0_start)
{
  // per-region parameters
  for (int j=0;j<NUMFB;j++)
  {
    cur->darkness[j] = 0.0f;
    cur->copy_multiplier[j] = 0.0f; // can drift very low
    cur->nuc_shape.t_mid_nuc[j]      = t0_start-6.0f; // really???
    cur->nuc_shape.t_mid_nuc_shift_per_flow[j] = 0.0f-3.0f;
  }

  cur->fit_taue = false;
  cur->tshift    = -1.5f;
  cur->nuc_shape.sigma  = 0.4f;
  cur->RatioDrift    = 0.0f;
  cur->CopyDrift    = 0.99f;

  cur->krate[TNUCINDEX] = 0.01f;
  cur->krate[ANUCINDEX] = 0.01f;
  cur->krate[CNUCINDEX] = 0.01f;
  cur->krate[GNUCINDEX] = 0.01f;
  cur->sens =  0.5f;

  cur->d[TNUCINDEX] =  0.1f;
  cur->d[ANUCINDEX] =  0.1f;
  cur->d[CNUCINDEX] =  0.1f;
  cur->d[GNUCINDEX] =  0.1f;

  cur->kmax[TNUCINDEX] = 5.0f;
  cur->kmax[ANUCINDEX] = 5.0f;
  cur->kmax[CNUCINDEX] = 5.0f;
  cur->kmax[GNUCINDEX] = 5.0f;
  cur->tau_R_m = -100.0f;
  cur->tau_R_o = -100.0f;
  cur->tauE = 1.0f;
  cur->NucModifyRatio[TNUCINDEX] = 0.9f;
  cur->NucModifyRatio[ANUCINDEX] = 0.9f;
  cur->NucModifyRatio[CNUCINDEX] = 0.9f;
  cur->NucModifyRatio[GNUCINDEX] = 0.9f;
  cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] = -3.0f;
  cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] = -3.0f;
  cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] = -3.0f;
  cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] = -3.0f;
  cur->nuc_shape.sigma_mult[TNUCINDEX] = 0.5f;
  cur->nuc_shape.sigma_mult[ANUCINDEX] = 0.5f;
  cur->nuc_shape.sigma_mult[CNUCINDEX] = 0.5f;
  cur->nuc_shape.sigma_mult[GNUCINDEX] = 0.5f;
  
  for (int i_nuc=0; i_nuc<NUMNUC; i_nuc++)
    cur->nuc_shape.C[i_nuc]    = 1.0f;
  
    cur->nuc_shape.valve_open = 15.0f; // frames(!)
    cur->nuc_shape.nuc_flow_span = 15.0f; // frames = 15.0f per second
    cur->nuc_shape.magic_divisor_for_timing = 20.7; // frames(!)
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
  cur->tauE = 5;
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

void reg_params_setStandardValue(reg_params *cur, float t_mid_nuc_start, float sigma_start, float *dntp_concentration_in_uM, bool _fit_taue)
{
  // per-region parameters
  for (int j=0;j<NUMFB;j++)
  {
    cur->darkness[j] = 0.0f;
    cur->copy_multiplier[j] = 1.0f;
    cur->nuc_shape.t_mid_nuc[j]      = t_mid_nuc_start;
    cur->nuc_shape.t_mid_nuc_shift_per_flow[j] = 0.0f;
  }

  cur->molecules_to_micromolar_conversion = 0.000062; // 3 micron wells


  cur->fit_taue = _fit_taue;
  cur->tshift = 0.4f;
  cur->nuc_shape.sigma  = sigma_start;
  cur->nuc_shape.nuc_flow_span = 22.5f;
  cur->CopyDrift    = 0.9987f;

  cur->RatioDrift    = 2.0f;


  for (int i_nuc=0; i_nuc<NUMNUC; i_nuc++)
    cur->nuc_shape.C[i_nuc]    =  dntp_concentration_in_uM[i_nuc];
  
    // defaults consistent w/ original v7 behavior
    cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] =  0.69f;
    cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] =  1.78f;
    cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] =  0.0f;
    cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] =  0.17f;

    cur->nuc_shape.sigma_mult[TNUCINDEX] = 1.162f;
    cur->nuc_shape.sigma_mult[ANUCINDEX] = 1.124f;
    cur->nuc_shape.sigma_mult[CNUCINDEX] = 1.0f;
    cur->nuc_shape.sigma_mult[GNUCINDEX] = 0.8533f;

    //@TODO: this is denominated in frames per second = 15
    //@TODO: please can we not do this operation at all
    cur->nuc_shape.valve_open = 15.0f; // frames(!)
    cur->nuc_shape.magic_divisor_for_timing = 20.7; // frames(!)

  cur->NucModifyRatio[TNUCINDEX] = 1.0f;
  cur->NucModifyRatio[ANUCINDEX] = 1.0f;
  cur->NucModifyRatio[CNUCINDEX] = 1.0f;
  cur->NucModifyRatio[GNUCINDEX] = 1.0f;
}

void DumpRegionParamsTitle(FILE *my_fp)
{
  fprintf(my_fp,"row\tcol\td[0]\td[1]\td[2]\td[3]\tkr[0]\tkr[1]\tkr[2]\tkr[3]\tkmax[0]\tkmax[1]\tkmax[2]\tkmax[3]\tt_mid_nuc\tt_mid_nuc[0]\tt_mid_nuc[1]\tt_mid_nuc[2]\tt_mid_nuc[3]\tsigma\tsigma[0]\tsigma[1]\tsigma[2]\tsigma[3]\tNucModifyRatio[0]\tNucModifyRatio[1]\tNucModifyRatio[2]\tNucModifyRatio[3]\ttau_m\ttau_o\ttauE\trdr\tpdr\ttshift");
  for (int i=0; i<NUMFB; i++)
    fprintf(my_fp,"\tt_mid_flow[%d]",i);
  fprintf(my_fp,"\n");
}



void DumpRegionParamsLine(FILE *my_fp,int my_row, int my_col, reg_params &rp)
{
  // officially the wrong way to do this
  fprintf(my_fp,"%4d\t%4d\t", my_row,my_col);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",rp.d[TNUCINDEX],rp.d[ANUCINDEX],rp.d[CNUCINDEX],rp.d[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",rp.krate[TNUCINDEX],rp.krate[ANUCINDEX],rp.krate[CNUCINDEX],rp.krate[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",rp.kmax[TNUCINDEX],rp.kmax[ANUCINDEX],rp.kmax[CNUCINDEX],rp.kmax[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t",rp.nuc_shape.t_mid_nuc[0]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",GetModifiedMidNucTime(&rp.nuc_shape,TNUCINDEX,0),GetModifiedMidNucTime(&rp.nuc_shape,ANUCINDEX,0),GetModifiedMidNucTime(&rp.nuc_shape,CNUCINDEX,0),GetModifiedMidNucTime(&rp.nuc_shape,GNUCINDEX,0));
  fprintf(my_fp,"%5.3f\t",rp.nuc_shape.sigma);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",GetModifiedSigma(&rp.nuc_shape,TNUCINDEX),GetModifiedSigma(&rp.nuc_shape,ANUCINDEX),GetModifiedSigma(&rp.nuc_shape,CNUCINDEX),GetModifiedSigma(&rp.nuc_shape,GNUCINDEX));
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t",rp.NucModifyRatio[TNUCINDEX],rp.NucModifyRatio[ANUCINDEX],rp.NucModifyRatio[CNUCINDEX],rp.NucModifyRatio[GNUCINDEX]);
  fprintf(my_fp,"%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t",rp.tau_R_m,rp.tau_R_o,rp.tauE,rp.RatioDrift,rp.CopyDrift,rp.tshift);
  for (int i=0; i<NUMFB; i++)
    fprintf(my_fp,"%5.3f\t",rp.nuc_shape.t_mid_nuc_shift_per_flow[i]);
  fprintf(my_fp,"\n");
}

float GetTypicalMidNucTime(nuc_rise_params *cur)
{
    return(cur->t_mid_nuc[0]);
}

void ResetPerFlowTimeShift(nuc_rise_params *cur)
{
  for (int fnum=0; fnum<NUMFB; fnum++)
    cur->t_mid_nuc_shift_per_flow[fnum] = 0.0f;
}

float GetModifiedMidNucTime(nuc_rise_params *cur, int NucID, int fnum)
{
  float retval_time = cur->t_mid_nuc[0];
  retval_time +=  cur->t_mid_nuc_delay[NucID]* (cur->t_mid_nuc[0]-cur->valve_open) /(cur->magic_divisor_for_timing+SAFETYZERO);
  retval_time += cur->t_mid_nuc_shift_per_flow[fnum];
  return(retval_time);
}

float GetModifiedSigma(nuc_rise_params *cur, int NucID)
{
    return(cur->sigma*cur->sigma_mult[NucID]);  // to make sure we knwo that this is modified
}

float CalculateCopyDrift(reg_params &rp, int absolute_flow)
{
    return pow (rp.CopyDrift,absolute_flow);
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
            reg_p->nuc_shape.sigma,reg_p->RatioDrift,reg_p->CopyDrift,reg_p->nuc_shape.t_mid_nuc[0],reg_p->tshift,reg_p->krate[0],reg_p->krate[1],reg_p->krate[2],reg_p->krate[3],reg_p->d[0],reg_p->d[1],reg_p->d[2],reg_p->d[3]);
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
  for ( int i=0; i<NUMFB; i++ )
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
  for ( int i=0; i<NUMFB; i++ )
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

