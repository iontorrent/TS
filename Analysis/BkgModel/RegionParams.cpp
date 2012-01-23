/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RegionParams.h"

void reg_params_ApplyUpperBound(reg_params *cur, reg_params *bound)
{
  MAX_BOUND_CHECK(nuc_shape.t_mid_nuc);
  
  for (int i=0;i<NUMFB;i++)
  {
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
}

void reg_params_ApplyLowerBound(reg_params *cur, reg_params *bound)
{
    MIN_BOUND_CHECK(nuc_shape.t_mid_nuc);
  for (int i=0;i<NUMFB;i++)
  {
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
}

float xComputeTauBfromEmptyUsingRegionLinearModel(float tau_R_m,float tau_R_o, float etbR)
{
  float tauB = (tau_R_m*etbR+tau_R_o);
  if (tauB < MINTAUB) tauB = MINTAUB;
  if (tauB > MAXTAUB) tauB = MAXTAUB;
  return (tauB);
}

float ComputeTauBfromEmptyUsingRegionLinearModel(reg_params *reg_p, float etbR)
{
    return(xComputeTauBfromEmptyUsingRegionLinearModel(reg_p->tau_R_m,reg_p->tau_R_o,etbR));
}

float xAdjustEmptyToBeadRatioForFlow(float etbR_original, float NucModifyRatio, float RatioDrift, int flow)
{
  float ModifiedRatio, TimeAdjust, etbR;
  ModifiedRatio = etbR_original*NucModifyRatio;
  TimeAdjust = RatioDrift*flow/SCALEOFBUFFERINGCHANGE;
  etbR = ModifiedRatio + (1.0-ModifiedRatio) * TimeAdjust;   // smooth adjustment towards being as fast as an empty
  return (etbR);
}

float AdjustEmptyToBeadRatioForFlow(float etbR_original, reg_params *reg_p, int nuc_id, int flow)
{
  return(xAdjustEmptyToBeadRatioForFlow(etbR_original,reg_p->NucModifyRatio[nuc_id],reg_p->RatioDrift,flow));
}


void reg_params_setStandardHigh(reg_params *cur, float t0_start)
{
  // per-region parameters
  for (int j=0;j<NUMFB;j++)
  {
    cur->darkness[j] = 2.0;
    cur->copy_multiplier[j] = 1.0;
  }


    cur->nuc_shape.t_mid_nuc      = t0_start+6; // really????


  cur->tshift    = 6.5;
  cur->nuc_shape.sigma = 5.5;
  cur->RatioDrift = 5.0;
  cur->CopyDrift = 1.0;

  cur->krate[TNUCINDEX] = 100.0;
  cur->krate[ANUCINDEX] = 100.0;
  cur->krate[CNUCINDEX] = 100.0;
  cur->krate[GNUCINDEX] = 100.0;
  cur->sens = 250.0;       // counts per 10K protons generated

  cur->d[TNUCINDEX] =  50000.0;
  cur->d[ANUCINDEX] =  50000.0;
  cur->d[CNUCINDEX] =  50000.0;
  cur->d[GNUCINDEX] =  50000.0;

  cur->kmax[TNUCINDEX] = 20000.0;
  cur->kmax[ANUCINDEX] = 20000.0;
  cur->kmax[CNUCINDEX] = 20000.0;
  cur->kmax[GNUCINDEX] = 20000.0;
  cur->tau_R_m = 100.0;
  cur->tau_R_o = 400.0;
  cur->NucModifyRatio[TNUCINDEX] = 1.1;
  cur->NucModifyRatio[ANUCINDEX] = 1.1;
  cur->NucModifyRatio[CNUCINDEX] = 1.1;
  cur->NucModifyRatio[GNUCINDEX] = 1.1;
  
  cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] = 3.1;
  cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] = 3.1;
  cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] = 3.1;
  cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] = 3.1;
  cur->nuc_shape.sigma_mult[TNUCINDEX] = 2.1;
  cur->nuc_shape.sigma_mult[ANUCINDEX] = 2.1;
  cur->nuc_shape.sigma_mult[CNUCINDEX] = 2.1;
  cur->nuc_shape.sigma_mult[GNUCINDEX] = 2.1;
  cur->nuc_shape.C    =  500.0;
}

void reg_params_setStandardLow(reg_params *cur, float t0_start)
{
  // per-region parameters
  for (int j=0;j<NUMFB;j++)
  {
    cur->darkness[j] = 0.0;
    cur->copy_multiplier[j] = 0.0; // can drift very low
  }


    cur->nuc_shape.t_mid_nuc      = t0_start-6; // really???


  cur->tshift    = 1.5;
  cur->nuc_shape.sigma  = 0.4;
  cur->RatioDrift    = 0.0;
  cur->CopyDrift    = 0.99;

  cur->krate[TNUCINDEX] = 0.01;
  cur->krate[ANUCINDEX] = 0.01;
  cur->krate[CNUCINDEX] = 0.01;
  cur->krate[GNUCINDEX] = 0.01;
  cur->sens =  0.5;

  cur->d[TNUCINDEX] =  0.1;
  cur->d[ANUCINDEX] =  0.1;
  cur->d[CNUCINDEX] =  0.1;
  cur->d[GNUCINDEX] =  0.1;

  cur->kmax[TNUCINDEX] = 5.0;
  cur->kmax[ANUCINDEX] = 5.0;
  cur->kmax[CNUCINDEX] = 5.0;
  cur->kmax[GNUCINDEX] = 5.0;
  cur->tau_R_m = -100.0;
  cur->tau_R_o = -100.0;
  cur->NucModifyRatio[TNUCINDEX] = 0.9;
  cur->NucModifyRatio[ANUCINDEX] = 0.9;
  cur->NucModifyRatio[CNUCINDEX] = 0.9;
  cur->NucModifyRatio[GNUCINDEX] = 0.9;
  cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] = -3;
  cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] = -3;
  cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] = -3;
  cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] = -3;
  cur->nuc_shape.sigma_mult[TNUCINDEX] = 0.5;
  cur->nuc_shape.sigma_mult[ANUCINDEX] = 0.5;
  cur->nuc_shape.sigma_mult[CNUCINDEX] = 0.5;
  cur->nuc_shape.sigma_mult[GNUCINDEX] = 0.5;
  cur->nuc_shape.C    = 1;
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

void reg_params_setSens(reg_params *cur, float sens_default)
{
  cur->sens = sens_default;

}

void reg_params_setBuffModel(reg_params *cur, float tau_R_m_default, float tau_R_o_default)
{
  cur->tau_R_m = tau_R_m_default;
  cur->tau_R_o = tau_R_o_default;
}

void reg_params_setNoRatioDriftValues(reg_params *cur)
{
    cur->RatioDrift    = 1.3;

    // in the no-RatioDrift fit version, we added fitting these parameters
    // in this case, they have unbiased defaults
    cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] =  0.0;
    cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] =  0.0;
    cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] =  0.0;
    cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] =  0.0;

    cur->nuc_shape.sigma_mult[TNUCINDEX] = 1.0;
    cur->nuc_shape.sigma_mult[ANUCINDEX] = 1.0;
    cur->nuc_shape.sigma_mult[CNUCINDEX] = 1.0;
    cur->nuc_shape.sigma_mult[GNUCINDEX] = 1.0;
}

void reg_params_setStandardValue(reg_params *cur, float t_mid_nuc_start, float sigma_start, float dntp_concentration_in_uM)
{
  // per-region parameters
  for (int j=0;j<NUMFB;j++)
  {
    cur->darkness[j] = 0.0;
    cur->copy_multiplier[j] = 1.0;
  }


    cur->nuc_shape.t_mid_nuc      = t_mid_nuc_start;

  cur->tshift = 3.4;
  cur->nuc_shape.sigma  = sigma_start;
  cur->CopyDrift    = 0.9987;

    cur->RatioDrift    = 2.0;


  cur->nuc_shape.C    =  dntp_concentration_in_uM;
    // defaults consistent w/ original v7 behavior
    cur->nuc_shape.t_mid_nuc_delay[TNUCINDEX] =  0.69;
    cur->nuc_shape.t_mid_nuc_delay[ANUCINDEX] =  1.78;
    cur->nuc_shape.t_mid_nuc_delay[CNUCINDEX] =  0.0;
    cur->nuc_shape.t_mid_nuc_delay[GNUCINDEX] =  0.17;

    cur->nuc_shape.sigma_mult[TNUCINDEX] = 1.162;
    cur->nuc_shape.sigma_mult[ANUCINDEX] = 1.124;
    cur->nuc_shape.sigma_mult[CNUCINDEX] = 1.0;
    cur->nuc_shape.sigma_mult[GNUCINDEX] = 0.8533;

  cur->NucModifyRatio[TNUCINDEX] = 1.0;
  cur->NucModifyRatio[ANUCINDEX] = 1.0;
  cur->NucModifyRatio[CNUCINDEX] = 1.0;
  cur->NucModifyRatio[GNUCINDEX] = 1.0;
}

void DumpRegionParamsTitle(FILE *my_fp)
{
  fprintf(my_fp,"row\tcol\td[0]\td[1]\td[2]\td[3]\tkr[0]\tkr[1]\tkr[2]\tkr[3]\tkmax[0]\tkmax[1]\tkmax[2]\tkmax[3]\tt_mid_nuc\tt_mid_nuc[0]\tt_mid_nuc[1]\tt_mid_nuc[2]\tt_mid_nuc[3]\tsigma\tsigma[0]\tsigma[1]\tsigma[2]\tsigma[3]\tNucModifyRatio[0]\tNucModifyRatio[1]\tNucModifyRatio[2]\tNucModifyRatio[3]\ttau_m\ttau_o\trdr\tpdr\ttshift\n");
}

/*void DumpRegionParamsLine(FILE *my_fp,int my_row, int my_col, reg_params &rp)
{
      fprintf(my_fp,"% 4d\t% 4d\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n",
          my_row,my_col,
          rp.d[0],rp.d[1],rp.d[2],rp.d[3],
          rp.krate[0],rp.krate[1],rp.krate[2],rp.krate[3],
          rp.nuc_shape.t_mid_nuc,rp.nuc_shape.sigma,rp.tau_R_m,rp.tau_R_o,rp.RatioDrift,rp.CopyDrift);  
}*/

void DumpRegionParamsLine(FILE *my_fp,int my_row, int my_col, reg_params &rp)
{
      fprintf(my_fp,"% 4d\t% 4d\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n",
          my_row,my_col,
          rp.d[TNUCINDEX],rp.d[ANUCINDEX],rp.d[CNUCINDEX],rp.d[GNUCINDEX],
          rp.krate[TNUCINDEX],rp.krate[ANUCINDEX],rp.krate[CNUCINDEX],rp.krate[GNUCINDEX],
          rp.kmax[TNUCINDEX],rp.kmax[ANUCINDEX],rp.kmax[CNUCINDEX],rp.kmax[GNUCINDEX],
          rp.nuc_shape.t_mid_nuc, 
              GetModifiedMidNucTime(&rp.nuc_shape,TNUCINDEX),GetModifiedMidNucTime(&rp.nuc_shape,ANUCINDEX),GetModifiedMidNucTime(&rp.nuc_shape,CNUCINDEX),GetModifiedMidNucTime(&rp.nuc_shape,GNUCINDEX),
              rp.nuc_shape.sigma,
              GetModifiedSigma(&rp.nuc_shape,TNUCINDEX),GetModifiedSigma(&rp.nuc_shape,ANUCINDEX),GetModifiedSigma(&rp.nuc_shape,CNUCINDEX),GetModifiedSigma(&rp.nuc_shape,GNUCINDEX),
              rp.NucModifyRatio[TNUCINDEX],rp.NucModifyRatio[ANUCINDEX],rp.NucModifyRatio[CNUCINDEX],rp.NucModifyRatio[GNUCINDEX], 
              rp.tau_R_m,rp.tau_R_o,rp.RatioDrift,rp.CopyDrift,rp.tshift);  
}


float GetModifiedMidNucTime(nuc_rise_params *cur, int NucID)
{
  return(cur->t_mid_nuc + cur->t_mid_nuc_delay[NucID]* (cur->t_mid_nuc-VALVEOPENFRAME) /TZERODELAYMAGICSCALE);
}

float GetModifiedSigma(nuc_rise_params *cur, int NucID)
{
    return(cur->sigma*cur->sigma_mult[NucID]);  // to make sure we knwo that this is modified
}

void ResetRegionBeadShiftsToZero(reg_params *reg_p)
{
        memset(reg_p->Ampl,0,sizeof(reg_p->Ampl));
        reg_p->R = 0.0;
        reg_p->Copies = 0.0;
}

void DumpRegionParamsCSV(FILE *my_fp, reg_params *reg_p)
{
    fprintf(my_fp,"%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,%10.5f,",
            reg_p->nuc_shape.sigma,reg_p->RatioDrift,reg_p->CopyDrift,reg_p->nuc_shape.t_mid_nuc,reg_p->tshift,reg_p->krate[0],reg_p->krate[1],reg_p->krate[2],reg_p->krate[3],reg_p->d[0],reg_p->d[1],reg_p->d[2],reg_p->d[3]);
}