/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DNTPRiseModel.h"

DntpRiseModel::DntpRiseModel(int _npts,float _C,float *_tvals,int _sub_steps)
{
    npts = _npts;
    tvals = _tvals;
    C = _C;
    i_start = 0;
    sub_steps = _sub_steps;
}




// spline with one knot
int SplineRiseFunction(float *output, int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc, float sigma, float tangent_zero, float tangent_one)
{
    int ndx = 0;
    float tlast = 0;
    float last_nuc_value = 0.0;
    int i_start;
    float scaled_dt = -1.0;
    float my_sigma = 3*sigma; // bring back into range for ERF

    i_start = -1;

    memset(output,0,sizeof(float[npts*sub_steps]));
    
    for (int i=0;(i < npts) && (scaled_dt<1);i++)
    {
        // get the frame number of this data point (might be fractional because this point could be
        // the average of several frames of data.  This number is the average time of all the averaged
        // data points
        float t=frame_times[i];

        for (int st=1;st <= sub_steps;st++)
        {
            float tnew = tlast+(t-tlast)*(float)st/sub_steps;
            scaled_dt = (tnew-t_mid_nuc)/my_sigma +0.5;

            if ((scaled_dt>0))
            {
              float scaled_dt_square = scaled_dt*scaled_dt;
              float scaled_dt_minus = scaled_dt-1;
              last_nuc_value = scaled_dt_square*(3-2*scaled_dt); //spline! with zero tangents at start and end points
              last_nuc_value += scaled_dt_square*scaled_dt_minus * tangent_one; // finishing tangent, tangent at zero = 0
              last_nuc_value += scaled_dt*scaled_dt_minus*scaled_dt_minus * tangent_zero; // tangent at start, tangent at 1 = 0
              
              // scale up to finish at C
              last_nuc_value *= C;
              if (scaled_dt>1)
                last_nuc_value = C;
            }
            output[ndx++] = last_nuc_value;
        }

        if ((i_start == -1) && (scaled_dt>0))
            i_start = i;

        tlast = t;
    }
    // if needed, can do a spline decrease to handle wash-off at end, but generally we're done with the reaction by then
    // so may be fancier than we need

    for (;ndx < sub_steps*npts;ndx++)
        output[ndx] = C;

    return(i_start);
}


int SigmaXRiseFunction(float *output,int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc,float sigma)
{
    int ndx = 0;
    float tlast = 0;
    float last_nuc_value = 0.0;
    int i_start;

    i_start = -1;

    memset(output,0,sizeof(float[npts*sub_steps]));
    for (int i=0;(i < npts) && (last_nuc_value < 0.999*C);i++)
    {
        // get the frame number of this data point (might be fractional because this point could be
        // the average of several frames of data.  This number is the average time of all the averaged
        // data points
        float t=frame_times[i];

        for (int st=1;st <= sub_steps;st++)
        {
            float tnew = tlast+(t-tlast)*(float)st/sub_steps;
            float erfdt = (tnew-t_mid_nuc)/sigma;

            if (erfdt >= -3)
                last_nuc_value = C*(1.0+ErfApprox(erfdt))/2.0;

            output[ndx++] = last_nuc_value;
        }

        if ((i_start == -1) && (last_nuc_value >= MIN_PROC_THRESHOLD))
            i_start = i;

        tlast = t;
    }

    for (;ndx < sub_steps*npts;ndx++)
        output[ndx] = C;

    return(i_start);
}


int SigmaRiseFunction(float *output,int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc,float sigma)
{
  if (true)
    return(SplineRiseFunction(output,npts,frame_times,sub_steps,C,t_mid_nuc,sigma,0,0));
  else
    return(SigmaXRiseFunction(output,npts,frame_times,sub_steps,C,t_mid_nuc,sigma));
}

#define CORRECTED_SIGMA_RISE_FUNC
#ifdef CORRECTED_SIGMA_RISE_FUNC
int DntpRiseModel::CalcCDntpTop(float *output,float t_mid_nuc,float sigma)
{
  //  why we have an object here is beyond me as we generate then destroy immediately without real shared state
    return(SigmaRiseFunction(output,npts,tvals,sub_steps,C, t_mid_nuc, sigma));
}

#else

// newer [dNTP] rise model empirically derived from measured data
int DntpRiseModel::CalcCDntpTop(float *output,float t_mid_nuc,float tau_mult)
{
    float tau,sigma;
    float s1,s2;
    (void)tau_mult;

    // compute tau
    tau = (((0.0009*t_mid_nuc-0.0887)*t_mid_nuc+3.2416)*t_mid_nuc-36.0491)*tau_mult;
    if (tau < 0.01)
        tau = 0.01;

    // compute sigma
    s1 = 0.7*tau+27.8161;
    s2 = 7.1733*tau+14.8696;

    if (s2 > s1)
        sigma = s1;
    else
        sigma = s2;

    sigma = sigma;

    int ndx = 0;
    float tlast = 0;

    i_start = -1;

    float isum1=0.0;
    float isum2=0.0;
    float last_nuc_value = 0.0;

    for (int i=0;(i < npts) && (last_nuc_value < 0.999*C);i++)
    {
        // get the frame number of this data point (might be fractional because this point could be
        // the average of several frames of data.  This number is the average time of all the averaged
        // data points
        float t=tvals[i];

        // the model produces the correct shape only if it is computed w/ 2 sub-steps
        // if what is needed is a single value per fitted data point, then we down-sample the 
        // output
        for (int st=1;st <= 2;st++)
        {
            float tnew = tlast+(t-tlast)*(double)st/2;
            float dt = tnew-tlast;

            float csigma = C*(1+ErfApprox((float)((tnew-t_mid_nuc)/(sigma/15))))/2.0;
            float fv1 = (isum1 + csigma*dt/tau)/(1+dt/tau);
            float fv2 = (isum2 + fv1*dt/tau)/(1+dt/tau);
            isum1=isum1+(csigma-fv1)*dt/tau;
            isum2=isum2+(fv1-fv2)*dt/tau;

            if ((st == 1) || (sub_steps == 2))
            {   
                last_nuc_value = fv2; 
                output[ndx++] = fv2;
            }
        }

        if ((i_start == -1) && (last_nuc_value >= MIN_PROC_THRESHOLD))
            i_start = i;

        // this isn't right!..in the wrong place, but I need to re-calibrate the model
        // to change it
        tlast = t;
    }

    for (;ndx < sub_steps*npts;ndx++)
        output[ndx] = C;

    return(i_start);
}

#endif

