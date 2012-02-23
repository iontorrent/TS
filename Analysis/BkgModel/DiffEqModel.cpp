/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "DiffEqModel.h"
#include "math.h"
#include <algorithm>

// ripped out to separate the math from the optimization procedure


// incorporation traces not yet quite right

// Compute the incorporation "red" hydrogen trace only without any "blue" hydrogen from the bulk or cross-talk
// does not adjust for gain?
void RedSolveHydrogenFlowInWell(float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB)
{
    float dv = 0.0;
    float dvn = 0.0;
    float dv_rs = 0.0;
    float scaledTau = 2.0*tauB;
    memset(vb_out,0,sizeof(float[i_start]));
    for (int i=i_start; i<len; i++)
    {
        // calculate the 'background' part (the accumulation/decay of the protons in the well
        // normally accounted for by the background calc)
        float dt = deltaFrame[i];
        float aval = dt/scaledTau;

        // calculate new dv
        dvn = (red_hydrogen[i] - dv_rs/tauB - dv*aval) / (1.0+aval);
        dv_rs += (dv+dvn)*dt/2.0;
        vb_out[i] = dv = dvn;
    }
}

// generates the background trace for a well given the "empty" well measurement of blue_hydrogen ions
void BlueSolveBackgroundTrace(float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
    float dv,dvn,dv_rs;
    float aval = 0;
    float dt;
    float shift_ratio = etbR-1.0;

    dv = 0.0;
    dv_rs = 0.0;
    dt = -1.0;
    dvn = 0.0;

    for (int i=0;i < len;i++)
    {
        dt = deltaFrame[i];
        aval = dt/(2.0*tauB);

        // calculate new dv
        dvn = (shift_ratio*blue_hydrogen[i] - dv_rs/tauB - dv*aval) / (1.0+aval);
        dv_rs += (dv+dvn)*dt/2.0;
        dv = dvn;

        vb_out[i] = (dv+blue_hydrogen[i]);
    }
}

// even although this is simply the sum of red and blue hydrogen ions, we'll compute it here because we think it is mildly more efficient
void PurpleSolveTotalTrace(float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
    float dv,dvn,dv_rs;
    float aval = 0;
    int   i;
    float dt;
    float shift_ratio = etbR-1.0;

    dv = 0.0;
    dv_rs = 0.0;
    dt = -1.0;
    dvn = 0.0;
    for (i=0;i < len;i++)
    {
        dt = deltaFrame[i];
        aval = dt/ (2.0*tauB);

        dvn = (red_hydrogen[i] + shift_ratio *blue_hydrogen[i] - dv_rs/tauB - dv*aval) / (1.0+aval);
        dv_rs += (dv+dvn) *dt/2.0;
        dv = dvn;

        vb_out[i] = (dv+blue_hydrogen[i]);
    }
}
