/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TraceCurry.h"

void TraceCurry::SingleFlowIncorporationTrace (float A,float *fval)
{
    // temporary
    //p->my_state.hits_by_flow[fnum]++;
    eval_count++;
    RedTrace (fval, ivalPtr, npts, deltaFrameSeconds, deltaFrame, c_dntp_top_pc, sub_steps, i_start, C, A, SP, kr, kmax, d, molecules_to_micromolar_conversion, sens, gain, tauB, math_poiss);
}

void TraceCurry::SingleFlowIncorporationTrace (float A,float kmult,float *fval)
{
    //p->my_state.hits_by_flow[fnum]++;
    float tkr = region_kr*kmult; // technically would be a side effect if I reset kr.
    eval_count++;
    RedTrace (fval, ivalPtr, npts, deltaFrameSeconds, deltaFrame, c_dntp_top_pc, sub_steps, i_start, C, A, SP, tkr, kmax, d, molecules_to_micromolar_conversion, sens, gain, tauB, math_poiss);
}

void TraceCurry::IntegrateRedObserved(float *red, float *red_obs)
{
    IntegrateRedFromRedTraceObserved (red,red_obs, npts, i_start, deltaFrame, tauB);
}

float TraceCurry::GuessAmplitude(float *red_obs)
{
    float red_guess[npts];
    IntegrateRedObserved(red_guess,red_obs);
    float offset = SP*sens;
    int test_pts = i_start+reg_p->nuc_shape.nuc_flow_span*0.75;  // nuc arrives at i_start, halfway through is a good guess for when we're done
    if (test_pts>npts-1)
        test_pts = npts-1;
    float a_guess = (red_guess[test_pts]-red_guess[i_start])/offset;  // take the difference to normalize out anything happening before incorporation
  return(a_guess);
}

void TraceCurry::SetWellRegionParams (struct bead_params *_p,struct reg_params *_rp,int _fnum,
                                      int _nnum,int _flow,
                                      int _i_start,float *_c_dntp_top)
{
    p = _p;
    reg_p = _rp;
    fnum = _fnum;
    NucID = _nnum;
    flow = _flow;


    // since this uses a library function..and the parameters involved aren't fit
    // it's helpful to compute this once and not in the model function
    SP = (float) (COPYMULTIPLIER * p->Copies) *pow (reg_p->CopyDrift,flow);

    etbR = AdjustEmptyToBeadRatioForFlow (p->R,reg_p,NucID,flow);
    tauB = ComputeTauBfromEmptyUsingRegionLinearModel (reg_p,etbR);

    sens = reg_p->sens*SENSMULTIPLIER;
    molecules_to_micromolar_conversion = reg_p->molecules_to_micromolar_conversion;
    d = reg_p->d[NucID]*p->dmult;
    kr = reg_p->krate[NucID]*p->kmult[fnum];
    region_kr = reg_p->krate[NucID];
    kmax = reg_p->kmax[NucID];
    C = reg_p->nuc_shape.C[NucID]; // setting a single flow
    gain = p->gain;
    // it is now necessary to have these precomputed somewhere else
    i_start = _i_start;
    c_dntp_top_pc = _c_dntp_top;
}

// what we're >really< setting
void TraceCurry::SetContextParams(int _i_start, float *c_dntp_top, int _sub_steps, float _C, float _SP, float _region_kr, float _kmax, float _d, float _sens, float _gain, float _tauB)
{
    i_start = _i_start;
    c_dntp_top_pc = c_dntp_top;
    // assume I directly know the parameters and don't have to recompute them
    C = _C;
    SP = _SP;
    region_kr = _region_kr;
    kr = region_kr;
    kmax = _kmax;
    d = _d;
    sens = _sens;
    gain = _gain;
    tauB = _tauB;
    sub_steps = _sub_steps;
}

TraceCurry::TraceCurry()
{
    ivalPtr = NULL;
    npts = 0;
    deltaFrame = NULL;
    deltaFrameSeconds = NULL;
    c_dntp_top_pc = NULL;
    math_poiss = NULL;
    p = NULL;
    reg_p = NULL;
    fnum = 0;
    NucID = 0;
    flow = 0;
    SP = 0.0f;
    etbR = 0.0f;
    tauB = 0.0f;
    sens = 0.0f;
    molecules_to_micromolar_conversion = 0.0f;
    d = 0.0f;
    kr = 0.0f;
    region_kr = 0.0f;
    kmax = 0.0f;
    C = 0.0f;
    i_start = 0;
    gain = 0.0f;
    eval_count = 0;
    sub_steps = ISIG_SUB_STEPS_SINGLE_FLOW;
}

// constructor
void   TraceCurry::Allocate (int _len,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss)
{
    ivalPtr = new float[_len];
    npts = _len;
    deltaFrame = _deltaFrame;
    deltaFrameSeconds = _deltaFrameSeconds;
    c_dntp_top_pc = NULL;
    math_poiss = _math_poiss;
}

TraceCurry::~TraceCurry()
{
    if (ivalPtr!=NULL)
        delete[] ivalPtr;
}
