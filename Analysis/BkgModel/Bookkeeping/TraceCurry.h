/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACECURRY_H
#define TRACECURRY_H

#include "Stats.h"
#include "BeadParams.h"
#include "RegionParams.h"
#include "Region.h"
#include "MathOptim.h"
#include "DiffEqModel.h"
// this object will "curry" a compute trace command by holding parameters that aren't being actively changed

class TraceCurry
{
  public:
    // pointers to the BkgModel parameters for the well/region being fit
    struct BeadParams *p;
    struct reg_params *reg_p;
    // indicates which flow we are fitting (used to set-up flow-specific parameter values)
    // this is the 'local' flow number (0-numfb)
    int fnum;
    // which nucleotide was in the flow
    int NucID;
    // global flow number (0-total number of flows in the run) ...needed for droop terms
    int flow;


    // single-flow delta-T between data points
    float *deltaFrame;
    float *deltaFrameSeconds;


    // pre-computed c_dntp_top...speedup
    float *c_dntp_top_pc;
    // pre-computed values
    float SP,etbR,tauB,sens,molecules_to_micromolar_conversion;
    int i_start;
    int sub_steps; // how finely divided is our timing
    int npts;

    float d;
    float kr,kmax;
    float C;
    float region_kr;
    float gain;

    int eval_count; // monitor number of times I call

    PoissonCDFApproxMemo *math_poiss;
    TraceCurry();
    void Allocate(int _len,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss);
    ~TraceCurry();
    void SingleFlowIncorporationTrace (float A,float *fval);
    void SingleFlowIncorporationTrace (float A,float kmult,float *fval);
    void SetWellRegionParams (struct BeadParams *_p,struct reg_params *_rp,int _fnum,
                              int _nnum,int _flow,
                              int _i_start,float *_c_dntp_top);
    float log_slope(float *trc_smooth,int start,int end,float ymin);
    float find_slope_logy_seg6(float *trc_smooth, int deltax=5);
    float find_slope_logy_min(float *trc_smooth, int deltax=5);
    float find_slope_logy_max(float *trc_smooth, int deltax=5);
    float find_slope_logy_valley(float *trc_smooth, int deltax=5);
    float find_slope_logy_regress(float *trc_smooth, int deltax=5);
    void GuessLogTaub(float *red_obs, BeadParams *p, int fnum, reg_params *_rp);
    bool validTaub(float taub, float minTaub, float maxTaub) {return (((taub>=minTaub) && (taub<=maxTaub)) ? true:false);}
    float boundTaub(float t,float minTaub,float maxTaub);
    float boundTaub_mid(float t,float minTaub,float maxTaub,float midTaub);
    float boundTaub_nuc(float t,float minTaub,float maxTaub,float default_tauB_bead,float default_tauB_region);
    void smooth_kern(float *out, float *in, int npts);
    void savitzky_golay(float *out, float *in, int npts);
    void set_peak(float *trace,int npts,int deltax=5);
    int minIndex(float *trace, int npts);
    int maxIndex(float *trace, int npts);  // findpeak
    int minIndex_local(float *in, int npts);
    float slope_logy(float *trace,int slen);
    double regression_slope(vector<float>& Y, vector<float>& X);
    void  SetContextParams(int _i_start, float *c_dntp_top, int _sub_steps, float _C,
                           float _SP, float _region_kr, float _kmax, float _d, float _sens, float _gain, float _tauB);
    float GetStartAmplitude(){ return(p->Ampl[fnum]);};
    float GuessAmplitude(float *red_obs);
    float GetStartKmult(){return(p->kmult[fnum]);};
    void IntegrateRedObserved(float *red, float *red_obs);
    void ErrorSignal(float *obs,float *fit, float *posptr, float *negptr);
    void ResetEval(){eval_count = 0;};
    int GetEvalCount(){return(eval_count);};
  private:
     // single-flow proton flux
    // don't assume constant, can have side effects
    float *ivalPtr;   
    //data to be passed to trace.h5
    int peak,valley,top,bot;
    int firstx;
    int lastx; // last uncompressed timeframe
    float ymin;

};

#endif // TRACECURRY_H
