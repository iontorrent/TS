/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACECURRY_H
#define TRACECURRY_H

#include "BeadParams.h"
#include "RegionParams.h"
#include "MathOptim.h"
#include "DiffEqModel.h"
// this object will "curry" a compute trace command by holding parameters that aren't being actively changed

class TraceCurry
{
  public:
    // pointers to the BkgModel parameters for the well/region being fit
    struct bead_params *p;
    struct reg_params *reg_p;
    // indicates which flow we are fitting (used to set-up flow-specific parameter values)
    // this is the 'local' flow number (0-NUMFB)
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
    void SetWellRegionParams (struct bead_params *_p,struct reg_params *_rp,int _fnum,
                              int _nnum,int _flow,
                              int _i_start,float *_c_dntp_top);
    void  SetContextParams(int _i_start, float *c_dntp_top, int _sub_steps, float _C,
                           float _SP, float _region_kr, float _kmax, float _d, float _sens, float _gain, float _tauB);
    float GetStartAmplitude(){ return(p->Ampl[fnum]);};
    float GetStartKmult(){return(p->kmult[fnum]);};
    void IntegrateRedObserved(float *red, float *red_obs);
    void ResetEval(){eval_count = 0;};
    int GetEvalCount(){return(eval_count);};
  private:
     // single-flow proton flux
    // don't assume constant, can have side effects
    float *ivalPtr;   
};

#endif // TRACECURRY_H
