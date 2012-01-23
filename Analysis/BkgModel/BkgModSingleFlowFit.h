/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODSINGLEFLOWFIT_H
#define BKGMODSINGLEFLOWFIT_H

#include "LevMarFitterV2.h"
#include "MathOptim.h"
#include "DiffEqModel.h"

#define BKG_CORRECT_BEFORE_FIT
#define n_to_uM_conv    (0.000062f)

struct BkgModSingleFlowFitParams {
    float Ampl;
};

class BkgModSingleFlowFit : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    BkgModSingleFlowFit(int _len,float *frameNumber,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss)
    {
        ivalPtr = new float[_len];
        xvals = frameNumber;
        deltaFrame = _deltaFrame;
        deltaFrameSeconds = _deltaFrameSeconds;
        c_dntp_top_pc = NULL;
        math_poiss = _math_poiss;
        Initialize(1,_len,xvals);
    }
    
    // useful for evaluating sum-of-squares difference without invoking the full Lev-Mar
    void SetJustAmplitude(float A){
      params.Ampl=A;
    }

    // passes in params and reg_params pointers for the well/region we are fitting
    // also indicates which flow we are fitting
    // !!!!! This function damages the _fg structure it is passed!!!!!!  It is not safe!!!!
    void SetWellRegionParams(struct bead_params *_p,struct reg_params *_rp,int _fnum,
                             int _nnum,int _flow,
                             int _i_start,float *_c_dntp_top)
    {
        p = _p;
        reg_p = _rp;
        params.Ampl = p->Ampl[_fnum];
        fnum = _fnum;
        NucID = _nnum;
        flow = _flow;


        // since this uses a library function..and the parameters involved aren't fit
        // it's helpful to compute this once and not in the model function
        SP =  (float)(COPYMULTIPLIER * p->Copies)*pow(reg_p->CopyDrift,flow);

        etbR = AdjustEmptyToBeadRatioForFlow(p->R,reg_p,NucID,flow);
        tauB = ComputeTauBfromEmptyUsingRegionLinearModel(reg_p,etbR);

        sens = reg_p->sens*SENSMULTIPLIER;

        // it is now necessary to make a separate call to
        i_start = _i_start;
        c_dntp_top_pc = _c_dntp_top;
    }
    


    // optionally set maximum value for parameters
    void SetParamMax(BkgModSingleFlowFitParams _max_params)
    {
        max_params = _max_params;
        LevMarFitterV2::SetParamMax((float *)&max_params);
    }

    // optionally set minimum value for parameters
    void SetParamMin(BkgModSingleFlowFitParams _min_params)
    {
        min_params = _min_params;
        LevMarFitterV2::SetParamMin((float *)&min_params);
    }

    // entry point for grid search
    void GridSearch(int steps,float *y)
    {
        LevMarFitterV2::GridSearch(steps,y,(float *)(&params));
    }

    // entry point for fitting
    virtual int Fit(int max_iter,float *y)
    {
        return(LevMarFitterV2::Fit(max_iter,y,(float *)(&params)));
    }

    // the starting point and end point of the fit
    BkgModSingleFlowFitParams params;

    ~BkgModSingleFlowFit()
    {
        delete [] ivalPtr;
    }

    // evaluates the function using the values in params
    virtual void Evaluate(float *y) {
        Evaluate(y,(float *)(&params));
    }

    // evaluates the function using the values in params
    void TestFval(float *y,int fnum) {
        Evaluate(y,(float *)&(p->Ampl[fnum]));
    }
    PoissonCDFApproxMemo *math_poiss;

protected:
    virtual void Evaluate(float *y, float *params) {
        SingleFlowIncorporationTrace(*params,y);
    }

    // loop exit condition
    virtual bool DoneTest(int iter,int max_iter,LaVectorDouble *delta,float lambda,int &done_cnt,float residual,float r_chg)
    {
        (void)done_cnt;
        (void)residual;
        (void)r_chg;
        (void)lambda;

        if ((*delta)(0)*(*delta)(0) < 0.0000025) done_cnt++;
        else done_cnt = 0;

        return((iter >= max_iter) || (done_cnt > 1));
    }

private:

    // pointers to the BkgModel parameters for the well/region being fit
    struct bead_params *p;
    struct reg_params *reg_p;
    
    // single-flow proton flux    
    float *ivalPtr;
    // single-flow delta-T between data points
    float *deltaFrame;
    float *deltaFrameSeconds;

    // indicates which flow we are fitting (used to set-up flow-specific parameter values)
    // this is the 'local' flow number (0-NUMFB)
    int fnum;
    // which nucleotide was in the flow
    int NucID;
    // global flow number (0-total number of flows in the run) ...needed for droop terms
    int flow;


    // pre-computed c_dntp_top...speedup
    float *c_dntp_top_pc;
    // pre-computed values
    float SP,etbR,tauB,sens;
    int i_start;

    void SingleFlowIncorporationTrace(float A,float *fval)
    {
        float d;
        float kr,kmax;
        float C;

        d = reg_p->d[NucID]*p->dmult;
        kr = reg_p->krate[NucID]*p->kmult[fnum];
        kmax = reg_p->kmax[NucID];
        C = reg_p->nuc_shape.C;

        ComputeCumulativeIncorporationHydrogens(ivalPtr, len, deltaFrameSeconds, c_dntp_top_pc, ISIG_SUB_STEPS_SINGLE_FLOW, i_start,  C, A, SP, kr, kmax, d, math_poiss);
        MultiplyVectorByScalar(ivalPtr, sens,len);  // transform hydrogens to signal       // variables used for solving background signal shape
        RedSolveHydrogenFlowInWell(fval,ivalPtr,len,i_start,deltaFrame,tauB);
        MultiplyVectorByScalar(fval,p->gain,len);
   }

    float *xvals;
    BkgModSingleFlowFitParams min_params,max_params;
};




#endif // BKGMODSINGLEFLOWFIT_H
