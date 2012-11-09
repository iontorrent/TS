/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODSINGLEFLOWFITTMIDNUC_H
#define BKGMODSINGLEFLOWFITTMIDNUC_H

#include "LevMarFitterV2.h"
#include "MathOptim.h"
#include "DiffEqModel.h"

#define BKG_CORRECT_BEFORE_FIT

struct BkgModSingleFlowFitTMidNucParams {
    float Ampl;
    float delta_t_mid_nuc;
    float kmult;
};

class BkgModSingleFlowFitTMidNuc : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    BkgModSingleFlowFitTMidNuc(int _len,float *frameNumber,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss)
    {
        ivalPtr = new float[_len];
        xvals = frameNumber;
        deltaFrame = _deltaFrame;
        deltaFrameSeconds = _deltaFrameSeconds;
        c_dntp_top_pc = new float[ISIG_SUB_STEPS_SINGLE_FLOW*_len];
        math_poiss = _math_poiss;
        Initialize(3,_len,xvals);
    }
    
    // useful for evaluating sum-of-squares difference without invoking the full Lev-Mar
    void SetJustAmplitude(float A){
      params.Ampl=A;
    }

    // passes in params and reg_params pointers for the well/region we are fitting
    // also indicates which flow we are fitting
    // !!!!! This function damages the _fg structure it is passed!!!!!!  It is not safe!!!!
    void SetWellRegionParams(float _Copies,float _R,float _gain,float _dmult,float _kmult,struct reg_params *_rp,int _fnum,
                             int _nnum,int _flow)
    {
        reg_p = _rp;
        params.Ampl = 1.0f;
        params.delta_t_mid_nuc = 0.0f;
        params.kmult = 1.0f;
        fnum = _fnum;
        NucID = _nnum;
        flow = _flow;
        Copies = _Copies;
        gain = _gain;
        dmult = _dmult;
        
        // since this uses a library function..and the parameters involved aren't fit
        // it's helpful to compute this once and not in the model function
        SP =  (float)(COPYMULTIPLIER * Copies)*pow(reg_p->CopyDrift,flow);

        etbR = AdjustEmptyToBeadRatioForFlow(_R,reg_p,NucID,flow);
        tauB = ComputeTauBfromEmptyUsingRegionLinearModel(reg_p,etbR);
        sens = reg_p->sens*SENSMULTIPLIER;

        sigma = GetModifiedSigma (& (_rp->nuc_shape),NucID);
        t_mid_nuc = GetModifiedMidNucTime (& (_rp->nuc_shape),NucID,fnum);
        nuc_flow_span = _rp->nuc_shape.nuc_flow_span;
    }
    


    // optionally set maximum value for parameters
    void SetParamMax(BkgModSingleFlowFitTMidNucParams _max_params)
    {
        max_params = _max_params;
        LevMarFitterV2::SetParamMax((float *)&max_params);
    }

    // optionally set minimum value for parameters
    void SetParamMin(BkgModSingleFlowFitTMidNucParams _min_params)
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
    BkgModSingleFlowFitTMidNucParams params;

    ~BkgModSingleFlowFitTMidNuc()
    {
        delete [] ivalPtr;
        delete [] c_dntp_top_pc;
    }

    // evaluates the function using the values in params
    virtual void Evaluate(float *y) {
        Evaluate(y,(float *)(&params));
    }

    PoissonCDFApproxMemo *math_poiss;

protected:
    virtual void Evaluate(float *y, float *params) {
        SingleFlowIncorporationTrace(params[0],params[1],params[2],y);
    }

    // loop exit condition
    virtual bool DoneTest(int iter,int max_iter,BkgFitLevMarDat *data,float lambda,int &done_cnt,float residual,float r_chg)
    {
        (void)done_cnt;
        (void)residual;
        (void)r_chg;
        (void)lambda;

        if (GetDataDelta (0) * GetDataDelta (0) < 0.0000025) done_cnt++;
        else done_cnt = 0;

        return((iter >= max_iter) || (done_cnt > 5));
    }

private:

    // pointers to the BkgModel parameters for the well/region being fit
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
    float dmult;
    float Copies,gain;
    float t_mid_nuc;
    float sigma;
    float nuc_flow_span;

    int i_start;

    void SingleFlowIncorporationTrace(float A,float delta_t_mid_nuc,float kmult,float *fval)
    {
        float d;
        float kr,kmax;
        float C;

        d = reg_p->d[NucID]*dmult;
        kr = reg_p->krate[NucID]*kmult;
        kmax = reg_p->kmax[NucID];
        C = reg_p->nuc_shape.C[NucID];

        i_start = SigmaRiseFunction(c_dntp_top_pc,npts,xvals,ISIG_SUB_STEPS_SINGLE_FLOW,C,t_mid_nuc+delta_t_mid_nuc,sigma,nuc_flow_span,true);
        ComputeCumulativeIncorporationHydrogens(ivalPtr,npts, deltaFrameSeconds, c_dntp_top_pc, ISIG_SUB_STEPS_SINGLE_FLOW, i_start,  C, A, SP, kr, kmax, d, reg_p->molecules_to_micromolar_conversion, math_poiss);
        MultiplyVectorByScalar(ivalPtr, sens,npts);  // transform hydrogens to signal       // variables used for solving background signal shape
        RedSolveHydrogenFlowInWell(fval,ivalPtr,npts,i_start,deltaFrame,tauB);
        MultiplyVectorByScalar(fval,gain,npts);
   }

    float *xvals;
    BkgModSingleFlowFitTMidNucParams min_params,max_params;
};




#endif // BKGMODSINGLEFLOWFITTMIDNUC_H
