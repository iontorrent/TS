/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODSINGLEFLOWFITKRATE_H
#define BKGMODSINGLEFLOWFITKRATE_H

#include "LevMarFitterV2.h"
#include "MathOptim.h"
#include "DiffEqModel.h"

#define n_to_uM_conv (0.000062f)

struct BkgModSingleFlowFitKrateParams {
    float Ampl;
    float kmult;
    float dmultX;
};

class BkgModSingleFlowFitKrate : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    BkgModSingleFlowFitKrate(int _len,float *frameNumber,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss, bool _fit_d = false)
    {
        ivalPtr = new float[_len];
        xvals = frameNumber;
        deltaFrame = _deltaFrame;
        deltaFrameSeconds = _deltaFrameSeconds;
        fit_d = _fit_d;
        c_dntp_top_pc = NULL;
        math_poiss = _math_poiss;
        if (_fit_d)
            Initialize(3,_len,xvals);
        else
            Initialize(2,_len,xvals);
        oldstyle = true;

    }
   // useful for evaluating sum-of-squares difference without invoking the full Lev-Mar
    void SetJustAmplitude(float A){
      params.Ampl=A;
    }

    // passes in params and reg_params pointers for the well/region we are fitting
    // also indicates which flow we are fitting
    // Not safe!!!!! Damages _fg structure without warning!!!!
    void SetWellRegionParams(struct bead_params *_p,struct reg_params *_rp,int _fnum,
                             int _nnum,int _flow,
                             int _i_start,float *_c_dntp_top)
    {
        p = _p;
        reg_p = _rp;
        params.Ampl = p->Ampl[_fnum];
        params.kmult = p->kmult[_fnum];
        params.dmultX = 1.0;
        fnum = _fnum;
        NucID = _nnum;
        flow = _flow;
 
        // Copies modifies the allowed kmult
        float decision_threshold = 2.0;
        max_offset = 1.0; // start at 1.0 when 0 amplitude
        max_slope = (param_max[1]-1)/(decision_threshold/p->Copies);
        min_offset = 1.0; // start at 1.0 when 0 amplitude
        min_slope = (param_min[1]-1)/(decision_threshold/p->Copies);
        
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
    void SetParamMax(BkgModSingleFlowFitKrateParams _max_params)
    {
        max_params = _max_params;
        LevMarFitterV2::SetParamMax((float *)&max_params);
    }

    // optionally set minimum value for parameters
    void SetParamMin(BkgModSingleFlowFitKrateParams _min_params)
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
    BkgModSingleFlowFitKrateParams params;

    ~BkgModSingleFlowFitKrate()
    {
        delete [] ivalPtr;
    }

    // evaluates the function using the values in params
    virtual void Evaluate(float *y) {
        Evaluate(y,(float *)(&params));
    }



protected:
   virtual void ApplyMoveConstraints(float *params_new)
    {

      // it's a special case here
      // parameter 0 is amplitude
      // parameter 1 is kmult
      if (params_new[0]>param_max[0]) params_new[0] = param_max[0];
      if (params_new[0]<param_min[0]) params_new[0] = param_min[0];
      if (oldstyle){
        if (params_new[1]>param_max[1]) params_new[1] = param_max[1];
        if (params_new[1]<param_min[1]) params_new[1] = param_min[1];
      } else
      {
        // so if we fit a larger amplitude, the allowed offset increases
        float kmult_max, kmult_min;
        // smoothly transition between no kmult variation and some kmult variation
        kmult_max = params_new[0]*max_slope+max_offset;
        kmult_min = params_new[0]*min_slope+min_offset;
        if (kmult_max>param_max[1]) kmult_max = param_max[1];
        if (kmult_max<1.0) kmult_max = 1.0;
        if (kmult_min<param_min[1]) kmult_min = param_min[1];
        if (kmult_min>1.0) kmult_min = 1.0;

        // now apply
        if (params_new[1]>kmult_max) params_new[1] = kmult_max;
        if (params_new[1]<kmult_min) params_new[1] = kmult_min;
      }
    }
  
    virtual void Evaluate(float *y, float *params) {
        // calculate proton flux and decay
        if (fit_d)
            SingleFlowIncorporationTrace(params[0],params[1],params[2],y);
        else
            SingleFlowIncorporationTrace(params[0],params[1],1.0,y);
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

    PoissonCDFApproxMemo *math_poiss;
private:

    // pointers to the BkgModel parameters for the well/region being fit
    struct bead_params *p;
    struct reg_params *reg_p;
    
    // single-flow proton flux    
    float *ivalPtr;
    // single-flow delta-T between data points
    float *deltaFrame;
    float *deltaFrameSeconds;

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
    
    // kmult special constraints
    float max_slope, max_offset;
    float min_slope, min_offset;
      bool oldstyle;

    void SingleFlowIncorporationTrace(float A,float kmult,float dmultX,float *fval)
    {
        (void)dmultX;

        float d;
        float kr,kmax;
        float C;

        d = reg_p->d[NucID]*p->dmult;
        // initialize diffusion/reaction simulation for this flow
        kr = reg_p->krate[NucID]*kmult;
        kmax = reg_p->kmax[NucID];
        C = reg_p->nuc_shape.C;

        ComputeCumulativeIncorporationHydrogens(ivalPtr, len, deltaFrameSeconds, c_dntp_top_pc, ISIG_SUB_STEPS_SINGLE_FLOW, i_start,  C, A, SP, kr, kmax, d,math_poiss);
        MultiplyVectorByScalar(ivalPtr, sens,len);  // transform hydrogens to signal       // variables used for solving background signal shape
        RedSolveHydrogenFlowInWell(fval,ivalPtr,len,i_start,deltaFrame,tauB);
        MultiplyVectorByScalar(fval,p->gain,len);
    }

    float *xvals;
    BkgModSingleFlowFitKrateParams min_params,max_params;
    bool fit_d;
};




#endif // BKGMODSINGLEFLOWFITKRATE_H
