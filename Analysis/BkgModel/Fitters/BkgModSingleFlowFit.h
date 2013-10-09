/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODSINGLEFLOWFIT_H
#define BKGMODSINGLEFLOWFIT_H

#include "LevMarFitterV2.h"

#include "TraceCurry.h"
#include "EmphasisVector.h"



class BkgModSingleFlowFit : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
  public:

    TraceCurry calc_trace;

    // constructor
    BkgModSingleFlowFit (int len,float *frameNumber,float *deltaFrame, float *deltaFrameSeconds, PoissonCDFApproxMemo *math_poiss, int nparams)
    {
      calc_trace.Allocate (len,deltaFrame, deltaFrameSeconds, math_poiss);
      xvals = frameNumber;
      weights = new float[len];
      Initialize (nparams,len,xvals);
    }

    // useful for evaluating sum-of-squares difference without invoking the full Lev-Mar
    void SetJustAmplitude (float A)
    {
      param_val[AMPLITUDE]=A;
    }

    void InitParams()
    {
      param_val[AMPLITUDE] = calc_trace.GetStartAmplitude();
      if (nparams>1)
        param_val[KMULT] = calc_trace.GetStartKmult();
    }

    // entry point for grid search
    void GridSearch (int steps,float *y)
    {
      LevMarFitterV2::GridSearch (steps,y, param_val);
    }
    
     void ErrorSignal(float *obs,float *fit, float *posptr, float *negptr)
     {
        calc_trace.ErrorSignal(obs,fit, posptr, negptr);
      }

    // entry point for fitting
    virtual int Fit (bool gauss_newton, int max_iter, float *y)
    {
      return (LevMarFitterV2::Fit (gauss_newton, max_iter, y, param_val));
    }


    // evaluates the function using the values in params
    virtual void Evaluate (float *y)
    {
      Evaluate (y, param_val);
    }

    int GetParams()
    {
      return nparams;
    }

    void SetUpEmphasis(EmphasisClass* emphasis_data)
    {
      _emphasis_data = emphasis_data;
    }

    virtual void DetermineAndSetWeightVector(float ampl) 
    {
      _emphasis_data->CustomEmphasis(weights, ampl);
      LevMarFitterV2::SetWeightVector(weights);
    }

    virtual ~BkgModSingleFlowFit()
    {
      delete [] weights;
    }

  protected:

    virtual void Evaluate (float *y, float *params)
    {
      if (nparams==1)
        calc_trace.SingleFlowIncorporationTrace (params[0],y);
      else
        calc_trace.SingleFlowIncorporationTrace (params[0],params[1],y);
    }

    bool DoneTest (int iter,int max_iter,BkgFitLevMarDat *data,float lambda,int &done_cnt,float residual,float r_chg)
    {
      (void) done_cnt;
      (void) residual;
      (void) r_chg;
      (void) lambda;

      if ( GetDataDelta(0) * GetDataDelta(0) < 0.0000025) 
        done_cnt++;
      else
        done_cnt = 0;

      return (done_cnt > 1);
    }

  private:

    float *xvals;
    float* weights;
    EmphasisClass* _emphasis_data;
};

#endif // BKGMODSINGLEFLOWFIT_H
