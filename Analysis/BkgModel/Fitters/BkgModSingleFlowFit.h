/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODSINGLEFLOWFIT_H
#define BKGMODSINGLEFLOWFIT_H

#include "LevMarFitterV2.h"

#include "TraceCurry.h"



class BkgModSingleFlowFit : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
  public:

    TraceCurry calc_trace;

    // constructor
    BkgModSingleFlowFit (int _len,float *frameNumber,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss, int _nparams)
    {
      calc_trace.Allocate (_len,_deltaFrame, _deltaFrameSeconds, _math_poiss);
      xvals = frameNumber;
      Initialize (_nparams,_len,xvals);
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

    // entry point for fitting
    virtual int Fit (int max_iter,float *y)
    {
      return (LevMarFitterV2::Fit (max_iter,y, param_val));
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


  protected:
    virtual void Evaluate (float *y, float *params)
    {
      if (nparams==1)
        calc_trace.SingleFlowIncorporationTrace (params[0],y);
      else
        calc_trace.SingleFlowIncorporationTrace (params[0],params[1],y);
    }

    // loop exit condition
    virtual bool DoneTest (int iter,int max_iter,BkgFitLevMarDat *data,float lambda,int &done_cnt,float residual,float r_chg)
    {
      (void) done_cnt;
      (void) residual;
      (void) r_chg;
      (void) lambda;

      //if (data->delta->at(0)*data->delta->at(0) < 0.0000025) done_cnt++;
      if (GetDataDelta (0) * GetDataDelta (0) < 0.0000025) done_cnt++;
      else done_cnt = 0;

      return ( (iter >= max_iter) || (done_cnt > 1));
    }

  private:

    float *xvals;
};

#endif // BKGMODSINGLEFLOWFIT_H
