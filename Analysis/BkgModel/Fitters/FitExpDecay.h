/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FITEXPDECAY_H
#define FITEXPDECAY_H

#include "LevMarFitterV2.h"
#include "MathOptim.h"
#include "DiffEqModel.h"

struct FitExpDecayParams {
    float  Signal ;  //units of counts, not amplitude
    float tau;  // units of frames
    float dc_offset;  // units of counts
};

class FitExpDecay : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    FitExpDecay(int _len,float *frameNumber)
    {
        xvals = frameNumber;
        Initialize(3,_len,xvals);
    }

    void SetStartAndEndPoints(int start,int end)
    {
       i_start = start;
       i_end = end;
    }

    // optionally set maximum value for parameters
    void SetParamMax(FitExpDecayParams _max_params)
    {
        max_params = _max_params;
        LevMarFitterV2::SetParamMax((float *)&max_params);
    }

    // optionally set minimum value for parameters
    void SetParamMin(FitExpDecayParams _min_params)
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
    virtual int Fit(bool gauss_newton, int max_iter,float *y)
    {
        return(LevMarFitterV2::Fit(gauss_newton, max_iter,y,(float *)(&params)));
    }

    // the starting point and end point of the fit
    FitExpDecayParams params;

    ~FitExpDecay()
    {
    }

    // evaluates the function using the values in params
    virtual void Evaluate(float *y) {
        Evaluate(y,(float *)(&params));
    }

protected:
    virtual void Evaluate(float *y, float *params) {
       memset(y,0,sizeof(float[npts]));

       for (int i=i_start;i < i_end;i++)
          y[i] = params[0]*ExpApprox(-(xvals[i]-xvals[i_start])/params[1]) + params[2];
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

        return(done_cnt > 5);
    }

private:

    int i_start,i_end;
    float *xvals;
    FitExpDecayParams min_params,max_params;
};




#endif // FITEXPDECAY_H
