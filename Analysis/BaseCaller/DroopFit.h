/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DROOPFIT_H
#define DROOPFIT_H

#include "LevMarFitter.h"

struct DroopParams {
    float dr;
    float base;
};

struct DroopParamsStdErr {
    float dr;
    float base;
};

class DroopFit : LevMarFitter
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    DroopFit(int _len)
    {
        Initialize(2,_len,NULL);
        dp[0] = (0.005 / 100.0);
    }

    // optionally set maximum value for parameters
    void SetParamMax(DroopParams _max_params)
    {
        max_params = _max_params;
        LevMarFitter::SetParamMax((float *)&max_params);
        dp[0] = (max_params.dr / 100.0);
    }

    // optionally set minimum value for parameters
    void SetParamMin(DroopParams _min_params)
    {
        min_params = _min_params;
        LevMarFitter::SetParamMin((float *)&min_params);
    }

    // set the X axis for the function
    void SetTimePoint(const std::vector<int> &_sampleTimePoint)
    {
    	sampleTimePoint = _sampleTimePoint;
    }

    // entry point for grid search
    void GridSearch(int steps,float *y)
    {
        LevMarFitter::GridSearch(steps,y,(float *)(&params));
    }

    // entry point for fitting
    int Fit(int max_iter,float *y)
    {
        return(LevMarFitter::Fit(max_iter,y,(float *)(&params), (float *)(&paramsStdErr)));
    }

    // get the mean squared error after the fit
    float GetResidual(void) {return residual;}

    // the starting point and end point of the fit
    DroopParams params;
    DroopParamsStdErr paramsStdErr;

    ~DroopFit()
    {
    }

protected:
    virtual void Evaluate(float *y, float *params) {
        
        DroopParams *p = (DroopParams *)params;

        if (sampleTimePoint.size() == 0) {	// Equidistant x axis
			for (int i=0;i < len;i++) {
			   y[i] = p->base * pow((1.0-p->dr), i+1);
			}

        } else {							// custom x spacing
			for (int i=0;i < len;i++) {
			   y[i] = p->base * pow((1.0-p->dr), sampleTimePoint[i]);
			}
        }
    }

private:
    DroopParams min_params,max_params;
    std::vector<int> sampleTimePoint;
};




#endif // DROOPFIT_H
