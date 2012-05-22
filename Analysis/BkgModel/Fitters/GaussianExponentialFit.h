/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GAUSSIANEXPONENTIALFIT_H
#define GAUSSIANEXPONENTIALFIT_H

#include "LevMarFitter.h"

struct GaussianExponentialParams {
    float A;
    float sigma;
    float t_mid_nuc;
    float tau;
};

class GaussianExponentialFit : LevMarFitter
{
    // evaluate the fitted function w/ the specified parameters
public:

    // constructor
    GaussianExponentialFit(int _len,float *_x)
    {
        xvals = new float[_len];
        memcpy(xvals,_x,sizeof(xvals));

        Initialize(4,_len,xvals);
    }

    GaussianExponentialFit(int _len)
    {
        xvals = new float[_len];
        
        for (int i=0;i < _len;i++) xvals[i] = (float)i;

        Initialize(4,_len,xvals);
    }

    // optionally set maximum value for parameters
    void SetParamMax(GaussianExponentialParams _max_params)
    {
        max_params = _max_params;
        LevMarFitter::SetParamMax((float *)&max_params);
    }

    // optionally set minimum value for parameters
    void SetParamMin(GaussianExponentialParams _min_params)
    {
        min_params = _min_params;
        LevMarFitter::SetParamMin((float *)&min_params);
    }

    // entry point for grid search
    void GridSearch(int steps,float *y)
    {
        LevMarFitter::GridSearch(steps,y,(float *)(&params));
    }

    // entry point for fitting
    int Fit(int max_iter,float *y)
    {
        return(LevMarFitter::Fit(max_iter,y,(float *)(&params)));
    }

    // get the mean squared error after the fit
    float GetResidual(void) {return residual;}

    // the starting point and end point of the fit
    GaussianExponentialParams params;

    ~GaussianExponentialFit()
    {
        delete [] xvals;
    }

protected:
    virtual void Evaluate(float *y, float *params) {
        
        GaussianExponentialParams *p = (GaussianExponentialParams *)params;
        float rt2s=p->sigma*sqrt(2.0f);
        float ssotau=p->sigma*p->sigma/p->tau;

        for (int i=0;i < len;i++)
        {
            float sigmoid = p->A*(1.0f - erf((ssotau - x[i] + p->t_mid_nuc)/rt2s));
            float exponent = exp((ssotau - 2.0f*x[i] + 2.0f*p->t_mid_nuc)/(2.0f*p->tau))/2.0f;
            y[i] = sigmoid * exponent;
        }
    }

private:
    float *xvals;
    GaussianExponentialParams min_params,max_params;
};




#endif // GAUSSIANEXPONENTIALFIT_H
