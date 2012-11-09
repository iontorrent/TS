/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PROJECTIONSEARCHFIT_H
#define PROJECTIONSEARCHFIT_H

#include "TraceCurry.h"
#include <DotProduct.h>

//@TODO: unify this single flow instance and the multiflow instance in BkgSearchAmplitude

//@TODO:  there is probably a "fitter" class which both this and levmar inherit from
// as this is >not< an instance of levmarfitterv2

class ProjectionSearchOneFlow
{
    // evaluate the fitted function w/ the specified parameters
  public:
    // the starting point and end point of the fit
    float paramA;
    float max_paramA, min_paramA;

    float projection_stop_threshold;

    float *fval_cache;
    float *residual_weight;
    float wtScale;
    float residual;
    int npts;

    bool enable_fval_cache;

    TraceCurry calc_trace;


    // set the weighting vector
    void SetWeightVector (float *vect)
    {
      wtScale = 0.0;
      for (int i=0;i < npts;i++)
      {
        residual_weight[i] = vect[i];
        wtScale += vect[i];
      }
    }

    // does the fit using a completely different (and faster) alternative
    void ProjectionSearch (float *y)
    {
      float tmp_eval[npts];
      float tmp_data[npts];
      float A_guess;
      float A_new;

      // pre-multiply data by emphasis vector
      MultiplyVectorByVector (tmp_data, y, residual_weight, npts);

      // evalutate function at the current amplitude estimate
      A_guess = calc_trace.GetStartAmplitude();

      bool done = false;
      int iter_cnt = 0;
      // only allow a few iterations here.  Reads that try to go beyond this are generally poor quality reads
      // with a very low copy count that can sometimes oscillate and fail to converge.
      while (!done && (iter_cnt < 30))
      {
        iter_cnt++;

        // if the estimate is really small evaluate it at A=projection_stop_threshold
        if (A_guess < projection_stop_threshold) A_guess = projection_stop_threshold;
        calc_trace.SingleFlowIncorporationTrace (A_guess,fval_cache);
        MultiplyVectorByVector (tmp_eval, fval_cache, residual_weight, npts);

        // compute the scaling
        float denom = DotProduct (npts,tmp_eval,tmp_eval);
        float amult = DotProduct (npts,tmp_eval,tmp_data) /denom;

        A_new = amult * A_guess;

        // if we are very unlucky...we may have had a divide by zero here
        if ( (A_new != A_new) || (denom == 0.0f))
        {
          A_new = A_guess;
          done = true;
          continue;
        }

        // check for min value
        if (A_new < min_paramA)
        {
          A_new = paramA = min_paramA;

          if (enable_fval_cache)
            calc_trace.SingleFlowIncorporationTrace (A_new,fval_cache);
          done = true;
          continue;
        }

        // check for max value
        if (A_new > max_paramA)
        {
          A_new = paramA = max_paramA;

          if (enable_fval_cache)
            calc_trace.SingleFlowIncorporationTrace (A_new,fval_cache);
          done = true;
          continue;
        }

        // check for done condition (change is less than 1-mer)
        if (fabs (A_new-A_guess) < projection_stop_threshold)
        {
          paramA = A_new;
          if (enable_fval_cache)
            MultiplyVectorByScalar (fval_cache,amult,npts);
          done = true;
          continue;
        }

        // do another iteration if necessary
        A_guess = A_new;
      }
    }
    ProjectionSearchOneFlow (int _len, float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss)
    {
      calc_trace.Allocate (_len,_deltaFrame, _deltaFrameSeconds, _math_poiss);
      Initialize (_len);
      enable_fval_cache = true;
    }

    void Initialize (int _npts)
    {
      npts = _npts;
      fval_cache = new float[npts];
      residual_weight = new float[npts];
      paramA = 0;
      min_paramA = 0;
      max_paramA = MAX_HPLEN-1;
      projection_stop_threshold = 0.1f;
      residual = 0.0f;
      wtScale = 0.0f;
    }
// get the mean squared error after the fit
    float  GetMeanSquaredError (float *y,bool use_fval_cache)
    {
      float *tmp;
      float tmp_eval[npts];

      if (enable_fval_cache && use_fval_cache)
      {
        tmp = fval_cache;
      }
      else
      {
        calc_trace.SingleFlowIncorporationTrace (paramA,tmp_eval);
        tmp = tmp_eval;
      }
      residual = CalcResidual (y, tmp);
      return residual;
    }

    float CalcResidual (float *y, float *tmp)
    {
      float r= 0.0f;
      float e;
      for (int i=0;i <npts;i++)
      {
        e = residual_weight[i] * (y[i] - tmp[i]);
        r += e*e;
      }

      // r = sqrt(r/wtScale)
      r = (r/wtScale);
      return r;
    }

    void SetFvalCacheEnable (bool _flag)
    {
      enable_fval_cache=_flag;
    };

    ~ProjectionSearchOneFlow()
    {
      delete[] fval_cache;
      delete[] residual_weight;
    }
};

#endif // PROJECTIONSEARCHFIT_H
