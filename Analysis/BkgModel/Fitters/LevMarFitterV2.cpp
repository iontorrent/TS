/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//#include <armadillo>
#include "LevMarFitterV2.h"
#include "BkgFitLevMarDat.h"
#include <float.h>

using namespace arma;

int LevMarFitterV2::Fit (bool gauss_newton,
                         int max_iter,
                         float *y,
                         float *params,
                         float *std_err)
{
  int iter=0;
  int done_cnt = 0;
  float fval[npts];
  float r_start = FLT_MAX;
  float r_chg = 0.0f;
  float bfjac[npts*nparams];
  float err_vect[npts];
  double bfjtj[nparams*nparams];
  double bfrhs[nparams];

  data->delta->set_size (nparams);
  data->delta->zeros (nparams);
  // evaluate where we are
  // always have something even if we never succeed in a step
  Evaluate (fval,params);
  if (enable_fval_cache)
    memcpy (fval_cache,fval,sizeof (float[npts]));
  
  r_start = CalcResidual (y, fval, npts, &err_vect[0]);

  bool converged = false;
  while (!(converged = DoneTest (iter,max_iter,data,lambda,done_cnt,r_start,r_chg)) and !ForceQuit(iter,max_iter, lambda))
  {

    // remember the last residual for comparison later
    r_chg = 0.0;
    MakeJacobian (bfjac, params, fval);
    CalculateJTJ (bfjtj,bfjac);
    CalculateRHS (bfrhs,bfjac, err_vect, params);

    // the following loop saves computation if the first attempt to adjust fails, but subsequent
    // attempts with larger values of lambda would succeed.  In this case, a new value of lambda
    // can be attempted, but the jtj and rhs matricies did not actually change and do not
    // need to be re-computed

    if (gauss_newton)
      TryGaussNewtonStep(y, &fval[0], params, &err_vect[0], &bfjtj[0], &bfrhs[0], done_cnt, r_start, r_chg);
    else 
      TryLevMarStep (y, &fval[0], params, &err_vect[0], &bfjtj[0], &bfrhs[0], done_cnt, r_start, r_chg);
    iter++;
  }

  SetConverged(converged);
  residual = r_start;

  // compute parameter std errs
  if (std_err)
  {

    Mat<double> tjac = (trans (* (data->jac)));
    Mat<double> std_err_jtj = (tjac * (* (data->jac)));


    Mat<double> std_err_jtj_inv = inv (std_err_jtj);
    double mrv = (residual * npts) / (npts - nparams); // mean residual variance
    std_err_jtj_inv *= mrv;

    for (int i=0;i < nparams;i++)
      std_err[i] = sqrt (std_err_jtj_inv (i, i));
  }
  if (numException >0)
    std::cout << "LevMarFitterV2: numException = " << numException << std::endl;
  regularizer = 0.0f; // reset
  numException = 0; // reset messages for next invocation of the same entity
  return (iter);
}

void LevMarFitterV2::Initialize (int _nparams,int _npts,float *_x)
{
  nparams = _nparams;
  npts = _npts;
  x = _x;

  param_val = new float[npts];
  param_max = new float[npts];
  param_min = new float[npts];

  // make all matricies the correct size
  if (data == NULL)
  {
    data = new BkgFitLevMarDat();
  }
  data->jac->set_size (npts,nparams);
  data->jtj->set_size (nparams,nparams);
  data->lhs->set_size (nparams,nparams);
  data->rhs->set_size (nparams);
  data->delta->set_size (nparams);

  //err_vect.resize(npts);

  dp = new float[_nparams];
  for (int i=0;i<_nparams;i++)
    dp[i] = 0.001;

  residualWeight = new float[npts];
  wtScale = npts;
  for (int i=0;i<npts;i++)
    residualWeight[i] = 1.0f;

  fval_cache = new float[npts];
  enable_fval_cache = false; // pick a start
  numException = 0;
}

void LevMarFitterV2::SetParamMax (float *_param_max)
{
  //param_max = _param_max;
  for (int i=0; i<nparams; i++)
    param_max[i] = _param_max[i];
}
void LevMarFitterV2::SetParamMin (float *_param_min)
{
  //param_min = _param_min;
  for (int i=0; i<nparams; i++)
    param_min[i] = _param_min[i];
}

void LevMarFitterV2::SetParamVal (float *_param_val)
{
  //param_min = _param_min;
  for (int i=0; i<nparams; i++)
    param_val[i] = _param_val[i];
}

void LevMarFitterV2::SetNthParam(float val, int idx)
{
  if (idx<nparams)
    param_val[idx] = val;
}

float LevMarFitterV2::ReturnNthParam(int idx)
{
  if (idx<nparams)
    return( param_val[idx]);
  else
    return(0.0f);
}
// generic grid search
void LevMarFitterV2::GridSearch (int steps,float *y,float *params)
{
  // if we were never configured with min/max parameter values, just return
  // we only allow grid search if the region in which to search has been defined
  if ( (param_min == NULL) || (param_max == NULL))
    return;

  float param_step[nparams];
  float p[nparams];

  // calculate step size for each parameter
  for (int pn=0;pn < nparams;pn++)
  {
    param_step[pn] = (param_max[pn] - param_min[pn]) / (steps-1);
    p[pn] = param_min[pn];
  }

  int step_num[nparams];
  memset (step_num,0,sizeof (step_num));

  int inum;
  int total_steps;
  int digit;
  float *fval = new float[npts];
  float min_err = FLT_MAX;
  float r1;

  total_steps = 1;
  for (int i=0;i < nparams;i++)
    total_steps *= steps;

  for (inum = 0;inum < total_steps;inum++)
  {
    // convert to baseN (where N==steps).  Each 'digit' of inum in baseN corresponds to the
    // step position of a parameter
    int inum_tmp = inum;
    int place_val = total_steps/steps;
    for (digit=nparams-1;digit >= 0;digit--)
    {
      int val = inum_tmp / place_val;
      inum_tmp -= val * place_val;
      p[digit] = param_min[digit]+param_step[digit]*val;
      place_val /= steps;
    }

    Evaluate (fval,p);
    r1 = CalcResidual (y, fval, npts);

    if (r1 < min_err)
    {
      memcpy (params,p,sizeof (p));
      min_err = r1;
    }
  }

  if (enable_fval_cache)
    memcpy (fval_cache,fval,sizeof (float[npts]));

  delete [] fval;
}


static void LinSolveErrMessage( const char* str )
{
  static int maxMessages = 100; //limit complaining to some number to avoid overflowing the log file
  if( --maxMessages > 0 )
    std::cout << str << "\t:" << maxMessages << std::endl;
}

void LevMarFitterV2::TryLevMarStep (float *y,
                                    float *fval,
                                    float *params,
                                    float *err_vect,
                                    double *bfjtj,
                                    double *bfrhs,
                                    int done_cnt,
                                    float &r_start,
                                    float &r_chg)
{
  float r_trial;
  float tmp[npts];
  float params_new[nparams];
  double bflhs[nparams*nparams];

  bool cont_proc = false;
  if (lambda>lambda_threshold)
    cont_proc = true; // skip, we can't do anything if we've escaped
  while (!cont_proc)
  {
    // add lambda parameter to jtj to form lhs of matrix equation
    memcpy (bflhs,bfjtj,sizeof (bflhs));
    for (int i=0;i < nparams;i++)
      bflhs[i*nparams+i] *= (1.0 + lambda);
    
    // can actually check against zero here because that is the problem
    if ((nparams==1) & (bflhs[0]==0.0f)) // one entry, it is zero, and our regularizer is zero armadillo does not throw exception but NaN
      regularizer += REGULARIZER_VAL;  // this is a problem with our derivative calculation when we hit the top of the box
      
    // add regularizer in case we have trouble
    for (int i=0;i < nparams;i++)
      bflhs[i*nparams+i] += regularizer;
    
    // solve for delta
    try
    {
      {

        // armadillo just as fast for 1 and 2 so no special case required
        // don't forget all our time is spent in other functions - simplifying here lets us make the code better and faster in the
        // actual time-sinks

        for (int r=0;r < nparams;r++)
        {
          data->rhs->at (r) = bfrhs[r];

          for (int c=0;c < nparams;c++)
            data->lhs->at (r,c) = bflhs[r*nparams+c];
        }

        * (data->delta) = solve (* (data->lhs),* (data->rhs));
      }

      bool NaN_detected = false;
      for (int i=0;i < nparams;i++)
      {
        double tmp_eval = data->delta->at(i);
        
        // test for NaN
        if (tmp_eval != tmp_eval)
        {
          NaN_detected = true;
          data->delta->at (i) = 0.0;
          tmp_eval = 0.0;
        }
        // if tmp_eval out of range for float, it's a disaster all around
        tmp_eval += params[i]; // safe promotion from float to double
        if ((tmp_eval>FLT_MAX) or (tmp_eval<-FLT_MAX)) // no adjust if disaster
        {
          tmp_eval = params[i]; 
          data->delta->at(i) = 0.0;
        }
        // adjust parameter from current baseline
        params_new[i] = tmp_eval; // demotion from double to float
      }
      // make this virtual in case constraints need to be non-orthogonal boxes
      ApplyMoveConstraints (params_new);
      // apply limits if necessary

      if (!NaN_detected)
      {
        // re-calculate error
        Evaluate (tmp,params_new);

        // calculate error bw function and data
        r_trial = CalcResidual (y, tmp, npts, &err_vect[0]);

        r_chg = r_trial - r_start;
      }
      else{
          char my_message[1024];
          sprintf(my_message,"LevMarFitterV2.cpp: NaN in TryLevMarStep matrix solve - at %f %f %f", lambda, regularizer,params[0]);
          LinSolveErrMessage(my_message);
      }

      if (!NaN_detected && (r_trial < r_start))
      {
        // safe way to check if this is step is feasible given that lambda is a float
        if (lambda> LEVMAR_STEP_V2*FLT_MIN)
          lambda /= LEVMAR_STEP_V2;

        memcpy (params,params_new,sizeof (float[nparams]));
        memcpy (fval,tmp,sizeof (float[npts]));
        cont_proc = true;
        r_start = r_trial;
        if (enable_fval_cache)
          memcpy (fval_cache,fval,sizeof (float[npts]));

      }
      else
      {
        lambda *= LEVMAR_STEP_V2;
      }

      if (debug_trace)
        printf ("lambda = %f, done = %d\n",lambda,done_cnt);

    }
    catch (std::runtime_error &le)
    {
      // a failed solution of the matrix should be treated just like a failed attempt
      // at improving the fit...increase lambda and try again
      if (2.0*regularizer>REGULARIZER_VAL)
      {
        // you're not an exception until you're an exception with a probably nonzero determinant
        numException++;
        LinSolveErrMessage("LevMarFitterV2.cpp: Fit - exception runtime ...");
      }
      data->delta->set_size (nparams);
      data->delta->zeros (nparams);
      lambda *= LEVMAR_STEP_V2;
      regularizer += REGULARIZER_VAL; // keep adding until we're nonzero on the diagonal
    }

    // we need a way to bail out of this loop if lambda continues to increase but we never
    // get a better fit
    if (lambda > lambda_threshold)
      cont_proc = true;

    if (debug_trace)
      printf ("lambda = %f, done = %d\n",lambda,done_cnt);
  }
  // bailed out of loop
}

void LevMarFitterV2::TryGaussNewtonStep(float *y,
                                    float *fval,
                                    float *params,
                                    float *err_vect,
                                    double *bfjtj,
                                    double *bfrhs,
                                    int done_cnt,
                                    float &r_start,
                                    float &r_chg)
{
  float r_trial;
  float tmp[npts];
  float params_new[nparams];

  {
    
    // can actually check against zero here because that is the problem
    if ((nparams==1) & (bfjtj[0]==0.0f)) // one entry, it is zero, and our regularizer is zero armadillo does not throw exception but NaN
      regularizer += REGULARIZER_VAL;  // this is a problem with our derivative calculation when we hit the top of the box
      
    // add regularizer in case we have trouble
    for (int i=0;i < nparams;i++)
      bfjtj[i*nparams+i] += regularizer;
    
    // solve for delta
    try
    {
      for (int r=0;r < nparams;r++)
      {
        data->rhs->at (r) = bfrhs[r];

        for (int c=0;c < nparams;c++)
          data->lhs->at (r,c) = bfjtj[r*nparams+c];
      }

      * (data->delta) = solve (* (data->lhs),* (data->rhs));

      bool NaN_detected = false;
      for (int i=0;i < nparams;i++)
      {
        double tmp_eval = data->delta->at(i);
        
        // test for NaN
        if (tmp_eval != tmp_eval)
        {
          NaN_detected = true;
          data->delta->at (i) = 0.0;
          tmp_eval = 0.0;
        }
        // if tmp_eval out of range for float, it's a disaster all around
        tmp_eval += params[i]; // safe promotion from float to double
        if ((tmp_eval>FLT_MAX) or (tmp_eval<-FLT_MAX)) // no adjust if disaster
        {
          tmp_eval = params[i]; 
          data->delta->at(i) = 0.0;
        }
        // adjust parameter from current baseline
        params_new[i] = tmp_eval; // demotion from double to float
      }
      // make this virtual in case constraints need to be non-orthogonal boxes
      ApplyMoveConstraints (params_new);
      // apply limits if necessary

      if (!NaN_detected)
      {
        // re-calculate error
        Evaluate (tmp,params_new);
        // dynamic emphasis
        DetermineAndSetWeightVector(params_new[0]);

        // calculate error bw function and data
        r_trial = CalcResidual (y, tmp, npts, &err_vect[0]);
        r_chg = r_trial - r_start;
      }
      else{
          char my_message[1024];
          sprintf(my_message,"LevMarFitterV2.cpp: NaN in TrygaussNewtonStep matrix solve - at %f %f", regularizer,params[0]);
          LinSolveErrMessage(my_message);
      }

      if (!NaN_detected && (r_trial < r_start))
      {
        memcpy (params,params_new,sizeof (float[nparams]));
        memcpy (fval,tmp,sizeof (float[npts]));
        r_start = r_trial;
        if (enable_fval_cache)
          memcpy (fval_cache,fval,sizeof (float[npts]));
      }
      else {
        DetermineAndSetWeightVector(params[0]);
      }

      if (debug_trace)
        printf ("lambda = %f, done = %d\n",lambda,done_cnt);

    }
    catch (std::runtime_error &le)
    {
      // a failed solution of the matrix should be treated just like a failed attempt
      // at improving the fit...increase lambda and try again
      if (2.0*regularizer>REGULARIZER_VAL)
      {
        // you're not an exception until you're an exception with a probably nonzero determinant
        numException++;
        LinSolveErrMessage("LevMarFitterV2.cpp: Fit - exception runtime ...");
      }
      data->delta->set_size (nparams);
      data->delta->zeros (nparams);
      regularizer += REGULARIZER_VAL; // keep adding until we're nonzero on the diagonal
    }

    if (debug_trace)
      printf ("lambda = %f, done = %d\n",lambda,done_cnt);
  }
}



void LevMarFitterV2::CalculateJTJ (double *bfjtj, float *bfjac)
{
  // calculate jtj matrix
  for (int r=0;r < nparams;r++)
    for (int c=r;c < nparams;c++)
      bfjtj[r*nparams+c] = bfjtj[c*nparams+r] = sdot (npts,&bfjac[r],nparams,&bfjac[c],nparams);

}

void LevMarFitterV2::CalculateRHS (double *bfrhs,
                                   float *bfjac,
                                   float *err_vect,
                                   float *params)
{
  // calculate rhs
  //Blas_Mat_Trans_Vec_Mult(jac,err_vect,rhs,1.0,0.0);
  for (int r=0;r < nparams;r++)
    bfrhs[r] = sdot (npts,&bfjac[r],nparams,&err_vect[0],1);

}
void LevMarFitterV2::SetLambdaThreshold (float _lambda_threshold)
{
  lambda_threshold = _lambda_threshold;
}

void LevMarFitterV2::SetLambdaStart (float _initial_lambda)
{
  // reset the extra variables here, as we're starting a new optimization
  lambda = _initial_lambda;
  regularizer = 0.0f;
  numException = 0;
}

float LevMarFitterV2::GetLambda (void)
{
  return (lambda);
}

void LevMarFitterV2::SetDebugTrace (bool _enable)
{
  debug_trace = _enable;
}

// set the weighting vector
void LevMarFitterV2::SetWeightVector (float *vect)
{
  wtScale = 0.0;
  for (int i=0;i < npts;i++)
  {
    residualWeight[i] = vect[i];
    wtScale += vect[i];
  }
}


// get the mean squared error after the fit
float LevMarFitterV2::GetMeanSquaredError (float *y,bool use_fval_cache)
{
  if (y == NULL)
    return residual;
  else
  {
    float *tmp;
    float tmp_eval[npts];

    if (enable_fval_cache && use_fval_cache)
    {
      tmp = fval_cache;
    }
    else
    {
      Evaluate (tmp_eval);
      tmp = tmp_eval;
    }
    residual = CalcResidual (y, tmp, npts);
    return residual;
  }
}

void LevMarFitterV2::ReturnPredicted(float *f_predict, bool use_fval_cache)
{
  if (f_predict!=NULL)
  {
    if (enable_fval_cache && use_fval_cache)
      memcpy(f_predict,fval_cache, sizeof(float[npts]));
    else
      Evaluate(f_predict);
  }
}

void LevMarFitterV2::SetDataDeltaNull()
{
  (void) data->delta ;
}
double LevMarFitterV2::GetDataDelta (int i)
{
  return data->delta->at (i);
}

LevMarFitterV2::~LevMarFitterV2()
{
  delete [] residualWeight;
  delete [] dp;

  delete [] fval_cache;
  if (data != NULL)
    delete data;
  delete[] param_val;
  delete[] param_max;
  delete[] param_min;
}
