/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//#include <armadillo>
#include "LevMarFitterV2.h"
#include "BkgFitLevMarDat.h"

using namespace arma;

int LevMarFitterV2::Fit (int max_iter,
                         float *y,
                         float *params,
                         float *std_err)
{
  int iter=0;
  int done_cnt = 0;
  float fval[npts];
  float r_start = FLT_MAX;
  float r_chg = 0.0;
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
  r_start += EvaluateParamFromPrior (params); // account for prior

  while (!DoneTest (iter,max_iter,data,lambda,done_cnt,r_start,r_chg))
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
    TryLevMarStep (y, &fval[0], params, &err_vect[0], &bfjtj[0], &bfrhs[0], done_cnt, r_start, r_chg);
    iter++;
  }

  residual = r_start;

  // compute parameter std errs
  if (std_err)
  {
    //LaSymmMatDouble std_err_jtj (nparams, nparams);
    Mat<double> tjac = (trans (* (data->jac)));
    Mat<double> std_err_jtj = (tjac * (* (data->jac)));
    //Blas_R1_Update (std_err_jtj, jac, 1.0, 0.0, false);

    //LaVectorLongInt piv;
    //piv.resize (nparams, nparams);

    //LaGenMatDouble std_err_jtj_inv = std_err_jtj;
    //LUFactorizeIP (std_err_jtj_inv, piv);
    //LaLUInverseIP (std_err_jtj_inv, piv);

    Mat<double> std_err_jtj_inv = inv (std_err_jtj);
    double mrv = (residual * npts) / (npts - nparams); // mean residual variance
    std_err_jtj_inv *= mrv;

    for (int i=0;i < nparams;i++)
      std_err[i] = sqrt (std_err_jtj_inv (i, i));
  }
  if (numException >0)
    std::cout << "LevMarFitterV2: numException = " << numException << std::endl;
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

  prior = new float [nparams];
  for (int i=0; i<nparams; i++)
    prior[i] = 0;
  dampers = new float [nparams];
  for (int i=0; i<nparams; i++)
    dampers[i] = 0.0; // no response to priors

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
//
float LevMarFitterV2::EvaluateParamFromPrior (float *param_new)
{
  float sum=0;
  for (int j=0; j<nparams; j++)
    sum += dampers[j]* (param_new[j]-prior[j]) * (param_new[j]-prior[j]);
  return (sum);
}


void LevMarFitterV2::TryLevMarStep (float *y,
                                    float *fval,
                                    float *params,
                                    float *err_vect,
                                    double *bfjtj,
                                    double *bfrhs,
                                    int done_cnt,
                                    float r_start,
                                    float &r_chg)
{
  float r_trial;
  float tmp[npts];
  float params_new[nparams];
  double bflhs[nparams*nparams];

  bool cont_proc = false;
  while (!cont_proc)
  {
    // add lambda parameter to jtj to form lhs of matrix equation
    memcpy (bflhs,bfjtj,sizeof (bflhs));
    for (int i=0;i < nparams;i++)
      bflhs[i*nparams+i] *= (1.0 + lambda);
    // solve for delta
    try
    {

      // these special cases handle the relatively trivial 1 and 2 parameter solutions
      // in a faster way
      if (nparams == 1)
        data->delta->at (0) = bfrhs[0]/bflhs[0];
      else if (nparams == 2)
      {
        double a,b,c,d,det;
        a = bflhs[0];
        b = bflhs[1];
        c = bflhs[2];
        d = bflhs[3];
        det = 1.0 / (a*d - b*c);
        data->delta->at (0) = (d*bfrhs[0]-b*bfrhs[1]) *det;
        data->delta->at (1) = (-c*bfrhs[0]+a*bfrhs[1]) *det;
      }
      else
      {
        // since we are doing a bigger matrix...package up everything into
        // a form the real matrix solve
        for (int r=0;r < nparams;r++)
        {
          data->rhs->at (r) = bfrhs[r];

          for (int c=0;c < nparams;c++)
            data->lhs->at (r,c) = bflhs[r*nparams+c];
        }
        //std:: cout <<"LevMarFitterV2.cpp: Begin solve()..." << endl;
        * (data->delta) = solve (* (data->lhs),* (data->rhs));
      }

      bool NaN_detected = false;
      for (int i=0;i < nparams;i++)
      {
        // test for NaN
        if (data->delta->at (i) != data->delta->at (i))
        {
          NaN_detected = true;
          data->delta->at (i) = 0;
        }

        // adjust parameter from current baseline
        params_new[i] = params[i] + data->delta->at (i);
      }
      // make this virtual in case constraints need to be non-orthogonal boxes
      ApplyMoveConstraints (params_new);
      // apply limits if necessary
      //if (param_max) params_new[i] = (params_new[i] > param_max[i] ? param_max[i] : params_new[i]);
      //if (param_min) params_new[i] = (params_new[i] < param_min[i] ? param_min[i] : params_new[i]);

      if (!NaN_detected)
      {
        // re-calculate error
        Evaluate (tmp,params_new);

        // calculate error bw function and data
        r_trial = CalcResidual (y, tmp, npts, &err_vect[0]);
        r_trial += EvaluateParamFromPrior (params_new);
        r_chg = r_trial - r_start;
      }

      if (!NaN_detected && (r_trial < r_start))
      {
        lambda /= LEVMAR_STEP_V2;
        if (lambda < FLT_MIN) lambda = FLT_MIN;
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
    catch (std::runtime_error le)
    {
      // a failed solution of the matrix should be treated just like a failed attempt
      // at improving the fit...increase lambda and try again
      data->delta->set_size (nparams);
      data->delta->zeros (nparams);
      lambda *= LEVMAR_STEP_V2;
      numException++;
      //std::cout <<"LevMarFitterV2.cpp: Fit - exception runtime ..." << endl;
    }

    // we need a way to bail out of this loop if lambda continues to increase but we never
    // get a better fit
    if (lambda > lambda_threshold)
      cont_proc = true;

    if (debug_trace)
      printf ("lambda = %f, done = %d\n",lambda,done_cnt);
  }
}


void LevMarFitterV2::CalculateJTJ (double *bfjtj, float *bfjac)
{
  // calculate jtj matrix
  for (int r=0;r < nparams;r++)
    for (int c=r;c < nparams;c++)
      bfjtj[r*nparams+c] = bfjtj[c*nparams+r] = cblas_sdot (npts,&bfjac[r],nparams,&bfjac[c],nparams);

  // adjust matrix for priors
  for (int r=0; r<nparams; r++)
  {
    // diagonal terms only
    bfjtj[r*nparams+r] += dampers[r]*dampers[r]; // damping not squared - this is >scaling< derivative automatically 1
  }
}

void LevMarFitterV2::CalculateRHS (double *bfrhs,
                                   float *bfjac,
                                   float *err_vect,
                                   float *params)
{
  // calculate rhs
  //Blas_Mat_Trans_Vec_Mult(jac,err_vect,rhs,1.0,0.0);
  for (int r=0;r < nparams;r++)
    bfrhs[r] = cblas_sdot (npts,&bfjac[r],nparams,&err_vect[0],1);


  // adjust vector for priors
  for (int r=0; r<nparams; r++)
  {
    bfrhs[r] += dampers[r]*dampers[r]* (prior[r]-params[r]);
  }
}
void LevMarFitterV2::SetLambdaThreshold (float _lambda_threshold)
{
  lambda_threshold = _lambda_threshold;
}

void LevMarFitterV2::SetLambdaStart (float _initial_lambda)
{
  lambda = _initial_lambda;
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
// set prior values for parameters
void LevMarFitterV2::SetPrior (float *input_prior)
{
  for (int j=0; j<nparams; j++)
    prior[j] = input_prior[j];
}
// dampening effect of prior for each parameter
void LevMarFitterV2::SetDampers (float *input_dampers)
{
  for (int j=0; j<nparams; j++)
    dampers[j] = input_dampers[j];
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
  delete [] prior;
  delete [] dampers;
  delete [] fval_cache;
  if (data != NULL)
    delete data;
  delete[] param_val;
  delete[] param_max;
  delete[] param_min;
}
