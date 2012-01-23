/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LEVMARFITTERV2_H
#define LEVMARFITTERV2_H

#include <string.h>
#include <stdlib.h>
#include <float.h>

// some of the code uses <complex.h>, and in <complex.h> 'I' is defined and this 
// interferes w/ lapackpp.  I undef it here in case anyone above has included <complex.h>
#undef I

#include <lapackpp.h>

// base class for fitting algorithms
class LevMarFitterV2
{
protected:
	LevMarFitterV2(): len(0), x(NULL), residual(0.0), dp(NULL), residualWeight(NULL),
        wtScale(0.0), nparams(0), param_max(NULL), param_min(NULL), debug_trace(false), 
		lambda(1.0), lambda_threshold(1.0E+10)
    {
    }

    ~LevMarFitterV2() {
        delete [] residualWeight;
        delete [] dp;
        delete [] prior;
        delete [] dampers;
    }

    void Initialize(int _nparams,int _len,float *_x)
    {
        nparams = _nparams;
        len = _len;
        x = _x;

        // make all matricies the correct size
        jac.resize(len,nparams);
        jtj.resize(nparams,nparams);
        lhs.resize(nparams,nparams);
        rhs.resize(nparams);
        delta.resize(nparams);
//        err_vect.resize(len);

        dp = new float[_nparams];
        for(int i=0;i<_nparams;i++)
            dp[i] = 0.001;

        residualWeight = new float[len];
        wtScale = len;
        for(int i=0;i<len;i++)
            residualWeight[i] = 1.0f;
        
        prior = new float [nparams];
        for (int i=0; i<nparams; i++)
          prior[i] = 0;
        dampers = new float [nparams];
        for (int i=0; i<nparams; i++)
          dampers[i] = 0.0; // no response to priors
        
    }

    void SetParamMax(float *_param_max) {param_max = _param_max;}
    void SetParamMin(float *_param_min) {param_min = _param_min;}

//    virtual float CalcResidual(float *refVals, float *testVals, int numVals, LaVectorDouble *err_vec = NULL) {
    virtual float CalcResidual(float *refVals, float *testVals, int numVals, float *err_vec = NULL) {
        double r = 0.0;
        double e;

        if (err_vec)
            for (int i=0;i < numVals;i++){
//                e = (*err_vec)(i) = residualWeight[i] * (refVals[i] - testVals[i]);
                err_vec[i] = e = residualWeight[i] * (refVals[i] - testVals[i]);
                r += e*e;
            } 
        else
            for (int i=0;i < numVals;i++){
                e = residualWeight[i] * (refVals[i] - testVals[i]);
                r += e*e;
            }

        // r = sqrt(r/wtScale) 
        r = (r/wtScale); 
        return r;
    }

    virtual bool DoneTest(int iter,int max_iter,LaVectorDouble *delta,float lambda,int &done_cnt,float residual,float r_chg)
    {
        (void)residual;
        (void)delta;
        (void)lambda;

        if (r_chg < 0.0) done_cnt = 0;
        else done_cnt++;

        return((iter >= max_iter) || (done_cnt >= 5));
    }

    // generic grid search
    void GridSearch(int steps,float *y,float *params)
    {
        // if we were never configured with min/max parameter values, just return
        // we only allow grid search if the region in which to search has been defined
        if ((param_min == NULL) || (param_max == NULL))
            return;

        float param_step[nparams];
        float p[nparams];

        // calculate step size for each parameter
        for (int pn=0;pn < nparams;pn++)
        {
            param_step[pn] = (param_max[pn] - param_min[pn])/(steps-1);
            p[pn] = param_min[pn];
        }

        int step_num[nparams];
        memset(step_num,0,sizeof(step_num));

        int inum;
        int total_steps;
        int digit;
        float *fval = new float[len];
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

            Evaluate(fval,p);
            r1 = CalcResidual(y, fval, len);

            if (r1 < min_err)
            {
                memcpy(params,p,sizeof(p));
                min_err = r1;
            }
        }

        delete [] fval;
    }
    
    virtual void ApplyMoveConstraints(float *params_new)
    {
        for (int i=0; i< nparams; i++)
        {
                          // apply limits if necessary
          if (param_max) params_new[i] = (params_new[i] > param_max[i] ? param_max[i] : params_new[i]);
          if (param_min) params_new[i] = (params_new[i] < param_min[i] ? param_min[i] : params_new[i]);
        }
    }
    // 
    float EvaluateParamFromPrior(float *param_new)
    {
      float sum=0;
      for (int j=0; j<nparams; j++)
        sum += dampers[j]*(param_new[j]-prior[j])*(param_new[j]-prior[j]);
      return(sum);
    }
    // main fit entry-point
    int Fit(int max_iter, float *y, float *params, float *std_err = NULL)
    {
        int iter=0;
        int done_cnt = 0;
        float fval[len];
        float tmp[len];
        float params_new[nparams];
        float r_start = FLT_MAX;
        float r_chg = 0.0;
        float r_trial;
        float bfjac[len*nparams];
        float err_vect[len];
        double bfjtj[nparams*nparams];
        double bflhs[nparams*nparams];
        double bfrhs[nparams];

        delta.resize(nparams);
        delta = 0.0;

        // evaluate where we are
        Evaluate(fval,params);
        r_start = CalcResidual(y, fval, len, &err_vect[0]);
        r_start += EvaluateParamFromPrior(params); // account for prior
        
        while (!DoneTest(iter,max_iter,&delta,lambda,done_cnt,r_start,r_chg))
        {

            // remember the last residual for comparison later
            r_chg = 0.0;

            // evaluate the partial derivatives w.r.t. each parameter
            memcpy(params_new,params,sizeof(float[nparams]));
            for (int i=0;i < nparams;i++)
            {
                // adjust parameter
                params_new[i] += dp[i];

                // re-evaluate function
                Evaluate(tmp,params_new);

                // store in jacobian
                for (int j=0;j < len;j++)
                    bfjac[j*nparams+i] = residualWeight[j] * (tmp[j]-fval[j]) / dp[i];

                // put param back
                params_new[i] = params[i];
            }

            // calculate jtj matrix
            for (int r=0;r < nparams;r++)
                for (int c=r;c < nparams;c++)
                    bfjtj[r*nparams+c] = bfjtj[c*nparams+r] = cblas_sdot(len,&bfjac[r],nparams,&bfjac[c],nparams);

            // calculate rhs
            //Blas_Mat_Trans_Vec_Mult(jac,err_vect,rhs,1.0,0.0);
            for (int r=0;r < nparams;r++)
                bfrhs[r] = cblas_sdot(len,&bfjac[r],nparams,&err_vect[0],1);
            
            // adjust matrix for priors
            for (int r=0; r<nparams; r++)
            {
              // diagonal terms only
              bfjtj[r*nparams+r] += dampers[r]*dampers[r]; // damping not squared - this is >scaling< derivative automatically 1
            } 
            // adjust vector for priors
            for (int r=0; r<nparams; r++)
            {
              bfrhs[r] += dampers[r]*dampers[r]*(prior[r]-params[r]);
            }

            // the following loop saves computation if the first attempt to adjust fails, but subsequent
            // attempts with larger values of lambda would succeed.  In this case, a new value of lambda
            // can be attempted, but the jtj and rhs matricies did not actually change and do not
            // need to be re-computed
            bool cont_proc = false;
            while(!cont_proc)
            {
                // add lambda parameter to jtj to form lhs of matrix equation
                memcpy(bflhs,bfjtj,sizeof(bflhs));
                for (int i=0;i < nparams;i++)
                    bflhs[i*nparams+i] *= (1.0 + lambda);

                // solve for delta
                try {
                    
                    // these special cases handle the relatively trivial 1 and 2 parameter solutions
                    // in a faster way
                    if (nparams == 1)
                        delta(0) = bfrhs[0]/bflhs[0];
                    else if (nparams == 2)
                    {
                        double a,b,c,d,det;
                        a = bflhs[0];
                        b = bflhs[1];
                        c = bflhs[2];
                        d = bflhs[3];
                        det = 1.0 / (a*d - b*c);
                        delta(0) = (d*bfrhs[0]-b*bfrhs[1])*det;
                        delta(1) = (-c*bfrhs[0]+a*bfrhs[1])*det;
                    }
                    else
                    {
                        // since we are doing a bigger matrix...package up everything into
                        // a form the real matrix solver can handle
                        for (int r=0;r < nparams;r++)
                        {
                            rhs(r)=bfrhs[r];
                                
                            for (int c=0;c < nparams;c++)
                                lhs(r,c) = bflhs[r*nparams+c];
                        }
 
                        LaLinearSolve(lhs,delta,rhs);
                    }

                    bool NaN_detected = false;
                    for (int i=0;i < nparams;i++)
                    {
                      // test for NaN
                      if (delta(i) != delta(i))
                      {
                          NaN_detected = true;
                          break;
                      }

                      // adjust parameter
                      params_new[i] += delta(i);
                     }
                      // make this virtual in case constraints need to be non-orthogonal boxes
                      ApplyMoveConstraints(params_new);
                      // apply limits if necessary
                      //if (param_max) params_new[i] = (params_new[i] > param_max[i] ? param_max[i] : params_new[i]);
                      //if (param_min) params_new[i] = (params_new[i] < param_min[i] ? param_min[i] : params_new[i]);
                   
                    if (!NaN_detected)
                    {
                        // re-calculate error
                        Evaluate(tmp,params_new);

                        // calculate error bw function and data
                        r_trial = CalcResidual(y, tmp, len, &err_vect[0]);
                        r_trial += EvaluateParamFromPrior(params_new);
                        r_chg = r_trial - r_start;
                    }

                    if (!NaN_detected && (r_trial < r_start))
                    {
                      lambda /= 10.0;
                      if (lambda < FLT_MIN) lambda = FLT_MIN;
                      memcpy(params,params_new,sizeof(float[nparams]));
                      memcpy(fval,tmp,sizeof(float[len]));
                      cont_proc = true;
                      r_start = r_trial;
                    }
                    else
                    {
                      lambda *= 10.0;
                    }

                    if (debug_trace)
                      printf("lambda = %f, done = %d\n",lambda,done_cnt);

                }
                catch (LaException le) {
                    // a failed solution of the matrix should be treated just like a failed attempt
                    // at improving the fit...increase lambda and try again
                    delta.resize(nparams);
                    delta = 0.0;

                    lambda *= 10.0;
                }

                // we need a way to bail out of this loop if lambda continues to increase but we never
                // get a better fit
                if (lambda > lambda_threshold)
                    cont_proc = true;

                if (debug_trace)
                    printf("lambda = %f, done = %d\n",lambda,done_cnt);
            }

            iter++;
        }

        residual = r_start;

        // compute parameter std errs
        if (std_err)
        {
            LaSymmMatDouble std_err_jtj(nparams, nparams);
            Blas_R1_Update(std_err_jtj, jac, 1.0, 0.0, false);

            LaVectorLongInt piv;
            piv.resize(nparams, nparams);

            LaGenMatDouble std_err_jtj_inv = std_err_jtj;
            LUFactorizeIP(std_err_jtj_inv, piv);
            LaLUInverseIP(std_err_jtj_inv, piv);

            double mrv = (residual * len) / (len - nparams); // mean residual variance
            std_err_jtj_inv *= mrv;

            for (int i=0;i < nparams;i++)
                std_err[i] = sqrt(std_err_jtj_inv(i, i));
        }

        return(iter);
    }

public:

    // evaluate the fitted function w/ the specified parameters
    virtual void Evaluate(float *y) = 0;

    // evaluate the fitted function w/ the specified parameters
    virtual void Evaluate(float *y,float *params) = 0;
    
    // fit the data
    virtual int Fit(int max_iter,float *y) = 0;

    void SetLambdaThreshold(float _lambda_threshold)
    {
        lambda_threshold = _lambda_threshold;
    }

    void SetLambdaStart(float _initial_lambda)
    {
        lambda = _initial_lambda;
    }

    float GetLambda(void)
    {
        return(lambda);
    }

    void SetDebugTrace(bool _enable)
    {
        debug_trace = _enable;
    }

    // set the weighting vector
    void SetWeightVector(float *vect)
    {
        wtScale = 0.0;
        for (int i=0;i < len;i++)
        {    
            residualWeight[i] = vect[i];
            wtScale += vect[i];
        }
    }
    // set prior values for parameters
    void SetPrior(float *input_prior)
    {
      for (int j=0; j<nparams; j++)
        prior[j] = input_prior[j];
    }
    // dampening effect of prior for each parameter
    void SetDampers(float *input_dampers)
    {
      for (int j=0; j<nparams; j++)
        dampers[j] = input_dampers[j];
    }



    // get the mean squared error after the fit
    float GetResidual(float *y = NULL)
    {
        if (y == NULL)
            return residual;
        else
        {
            float tmp[len];
            Evaluate(tmp);
            residual = CalcResidual(y, tmp, len);
            return residual;
        }
    }



protected:
    int len;
    float *x;
    float residual;
    float *dp;
    float *residualWeight;
    float *prior;
    float *dampers;
    float wtScale;
    int nparams;
    float *param_max;
    float *param_min;
    bool  debug_trace;

public:

private:
    float lambda;
    float lambda_threshold;
    LaGenMatDouble  jac;
    LaSymmMatDouble jtj;
    LaSymmMatDouble lhs;
    LaVectorDouble  rhs;
    LaVectorDouble  delta;
};





#endif // LEVMARFITTERV2_H

