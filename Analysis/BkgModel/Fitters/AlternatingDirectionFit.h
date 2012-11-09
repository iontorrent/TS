/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ALTERNATINGDIRECTIONFIT_H
#define ALTERNATINGDIRECTIONFIT_H

#include "TraceCurry.h"
// need this to dynamically allocate vectors
#include "EmphasisVector.h"



class AlternatingDirectionOneFlow
{
    // evaluate the fitted function w/ the specified parameters
  public:
    // the starting point and end point of the fit
    float *paramA;
    float *max_paramA, *min_paramA;
    int nparams;


    float *fval_cache;
    float *step_cache;
    float *delta_fval;
    float *cur_resid;
    float *residual_weight;
    float wtScale;
    float residual;
    int npts;
    bool local_krate_fit;
    float local_krate_fit_decision_threshold;
    float expand_rate;

    EmphasisClass *emphasis_ptr;


    TraceCurry calc_trace;

    float my_project_vector ( float *X, float *Y, float *em, int len )
    {
      float numX,denX;
      numX=denX = 0.0f;
      denX = 0.0001f; // safety catch to prevent nans
      for ( int i=0; i<len; i++ )
      {
        numX += X[i]*Y[i]*em[i]*em[i];
        denX += Y[i]*Y[i]*em[i]*em[i];
      }
      return ( numX/denX ); // throws very large value if something bad happens to model trace
    };
    // set the weighting vector
    void SetWeightVector ( float *vect )
    {

      for ( int i=0;i < npts;i++ )
      {
        residual_weight[i] = vect[i];
      }
      ComputeWeightScale();
    }
    
    void ComputeWeightScale()
    {
      wtScale = 0.0f;
      for (int i=0; i<npts; i++)
        wtScale += residual_weight[i]*residual_weight[i];
    }

    // does the fit using a completely different (and faster) alternative
    void AlternatingDirection ( float *y, bool do_krate )
    {
      float lambda = 1.01; // made no noticeable progress
      bool seeking_convergence = true;
      local_krate_fit = do_krate;

      // one hit of compute to start us off
      calc_trace.SingleFlowIncorporationTrace ( paramA[0],paramA[1],fval_cache );

      SetResiduals ( y ,fval_cache );
      float r_level = QuickResiduals();
      float x_level = r_level;
      int defend_against_infinity = 0;
      while ( seeking_convergence )
      {

        x_level = DoOneStep ( y,0 ,x_level);

        if ( local_krate_fit )
          x_level = DoOneStep ( y,1,x_level );

        if ( x_level*lambda>r_level ) // check after doing both steps only to exit the loop if we didn't drop quickly enough
          seeking_convergence=false;
        else
          r_level = x_level;
        
        // I hate guarding against loops, but NANs can happen anywhere
        defend_against_infinity++;
        if (defend_against_infinity>50)
        {
          // usually means something very bad in the input data
          printf("ALERT: fail to defend against infinity %d %d %d\n", calc_trace.p->x, calc_trace.p->y,calc_trace.flow);
          seeking_convergence = false;
        }
      }

    }

    float QuickResiduals()
    {
      float r=0;
      for ( int j=0; j<npts; j++ )
        r += cur_resid[j]*cur_resid[j]*residual_weight[j]*residual_weight[j];
      r /= wtScale;
      return ( r );
    }

    void ApplyConstraints ( float *tmp_param )
    {
      for ( int i=0; i<nparams; i++ )
      {
        if ( tmp_param[i]<min_paramA[i] )
          tmp_param[i] = min_paramA[i];
        if ( tmp_param[i]>max_paramA[i] )
          tmp_param[i] = max_paramA[i];
      }
    }

    void DynamicEmphasis ( float control_val )
    {
      if ( emphasis_ptr!=NULL )
        emphasis_ptr->CustomEmphasis ( residual_weight,control_val );
      ComputeWeightScale();
    }
    
    void FakeEmphasis(float cutoff)
    {
      float tval;
      float top = 0.0f;
      int halfmax = 0;
       // use the current incorporation as our weighting factor
       for (int i=0; i<npts; i++)
       {
         tval = abs(fval_cache[i]);
         if (tval>=top)
         {
           top = tval;
           halfmax = i; // advance while we rise
         }
         if (tval>=top*cutoff)
           halfmax = i; // advance until we drop off
         
       }
       for (int i=0; i<halfmax; i++)
         residual_weight[i] = 1.0f;
       for (int i=halfmax; i<npts; i++)
         residual_weight[i] = 0.001f; // pretend to be zero but not quite in case of trouble 
       ComputeWeightScale();
    }

    void DynamicConstraintKrate ( float *tmp_param )
    {
      // dynamic choice of kmult
      float magic = local_krate_fit_decision_threshold/calc_trace.p->Copies;
      float thresh = tmp_param[0]>0.0f ? tmp_param[0] : 0.0f;

      // obviously, at threshold/magic =1, the bounds are equal and 1
      // as we increase above this, the lower bound drops smoothly towards zero, allowing us to fit varying rates
      float lower_bound = ( 1.0f+expand_rate ) *magic/ ( magic+expand_rate*thresh );
      float upper_bound = 1.0f/lower_bound;
      if ( lower_bound>1.0f )
      {
        tmp_param[1] = 1.0f;
        local_krate_fit = false;
      }
      else
      {
        if ( tmp_param[1]>upper_bound )
          tmp_param[1] = upper_bound;
        if ( tmp_param[1]<lower_bound )
          tmp_param[1]=lower_bound;
        local_krate_fit = true;
      }
    }

    void ApplyDynamicConstraints ( float *tmp_param )
    {
      // kmult depends on the scale of the amplitude as well as the copy-count
      // which should really be signal/scale
      int i=0; // Amplitude first
      if ( tmp_param[i]<min_paramA[i] )
        tmp_param[i] = min_paramA[i];
      if ( tmp_param[i]>max_paramA[i] )
        tmp_param[i] = max_paramA[i];

      // choose krate fit bounds and type based on current amplitude
      DynamicConstraintKrate ( tmp_param );
      DynamicEmphasis ( tmp_param[0] );
      //FakeEmphasis(0.37);
      //ReweightHuber(2000);
    }
    
    void ReweightHuber(float huber_val)
    {
      // adjust current residual weights by a discounting factor
      for (int i=0; i<npts; i++)
        residual_weight[i] *= 1.0f/sqrt(sqrt(1+cur_resid[i]*cur_resid[i]/huber_val));  // like L1 for large residuals, L2 for small
      ComputeWeightScale();
    }

    void CopyParam ( float *sink, float *source )
    {
      for ( int i=0; i<nparams; i++ )
        sink[i] = source[i];
    }

    float DoOneStep ( float *y, int which_param , float start_res )
    {
      float tmp_param[npts];
      FindStepDirection ( which_param );
      float cur_scale=BasicStepScale();
      CopyParam ( tmp_param,paramA );
      tmp_param[which_param] += cur_scale;
      //ApplyConstraints ( tmp_param );
      ApplyDynamicConstraints ( tmp_param );
      calc_trace.SingleFlowIncorporationTrace ( tmp_param[0],tmp_param[1],step_cache );
      SetResiduals ( y ,step_cache );
      float qr = QuickResiduals();
      float retval;
      if ( qr<start_res )
      {
        CopyParam ( paramA,tmp_param );
        memcpy ( fval_cache,step_cache,sizeof ( float[npts] ) );
        retval=qr;
      }
      else
      {
        SetResiduals ( y,fval_cache ); // revert residuals
        ApplyDynamicConstraints(paramA); // revert emphasis as well
        // fail to descend, revert
        retval = start_res;
      }
      return(retval);
    }

    void FindStepDirection ( int which_param ) // step downwards along this direction
    {
      // fval_cache contains the current evauation
      float tmp_param[nparams];
      float epsilon = 0.01;

      CopyParam ( tmp_param,paramA );
      if ( (tmp_param[which_param]+epsilon)>max_paramA[which_param] )
        epsilon = -1.0f* epsilon;
      
      tmp_param[which_param] += epsilon;  // can blow out at maximum point

      calc_trace.SingleFlowIncorporationTrace ( tmp_param[0],tmp_param[1],delta_fval );
      for ( int i=0; i<npts; i++ )
        delta_fval[i] -= fval_cache[i];
      for ( int i=0; i<npts; i++ )
        delta_fval[i] /= epsilon;
    }

    void SetResiduals ( float *y , float *my_cache )
    {
      for ( int i=0; i<npts; i++ )
        cur_resid[i] = y[i]-my_cache[i];
    }

    float BasicStepScale()
    {
      return ( my_project_vector ( cur_resid,delta_fval,residual_weight, npts ) );
    }


    AlternatingDirectionOneFlow ( int _len, float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss )
    {
      calc_trace.Allocate ( _len,_deltaFrame, _deltaFrameSeconds, _math_poiss );
      Initialize ( _len, 2 );
      Defaults();
    }

    void Defaults()
    {
      paramA[0] = 0.5;
      paramA[1] = 1.0;
      min_paramA[0] = 0.001;
      min_paramA[1] = 0.65;
      max_paramA[0] = 9.9;
      max_paramA[1] = 1.75;
    }

    void SetMin ( float *_min_param )
    {
      for ( int i=0; i<2; i++ )
        min_paramA[i] = _min_param[i];
    }

    void SetMax ( float *_max_param )
    {
      for ( int i=0; i<2; i++ )
        max_paramA[i] = _max_param[i];
    }

    void Initialize ( int _npts, int _nparams )
    {
      npts = _npts;
      nparams = _nparams;
      fval_cache = new float[npts];
      step_cache = new float[npts];
      delta_fval = new float[npts];
      cur_resid = new float [npts];
      residual_weight = new float[npts];
      paramA = new float[nparams];
      min_paramA = new float [nparams];
      max_paramA = new float [nparams];
      residual = 0.0f;
      wtScale = 0.0f;
      local_krate_fit = false;
      local_krate_fit_decision_threshold = 2.0f;
      expand_rate = 1.0f;
      emphasis_ptr = NULL;
    }

// get the mean squared error after the fit
    float  GetMeanSquaredError ( float *y,bool use_fval_cache )
    {
      float *tmp;
      float tmp_eval[npts];

      if ( use_fval_cache )
      {
        tmp = fval_cache;
      }
      else
      {
        calc_trace.SingleFlowIncorporationTrace ( paramA[0],paramA[1],tmp_eval );
        tmp = tmp_eval;
      }
      residual = CalcResidual ( y, tmp );
      return residual;
    }
    
    void ReturnPredicted(float *f_predict, bool use_fval_cache)
    {
      if (f_predict!=NULL)
      {
        if (use_fval_cache)
          memcpy(f_predict, fval_cache, sizeof(float[npts]));
        else
          calc_trace.SingleFlowIncorporationTrace(paramA[0],paramA[1],f_predict);
      }
    }



    float CalcResidual ( float *y, float *tmp )
    {
      float r= 0.0f;
      float e;
      for ( int i=0;i <npts;i++ )
      {
        e = residual_weight[i] * ( y[i]-tmp[i] );
        r += e*e;
      }

      // r = sqrt(r/wtScale)
      r = ( r/wtScale );
      return r;
    }


    ~AlternatingDirectionOneFlow()
    {
      delete[] paramA;
      delete[] min_paramA;
      delete[] max_paramA;
      delete[] fval_cache;
      delete[] step_cache;
      delete[] cur_resid;
      delete[] delta_fval;
      delete[] residual_weight;
    }
};

#endif // ALTERNATINGDIRECTIONFIT_H
