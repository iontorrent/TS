/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODSINGLEFLOWFITKRATE_H
#define BKGMODSINGLEFLOWFITKRATE_H

#include "LevMarFitterV2.h"
#include "MathOptim.h"
#include "DiffEqModel.h"

#define n_to_uM_conv (0.000062f)
#define DO_MESH_INTERPOLATION 0

struct BkgModSingleFlowFitKrateParams
{
  float Ampl;
  float kmult;
  float dmultX;
};

class Interpolator
{
  public:
    float *scratch;
    float *Triple[3];
    int npts;
    float x[3];
    float y[3];
    float bary_c[3];
    Interpolator()
    {
      scratch=NULL;
      for (int i=0; i<3; i++)
      {
        Triple[i]=NULL;
      };
      npts = 0;
      for (int i=0; i<3; i++)
      {
        x[i]=y[i]=bary_c[i] = 0;
      }
    };
    void Allocate (int _npts)
    {
      npts = _npts;
      scratch = new float [npts *3];
      for (int i=0; i<3; i++)
        Triple[i] = &scratch[i*npts];
    };
    void Delete()
    {
      if (scratch!=NULL) delete[] scratch;
      for (int i=0; i<3; i++)
        Triple[i] = NULL;
    }
    ~Interpolator()
    {
      Delete();
    };
    void LoadPoint (int which_pt, float tx, float ty, float *vals)
    {
      x[which_pt] = tx;
      y[which_pt] = ty;
      for (int i=0; i<npts; i++)
        Triple[which_pt][i] = vals[i];
    }
    bool Check (float tx, float ty)
    {
      float tc[3];
      Bary_Coord (tc,tx,ty,x,y);
      if (tc[0]<0 || tc[1]<0 || tc[2]<0)
        return (false);
      return (true);
    }
    void Interpolate (float *output, float tx, float ty)
    {
      float tc[3];
      Bary_Coord (tc, tx,ty, x, y);
      //printf("%f\t%f\t%f\n", tc[0],tc[1],tc[2]);
      Bary_Interpolate (output,Triple,tc, npts);
    }
};




class BkgModSingleFlowFitKrate : public LevMarFitterV2
{
    // evaluate the fitted function w/ the specified parameters
  public:

    // constructor
    BkgModSingleFlowFitKrate (int _len,float *frameNumber,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss, bool _fit_d = false)
    {
      ivalPtr = new float[_len];
      xvals = frameNumber;
      deltaFrame = _deltaFrame;
      deltaFrameSeconds = _deltaFrameSeconds;
      fit_d = _fit_d;
      c_dntp_top_pc = NULL;
      math_poiss = _math_poiss;
      if (_fit_d)
        Initialize (3,_len,xvals);
      else
        Initialize (2,_len,xvals);
      oldstyle = true;
      // set up interpolation
      if (DO_MESH_INTERPOLATION>0)
      {
        for (int i=0; i<4; i++)
          my_guess[i].Allocate (_len);
      }
    }
    // useful for evaluating sum-of-squares difference without invoking the full Lev-Mar
    void SetJustAmplitude (float A)
    {
      params.Ampl=A;
    }

    // passes in params and reg_params pointers for the well/region we are fitting
    // also indicates which flow we are fitting
    // Not safe!!!!! Damages _fg structure without warning!!!!
    void SetWellRegionParams (struct bead_params *_p,struct reg_params *_rp,int _fnum,
                              int _nnum,int _flow,
                              int _i_start,float *_c_dntp_top)
    {
      p = _p;
      reg_p = _rp;
      params.Ampl = p->Ampl[_fnum];
      params.kmult = p->kmult[_fnum];
      params.dmultX = 1.0;
      fnum = _fnum;
      NucID = _nnum;
      flow = _flow;

      // Copies modifies the allowed kmult
      float decision_threshold = 2.0;
      max_offset = 1.0; // start at 1.0 when 0 amplitude
      max_slope = (param_max[1]-1) / (decision_threshold/p->Copies);
      min_offset = 1.0; // start at 1.0 when 0 amplitude
      min_slope = (param_min[1]-1) / (decision_threshold/p->Copies);

      // since this uses a library function..and the parameters involved aren't fit
      // it's helpful to compute this once and not in the model function
      SP = (float) (COPYMULTIPLIER * p->Copies) *pow (reg_p->CopyDrift,flow);
      etbR = AdjustEmptyToBeadRatioForFlow (p->R,reg_p,NucID,flow);
      tauB = ComputeTauBfromEmptyUsingRegionLinearModel (reg_p,etbR);

      sens = reg_p->sens*SENSMULTIPLIER;

      // it is now necessary to make a separate call to
      i_start = _i_start;
      c_dntp_top_pc = _c_dntp_top;

      if (DO_MESH_INTERPOLATION>0)
      {
        // interpolation:: set up square
        float tval[len];
        float mid = params.Ampl;
        float top = mid+1.0;
        float bot = mid/2.0;
        if (bot<mid-1.0)
          bot = mid-1.0;
        float left = 0.6;
        float right = 1.8;

        SingleFlowIncorporationTrace (top,left,1.0,tval);
        my_guess[3].LoadPoint (1,top,left,tval);
        my_guess[0].LoadPoint (0,top,left,tval);
        SingleFlowIncorporationTrace (top,right,1.0,tval);
        my_guess[0].LoadPoint (1,top,right,tval);
        my_guess[1].LoadPoint (0,top,right,tval);
        SingleFlowIncorporationTrace (bot,right,1.0,tval);
        my_guess[1].LoadPoint (1,bot,right,tval);
        my_guess[2].LoadPoint (0,bot,right,tval);
        SingleFlowIncorporationTrace (bot,left,1.0,tval);
        my_guess[2].LoadPoint (1,bot,left,tval);
        my_guess[3].LoadPoint (0,bot,left,tval);
        SingleFlowIncorporationTrace (mid,1.0,1.0,tval);
        my_guess[0].LoadPoint (2,mid,1.0,tval);
        my_guess[1].LoadPoint (2,mid,1.0,tval);
        my_guess[2].LoadPoint (2,mid,1.0,tval);
        my_guess[3].LoadPoint (2,mid,1.0,tval);
      }
    }


    // optionally set maximum value for parameters
    void SetParamMax (BkgModSingleFlowFitKrateParams _max_params)
    {
      max_params = _max_params;
      LevMarFitterV2::SetParamMax ( (float *) &max_params);
    }

    // optionally set minimum value for parameters
    void SetParamMin (BkgModSingleFlowFitKrateParams _min_params)
    {
      min_params = _min_params;
      LevMarFitterV2::SetParamMin ( (float *) &min_params);
    }

    // entry point for grid search
    void GridSearch (int steps,float *y)
    {
      LevMarFitterV2::GridSearch (steps,y, (float *) (&params));
    }

    // entry point for fitting
    virtual int Fit (int max_iter,float *y)
    {
      return (LevMarFitterV2::Fit (max_iter,y, (float *) (&params)));
    }

    // the starting point and end point of the fit
    BkgModSingleFlowFitKrateParams params;

    ~BkgModSingleFlowFitKrate()
    {
      delete [] ivalPtr;
    }

    // evaluates the function using the values in params
    virtual void Evaluate (float *y)
    {
      Evaluate (y, (float *) (&params));
    }

    void SetNewKmultBox (bool _oldstyle)
    {
      oldstyle = _oldstyle;
    };


  protected:
    virtual void ApplyMoveConstraints (float *params_new)
    {

      // it's a special case here
      // parameter 0 is amplitude
      // parameter 1 is kmult
      if (params_new[0]>param_max[0]) params_new[0] = param_max[0];
      if (params_new[0]<param_min[0]) params_new[0] = param_min[0];
      if (oldstyle)
      {
        if (params_new[1]>param_max[1]) params_new[1] = param_max[1];
        if (params_new[1]<param_min[1]) params_new[1] = param_min[1];
      }
      else
      {
        // so if we fit a larger amplitude, the allowed offset increases
        float kmult_max, kmult_min;
        // smoothly transition between no kmult variation and some kmult variation
        kmult_max = params_new[0]*max_slope+max_offset;
        kmult_min = params_new[0]*min_slope+min_offset;
        if (kmult_max>param_max[1]) kmult_max = param_max[1];
        if (kmult_max<1.0) kmult_max = 1.0;
        if (kmult_min<param_min[1]) kmult_min = param_min[1];
        if (kmult_min>1.0) kmult_min = 1.0;

        // now apply
        if (params_new[1]>kmult_max) params_new[1] = kmult_max;
        if (params_new[1]<kmult_min) params_new[1] = kmult_min;
      }
    }

    virtual void Evaluate (float *y, float *params)
    {
      // calculate proton flux and decay
      if (DO_MESH_INTERPOLATION==0)
      {
        if (fit_d)
          SingleFlowIncorporationTrace (params[0],params[1],params[2],y);
        else
          SingleFlowIncorporationTrace (params[0],params[1],1.0,y);
      }
      else
      {
        bool did_it = false;
        for (int i=0;  i<4; i++)
        {
          if (my_guess[i].Check (params[0],params[1]))
          {
            did_it=true;
            my_guess[i].Interpolate (y, params[0], params[1]);
          }
        }
        if (!did_it)
        {
          if (params[0]>my_guess[0].x[2])
            my_guess[0].Interpolate (y,params[0],params[1]);
          else
            my_guess[2].Interpolate (y,params[0],params[1]);
        }
        /*float ty[len];
        SingleFlowIncorporationTrace(params[0],params[1],1.0,ty);
        for (int i=0; i<len; i++)
          printf("%d\t%f\t%f\t%f\t%f\n", i, y[i],ty[i],params[0],params[1]);*/
      }
    }

    // loop exit condition
    virtual bool DoneTest (int iter,int max_iter,LaVectorDouble *delta,float lambda,int &done_cnt,float residual,float r_chg)
    {
      (void) done_cnt;
      (void) residual;
      (void) r_chg;
      (void) lambda;

      if ( (*delta) (0) * (*delta) (0) < 0.0000025) done_cnt++;
      else done_cnt = 0;

      return ( (iter >= max_iter) || (done_cnt > 1));
    }

    PoissonCDFApproxMemo *math_poiss;
  private:
    Interpolator my_guess[4];
    // pointers to the BkgModel parameters for the well/region being fit
    struct bead_params *p;
    struct reg_params *reg_p;

    // single-flow proton flux
    float *ivalPtr;
    // single-flow delta-T between data points
    float *deltaFrame;
    float *deltaFrameSeconds;

    int fnum;
    // which nucleotide was in the flow
    int NucID;
    // global flow number (0-total number of flows in the run) ...needed for droop terms
    int flow;

    // pre-computed c_dntp_top...speedup
    float *c_dntp_top_pc;
    // pre-computed values
    float SP,etbR,tauB,sens;
    int i_start;

    // kmult special constraints
    float max_slope, max_offset;
    float min_slope, min_offset;
    bool oldstyle;


    void SingleFlowIncorporationTrace (float A,float kmult,float dmultX,float *fval)
    {
      (void) dmultX;

      float d;
      float kr,kmax;
      float C;

      d = reg_p->d[NucID]*p->dmult;
      // initialize diffusion/reaction simulation for this flow
      kr = reg_p->krate[NucID]*kmult;
      kmax = reg_p->kmax[NucID];
      C = reg_p->nuc_shape.C;

      ComputeCumulativeIncorporationHydrogens (ivalPtr, len, deltaFrameSeconds, c_dntp_top_pc, ISIG_SUB_STEPS_SINGLE_FLOW, i_start,  C, A, SP, kr, kmax, d,math_poiss);
      MultiplyVectorByScalar (ivalPtr, sens,len); // transform hydrogens to signal       // variables used for solving background signal shape
      RedSolveHydrogenFlowInWell (fval,ivalPtr,len,i_start,deltaFrame,tauB);
      MultiplyVectorByScalar (fval,p->gain,len);
    }

// COMMENT:  I HATE IFDEFS, BUT AM USING THEM ANYWAY TO DISTINGUISH THE TWO EXPERIMENTAL FUNCTIONS
// @TODO: GET RID OF THIS AS SOON AS POSSIBLE
#ifdef USEDUALTRACE
    // combined version
    // computes trace and derivatives in single function using dual numbers
     void MakeJacobian(float *bfjac,float *params, float *fval)
    {
            float d;
      float kr,kmax;
      float DA[100],DK[100]; // plenty of room, nonsense

      d = reg_p->d[NucID]*p->dmult;
      // initialize diffusion/reaction simulation for this flow
      kr = reg_p->krate[NucID]*params[1];
      kmax = reg_p->kmax[NucID];

      Dual tA(params[0],1,0);
      Dual tK(kr,0,reg_p->krate[NucID]);

      DerivativeRedTrace(fval, ivalPtr,DA,DK,len,deltaFrameSeconds, deltaFrame,c_dntp_top_pc,ISIG_SUB_STEPS_SINGLE_FLOW, i_start, tA,SP,tK,kmax,d, sens,p->gain, tauB,math_poiss);
      int i=0;
    for (int j=0;j < len;j++)
          bfjac[j*nparams+i] = residualWeight[j] *DA[j];
          //fval contains first derivative da

      i=1;
    for (int j=0;j < len;j++)
          bfjac[j*nparams+i] = residualWeight[j] *DK[j];
    // fval contains first derivative dkmult

    }
#endif

#ifdef USEDUALDERIVATIVE
    // computes incorporation and derivative of incorporation
    // applies diffeq to recover derivative of trace
    void MakeJacobian(float *bfjac,float *params, float *fval)
    {
            float d;
      float kr,kmax;
      float DA[100],DK[100]; // plenty of room, nonsense

      d = reg_p->d[NucID]*p->dmult;
      // initialize diffusion/reaction simulation for this flow
      kr = reg_p->krate[NucID]*params[1];
      kmax = reg_p->kmax[NucID];

      Dual tA(params[0],1,0);
      Dual tK(kr,0,reg_p->krate[NucID]);

      DerivativeComputeCumulativeIncorporationHydrogens (ivalPtr, DA, DK, len, deltaFrameSeconds, c_dntp_top_pc, ISIG_SUB_STEPS_SINGLE_FLOW, i_start, tA, SP, tK, kmax, d,math_poiss);
      //MultiplyVectorByScalar (ivalPtr, sens,len); // transform hydrogens to signal       // variables used for solving background signal shape
      MultiplyVectorByScalar (DA, sens,len); // transform hydrogens to signal       // variables used for solving background signal shape
       MultiplyVectorByScalar (DK, sens,len); // transform hydrogens to signal       // variables used for solving background signal shape
     //RedSolveHydrogenFlowInWell (fval,ivalPtr,len,i_start,deltaFrame,tauB);
      //MultiplyVectorByScalar (fval,p->gain,len);
      // fval contains function here
     RedSolveHydrogenFlowInWell (fval,DA,len,i_start,deltaFrame,tauB);
      MultiplyVectorByScalar (fval,p->gain,len);
      int i=0;
    for (int j=0;j < len;j++)
          bfjac[j*nparams+i] = residualWeight[j] *fval[j];
          //fval contains first derivative da
     RedSolveHydrogenFlowInWell (fval,DK,len,i_start,deltaFrame,tauB);
      MultiplyVectorByScalar (fval,p->gain,len);
      i=1;
    for (int j=0;j < len;j++)
          bfjac[j*nparams+i] = residualWeight[j] *fval[j];
    // fval contains first derivative dkmult
      
    }
#endif

/*    void ApproximateAmplitudeDerivative(float *fval, float A, float kmult, float dx)
    {
      // ivalPtr pre-existing
      float mdx = (A+dx)/A;
      MultiplyVectorByScalar(ivalPtr, mdx,len);
      RedSolveHydrogenFlowInWell(fval,ivalPtr,len,i_start,deltaFrame,tauB);
      MultiplyVectorByScalar(fval,p->gain,len);
      MultiplyVectorByScalar(ivalPtr, 1.0/mdx,len); // dumb reversion
      //SingleFlowIncorporationTrace(A+dx,kmult,1.0,fval);
    }

    void ApproximateKmultDerivative(float *fval, float A, float kmult,float dx)
    {
      // note side effects need to be removed!!!!
      float kdx = (kmult+dx)/kmult; // "running quicker" if dx>0, which is what we want
      StretchTime(ivalPtr,kdx,xvals,i_start,len);
      RedSolveHydrogenFlowInWell(fval,ivalPtr,len,i_start,deltaFrame,tauB);
      MultiplyVectorByScalar(fval,p->gain,len);
      //SingleFlowIncorporationTrace(A,kmult+dx,1.0,fval);
    }
    

       // allow replacement
    void MakeJacobian (float *bfjac, float *params, float *fval)
    {
      float tmp[len];

      // evaluate the partial derivatives w.r.t. each parameter
      int i=0;
      ApproximateAmplitudeDerivative(tmp,params[0],params[1], dp[i]);
              // store in jacobian
        for (int j=0;j < len;j++)
          bfjac[j*nparams+i] = residualWeight[j] * (tmp[j]-fval[j]) / dp[i];
      i=1;
      ApproximateKmultDerivative(tmp,params[0], params[1],dp[i]);
        for (int j=0;j < len;j++)
          bfjac[j*nparams+i] = residualWeight[j] * (tmp[j]-fval[j]) / dp[i];
    }
// st>1.0
void StretchTime(float *vals, float st, float *timeFrame, int frame_zero, int len)
{
  float tz = timeFrame[frame_zero];
  for (int i=frame_zero; i<len-1; i++)
  {

    float t = st*(timeFrame[i]-tz)+tz;
    // assume we're small enough to not shift between frames
    float delta = (t-timeFrame[i])/(timeFrame[i+1]-timeFrame[i]);
    vals[i] = (1-delta)*vals[i]+delta*vals[i+1];
  }
}*/
    float *xvals;
    BkgModSingleFlowFitKrateParams min_params,max_params;
    bool fit_d;
};



#endif // BKGMODSINGLEFLOWFITKRATE_H
