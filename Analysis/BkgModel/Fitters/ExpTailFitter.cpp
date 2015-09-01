/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MathOptim.h"
#include "BkgMagicDefines.h"
#include "MathUtil.h"
#include "ExpTailFitter.h"
#include "FitExpDecay.h"

#define MIN_VALID_FLOWS 6

bool ExpTailFitter::ComputeAverageValidTrace(float *avg_trc, float *incorporation_traces,
    BeadParams *p, int npts, float low_A, float hi_A,
    int flow_block_size
  )
{
  int navg = 0;
  memset(avg_trc,0,sizeof(float[npts]));
   for (int fnum=0;fnum < flow_block_size;fnum++)
   {
      // keep out long HPs in order to prevent pollution of data
      // points with additional generated protons
      // keep out non-incorporating flows because they don't contain
      // exponential decay information
      if ((p->Ampl[fnum] > low_A) && (p->Ampl[fnum] < hi_A))
      {
         AccumulateVector(avg_trc,&incorporation_traces[fnum*npts],npts);
          navg++;
      }
   }
   if (navg>MIN_VALID_FLOWS){
    MultiplyVectorByScalar(avg_trc,1.0f/navg,npts);
    return(true);
   }
   return(false);
}

int SetWeightVector(float *weight_vect, int npts, std::vector<float> &ftimes, float fi_end){
  int i_start = -1;
  for (int i=0;i < npts;i++)
    {
       if (ftimes[i] < fi_end)
          weight_vect[i] = 0.0f;
       else
       {
          weight_vect[i] = 1.0f;
          if (i_start < 0)
             i_start = i;
       }
    }
  return(i_start);
}

void  ExpTailFitter::FitTauAdj(float *incorporation_traces, float *bkg_traces, BeadParams *p, reg_params *rp, FlowBufferInfo *my_flow, TimeCompression &time_c, int flow_block_size, int flow_block_start
  )
{
  int npts = time_c.npts();
   float tau_adj = p->tau_adj;

  std::vector<float> ftimes = time_c.frameNumber;

  if ( flow_block_start == 0 )    // First block ever.
   {
      // create average trace across all flows and process it
      float avg_trc[npts];
      const float LOW_AMPLITUDE = 0.5f;
      const float HI_AMPLITUDE = 3.0f;
      const float AVG_AMPLITUDE = 1.5f;
      bool find_adj = ComputeAverageValidTrace(avg_trc, incorporation_traces, p, npts,LOW_AMPLITUDE, HI_AMPLITUDE, flow_block_size);

      if (find_adj)
      {

         float fi_end = GetModifiedIncorporationEnd(&rp->nuc_shape,my_flow->flow_ndx_map[0],0, AVG_AMPLITUDE);
         float weight_vect[npts];
         int i_start = SetWeightVector(weight_vect,npts,ftimes, fi_end);

          float my_etbR = rp->AdjustEmptyToBeadRatioForFlow(p->R, p->Ampl[0],  p->Copies, p->phi, my_flow->flow_ndx_map[0], flow_block_start );
         float my_tauB = rp->ComputeTauBfromEmptyUsingRegionLinearModel(my_etbR);

         FitExpDecayParams min_params,max_params;
         min_params.Signal = 0.0f;
         min_params.tau = my_tauB*0.9f;
         min_params.dc_offset = -50.0f;
         max_params.Signal = 500.0f;
         max_params.tau = my_tauB*1.1f;
         max_params.dc_offset =  50.0f;
         FitExpDecay exp_fitter(npts,&ftimes[0]);

         exp_fitter.SetWeightVector(weight_vect);
         exp_fitter.SetLambdaStart(1.0E-20f);
         exp_fitter.SetLambdaThreshold(100.0f);
         exp_fitter.SetParamMax(max_params);
         exp_fitter.SetParamMin(min_params);
         exp_fitter.params.Signal = 20.0f;
         exp_fitter.params.tau = my_tauB;
         exp_fitter.params.dc_offset = 0.0f;
         exp_fitter.SetStartAndEndPoints(i_start,npts);
         exp_fitter.Fit(false, 200, avg_trc);

         tau_adj = exp_fitter.params.tau/my_tauB;
      }
   }

  p->tau_adj = tau_adj;
}

void ExpTailFitter::AdjustBackground(float *incorporation_traces,float *bkg_traces,
    BeadParams *p,reg_params *rp,FlowBufferInfo *my_flow,TimeCompression &time_c,
    int flow_block_size, int flow_block_start
  )
{
  int npts = time_c.npts();

  // now process each flow indepdendently
  for (int fnum=0;fnum < flow_block_size;fnum++)
  {
     float my_etbR = rp->AdjustEmptyToBeadRatioForFlow(p->R, p->Ampl[fnum],  p->Copies, p->phi, my_flow->flow_ndx_map[fnum], flow_block_start + fnum );
     float my_tauB = rp->ComputeTauBfromEmptyUsingRegionLinearModel(my_etbR)*p->tau_adj;
     float my_t_mid_nuc = GetModifiedMidNucTime(&rp->nuc_shape,my_flow->flow_ndx_map[fnum],fnum);

     float bkg_leakage_fraction = generic_exp_tail_fit(&incorporation_traces[fnum*npts],&bkg_traces[fnum*npts],time_c,my_tauB,p->Ampl[fnum],my_t_mid_nuc);
     AddScaledVector(&incorporation_traces[fnum*npts],&bkg_traces[fnum*npts],-bkg_leakage_fraction,npts);
  }

}

/*
// this routine inappropriately combines two separate fitting processes that happen to use the same data
// do not call it if avoidable
float ExpTailFitter::CorrectTraces(float *incorporation_traces,float *bkg_traces,BeadParams *p,reg_params *rp,FlowBufferInfo *my_flow,TimeCompression &time_c, int flow_block_size, int flow_block_start )
{
  // only happens in first block of compute
  FitTauAdj(incorporation_traces, bkg_traces, p,rp,my_flow, time_c, flow_block_size, flow_block_start);
  
  // adjusts in all compute flows
  AdjustBackground(incorporation_traces,bkg_traces,p,rp,my_flow, time_c, flow_block_size, flow_block_start);

  return(p->tau_adj);
}*/

bool FindNonIncorporatingInterval(int &t_start, int &t_end, std::vector<float> &ftimes, float t_mid_nuc, float mer_guess, int npts){
  float fi_end = t_mid_nuc + MIN_INCORPORATION_TIME_PI + MIN_INCORPORATION_TIME_PER_MER_PI_VERSION_TWO* mer_guess;

t_start = -1;
t_end = -1;

 for (int i = 0; (i < npts); i++) {

    if ((t_start == -1) && (ftimes[i] >= fi_end))t_start = i;
    if ((t_end == -1) && (ftimes[i] >= (fi_end + 60.0f))) t_end = i;
 }

 // not enough frames warranting an exponenetial tail fit
 if (t_start == -1) {
   return(false);
 }

 if (t_end == -1)
   t_end = npts;

 // not enough data points to inspect
 if (t_end -t_start < 5)
   return(false);
 return(true);
}

void GenerateSmoothIncTrace(float *smooth_inc_trace, float *inc_trc, std::vector<float> &ftimes, int t_start, int t_end, float tau, int npts){
  // 7 = 2*3+1
  const int kern_dist =3;
  float kern[7];

  int iexp_start =t_start;
  int iexp_offset =2*kern_dist+1;
  if ((t_end-t_start) < iexp_offset)
     iexp_start =t_end-iexp_offset;

  for (int i = 0; i < iexp_offset; i++)
  {
     float dt = (ftimes[i+iexp_start]-ftimes[iexp_start+kern_dist])/tau;

     if (dt > 0.0f)
        kern[i] = 1.0f/ExpApprox(-dt);
     else
        kern[i] = ExpApprox(dt);
  }

  smooth_kern(smooth_inc_trace, inc_trc, kern, kern_dist, npts);
}

float IntegratedBackgroundScale(float *bkg_trace, int t_start, int t_end){
  float bkg_scale = 0.0f;

  for (int i=t_start;i <t_end;i++)
     bkg_scale += bkg_trace[i];

  bkg_scale /= (t_end-t_start);
  return(bkg_scale);
}


float ExpTailFitter::generic_exp_tail_fit(float *inc_trc, float *bkg_trace, TimeCompression& time_c, float tau, float mer_guess, float t_mid_nuc, bool debug)
{
  if (tau <= 0.0f) return 0.0f;
   int npts = time_c.npts();

   std::vector<float> ftimes = time_c.frameNumber;

   int t_start,t_end;
   if (!FindNonIncorporatingInterval(t_start,t_end,ftimes, t_mid_nuc, mer_guess, npts))
     return(0.0f);


   float smooth_inc_trace[npts];
   GenerateSmoothIncTrace(smooth_inc_trace,inc_trc, ftimes, t_start, t_end, tau, npts);


   float exp_amp;
   float mse_err;
   float integrated_delta = fit_exp_tail_to_data(smooth_inc_trace,t_start,t_end, ftimes, tau, &exp_amp, &mse_err, debug);
   float bkg_scale = IntegratedBackgroundScale(bkg_trace, t_start, t_end);
//@TODO: note potential divide by zero error here - fixed?
   // @TODO: this is simply decomposing SI = signal*exp(-t/tau) + alpha*bkg  - but we're ignoring the actual bkg shape and substituting an averaage dc value
   float tmp_fraction = 0.0f;
   if (bkg_scale>sanity_nuc_step_lower){
     // bkg ph step well enough defined to take a fraction
    tmp_fraction = integrated_delta/bkg_scale;
   }
   // but might have an insane fit anyway due to the vagaries of data
   // clamp to sensible bounds
   if (tmp_fraction>sanity_ratio_limit){
      tmp_fraction = sanity_ratio_limit;
   }
   if (tmp_fraction<(-sanity_ratio_limit)){
     tmp_fraction = -sanity_ratio_limit;
   }
   return (tmp_fraction);
}

float ExpTailFitter::fit_exp_tail_to_data(float *trc, int start_pt, int end_pt, std::vector<float> &ftimes, float tau, float *exp_amp, float *mse_err, bool debug)
{
   // initialize matricies
   lhs(0, 0) = 0.0;
   lhs(0, 1) = 0.0;
   lhs(1, 0) = 0.0;
   lhs(1, 1) = 0.0;
   rhs(0) = 0.0;
   rhs(1) = 0.0;

   for (int i = start_pt; i < end_pt; i++) {
      lhs(0, 0) += 1.0;

      float dt = (ftimes[i] - ftimes[start_pt]);
      float expval = ExpApprox(-dt / tau);
      lhs(0, 1) += expval;
      lhs(1, 1) += expval * expval;

      rhs(0) += trc[i];
      rhs(1) += trc[i] * expval;
   }
   lhs(1, 0) = lhs(0, 1);
   // I'm not sure if the solve method leaves the original matricies intact or not, some I'm saving rhs(0)
   // for later use just in case
   double old_rhs_0 = rhs(0);

   // we *shouldn't ever have to worry about this throwing an exception since, by construction, the matrix
   // equation is solvable (as long as tau is non-zero)
   vals = arma::solve(lhs, rhs);

   if (debug) fprintf(stdout, "start_pt:%d end_pt:%d dc_offset:%f exp_amplitude:%f\n", start_pt, end_pt, vals(0), vals(1));

   // a fit of a negative exponential amplitude is not physical
   // so we revert to assuming that there was no incorporation signal
   // and just use the dc average of the trace as the dc offset
   if (vals(1) < -20.0) {
      if (exp_amp != NULL) *exp_amp = 0.0f;

      if (mse_err != NULL)
      {
         float msesum = 0.0f;
         for (int i=start_pt;i < end_pt;i++)
         {
            float err = trc[i]-old_rhs_0;
            msesum += err*err;
         }
         *mse_err = msesum/(end_pt-start_pt);
      }

      return ((float)old_rhs_0 / (end_pt - start_pt));
   }

   if (exp_amp != NULL) *exp_amp = vals(1);

   if (mse_err != NULL)
   {
      float msesum = 0.0f;
      for (int i=start_pt;i < end_pt;i++)
      {
         float dt = (ftimes[i] - ftimes[start_pt]);
         float expval = vals(1)*ExpApprox(-dt / tau);
         float err = trc[i]-(expval+vals(0));
         msesum += err*err;
      }
      *mse_err = msesum/(end_pt-start_pt);
   }

   return ((float)vals(0));
}

void smooth_kern(float *out, float *in, float *kern, int dist, int npts)
{
   float sum;
   float scale;

   for (int i = 0; i < npts; i++) {
      sum = 0.0f;
      scale = 0.0f;

      for (int j = i - dist, k = 0; j <= (i + dist); j++, k++) {
         if ((j >= 0) && (j < npts)) {
            sum += kern[k] * in[j];
            scale += kern[k];
         }
      }
      out[i] = sum / scale;
   }
}



