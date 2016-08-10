/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MathOptim.h"
#include "BkgMagicDefines.h"
#include "MathUtil.h"
#include "ExpTailFitter.h"

void ExpTailBkgFitter::AdjustBackground(float *incorporation_traces,float *bkg_leakage_fraction, float *bkg_traces,
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

     bkg_leakage_fraction[fnum] = generic_exp_tail_fit(&incorporation_traces[fnum*npts],&bkg_traces[fnum*npts],time_c,my_tauB,p->Ampl[fnum],my_t_mid_nuc);
     AddScaledVector(&incorporation_traces[fnum*npts],&bkg_traces[fnum*npts],-bkg_leakage_fraction[fnum],npts);
  }

}


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


float ExpTailBkgFitter::generic_exp_tail_fit(float *inc_trc, float *bkg_trace, TimeCompression& time_c, float tau, float mer_guess, float t_mid_nuc, bool debug)
{
  if (tau <= 0.0f) return 0.0f;
   int npts = time_c.npts();

   std::vector<float> ftimes = time_c.frameNumber;

   int t_start,t_end;
   if (!FindNonIncorporatingInterval(t_start,t_end,ftimes, t_mid_nuc, mer_guess, npts))
     return(0.0f);


   float smooth_inc_trace[npts];
   GenerateSmoothIncTrace(smooth_inc_trace,inc_trc, ftimes, t_start, t_end, tau, npts);



   float integrated_delta = fit_exp_tail_to_data(smooth_inc_trace,t_start,t_end, ftimes, tau,  debug);
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

void ExpTailBkgFitter::ResetMat(){
  // initialize matricies
  lhs(0, 0) = 0.0;
  lhs(0, 1) = 0.0;
  lhs(1, 0) = 0.0;
  lhs(1, 1) = 0.0;
  rhs(0) = 0.0;
  rhs(1) = 0.0;
}

void ExpTailBkgFitter::SetupXtX(float *trc, int start_pt, int end_pt, std::vector<float> &ftimes, float tau){
  // XtX = v*v, exp*v, exp*v exp*exp
  // Xt*y = v*y, exp*y

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
}



float ExpTailBkgFitter::fit_exp_tail_to_data(float *trc, int start_pt, int end_pt, std::vector<float> &ftimes, float tau, bool debug)
{
  ResetMat();
  SetupXtX(trc, start_pt, end_pt, ftimes,tau);

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
      return ((float)old_rhs_0 / (end_pt - start_pt));
   }

   return ((float)vals(0));
}


