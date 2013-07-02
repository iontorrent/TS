/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MathOptim.h"
#include "BkgMagicDefines.h"
#include "MathUtil.h"
#include "ExpTailFitter.h"
#include "FitExpDecay.h"

float ExpTailFitter::CorrectTraces(float *incorporation_traces,float *bkg_traces,bead_params *p,reg_params *rp,flow_buffer_info *my_flow,TimeCompression &time_c)
{
   int npts = time_c.npts();
   std::vector<float> ftimes = time_c.frameNumber;
   FitExpDecay exp_fitter(npts,&ftimes[0]);
   float tau_adj = p->tau_adj;

#if 1
   if (my_flow->buff_flow[0] == 0)
   {
      // create average trace across all flows and process it
      float avg_trc[npts];
      float avg_bkg[npts];
      int navg = 0;
      memset(avg_trc,0,sizeof(avg_trc));
      memset(avg_bkg,0,sizeof(avg_bkg));
      for (int fnum=0;fnum < NUMFB;fnum++)
      {
         // keep out long HPs in order to prevent pollution of data
         // points with additional generated protons
         // keep out non-incorporating flows because they don't contain
         // exponential decay information
         if ((p->Ampl[fnum] > 0.5f) && (p->Ampl[fnum] < 3.0f))
         {
            AccumulateVector(avg_trc,&incorporation_traces[fnum*npts],npts);
            AccumulateVector(avg_bkg,&bkg_traces[fnum*npts],npts);
            navg++;
         }
      }

      if (navg > 6)
      {
         MultiplyVectorByScalar(avg_trc,1.0f/navg,npts);

         float my_t_mid_nuc = GetModifiedMidNucTime(&rp->nuc_shape,my_flow->flow_ndx_map[0],0);

         // estimated average end of incorporation
         float fi_end = my_t_mid_nuc + 6.0f + 2.0f*1.5f;
         float weight_vect[npts];
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

         float my_etbR = AdjustEmptyToBeadRatioForFlow(p->R,rp,my_flow->flow_ndx_map[0],my_flow->buff_flow[0]);
         float my_tauB = ComputeTauBfromEmptyUsingRegionLinearModel(rp,my_etbR);

         FitExpDecayParams min_params,max_params;
         min_params.Ampl = 0.0f;
         min_params.tau = my_tauB*0.9f;
         min_params.dc_offset = -50.0f;
         max_params.Ampl = 500.0f;
         max_params.tau = my_tauB*1.1f;
         max_params.dc_offset =  50.0f;

         exp_fitter.SetWeightVector(weight_vect);
         exp_fitter.SetLambdaStart(1.0E-20f);
         exp_fitter.SetLambdaThreshold(100.0f);
         exp_fitter.SetParamMax(max_params);
         exp_fitter.SetParamMin(min_params);
         exp_fitter.params.Ampl = 20.0f;
         exp_fitter.params.tau = my_tauB;
         exp_fitter.params.dc_offset = 0.0f;
         exp_fitter.SetStartAndEndPoints(i_start,npts);
         exp_fitter.Fit(200,avg_trc);

         tau_adj = exp_fitter.params.tau/my_tauB;
      }
   }
#endif

   p->tau_adj = tau_adj;

   // now process each flow indepdendently
   for (int fnum=0;fnum < NUMFB;fnum++)
   {
      float my_etbR = AdjustEmptyToBeadRatioForFlow(p->R,rp,my_flow->flow_ndx_map[fnum],my_flow->buff_flow[fnum]);
      float my_tauB = ComputeTauBfromEmptyUsingRegionLinearModel(rp,my_etbR)*tau_adj;
      float my_t_mid_nuc = GetModifiedMidNucTime(&rp->nuc_shape,my_flow->flow_ndx_map[fnum],fnum);

      float dc_offset = generic_exp_tail_fit(&incorporation_traces[fnum*npts],&bkg_traces[fnum*npts],time_c,my_tauB,p->Ampl[fnum],my_t_mid_nuc);
      AddScaledVector(&incorporation_traces[fnum*npts],&bkg_traces[fnum*npts],-dc_offset,npts);
   }

   return(tau_adj);
}

float ExpTailFitter::generic_exp_tail_fit(float *trc, float *bkg_trace, TimeCompression& time_c, float tau, float mer_guess, float t_mid_nuc, bool debug)
{
   int npts = time_c.npts();
   int i0, i1, i2, i3;
   float fi_start = t_mid_nuc - 9.0f;
   float fi_end = t_mid_nuc + 6.0f + 1.75f * mer_guess;
   std::vector<float> ftimes = time_c.frameNumber;
   float bkg_ampl = 0.0f;

   if (tau <= 0.0f) return 0.0f;

   i0 = 0;
   i1 = -1;
   i2 = -1;
   i3 = -1;

   for (int i = i0; (i < npts); i++) {
      if ((i1 == -1) && (ftimes[i] >= fi_start)) i1 = i;
      if ((i2 == -1) && (ftimes[i] >= fi_end)) i2 = i;
      if ((i3 == -1) && (ftimes[i] >= (fi_end + 60.0f))) i3 = i;
   }

   // not enough frames warranting an exponenetial tail fit
   if (i2 == -1) {
     return (0.0f);
   }

   if (i3 == -1)
      i3 = npts;

   // not enough data points to inspect
   if (i3 - i2 < 5)
      return (0.0f);

   float tmp[npts];
   float kern[7];

   int iexp_start = i2;
   if ((i3-i2) < 7)
      iexp_start = i3-7;

   for (int i = 0; i < 7; i++)
   {
      float dt = (ftimes[i+iexp_start]-ftimes[iexp_start+3])/tau;

      if (dt > 0.0f)
         kern[i] = 1.0f/ExpApprox(-dt);
      else
         kern[i] = ExpApprox(dt);
   }

   smooth_kern(tmp, trc, kern, 3, npts);

   float exp_amp;
   float mse_err;
   float dc_offset = fit_exp_tail_to_data(tmp, i2, i3, ftimes, tau, &exp_amp, &mse_err, debug);

   for (int i=i2;i < i3;i++)
      bkg_ampl += bkg_trace[i];

   bkg_ampl /= (i3-i2);

   return (dc_offset/bkg_ampl);
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

void ExpTailFitter::smooth_kern(float *out, float *in, float *kern, int dist, int npts)
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

float ExpTailFitter::flux_integral_analysis(float *trc, TimeCompression &time_c, float tau, float t0)
{
   int npts = time_c.npts();
   std::vector<float> ftimes = time_c.frameNumber;
   float trc_smooth[npts];
   float trc_smooth_2[npts];
   float kern[5] = {-0.0857,0.3429,0.4857,0.3429,-0.0857};
   float kern_2[9] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};

   // do some minimal smoothing
   smooth_kern(trc_smooth, trc, kern, 2, npts);

   // calculate the proton flux
   float last = 0.0f;
   float tlast = 0.0f;
   for (int i=0;i < npts;i++) {
      float dt = ftimes[i]-tlast;
      float dv = trc_smooth[i]-last;

      last = trc_smooth[i];
      trc_smooth[i] = (dv/dt)+trc_smooth[i]/tau;
      tlast = ftimes[i];
   }

   smooth_kern(trc_smooth_2, trc_smooth, kern_2, 4, npts);

   // find the peak flux
   float max_flux = 0.0f;
   int max_flux_pt = 0;
   for (int i = 0; (i < npts); i++) {
      if (ftimes[i] < t0)
         continue;

      if (trc_smooth_2[i] > max_flux) {
         max_flux = trc_smooth_2[i];
         max_flux_pt = i;
      }
   }

   float tend = 2.0f*ftimes[max_flux_pt]-t0;
   return(tend);
}

void ExpTailFitter::calc_proton_flux(float *trc,TimeCompression &time_c,float tau,float mer_guess)
{
   int npts = time_c.npts();
   std::vector<float> ftimes = time_c.frameNumber;
   float tmp[npts];

   tmp[0] = 0.0f;
   tmp[npts-1] = 0.0f;

   for (int i=1;i < npts-1;i++)
      tmp[i] = trc[i]/tau + (trc[i+1]-trc[i-1])/(ftimes[i+1]-ftimes[i-1]); 

   float kern[7];
   float mg = mer_guess;
   if (mg < 0.0f) mg = 0.0f;
   for (int i=0;i < 7;i++)
   {
      float dt=i-3;
      kern[i] = ExpApprox(-dt*dt/(15.0f*(mg+0.5f)));
   }
   smooth_kern(trc,tmp,kern,3,npts);
}



