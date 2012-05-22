/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef STREAMINGKERNELS_H
#define STREAMINGKERNELS_H

#include "BkgModelCudaKernels.h"

__device__ void
ModelFuncEvaluation(
  float A, 
  float Krate,
  float tau,
  float gain,
  float SP,
  float d,
  float kmax,
  float sens,
  int start,
  int c_dntp_top_ndx,
  float* nucRise,
  int ibd,
  int num_frames,
  int num_beads,
  float* fval)
{
  if (A > MAX_HPLEN)
  {
    A = MAX_HPLEN;
  }

  int ileft, iright;
  float ifrac;

  // step 2
  float occ_l,occ_r;
  float totocc;
  float totgen;
  float pact;
  int i, st;

  // step 3
  float ldt;

  // step 4
  float c_dntp_int;
  float pact_new;

  // variables used for solving background signal shape
  float dv = 0.0;
  float dvn = 0.0;
  float dv_rs = 0.0;

  //int c_dntp_top_ndx = ISIG_SUB_STEPS_SINGLE_FLOW*start;

  // initialize diffusion/reaction simulation for this flow
  ileft = (int) A;
  iright = ileft + 1;
  ifrac = iright - A;
  occ_l = ifrac;
  occ_r = A - ileft;

  ileft--;
  iright--;

  if (ileft < 0)
  {
    occ_l = 0.0;
  }

  if (iright == MAX_HPLEN)
  {
    iright = ileft;
    occ_r = occ_l;
    occ_l = 0;
  }

  occ_l *= SP;
  occ_r *= SP;
  pact = occ_l + occ_r;
  totocc = SP*A;
  totgen = totocc;

  const float* rptr = precompute_pois_params (iright);
  const float* lptr = precompute_pois_params (ileft);

  float c_dntp_bot = 0.0; // concentration of dNTP in the well
  float c_dntp_top = 0.0;
  float c_dntp_sum = 0.0;
  float c_dntp_old_rate = 0;
  float c_dntp_new_rate = 0;

  float c_dntp_bot_plus_kmax = 1.0f/kmax;

  float aval;
  float scaledTau = 2.0f*tau;
  float scaled_kr = Krate*n_to_uM_conv/d;
  float half_kr = Krate*0.5f;
  float tmp;
  for (i=start;i < num_frames;i++)
  {
    if (totgen > 0.0f)
    {
      ldt = (DELTAFRAME_CONST_CUDA[i]/( ISIG_SUB_STEPS_SINGLE_FLOW * FRAMESPERSEC)) * half_kr;
      for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0f);st++)
      {
        c_dntp_top = nucRise[c_dntp_top_ndx];
        c_dntp_top_ndx += 1;

        // assume instantaneous equilibrium
        c_dntp_old_rate = c_dntp_new_rate;
        c_dntp_bot = c_dntp_top/ (1 + scaled_kr*pact*c_dntp_bot_plus_kmax);
        c_dntp_bot_plus_kmax = 1.0f/ (c_dntp_bot + kmax);

        c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
        c_dntp_int = ldt* (c_dntp_new_rate+c_dntp_old_rate);
        c_dntp_sum += c_dntp_int;

        // calculate new number of active polymerase
        pact_new = poiss_cdf_approx (iright,c_dntp_sum,rptr) * occ_r;
        if (occ_l > 0.0f)
          pact_new += poiss_cdf_approx (ileft,c_dntp_sum,lptr) * occ_l;

        totgen -= ( (pact+pact_new) * 0.5f) * c_dntp_int;
        pact = pact_new;
      }

      if (totgen < 0.0f) totgen = 0.0f;
    }

    // calculate the 'background' part (the accumulation/decay of the protons in the well
    // normally accounted for by the background calc)
    aval = DELTAFRAME_CONST_CUDA[i]/scaledTau;

    // calculate new dv
    dvn = ( (totocc-totgen) * sens - dv_rs/tau - dv*aval) / (1.0f+aval);
    dv_rs += (dv+dvn) *aval*tau;
    tmp = dv = dvn;
    fval[num_beads*i + ibd] = tmp*gain;
  }
}

__device__ void 
ResidualCalculationPerFlow(
  float* fg_buffers, 
  float* fval, 
  float* weight, 
  float* err, 
  float& residual,
  float wtScale,
  int ibd,
  int num_beads,
  int num_frames) {

  int i;
  float e;
  residual = 0;
  for (i=0; i<num_frames; ++i) {
    err[num_beads*i + ibd] = e = weight[num_beads*i + ibd] * 
                                    (fg_buffers[num_beads*i + ibd] - fval[num_beads*i + ibd]);
    residual += e*e;
  }
  residual /= wtScale;
}

// Let number of beads be N and frames be F. The size for each input argument in
// comments is in bytes

template<int iterations>
__global__ void 
PerFlowLevMarFit_k(
  // inputs
  float* fg_buffers, // NxF

  // weights for frames
  float* weight, // NxF 
  float wtScale, // 4

  // bead params
  float* pAmpl, // N
  float* pKmult, // N
  float* pdmult, // N
  float* ptauB, // N
  float* pgain, // N
  float* pSP, // 4
  float sens, // 4
  float maxAmpl, // 4
  float minAmpl, // 4
  float maxKrate, // 4
  float minKrate, // 4
 
  // NucRise params are for the region 
  float* nucRise, // 2xF
  int start, // 4
  int c_dntp_top_ndx, // 4

  // region params
  float krate_reg, // 4
  float d_reg, // 4
  float kmax, // 4

  // scratch space in global memory
  float* err, // NxF
  float* fval, // NxF
  float* tmp_fval, // NxF
  float* jac, // 2xNxF 

  // other inputs 
  int num_beads, // 4
  int num_frames, // 4
  int beadFrameProduct // 4
  ) 
{
  int ibd = blockIdx.x * blockDim.x + threadIdx.x;

  float krate = pKmult[ibd]*krate_reg;
  float Ampl = pAmpl[ibd];
  float tauB = ptauB[ibd];
  float d = pdmult[ibd]*d_reg;
  float gain = pgain[ibd];
  float SP = pSP[ibd]; 
  
  float residual, newresidual;
  int i, iter;

  // These values before start are always zero since there is no nucrise yet. Don't need to
  // zero it out. Have to change the residual calculation accordingly for the frames before the
  // start.
  for (i=0; i < start; i++) {
    fval[num_beads*i + ibd] = 0;
    tmp_fval[num_beads*i + ibd] = 0;
  }

  // first step
  // Evaluate model function using input Ampl and Krate and get starting residual
  ModelFuncEvaluation(Ampl, krate, tauB, gain, SP, d, kmax, sens, start, c_dntp_top_ndx,
      nucRise, ibd, num_frames, num_beads, fval);

  // calculating weighted sum of square residuals for the convergence test
  ResidualCalculationPerFlow(fg_buffers, fval, weight, err, residual, wtScale, 
      ibd, num_beads, num_frames);

  // new Ampl and Krate generated from the Lev mar Fit
  float newAmpl, newKrate;

  // initialize lambda
  float lambda = 1E-20;

  // convergence test variables 
  float delta0 = 0, delta1 = 0;

  // determinant for the JTJ matrix in Lev Mar Solve
  float det;

  // Indicates whether a flow has converged
  int flowDone = 0;

  // Lev Mar Fit Outer Loop
  for (iter = 0; iter < iterations; ++iter) {

     // convergence test...need to think of an alternate approach
     if ((delta0*delta0) < 0.0000025)
       flowDone++;
     else
       flowDone = 0;

     // stop the loop for this bead here
     if (flowDone  == 2)
       break;
     
     // new Ampl and krate by adding delta to existing values
     newAmpl = Ampl + 0.001f;
     newKrate = krate + 0.001f;
  
     // Evaluate model function for new Ampl keeping Krate constant
     ModelFuncEvaluation(newAmpl, krate, tauB, gain, SP, d, kmax, sens, 
       start, c_dntp_top_ndx, nucRise, ibd, num_frames, num_beads, tmp_fval);
    
     // Write jacobian entry for model function derivative w.r.t Amplitude
     for (i=start; i < num_frames; ++i) {
       jac[num_beads*i + ibd] = weight[num_beads*i + ibd] * (tmp_fval[num_beads*i + ibd] - 
                              fval[num_beads*i + ibd]) * 1000.0f;
     }

     // Evaluate model function for new Krate keeping amplitude constant
     ModelFuncEvaluation(Ampl, newKrate, tauB, gain, SP, d, kmax, sens, 
       start, c_dntp_top_ndx, nucRise, ibd, num_frames, num_beads, tmp_fval);

     // Write jacobian entry for model function derivative w.r.t Krate
     for (i=start; i < num_frames; ++i) {
       jac[beadFrameProduct + num_beads*i + ibd] = weight[num_beads*i + ibd] * (tmp_fval[num_beads*i + ibd] - 
                              fval[num_beads*i + ibd]) * 1000.0f;
     }

     // Create JTJ and RHS entries 
     //1. dF/dA * dF/dA -> aa
     //2. dF/dA * dF/dKr -> akr
     //3. dF/dKr * dF/dA -> akr (commutative)
     //4. dF/dKr * dF/dKr -> krkr
     float aa = 0, akr= 0, krkr = 0, rhs0 = 0, rhs1 = 0;
     for (i=start; i < num_frames; ++i) {
       aa += jac[num_beads*i + ibd] * jac[num_beads*i + ibd];
       akr += jac[num_beads*i + ibd] * jac[beadFrameProduct + num_beads*i + ibd];
       krkr += jac[beadFrameProduct + num_beads*i + ibd] * jac[beadFrameProduct + num_beads*i + ibd];
       rhs0 += jac[num_beads*i + ibd] * err[num_beads*i + ibd];
       rhs1 += jac[beadFrameProduct + num_beads*i + ibd] * err[num_beads*i + ibd];
     } 

     // write jtj. Skip akr entry as it is same as kra
     //jtj[ibd] = aa;
     //jtj[num_beads + ibd] = akr;
     //jtj[num_beads*2 + ibd] = krkr;

     //rhs[ibd] = rhs0;
     //rhs[num_beads + ibd] = rhs1;

     // Now start the solving. Scale jtj by (1 + lambda) on the fly 
     // Inner loop for running steps of lambda
     bool cont_proc = false;
    while (!cont_proc) {
      // use inv() here
      det = 1.0f / (aa*krkr*(1.0f + lambda)*(1.0f + lambda) - akr*akr);
      delta0 = (krkr*(1.0f + lambda)*rhs0 - akr*rhs1)*det;
      delta1 = (-akr*rhs0 + aa*(1.0f + lambda)*rhs1)*det;
     
      // NAN check

      // add delta to params to obtain new params
      newAmpl = Ampl + delta0;
      newKrate = krate + delta1;

      clamp(newAmpl, minAmpl, maxAmpl);
      clamp(newKrate, minKrate, maxKrate);

      // Evaluate using new params
      ModelFuncEvaluation(newAmpl, newKrate, tauB, gain, SP, d, kmax, sens, 
        start, c_dntp_top_ndx, nucRise, ibd, num_frames, num_beads, tmp_fval);
      // residual calculation using new parameters
      ResidualCalculationPerFlow(fg_buffers, tmp_fval, weight, err, newresidual, 
        wtScale, ibd, num_beads, num_frames);

      // this might be killing...Need to rethink for some alternative here
      // If new residual is less than the earlier recorded residual, accept the solution and
      // obtain new parameters and copy them to original parameters and copy the new model function 
      // to the earlier recorded model function till this point
      if (newresidual < residual) {
        lambda /= 10.0f;
        if (lambda < FLT_MIN)
          lambda = FLT_MIN;
        Ampl = newAmpl;
        krate = newKrate;

        // copy new function val to fval
        for (i=start; i<num_frames; ++i)
          fval[num_beads*i + ibd] = tmp_fval[num_beads*i + ibd];
        cont_proc = true;
        residual = newresidual;
      }
      else {
        lambda *= 10.0f;
      }
        
      if (lambda > 1.0f)
        cont_proc = true;
    }
  } 
}

#endif // STREAMINGKERNELS_H
