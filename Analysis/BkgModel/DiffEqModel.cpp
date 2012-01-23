/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "DiffEqModel.h"
#include "math.h"
#include <algorithm>

// ripped out to separate the math from the optimization procedure

// utility functions because C++ is too low level a language to be useful
void MultiplyVectorByScalar(float *my_vec, float my_scalar, int len)
{
    for (int i=0; i<len; i++)
        my_vec[i] *= my_scalar;
}

void AddScaledVector(float *start_vec, float *my_vec, float my_scalar, int len)
{
    for (int i=0; i<len; i++)
        start_vec[i] += my_scalar * my_vec[i];
}

void CopyVector(float *destination_vec, float *my_vec, int len)
{
    memcpy(destination_vec, my_vec, sizeof(float)*len);
}

void AccumulateVector(float *destination_vec, float *my_vec, int len)
{
    for (int i=0; i<len; i++)
        destination_vec[i] += my_vec[i];
}

void DiminishVector(float *destination_vec, float *my_vec, int len)
{
    for (int i=0; i<len; i++)
        destination_vec[i] -= my_vec[i];
}

void MultiplyVectorByVector(float *my_vec, float *my_other_vec, int len)
{
    for (int i=0; i<len; i++)
        my_vec[i] *= my_other_vec[i];
}

float CheckVectorDiffSSQ(float *my_vec, float *my_other_vec, int len)
{
    float retval = 0;
    for (int i=0; i<len; i++)
        retval += (my_vec[i]-my_other_vec[i])*(my_vec[i]-my_other_vec[i]);
    return(retval);
}

// generic local derivatives
void CALC_PartialDeriv(float *p1,float *p2,int len,float dp)
{
    float *pend = p2+len;
    for (;p2 < pend;p2++) *p2 = (*p2 - *p1++)/dp;
}

void CALC_PartialDeriv_W_EMPHASIS(float *p1,float *p2,float *pe,int len,float dp,float pelen)
{
    float *pend = p2+len;
    int ipe;
    for (ipe=0;p2 < pend;p2++)
    {
        *p2 = (*p2 - *p1++)*pe[ipe++]/dp;
        if (ipe >= pelen) ipe = 0;
    }
}


// computes the partial derivative for a parameter with emphasis.  len is the total length
// of data points to compute over, and all vectors should have len values.  dp is the amount
// the parameter was stepped by when computing the function values in p2.
void CALC_PartialDeriv_W_EMPHASIS_LONG(float *p1,float *p2,float *pe,int len,float dp)
{
    for (int i=0;i < len;i++)
        p2[i] = (p2[i] - p1[i])*pe[i]/dp;
}



// incorporation traces not yet quite right

// Compute the incorporation "red" hydrogen trace only without any "blue" hydrogen from the bulk or cross-talk
// does not adjust for gain?
void RedSolveHydrogenFlowInWell(float *vb_out, float *red_hydrogen, int len, int i_start,float *deltaFrame, float tauB)
{
    float dv = 0.0;
    float dvn = 0.0;
    float dv_rs = 0.0;
    float scaledTau = 2.0*tauB;
    memset(vb_out,0,sizeof(float[i_start]));
    for (int i=i_start; i<len; i++)
    {
        // calculate the 'background' part (the accumulation/decay of the protons in the well
        // normally accounted for by the background calc)
        float dt = deltaFrame[i];
        float aval = dt/scaledTau;

        // calculate new dv
        dvn = (red_hydrogen[i] - dv_rs/tauB - dv*aval) / (1.0+aval);
        dv_rs += (dv+dvn)*dt/2.0;
        vb_out[i] = dv = dvn;
    }
}

// generates the background trace for a well given the "empty" well measurement of blue_hydrogen ions
void BlueSolveBackgroundTrace(float *vb_out, float *blue_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
    float dv,dvn,dv_rs;
    float aval = 0;
    float dt;
    float shift_ratio = etbR-1.0;

    dv = 0.0;
    dv_rs = 0.0;
    dt = -1.0;
    dvn = 0.0;

    for (int i=0;i < len;i++)
    {
        dt = deltaFrame[i];
        aval = dt/(2.0*tauB);

        // calculate new dv
        dvn = (shift_ratio*blue_hydrogen[i] - dv_rs/tauB - dv*aval) / (1.0+aval);
        dv_rs += (dv+dvn)*dt/2.0;
        dv = dvn;

        vb_out[i] = (dv+blue_hydrogen[i]);
    }
}

// even although this is simply the sum of red and blue hydrogen ions, we'll compute it here because we think it is mildly more efficient
void PurpleSolveTotalTrace(float *vb_out, float *blue_hydrogen, float *red_hydrogen, int len, float *deltaFrame, float tauB, float etbR)
{
    float dv,dvn,dv_rs;
    float aval = 0;
    int   i;
    float dt;
    float shift_ratio = etbR-1.0;

    dv = 0.0;
    dv_rs = 0.0;
    dt = -1.0;
    dvn = 0.0;
    for (i=0;i < len;i++)
    {
        dt = deltaFrame[i];
        aval = dt/ (2.0*tauB);

        dvn = (red_hydrogen[i] + shift_ratio *blue_hydrogen[i] - dv_rs/tauB - dv*aval) / (1.0+aval);
        dv_rs += (dv+dvn) *dt/2.0;
        dv = dvn;

        vb_out[i] = (dv+blue_hydrogen[i]);
    }
}

// shielding layer to insulate from choice
void ComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrameSeconds,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
        float A, float SP,
        float kr, float kmax, float d, PoissonCDFApproxMemo *math_poiss, bool do_simple) // default value for external calls
{
  bool purely_local = false;
  if (math_poiss==NULL)
  {
    math_poiss = new PoissonCDFApproxMemo;
    math_poiss->Allocate(MAX_HPLEN+1,512,0.05);
    math_poiss->GenerateValues();
    purely_local = true;
  }
    if (do_simple)
        SimplifyComputeCumulativeIncorporationHydrogens(ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,C,A,SP,kr,kmax,d,math_poiss);
    else
        ComplexComputeCumulativeIncorporationHydrogens(ival_offset,npts,deltaFrameSeconds,nuc_rise_ptr,SUB_STEPS,my_start,C,A,SP,kr,kmax,d,math_poiss);
    
   if (purely_local)
   {
     delete math_poiss;
     math_poiss=NULL;
   }
}


#define FLOW_STEP 4

//assumptions: get 4 flows passed here
// deltaFrameSeconds always the same
// SUB_STEPS always the same
// pretend we are vectorizing this function
// by the crude example of doing arrays pretending to be vectors for all.
void ParallelSimpleComputeCumulativeIncorporationHydrogens(float **ival_offset, int npts, float *deltaFrameSeconds,
        float **nuc_rise_ptr, int SUB_STEPS, int *my_start, 
        float *A, float *SP,
        float *kr, float *kmax, float *d, PoissonCDFApproxMemo *math_poiss)
{
    int i;
    int common_start;
    
    float totocc[FLOW_STEP], totgen[FLOW_STEP];
    float pact[FLOW_STEP],pact_new[FLOW_STEP];
    float  c_dntp_top[FLOW_STEP], c_dntp_bot[FLOW_STEP];
    float  hplus_events_sum[FLOW_STEP], hplus_events_current[FLOW_STEP]; // mean events per molecule, cumulative and current
    
    MixtureMemo mix_memo[4];
    float tA[FLOW_STEP];
    for (int q=0; q<FLOW_STEP; q++)
        tA[q] = mix_memo[q].Generate(A[q],math_poiss); // don't damage A now that it is a pointer

    for (int q=0; q<FLOW_STEP; q++)
      mix_memo[q].ScaleMixture(SP[q]);
    for (int q=0; q<FLOW_STEP; q++)
      pact[q] = mix_memo[q].total_live;  // active polymerases
    for (int q=0; q<FLOW_STEP; q++)
    totocc[q] = SP[q]*tA[q];  // how many hydrogens we'll eventually generate
   
    for (int q=0; q<FLOW_STEP; q++)
      totgen[q] = totocc[q];  // number remaining to generate

    for (int q=0; q<FLOW_STEP; q++)
      c_dntp_bot[q] = 0.0; // concentration of dNTP in the well
    for (int q=0; q<FLOW_STEP; q++)
      c_dntp_top[q] = 0.0; // concentration at top
    for (int q=0; q<FLOW_STEP; q++)
      hplus_events_sum[q] = 0.0;
    for (int q=0; q<FLOW_STEP; q++)
      hplus_events_current[q] = 0.0; // Events per molecule

    for (int q=0; q<FLOW_STEP; q++)
      memset(ival_offset[q],0,sizeof(float[npts]));  // zero the points we don't compute

    float scaled_kr[FLOW_STEP];
    for (int q=0; q<FLOW_STEP; q++)
      scaled_kr[q] = kr[q]*n_to_uM_conv/d[q]; // convert molecules of polymerase to active concentraction
    float half_kr[FLOW_STEP];
    for (int q=0; q<FLOW_STEP; q++)
       half_kr[q] = kr[q] *0.5; // for averaging

    
    float c_dntp_bot_plus_kmax[FLOW_STEP];
    for (int q=0; q<FLOW_STEP; q++)
      c_dntp_bot_plus_kmax[q] = 1.0/kmax[q];
    
    float c_dntp_old_effect[FLOW_STEP];
    for (int q=0; q<FLOW_STEP; q++)
        c_dntp_old_effect[q] = 0.0;
    float c_dntp_new_effect[FLOW_STEP];
    for (int q=0; q<FLOW_STEP; q++)
        c_dntp_new_effect[q] = 0.0;

    
    float sum_totgen = 0.0;
    for (int q=0; q<FLOW_STEP; q++)
      sum_totgen += totgen[q];
    
    common_start = my_start[0]; 
    for (int q=0; q<FLOW_STEP; q++)
      if (common_start<my_start[q])
        common_start = my_start[q];
    // first non-zero index of the computed [dNTP] array for this nucleotide
    int c_dntp_top_ndx = common_start*SUB_STEPS;

    for (i=common_start;i < npts;i++)
    {
        if (sum_totgen > 0.0)
        {
            float ldt;
            ldt = deltaFrameSeconds[i];
            ldt /= SUB_STEPS;
            for (int st=1; (st <= SUB_STEPS) && (sum_totgen > 0.0);st++)  // someone needs computation
            {
                // update top of well concentration in the bulk for FLOW_STEP flows
                for (int q=0; q<FLOW_STEP; q++)
                  c_dntp_top[q] = nuc_rise_ptr[q][c_dntp_top_ndx];
                c_dntp_top_ndx += 1; 

                // assume instantaneous equilibrium within the well
                for (int q=0; q<FLOW_STEP; q++)
                  c_dntp_bot[q] = c_dntp_top[q]/(1+ scaled_kr[q]*pact[q]*c_dntp_bot_plus_kmax[q]); // the level at which new nucs are used up as fast as they diffuse in
                for (int q=0; q<FLOW_STEP; q++)
                  c_dntp_bot_plus_kmax[q] = 1.0/(c_dntp_bot[q] + kmax[q]); // scale for michaelis-menten kinetics, assuming nucs are limiting factor

                // Now compute effect of concentration on enzyme rate
                for (int q=0; q<FLOW_STEP; q++)
                  c_dntp_old_effect[q] = c_dntp_new_effect[q];
                for (int q=0; q<FLOW_STEP; q++)
                  c_dntp_new_effect[q] = c_dntp_bot[q]*c_dntp_bot_plus_kmax[q]; // current effect of concentration on enzyme rate
                
                // update events per molecule
                for (int q=0; q<FLOW_STEP; q++)
                  hplus_events_current[q] = ldt*half_kr[q]*(c_dntp_new_effect[q]+c_dntp_old_effect[q]);  // events per molecule is average rate * time of rate
                for (int q=0; q<FLOW_STEP; q++)
                  hplus_events_sum[q] += hplus_events_current[q];
                
                // how many active molecules left at end of time period given poisson process with total intensity of events
                for (int q=0; q<FLOW_STEP; q++)
                  pact_new[q] = mix_memo[q].GetStep(hplus_events_sum[q]); 

                
                // how many hplus were generated
                for (int q=0; q<FLOW_STEP; q++)
                  totgen[q] -= ((pact[q]+pact_new[q]) * 0.5) * hplus_events_current[q];  // active molecules * events per molecule
                for (int q=0; q<FLOW_STEP; q++)
                  pact[q] = pact_new[q];
              for (int q=0; q<FLOW_STEP; q++)
                totgen[q] = std::max(totgen[q],0.0f);
              // or is there a "max" within command?
              sum_totgen = 0.0;
              for (int q=0; q<FLOW_STEP; q++)
                sum_totgen += totgen[q];
            }

        }
        for (int q=0; q<FLOW_STEP; q++)
          ival_offset[q][i] = (totocc[q]-totgen[q]);

    }
}


// try to simplify
void SimplifyComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrameSeconds,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
        float A, float SP,
        float kr, float kmax, float d,PoissonCDFApproxMemo *math_poiss)
{
    int i;
    float totocc, totgen;
//    mixed_poisson_struct mix_ctrl;
    MixtureMemo mix_memo;

    float pact,pact_new;
    float  c_dntp_top, c_dntp_bot;
    float  hplus_events_sum, hplus_events_current; // mean events per molecule, cumulative and current

    float ldt;

    A = mix_memo.Generate(A,math_poiss);
    //A = InitializeMixture(&mix_ctrl,A,MAX_HPLEN); // initialize Poisson with correct amplitude which maxes out at MAX_HPLEN
    mix_memo.ScaleMixture(SP);
    //ScaleMixture(&mix_ctrl,SP); // scale mixture fractions to proper number of molecules
    pact = mix_memo.total_live;  // active polymerases
    totocc = SP*A;  // how many hydrogens we'll eventually generate

    totgen = totocc;  // number remaining to generate

    c_dntp_bot = 0.0; // concentration of dNTP in the well
    c_dntp_top = 0.0; // concentration at top
    hplus_events_sum = hplus_events_current = 0.0; // Events per molecule

    memset(ival_offset,0,sizeof(float[my_start]));  // zero the points we don't compute

    float scaled_kr = kr*n_to_uM_conv/d; // convert molecules of polymerase to active concentraction
    float half_kr = kr *0.5; // for averaging

    // first non-zero index of the computed [dNTP] array for this nucleotide
    int c_dntp_top_ndx = my_start*SUB_STEPS;
    float c_dntp_bot_plus_kmax = 1.0/kmax;
    float c_dntp_old_effect = 0;
    float c_dntp_new_effect = 0;
    int st;

    for (i=my_start;i < npts;i++)
    {
        if (totgen > 0.0)
        {
            ldt = deltaFrameSeconds[i];
            ldt /= SUB_STEPS;
            for (st=1; (st <= SUB_STEPS) && (totgen > 0.0);st++)
            {
                // update top of well concentration in the bulk
                c_dntp_top = nuc_rise_ptr[c_dntp_top_ndx];
                c_dntp_top_ndx += 1; // jump full distance

                // assume instantaneous equilibrium within the well
                c_dntp_bot = c_dntp_top/(1+ scaled_kr*pact*c_dntp_bot_plus_kmax); // the level at which new nucs are used up as fast as they diffuse in
                c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax); // scale for michaelis-menten kinetics, assuming nucs are limiting factor

                // Now compute effect of concentration on enzyme rate
                c_dntp_old_effect = c_dntp_new_effect;
                c_dntp_new_effect = c_dntp_bot*c_dntp_bot_plus_kmax; // current effect of concentration on enzyme rate
                
                // update events per molecule
                hplus_events_current = ldt*half_kr*(c_dntp_new_effect+c_dntp_old_effect);  // events per molecule is average rate * time of rate
                hplus_events_sum += hplus_events_current;

                // how many active molecules left at end of time period given poisson process with total intensity of events
                // exp(-t) * (1+t+t^2/+t^3/6+...) where we interpolate between polynomial lengths by A
                // exp(-t) ( 1+... + frac*(t^k/k!)) where k = ceil(A-1) and frac = A-floor(A), for A>=1
                pact_new = mix_memo.GetStep(hplus_events_sum); 
                
                // how many hplus were generated
                totgen -= ((pact+pact_new) * 0.5) * hplus_events_current;  // active molecules * events per molecule
                pact = pact_new;
            }

            if (totgen < 0.0) totgen = 0.0;
        }

        ival_offset[i] = (totocc-totgen);
    }
}




// this is complex

// computes the incorporation signal for a single flow
// One of the two single most important functions
//
// basic parameters for a well and a flow
// iValOffset is the scratch space for this flow to put the cumulative signal
// nuc_rise_ptr & my_start describe the concentration this well will see at the top
// C = max concentration, not actually altered
// A is the "Amplitude" = mixture of homopolymer lengths, approximated by adjacent lengths
// SP is the current number of active copies of the template
// kr = kurrent rate for enzyme activity [depends on nuc type, possibly on flow]
// kmax = kurrent max rate for activity "Michaelis Menten",
// d = diffusion rate into the well of nucleotide [depends on nuc type, possibly on well]
// this equation works the way we want if pact is the # of active pol in the well
// c_dntp_int is in uM-seconds, and dt is in frames @ 15fps
// it calculates the number of incorporations based on the average active polymerase
// and the average dntp concentration in the well during this time step.
// note c_dntp_int is the integral of the dntp concentration in the well during the
// time step, which is equal to the average [dntp] in the well times the timestep duration
void ComplexComputeCumulativeIncorporationHydrogens(float *ival_offset, int npts, float *deltaFrameSeconds,
        float *nuc_rise_ptr, int SUB_STEPS, int my_start, float C,
        float A, float SP,
        float kr, float kmax, float d, PoissonCDFApproxMemo *math_poiss)
{
    int i;
    float totocc, totgen;

    MixtureMemo mix_memo;

    float pact,pact_new;
    float c_dntp_sum, c_dntp_bot;
    float c_dntp_top, c_dntp_int;

    float ldt;
    int st;

    // step 4
    float alpha;
    float expval;


    A = mix_memo.Generate(A,math_poiss);
    //A = InitializeMixture(&mix_ctrl,A,MAX_HPLEN); // initialize Poisson with correct amplitude which maxes out at MAX_HPLEN
    mix_memo.ScaleMixture(SP);
    //ScaleMixture(&mix_ctrl,SP); // scale mixture fractions to proper number of molecules
    pact = mix_memo.total_live;  // active polymerases
    totocc = SP*A;  // how many hydrogens we'll eventually generate

    totgen = totocc;  // number remaining to generate

    c_dntp_bot = 0.0; // concentration of dNTP in the well
    c_dntp_top = 0.0; // concentration at top
    c_dntp_sum = 0.0; // running sum of kr*[dNTP]

    // some pre-computed things
    float c_dntp_bot_plus_kmax = 1.0/kmax;
    float last_tmp1 = 0.0;
    float last_tmp2 = 0.0;
    float c_dntp_fast_inc = kr*(C/ (C+kmax));

    // [dNTP] in the well after which we switch to simpler model
    float fast_start_threshold = 0.99*C;
    float scaled_kr = kr*n_to_uM_conv;

    // first non-zero index of the computed [dNTP] array for this nucleotide
    int c_dntp_top_ndx = my_start*SUB_STEPS;

    memset(ival_offset,0,sizeof(float[my_start]));  // zero the points we don't compute

    for (i=my_start;i < npts;i++)
    {
        if (totgen > 0.0)
        {
            ldt = deltaFrameSeconds[i];

            // once the [dNTP] pretty much reaches full strength in the well
            // the math becomes much simpler
            if (c_dntp_bot > fast_start_threshold)
            {
                c_dntp_int = c_dntp_fast_inc*ldt;
                c_dntp_sum += c_dntp_int;

                pact_new = mix_memo.GetStep(c_dntp_sum);
                
                totgen -= ((pact+pact_new) /2.0) * c_dntp_int;
                pact = pact_new;
            }
            // can also use fast step math when c_dntp_bot ~ c_dntp_top
            else if (c_dntp_top > 0 && c_dntp_bot/c_dntp_top > 0.95)
            {
                c_dntp_bot = nuc_rise_ptr[c_dntp_top_ndx++];
                c_dntp_fast_inc = kr*(c_dntp_bot/ (c_dntp_bot+kmax));
                c_dntp_int = c_dntp_fast_inc*ldt;
                c_dntp_sum += c_dntp_int;

                pact_new = mix_memo.GetStep(c_dntp_sum);
                
                totgen -= ((pact+pact_new) /2.0) * c_dntp_int;
                pact = pact_new;
            }
            else
            {
                ldt /= SUB_STEPS;
                for (st=1; (st <= SUB_STEPS) && (totgen > 0.0);st++)
                {
                    c_dntp_top = nuc_rise_ptr[c_dntp_top_ndx++];

                    // we're doing the "exponential euler method" approximation for taking a time-step here
                    // so we don't need to do ultra-fine steps when the two effects are large and nearly canceling each other
                    alpha = d+scaled_kr*pact*c_dntp_bot_plus_kmax;
                    expval = ExpApprox(-alpha*ldt);

                    c_dntp_bot = c_dntp_bot*expval + d*c_dntp_top* (1-expval) /alpha;

                    c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax);
                    last_tmp1 = c_dntp_bot * c_dntp_bot_plus_kmax;
                    c_dntp_int = kr*(last_tmp2 + last_tmp1) *ldt/2.0;
                    last_tmp2 = last_tmp1;

                    c_dntp_sum += c_dntp_int;

                    // calculate new number of active polymerase
                    pact_new = mix_memo.GetStep(c_dntp_sum);
                    
                    totgen -= ((pact+pact_new)/2.0) * c_dntp_int;
                    pact = pact_new;
                }
            }

            if (totgen < 0.0) totgen = 0.0;
        }

        ival_offset[i] = (totocc-totgen);
    }
}


