/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MultiFlowModel.h"

// wrapping ways to solve multiple flows simultaneously for the predicted trace



void FillBufferParamsBlockFlows(buffer_params_block_flows *my_buff, bead_params *p, reg_params *reg_p, int *flow_ndx_map, int *buff_flow)
{
    int NucID;
    for (int fnum=0;fnum<NUMFB;fnum++)
    {
        NucID  =flow_ndx_map[fnum];
        // calculate some constants used for this flow
        my_buff->etbR[fnum] = AdjustEmptyToBeadRatioForFlow(p->R,reg_p,NucID,buff_flow[fnum]);
        my_buff->tauB[fnum] = ComputeTauBfromEmptyUsingRegionLinearModel(reg_p,my_buff->etbR[fnum]);
    }
}

void FillIncorporationParamsBlockFlows(incorporation_params_block_flows *my_inc, bead_params *p, reg_params *reg_p,int *flow_ndx_map, int *buff_flow)
{
    int NucID;

    for (int fnum=0; fnum<NUMFB; fnum++)
      reg_p->copy_multiplier[fnum] = CalculateCopyDrift(*reg_p, buff_flow[fnum]);
    for (int fnum=0; fnum<NUMFB; fnum++)
    {

        NucID=flow_ndx_map[fnum];
        my_inc->NucID[fnum] = NucID;
        my_inc->SP[fnum] = (float)(COPYMULTIPLIER * p->Copies) *reg_p->copy_multiplier[fnum];

        my_inc->sens[fnum] = reg_p->sens*SENSMULTIPLIER;
        my_inc->molecules_to_micromolar_conversion[fnum] = reg_p->molecules_to_micromolar_conversion;
        
        my_inc->d[fnum] = (reg_p->d[NucID]) *p->dmult;
        my_inc->kr[fnum] = reg_p->krate[NucID]*p->kmult[fnum];
        my_inc->kmax[fnum] = reg_p->kmax[NucID];
        my_inc->C[fnum] = reg_p->nuc_shape.C;
    }
}

void ApplyDarkMatter(float *fval,reg_params *reg_p, float *dark_matter_compensator, int *flow_ndx_map, int npts)
{
    // dark matter vectorized in different loop
    float darkness = reg_p->darkness[0];
    float *dark_matter_for_flow;
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
        dark_matter_for_flow = &dark_matter_compensator[flow_ndx_map[fnum]*npts];

        AddScaledVector(fval+fnum*npts,dark_matter_for_flow,darkness,npts);
    }
}


// 2nd-order background function with non-uniform bead well
void MultiFlowComputeIncorporationPlusBackground(float *fval,struct bead_params *p,struct reg_params *reg_p, float *ival, float *sbg, 
                                                           RegionTracker &my_regions, buffer_params_block_flows &cur_buffer_block, 
                                                           TimeCompression &time_c, flow_buffer_info &my_flow, bool use_vectorization, int bead_flow_t)
{
    float *vb_out[NUMFB];
    float *bkg_for_flow[NUMFB];
    float *new_hydrogen_for_flow[NUMFB];

    //@TODO: the natural place for vectorization is here at the flow level
    // flows are logically independent: apply "compute trace" to all flows
    // this makes for an obvious fit to solving 4 at once using the processor vectorization routines

    // parallel fill one bead parameter for block of flows
    FillBufferParamsBlockFlows(&cur_buffer_block,p,reg_p,my_flow.flow_ndx_map,my_flow.buff_flow);

    // parallel compute across flows
    for (int fnum=0;fnum<NUMFB;fnum++)
    {
        vb_out[fnum] = fval + fnum*time_c.npts;        // get ptr to start of the function evaluation for the current flow
        bkg_for_flow[fnum] = &sbg[fnum*time_c.npts];             // get ptr to pre-shifted background
        new_hydrogen_for_flow[fnum] = &ival[fnum*time_c.npts];
    }

    // do the actual computation
#ifdef __INTEL_COMPILER
    {
      for (int fnum=0; fnum<NUMFB; fnum++)
            PurpleSolveTotalTrace(vb_out[fnum],bkg_for_flow[fnum], new_hydrogen_for_flow[fnum], time_c.npts,
                                  time_c.deltaFrame, cur_buffer_block.tauB[fnum], cur_buffer_block.etbR[fnum]);
    }
#else // assumed to be GCC
    if (use_vectorization) {
        PurpleSolveTotalTrace_Vec(NUMFB, vb_out, bkg_for_flow, new_hydrogen_for_flow, time_c.npts,
                                  time_c.deltaFrame, cur_buffer_block.tauB, cur_buffer_block.etbR, p->gain);
    } else {
        for (int fnum=0; fnum<NUMFB; fnum++)
            PurpleSolveTotalTrace(vb_out[fnum],bkg_for_flow[fnum], new_hydrogen_for_flow[fnum], time_c.npts,
                                  time_c.deltaFrame, cur_buffer_block.tauB[fnum], cur_buffer_block.etbR[fnum]);
    }
#endif

    // adjust for well sensitivity, unexplained systematic effects
    // gain naturally parallel across flows
    MultiplyVectorByScalar(fval,p->gain,bead_flow_t);

    // Dark Matter is extra background term of unexplained origin
    // Possibly should be applied directly to the observed signal rather than synthesized here inside a loop.
    ApplyDarkMatter(fval,reg_p,my_regions.missing_mass.dark_matter_compensator,my_flow.flow_ndx_map,time_c.npts);
}


void MultiFlowComputeCumulativeIncorporationSignal(struct bead_params *p,struct reg_params *reg_p, float *ivalPtr,
                                                             RegionTracker &my_regions, incorporation_params_block_flows &cur_bead_block, 
                                                             TimeCompression &time_c, flow_buffer_info &my_flow,  PoissonCDFApproxMemo *math_poiss )
{
  // only side effect should be new values in ivalPtr

    // this is region wide
    //@TODO this should be exported to the places we switch regions so it is not recomputed
    my_regions.cache_step.CalculateNucRiseCoarseStep(reg_p,time_c,my_flow);
    // pretend I'm making a parallel process
    FillIncorporationParamsBlockFlows(&cur_bead_block, p,reg_p,my_flow.flow_ndx_map,my_flow.buff_flow);
    // "In parallel, across flows"
    float* nuc_rise_ptr[NUMFB];
    float* incorporation_rise[NUMFB];
    int my_start[NUMFB];
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      nuc_rise_ptr[fnum]      = my_regions.cache_step.NucCoarseStep(fnum);
      incorporation_rise[fnum]= &ivalPtr[fnum*time_c.npts];
      my_start[fnum] = my_regions.cache_step.i_start_coarse_step[fnum];
    }
    // this is >almost< a parallel operation by flows now
    bool use_my_parallel=true;
    if (use_my_parallel){
      //@TODO handle cases of fewer than 4 flows remaining
      for (int fnum=0; fnum<NUMFB; fnum+=4)
      {

        ParallelSimpleComputeCumulativeIncorporationHydrogens(&incorporation_rise[fnum], time_c.npts, time_c.deltaFrameSeconds,&nuc_rise_ptr[fnum],
                                                              ISIG_SUB_STEPS_MULTI_FLOW,  &my_start[fnum], &p->Ampl[fnum],
                                                              &cur_bead_block.SP[fnum],&cur_bead_block.kr[fnum], &cur_bead_block.kmax[fnum], &cur_bead_block.d[fnum], &cur_bead_block.molecules_to_micromolar_conversion[fnum], math_poiss);
        for (int q=0; q<4; q++)  
             MultiplyVectorByScalar(incorporation_rise[fnum+q], cur_bead_block.sens[fnum+q],time_c.npts);  // transform hydrogens to signal
      } 
    } else {
      for (int fnum=0;fnum<NUMFB;fnum++)
      { 
          ComputeCumulativeIncorporationHydrogens(incorporation_rise[fnum], time_c.npts,
                                                time_c.deltaFrameSeconds, nuc_rise_ptr[fnum], ISIG_SUB_STEPS_MULTI_FLOW, my_start[fnum],
                                                cur_bead_block.C[fnum], p->Ampl[fnum], cur_bead_block.SP[fnum], cur_bead_block.kr[fnum], cur_bead_block.kmax[fnum], cur_bead_block.d[fnum],cur_bead_block.molecules_to_micromolar_conversion[fnum], math_poiss);

          MultiplyVectorByScalar(incorporation_rise[fnum], cur_bead_block.sens[fnum],time_c.npts);  // transform hydrogens to signal
      }
    }
}



void MultiCorrectBeadBkg(float *block_signal_corrected, bead_params *p,
                         BeadScratchSpace &my_scratch, flow_buffer_info &my_flow, TimeCompression &time_c, RegionTracker &my_regions, float *sbg, bool use_vectorization)
{
  float vb[my_scratch.bead_flow_t];
  float* vb_out[my_flow.numfb];
  float* sbgPtr[my_flow.numfb];
  float block_bkg_plus_xtalk[my_scratch.bead_flow_t]; // set up instead of shifted background
  memset (vb,0,sizeof (float[my_scratch.bead_flow_t]));

  // add cross-talk for this bead to the empty-trace
  CopyVector (block_bkg_plus_xtalk,sbg,my_scratch.bead_flow_t);
  AccumulateVector (block_bkg_plus_xtalk,my_scratch.cur_xtflux_block,my_scratch.bead_flow_t);

  // compute the zeromer
  // setup pointers into the arrays
  for (int fnum=0; fnum<my_flow.numfb; fnum++)
  {
    // remove zeromer background - just like oneFlowFit.
    // should include xtalk (0) so I can reuse this routine
    sbgPtr[fnum] = &block_bkg_plus_xtalk[fnum*time_c.npts];
    vb_out[fnum] = &vb[fnum*time_c.npts];
  }

  // do the actual calculation in parallel or not
#ifdef __INTEL_COMPILER
  {
    for (int fnum=0; fnum<my_flow.numfb; fnum++)
      BlueSolveBackgroundTrace (vb_out[fnum],sbgPtr[fnum],time_c.npts,time_c.deltaFrame,
                                my_scratch.cur_buffer_block.tauB[fnum],my_scratch.cur_buffer_block.etbR[fnum]);
  }
#else
  if (use_vectorization)
  {
    BlueSolveBackgroundTrace_Vec (my_flow.numfb, vb_out, sbgPtr, time_c.npts, time_c.deltaFrame,
                                  my_scratch.cur_buffer_block.tauB, my_scratch.cur_buffer_block.etbR);
  }
  else
  {
    for (int fnum=0; fnum<my_flow.numfb; fnum++)
      BlueSolveBackgroundTrace (vb_out[fnum],sbgPtr[fnum],time_c.npts,time_c.deltaFrame,
                                my_scratch.cur_buffer_block.tauB[fnum],my_scratch.cur_buffer_block.etbR[fnum]);
  }
#endif

  MultiplyVectorByScalar (vb,p->gain,my_scratch.bead_flow_t);
  
  ApplyDarkMatter (vb,&my_regions.rp, my_regions.missing_mass.dark_matter_compensator,my_flow.flow_ndx_map,time_c.npts);

  // zeromer computed, now remove from observed
  DiminishVector (block_signal_corrected,vb,my_scratch.bead_flow_t); // remove calculated background to produce corrected signal

}


// Multiflow: Solve incorporation
// Solve lost hydrogens to bulk
// solve bulk resistance to lost hydrogens
// this function too large, should be componentized
void AccumulateSingleNeighborXtalkTrace(float *my_xtflux, bead_params *p, reg_params *reg_p,
                                                  BeadScratchSpace &my_scratch, TimeCompression &time_c, RegionTracker &my_regions, flow_buffer_info my_flow,
                                                  PoissonCDFApproxMemo *math_poiss, bool use_vectorization,
                                                  float tau_top, float tau_bulk, float multiplier)
{

    // Compute the hydrogen signal of xtalk in the bulk fluid above the affected well
    // 1) Xtalk happens fast -> we compute the cumulative lost hydrogen ions at the top of the bead instead of at the bottom (tau_top)
    // 1a) xtalk happens as the well loses hydrogen ions so the cumulative lost = cumulative total generated - number in well currently
    // 2) hydrogen ions are "retained" by the bulk for a finite amount of time as they are swept along
    // 2a) this 'retainer' may be asymmetric (tau_bulk) models the decay rate
    // 3) multiplier: hydrogen ions spread out, so are of different proportion to the original signal

    // this is pre-calculated outside for the current region parameters
    //my_regions.cache_step.CalculateNucRiseCoarseStep(reg_p,time_c);


    // over-ride buffering parameters for bead
    // use the same incorporation parameters, though
    FillIncorporationParamsBlockFlows(&my_scratch.cur_bead_block, p,reg_p,my_flow.flow_ndx_map,my_flow.buff_flow);
    //params_IncrementHits(p);

    float block_model_trace[my_scratch.bead_flow_t], block_incorporation_rise[my_scratch.bead_flow_t];

    // "In parallel, across flows"
    float* nuc_rise_ptr[NUMFB];
    float* model_trace[NUMFB];
    float* incorporation_rise[NUMFB];
    // should this be using cur_buffer_block as usual?
    float vec_tau_top[NUMFB];
    float vec_tau_bulk[NUMFB];

    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      vec_tau_top[fnum] = tau_top;
      vec_tau_bulk[fnum] = tau_bulk;
    }

    // setup parallel pointers into the structure
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
        nuc_rise_ptr[fnum] = my_regions.cache_step.NucCoarseStep(fnum);
        model_trace[fnum] = &block_model_trace[fnum*time_c.npts];
        incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c.npts];    // set up each flow information
    }
    // fill in each flow incorporation
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
        // compute newly generated ions for the amplitude of each flow
        ComputeCumulativeIncorporationHydrogens(incorporation_rise[fnum],
                                                time_c.npts, time_c.deltaFrameSeconds, nuc_rise_ptr[fnum], ISIG_SUB_STEPS_MULTI_FLOW, my_regions.cache_step.i_start_coarse_step[fnum],
                                                my_scratch.cur_bead_block.C[fnum], p->Ampl[fnum], my_scratch.cur_bead_block.SP[fnum], my_scratch.cur_bead_block.kr[fnum], my_scratch.cur_bead_block.kmax[fnum], my_scratch.cur_bead_block.d[fnum],my_scratch.cur_bead_block.molecules_to_micromolar_conversion[fnum], math_poiss);
        MultiplyVectorByScalar(incorporation_rise[fnum], my_scratch.cur_bead_block.sens[fnum],time_c.npts);  // transform hydrogens to signal       // variables used for solving background signal shape
    }

    // Now solve the top of the well cumulative lost hydrogen ions
    // happen faster, hence tau_top
#ifdef __INTEL_COMPILER
    {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_top); // we lose hydrogen ions fast!

            DiminishVector(incorporation_rise[fnum],model_trace[fnum],time_c.npts);  // cumulative lost hydrogen ions instead of retained hydrogen ions
  }
    }
#else
    if (use_vectorization) {
        RedSolveHydrogenFlowInWell_Vec(NUMFB,model_trace,incorporation_rise,time_c.npts,time_c.deltaFrame,vec_tau_top); // we lose hydrogen ions fast!

        DiminishVector(block_incorporation_rise,block_model_trace,my_scratch.bead_flow_t);  // cumulative lost hydrogen ions instead of retained hydrogen ions

    } else {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_top); // we lose hydrogen ions fast!

            DiminishVector(incorporation_rise[fnum],model_trace[fnum],time_c.npts);  // cumulative lost hydrogen ions instead of retained hydrogen ions
        }
    }
#endif

    // finally solve the way hydrogen ions diffuse out of the bulk
#ifdef __INTEL_COMPILER
    {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            // Now solve the bulk
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,
                                       my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_bulk);  // we retain hydrogen ions variably in the bulk depending on direction
        }
    }
#else
    if (use_vectorization){
        // Now solve the bulk
        RedSolveHydrogenFlowInWell_Vec(NUMFB,model_trace,incorporation_rise,time_c.npts,time_c.deltaFrame,vec_tau_bulk);

    } else {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            // Now solve the bulk
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,
                                       my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_bulk);  // we retain hydrogen ions variably in the bulk depending on direction

        }
    }
#endif

    // universal
    MultiplyVectorByScalar(block_model_trace,multiplier,my_scratch.bead_flow_t); // scale down the quantity of ions

    // add to the bulk cross-talk we're creating
    AccumulateVector(my_xtflux, block_model_trace,my_scratch.bead_flow_t);
}



// Multiflow: Solve incorporation
// Solve lost hydrogens to bulk
// solve bulk resistance to lost hydrogens
// this function too large, should be componentized
void AccumulateSingleNeighborExcessHydrogen(float *my_xtflux, float *neighbor_signal, bead_params *p, reg_params *reg_p,
                                                  BeadScratchSpace &my_scratch, TimeCompression &time_c,
                                                  RegionTracker &my_regions, flow_buffer_info my_flow,
                                                   bool use_vectorization,
                                                  float tau_top, float tau_bulk, float multiplier)
{

    // Compute the hydrogen signal of xtalk in the bulk fluid above the affected well
    // 1) Xtalk happens fast -> we compute the cumulative lost hydrogen ions at the top of the bead instead of at the bottom (tau_top)
    // 1a) xtalk happens as the well loses hydrogen ions so the cumulative lost = cumulative total generated - number in well currently
    // 2) hydrogen ions are "retained" by the bulk for a finite amount of time as they are swept along
    // 2a) this 'retainer' may be asymmetric (tau_bulk) models the decay rate
    // 3) multiplier: hydrogen ions spread out, so are of different proportion to the original signal

    // this is pre-calculated outside for the current region parameters
    //my_regions.cache_step.CalculateNucRiseCoarseStep(reg_p,time_c);


    // over-ride buffering parameters for bead
    // use the same incorporation parameters, though

    FillBufferParamsBlockFlows(&my_scratch.cur_buffer_block,p,reg_p,my_flow.flow_ndx_map,my_flow.buff_flow);

    //params_IncrementHits(p);

    float block_model_trace[my_scratch.bead_flow_t], block_incorporation_rise[my_scratch.bead_flow_t];

    // "In parallel, across flows"

    float* model_trace[NUMFB];
    float* incorporation_rise[NUMFB];
    // should this be using cur_buffer_block as usual?
    float vec_tau_top[NUMFB];
    float vec_tau_bulk[NUMFB];

    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      vec_tau_top[fnum] = tau_top;
      vec_tau_bulk[fnum] = tau_bulk;
    }

    // setup parallel pointers into the structure
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
       
        model_trace[fnum] = &block_model_trace[fnum*time_c.npts];
        incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c.npts];    // set up each flow information
    }
    // fill in each flow incorporation
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      // grab my excess hydrogen ions from the observed signal
      IntegrateRedFromObservedTotalTrace(incorporation_rise[fnum], &neighbor_signal[fnum*time_c.npts], &my_scratch.shifted_bkg[fnum*time_c.npts], time_c.npts,time_c.deltaFrame, my_scratch.cur_buffer_block.tauB[fnum], my_scratch.cur_buffer_block.etbR[fnum]);
      // variables used for solving background signal shape
    }

    // Now solve the top of the well cumulative lost hydrogen ions
    // happen faster, hence tau_top
#ifdef __INTEL_COMPILER
    {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_top); // we lose hydrogen ions fast!

            DiminishVector(incorporation_rise[fnum],model_trace[fnum],time_c.npts);  // cumulative lost hydrogen ions instead of retained hydrogen ions
  }
    }
#else
    if (use_vectorization) {
        RedSolveHydrogenFlowInWell_Vec(NUMFB,model_trace,incorporation_rise,time_c.npts,time_c.deltaFrame,vec_tau_top); // we lose hydrogen ions fast!

        DiminishVector(block_incorporation_rise,block_model_trace,my_scratch.bead_flow_t);  // cumulative lost hydrogen ions instead of retained hydrogen ions

    } else {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_top); // we lose hydrogen ions fast!

            DiminishVector(incorporation_rise[fnum],model_trace[fnum],time_c.npts);  // cumulative lost hydrogen ions instead of retained hydrogen ions
        }
    }
#endif

    // finally solve the way hydrogen ions diffuse out of the bulk
#ifdef __INTEL_COMPILER
    {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            // Now solve the bulk
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,
                                       my_regions.cache_step.i_start_coarse_step[my_flow.flow_ndx_map[fnum]],time_c.deltaFrame,tau_bulk);  // we retain hydrogen ions variably in the bulk depending on direction
        }
    }
#else
    if (use_vectorization){
        // Now solve the bulk
        RedSolveHydrogenFlowInWell_Vec(NUMFB,model_trace,incorporation_rise,time_c.npts,time_c.deltaFrame,vec_tau_bulk);

    } else {
        for (int fnum=0; fnum<NUMFB; fnum++) {
            // Now solve the bulk
            RedSolveHydrogenFlowInWell(model_trace[fnum],incorporation_rise[fnum],time_c.npts,
                                       my_regions.cache_step.i_start_coarse_step[fnum],time_c.deltaFrame,tau_bulk);  // we retain hydrogen ions variably in the bulk depending on direction

        }
    }
#endif

    // universal
    MultiplyVectorByScalar(block_model_trace,multiplier,my_scratch.bead_flow_t); // scale down the quantity of ions

    // add to the bulk cross-talk we're creating
    AccumulateVector(my_xtflux, block_model_trace,my_scratch.bead_flow_t);
}
