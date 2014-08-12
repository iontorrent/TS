/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MultiFlowModel.h"

using namespace std;

// wrapping ways to solve multiple flows simultaneously for the predicted trace



void MathModel::FillBufferParamsBlockFlows ( 
    buffer_params_block_flows *my_buff, BeadParams *p, const reg_params *reg_p, 
    const int *flow_ndx_map, 
    int flow_block_start,
    int flow_block_size
  )
{
  int NucID;
  for ( int fnum=0;fnum<flow_block_size;fnum++ )
  {
    NucID  =flow_ndx_map[fnum];
    // calculate some constants used for this flow
    my_buff->etbR[fnum] = reg_p->AdjustEmptyToBeadRatioForFlow ( p->R, p->Ampl[fnum], p->Copies, p->phi, NucID, flow_block_start + fnum );
    my_buff->tauB[fnum] = reg_p->ComputeTauBfromEmptyUsingRegionLinearModel ( my_buff->etbR[fnum] );
  }
}

void MathModel::FillIncorporationParamsBlockFlows ( 
    incorporation_params_block_flows *my_inc, BeadParams *p, reg_params *reg_p,
    const int *flow_ndx_map, int flow_block_start,
    int flow_block_size
  )
{
  int NucID;

  for ( int fnum=0; fnum<flow_block_size; fnum++ )
    reg_p->copy_multiplier[fnum] = reg_p->CalculateCopyDrift ( flow_block_start + fnum );
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {

    NucID=flow_ndx_map[fnum];
    my_inc->NucID[fnum] = NucID;
    my_inc->SP[fnum] = ( float ) ( COPYMULTIPLIER * p->Copies ) *reg_p->copy_multiplier[fnum];

    my_inc->sens[fnum] = reg_p->sens*SENSMULTIPLIER;
    my_inc->molecules_to_micromolar_conversion[fnum] = reg_p->molecules_to_micromolar_conversion;

    my_inc->d[fnum] = ( reg_p->d[NucID] ) *p->dmult;
    my_inc->kr[fnum] = reg_p->krate[NucID]*p->kmult[fnum];
    my_inc->kmax[fnum] = reg_p->kmax[NucID];
    my_inc->C[fnum] = reg_p->nuc_shape.C[NucID];
  }
}

void MathModel::ApplyDarkMatter ( float *fval,const reg_params *reg_p, 
    const vector<float>& dark_matter_compensator, 
    const int *flow_ndx_map, int npts,
    int flow_block_size )
{
  // dark matter vectorized in different loop
  float darkness = reg_p->darkness[0];
  const float *dark_matter_for_flow;
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    dark_matter_for_flow = &dark_matter_compensator[flow_ndx_map[fnum]*npts];

    AddScaledVector ( fval+fnum*npts,dark_matter_for_flow,darkness,npts );
  }
}

void MathModel::ApplyPCADarkMatter ( float *fval,BeadParams *p, 
    const vector<float>& dark_matter_compensator, int npts,
    int flow_block_size )
{
  // dark matter vectorized in different loop
  float dm[npts];
  memset(dm,0,sizeof(dm));
  for (int icomp=0;icomp < NUM_DM_PCA;icomp++)
     AddScaledVector( dm,&dark_matter_compensator[icomp*npts],p->pca_vals[icomp],npts);

  for ( int fnum=0; fnum<flow_block_size; fnum++ )
      AddScaledVector ( fval+fnum*npts,dm,1.0f,npts );
}


// 2nd-order background function with non-uniform bead well
void MathModel::MultiFlowComputeTraceGivenIncorporationAndBackground ( 
    float *fval, BeadParams *p,const reg_params *reg_p, float *ival, float *sbg,
    RegionTracker &my_regions, buffer_params_block_flows &cur_buffer_block,
    const TimeCompression &time_c, const FlowBufferInfo &my_flow, 
    bool use_vectorization, int bead_flow_t,
    int flow_block_size, int flow_block_start )
{
  // Save on allocation complexity; allocate enough to get to the end.
  float **vb_out = new float*[ flow_block_size ];
  float **bkg_for_flow = new float*[ flow_block_size ];
  float **new_hydrogen_for_flow = new float*[ flow_block_size ];

  //@TODO: the natural place for vectorization is here at the flow level
  // flows are logically independent: apply "compute trace" to all flows
  // this makes for an obvious fit to solving 4 at once using the processor vectorization routines

  // parallel fill one bead parameter for block of flows
  FillBufferParamsBlockFlows ( &cur_buffer_block,p,reg_p,my_flow.flow_ndx_map, flow_block_start, 
    flow_block_size );

  // parallel compute across flows
  for ( int fnum=0;fnum<flow_block_size;fnum++ )
  {
    vb_out[fnum] = fval + fnum*time_c.npts();        // get ptr to start of the function evaluation for the current flow
    bkg_for_flow[fnum] = &sbg[fnum*time_c.npts() ];            // get ptr to pre-shifted background
    new_hydrogen_for_flow[fnum] = &ival[fnum*time_c.npts() ];
  }

  // do the actual computation
#ifdef __INTEL_COMPILER
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
      PurpleSolveTotalTrace ( vb_out[fnum],bkg_for_flow[fnum], new_hydrogen_for_flow[fnum], time_c.npts(),
                              &time_c.deltaFrame[0], cur_buffer_block.tauB[fnum], cur_buffer_block.etbR[fnum] );
  }
#else // assumed to be GCC
  if ( use_vectorization )
  {
    MathModel::PurpleSolveTotalTrace_Vec ( vb_out, bkg_for_flow, new_hydrogen_for_flow, 
                  time_c.npts(), &time_c.deltaFrame[0], 
                  cur_buffer_block.tauB, cur_buffer_block.etbR, p->gain, flow_block_size );
  }
  else
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
      PurpleSolveTotalTrace ( vb_out[fnum],bkg_for_flow[fnum], new_hydrogen_for_flow[fnum],time_c.npts(),
                              &time_c.deltaFrame[0], cur_buffer_block.tauB[fnum], cur_buffer_block.etbR[fnum] );
  }
#endif

  // adjust for well sensitivity, unexplained systematic effects
  // gain naturally parallel across flows
  MultiplyVectorByScalar ( fval,p->gain,bead_flow_t );

  // Dark Matter is extra background term of unexplained origin
  // Possibly should be applied directly to the observed signal rather than synthesized here inside a loop.
  if (my_regions.missing_mass.mytype == PerNucAverage)
  {
    // used to rely on "darkness" being 0.0 when this happened.
    // making this trap more explicit.
    if (!my_regions.missing_mass.training_only)
     ApplyDarkMatter ( fval,reg_p,my_regions.missing_mass.dark_matter_compensator,my_flow.flow_ndx_map,time_c.npts(), flow_block_size );
  }

  if (my_regions.missing_mass.mytype==PCAVector)
     ApplyPCADarkMatter ( fval,p,my_regions.missing_mass.dark_matter_compensator,time_c.npts(), flow_block_size );

  // Cleanup.
  delete [] vb_out;
  delete [] bkg_for_flow;
  delete [] new_hydrogen_for_flow;
}


void MathModel::MultiFlowComputeCumulativeIncorporationSignal ( 
    struct BeadParams *p,struct reg_params *reg_p, float *ivalPtr,
    NucStep &cache_step, incorporation_params_block_flows &cur_bead_block,
    const TimeCompression &time_c, const FlowBufferInfo &my_flow,  
    PoissonCDFApproxMemo *math_poiss, int flow_block_size,
    int flow_block_start )
{
  // only side effect should be new values in ivalPtr

  // this is region wide
  //This will short-circuit if has been computed
  cache_step.CalculateNucRiseCoarseStep ( reg_p,time_c,my_flow );

  // pretend I'm making a parallel process
  FillIncorporationParamsBlockFlows ( &cur_bead_block, p,reg_p,my_flow.flow_ndx_map,
                                      flow_block_start, flow_block_size );

  // "In parallel, across flows"
  const float** nuc_rise_ptr = new const float *[flow_block_size];
  float** incorporation_rise = new float *[flow_block_size];
  int* my_start = new int[flow_block_size];
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    nuc_rise_ptr[fnum]       = cache_step.NucCoarseStep ( fnum );
    incorporation_rise[fnum] = &ivalPtr[fnum*time_c.npts() ];
    my_start[fnum]           = cache_step.i_start_coarse_step[fnum];
  }
  // this is >almost< a parallel operation by flows now
  bool use_my_parallel= (flow_block_size%4 == 0);
  if ( use_my_parallel )
  {
    //@TODO handle cases of fewer than 4 flows remaining
    for ( int fnum=0; fnum<flow_block_size; fnum+=4 )
    {
      MathModel::ParallelSimpleComputeCumulativeIncorporationHydrogens ( 
          &incorporation_rise[fnum], time_c.npts(), &time_c.deltaFrameSeconds[0],
          &nuc_rise_ptr[fnum],
          ISIG_SUB_STEPS_MULTI_FLOW,  &my_start[fnum], &p->Ampl[fnum],
          &cur_bead_block.SP[fnum],&cur_bead_block.kr[fnum], &cur_bead_block.kmax[fnum], 
          &cur_bead_block.d[fnum], &cur_bead_block.molecules_to_micromolar_conversion[fnum], 
          math_poiss, reg_p->hydrogenModelType );
    }
  }
  else
  {
    for ( int fnum=0;fnum<flow_block_size;fnum++ )
    {
      MathModel::ComputeCumulativeIncorporationHydrogens ( incorporation_rise[fnum], time_c.npts(),
          &time_c.deltaFrameSeconds[0], nuc_rise_ptr[fnum], ISIG_SUB_STEPS_MULTI_FLOW, 
          my_start[fnum], cur_bead_block.C[fnum], p->Ampl[fnum], cur_bead_block.SP[fnum], 
          cur_bead_block.kr[fnum], cur_bead_block.kmax[fnum], 
          cur_bead_block.d[fnum],cur_bead_block.molecules_to_micromolar_conversion[fnum], 
          math_poiss, reg_p->hydrogenModelType );
    }
  }

  // transform hydrogens to signal
  for ( int fnum=0;fnum<flow_block_size;fnum++ )
      MultiplyVectorByScalar ( incorporation_rise[fnum], cur_bead_block.sens[fnum],time_c.npts() );

  // Cleanup.
  delete [] nuc_rise_ptr;
  delete [] incorporation_rise;
  delete [] my_start;
}



void MathModel::MultiCorrectBeadBkg ( 
    float *block_signal_corrected, BeadParams *p,
    const BeadScratchSpace &my_scratch, 
    const buffer_params_block_flows &my_cur_buffer_block,
    const FlowBufferInfo &my_flow, 
    const TimeCompression &time_c, const RegionTracker &my_regions, float *sbg, 
    bool use_vectorization, int flow_block_size )
{
  float vb[my_scratch.bead_flow_t];
  float* vb_out[flow_block_size];
  float* sbgPtr[flow_block_size];
  float block_bkg_plus_xtalk[my_scratch.bead_flow_t]; // set up instead of shifted background
  memset ( vb,0,sizeof ( float[my_scratch.bead_flow_t] ) );

  // add cross-talk for this bead to the empty-trace
  CopyVector ( block_bkg_plus_xtalk,sbg,my_scratch.bead_flow_t );
  AccumulateVector ( block_bkg_plus_xtalk,my_scratch.cur_xtflux_block,my_scratch.bead_flow_t );

  // compute the zeromer
  // setup pointers into the arrays
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    // remove zeromer background - just like oneFlowFit.
    // should include xtalk (0) so I can reuse this routine
    sbgPtr[fnum] = &block_bkg_plus_xtalk[fnum*time_c.npts() ];
    vb_out[fnum] = &vb[fnum*time_c.npts() ];
  }

  // do the actual calculation in parallel or not
#ifdef __INTEL_COMPILER
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
      BlueSolveBackgroundTrace ( vb_out[fnum],sbgPtr[fnum],time_c.npts(),&time_c.deltaFrame[0],
                                 my_cur_buffer_block.tauB[fnum],my_cur_buffer_block.etbR[fnum] );
  }
#else
  if ( use_vectorization )
  {
    MathModel::BlueSolveBackgroundTrace_Vec ( vb_out, sbgPtr, time_c.npts(), &time_c.deltaFrame[0],
                                   my_cur_buffer_block.tauB, my_cur_buffer_block.etbR, flow_block_size );
  }
  else
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
      BlueSolveBackgroundTrace ( vb_out[fnum],sbgPtr[fnum],time_c.npts(),&time_c.deltaFrame[0],
                                 my_cur_buffer_block.tauB[fnum],my_cur_buffer_block.etbR[fnum] );
  }
#endif

  MultiplyVectorByScalar ( vb,p->gain,my_scratch.bead_flow_t );

  if (my_regions.missing_mass.mytype == PerNucAverage)
     ApplyDarkMatter ( vb,&my_regions.rp, my_regions.missing_mass.dark_matter_compensator,my_flow.flow_ndx_map,time_c.npts(), flow_block_size );
  else
     ApplyPCADarkMatter ( vb,p,my_regions.missing_mass.dark_matter_compensator,time_c.npts(), flow_block_size );

  // zeromer computed, now remove from observed
  DiminishVector ( block_signal_corrected,vb,my_scratch.bead_flow_t ); // remove calculated background to produce corrected signal

}


static void IonsFromBulk ( float **model_trace, float **incorporation_rise,
                    TimeCompression &time_c, RegionTracker &my_regions, const FlowBufferInfo & my_flow,
                    bool use_vectorization,
                    float *vec_tau_bulk, int flow_block_size )
{
  // finally solve the way hydrogen ions diffuse out of the bulk
#ifdef __INTEL_COMPILER
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
    {
      // Now solve the bulk
      MathModel::RedSolveHydrogenFlowInWell ( model_trace[fnum],incorporation_rise[fnum],
        time_c.npts(), my_regions.cache_step.i_start_coarse_step[my_flow.flow_ndx_map[fnum]],
        &time_c.deltaFrame[0],vec_tau_bulk[fnum] ); 
        // we retain hydrogen ions variably in the bulk depending on direction
    }
  }
#else
  if ( use_vectorization )
  {
    // Now solve the bulk
    MathModel::RedSolveHydrogenFlowInWell_Vec ( model_trace,incorporation_rise,time_c.npts(),
        &time_c.deltaFrame[0],vec_tau_bulk, flow_block_size );

  }
  else
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
    {
      // Now solve the bulk
      MathModel::RedSolveHydrogenFlowInWell ( model_trace[fnum],incorporation_rise[fnum],
        time_c.npts(), my_regions.cache_step.i_start_coarse_step[fnum],
        &time_c.deltaFrame[0],vec_tau_bulk[fnum] ); 
        // we retain hydrogen ions variably in the bulk depending on direction
    }
  }
#endif
}


// note: input is incorporation_rise, returns lost_hydrogens in the same buffer, recycling the memory
static void CumulativeLostHydrogens ( float **incorporation_rise_to_lost_hydrogens, float **scratch_trace, 
                               TimeCompression &time_c, RegionTracker &my_regions,
                               bool use_vectorization,
                               float *vec_tau_top, 
                               int flow_block_size )
{
  // Put the model trace from buffering the incorporation_rise into scratch_trace
#ifdef __INTEL_COMPILER
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
    {
      MathModel::RedSolveHydrogenFlowInWell ( scratch_trace[fnum],incorporation_rise_to_lost_hydrogens[fnum],time_c.npts(),my_regions.cache_step.i_start_coarse_step[fnum],&time_c.deltaFrame[0],vec_tau_top[fnum] ); // we lose hydrogen ions fast!

    }
  }
#else
  if ( use_vectorization )
  {
    MathModel::RedSolveHydrogenFlowInWell_Vec ( scratch_trace,incorporation_rise_to_lost_hydrogens,time_c.npts(),&time_c.deltaFrame[0],vec_tau_top, flow_block_size ); // we lose hydrogen ions fast!

  }
  else
  {
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
    {
      MathModel::RedSolveHydrogenFlowInWell ( scratch_trace[fnum],incorporation_rise_to_lost_hydrogens[fnum],time_c.npts(),my_regions.cache_step.i_start_coarse_step[fnum],&time_c.deltaFrame[0],vec_tau_top[fnum] ); // we lose hydrogen ions fast!

    }
  }
#endif

  // return lost_hydrogens in the incorporation_rise variables by subtracting the trace from the cumulative
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
    DiminishVector ( incorporation_rise_to_lost_hydrogens[fnum],scratch_trace[fnum],time_c.npts() );  // cumulative lost hydrogen ions instead of retained hydrogen ions

}

static void IncorporationRiseFromNeighborParameters ( 
    float **incorporation_rise, const float * const *nuc_rise_ptr,
    BeadParams *p,
    TimeCompression &time_c, RegionTracker &my_regions,
    BeadScratchSpace &my_scratch, 
    incorporation_params_block_flows &my_cur_bead_block,
    PoissonCDFApproxMemo *math_poiss, int hydrogenModelType,
    int flow_block_size)
{
  // fill in each flow incorporation
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    // compute newly generated ions for the amplitude of each flow
    MathModel::ComputeCumulativeIncorporationHydrogens ( incorporation_rise[fnum],
        time_c.npts(), &time_c.deltaFrameSeconds[0], nuc_rise_ptr[fnum], ISIG_SUB_STEPS_MULTI_FLOW, my_regions.cache_step.i_start_coarse_step[fnum],
        my_cur_bead_block.C[fnum], p->Ampl[fnum], my_cur_bead_block.SP[fnum], my_cur_bead_block.kr[fnum], my_cur_bead_block.kmax[fnum], my_cur_bead_block.d[fnum],my_cur_bead_block.molecules_to_micromolar_conversion[fnum], math_poiss, hydrogenModelType );
    MultiplyVectorByScalar ( incorporation_rise[fnum], my_cur_bead_block.sens[fnum],time_c.npts() );  // transform hydrogens to signal       // variables used for solving background signal shape
  }
}

// rescale final trace by differences in buffering between wells
// wells with more buffering shield from the total hydrogens floating around
static void RescaleTraceByBuffering ( float **my_trace, float *vec_tau_source, float *vec_tau_dest, int npts, int flow_block_size )
{
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    MultiplyVectorByScalar ( my_trace[fnum],vec_tau_source[fnum],npts );
    MultiplyVectorByScalar ( my_trace[fnum],1.0f/vec_tau_dest[fnum],npts );
  }
}


// Multiflow: Solve incorporation
// Solve lost hydrogens to bulk
// solve bulk resistance to lost hydrogens
// this function too large, should be componentized
void MathModel::AccumulateSingleNeighborXtalkTrace ( float *my_xtflux, BeadParams *p, reg_params *reg_p,
    BeadScratchSpace &my_scratch, 
    incorporation_params_block_flows & my_cur_bead_block,
    TimeCompression &time_c, RegionTracker &my_regions, const FlowBufferInfo & my_flow,
    PoissonCDFApproxMemo *math_poiss, bool use_vectorization,
    float tau_top, float tau_bulk, float multiplier,
    int flow_block_size, int flow_block_start )
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
  FillIncorporationParamsBlockFlows ( &my_cur_bead_block, p,reg_p,my_flow.flow_ndx_map,
    flow_block_start, flow_block_size );
  //params_IncrementHits(p);

  float block_model_trace[my_scratch.bead_flow_t], block_incorporation_rise[my_scratch.bead_flow_t];

  // "In parallel, across flows"
  const float* nuc_rise_ptr[flow_block_size];
  float* model_trace[flow_block_size];
  float* scratch_trace[flow_block_size];
  float* incorporation_rise[flow_block_size];
  float* lost_hydrogens[flow_block_size];
  // should this be using cur_buffer_block as usual?
  float vec_tau_top[flow_block_size];
  float vec_tau_bulk[flow_block_size];

  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    vec_tau_top[fnum] = tau_top;
    vec_tau_bulk[fnum] = tau_bulk;
  }

  // setup parallel pointers into the structure
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    nuc_rise_ptr[fnum] = my_regions.cache_step.NucCoarseStep ( fnum );
    scratch_trace[fnum]=model_trace[fnum] = &block_model_trace[fnum*time_c.npts() ];
    lost_hydrogens[fnum]=incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c.npts() ];   // set up each flow information
  }

  IncorporationRiseFromNeighborParameters ( incorporation_rise, nuc_rise_ptr, p, time_c, my_regions, my_scratch, my_cur_bead_block, math_poiss, reg_p->hydrogenModelType, flow_block_size );
  // temporarily use the model_trace memory structure as scratch space
  // turn incorporation_rise into lost hydrogens
  CumulativeLostHydrogens ( incorporation_rise, scratch_trace,time_c, my_regions, use_vectorization, vec_tau_top, flow_block_size );
  // lost_hydrogens = incorporation_rise
  // now fill in the model_trace structure for real, overwriting any temporary use of that space
  IonsFromBulk ( model_trace,lost_hydrogens, time_c, my_regions, my_flow, use_vectorization, vec_tau_bulk, flow_block_size );

  // universal
  MultiplyVectorByScalar ( block_model_trace,multiplier,my_scratch.bead_flow_t ); // scale down the quantity of ions

  // add to the bulk cross-talk we're creating
  AccumulateVector ( my_xtflux, block_model_trace,my_scratch.bead_flow_t );
}

static void IncorporationRiseFromNeighborSignal( float **incorporation_rise, float **neighbor_local,
    TimeCompression &time_c, RegionTracker &my_regions,
    BeadScratchSpace &my_scratch,
    buffer_params_block_flows & my_cur_buffer_block,
    int flow_block_size
  )
{
  // fill in each flow incorporation
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    // grab my excess hydrogen ions from the observed signal
    MathModel::IntegrateRedFromObservedTotalTrace ( incorporation_rise[fnum], neighbor_local[fnum], &my_scratch.shifted_bkg[fnum*time_c.npts() ], time_c.npts(),&time_c.deltaFrame[0], my_cur_buffer_block.tauB[fnum], my_cur_buffer_block.etbR[fnum] );
    // variables used for solving background signal shape
  }

  // construct the incorporation in the neighbor well just for research purposes
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    // bad! put trace back in neighbor signal
    MathModel::RedSolveHydrogenFlowInWell ( neighbor_local[fnum],incorporation_rise[fnum],time_c.npts(),my_regions.cache_step.i_start_coarse_step[fnum],&time_c.deltaFrame[0],my_cur_buffer_block.tauB[fnum] ); // we lose hydrogen ions fast!
  }
}



void MathModel::AccumulateSingleNeighborExcessHydrogen ( float *my_xtflux, float *neighbor_signal, BeadParams *p, reg_params *reg_p,
    BeadScratchSpace &my_scratch, 
    buffer_params_block_flows &my_cur_buffer_block,
    TimeCompression &time_c,
    RegionTracker &my_regions, const FlowBufferInfo & my_flow,
    bool use_vectorization,
    float tau_top, float tau_bulk, float multiplier,
    int flow_block_size, int flow_block_start )
{
  // over-ride buffering parameters for bead

  FillBufferParamsBlockFlows ( &my_cur_buffer_block,p,reg_p,my_flow.flow_ndx_map,flow_block_start,
    flow_block_size );

  float block_model_trace[my_scratch.bead_flow_t], block_incorporation_rise[my_scratch.bead_flow_t];

  // "In parallel, across flows"

  float* model_trace[flow_block_size];
  float* scratch_trace[flow_block_size];
  float* incorporation_rise[flow_block_size];
  float* lost_hydrogens[flow_block_size];
  float* neighbor_local[flow_block_size];
  // should this be using cur_buffer_block as usual?
  float vec_tau_top[flow_block_size];
  float vec_tau_bulk[flow_block_size];

  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    vec_tau_top[fnum] = tau_top;
    vec_tau_bulk[fnum] = tau_bulk;
  }

  // setup parallel pointers into the structure
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {

    scratch_trace[fnum]=model_trace[fnum] = &block_model_trace[fnum*time_c.npts() ];
    lost_hydrogens[fnum]=incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c.npts() ];   // set up each flow information
    neighbor_local[fnum] = &neighbor_signal[fnum*time_c.npts() ];
  }
  // make incorporation_rise
  IncorporationRiseFromNeighborSignal ( incorporation_rise,neighbor_local, time_c,my_regions, my_scratch, my_cur_buffer_block, flow_block_size );
  // use scratch_trace to hold temporary trace - same memory as model_trace because we don't need it yet
  // turn incorporation_rise into lost_hydrogens
  CumulativeLostHydrogens ( incorporation_rise, scratch_trace,time_c, my_regions, use_vectorization, vec_tau_top, flow_block_size );
  // lost_hydrogens = incorporation_rise 
  // now get model_trace for real in cross-talk and overwrite any temporary uses of that space
  IonsFromBulk ( model_trace,lost_hydrogens, time_c, my_regions, my_flow, use_vectorization, vec_tau_bulk, flow_block_size );
  

  // universal
  MultiplyVectorByScalar ( block_model_trace,multiplier,my_scratch.bead_flow_t ); // scale down the quantity of ions

  // add to the bulk cross-talk we're creating
  AccumulateVector ( my_xtflux, block_model_trace,my_scratch.bead_flow_t );
}

// Multiflow: Solve incorporation
// Solve lost hydrogens to bulk
// simplify cross-talk
// a) well >loses< ions - excess hydrogen lost based on well parameters
// b) "empty" well >gains< ions - based on tauE
// c) "gained" ions will then predict bulk

void MathModel::AccumulateSingleNeighborExcessHydrogenOneParameter ( float *my_xtflux, float *neighbor_signal,
    BeadParams *p, reg_params *reg_p,
    BeadScratchSpace &my_scratch, 
    buffer_params_block_flows &my_cur_buffer_block,
    TimeCompression &time_c,
    RegionTracker &my_regions, const FlowBufferInfo & my_flow,
    bool use_vectorization,
    float multiplier, bool rescale_flag,
    int flow_block_size, int flow_block_start )
{

  FillBufferParamsBlockFlows ( &my_cur_buffer_block,p,reg_p,my_flow.flow_ndx_map, flow_block_start,
    flow_block_size );

  float block_model_trace[my_scratch.bead_flow_t], block_incorporation_rise[my_scratch.bead_flow_t];

  // "In parallel, across flows"

  float* model_trace[flow_block_size];
  float* scratch_trace[flow_block_size];
  float* incorporation_rise[flow_block_size];
  float* lost_hydrogens[flow_block_size];
  float* neighbor_local[flow_block_size];

  // should this be using cur_buffer_block as usual?
  float vec_tau_well[flow_block_size];
  float vec_tau_empty[flow_block_size];

  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    vec_tau_well[fnum] = my_cur_buffer_block.tauB[fnum];
    vec_tau_empty[fnum] = my_cur_buffer_block.etbR[fnum]*vec_tau_well[fnum];
  }

  // setup parallel pointers into the structure
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {

    scratch_trace[fnum] = model_trace[fnum] = &block_model_trace[fnum*time_c.npts() ];
    lost_hydrogens[fnum] = incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c.npts() ];   // set up each flow information
    neighbor_local[fnum] = &neighbor_signal[fnum*time_c.npts() ];
  }

  IncorporationRiseFromNeighborSignal ( incorporation_rise,neighbor_local, time_c,my_regions, my_scratch, my_cur_buffer_block, flow_block_size );
  // uses a scratch buffer here [recycles model_trace as we don't need it yet], 
  // turns incorporation_rise into lost_hydrogens
  CumulativeLostHydrogens ( incorporation_rise, scratch_trace, time_c, my_regions, use_vectorization, vec_tau_well, flow_block_size );
  // lost_hydrogens=incorporation_rise returned as lost_hydrogens above
  // now we generate the real model_trace we're accumulating
  IonsFromBulk ( model_trace,lost_hydrogens, time_c, my_regions, my_flow, use_vectorization, vec_tau_empty, flow_block_size );
  
  if (rescale_flag)
    RescaleTraceByBuffering(model_trace, vec_tau_well, vec_tau_empty,time_c.npts(), flow_block_size);

  // universal
  MultiplyVectorByScalar ( block_model_trace,multiplier,my_scratch.bead_flow_t ); // scale down the quantity of ions

  // add to the bulk cross-talk we're creating
  AccumulateVector ( my_xtflux, block_model_trace,my_scratch.bead_flow_t );
}

