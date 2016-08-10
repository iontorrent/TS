/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgSearchAmplitude.h"


SearchAmplitude::SearchAmplitude()
{
  math_poiss = NULL;
  my_trace = NULL;
  empty_trace = NULL;
  my_scratch = NULL;
  pointer_regions = NULL;
  time_c = NULL;
  my_flow = NULL;
  emphasis_data = NULL;

  negative_amplitude_limit = 0.001f;
  positive_amplitude_limit = LAST_POISSON_TABLE_COL-0.5 ;  // avoid zero derivatives if we hit the top of the table
  use_vectorization = true;
  num_iterations = 2;
}

SearchAmplitude::~SearchAmplitude()
{
  // nothing allocated here, just get rid of everything
  math_poiss = NULL;
  my_trace = NULL;
  empty_trace = NULL;
  my_scratch = NULL;
  pointer_regions = NULL;
  time_c = NULL;
  my_flow = NULL;
  emphasis_data = NULL;
}

/////// Another means of getting a crude amplitude: projection onto model vector

// project X onto Y

static float project_vector ( const float *X, const float *Y, const float *em, int len )
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
}

void SearchAmplitude::ProjectionSearchAmplitude ( 
    BeadTracker &my_beads, bool , bool sampledOnly, int flow_block_size, int flow_block_start
  ) const
{
  // should be using a specific region pointer, not the universal one, to match the bead list
  // as we may generate more region pointers from more beads

  my_scratch->FillShiftedBkg ( *empty_trace, pointer_regions->rp.tshift, *time_c, true, flow_block_size );
  my_scratch->ResetXtalkToZero();
  // "emphasis" originally set to zero here, boosted up to the calling function to be explicit
  int TmpEmphasis[flow_block_size];
  for( int fnum = 0 ; fnum < flow_block_size ; ++fnum )
    TmpEmphasis[fnum] = 0; // all zero
  my_scratch->FillEmphasis ( TmpEmphasis,emphasis_data->EmphasisVectorByHomopolymer,emphasis_data->EmphasisScale, flow_block_size );
  pointer_regions->cache_step.ForceLockCalculateNucRiseCoarseStep ( &pointer_regions->rp, *time_c, *my_flow ); // technically the same over the whole region

  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    // want this for all beads, even if they are polyclonal.
    if (!sampledOnly || (sampledOnly && my_beads.Sampled(ibd))) { 
      my_scratch->FillObserved ( *my_trace,ibd, flow_block_size ); // set scratch space for this bead

      ProjectionSearchOneBead ( &my_beads.params_nn[ibd], flow_block_size, flow_block_start );
       my_beads.params_nn[ibd].ApplyAmplitudeZeros ( my_flow->dbl_tap_map, flow_block_size ); // force double-taps to zero

    }
  }
  // undo any problems we may have caused for the rest of the code
  pointer_regions->cache_step.Unlock();
}

static void ProjectAmplitude ( 
    BeadParams *p, 
    const float *observed, 
    const float *const *model_trace, 
    const float *em_vector, 
    int npts, 
    float negative_amplitude_limit ,
    float positive_amplitude_limit,
    int flow_block_size
  )
{
  static int numWarnings = 0;
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    float tmp_mult= project_vector ( & ( observed[npts*fnum] ),model_trace[fnum],em_vector,npts );
    p->Ampl[fnum] *=tmp_mult;
    // Some limited warnings
    if ( p->Ampl[fnum]!=p->Ampl[fnum] && numWarnings++ < 10 )
    {
      printf ( "BSA: tmp_mult:\t%f\n",tmp_mult );
      printf ( "BSA: em\t" );
      for ( int ix=0; ix<npts; ix++ )
        printf ( "%f\t",em_vector[ix] );
      printf ( "\n" );
      printf ( "BSA: obser\t" );
      for ( int ix=0; ix<npts; ix++ )
        printf ( "%f\t",observed[npts*fnum+ix] );
      printf ( "\n" );
      printf ( "BSA: model\t" );
      for ( int ix=0; ix<npts; ix++ )
        printf ( "%f\t",model_trace[fnum][ix] );
      printf ( "\n" );

    }
    // Not sure why 1.0f but this is how I found it...
    if ( p->Ampl[fnum]!=p->Ampl[fnum] )
    {
      p->Ampl[fnum]=1.0f;
    }
    // bounds check
    if ( p->Ampl[fnum]<negative_amplitude_limit )
      p->Ampl[fnum] = negative_amplitude_limit; // Projection onto zero returns potentially large number, but we may want negative values here
    if ( p->Ampl[fnum]>positive_amplitude_limit )
      p->Ampl[fnum]=positive_amplitude_limit; // we limit this everywhere else, so keep from explosion
  }
}

// exploit vectors to get a quick & dirty estimate of amplitude
// project onto model trace to get a multiplicative step bringing us closer to alignment with the trace.
void SearchAmplitude::ProjectionSearchOneBead ( BeadParams *p, int flow_block_size, int flow_block_start ) const
{
  reg_params *reg_p = &pointer_regions->rp;

  // technically, need to setup bead correctly here
  for (int fnum=0; fnum<flow_block_size; fnum++){
    p->Ampl[fnum] = 1.0f; // set for projection initiation
    //p->kmult[fnum] = 1.0f; //? should be set because it might be set differently in some routine before this
  }

  // set up bead parameters across flows for this bead
  // they >might< depend on initial kmult, amplitude
  MathModel::FillBufferParamsBlockFlows ( my_cur_buffer_block,p,reg_p,my_flow->flow_ndx_map, flow_block_start, flow_block_size );
  MathModel::FillIncorporationParamsBlockFlows ( my_cur_bead_block, p,reg_p,my_flow->flow_ndx_map, flow_block_start, flow_block_size );

  // this is a more generic "subtract background" routine that we need to think about
  // reusing for 'oneflowfit' amplitude fitting
  // need to add cross-talk (0) here so that everything is compatible.
  // hold all flows of signal at once

  MathModel::MultiCorrectBeadBkg ( my_scratch->observed,p, *my_scratch,*my_cur_buffer_block,
                                   *my_flow,*time_c, *pointer_regions,my_scratch->shifted_bkg,
                                   use_vectorization, flow_block_size );

  const float *em_vector = emphasis_data->EmphasisVectorByHomopolymer[0]; // incorrect @TODO

// @TODO make parallel
  float block_model_trace[my_scratch->bead_flow_t], block_incorporation_rise[my_scratch->bead_flow_t];
  // "In parallel, across flows"
  const float* nuc_rise_ptr[flow_block_size];
  float* model_trace[flow_block_size];
  float* incorporation_rise[flow_block_size];
  int my_start[flow_block_size];

  // setup parallel pointers into the structure
  for ( int fnum=0; fnum<flow_block_size; fnum++ )
  {
    nuc_rise_ptr[fnum] = pointer_regions->cache_step.NucCoarseStep ( fnum );
    model_trace[fnum] = &block_model_trace[fnum*time_c->npts() ];
    //tmp_model_trace[fnum] = &tmp_block_model_trace[fnum*time_c->npts() ];
    incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c->npts() ];   // set up each flow information

    my_start[fnum] = pointer_regions->cache_step.i_start_coarse_step[fnum];
  }

  // now loop calculating model and projecting, multiflow
  // at this point, we have amplitude = 1.0, krate= default in each flow
  for ( int projection_loop=0; projection_loop<num_iterations; projection_loop++ )
  {

    // calculate incorporation given parameters
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
    {
      // can use the parallel trick
      // compute model based inccorporation trace
      // also done again and again
      //p->my_state.hits_by_flow[fnum]++; // temporary

      MathModel::ComputeCumulativeIncorporationHydrogens ( incorporation_rise[fnum],
          time_c->npts(), &time_c->deltaFrameSeconds[0], nuc_rise_ptr[fnum], ISIG_SUB_STEPS_MULTI_FLOW, my_start[fnum],
          my_cur_bead_block->C[fnum], p->Ampl[fnum], my_cur_bead_block->SP[fnum],
          my_cur_bead_block->kr[fnum], my_cur_bead_block->kmax[fnum], my_cur_bead_block->d[fnum], my_cur_bead_block->molecules_to_micromolar_conversion[fnum], math_poiss, reg_p->hydrogenModelType );

      MultiplyVectorByScalar ( incorporation_rise[fnum], my_cur_bead_block->sens[fnum],time_c->npts() ); // transform hydrogens to signal       // variables used for solving background signal shape
    }

    // can be parallel
    if ( use_vectorization )
    {
      MathModel::RedSolveHydrogenFlowInWell_Vec ( model_trace,incorporation_rise,time_c->npts(),&time_c->deltaFrame[0],my_cur_buffer_block->tauB, flow_block_size ); // we lose hydrogen ions fast!

    }
    else
    {
      for ( int fnum=0; fnum<flow_block_size; fnum++ )
      {
        // we do this bit again and again
        MathModel::RedSolveHydrogenFlowInWell ( model_trace[fnum],incorporation_rise[fnum],
                                     time_c->npts(),my_start[fnum],&time_c->deltaFrame[0],my_cur_buffer_block->tauB[fnum] );
      }
    }
      MultiplyVectorByScalar ( block_model_trace,p->gain,my_scratch->bead_flow_t );

    // magic trick: project onto model incorporation trace to find amplitude
    // because we're "almost" linear this trick works
    ProjectAmplitude ( p, my_scratch->observed, model_trace, em_vector,time_c->npts(), negative_amplitude_limit, positive_amplitude_limit, flow_block_size );

  }
}

