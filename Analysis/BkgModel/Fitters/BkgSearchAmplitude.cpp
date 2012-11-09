/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgSearchAmplitude.h"


void SearchAmplitude::EvaluateAmplitudeFit ( bead_params *p, float *avals,float *error_by_flow )
{
  //bead_params eval_params = *p; // temporary copy - not needed as we over-write this object
  reg_params *reg_p = & ( pointer_regions->rp );

  params_SetAmplitude ( p,avals );
  // evaluate the function
  MultiFlowComputeCumulativeIncorporationSignal ( p,reg_p,my_scratch->ival,pointer_regions->cache_step,my_scratch->cur_bead_block,*time_c,*my_flow,math_poiss );
  MultiFlowComputeTraceGivenIncorporationAndBackground ( my_scratch->fval,p,reg_p,my_scratch->ival,my_scratch->shifted_bkg,*pointer_regions,my_scratch->cur_buffer_block,*time_c,*my_flow,use_vectorization, my_scratch->bead_flow_t );


  my_scratch->CalculateFitError ( error_by_flow,my_flow->numfb );
}


void InitializeBinarySearch ( bead_params *p, bool restart, float *ac, float *ub, float *lb, float *step, bool *done )
{
  if ( restart )
    for ( int fnum=0;fnum<NUMFB;fnum++ )
    {
      ac[fnum] = 0.5;
      ub[fnum] = 1.5*MAX_HPLEN; // top value that has been rejected
      lb[fnum] = -0.5; // lowest value that has been rejected
      done[fnum] = false;
      step[fnum] = 1.0;
    }
  else
    for ( int fnum=0;fnum<NUMFB;fnum++ )
    {
      ac[fnum] =p->Ampl[fnum];
      done[fnum] = false;
      step[fnum] = 0.02;
    }
}

void TakeStepBinarySearch ( float *ap, float *ep, float *ac, float *ec,float *ub, float *lb, float *step,  float deltaAmp, float min_a, float max_a )
{
  for ( int i=0;i < NUMFB;i++ )
  {
    float slope = ( ep[i] - ec[i] ) /deltaAmp;

    if ( slope < 0.0 )
    {
      // step up to lower err
      lb[i] = ac[i]; // leaving here
      ap[i] = ac[i] + step[i];
      if ( ( ap[i]+deltaAmp ) >ub[i] )
      {
        ap[i] = ( ac[i]+ub[i] ) /2;
        step[i] = ap[i] - ac[i];
      }
    }
    else
    {
      // step up to lower err
      ub[i] = ac[i]; // leaving here, must be upper bound
      ap[i] = ac[i] - step[i];
      if ( ap[i]< ( lb[i]+deltaAmp ) )
      {
        ap[i] = ( ac[i]+lb[i] ) /2;
        step[i] = ac[i]-ap[i];
      }
    }

    if ( ap[i] < min_a )
    {
      ap[i] = min_a;
      step[i] = ac[i]-min_a; // the step we're really taking
    }
    if ( ap[i] > max_a )
    {
      ap[i] = max_a;
      step[i] = max_a-ac[i]; // the step we're really taking
    }
  }
}


void UpdateStepBinarySearch ( float *ac, float *ec, float *ap, float *ep, float *ub, float *lb, float *step, float min_step, bool *done )
{
  for ( int i=0;i < NUMFB;i++ )
  {
    if ( ep[i] < ec[i] )
    {
      if ( ac[i]>ap[i] )
        ub[i] = ac[i];
      if ( ac[i]<ap[i] )
        lb[i] = ac[i]; // last point tried bounds direction of search

      ac[i] = ap[i];
      ec[i] = ep[i];
      if ( step[i]<min_step )
        done[i] = true; // succeeded using a small step
      step[i] *= 1.4;
      // because expect exponential decay, each +1 = 1/4 the odds, so optimal step is in fact small
    }
    else
    {
      if ( ap[i]>ac[i] )
        ub[i]=ap[i]; // loser upper bound
      if ( ap[i]<ac[i] )
        lb[i] = ap[i]; // loser lower bound
      step[i] /= 2.0;  // lose shrink
      if ( step[i] < min_step )
      {
        done[i] = true;
      }
    }
  }
}

int CheckDone ( bool *done )
{
  int done_cnt = NUMFB;
  for ( int i=0; i<NUMFB; i++ )
    if ( done[i] )
      done_cnt--;
  return ( done_cnt );
}


void SearchAmplitude::BinarySearchOneBead ( bead_params *p, float min_step, bool restart )
{
  float ac[NUMFB] __attribute__ ( ( aligned ( 16 ) ) );
  float ec[NUMFB] __attribute__ ( ( aligned ( 16 ) ) );
  float ap[NUMFB] __attribute__ ( ( aligned ( 16 ) ) );
  float ep[NUMFB] __attribute__ ( ( aligned ( 16 ) ) );
  float ub[NUMFB];
  float lb[NUMFB];
  float step[NUMFB] __attribute__ ( ( aligned ( 16 ) ) );
  float min_a = 0.001;
  float max_a = ( MAX_HPLEN - 1 )-0.001;
  bool done[NUMFB];
  int done_cnt;
  int iter;
  float deltaAmp = 0.00025;


  InitializeBinarySearch ( p,restart,ac,ub,lb,step,done );

  EvaluateAmplitudeFit ( p, ac,ec );

  done_cnt = NUMFB; // number remaining to finish

  iter = 0;

  while ( ( done_cnt > 0 ) && ( iter <= 30 ) )
  {
    // figure out which direction to go in from here
    for ( int fnum=0;fnum<NUMFB;fnum++ )
      ap[fnum] = ac[fnum] + deltaAmp;

    EvaluateAmplitudeFit ( p, ap,ep );
    // computes derivative to determine direction
    TakeStepBinarySearch ( ap,ep,ac,ec,ub,lb,step,deltaAmp,min_a,max_a );

    // determine if new location is better
    EvaluateAmplitudeFit ( p, ap,ep );

    UpdateStepBinarySearch ( ac,ec,ap,ep, ub,lb,step, min_step, done );
    done_cnt = CheckDone ( done );
    iter++;
  }
  params_SetAmplitude ( p,ac );

  // It is important to have amplitude zero so that regional fits do the right thing for other parameters
  params_ApplyAmplitudeZeros ( p,my_flow->dbl_tap_map );
}

void SearchAmplitude::BinarySearchAmplitude ( BeadTracker &my_beads, float min_step,bool restart )
{
  // should be using a specific region pointer, not the universal one, to match the bead list
  // as we may generate more region pointers from more beads
  // force refresh of cached background
  my_scratch->FillShiftedBkg ( *empty_trace, pointer_regions->rp.tshift, *time_c, true );
  my_scratch->ResetXtalkToZero();
  // "emphasis" originally set to zero here, boosted up to the calling function to be explicit
  int TmpEmphasis[NUMFB] = {0}; // all zero
  my_scratch->FillEmphasis ( TmpEmphasis,emphasis_data->EmphasisVectorByHomopolymer,emphasis_data->EmphasisScale );
  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    // want this for all beads, even if they are polyclonal.
    //@TODO this is of course really weird as we have several options for fitting one flow at a time
    // which might be easily parallelizable and not need this weird global level binary search
    my_scratch->FillObserved ( *my_trace,ibd ); // set scratch space for this bead

    BinarySearchOneBead ( &my_beads.params_nn[ibd],min_step,restart );
  }
}

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
  use_vectorization = true;
  rate_fit = false;
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
// possibly I need to include "emphasis" function here
float project_vector ( float *X, float *Y, float *em, int len )
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

float em_diff ( float *X, float *Y, float *em, int len )
{
  float tot=0.0f;
  float eval;
  for ( int i=0; i<len; i++ )
  {
    eval = ( X[i]-Y[i] ) *em[i];
    tot += eval*eval;
  }
  return ( tot );
}


void SearchAmplitude::ProjectionSearchAmplitude ( BeadTracker &my_beads, bool _rate_fit )
{
  rate_fit = _rate_fit;
  // should be using a specific region pointer, not the universal one, to match the bead list
  // as we may generate more region pointers from more beads

  my_scratch->FillShiftedBkg ( *empty_trace, pointer_regions->rp.tshift, *time_c, true );
  my_scratch->ResetXtalkToZero();
  // "emphasis" originally set to zero here, boosted up to the calling function to be explicit
  int TmpEmphasis[NUMFB] = {0}; // all zero
  my_scratch->FillEmphasis ( TmpEmphasis,emphasis_data->EmphasisVectorByHomopolymer,emphasis_data->EmphasisScale );
  pointer_regions->cache_step.ForceLockCalculateNucRiseCoarseStep ( &pointer_regions->rp, *time_c, *my_flow ); // technically the same over the whole region

  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    // want this for all beads, even if they are polyclonal.
    //@TODO this is of course really weird as we have several options for fitting one flow at a time
    // which might be easily parallelizable and not need this weird global level binary search
    my_scratch->FillObserved ( *my_trace,ibd ); // set scratch space for this bead

    ProjectionSearchOneBead ( &my_beads.params_nn[ibd] );
  }
  // undo any problems we may have caused for the rest of the code
  pointer_regions->cache_step.Unlock();
}

void ProjectAmplitude ( bead_params *p, float *observed, float **model_trace, float *em_vector, int npts, float negative_amplitude_limit )
{
  static int numWarnings = 0;
  for ( int fnum=0; fnum<NUMFB; fnum++ )
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
    if ( p->Ampl[fnum]>MAX_HPLEN )
      p->Ampl[fnum]=MAX_HPLEN; // we limit this everywhere else, so keep from explosion
  }
}

// exploit vectors to get a quick & dirty estimate of amplitude
// project onto model trace to get a multiplicative step bringing us closer to alignment with the trace.
void SearchAmplitude::ProjectionSearchOneBead ( bead_params *p )
{
  reg_params *reg_p = &pointer_regions->rp;


  // set up bead parameters across flows for this bead
  FillBufferParamsBlockFlows ( & ( my_scratch->cur_buffer_block ),p,reg_p,my_flow->flow_ndx_map,my_flow->buff_flow );
  FillIncorporationParamsBlockFlows ( & ( my_scratch->cur_bead_block ), p,reg_p,my_flow->flow_ndx_map,my_flow->buff_flow );

  // this is a more generic "subtract background" routine that we need to think about
  // reusing for 'oneflowfit' amplitude fitting
  // need to add cross-talk (0) here so that everything is compatible.
  // hold all flows of signal at once

  MultiCorrectBeadBkg ( my_scratch->observed,p,
                        *my_scratch,*my_flow,*time_c,*pointer_regions,my_scratch->shifted_bkg,use_vectorization );

  float *em_vector = emphasis_data->EmphasisVectorByHomopolymer[0]; // incorrect @TODO

// @TODO make parallel
  float block_model_trace[my_scratch->bead_flow_t], block_incorporation_rise[my_scratch->bead_flow_t];
  //float tmp_block_model_trace[my_scratch->bead_flow_t];
  //float* tmp_model_trace[NUMFB];
  // "In parallel, across flows"
  float* nuc_rise_ptr[NUMFB];
  float* model_trace[NUMFB];
  float* incorporation_rise[NUMFB];
  int my_start[NUMFB];

  // setup parallel pointers into the structure
  for ( int fnum=0; fnum<NUMFB; fnum++ )
  {
    nuc_rise_ptr[fnum] = pointer_regions->cache_step.NucCoarseStep ( fnum );
    model_trace[fnum] = &block_model_trace[fnum*time_c->npts() ];
    //tmp_model_trace[fnum] = &tmp_block_model_trace[fnum*time_c->npts() ];
    incorporation_rise[fnum] = &block_incorporation_rise[fnum*time_c->npts() ];   // set up each flow information

    my_start[fnum] = pointer_regions->cache_step.i_start_coarse_step[fnum];
    p->Ampl[fnum] = 1.0f; // set for projection
  }

  int numiterations = 2;
  // now loop calculating model and projecting, multiflow
  for ( int projection_loop=0; projection_loop<numiterations; projection_loop++ )
  {

    // calculate incorporation given parameters
    for ( int fnum=0; fnum<my_flow->numfb; fnum++ )
    {
      // can use the parallel trick
      // compute model based inccorporation trace
      // also done again and again
      //p->my_state.hits_by_flow[fnum]++; // temporary

      ComputeCumulativeIncorporationHydrogens ( incorporation_rise[fnum],
          time_c->npts(), &time_c->deltaFrameSeconds[0], nuc_rise_ptr[fnum], ISIG_SUB_STEPS_MULTI_FLOW, my_start[fnum],
          my_scratch->cur_bead_block.C[fnum], p->Ampl[fnum], my_scratch->cur_bead_block.SP[fnum],
          my_scratch->cur_bead_block.kr[fnum], my_scratch->cur_bead_block.kmax[fnum], my_scratch->cur_bead_block.d[fnum], my_scratch->cur_bead_block.molecules_to_micromolar_conversion[fnum], math_poiss );

      MultiplyVectorByScalar ( incorporation_rise[fnum], my_scratch->cur_bead_block.sens[fnum],time_c->npts() ); // transform hydrogens to signal       // variables used for solving background signal shape
    }

    // can be parallel
#ifndef __INTEL_COMPILER
    if ( use_vectorization )
    {
      RedSolveHydrogenFlowInWell_Vec ( NUMFB,model_trace,incorporation_rise,time_c->npts(),&time_c->deltaFrame[0],my_scratch->cur_buffer_block.tauB ); // we lose hydrogen ions fast!

    }
    else
#else // use Intel Compiler
    {
      for ( int fnum=0; fnum<my_flow->numfb; fnum++ )
      {
        // we do this bit again and again
        RedSolveHydrogenFlowInWell ( model_trace[fnum],incorporation_rise[fnum],
                                     time_c->npts(),my_start[fnum],&time_c->deltaFrame[0],my_scratch->cur_buffer_block.tauB[fnum] );
      }
    }
#endif
      MultiplyVectorByScalar ( block_model_trace,p->gain,my_scratch->bead_flow_t );

    // magic trick: project onto model incorporation trace to find amplitude
    // because we're "almost" linear this trick works
    ProjectAmplitude ( p, my_scratch->observed, model_trace, em_vector,time_c->npts(), negative_amplitude_limit );

  }
}

