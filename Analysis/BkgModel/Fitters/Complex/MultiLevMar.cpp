/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MultiLevMar.h"
#include "MiscVec.h"

MultiFlowLevMar::MultiFlowLevMar ( SignalProcessingMasterFitter &_bkg ) :
    bkg ( _bkg )
{
// create matrix packing object(s)
  //Note:  scratch-space is used directly by the matrix packer objects to get the derivatives
  // so this object >must< persist in order to be used by the fit control object in the Lev_Mar fit.
  // Allocate directly the annoying pointers that the lev-mar object uses for control

  lm_state.SetNonIntegerPenalty ( bkg.global_defaults.fitter_defaults.clonal_call_scale,bkg.global_defaults.fitter_defaults.clonal_call_penalty,MAGIC_MAX_CLONAL_HP_LEVEL );
  lm_state.shrink_factor = bkg.global_defaults.fitter_defaults.shrink_factor;
  lm_state.AllocateBeadFitState ( bkg.region_data->my_beads.numLBeads );
  lm_state.AssignBeadsToRegionGroups ();

  lev_mar_scratch.Allocate ( bkg.region_data->time_c.npts(),fit_control.fitParams.NumSteps );
  fit_control.AllocPackers ( lev_mar_scratch.scratchSpace, bkg.global_defaults.signal_process_control.no_RatioDrift_fit_first_20_flows,bkg.global_defaults.signal_process_control.fitting_taue, lev_mar_scratch.bead_flow_t, bkg.region_data->time_c.npts() );
  use_vectorization = bkg.global_defaults.signal_process_control.use_vectorization;

  // regional parameters are the same for each bead,
  // we recalculate these for each derivative * all beads
  // so instead calculate once and reuse for each bead the little steps we will take
  step_rp.resize ( fit_control.fitParams.NumSteps ); // waste a little space as we don't use all steps in all fits, but good enough
  step_nuc_cache.resize ( fit_control.fitParams.NumSteps ); // waste a little space as we don't use all steps, but good enough
  for ( unsigned int istep=0; istep<step_nuc_cache.size(); istep++ )
    step_nuc_cache[istep].Alloc ( bkg.region_data->time_c.npts() );

  InitTshiftCache();
  InitRandomCache();

  // assumed the same size
  assert ( lm_state.numLBeads == bkg.region_data->my_beads.numLBeads );
}

MultiFlowLevMar::~MultiFlowLevMar()
{
  DeleteRandomCache();
}

///-------------Entry points to this optimizer

// this does work on a sampling of wells >only< to find regional parameters
int MultiFlowLevMar::MultiFlowSpecializedSampledLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction )
{

  EnterTheOptimization ( well_fit,reg_fit,lambda_start,clonal_restriction );

  // if just regional steps
  bool do_just_region = ( well_fit==NULL );
  bool do_just_well = ( reg_fit==NULL );
  bool do_both = !do_just_region & !do_just_well;

  int total_iter = 0; // total number of steps taken

  // just regional parameter updates without any well updates
  if ( do_just_region )
  {
    for ( int loc_iter=0 ; ( loc_iter< number_region_iterations_wanted )  & do_just_region; loc_iter++ )
    {
      DoSampledRegionIteration ( reg_fit,total_iter );
      total_iter++;
    }
  }

  // if alternating bead and regional steps
  if ( do_both )
  {
    for ( int loc_iter=0 ; ( loc_iter<number_region_iterations_wanted )  ; loc_iter++ )
    {
      // do one well iteration
      bool skip_region = DoSampledBeadIteration ( false, well_fit, total_iter );
      total_iter++;
      if ( skip_region )
      {
        total_iter = 2*number_region_iterations_wanted;
        break;
      }
      // do one region iteration
      if ( !skip_region )
      {
        DoSampledRegionIteration ( reg_fit,total_iter );
        total_iter++;
      }
    }
  }
  // just bead steps to finish off
  for ( int loc_iter=0 ; loc_iter<additional_bead_only_iterations; loc_iter++ )
  {
    DoSampledBeadIteration ( true, well_fit, total_iter );
    total_iter++;
  }
  CleanTerminateOptimization();
  return ( total_iter );
}



// fitting all beads and some beads per region
int MultiFlowLevMar::MultiFlowSpecializedLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction )
{

  EnterTheOptimization ( well_fit,reg_fit,lambda_start,clonal_restriction );


  // if just regional steps
  bool do_just_region = ( well_fit==NULL );
  bool do_just_well = ( reg_fit==NULL );
  bool do_both = !do_just_region & !do_just_well;

  int total_iter=0;

  // just regional parameter updates without any well updates
  if ( do_just_region )
  {
    for ( int loc_iter=0 ; loc_iter<number_region_iterations_wanted; loc_iter++ )
    {
      DoRegionIteration ( reg_fit,total_iter );
      total_iter++;
    }
  }

  // if alternating bead and regional steps
  if ( do_both )
  {
    for ( int loc_iter=0 ; loc_iter<number_region_iterations_wanted ; loc_iter++ )
    {
      // do one well iteration
      bool skip_region = DoAllBeadIteration ( false, well_fit, total_iter );
      total_iter++;
      if ( skip_region )
      {
        total_iter = 2*number_region_iterations_wanted;
        break;
      }
      // do one region iteration
      if ( !skip_region )
      {
        DoRegionIteration ( reg_fit,total_iter );
        total_iter++;
      }
    }
  }

  // just bead steps to finish off
  for ( int loc_iter=0 ; loc_iter<additional_bead_only_iterations; loc_iter++ )
  {
    DoAllBeadIteration ( true, well_fit, total_iter );
    total_iter++;
  }


  CleanTerminateOptimization();
  return ( total_iter );
}

// If I'm not fitting wells, make it easy to see I'm never going to fit wells
int MultiFlowLevMar::MultiFlowSpecializedLevMarFitParametersOnlyRegion ( int number_region_iterations_wanted,  BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction )
{

  EnterTheOptimization ( NULL,reg_fit,lambda_start,clonal_restriction );

  int total_iter=0;
  // just regional parameter updates without any well updates
  if ( reg_fit!=NULL )
  {
    for ( int loc_iter=0 ; loc_iter<number_region_iterations_wanted; loc_iter++ )
    {
      DoRegionIteration ( reg_fit,total_iter );
      total_iter++;
    }
  }

  CleanTerminateOptimization();
  return ( total_iter );
}

// This is to finalize the well parameters conditional on the regional parameters
// strong candidate for export to the GPU
void MultiFlowLevMar::MultiFlowSpecializedLevMarFitAllWells ( int bead_only_iterations, BkgFitMatrixPacker *well_fit, float lambda_start,int clonal_restriction )
{
  EnterTheOptimization ( well_fit,NULL, lambda_start,clonal_restriction );

  if ( well_fit != NULL )
  {
    DoAllBeadIteration ( true, well_fit, /*iter*/ 1, bead_only_iterations );
  }

  CleanTerminateOptimization();
}

///-----------------------------------done with entry points
void MultiFlowLevMar::DoSampledRegionIteration (
  BkgFitMatrixPacker *reg_fit,
  int iter )
{

  reg_params eval_rp;
  SetupAnyIteration ( eval_rp,iter );
  // do my region iteration
  int reg_wells = LevMarAccumulateRegionDerivsForSampledActiveBeadList ( eval_rp,
                  reg_fit, lm_state.reg_mask,
                  iter );
  // solve per-region equation and adjust parameters
  if ( reg_wells > lm_state.min_bead_to_fit_region )
  {
    LevMarFitRegion ( reg_fit );
  }
  IdentifyParametersFromSample ( bkg.region_data->my_beads,bkg.region_data->my_regions, lm_state.well_mask, lm_state.reg_mask,lm_state.skip_beads,lm_state );
}

void MultiFlowLevMar::DoRegionIteration (
  BkgFitMatrixPacker *reg_fit,
  int iter )
{

  reg_params eval_rp;
  SetupAnyIteration ( eval_rp, iter );
  // do my region iteration
  int reg_wells = LevMarAccumulateRegionDerivsForActiveBeadList ( eval_rp,
                  reg_fit, lm_state.reg_mask,
                  iter );
  // solve per-region equation and adjust parameters
  if ( reg_wells > lm_state.min_bead_to_fit_region )
  {
    LevMarFitRegion ( reg_fit );
    if ( !bkg.global_defaults.signal_process_control.regional_sampling )
      lm_state.IncrementRegionGroup();
  }

  IdentifyParameters ( bkg.region_data->my_beads,bkg.region_data->my_regions, lm_state.well_mask, lm_state.reg_mask,lm_state.skip_beads );
}


bool MultiFlowLevMar::DoSampledBeadIteration (
  bool well_only_fit,
  BkgFitMatrixPacker *well_fit,
  int iter )
{
  reg_params eval_rp;
  SetupAnyIteration ( eval_rp, iter );
  float failed_frac = LevMarFitToRegionalActiveBeadList (
                        well_only_fit,
                        eval_rp,
                        well_fit, lm_state.well_mask,
                        iter );
  // if more than 1/2 the beads aren't improving any longer, stop trying to do the
  // region-wide fit
  bool skip_region = false;
  if ( !well_only_fit && ( failed_frac > lm_state.bead_failure_rate_to_abort ) )
    skip_region=true; // which will be incremented later
  return ( skip_region );
}
bool MultiFlowLevMar::DoAllBeadIteration (
  bool well_only_fit,
  BkgFitMatrixPacker *well_fit, int iter,
  int bead_iterations, bool isSample )
{

  reg_params eval_rp;
  SetupAnyIteration ( eval_rp,  iter );
  float failed_frac = LevMarFitToActiveBeadList (
                        well_only_fit,
                        eval_rp,
                        well_fit, lm_state.well_mask,
                        bead_iterations );
  // if more than 1/2 the beads aren't improving any longer, stop trying to do the
  // region-wide fit
  bool skip_region = false;
  if ( !well_only_fit && ( failed_frac > lm_state.bead_failure_rate_to_abort ) )
    skip_region=true;
  return ( skip_region );
}

void MultiFlowLevMar::SetupAnyIteration ( reg_params &eval_rp,  int iter )
{
#ifdef FIT_ITERATION_DEBUG_TRACE
  bkg.DebugIterations();
#endif

  lm_state.PhaseInClonalRestriction ( iter,lm_state.nonclonal_call_penalty_enforcement );
  FillTshiftCache ( bkg.region_data->my_regions.rp.tshift );
  lm_state.reg_error = 0.0f;
  eval_rp = bkg.region_data->my_regions.rp;
}

void MultiFlowLevMar::EnterTheOptimization ( BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit, float lambda_start, int clonal_restriction )
{
  lm_state.InitializeLevMarFit ( well_fit,reg_fit );
  lm_state.nonclonal_call_penalty_enforcement = clonal_restriction;
  lev_mar_scratch.ResetXtalkToZero();
  InitTshiftCache();
  bkg.region_data->my_beads.CorruptedBeadsAreLowQuality(); // make sure we're up to date with quality estimates

  lm_state.SetupActiveBeadList ( lambda_start );
}

void MultiFlowLevMar::CleanTerminateOptimization()
{
  // @ TODO: can this be cleaned up?
  lm_state.FinalComputeAndSetAverageResidual ( bkg.region_data->my_beads );
  lm_state.restrict_clonal = 0.0f; // we only restrict clonal within this routine
}



void MultiFlowLevMar::DynamicEmphasis ( bead_params &p )
{
  // put together the emphasis needed
  lev_mar_scratch.SetEmphasis ( p.Ampl,bkg.region_data->my_beads.max_emphasis );
  lev_mar_scratch.CreateEmphasis ( bkg.region_data->emphasis_data.EmphasisVectorByHomopolymer, bkg.region_data->emphasis_data.EmphasisScale );
}


void MultiFlowLevMar::FillScratchForEval ( struct bead_params *p,struct reg_params *reg_p, NucStep &cache_step )
{
//  params_IncrementHits(p);
  // evaluate the function
  MultiFlowComputeCumulativeIncorporationSignal ( p,reg_p,lev_mar_scratch.ival,cache_step,lev_mar_scratch.cur_bead_block,bkg.region_data->time_c,bkg.region_data->my_flow,bkg.math_poiss );
  MultiFlowComputeTraceGivenIncorporationAndBackground ( lev_mar_scratch.fval,p,reg_p,lev_mar_scratch.ival,cache_sbg,bkg.region_data->my_regions,lev_mar_scratch.cur_buffer_block,bkg.region_data->time_c,bkg.region_data->my_flow,use_vectorization, lev_mar_scratch.bead_flow_t );

  // add clonal restriction here to penalize non-integer clonal reads
  // this of course does not belong here and should be in the optimizer section of the code
  lm_state.ApplyClonalRestriction ( lev_mar_scratch.fval, p,bkg.region_data->time_c.npts() );

  // put together the emphasis needed
  DynamicEmphasis ( *p );
}

// assumes use of the cached regional derivative steps & nuc_step precomputation
void MultiFlowLevMar::AccumulateRegionDerivForOneBead (
  int ibd, int &reg_wells,
  BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
  int iter )
{
  // get the current parameter values for this bead
  bead_params eval_params = bkg.region_data->my_beads.params_nn[ibd];
  // make custom emphasis vector for this well using pointers to the per-HP vectors
  DynamicEmphasis ( eval_params );
  lev_mar_scratch.FillObserved ( bkg.region_data->my_trace, eval_params.trace_ndx );
  // now we're set up, do the individual steps
  ComputeCachedPartialDerivatives ( eval_params, PartialDeriv_mask );
  lm_state.residual[ibd] = lev_mar_scratch.CalculateFitError ( NULL,NUMFB );

  if ( ( ibd == bkg.region_data->my_beads.DEBUG_BEAD ) && ( bkg.my_debug.trace_dbg_file != NULL ) )
  {
    bkg.my_debug.DebugBeadIteration ( eval_params,step_rp[0], iter,lm_state.residual[ibd],&bkg.region_data->my_regions );
  }

  lm_state.reg_error += lm_state.residual[ibd];

  if ( lm_state.WellBehavedBead ( ibd ) ) // only use "well-behaved" wells at any iteration
  {
    // if  reg_wells>0, continue same matrix, otherwise start a new one
    BuildMatrix ( reg_fit, ( reg_wells>0 ), ( ibd==bkg.region_data->my_beads.DEBUG_BEAD ) );
    reg_wells++;
  }
}

// reg_proc = TRUE
int MultiFlowLevMar::LevMarAccumulateRegionDerivsForSampledActiveBeadList (
  reg_params &eval_rp,
  BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
  int iter )
{
  int reg_wells = 0;
  lm_state.avg_resid = CalculateCurrentResidualForTestBeads (); // execute on every bead once to get current avg residual // @TODO: execute on every bead we care about in this fit???

  bead_params eval_params; // each bead over-rides this, so we only need for function call
  FillDerivativeStepCache ( eval_params,eval_rp,PartialDeriv_mask );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;
    AccumulateRegionDerivForOneBead ( ibd, reg_wells, reg_fit, PartialDeriv_mask, iter );
  }
  return ( reg_wells ); // number of live wells fit for region
}

// reg_proc = TRUE
int MultiFlowLevMar::LevMarAccumulateRegionDerivsForActiveBeadList (
  reg_params &eval_rp,
  BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
  int iter )
{
  int reg_wells = 0;
  lm_state.avg_resid = CalculateCurrentResidualForTestBeads ( ); // execute on every bead once to get current avg residual // @TODO: execute on every bead we care about in this fit???

  // Tricky here:  we fill the regional parameter & nuc_step and >apply< them across all beads
  // That way we don't recalculate regional parameters/steps for each bead as we take derivatives
  bead_params eval_params; // each bead uses own parameters, but we need this for the function call
  FillDerivativeStepCache ( eval_params,eval_rp,PartialDeriv_mask );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;

    AccumulateRegionDerivForOneBead ( ibd, reg_wells,  reg_fit, PartialDeriv_mask, iter );
  }
  return ( reg_wells ); // number of live wells fit for region
}


void MultiFlowLevMar::LevMarFitRegion (
  BkgFitMatrixPacker *reg_fit )
{
  bool cont_proc = false;
  if (lm_state.reg_lambda>lm_state.reg_lambda_max)
    cont_proc = true; // we've quit already but don't know it

  reg_params eval_rp;
  int defend_against_infinity = 0; // make sure we can't get trapped forever

  reg_fit->resetNumException();
  while ( !cont_proc && defend_against_infinity<EFFECTIVEINFINITY )
  {
    defend_against_infinity++;
    eval_rp = bkg.region_data->my_regions.rp;
    ResetRegionBeadShiftsToZero ( &eval_rp );

    if ( reg_fit->GetOutput ( ( float * ) ( &eval_rp ),lm_state.reg_lambda,lm_state.reg_regularizer ) != LinearSolverException )
    {
      reg_params_ApplyLowerBound ( &eval_rp,&bkg.region_data->my_regions.rp_low );
      reg_params_ApplyUpperBound ( &eval_rp,&bkg.region_data->my_regions.rp_high );
      if ( bkg.global_defaults.signal_process_control.generic_test_flag )
        SetAverageDiffusion ( eval_rp );

      // make a copy so we can modify it to null changes in new_rp that will
      // we push into individual bead parameters
      reg_params new_rp = eval_rp;
      FillTshiftCache ( new_rp.tshift );

      float new_reg_error = TryNewRegionalParameters ( &new_rp );
      // are the new parameters better?
      if ( new_reg_error < lm_state.reg_error )
      {
        // it's better...apply the change to all the beads and the region
        // update regional parameters
        bkg.region_data->my_regions.rp = new_rp;

        // re-calculate current parameter values for each bead as necessary
        UpdateBeadParametersFromRegion ( &new_rp );

        lm_state.ReduceRegionStep();
        cont_proc = true;
      }
      else
      {
        cont_proc = lm_state.IncreaseRegionStep();
      }
    }
    else
    {
      if ((lm_state.reg_regularizer>0.0f) & lm_state.LogMessage()){
        printf ( "LM solver region exception %f %f in %d %d:\n", lm_state.reg_lambda , lm_state.reg_regularizer,bkg.region_data->region->col, bkg.region_data->region->row);
      }
      lm_state.IncreaseRegionRegularizer(); // linear solver exception often from zero derivative - handle this case
      cont_proc = lm_state.IncreaseRegionStep();
    }
  }

  if ( defend_against_infinity>100 )
    printf ( "RegionLevMar taking a while: %d\n",defend_against_infinity );
}


float MultiFlowLevMar::CalculateCurrentResidualForTestBeads ( )
{
  float avg_error = 0.0;
  int cnt = 0;
  // use the current region parameter for all beads evaluated
  // don't update nuc step as we don't need to recalculate it
  bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep ( &bkg.region_data->my_regions.rp,bkg.region_data->time_c,bkg.region_data->my_flow );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;

    lev_mar_scratch.FillObserved ( bkg.region_data->my_trace,bkg.region_data->my_beads.params_nn[ibd].trace_ndx ); // set scratch space for this bead
    FillScratchForEval ( &bkg.region_data->my_beads.params_nn[ibd], &bkg.region_data->my_regions.rp, bkg.region_data->my_regions.cache_step );
    avg_error += lev_mar_scratch.CalculateFitError ( NULL,NUMFB );
    cnt++;
  }
  bkg.region_data->my_regions.cache_step.Unlock();

  if ( cnt > 0 )
  {
    avg_error = avg_error / cnt;
  }
  else   // in case the semantics need to be traced
  {
    avg_error = std::numeric_limits<float>::quiet_NaN();
  }
  return ( avg_error );
}



float MultiFlowLevMar::TryNewRegionalParameters ( reg_params *new_rp )
{
  float new_reg_error = 0.0f;
  //@TODO make this own cache not universal cache override
  bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep ( new_rp,bkg.region_data->time_c,bkg.region_data->my_flow );

  // calculate new parameters for everything and re-check residual error
  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;

    bead_params eval_params = bkg.region_data->my_beads.params_nn[ibd];
    // apply the region-wide adjustments to each individual well
    UpdateOneBeadFromRegion ( &eval_params,&bkg.region_data->my_beads.params_high, &bkg.region_data->my_beads.params_low,new_rp,bkg.region_data->my_flow.dbl_tap_map );

    lev_mar_scratch.FillObserved ( bkg.region_data->my_trace, bkg.region_data->my_beads.params_nn[ibd].trace_ndx );
    FillScratchForEval ( &eval_params, new_rp, bkg.region_data->my_regions.cache_step );

    new_reg_error += lev_mar_scratch.CalculateFitError ( NULL,NUMFB );
  }
  // undo lock so we can reuse
  bkg.region_data->my_regions.cache_step.Unlock();

  return ( new_reg_error );
}


void MultiFlowLevMar::UpdateBeadParametersFromRegion ( reg_params *new_rp )
{
  // update all but ignored beads
  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    UpdateOneBeadFromRegion ( &bkg.region_data->my_beads.params_nn[ibd],&bkg.region_data->my_beads.params_high,&bkg.region_data->my_beads.params_low,new_rp, bkg.region_data->my_flow.dbl_tap_map );
  }
}

void MultiFlowLevMar::LevMarBuildMatrixForBead ( int ibd,
    bool well_only_fit,
    reg_params &eval_rp, NucStep &cache_step,

    BkgFitMatrixPacker *well_fit,
    unsigned int PartialDeriv_mask,
    int iter )
{
  // get the current parameter values for this bead
  bead_params eval_params = bkg.region_data->my_beads.params_nn[ibd];
  // make custom emphasis vector for this well using pointers to the per-HP vectors
  DynamicEmphasis ( eval_params );
  lev_mar_scratch.FillObserved ( bkg.region_data->my_trace, eval_params.trace_ndx );
  // now we're set up, do the individual steps
  ComputePartialDerivatives ( eval_params, eval_rp, cache_step, PartialDeriv_mask );
  lm_state.residual[ibd] = lev_mar_scratch.CalculateFitError ( NULL,NUMFB );

  if ( ( ibd == bkg.region_data->my_beads.DEBUG_BEAD ) && ( bkg.my_debug.trace_dbg_file != NULL ) )
  {
    bkg.my_debug.DebugBeadIteration ( eval_params,eval_rp, iter, lm_state.residual[ibd],&bkg.region_data->my_regions );
  }

  // assemble jtj matrix and rhs matrix for per-well fitting
  // only the non-zero elements of computed
  // automatically start a new matrix
  BuildMatrix ( well_fit,false, ( ibd == bkg.region_data->my_beads.DEBUG_BEAD ) );
}


// reg_proc ==FALSE
float MultiFlowLevMar::LevMarFitToActiveBeadList (
  bool well_only_fit,
  reg_params &eval_rp,
  BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
  int bead_iterations, bool isSample )
{

  float req_done = 0.0f; // beads have improved?
  float executed_bead = 0.001f; // did we actually check this bead, or was it skipped for some reason

  step_nuc_cache[0].ForceLockCalculateNucRiseCoarseStep ( &eval_rp,bkg.region_data->time_c,bkg.region_data->my_flow );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ( !bkg.region_data->my_beads.high_quality[ibd] & lm_state.skip_beads ) || lm_state.well_completed[ibd] )  // primarily concerned with regional fits for this iteration, catch up remaining wells later
      continue;

    for ( int iter=0; iter < bead_iterations; ++iter )
    {
      LevMarBuildMatrixForBead ( ibd,  well_only_fit,eval_rp, step_nuc_cache[0], well_fit,PartialDeriv_mask, iter );
      req_done += LevMarFitOneBead ( ibd, eval_rp,well_fit,well_only_fit );
    }
    executed_bead+=1;
  }
  // free up
  step_nuc_cache[0].Unlock();
  return ( req_done/executed_bead );
}

//TODO: Refactor to reuse previous function
float MultiFlowLevMar::LevMarFitToRegionalActiveBeadList (

  bool well_only_fit,
  reg_params &eval_rp,
  BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
  int iter )
{

  float req_done = 0.0f; // beads have improved?
  float executed_bead = 0.001f; // did we actually check this bead, or was it skipped for some reason

  step_nuc_cache[0].ForceLockCalculateNucRiseCoarseStep ( &eval_rp,bkg.region_data->time_c,bkg.region_data->my_flow );
  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    // if this iteration is a region-wide parameter fit, then only process beads
    // in the selection sub-group
    if ( ExcludeBead ( ibd ) )
      continue;


    LevMarBuildMatrixForBead ( ibd,  well_only_fit,eval_rp,step_nuc_cache[0],well_fit,PartialDeriv_mask,iter );

    executed_bead+=1;
    req_done += LevMarFitOneBead ( ibd, eval_rp,well_fit,well_only_fit );
  }
  step_nuc_cache[0].Unlock();
  return ( req_done/executed_bead );
}


// the decision logic per iteration is particularly tortured in these routines
// and needs to be revisited
int MultiFlowLevMar::LevMarFitOneBead ( int ibd,
                                        reg_params &eval_rp,
                                        BkgFitMatrixPacker *well_fit,
                                        bool well_only_fit )
{
  int bead_not_improved = 0;
  bead_params eval_params;

  int defend_against_infinity=0;
  // we only need to re-calculate the PartialDeriv's if we actually adjust something
  // if the fit didn't improve...adjust lambda and retry right here, that way
  // we can save all the work of re-calculating the PartialDeriv's
  bool cont_proc = false;
  // check to see if we're out of bounds and continuing by intertia
  if (lm_state.lambda[ibd]>lm_state.lambda_max)
    cont_proc = true; // done with this bead but don't know it
    
  well_fit->resetNumException();
  while ( ( !cont_proc ) && ( defend_against_infinity<EFFECTIVEINFINITY ) )
  {
    defend_against_infinity++;
    eval_params = bkg.region_data->my_beads.params_nn[ibd];
    float achg = 0.0f;

    // solve equation and adjust parameters

    if ( well_fit->GetOutput ( ( float * ) ( &eval_params ),lm_state.lambda[ibd] ,lm_state.regularizer[ibd]) != LinearSolverException )
    {
      // bounds check new parameters
      params_ApplyLowerBound ( &eval_params,&bkg.region_data->my_beads.params_low );
      params_ApplyUpperBound ( &eval_params,&bkg.region_data->my_beads.params_high );
      params_ApplyAmplitudeZeros ( &eval_params, bkg.region_data->my_flow.dbl_tap_map ); // double-tap

      FillScratchForEval ( &eval_params, &eval_rp, bkg.region_data->my_regions.cache_step );
      float res = lev_mar_scratch.CalculateFitError ( NULL, NUMFB );

      if ( res < ( lm_state.residual[ibd] ) )
      {
        achg=CheckSignificantSignalChange ( &bkg.region_data->my_beads.params_nn[ibd],&eval_params,NUMFB );
        bkg.region_data->my_beads.params_nn[ibd] = eval_params;

        lm_state.ReduceBeadLambda ( ibd );
        lm_state.residual[ibd] = res;

        cont_proc = true;
      }
      else
      {
//        params_CopyHits(&eval_params,&bkg.region_data->my_beads.params_nn[ibd]); // store hits
        lm_state.IncreaseBeadLambda ( ibd );
      }
    }
    else
    {
      if ( ( ibd == bkg.region_data->my_beads.DEBUG_BEAD ) && ( bkg.my_debug.trace_dbg_file != NULL ) )
      {
        fprintf ( bkg.my_debug.trace_dbg_file,"singular matrix\n" );
        fflush ( bkg.my_debug.trace_dbg_file );
      }
      if ( ((2.0*lm_state.regularizer[ibd])>LM_BEAD_REGULARIZER ) & lm_state.LogMessage())
      {
        //regularization has failed to stabilize as well
        // show this result only if total failure (such as nans contaminating the matrix)
        printf ( "Well singular matrix: %d %f %f\n", ibd, lm_state.lambda[ibd], lm_state.regularizer[ibd] );
      }
      lm_state.IncreaseBeadLambda ( ibd );
      // failed the solver therefore must regularize in case we have a zero row or column in the derivatives
      lm_state.IncreaseRegularizer ( ibd );
    }
    // if signal isn't making much progress, and we're ready to abandon lev-mar, deal with it
    if ( ( achg < lm_state.min_amplitude_change ) && ( lm_state.lambda[ibd] >= lm_state.lambda_max ) )
    {
      bead_not_improved = 1;
      if ( well_only_fit )
      {
        // this well is finished
        // lm_state.FinishCurrentBead ( ibd );
        FinishBead ( ibd );
        cont_proc = true;
      }
    }

    // if regional fitting...we can get stuck here if we can't improve until the next
    // regional fit
    if ( !well_only_fit && ( lm_state.lambda[ibd] >= lm_state.lambda_escape ) )
      cont_proc = true;
    if (lm_state.lambda[ibd]>lm_state.lambda_max)
      cont_proc = true; // done with this bead but don't know it - how would this get reset?
  }
  
  // we have taken an optimization step
  // this drives the amplitude data away from the desired near-integer values in the first few flows
  // we now shrink the values towards the theoretically best values ignoring the effect on the likelihood
  // because the other parameters will make up for them
  if ((lm_state.shrink_factor>0.0f) & (lm_state.nonclonal_call_penalty_enforcement>0)){
    params_ShrinkTowardsIntegers(&bkg.region_data->my_beads.params_nn[ibd],lm_state.shrink_factor);
  }

  if ( defend_against_infinity>100 )
    printf ( "Problem with bead %d %d\n", ibd, defend_against_infinity );
  return ( bead_not_improved );
}


// arguably this is part of "scratch space" operations and should be part of that object
// must have "pointed" scratch space at the current bead parameters
void MultiFlowLevMar::ComputePartialDerivatives ( bead_params &eval_params, reg_params &eval_rp, NucStep &cache_step, unsigned int PartialDeriv_mask )
{
  float *output;
  CpuStep_t *StepP;

  for ( int step=0;step<fit_control.fitParams.NumSteps;step++ )
  {
    StepP = &fit_control.fitParams.Steps[step];
    if ( ( StepP->PartialDerivMask & PartialDeriv_mask ) == 0 )
      continue; // only do the steps we are interested in.

    output = lev_mar_scratch.scratchSpace + step*lev_mar_scratch.bead_flow_t;

    ComputeOnePartialDerivative ( output, StepP,  eval_params, eval_rp, cache_step );

  }
}

// if I have cached region parameters and steps
void MultiFlowLevMar::ComputeCachedPartialDerivatives ( bead_params &eval_params,  unsigned int PartialDeriv_mask )
{
  float *output;
  CpuStep_t *StepP;

  for ( int step=0;step<fit_control.fitParams.NumSteps;step++ )
  {
    StepP = &fit_control.fitParams.Steps[step];
    if ( ( StepP->PartialDerivMask & PartialDeriv_mask ) == 0 )
      continue; // only do the steps we are interested in.

    output = lev_mar_scratch.scratchSpace + step*lev_mar_scratch.bead_flow_t;

    ComputeOnePartialDerivative ( output, StepP,  eval_params, step_rp[step], step_nuc_cache[step] );

  }
}

void MultiFlowLevMar::FillDerivativeStepCache ( bead_params &eval_params, reg_params &eval_rp, unsigned int PartialDeriv_mask )
{
  CpuStep_t *StepP;
  float backup[lev_mar_scratch.bead_flow_t]; // more than we need

  for ( int step=0;step<fit_control.fitParams.NumSteps;step++ )
  {
    StepP = &fit_control.fitParams.Steps[step];
    if ( ( StepP->PartialDerivMask & PartialDeriv_mask ) == 0 )
      continue; // only do the steps we are interested in.
    step_rp[step] = eval_rp; // start where we should be
    // do my special step for each thing we're executing
    DoStepDiff ( 1,backup, StepP,&eval_params,&step_rp[step] );
    step_nuc_cache[step].ForceLockCalculateNucRiseCoarseStep ( &step_rp[step],bkg.region_data->time_c,bkg.region_data->my_flow );

    // because our codebase is somewhat toxic, we cannot effectively tell when we're doing region derivatives or bead derivatives
    // so we'll reset these guys individually each time we take a derivative (crazy)
    DoStepDiff ( 0,backup, StepP,&eval_params,&step_rp[step] ); // reset parameter to default value
  }
}


void MultiFlowLevMar::ComputeOnePartialDerivative (
  float *output,
  CpuStep_t *StepP,
  bead_params &eval_params,
  reg_params &eval_rp,
  NucStep &cache_step )
{
  if ( StepP->PartialDerivMask == DFDGAIN )
  {
    Dfdgain_Step ( output, &eval_params );
  }
  else if ( StepP->PartialDerivMask == DFDERR )
  {
    Dfderr_Step ( output, &eval_params );
  }
  else if ( StepP->PartialDerivMask == YERR )
  {
    Dfyerr_Step ( output );
  }
  else if ( StepP->PartialDerivMask == DFDTSH )
  {
    // check cache if we've evaluated this guy already - should be unnecessary
    //FillTshiftCache(eval_rp.tshift);
    MultiFlowComputePartialDerivOfTimeShift ( output,&eval_params, &eval_rp,cache_slope );
  }
  else if ( StepP->PartialDerivMask == FVAL ) // set up the baseline for everything else
  {
    // for safety
    memset ( output, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );
    memset ( lev_mar_scratch.ival, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );

    // fill in the function value & incorporation trace
    MultiFlowComputeCumulativeIncorporationSignal ( &eval_params,&eval_rp,lev_mar_scratch.ival,cache_step,lev_mar_scratch.cur_bead_block,bkg.region_data->time_c,bkg.region_data->my_flow,bkg.math_poiss );
    MultiFlowComputeTraceGivenIncorporationAndBackground ( output,&eval_params,&eval_rp,lev_mar_scratch.ival,cache_sbg,bkg.region_data->my_regions,lev_mar_scratch.cur_buffer_block,bkg.region_data->time_c,bkg.region_data->my_flow,use_vectorization, lev_mar_scratch.bead_flow_t );
    // add clonal restriction here to penalize non-integer clonal reads
    lm_state.ApplyClonalRestriction ( output, &eval_params,bkg.region_data->time_c.npts() );
  }
  else if ( StepP->diff != 0 )
  {
    float ivtmp[lev_mar_scratch.bead_flow_t];
    float backup[lev_mar_scratch.bead_flow_t]; // more than we need
    DoStepDiff ( 1,backup, StepP,&eval_params,&eval_rp );
    float *local_iv = lev_mar_scratch.ival;

    if ( StepP->doBoth )
    {
      local_iv = ivtmp;
      //@TODO nuc rise recomputation?
      MultiFlowComputeCumulativeIncorporationSignal ( &eval_params,&eval_rp,local_iv,cache_step,lev_mar_scratch.cur_bead_block,bkg.region_data->time_c,bkg.region_data->my_flow,bkg.math_poiss );
    }

    MultiFlowComputeTraceGivenIncorporationAndBackground ( output,&eval_params,&eval_rp,local_iv,cache_sbg,bkg.region_data->my_regions,lev_mar_scratch.cur_buffer_block,bkg.region_data->time_c,bkg.region_data->my_flow,use_vectorization, lev_mar_scratch.bead_flow_t );
    // add clonal restriction here to penalize non-integer clonal reads
    lm_state.ApplyClonalRestriction ( output, &eval_params,bkg.region_data->time_c.npts() );

    CALC_PartialDeriv_W_EMPHASIS_LONG ( lev_mar_scratch.fval,output,lev_mar_scratch.custom_emphasis,lev_mar_scratch.bead_flow_t,StepP->diff );

    DoStepDiff ( 0,backup, StepP,&eval_params,&eval_rp ); // reset parameter to default value
  }
}


void MultiFlowLevMar::Dfdgain_Step ( float *output,bead_params *eval_p )
{

// partial w.r.t. gain is the function value divided by the current gain

  float* src[NUMFB];
  float* dst[NUMFB];
  float* em[NUMFB];
  // set up across flows
  for ( int fnum=0;fnum < NUMFB;fnum++ )
  {
    src[fnum] = &lev_mar_scratch.fval[bkg.region_data->time_c.npts() *fnum];
    dst[fnum] = &output[bkg.region_data->time_c.npts() *fnum];
    em[fnum] = &lev_mar_scratch.custom_emphasis[bkg.region_data->time_c.npts() *fnum];
  }
  // execute in parallel
#ifdef __INTEL_COMPILER
  {
    for ( int fnum=0; fnum<NUMFB; fnum++ )
    {
      for ( int i=0;i < bkg.region_data->time_c.npts();i++ )
        ( dst[fnum] ) [i] = ( src[fnum] ) [i]* ( em[fnum] ) [i]/eval_p->gain;
    }
  }
#else
  if ( use_vectorization )
    Dfdgain_Step_Vec ( NUMFB, dst, src, em, bkg.region_data->time_c.npts(), eval_p->gain );
  else
    for ( int fnum=0; fnum<NUMFB; fnum++ )
      for ( int i=0;i < bkg.region_data->time_c.npts();i++ )
        ( dst[fnum] ) [i] = ( src[fnum] ) [i]* ( em[fnum] ) [i]/eval_p->gain;
#endif
}

void MultiFlowLevMar::Dfderr_Step ( float *output, bead_params *eval_p )
{
  // partial w.r.t. darkness is the dark_matter_compensator multiplied by the emphasis

  float* dst[NUMFB];
  float* et[NUMFB];
  float* em[NUMFB];
  // set up
  for ( int fnum=0;fnum < NUMFB;fnum++ )
  {
    dst[fnum] = &output[bkg.region_data->time_c.npts() *fnum];
    em[fnum] = &lev_mar_scratch.custom_emphasis[bkg.region_data->time_c.npts() *fnum];
    et[fnum] = bkg.region_data->my_regions.missing_mass.dark_nuc_comp[bkg.region_data->my_flow.flow_ndx_map[fnum]];
  }
  //execute
#ifdef __INTEL_COMPILER
  {
    for ( int fnum=0; fnum<NUMFB;fnum++ )
    {
      for ( int i=0;i < bkg.region_data->time_c.npts();i++ )
        ( dst[fnum] ) [i] = ( et[fnum] ) [i]* ( em[fnum] ) [i];
    }
  }
#else
  if ( use_vectorization )
    Dfderr_Step_Vec ( NUMFB, dst, et, em, bkg.region_data->time_c.npts() );
  else
  {
    for ( int fnum=0; fnum<NUMFB;fnum++ )
    {
      for ( int i=0;i < bkg.region_data->time_c.npts();i++ )
        ( dst[fnum] ) [i] = ( et[fnum] ) [i]* ( em[fnum] ) [i];
    }
  }
#endif
}

//@TODO: this is closely related to fit error, but the goal here is to get y-observed for the lev-mar step
void MultiFlowLevMar::Dfyerr_Step ( float *y_minus_f_emphasized )
{

  for ( int fnum=0;fnum < NUMFB;fnum++ )
  {
    float eval;

    for ( int i=0;i<bkg.region_data->time_c.npts();i++ ) // real data only
    {
      int ti= i+fnum*bkg.region_data->time_c.npts();
      eval = lev_mar_scratch.observed[ti]-lev_mar_scratch.fval[ti];
      eval = eval*lev_mar_scratch.custom_emphasis[ti];
      y_minus_f_emphasized[ti] = eval;

    }
  }
}

// this is the one PartialDeriv that really isn't computed very well w/ the stansard numeric computation method
void MultiFlowLevMar::MultiFlowComputePartialDerivOfTimeShift ( float *fval,struct bead_params *p, struct reg_params *reg_p, float *neg_sbg_slope )
{

  int fnum;

  float *vb_out;
  float *flow_deriv;

  // parallel fill one bead parameter for block of flows
  FillBufferParamsBlockFlows ( &lev_mar_scratch.cur_buffer_block,p,reg_p,bkg.region_data->my_flow.flow_ndx_map,bkg.region_data->my_flow.buff_flow );

  for ( fnum=0;fnum<NUMFB;fnum++ )
  {
    vb_out = fval + fnum*bkg.region_data->time_c.npts();  // get ptr to start of the function evaluation for the current flow
    flow_deriv = &neg_sbg_slope[fnum*bkg.region_data->time_c.npts() ];                  // get ptr to pre-shifted slope

    // now this diffeq looks like all the others
    // because linearity of derivatives gets passed through
    // flow_deriv in this case is in fact the local derivative with respect to time of the background step
    // so it can be passed through the equation as though it were a background term
    BlueSolveBackgroundTrace ( vb_out,flow_deriv,bkg.region_data->time_c.npts(),&bkg.region_data->time_c.deltaFrame[0],lev_mar_scratch.cur_buffer_block.tauB[fnum],lev_mar_scratch.cur_buffer_block.etbR[fnum] );
    // isolate gain and emphasis so we can reuse diffeq code
  }
  MultiplyVectorByVector ( fval,lev_mar_scratch.custom_emphasis,lev_mar_scratch.bead_flow_t );
  // gain invariant so can be done to all at once
  MultiplyVectorByScalar ( fval,p->gain,lev_mar_scratch.bead_flow_t );
}

char *MultiFlowLevMar::findName ( float *ptr )
{
  int i;
  for ( i=0;i<fit_control.fitParams.NumSteps;i++ )
  {
    if ( ptr >= fit_control.fitParams.Steps[i].ptr && ptr < ( fit_control.fitParams.Steps[i].ptr + lev_mar_scratch.bead_flow_t ) )
      return ( fit_control.fitParams.Steps[i].name );
  }
  return ( NULL );
}


void MultiFlowLevMar::InitRandomCache()
{

  cache_sbg = new float [lev_mar_scratch.bead_flow_t];
  cache_slope = new float [lev_mar_scratch.bead_flow_t];
}

void MultiFlowLevMar::DeleteRandomCache()
{

  delete[] cache_sbg;
  delete[] cache_slope;
  cache_sbg = NULL;
  cache_slope = NULL;
}




void MultiFlowLevMar::ChooseSkipBeads ( bool _skip_beads )
{
  lm_state.skip_beads = _skip_beads;  // make sure we fit every bead here
}

bool MultiFlowLevMar::SkipBeads()
{
  return ( lm_state.skip_beads );
}

void MultiFlowLevMar::InitTshiftCache()
{
  tshift_cache = -10.0f;
}

//bkg.region_data->my_regions.rp.tshift
void MultiFlowLevMar::FillTshiftCache ( float my_tshift )
{
  if ( my_tshift != tshift_cache )
  {
    // for safety
    memset ( cache_sbg, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );
    bkg.region_data->emptytrace->GetShiftedBkg ( my_tshift, bkg.region_data->time_c, cache_sbg );
    memset ( cache_slope, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );
    bkg.region_data->emptytrace->GetShiftedSlope ( my_tshift, bkg.region_data->time_c, cache_slope );

    tshift_cache = my_tshift;
  }
}

bool MultiFlowLevMar::ExcludeBead ( int ibd )
{
  // test to see whether this bead is excluded in the regional sampling
  // or in the different region groups

  bool exclude = lm_state.well_completed[ibd];
                 
  if (bkg.region_data->my_beads.isSampled) {
    // regional sampling enabled
    exclude = exclude || !bkg.region_data->my_beads.StillSampled(ibd);
  }
  else
  {
    // rolling regional groups enabled
    exclude = exclude || !bkg.region_data->my_beads.BeadIncluded(ibd, true) || !lm_state.ValidBeadGroup ( ibd );
  }
  return ( exclude );
}

void MultiFlowLevMar::FinishBead ( int ibd )
{
  // enforces synchronization between lm_state & my_beads
  lm_state.FinishCurrentBead ( ibd );
  bkg.region_data->my_beads.ExcludeFromSampled ( ibd );
}



void DoStepDiff ( int add, float *archive, CpuStep_t *step, struct bead_params *p, struct reg_params *reg_p )
{
  int i;
  float *dptr = NULL;
  // choose my parameter
  if ( step->paramsOffset != NOTBEADPARAM )
  {
    dptr = ( float * ) ( ( char * ) p + step->paramsOffset );
  }
  else if ( step->regParamsOffset != NOTREGIONPARAM )
  {
    dptr = ( float * ) ( ( char * ) reg_p + step->regParamsOffset );
  }
  else if ( step->nucShapeOffset != NOTNUCRISEPARAM )
  {
    dptr = ( float * ) ( ( char * ) & ( reg_p->nuc_shape ) + step->nucShapeOffset );
  }
  // update my parameter
  if ( dptr != NULL )
  {
    for ( i=0;i<step->len;i++ )
    {
      if ( add )
      {
        archive[i] = *dptr;
        *dptr += step->diff; // change
      }
      else
        *dptr = archive[i]; // reset
      dptr++;
    }
  }
}

void UpdateOneBeadFromRegion ( bead_params *p, bound_params *hi, bound_params *lo, reg_params *new_rp, int *dbl_tap_map )
{
  for ( int i=0;i < NUMFB;i++ )
    p->Ampl[i] += new_rp->Ampl[i];
  p->R += new_rp->R;
  p->Copies += new_rp->Copies;

  params_ApplyLowerBound ( p,lo );
  params_ApplyUpperBound ( p,hi );
  params_ApplyAmplitudeZeros ( p,dbl_tap_map ); // double-taps
}

void IdentifyParametersFromSample ( BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask, bool skip_beads, const LevMarBeadAssistant& lm_state )
{
  if ( ( well_mask & DFDPDM ) >0 )  // only if actually fitting dmult do we need to do this step
    IdentifyDmultFromSample ( my_beads,my_regions,skip_beads, lm_state );
  if ( ( reg_mask & DFDMR ) >0 ) // only if actually fitting NucMultiplyRatio do I need to identify NMR so we don't slide around randomly
    IdentifyNucMultiplyRatioFromSample ( my_beads, my_regions );
}

void IdentifyDmultFromSample ( BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads, const LevMarBeadAssistant& lm_state )
{
  float mean_dmult = my_beads.CenterDmultFromSample ( skip_beads );

  for ( int nnuc=0;nnuc < NUMNUC;nnuc++ )
  {
    my_regions.rp.d[nnuc] *= mean_dmult;
  }
}

void IdentifyNucMultiplyRatioFromSample ( BeadTracker &my_beads, RegionTracker &my_regions )
{
  // provide identifiability constraint
  float mean_x = 0.0f;
  for ( int nnuc=0; nnuc<NUMNUC; nnuc++ )
  {
    mean_x += my_regions.rp.NucModifyRatio[nnuc];
  }
  mean_x /=NUMNUC;

  for ( int nnuc=0; nnuc<NUMNUC; nnuc++ )
  {
    my_regions.rp.NucModifyRatio[nnuc] /=mean_x;
  }

  my_beads.RescaleRatio ( 1.0f/mean_x );
}

void IdentifyParameters ( BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask, unsigned int reg_mask, bool skip_beads )
{
  if ( ( well_mask & DFDPDM ) >0 )  // only if actually fitting dmult do we need to do this step
    IdentifyDmult ( my_beads,my_regions,skip_beads );
  if ( ( reg_mask & DFDMR ) >0 ) // only if actually fitting NucMultiplyRatio do I need to identify NMR so we don't slide around randomly
    IdentifyNucMultiplyRatio ( my_beads, my_regions );
}

void IdentifyDmult ( BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads )
{
  float mean_dmult = my_beads.CenterDmult ( skip_beads ); // only active set

  for ( int nnuc=0;nnuc < NUMNUC;nnuc++ )
  {
    my_regions.rp.d[nnuc] *= mean_dmult;
  }
}

void IdentifyNucMultiplyRatio ( BeadTracker &my_beads, RegionTracker &my_regions )
{
  // provide identifiability constraint
  float mean_x = 0.0f;
  for ( int nnuc=0; nnuc<NUMNUC; nnuc++ )
  {
    mean_x += my_regions.rp.NucModifyRatio[nnuc];
  }
  mean_x /=NUMNUC;

  for ( int nnuc=0; nnuc<NUMNUC; nnuc++ )
  {
    my_regions.rp.NucModifyRatio[nnuc] /=mean_x;
  }

  my_beads.RescaleRatio ( 1.0f/mean_x );
}
