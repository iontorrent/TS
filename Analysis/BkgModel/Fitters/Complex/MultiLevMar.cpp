/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MultiLevMar.h"
#include "MiscVec.h"
#include <assert.h>
#include "BkgFitMatDat.h"

MultiFlowLevMar::MultiFlowLevMar ( SignalProcessingMasterFitter &_bkg, int flow_block_size,
                                   master_fit_type_table *table ) :
  bkg ( _bkg ),
  lev_mar_cur_bead_block( flow_block_size ),
  lev_mar_cur_buffer_block( flow_block_size ),
  fit_control( table )
{
  // create matrix packing object(s)
  //Note:  scratch-space is used directly by the matrix packer objects to get the derivatives
  // so this object >must< persist in order to be used by the fit control object in the Lev_Mar fit.
  // Allocate directly the annoying pointers that the lev-mar object uses for control

  lm_state.SetNonIntegerPenalty ( bkg.global_defaults.fitter_defaults.clonal_call_scale,bkg.global_defaults.fitter_defaults.clonal_call_penalty,MAGIC_MAX_CLONAL_HP_LEVEL );
  lm_state.kmult_penalty_scale = bkg.global_defaults.signal_process_control.kmult_penalty; // make sure this is set on initialization(!)
  lm_state.AllocateBeadFitState ( bkg.region_data->my_beads.numLBeads );
  lm_state.AssignBeadsToRegionGroups ();

  lev_mar_scratch.Allocate ( bkg.region_data->time_c.npts(),BkgFitStructures::NumSteps, flow_block_size );
  fit_control.AllocPackers ( lev_mar_scratch.scratchSpace, bkg.global_defaults.signal_process_control.no_RatioDrift_fit_first_20_flows,bkg.global_defaults.signal_process_control.fitting_taue,
                             bkg.global_defaults.signal_process_control.hydrogenModelType, lev_mar_scratch.bead_flow_t, bkg.region_data->time_c.npts(), flow_block_size );
  use_vectorization = bkg.global_defaults.signal_process_control.use_vectorization;

  // regional parameters are the same for each bead,
  // we recalculate these for each derivative * all beads
  // so instead calculate once and reuse for each bead the little steps we will take
  step_rp.resize ( BkgFitStructures::NumSteps ); // waste a little space as we don't use all steps in all fits, but good enough
  step_nuc_cache.resize ( BkgFitStructures::NumSteps ); // waste a little space as we don't use all steps, but good enough
  for ( unsigned int istep=0; istep<step_nuc_cache.size(); istep++ )
    step_nuc_cache[istep].Alloc ( bkg.region_data->time_c.npts(), flow_block_size );

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
int MultiFlowLevMar::MultiFlowSpecializedSampledLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction, int flow_key, int flow_block_size,
                                                                      int flow_block_start )
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
      int nbeads = DoSampledRegionIteration ( reg_fit,total_iter, flow_key, flow_block_size, flow_block_start );
      IF_OPTIMIZER_DEBUG(bkg.inception_state, bkg.debugSaver.WriteData(reg_fit, bkg.region_data->my_regions.rp ,flow_block_start, bkg.GetRegion(),reg_fit->compNames, nbeads));
      total_iter++;
    }
  }

  // if alternating bead and regional steps
  if ( do_both )
  {
    for ( int loc_iter=0 ; ( loc_iter<number_region_iterations_wanted )  ; loc_iter++ )
    {
      // do one well iteration
      bool skip_region = DoSampledBeadIteration ( false, well_fit, total_iter, flow_key, flow_block_size, flow_block_start );
      total_iter++;
      if ( skip_region )
      {
        if (loc_iter==0){
          printf("Failed beads lead to skip region entirely\n");
        }
        total_iter = 2*number_region_iterations_wanted;
        break;
      }
      // do one region iteration
      if ( !skip_region )
      {
        int nbeads = DoSampledRegionIteration ( reg_fit,total_iter, flow_key, flow_block_size, flow_block_start );
        IF_OPTIMIZER_DEBUG(bkg.inception_state, bkg.debugSaver.WriteData(reg_fit, bkg.region_data->my_regions.rp ,flow_block_start, bkg.GetRegion(),reg_fit->compNames, nbeads));
        total_iter++;
      }
    }
  }
  // just bead steps to finish off
  for ( int loc_iter=0 ; loc_iter<additional_bead_only_iterations; loc_iter++ )
  {
    DoSampledBeadIteration ( true, well_fit, total_iter, flow_key, flow_block_size, flow_block_start );
    total_iter++;
  }
  if ((lm_state.region_success_step<1) & (reg_fit!=NULL)){
    printf("No successes: MultiFlowSpecializedSampledLevMarFitParameters in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  CleanTerminateOptimization();
  return ( total_iter );
}



// fitting all beads and some beads per region
int MultiFlowLevMar::MultiFlowSpecializedLevMarFitParameters ( int additional_bead_only_iterations, int number_region_iterations_wanted, BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit,float lambda_start,int clonal_restriction, int flow_key, int flow_block_size,
                                                               int flow_block_start )
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
      int nbeads = DoRegionIteration ( reg_fit,total_iter, flow_key, flow_block_size, flow_block_start );
      IF_OPTIMIZER_DEBUG(bkg.inception_state, bkg.debugSaver.WriteData(reg_fit, bkg.region_data->my_regions.rp ,flow_block_start, bkg.GetRegion(),reg_fit->compNames, nbeads));
      total_iter++;
    }
  }

  // if alternating bead and regional steps
  if ( do_both )
  {
    for ( int loc_iter=0 ; loc_iter<number_region_iterations_wanted ; loc_iter++ )
    {
      // do one well iteration
      bool skip_region = DoAllBeadIteration ( false, well_fit, total_iter, 1, false, flow_key, flow_block_size, flow_block_start );
      total_iter++;
      if ( skip_region )
      {
        total_iter = 2*number_region_iterations_wanted;
        break;
      }
      // do one region iteration
      if ( !skip_region )
      {
        int nbeads = DoRegionIteration ( reg_fit, total_iter, flow_key, flow_block_size, flow_block_start );
        IF_OPTIMIZER_DEBUG(bkg.inception_state, bkg.debugSaver.WriteData(reg_fit, bkg.region_data->my_regions.rp ,flow_block_start, bkg.GetRegion(),reg_fit->compNames, nbeads));
        total_iter++;
      }
    }
  }

  // just bead steps to finish off
  for ( int loc_iter=0 ; loc_iter<additional_bead_only_iterations; loc_iter++ )
  {
    DoAllBeadIteration ( true, well_fit, total_iter, 1, false, flow_key, flow_block_size,
                         flow_block_start );
    total_iter++;
  }
  if ((lm_state.region_success_step<1) & (reg_fit!=NULL)){
    printf("No successes: MultiFlowSpecializedLevMarFitParameters in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  CleanTerminateOptimization();
  return ( total_iter );
}

// If I'm not fitting wells, make it easy to see I'm never going to fit wells
int MultiFlowLevMar::MultiFlowSpecializedLevMarFitParametersOnlyRegion ( 
    int number_region_iterations_wanted,  BkgFitMatrixPacker *reg_fit,
    float lambda_start,int clonal_restriction, int flow_key, int flow_block_size,
    int flow_block_start
    )
{

  EnterTheOptimization ( NULL,reg_fit,lambda_start,clonal_restriction );

  int total_iter=0;
  // just regional parameter updates without any well updates
  if ( reg_fit!=NULL )
  {
    for ( int loc_iter=0 ; loc_iter<number_region_iterations_wanted; loc_iter++ )
    {
      int nbeads = DoRegionIteration ( reg_fit,total_iter, flow_key, flow_block_size, flow_block_start );
      IF_OPTIMIZER_DEBUG(bkg.inception_state, bkg.debugSaver.WriteData(reg_fit, bkg.region_data->my_regions.rp ,flow_block_start, bkg.GetRegion(),reg_fit->compNames, nbeads));
      total_iter++;
    }
  }
  if (lm_state.region_success_step<1){
    printf("No successes: MultiFlowSpecializedLevMarFitParametersOnlyRegion in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  CleanTerminateOptimization();
  return ( total_iter );
}

// This is to finalize the well parameters conditional on the regional parameters
// strong candidate for export to the GPU
void MultiFlowLevMar::MultiFlowSpecializedLevMarFitAllWells ( int bead_only_iterations, BkgFitMatrixPacker *well_fit, float lambda_start,int clonal_restriction, int flow_key, int flow_block_size, int flow_block_start )
{
  EnterTheOptimization ( well_fit,NULL, lambda_start,clonal_restriction );

  if ( well_fit != NULL )
  {
    DoAllBeadIteration ( true, well_fit, /*iter*/ 1, bead_only_iterations, /*isSample*/ false, flow_key, flow_block_size, flow_block_start );
  }

  CleanTerminateOptimization();
}

///-----------------------------------done with entry points
int MultiFlowLevMar::DoSampledRegionIteration (
    BkgFitMatrixPacker *reg_fit,
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{

  reg_params eval_rp;
  SetupAnyIteration ( eval_rp, iter, flow_block_size );
  // do my region iteration
  int reg_wells = LevMarAccumulateRegionDerivsForSampledActiveBeadList ( eval_rp,
                                                                         reg_fit, lm_state.reg_mask,
                                                                         iter, flow_key, flow_block_size, flow_block_start );
  // solve per-region equation and adjust parameters
  if ( reg_wells > lm_state.min_bead_to_fit_region )
  {
    LevMarFitRegion ( reg_fit, flow_key, flow_block_size, flow_block_start );
  } else {
    printf("DoSampledRegionIteration: %d beads less than minimum %d required in region(col=%d,row=%d)\n",reg_wells, lm_state.min_bead_to_fit_region, bkg.region_data->region->col, bkg.region_data->region->row);
  }
  IdentifyParametersFromSample ( bkg.region_data->my_beads,bkg.region_data->my_regions, lm_state.well_mask, lm_state.reg_mask, lm_state , flow_block_size);
  return reg_wells;
}

int MultiFlowLevMar::DoRegionIteration (
    BkgFitMatrixPacker *reg_fit,
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{

  reg_params eval_rp;
  SetupAnyIteration ( eval_rp, iter, flow_block_size );
  // do my region iteration
  int reg_wells = LevMarAccumulateRegionDerivsForActiveBeadList ( eval_rp,
                                                                  reg_fit, lm_state.reg_mask,
                                                                  iter, flow_key, flow_block_size, flow_block_start );
  // solve per-region equation and adjust parameters
  if ( reg_wells > lm_state.min_bead_to_fit_region )
  {
    LevMarFitRegion ( reg_fit, flow_key, flow_block_size, flow_block_start );
    if ( !bkg.global_defaults.signal_process_control.regional_sampling )
      lm_state.IncrementRegionGroup();
  } else {
    int nhighqual=bkg.region_data->my_beads.NumHighQuality();
    printf("DoRegionIteration: %d beads less than minimum %d required, high qual %d, in region group %d  in region(col=%d,row=%d)\n",reg_wells, lm_state.min_bead_to_fit_region, nhighqual, lm_state.current_bead_region_group, bkg.region_data->region->col, bkg.region_data->region->row);
    if ( !bkg.global_defaults.signal_process_control.regional_sampling )
      lm_state.IncrementRegionGroup(); // better try another bead set if this one doesn't work!
  }
  IdentifyParameters ( bkg.region_data->my_beads,bkg.region_data->my_regions, *bkg.region_data_extras.my_flow, flow_block_size, lm_state.well_mask, lm_state.reg_mask,lm_state.skip_beads );
  return reg_wells;
}


bool MultiFlowLevMar::DoSampledBeadIteration (
    bool well_only_fit,
    BkgFitMatrixPacker *well_fit,
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{
  reg_params eval_rp;
  SetupAnyIteration ( eval_rp, iter, flow_block_size );
  float failed_frac = LevMarFitToRegionalActiveBeadList (
        well_only_fit,
        eval_rp,
        well_fit, lm_state.well_mask,
        iter, flow_key, flow_block_size, flow_block_start );
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
    int bead_iterations, bool isSample,
    int flow_key, int flow_block_size, int flow_block_start )
{

  reg_params eval_rp;
  SetupAnyIteration ( eval_rp, iter, flow_block_size );
  float failed_frac = LevMarFitToActiveBeadList (
        well_only_fit,
        eval_rp,
        well_fit, lm_state.well_mask,
        bead_iterations, false, flow_key, flow_block_size, flow_block_start );
  // if more than 1/2 the beads aren't improving any longer, stop trying to do the
  // region-wide fit
  bool skip_region = false;
  if ( !well_only_fit && ( failed_frac > lm_state.bead_failure_rate_to_abort ) )
    skip_region=true;
  return ( skip_region );
}

void MultiFlowLevMar::SetupAnyIteration ( reg_params &eval_rp,  int iter, int flow_block_size )
{
#ifdef FIT_ITERATION_DEBUG_TRACE
  bkg.DebugIterations();
#endif

  lm_state.PhaseInClonalRestriction ( iter,lm_state.nonclonal_call_penalty_enforcement );
  FillTshiftCache ( bkg.region_data->my_regions.rp.tshift, flow_block_size );
  lm_state.reg_error = 0.0f;
  eval_rp = bkg.region_data->my_regions.rp;
  //lm_state.derivative_direction = -lm_state.derivative_direction; // flip derivatives in case we hit a boundary box and get stuck
}

void MultiFlowLevMar::EnterTheOptimization ( BkgFitMatrixPacker *well_fit, BkgFitMatrixPacker *reg_fit, float lambda_start, int clonal_restriction )
{
  lm_state.InitializeLevMarFit ( well_fit,reg_fit );
  lm_state.nonclonal_call_penalty_enforcement = clonal_restriction;
  lev_mar_scratch.ResetXtalkToZero();
  InitTshiftCache();
  bkg.region_data->my_beads.CorruptedBeadsAreLowQuality(); // make sure we're up to date with quality estimates

  lm_state.ReAssignBeadsToRegionGroups(bkg.region_data->my_beads, NUMBEADSPERGROUP); // make sure we don't run out of good quality beads in any subgroup
  lm_state.SetupActiveBeadList ( lambda_start );
  lm_state.derivative_direction = 1;
}

void MultiFlowLevMar::CleanTerminateOptimization()
{
  // @ TODO: can this be cleaned up?
  lm_state.FinalComputeAndSetAverageResidual ( bkg.region_data->my_beads );
  lm_state.restrict_clonal = 0.0f; // we only restrict clonal within this routine
}



void MultiFlowLevMar::DynamicEmphasis ( BeadParams &p, int flow_block_size )
{
  // put together the emphasis needed
  lev_mar_scratch.SetEmphasis ( p.Ampl,bkg.region_data->my_beads.max_emphasis, flow_block_size );
  lev_mar_scratch.CreateEmphasis ( bkg.region_data->emphasis_data.EmphasisVectorByHomopolymer, bkg.region_data->emphasis_data.EmphasisScale, flow_block_size );
}


void MultiFlowLevMar::FillScratchForEval ( BeadParams *p, BeadParams *ref_p, int ref_span, reg_params *reg_p, NucStep &cache_step, int flow_key, int flow_block_size, int flow_block_start )
{
  //  params_IncrementHits(p);
  // evaluate the function
  MathModel::MultiFlowComputeCumulativeIncorporationSignal ( p,reg_p,lev_mar_scratch.ival,
                                                             cache_step,lev_mar_cur_bead_block,bkg.region_data->time_c,*bkg.region_data_extras.my_flow,
                                                             bkg.math_poiss, flow_block_size, flow_block_start );
  MathModel::MultiFlowComputeTraceGivenIncorporationAndBackground ( lev_mar_scratch.fval,
                                                                    p,reg_p,lev_mar_scratch.ival,cache_sbg,bkg.region_data->my_regions,
                                                                    lev_mar_cur_buffer_block,bkg.region_data->time_c,*bkg.region_data_extras.my_flow,
                                                                    use_vectorization, lev_mar_scratch.bead_flow_t, flow_block_size, flow_block_start );

  // add clonal restriction here to penalize non-integer clonal reads
  // this of course does not belong here and should be in the optimizer section of the code
  lm_state.ApplyClonalRestriction ( lev_mar_scratch.fval, p,bkg.region_data->time_c.npts(), flow_key, flow_block_size );
  lm_state.PenaltyForDeviationFromRef(lev_mar_scratch.fval, p, ref_p, ref_span, bkg.region_data->time_c.npts(), flow_block_size );
  lm_state.PenaltyForDeviationFromKmult(lev_mar_scratch.fval,p,bkg.region_data->time_c.npts(),flow_block_size);
  // put together the emphasis needed
  DynamicEmphasis ( *p, flow_block_size );
}

// assumes use of the cached regional derivative steps & nuc_step precomputation
void MultiFlowLevMar::AccumulateRegionDerivForOneBead (
    int ibd, int &reg_wells,
    BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{
  // get the current parameter values for this bead
  BeadParams eval_params = bkg.region_data->my_beads.params_nn[ibd];
  // make custom emphasis vector for this well using pointers to the per-HP vectors
  DynamicEmphasis ( eval_params, flow_block_size );
  lev_mar_scratch.FillObserved ( bkg.region_data->my_trace, eval_params.trace_ndx, flow_block_size );
  // now we're set up, do the individual steps
  SynchRefBead(ibd);
  ComputeCachedPartialDerivatives ( eval_params, &lm_state.ref_bead, lm_state.ref_span, PartialDeriv_mask, flow_key, flow_block_size,
                                    flow_block_start );
  lm_state.residual[ibd] = lev_mar_scratch.CalculateFitError ( NULL, flow_block_size );

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
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{
  int reg_wells = 0;
  // execute on every bead once to get current avg residual
  // @TODO: execute on every bead we care about in this fit???
  lm_state.avg_resid = CalculateCurrentResidualForTestBeads ( flow_key, flow_block_size, flow_block_start );

  BeadParams eval_params; // each bead over-rides this, so we only need for function call
  FillDerivativeStepCache ( eval_params,eval_rp,PartialDeriv_mask, flow_block_size );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;
    AccumulateRegionDerivForOneBead ( ibd, reg_wells, reg_fit, PartialDeriv_mask, iter,
                                      flow_key, flow_block_size, flow_block_start );
  }
  if (reg_wells<1){
    printf("LevMarAccumulateRegionDerivsForSampledActiveBeadList: All beads excluded! in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  return ( reg_wells ); // number of live wells fit for region
}

// reg_proc = TRUE
int MultiFlowLevMar::LevMarAccumulateRegionDerivsForActiveBeadList (
    reg_params &eval_rp,
    BkgFitMatrixPacker *reg_fit, unsigned int PartialDeriv_mask,
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{
  int reg_wells = 0;
  // execute on every bead once to get current avg residual
  // @TODO: execute on every bead we care about in this fit???
  lm_state.avg_resid = CalculateCurrentResidualForTestBeads ( flow_key, flow_block_size, flow_block_start  );

  // Tricky here:  we fill the regional parameter & nuc_step and >apply< them across all beads
  // That way we don't recalculate regional parameters/steps for each bead as we take derivatives
  BeadParams eval_params; // each bead uses own parameters, but we need this for the function call
  FillDerivativeStepCache ( eval_params,eval_rp,PartialDeriv_mask, flow_block_size );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;

    AccumulateRegionDerivForOneBead ( ibd, reg_wells,  reg_fit, PartialDeriv_mask, iter,
                                      flow_key, flow_block_size, flow_block_start );
  }
  if (reg_wells<1){
    printf("LevMarAccumulateRegionDerivsForActiveBeadList: All beads excluded! in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  return ( reg_wells ); // number of live wells fit for region
}


void MultiFlowLevMar::LevMarFitRegion (
    BkgFitMatrixPacker *reg_fit, int flow_key, int flow_block_size, int flow_block_start )
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

    if ( reg_fit->GetOutput ( 0, &eval_rp, lm_state.reg_lambda,lm_state.reg_regularizer ) != LinearSolverException )
    {
      eval_rp.ApplyLowerBound ( &bkg.region_data->my_regions.rp_low, flow_block_size );
      eval_rp.ApplyUpperBound ( &bkg.region_data->my_regions.rp_high, flow_block_size );

      // make a copy so we can modify it to null changes in new_rp that will
      // we push into individual bead parameters
      reg_params new_rp = eval_rp;
      FillTshiftCache ( new_rp.tshift, flow_block_size );

      float new_reg_error = TryNewRegionalParameters ( &new_rp, flow_key, flow_block_size, flow_block_start );
      // are the new parameters better?
      if ( new_reg_error < lm_state.reg_error )
      {
        // it's better...apply the change to all the beads and the region
        // update regional parameters
        lm_state.reg_error =  new_reg_error;
        lm_state.region_success_step++; // succeeded!
        new_rp.reg_error = new_reg_error;  // save reg_error to be dumped to region_param.h5
        bkg.region_data->my_regions.rp = new_rp;

        // re-calculate current parameter values for each bead as necessary
        UpdateBeadParametersFromRegion ( &new_rp, flow_block_size );

        lm_state.ReduceRegionStep();
        cont_proc = true;
      }
      else
      {
        cont_proc = lm_state.IncreaseRegionStep();
        if (cont_proc)
          printf ( "LevMarFitRegion: Reached max lambda %f in region(col=%d,row=%d):\n", lm_state.reg_lambda,bkg.region_data->region->col, bkg.region_data->region->row);
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

  if ( defend_against_infinity>SMALLINFINITY )
    printf ( "RegionLevMar taking a while: %d in region(col=%d,row=%d)\n",defend_against_infinity , bkg.region_data->region->col, bkg.region_data->region->row);
  //return (lm_state.reg_error);
}


float MultiFlowLevMar::CalculateCurrentResidualForTestBeads ( int flow_key, int flow_block_size, int flow_block_start )
{
  float avg_error = 0.0;
  int cnt = 0;
  // use the current region parameter for all beads evaluated
  // don't update nuc step as we don't need to recalculate it
  bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep ( &bkg.region_data->my_regions.rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;

    // set scratch space for this bead
    lev_mar_scratch.FillObserved ( bkg.region_data->my_trace,
                                   bkg.region_data->my_beads.params_nn[ibd].trace_ndx, flow_block_size );
    // update for known reference values for this bead
    lm_state.ref_span = bkg.region_data->my_beads.barcode_info.SetBarcodeFlows(lm_state.ref_bead.Ampl,bkg.region_data->my_beads.barcode_info.barcode_id[ibd]);
    FillScratchForEval ( &bkg.region_data->my_beads.params_nn[ibd], &lm_state.ref_bead, lm_state.ref_span,
                         &bkg.region_data->my_regions.rp, bkg.region_data->my_regions.cache_step,
                         flow_key, flow_block_size, flow_block_start );
    avg_error += lev_mar_scratch.CalculateFitError ( NULL, flow_block_size );
    cnt++;
  }
  bkg.region_data->my_regions.cache_step.Unlock();

  if ( cnt > 0 )
  {
    avg_error = avg_error / cnt;
  }
  else   // in case the semantics need to be traced
  {
    printf("CalculateCurrentResidualForTestBeads: Ran out of beads!  No progress made! in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
    avg_error = std::numeric_limits<float>::quiet_NaN();
  }
  return ( avg_error );
}



float MultiFlowLevMar::TryNewRegionalParameters ( reg_params *new_rp, int flow_key, int flow_block_size,
                                                  int flow_block_start
                                                  )
{
  float new_reg_error = 0.0f;
  //@TODO make this own cache not universal cache override
  bkg.region_data->my_regions.cache_step.ForceLockCalculateNucRiseCoarseStep ( new_rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow );

  // calculate new parameters for everything and re-check residual error
  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ExcludeBead ( ibd ) )
      continue;

    BeadParams eval_params = bkg.region_data->my_beads.params_nn[ibd];
    // apply the region-wide adjustments to each individual well
    UpdateOneBeadFromRegion ( &eval_params,&bkg.region_data->my_beads.params_high, &bkg.region_data->my_beads.params_low,new_rp,bkg.region_data_extras.my_flow->dbl_tap_map, 2.0f, flow_block_size );

    lev_mar_scratch.FillObserved ( bkg.region_data->my_trace, bkg.region_data->my_beads.params_nn[ibd].trace_ndx, flow_block_size );
    lm_state.ref_span = bkg.region_data->my_beads.barcode_info.SetBarcodeFlows(lm_state.ref_bead.Ampl,bkg.region_data->my_beads.barcode_info.barcode_id[ibd]);
    FillScratchForEval ( &eval_params,&lm_state.ref_bead, lm_state.ref_span, new_rp, bkg.region_data->my_regions.cache_step, flow_key, flow_block_size, flow_block_start );

    new_reg_error += lev_mar_scratch.CalculateFitError ( NULL, flow_block_size );
  }
  // undo lock so we can reuse
  bkg.region_data->my_regions.cache_step.Unlock();

  return ( new_reg_error );
}


void MultiFlowLevMar::UpdateBeadParametersFromRegion ( reg_params *new_rp, int flow_block_size )
{
  // update all but ignored beads
  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    // only update sampled beads if sampling(!)
    if (bkg.region_data->my_beads.isSampled){
      if (bkg.region_data->my_beads.Sampled(ibd))
        UpdateOneBeadFromRegion ( &bkg.region_data->my_beads.params_nn[ibd],&bkg.region_data->my_beads.params_high,&bkg.region_data->my_beads.params_low,new_rp, bkg.region_data_extras.my_flow->dbl_tap_map, 2.0f, flow_block_size );

    } else {
      // update everything
      UpdateOneBeadFromRegion ( &bkg.region_data->my_beads.params_nn[ibd],&bkg.region_data->my_beads.params_high,&bkg.region_data->my_beads.params_low,new_rp, bkg.region_data_extras.my_flow->dbl_tap_map, 2.0f, flow_block_size );
    }
  }
}

void MultiFlowLevMar::SynchRefBead(int ibd){
  lm_state.ref_span = bkg.region_data->my_beads.barcode_info.SetBarcodeFlows(lm_state.ref_bead.Ampl,bkg.region_data->my_beads.barcode_info.barcode_id[ibd]);

}

void MultiFlowLevMar::LevMarBuildMatrixForBead ( int ibd,
                                                 bool well_only_fit,
                                                 reg_params &eval_rp, NucStep &cache_step,
                                                 BkgFitMatrixPacker *well_fit,
                                                 unsigned int PartialDeriv_mask,
                                                 int iter, int flow_key, int flow_block_size, int flow_block_start )
{
  // get the current parameter values for this bead
  BeadParams eval_params = bkg.region_data->my_beads.params_nn[ibd];
  // make custom emphasis vector for this well using pointers to the per-HP vectors
  DynamicEmphasis ( eval_params, flow_block_size );
  lev_mar_scratch.FillObserved ( bkg.region_data->my_trace, eval_params.trace_ndx, flow_block_size );
  // now we're set up, do the individual steps
  SynchRefBead(ibd);
  ComputePartialDerivatives ( eval_params, &lm_state.ref_bead, lm_state.ref_span, eval_rp, cache_step, PartialDeriv_mask, flow_key, flow_block_size, flow_block_start );
  lm_state.residual[ibd] = lev_mar_scratch.CalculateFitError ( NULL, flow_block_size );

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
    int bead_iterations, bool isSample, int flow_key, int flow_block_size,
    int flow_block_start )
{

  float req_done = 0.0f; // beads have improved?
  float executed_bead = 0.001f; // did we actually check this bead, or was it skipped for some reason

  step_nuc_cache[0].ForceLockCalculateNucRiseCoarseStep ( &eval_rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow );

  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    if ( ( !bkg.region_data->my_beads.high_quality[ibd] & lm_state.skip_beads ) || lm_state.well_completed[ibd] )  // primarily concerned with regional fits for this iteration, catch up remaining wells later
      continue;

    for ( int iter=0; iter < bead_iterations; ++iter )
    {
      LevMarBuildMatrixForBead ( ibd,  well_only_fit,eval_rp, step_nuc_cache[0], well_fit,PartialDeriv_mask, iter, flow_key, flow_block_size, flow_block_start );
      req_done += LevMarFitOneBead ( ibd, eval_rp,well_fit,well_only_fit, flow_key, flow_block_size,
                                     flow_block_start );
    }
    executed_bead+=1;
  }
  // free up
  step_nuc_cache[0].Unlock();
  if (executed_bead<1){
    printf("LevMarFitToActiveBeadList: Ran out of beads!  No progress made!in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  return ( req_done/executed_bead );
}

//TODO: Refactor to reuse previous function
float MultiFlowLevMar::LevMarFitToRegionalActiveBeadList (

    bool well_only_fit,
    reg_params &eval_rp,
    BkgFitMatrixPacker *well_fit, unsigned int PartialDeriv_mask,
    int iter, int flow_key, int flow_block_size, int flow_block_start )
{

  float req_done = 0.0f; // beads have improved?
  float executed_bead = 0.001f; // did we actually check this bead, or was it skipped for some reason

  step_nuc_cache[0].ForceLockCalculateNucRiseCoarseStep ( &eval_rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow );
  for ( int ibd=0; ibd < lm_state.numLBeads; ibd++ )
  {
    // if this iteration is a region-wide parameter fit, then only process beads
    // in the selection sub-group
    if ( ExcludeBead ( ibd ) )
      continue;


    LevMarBuildMatrixForBead ( ibd,  well_only_fit,eval_rp,step_nuc_cache[0],well_fit,PartialDeriv_mask,iter, flow_key, flow_block_size, flow_block_start );

    executed_bead+=1;
    req_done += LevMarFitOneBead ( ibd, eval_rp,well_fit,well_only_fit, flow_key, flow_block_size, flow_block_start );
  }
  step_nuc_cache[0].Unlock();
  if (executed_bead<1){
    printf("LevMarFitToRegionalActiveBeadList: Ran out of beads!  No progress made!in region(col=%d,row=%d)\n", bkg.region_data->region->col, bkg.region_data->region->row);
  }
  return ( req_done/executed_bead );
}


// the decision logic per iteration is particularly tortured in these routines
// and needs to be revisited
int MultiFlowLevMar::LevMarFitOneBead ( int ibd,
                                        reg_params &eval_rp,
                                        BkgFitMatrixPacker *well_fit,
                                        bool well_only_fit,
                                        int flow_key,
                                        int flow_block_size,
                                        int flow_block_start
                                        )
{
  int bead_not_improved = 0;
  BeadParams eval_params;

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

    if ( well_fit->GetOutput ( &eval_params, 0, lm_state.lambda[ibd] ,lm_state.regularizer[ibd]) != LinearSolverException )
    {
      // bounds check new parameters
      eval_params.ApplyLowerBound ( &bkg.region_data->my_beads.params_low, flow_block_size );
      eval_params.ApplyUpperBound ( &bkg.region_data->my_beads.params_high, flow_block_size );
      eval_params.ApplyAmplitudeZeros ( bkg.region_data_extras.my_flow->dbl_tap_map, flow_block_size ); // double-tap
      eval_params.ApplyAmplitudeDrivenKmultLimit(flow_block_size, bkg.global_defaults.signal_process_control.single_flow_master.krate_adj_limit);

      SynchRefBead(ibd);
      FillScratchForEval ( &eval_params, &lm_state.ref_bead, lm_state.ref_span,&eval_rp, bkg.region_data->my_regions.cache_step, flow_key, flow_block_size, flow_block_start );
      float res = lev_mar_scratch.CalculateFitError ( NULL, flow_block_size );

      if ( res < ( lm_state.residual[ibd] ) )
      {
        achg = bkg.region_data->my_beads.params_nn[ibd].LargestAmplitudeCopiesChange ( &eval_params,flow_block_size );
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
        printf ( "Well singular matrix: %d %f %f in region(col=%d,row=%d)\n", ibd, lm_state.lambda[ibd], lm_state.regularizer[ibd] , bkg.region_data->region->col, bkg.region_data->region->row);
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
  

  if ( defend_against_infinity>SMALLINFINITY )
    printf ( "Problem with bead %d %d in region(col=%d,row=%d)\n", ibd, defend_against_infinity , bkg.region_data->region->col, bkg.region_data->region->row);
  return ( bead_not_improved );
}


// arguably this is part of "scratch space" operations and should be part of that object
// must have "pointed" scratch space at the current bead parameters
void MultiFlowLevMar::ComputePartialDerivatives ( BeadParams &eval_params, BeadParams *ref_p, int ref_span, reg_params &eval_rp, NucStep &cache_step, unsigned int PartialDeriv_mask, int flow_key, int flow_block_size, int flow_block_start )
{
  float *output;
  CpuStep *StepP;

  for ( int step=0;step<BkgFitStructures::NumSteps;step++ )
  {
    StepP = &BkgFitStructures::Steps[step];
    if ( ( StepP->PartialDerivMask & PartialDeriv_mask ) == 0 )
      continue; // only do the steps we are interested in.

    output = lev_mar_scratch.scratchSpace + step*lev_mar_scratch.bead_flow_t;

    ComputeOnePartialDerivative ( output, StepP,  eval_params, ref_p, ref_span, eval_rp, cache_step, flow_key, flow_block_size, flow_block_start );

  }
}

// if I have cached region parameters and steps
void MultiFlowLevMar::ComputeCachedPartialDerivatives ( BeadParams &eval_params,  BeadParams *ref_p, int ref_span, unsigned int PartialDeriv_mask, int flow_key, int flow_block_size, int flow_block_start )
{
  float *output;
  CpuStep *StepP;

  for ( int step=0;step<BkgFitStructures::NumSteps;step++ )
  {
    StepP = &BkgFitStructures::Steps[step];
    if ( ( StepP->PartialDerivMask & PartialDeriv_mask ) == 0 )
      continue; // only do the steps we are interested in.

    output = lev_mar_scratch.scratchSpace + step*lev_mar_scratch.bead_flow_t;

    ComputeOnePartialDerivative ( output, StepP,  eval_params,  ref_p, ref_span, step_rp[step], step_nuc_cache[step], flow_key, flow_block_size, flow_block_start );

  }
}

void MultiFlowLevMar::FillDerivativeStepCache ( BeadParams &eval_params, reg_params &eval_rp, unsigned int PartialDeriv_mask, int flow_block_size )
{
  CpuStep *StepP;
  float backup[lev_mar_scratch.bead_flow_t]; // more than we need

  for ( int step=0;step<BkgFitStructures::NumSteps;step++ )
  {
    StepP = &BkgFitStructures::Steps[step];
    if ( ( StepP->PartialDerivMask & PartialDeriv_mask ) == 0 )
      continue; // only do the steps we are interested in.
    step_rp[step] = eval_rp; // start where we should be
    // do my special step for each thing we're executing
    DoStepDiff ( lm_state.derivative_direction,backup, StepP,&eval_params,&step_rp[step], flow_block_size );
    step_nuc_cache[step].ForceLockCalculateNucRiseCoarseStep ( &step_rp[step],bkg.region_data->time_c,*bkg.region_data_extras.my_flow );

    // because our codebase is somewhat toxic, we cannot effectively tell when we're doing region derivatives or bead derivatives
    // so we'll reset these guys individually each time we take a derivative (crazy)
    DoStepDiff ( 0,backup, StepP,&eval_params,&step_rp[step], flow_block_size ); // reset parameter to default value
  }
}


void MultiFlowLevMar::ComputeOnePartialDerivative (
    float *output,
    CpuStep *StepP,
    BeadParams &eval_params,
    BeadParams *ref_p,
    int ref_span,
    reg_params &eval_rp,
    NucStep &cache_step,
    int flow_key,
    int flow_block_size, int flow_block_start
    )
{
  if ( StepP->PartialDerivMask == DFDGAIN )
  {
    Dfdgain_Step ( output, &eval_params, flow_block_size );
  }
  else if ( StepP->PartialDerivMask == DFDERR )
  {
    Dfderr_Step ( output, &eval_params, flow_block_size );
  }
  else if ( StepP->PartialDerivMask == YERR )
  {
    Dfyerr_Step ( output, flow_block_size );
  }
  else if ( StepP->PartialDerivMask == DFDTSH )
  {
    // check cache if we've evaluated this guy already - should be unnecessary
    //FillTshiftCache(eval_rp.tshift);
    MultiFlowComputePartialDerivOfTimeShift ( output,&eval_params, &eval_rp,cache_slope, flow_block_size, flow_block_start );
  }
  else if ( StepP->PartialDerivMask == FVAL ) // set up the baseline for everything else
  {
    // for safety
    memset ( output, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );
    memset ( lev_mar_scratch.ival, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );

    // fill in the function value & incorporation trace
    MathModel::MultiFlowComputeCumulativeIncorporationSignal ( &eval_params,&eval_rp,lev_mar_scratch.ival,cache_step,lev_mar_cur_bead_block,bkg.region_data->time_c,
                                                               *bkg.region_data_extras.my_flow,bkg.math_poiss, flow_block_size, flow_block_start );
    MathModel::MultiFlowComputeTraceGivenIncorporationAndBackground ( output,&eval_params,&eval_rp,lev_mar_scratch.ival,cache_sbg,bkg.region_data->my_regions,lev_mar_cur_buffer_block,bkg.region_data->time_c,
                                                                      *bkg.region_data_extras.my_flow,use_vectorization, lev_mar_scratch.bead_flow_t, flow_block_size,
                                                                      flow_block_start );
    // add clonal restriction here to penalize non-integer clonal reads
    lm_state.ApplyClonalRestriction ( output, &eval_params,bkg.region_data->time_c.npts(), flow_key, flow_block_size );
    lm_state.PenaltyForDeviationFromRef(output, &eval_params, ref_p, ref_span, bkg.region_data->time_c.npts(), flow_block_size );
    lm_state.PenaltyForDeviationFromKmult(output,&eval_params,bkg.region_data->time_c.npts(),flow_block_size);
  }
  else if ( StepP->diff != 0 )
  {
    float ivtmp[lev_mar_scratch.bead_flow_t];
    float backup[lev_mar_scratch.bead_flow_t]; // more than we need
    DoStepDiff ( lm_state.derivative_direction,backup, StepP,&eval_params,&eval_rp, flow_block_size );
    float *local_iv = lev_mar_scratch.ival;

    if ( StepP->doBoth )
    {
      local_iv = ivtmp;
      //@TODO nuc rise recomputation?
      MathModel::MultiFlowComputeCumulativeIncorporationSignal ( &eval_params,&eval_rp,local_iv,cache_step,lev_mar_cur_bead_block,bkg.region_data->time_c,*bkg.region_data_extras.my_flow,bkg.math_poiss, flow_block_size, flow_block_start );
    }

    MathModel::MultiFlowComputeTraceGivenIncorporationAndBackground ( output,&eval_params,&eval_rp,local_iv,cache_sbg,bkg.region_data->my_regions,lev_mar_cur_buffer_block,bkg.region_data->time_c,*bkg.region_data_extras.my_flow,use_vectorization, lev_mar_scratch.bead_flow_t, flow_block_size, flow_block_start );
    // add clonal restriction here to penalize non-integer clonal reads
    lm_state.ApplyClonalRestriction ( output, &eval_params,bkg.region_data->time_c.npts(), flow_key, flow_block_size );
    lm_state.PenaltyForDeviationFromRef(output, &eval_params, ref_p, ref_span, bkg.region_data->time_c.npts(), flow_block_size );
    lm_state.PenaltyForDeviationFromKmult(output,&eval_params,bkg.region_data->time_c.npts(),flow_block_size);

    CALC_PartialDeriv_W_EMPHASIS_LONG ( lev_mar_scratch.fval,output,lev_mar_scratch.custom_emphasis,lev_mar_scratch.bead_flow_t,lm_state.derivative_direction*StepP->diff );

    DoStepDiff ( 0,backup, StepP,&eval_params,&eval_rp, flow_block_size ); // reset parameter to default value
  }
}


void MultiFlowLevMar::Dfdgain_Step ( float *output,BeadParams *eval_p, int flow_block_size )
{

  // partial w.r.t. gain is the function value divided by the current gain

  float** src = new float *[ flow_block_size ];
  float** dst = new float *[ flow_block_size ];
  float** em = new float *[ flow_block_size ];
  // set up across flows
  for ( int fnum=0;fnum < flow_block_size;fnum++ )
  {
    src[fnum] = &lev_mar_scratch.fval[bkg.region_data->time_c.npts() *fnum];
    dst[fnum] = &output[bkg.region_data->time_c.npts() *fnum];
    em[fnum] = &lev_mar_scratch.custom_emphasis[bkg.region_data->time_c.npts() *fnum];
  }
  // execute in parallel
  if ( use_vectorization )
    Dfdgain_Step_Vec ( flow_block_size, dst, src, em, bkg.region_data->time_c.npts(), eval_p->gain );
  else
    for ( int fnum=0; fnum<flow_block_size; fnum++ )
      for ( int i=0;i < bkg.region_data->time_c.npts();i++ )
        ( dst[fnum] ) [i] = ( src[fnum] ) [i]* ( em[fnum] ) [i]/eval_p->gain;

  // Cleanup.
  delete [] src;
  delete [] em;
  delete [] dst;
}

void MultiFlowLevMar::Dfderr_Step ( float *output, BeadParams *eval_p, int flow_block_size )
{
  // partial w.r.t. darkness is the dark_matter_compensator multiplied by the emphasis

  float** dst = new float*[ flow_block_size ];
  float** et  = new float*[ flow_block_size ];
  float** em  = new float*[ flow_block_size ];
  // set up
  for ( int fnum=0;fnum < flow_block_size;fnum++ )
  {
    dst[fnum] = &output[bkg.region_data->time_c.npts() *fnum];
    em[fnum] = &lev_mar_scratch.custom_emphasis[bkg.region_data->time_c.npts() *fnum];
    et[fnum] = bkg.region_data->my_regions.missing_mass.dark_nuc_comp[bkg.region_data_extras.my_flow->flow_ndx_map[fnum]];
  }
  //execute
  if ( use_vectorization )
    Dfderr_Step_Vec ( flow_block_size, dst, et, em, bkg.region_data->time_c.npts() );
  else
  {
    for ( int fnum=0; fnum<flow_block_size;fnum++ )
    {
      for ( int i=0;i < bkg.region_data->time_c.npts();i++ )
        ( dst[fnum] ) [i] = ( et[fnum] ) [i]* ( em[fnum] ) [i];
    }
  }

  // Cleanup.
  delete [] dst;
  delete [] et;
  delete [] em;
}

//@TODO: this is closely related to fit error, but the goal here is to get y-observed for the lev-mar step
void MultiFlowLevMar::Dfyerr_Step ( float *y_minus_f_emphasized, int flow_block_size )
{

  for ( int fnum=0;fnum < flow_block_size;fnum++ )
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
void MultiFlowLevMar::MultiFlowComputePartialDerivOfTimeShift ( float *fval,struct BeadParams *p, struct reg_params *reg_p, float *neg_sbg_slope, int flow_block_size, int flow_block_start )
{

  int fnum;

  float *vb_out;
  float *flow_deriv;

  // parallel fill one bead parameter for block of flows
  MathModel::FillBufferParamsBlockFlows ( &lev_mar_cur_buffer_block,p,reg_p,bkg.region_data_extras.my_flow->flow_ndx_map, flow_block_start, flow_block_size );

  for ( fnum=0;fnum<flow_block_size;fnum++ )
  {
    vb_out = fval + fnum*bkg.region_data->time_c.npts();  // get ptr to start of the function evaluation for the current flow
    flow_deriv = &neg_sbg_slope[fnum*bkg.region_data->time_c.npts() ];                  // get ptr to pre-shifted slope

    // now this diffeq looks like all the others
    // because linearity of derivatives gets passed through
    // flow_deriv in this case is in fact the local derivative with respect to time of the background step
    // so it can be passed through the equation as though it were a background term
    MathModel::BlueSolveBackgroundTrace ( vb_out,flow_deriv,bkg.region_data->time_c.npts(),&bkg.region_data->time_c.deltaFrame[0],lev_mar_cur_buffer_block.tauB[fnum],lev_mar_cur_buffer_block.etbR[fnum] );
    // isolate gain and emphasis so we can reuse diffeq code
  }
  MultiplyVectorByVector ( fval,lev_mar_scratch.custom_emphasis,lev_mar_scratch.bead_flow_t );
  // gain invariant so can be done to all at once
  MultiplyVectorByScalar ( fval,p->gain,lev_mar_scratch.bead_flow_t );
}

const char* MultiFlowLevMar::findName ( float *ptr )
{
  int i;
  for ( i=0;i<BkgFitStructures::NumSteps;i++ )
  {
    if ( ptr >= BkgFitStructures::Steps[i].ptr && ptr < ( BkgFitStructures::Steps[i].ptr + lev_mar_scratch.bead_flow_t ) )
      return ( BkgFitStructures::Steps[i].name );
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
void MultiFlowLevMar::FillTshiftCache ( float my_tshift, int flow_block_size )
{
  if ( my_tshift != tshift_cache )
  {
    // for safety
    memset ( cache_sbg, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );
    bkg.region_data->emptytrace->GetShiftedBkg ( my_tshift, bkg.region_data->time_c, cache_sbg,
                                                 flow_block_size );
    memset ( cache_slope, 0, sizeof ( float[lev_mar_scratch.bead_flow_t] ) );
    bkg.region_data->emptytrace->GetShiftedSlope( my_tshift, bkg.region_data->time_c, cache_slope,
                                                  flow_block_size );

    tshift_cache = my_tshift;
  }
}

bool MultiFlowLevMar::ExcludeBead ( int ibd )
{
  // test to see whether this bead is excluded in further computation
  // in this fit, restricted to beads of interest
  // regional sampling is handled differently than rolling regional fits

  bool exclude = lm_state.well_completed[ibd];

  if (bkg.region_data->my_beads.isSampled) {
    // regional sampling enabled
    exclude = exclude || !bkg.region_data->my_beads.Sampled(ibd);
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
  // mark this well as finished so it can be excluded
  // from further computation in this fit
  lm_state.FinishCurrentBead ( ibd );
}



void MultiFlowLevMar::DoStepDiff ( int add, float *archive, CpuStep *step, BeadParams *p, 
                                   reg_params *reg_p,
                                   int flow_block_size
                                   )
{
  int i;
  float *dptr = NULL;
  // choose my parameter
  // This is how we call a pointer to a member function in C++.
  if ( step->paramsFunc != NOTBEADPARAM )
  {
    dptr = (p->*(step->paramsFunc))();
  }
  else if ( step->regParamsFunc != NOTREGIONPARAM )
  {
    dptr = (reg_p->*( step->regParamsFunc ))();
  }
  else if ( step->nucShapeFunc != NOTNUCRISEPARAM )
  {
    dptr = (reg_p->nuc_shape.*( step->nucShapeFunc ))();
  }
  // update my parameter
  if ( dptr != NULL )
  {
    // How many items do we need here?
    int length = -1;
    switch( step->length )
    {
      case CpuStep::SpecialCalculation:   length = 0;         break;
      case CpuStep::Singleton:            length = 1;         break;
      case CpuStep::PerNuc:               length = NUMNUC;    break;
      case CpuStep::PerFlow:              length = flow_block_size;  break;
        // No default here; we want the compiler to complain if this enum is changed.
    }
    for ( i=0;i<length;i++ )
    {
      if ( add )
      {
        archive[i] = *dptr;
        *dptr += add*step->diff; // change direction if we go negative or positive
      }
      else
        *dptr = archive[i]; // reset
      dptr++;
    }
  }
}

void UpdateOneBeadFromRegion ( BeadParams *p, bound_params *hi, bound_params *lo, reg_params *new_rp, int *dbl_tap_map, float krate_adj_limit, int flow_block_size )
{
  for ( int i=0;i < flow_block_size;i++ )
    p->Ampl[i] += new_rp->Ampl[i];
  p->R += new_rp->R;
  p->Copies += new_rp->Copies;

  p->ApplyLowerBound ( lo, flow_block_size );
  p->ApplyUpperBound ( hi, flow_block_size );
  p->ApplyAmplitudeZeros ( dbl_tap_map, flow_block_size ); // double-taps
  p->ApplyAmplitudeDrivenKmultLimit(flow_block_size, krate_adj_limit);
}

void IdentifyParametersFromSample ( BeadTracker &my_beads, RegionTracker &my_regions, unsigned int well_mask,
                                    unsigned int reg_mask, const LevMarBeadAssistant& lm_state, int flow_block_size )
{
  if ( ( well_mask & DFDPDM ) >0 )  // only if actually fitting dmult do we need to do this step
    IdentifyDmultFromSample ( my_beads,my_regions, lm_state , flow_block_size);
  if ( ( reg_mask & DFDMR ) >0 ) // only if actually fitting NucMultiplyRatio do I need to identify NMR so we don't slide around randomly
    IdentifyNucMultiplyRatioFromSample ( my_beads, my_regions );
}

void IdentifyDmultFromSample ( BeadTracker &my_beads, RegionTracker &my_regions, const LevMarBeadAssistant& lm_state, int flow_block_size )
{
  float mean_dmult = my_beads.CenterDmultFromSample ();

  for ( int nnuc=0;nnuc < NUMNUC;nnuc++ )
  {
    my_regions.rp.d[nnuc] *= mean_dmult;
  }
  my_regions.rp.ApplyLowerBound(&my_regions.rp_low,flow_block_size);
  my_regions.rp.ApplyUpperBound(&my_regions.rp_high, flow_block_size);
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

void IdentifyParameters ( BeadTracker &my_beads, RegionTracker &my_regions, FlowBufferInfo &my_flow, int flow_block_size, unsigned int well_mask, unsigned int reg_mask, bool skip_beads )
{
  if ( ( well_mask & DFDPDM ) >0 )  // only if actually fitting dmult do we need to do this step
    IdentifyDmult ( my_beads,my_regions,skip_beads , flow_block_size);
  if ( ( well_mask & DFDDKR) > 0 )
    IdentifyKmult(my_beads,my_regions,my_flow, flow_block_size, skip_beads);

  if ( ( reg_mask & DFDMR ) >0 ) // only if actually fitting NucMultiplyRatio do I need to identify NMR so we don't slide around randomly
    IdentifyNucMultiplyRatio ( my_beads, my_regions );
}


void IdentifyKmult(BeadTracker &my_beads, RegionTracker &my_regions, FlowBufferInfo &my_flow, int flow_block_size, bool skip_beads )
{
  // identify kmult to krate to make sure we don't slide around randomly
  float mean_kmult[NUMNUC];

  my_beads.CenterKmult(mean_kmult, skip_beads, my_flow.flow_ndx_map, flow_block_size);
  for (int nnuc=0; nnuc<NUMNUC; nnuc++){
    my_regions.rp.krate[nnuc] *= mean_kmult[nnuc]; //
  }
}


void IdentifyDmult ( BeadTracker &my_beads, RegionTracker &my_regions, bool skip_beads , int flow_block_size)

{
  float mean_dmult = my_beads.CenterDmult ( skip_beads ); // only active set

  for ( int nnuc=0;nnuc < NUMNUC;nnuc++ )
  {
    my_regions.rp.d[nnuc] *= mean_dmult;
  }
  my_regions.rp.ApplyLowerBound(&my_regions.rp_low,flow_block_size);
  my_regions.rp.ApplyUpperBound(&my_regions.rp_high, flow_block_size);
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
