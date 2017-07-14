/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ProcessImageToWell.h"
#include "BkgFitterTracker.h"
#include "BkgModelHdf5.h"
#include "EmptyTraceTracker.h"
#include "Image.h"
#include "ImageTransformer.h"
#include "IonErr.h"
#include "json/json.h"
#include "MaskFunctions.h"
#include "FlowSequence.h"

#include "SeparatorInterface.h"
#include "TrackProgress.h"
#include "WellFileManipulation.h"
#include "ClonalFilter.h"
#include "ComplexMask.h"
#include "Serialization.h"
#include "GpuMultiFlowFitControl.h"
#include "ChipIdDecoder.h"

#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <fenv.h>

#define DUMP_NEW_PIPELINE_REG_FITTING 0

using namespace std;


// Need to be aware whether 1.wells writing in background or not
// Need to sync after writing a flow

/*
 static void setUpFlowByFlowHandshakeWorker(
  const CommandLineOpts &inception_state,
  const ImageSpecClass &my_image_spec,
  BkgFitterTracker &fitterTracker,
  SemQueue *packQueue,
  SemQueue *writeQueue,
  ChunkyWells *rawWells,
  GPUFlowByFlowPipelineInfo &info)
{
  fitterTracker.CreateRingBuffer(20, my_image_spec.rows * my_image_spec.cols);

  // set up information for the thread here
  info.ampEstimatesBuf = fitterTracker.getRingBuffer();
  info.fitters = &(fitterTracker.signal_proc_fitters);
  info.startingFlow = inception_state.bkg_control.gpuControl.switchToFlowByFlowAt;
  info.endingFlow = (int)inception_state.flow_context.endingFlow;
  info.packQueue = packQueue;
  info.writeQueue = writeQueue;
  info.rawWells = rawWells;
}
 */

static bool CheckForMinimumNumberOfFlows(const CommandLineOpts &inception_state) {


  int startingFlow = inception_state.flow_context.startingFlow;
  if (startingFlow == 0) {

    const FlowBlockSequence &flow_block_sequence =
        inception_state.bkg_control.signal_chunks.flow_block_sequence;
    FlowBlockSequence::const_iterator flow_block = flow_block_sequence.BlockAtFlow( startingFlow );

    size_t minFlows = inception_state.flow_context.endingFlow - startingFlow + 1;
    if (minFlows < flow_block->size())
      return false;
  }

  return true;
}



//////////////////////////////////////////////////////////////////////////////
//ImageToWells class methods:



ImageToWells::ImageToWells(
    OptArgs &refOpts,
    CommandLineOpts &refInception_state,
    Json::Value &refJson_params, //get rid of this in the future and set Opts from the outside
    SeqListClass &refMy_keys,
    TrackProgress &refMy_progress,
    ImageSpecClass &refMy_image_spec,
    SlicedPrequel &refMy_prequel_setup)
:
            opts(refOpts),
            inception_state(refInception_state),
            my_keys(refMy_keys),
            my_progress(refMy_progress),
            my_image_spec(refMy_image_spec),
            my_prequel_setup(refMy_prequel_setup),
            WholeChip(0, 0, refMy_image_spec.cols, refMy_image_spec.rows),
            GlobalFitter(my_prequel_setup.num_regions)
{

  ptrMyImgSet = NULL;
  ptrRawWells=NULL;
  ptrWriteFlowData = NULL;

  GetRestartData(refJson_params);
  InitMask();
  SetupDirectoriesAndFiles();
  SetUpThreadedSignalProcessing(refJson_params);
}



ImageToWells::~ImageToWells(){

  if(ptrMyImgSet != NULL)
    delete ptrMyImgSet;

  if(ptrRawWells !=NULL )
    delete ptrRawWells;

}


void ImageToWells::SetUpThreadedSignalProcessing( Json::Value &json_params){

  // So... Let's start with the monitoring tools, for gathering statistics on these runs.
  MemUsage ( "StartingBackground" );
  time_t init_start;
  time ( &init_start );

  InitGlobalFitter(json_params);
  CreateAndInitRawWells();
  CreateAndInitImageTracker();
  AllocateAndStartUpGlobalFitter();
  InitFlowDataWriter();

  MemUsage ( "AfterBgInitialization" );
  time_t init_end;
  time ( &init_end );
  fprintf ( stdout, "InitModel: %0.3lf sec.\n", difftime ( init_end,init_start ) );

}


void ImageToWells::DoThreadedSignalProcessing()
{
  // This is the main routine, more or less, as far as the background model goes.
  // Our job is to wait for the ImageLoader to load everything necessary,
  // and then apply the background model to flow blocks.

  //ToDo: still too much redundant code between those two functions. some more cleanup should be performed.
  if( GlobalFitter.GpuQueueControl.useFlowByFlowExecution())
  {
    ExecuteFlowByFlowSignalProcessing(); //flow by flow execution after first 20
  }else{
    ExecuteFlowBlockSignalProcessing(); //current pipeline with blokc of 20 flow executoon for whole experiment
  }
}


void ImageToWells::FinalizeThreadedSignalProcessing(){

  // saving reg params of all regions in a json file which can be read back in
  // after restart for subsequent flows
  GlobalFitter.SaveRegParamsJson(inception_state.bkg_control.restartRegParamsFile);

  ptrRawWells->WriteWellsCopies();
  // Need a commandline option to decide if we want post fit steps in separate background
  // thread or not when running new flow by flow pipeline

  DestroyFlowDataWriter();

  if (inception_state.bkg_control.gpuControl.postFitHandshakeWorker) {
    //if ( handshakeCreated) {
    if(GlobalFitter.GpuQueueControl.handshakeCreated()){
      GlobalFitter.GpuQueueControl.joinFlowByFlowHandshakeWorker();
      //  pthread_join(flowByFlowHandshakeThread, NULL);
    }
  }

  StoreRestartData();

  ptrRawWells->Close();
  ptrRawWells->GetWriteTimer().PrintMilliSeconds(std::cout, "Timer: Wells total writing time:");

  GlobalFitter.UnSpinGpuThreads();
  GlobalFitter.UnSpinCpuThreads();


  if ( inception_state.bkg_control.signal_chunks.updateMaskAfterBkgModel )
    FromBeadfindMask.pinnedInFlow->UpdateMaskWithPinned ( FromBeadfindMask.my_mask ); //update maskPtr

  FromBeadfindMask.pinnedInFlow->DumpSummaryPinsPerFlow ( inception_state.sys_context.GetResultsFolder() );

  MoveWellsFileAndFinalize();

}







void ImageToWells::GetRestartData(Json::Value &json_params){

  if(!isRestart())
  {
    // do separator if necessary: generate mask
    string chipType = GetParamsString(json_params, "chipType", "");
    // do separator if necessary: generate mask
    IsolatedBeadFind ( my_prequel_setup, my_image_spec, WholeChip, inception_state,
        inception_state.sys_context.GetResultsFolder(), inception_state.sys_context.analysisLocation,  my_keys, my_progress, chipType );

  }
  else
  {
    // restarting execution from known state, restore my_prequel_setup
    // no matter what I do seems like all restoration has to happen in this scope

    string filePath = inception_state.sys_context.analysisLocation + inception_state.bkg_control.signal_chunks.restart_from;

    time_t begin_load_time;
    time ( &begin_load_time );

    ifstream ifs(filePath.c_str(), ifstream::in);
    assert(ifs.good());

    //boost::archive::text_iarchive in_archive(ifs);
    boost::archive::binary_iarchive in_archive(ifs);

    string saved_git_hash;
    in_archive >> saved_git_hash
    >> my_prequel_setup
    >> FromBeadfindMask
    >> GlobalFitter;

    ifs.close();

    time_t finish_load_time;
    time ( &finish_load_time );
    fprintf ( stdout, "Loading restart state from archive %s took %0.1f sec\n",
        filePath.c_str(), difftime ( finish_load_time, begin_load_time ));

    if ( inception_state.bkg_control.signal_chunks.restart_check ){
      string git_hash = IonVersion::GetGitHash();
      ION_ASSERT( (saved_git_hash.compare(git_hash) == 0),
          "This GIT hash " + git_hash + " of Analysis does not match the GIT hash " + saved_git_hash + " where " + filePath + " was saved; disable this check by using --restart-check false");
    }

    // set file locations
    my_prequel_setup.FileLocations ( inception_state.sys_context.analysisLocation );
  }

  // Check if we have enoguh flows to perform signal processing.
  // If we are starting from flow 0 we should have at least number of flows
  // equalling flow block size (currently 20) to perform regional paramter
  // fitting in background model and derive any actional information for the
  // next set of flows. If we dont have thes minimum number of flows then give
  // an appropriate message and error out
  if (!CheckForMinimumNumberOfFlows(inception_state)) {
    std::cout << "ERROR: Insufficient number of flows to start signal processing. "
        << "Minimum number of flows required is 20." << std::endl;
    exit(EXIT_FAILURE);
  }
}


void ImageToWells::StoreRestartData(){

  if ( doSerialize() ){
    string filePath = inception_state.sys_context.analysisLocation + inception_state.bkg_control.signal_chunks.restart_next;
    ofstream outStream(filePath.c_str(), ios_base::trunc);
    assert(outStream.good());
    //boost::archive::text_oarchive outArchive(outStream);
    boost::archive::binary_oarchive outArchive(outStream);


    // get region associated objects on disk first

    time_t begin_save_time;
    time ( &begin_save_time );

    //const ComplexMask *from_beadfind_mask_ptr = &from_beadfind_mask;
    //BkgFitterTracker *GlobalFitter_ptr = &GlobalFitter;
    string git_hash = IonVersion::GetGitHash();

    GlobalFitter.GpuQueueControl.mirrorDeviceBuffersToHostForSerialization();

    outArchive
    << git_hash
    << my_prequel_setup
    << FromBeadfindMask
    << GlobalFitter;
    outStream.close();

    time_t finish_save_time;
    time ( &finish_save_time );
    fprintf ( stdout, "Writing restart state to archive %s took %0.1f secs",
        filePath.c_str(), difftime ( finish_save_time, begin_save_time ));
  }

}


void ImageToWells::InitMask()
{

  //string analysisLocation;

  if(!isRestart())
  {
    // starting execution fresh
    FromBeadfindMask.InitMask();

    my_prequel_setup.LoadBeadFindForSignalProcessing ( true );

    if(!inception_state.sys_context.region_list.empty())
    {
      my_prequel_setup.RestrictRegions(inception_state.sys_context.region_list);
    }

    if ( inception_state.bfd_control.beadMaskFile != NULL )
    {
      fprintf ( stdout, "overriding beadfind mask with %s\n", inception_state.bfd_control.beadMaskFile );
      FromBeadfindMask.my_mask->LoadMaskAndAbortOnFailure ( inception_state.bfd_control.beadMaskFile );
    }
    else
    {
      FromBeadfindMask.my_mask->LoadMaskAndAbortOnFailure ( my_prequel_setup.bfMaskFile.c_str() );
    }

    FromBeadfindMask.InitPinnedInFlow ( inception_state.flow_context.GetNumFlows() );
  }
}


void ImageToWells::SetupDirectoriesAndFiles()
{
  // set up the directory for a new temporary wells file
  // clear out previous ones (can be bad with re-entrant processing!)
  ClearStaleWellsFile();
  inception_state.sys_context.MakeNewTmpWellsFile ( inception_state.sys_context.GetResultsFolder() );
}


void ImageToWells::InitGlobalFitter( Json::Value &json_params)
{


  std::cout << "Wells file name being opened: " << getWellsFile() << std::endl;

  // Make a BkgFitterTracker object, and then either load its state from above,
  // or build it fresh.
  if( !isRestart()){
    // Get the object that came from serialization.
    // GlobalFitter.LoadRegParamsFromJson(inception_state.bkg_control.restartRegParamsFile);
    //}
    //else {
    GlobalFitter.global_defaults.flow_global.SetFlowOrder ( inception_state.flow_context.flowOrder ); // @TODO: 2nd duplicated code instance

    GlobalFitter.global_defaults.SetOpts(opts, json_params);

    // >creates but does not open wells file<
    fprintf(stdout, "Opening wells file %s ... ", getWellsFile().c_str());
    RawWells preWells ( inception_state.sys_context.wellsFilePath, inception_state.sys_context.wellsFileName, inception_state.sys_context.well_convert, inception_state.sys_context.well_lower, inception_state.sys_context.well_upper );
    fprintf(stdout, "done\n");

    string chipType = GetParamsString(json_params, "chipType", "");

    CreateWellsFileForWriting ( preWells,FromBeadfindMask.my_mask, inception_state,
        inception_state.flow_context.GetNumFlows(),
        my_image_spec.rows, my_image_spec.cols, chipType.c_str() );

    // build trace tracking
    GlobalFitter.SetUpTraceTracking ( my_prequel_setup, inception_state, my_image_spec,
        FromBeadfindMask,
        getFlowBlockSeq().MaxFlowsInAnyFlowBlock() );
    GlobalFitter.AllocateRegionData(my_prequel_setup.region_list.size(), & inception_state );
  }
}





void ImageToWells::AllocateAndStartUpGlobalFitter()
{
  // One way (fresh creation) or the other (serialization), we need to allocate scratch space,
  // now that the GlobalFitter has been built.

  //(inception_state,my_image_spec,my_prequel_setup,my_keys, *ptrRawWells,FromBeadfindMask,getFlowBlockSeq(),isRestart());
  GlobalFitter.AllocateSlicedChipScratchSpace( getFlowBlockSeq().MaxFlowsInAnyFlowBlock() );
  GlobalFitter.UpdateAndCheckGPUCommandlineOptions(inception_state);
  GlobalFitter.SetUpCpuAndGpuPipelines(inception_state.bkg_control);
  GlobalFitter.UpdateGPUPipelineExecutionConfiguration(inception_state);
  //GlobalFitter.SetUpCpuPipeline(inception_state.bkg_control);
  //GlobalFitter.SetUpGpuPipeline(inception_state.bkg_control, getFlowBlockSeq().HasFlowInFirstFlowBlock( inception_state.flow_context.startingFlow ));
  GlobalFitter.SpinnUpCpuThreads();



  // The number of flow blocks here is going to the hdf5 writer,
  // which needs to build structures containing all the flow blocks,
  // not just the ones that might be part of this run.
  int num_flow_blocks = getFlowBlockSeq().FlowBlockCount( 0, inception_state.flow_context.endingFlow );

  GlobalFitter.ThreadedInitialization ( *ptrRawWells, inception_state, FromBeadfindMask,
      inception_state.sys_context.GetResultsFolder(),
      my_image_spec,
      my_prequel_setup.smooth_t0_est,
      my_prequel_setup.region_list,
      my_prequel_setup.region_timing, my_keys, isRestart(),
      num_flow_blocks );

  GlobalFitter.LoadRegParamsFromJson(inception_state.bkg_control.restartRegParamsFile);

  // need to have initialized the regions for this
  GlobalFitter.SetRegionProcessOrder (inception_state);

  // init trace-output for the --bkg-debug-trace-sse/xyflow/rcflow options
  GlobalFitter.InitBeads_xyflow(inception_state);

  // Get the GPU ready, if we're using it.
  GlobalFitter.DetermineAndSetGPUAllocationAndKernelParams( inception_state.bkg_control, KEY_LEN, getFlowBlockSeq().MaxFlowsInAnyFlowBlock() );
  GlobalFitter.SpinnUpGpuThreads();
  //GlobalFitter.SpinUpGPUThreads(); //now is done within pipelinesetup

  GlobalFitter.setWashoutThreshold(inception_state.bkg_control.washout_threshold);
  GlobalFitter.setWashoutFlowDetection(inception_state.bkg_control.washout_flow_detection);

}

void ImageToWells::CreateAndInitRawWells()
{

  // do we have a wells file?
  ION_ASSERT( isFile(getWellsFile().c_str()), "Wells file "+ getWellsFile() + " does not exist" );
  ptrRawWells = new ChunkyWells( inception_state.sys_context.wellsFilePath,
      inception_state.sys_context.wellsFileName,
      inception_state.bkg_control.signal_chunks.save_wells_flow,
      inception_state.flow_context.startingFlow,
      inception_state.flow_context.endingFlow
  );


  bool saveCopies = inception_state.sys_context.wells_save_number_copies;
  if(isRestart()) {
    saveCopies = false;
  }
  ptrRawWells->SetSaveCopies(saveCopies);
  ptrRawWells->SetConvertWithCopies(inception_state.sys_context.wells_convert_with_copies);

}

void ImageToWells::CreateAndInitImageTracker()
{

  // Image Loading thread setup to grab flows in the background

  // ImageTracker constructed to load flows
  // must contact the GlobalFitter data that it will be associated with

  // from thin air each time:
  ptrMyImgSet = new ImageTracker ( inception_state.flow_context.getFlowSpan(),
      inception_state.img_control.ignoreChecksumErrors,
      inception_state.img_control.total_timeout );

  ptrMyImgSet->SetUpImageLoaderInfo ( inception_state, FromBeadfindMask, my_image_spec, getFlowBlockSeq() );

  ptrMyImgSet->DecideOnRawDatsToBufferForThisFlowBlock();

  ptrMyImgSet->FireUpThreads();


}


void ImageToWells::InitFlowDataWriter()
{
  // JZ start flow data writer thread

  unsigned int saveQueueSize;

  if (GlobalFitter.GpuQueueControl.useFlowByFlowExecution())
    saveQueueSize = 2;
  else
    saveQueueSize = (unsigned int) inception_state.sys_context.wells_save_queue_size;

  if(saveQueueSize > 0) {

    ptrWriteFlowData = new WriteFlowDataClass(saveQueueSize,inception_state, my_image_spec,*ptrRawWells);
    ptrWriteFlowData->start();
    //todo: error handling
  }

}


void ImageToWells::DestroyFlowDataWriter()
{
  if(ptrWriteFlowData != NULL){
    ptrWriteFlowData->join();
    delete ptrWriteFlowData;
  }
}




void ImageToWells::MoveWellsFileAndFinalize()
{
  // whatever model we run, copy the signals to permanent location
  inception_state.sys_context.CopyTmpWellFileToPermanent (  inception_state.sys_context.GetResultsFolder() );

  inception_state.sys_context.CleanupTmpWellsFile (); // remove local file(!) how will this work with re-entrant processing????

  // check for DataCollect messages in explog_final.txt and mask affected regions as ignore
  // WARNING: masking entire block asas ignore, cannot be used on thumbnails
  // WARNING: waiting for explog_final.txt if enabled and if there is not restart_next file
  if (inception_state.img_control.mask_datacollect_exclude_regions and inception_state.bkg_control.signal_chunks.restart_next.empty()){
    inception_state.sys_context.WaitForExpLogFinalPath();
    // Mask all wells as ignore if includes DataCollect exclude regions
    if (inception_state.sys_context.CheckDatacollectExcludeRegions(inception_state.loc_context.chip_offset_x, 
								   inception_state.loc_context.chip_offset_x + inception_state.loc_context.cols - 1, 
								   inception_state.loc_context.chip_len_x,
								   inception_state.loc_context.chip_offset_y, 
								   inception_state.loc_context.chip_offset_y + inception_state.loc_context.rows - 1, 
								   inception_state.loc_context.chip_len_y) ){
      printf("Masking DataCollect exclude regions as Ignore\n");
      FromBeadfindMask.my_mask->SetAll((MaskType)(MaskIgnore));
    }
  }

  // mask file may be updated by model processing
  // analysis.bfmask.bin is what BaseCaller expects
  string analysisMaskFile  = inception_state.sys_context.analysisLocation + "analysis.bfmask.bin";
  string analysisStatFile  = inception_state.sys_context.analysisLocation + "analysis.bfmask.stats";
  FromBeadfindMask.my_mask->UpdateBeadFindOutcomes ( WholeChip, analysisMaskFile.c_str(), !inception_state.bfd_control.SINGLEBF, 0, analysisStatFile.c_str() );
  my_progress.ReportState ( "Raw flowgrams complete" );

}

void ImageToWells::DoClonalFilter(int flow)
{
  // Find the flow that's the last runnable flow in the "mixed_last_flow" block.
  int applyFlow = getFlowBlockSeq().BlockAtFlow( inception_state.bkg_control.polyclonal_filter.mixed_last_flow - 1 )->end() - 1;

  if ( flow == applyFlow && inception_state.bkg_control.polyclonal_filter.enable ){

    cout << "Applying Clonal Filter at " << applyFlow << endl;
    ApplyClonalFilter ( *FromBeadfindMask.my_mask,
        inception_state.sys_context.GetResultsFolder(),
        GlobalFitter.sliced_chip, inception_state.bkg_control.polyclonal_filter );
  }
}



//////////////////////////////////////////////////////
//actual Background Model loop where all the work is performed.
//this one does the basic block of n flow execution for all flows of the experiment
void ImageToWells::ExecuteFlowBlockSignalProcessing(){

  Timer flow_block_timer;
  Timer signal_proc_timer;
  master_fit_type_table *LevMarSparseMatrices = NULL;

  for ( int flow = inception_state.flow_context.startingFlow;
      flow < (int)inception_state.flow_context.endingFlow; flow++ )
  {

    FlowBlockSequence::const_iterator flow_block = getFlowBlockSeq().BlockAtFlow( flow );
    if ( flow == flow_block->begin() || flow == inception_state.flow_context.startingFlow ) {
      flow_block_timer.restart();

      // Build some matrices to work with. Only needed if not running flow by flow pipeline
      LevMarSparseMatrices = new master_fit_type_table( GlobalFitter.global_defaults.flow_global,
          flow_block->begin(), max( KEY_LEN - flow_block->begin() , 0),
          flow_block->size() );

    }

    // coordinate with the ImageLoader threads for this flow to be read in
    // WaitForFlowToLoad guarantees all flows up this one have been read in
    ptrMyImgSet->WaitForFlowToLoad ( flow );

    // ----- handle set up for processing this flow before we do anything needing
    bool last_flow = isLastFlow(flow); // actually the literal >last< flow, not just the flow in a chunk, so we can handle not having a full chunk.

    // done with set up for anything this flow needs
    signal_proc_timer.restart();
    // computation that modifies data

    GlobalFitter.ExecuteFitForFlow ( flow, *ptrMyImgSet, last_flow,
        max( KEY_LEN - flow_block->begin(), 0 ),
        LevMarSparseMatrices, & inception_state );

    // no more computation
    signal_proc_timer.elapsed();
    printf ( "SigProc: pure compute time for flow %d: %.1f sec.\n",
        flow, signal_proc_timer.elapsed());
    MemUsage ( "Memory_Flow: " + ToStr ( flow ) );


    DoClonalFilter(flow);


    if (inception_state.bkg_control.pest_control.bkg_debug_files) {
      GlobalFitter.DumpBkgModelRegionInfo ( inception_state.sys_context.GetResultsFolder(),
          flow, last_flow, flow_block );
      GlobalFitter.DumpBkgModelBeadInfo( inception_state.sys_context.GetResultsFolder(),
          flow,last_flow,
          inception_state.bkg_control.pest_control.debug_bead_only>0, flow_block );
    }

    // hdf5 dump of bead and regional parameters in bkgmodel
    if (inception_state.bkg_control.pest_control.bkg_debug_files)
      GlobalFitter.all_params_hdf.IncrementalWrite ( flow, last_flow, flow_block,
          getFlowBlockSeq().FlowBlockIndex( flow ) );

    // Needed for 318 chips. Decide how many DATs to read ahead for every flow block
    // also report timing for block of 20 flows from reading dat to writing 1.wells for this block
    if ( flow == flow_block->end() - 1 )
      ptrMyImgSet->DecideOnRawDatsToBufferForThisFlowBlock();

    // End of block cleanup.
    if ( (flow == flow_block->end() - 1 ) || last_flow) {
      // Make sure that we've written everything.
      if(useFlowDataWriter()) {
        ptrRawWells->DoneUpThroughFlow( flow, ptrWriteFlowData->GetPackQueue(), ptrWriteFlowData->GetWriteQueue());
      }
      else {
        ptrRawWells->DoneUpThroughFlow( flow );
      }

      // Cleanup.
      delete LevMarSparseMatrices;
      LevMarSparseMatrices = 0;
    }

    // report timing for block of 20 flows from reading dat to writing 1.wells for this block
    fprintf ( stdout, "Flow Block compute time for flow %d to %d: %.1f sec.\n",
        flow_block->begin(), flow, flow_block_timer.elapsed());

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    // my_img_set knows what buffer is associated with the absolute flow
    ptrMyImgSet->FinishFlow ( flow );

  } // end flow loop

}


//////////////////////////////////////////////////////
//actual Background Model loop where all the work is performed.
//this one does the fist block of flows in the original fashion and then switches to flow by flow execution
void ImageToWells::ExecuteFlowByFlowSignalProcessing()
{
  // process all flows...
  // using actual flow values
  Timer flow_block_timer;
  Timer signal_proc_timer;

  master_fit_type_table *LevMarSparseMatrices = NULL;

  for ( int flow = inception_state.flow_context.startingFlow;
      flow < (int)inception_state.flow_context.endingFlow; flow++ )
  {

    FlowBlockSequence::const_iterator flow_block = getFlowBlockSeq().BlockAtFlow( flow );
    if ( flow == flow_block->begin() || flow == inception_state.flow_context.startingFlow ) {
      flow_block_timer.restart();
      // Build some matrices to work with. Only needed for first 20 flows when not running flow by flow pipeline
      if ( ! GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
      {
        LevMarSparseMatrices = new master_fit_type_table( GlobalFitter.global_defaults.flow_global,
            flow_block->begin(), max( KEY_LEN - flow_block->begin() , 0),
            flow_block->size() );
      }
    }

    // coordinate with the ImageLoader threads for this flow to be read in
    // WaitForFlowToLoad guarantees all flows up this one have been read in
    ptrMyImgSet->WaitForFlowToLoad ( flow );

    // ----- handle set up for processing this flow before we do anything needing
    bool last_flow = isLastFlow(flow); // actually the literal >last< flow, not just the flow in a chunk, so we can handle not having a full chunk.

    // done with set up for anything this flow needs
    signal_proc_timer.restart();

    // computation that modifies data
    // isolate this object so it can carry out actions in any order it chooses.
    if(GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
    {// flow by flow execution after first block of 20

      GlobalFitter.checkAndInitGPUPipelineSwitch(inception_state,
          my_image_spec,
          &(ptrWriteFlowData->GetPackQueue()),
          &(ptrWriteFlowData->GetWriteQueue()),
          ptrRawWells,
          flow,
          isRestart() );

      GlobalFitter.ExecuteGPUFlowByFlowSignalProcessing(
          flow,
          *ptrMyImgSet,
          last_flow,
          max( KEY_LEN - flow_block->begin(), 0 ),
          &inception_state,
          &(my_prequel_setup.smooth_t0_est));
    }
    else
    {
      GlobalFitter.ExecuteFitForFlow ( flow, *ptrMyImgSet, last_flow,
          max( KEY_LEN - flow_block->begin(), 0 ),
          LevMarSparseMatrices, & inception_state );
    }

#if DUMP_NEW_PIPELINE_REG_FITTING
    if (flow >= 20) {
      for (size_t i=0; i<GlobalFitter.signal_proc_fitters.size(); ++i) {
        std::cout << "flow:" << flow << ",";
        std::cout << "regCol:" << GlobalFitter.signal_proc_fitters[i]->region_data->region->col << ",";
        std::cout << "regRow:" << GlobalFitter.signal_proc_fitters[i]->region_data->region->row << ",";
        std::cout << "RegId:" << GlobalFitter.signal_proc_fitters[i]->region_data->region->index << ",";
        std::cout << "tmidNuc:" <<*(GlobalFitter.signal_proc_fitters[i]->region_data->my_regions.rp.AccessTMidNuc()) << ",";
        std::cout << "rdr:" <<*(GlobalFitter.signal_proc_fitters[i]->region_data->my_regions.rp.AccessRatioDrift()) << ",";
        std::cout << "pdr:" <<*(GlobalFitter.signal_proc_fitters[i]->region_data->my_regions.rp.AccessCopyDrift()) << ",";
        std::cout << std::endl;
      }
    }
#endif

    // no more computation
    signal_proc_timer.elapsed();
    printf ( "SigProc: pure compute time for flow %d: %.1f sec.\n",
        flow, signal_proc_timer.elapsed());
    MemUsage ( "Memory_Flow: " + ToStr ( flow ) );


    //In the last flow before switching to FlowByFlow we collect the samples for all the regions and n flows for regional fitting
    if(inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution){
      if(flow == (inception_state.bkg_control.gpuControl.switchToFlowByFlowAt -1) ){
        GlobalFitter.CollectSampleWellsForGPUFlowByFlowSignalProcessing(
            flow,
            flow_block->size(),
            *ptrMyImgSet,
            last_flow,
            max( KEY_LEN - flow_block->begin(), 0 ),
            &inception_state,
            &(my_prequel_setup.smooth_t0_est));
      }
    }


    //all this only needs to be done before we switched to flow by flow and if we did not create the handshake
    if (!GlobalFitter.GpuQueueControl.handshakeCreated() && ! GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
    {

      // Find the flow that's the last runnable flow in the "mixed_last_flow" block.
      DoClonalFilter(flow);
      //TODO: make dump of region info and bead info work with flow by flow pipeline
      if (inception_state.bkg_control.pest_control.bkg_debug_files) {
        GlobalFitter.DumpBkgModelRegionInfo ( inception_state.sys_context.GetResultsFolder(),
            flow, last_flow, flow_block );
        GlobalFitter.DumpBkgModelBeadInfo( inception_state.sys_context.GetResultsFolder(),
            flow,last_flow,
            inception_state.bkg_control.pest_control.debug_bead_only>0, flow_block );

      }
    }

    // hdf5 dump of bead and regional parameters in bkgmodel
    if (inception_state.bkg_control.pest_control.bkg_debug_files)
      GlobalFitter.all_params_hdf.IncrementalWrite ( flow, last_flow, flow_block,
          getFlowBlockSeq().FlowBlockIndex( flow ) );

    // Needed for 318 chips. Decide how many DATs to read ahead for every flow block
    // also report timing for block of 20 flows from reading dat to writing 1.wells for this block
    if ( flow == flow_block->end() - 1 )
      ptrMyImgSet->DecideOnRawDatsToBufferForThisFlowBlock();


    //all this only needs to be done before we switched to flow by flow
    if ( ! GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
    {
      // End of block cleanup.
      if ( (flow == flow_block->end() - 1 ) || last_flow) {
        // Make sure that we've written everything.
        if(useFlowDataWriter()) {
          ptrRawWells->DoneUpThroughFlow( flow, ptrWriteFlowData->GetPackQueue(), ptrWriteFlowData->GetWriteQueue());
        }
        else {
          ptrRawWells->DoneUpThroughFlow( flow );
        }


        // Cleanup.

        delete LevMarSparseMatrices;
        LevMarSparseMatrices = NULL;
      }

    } //post fit steps of bkgmodel in separate thread so as not to slow down GPU thread


    // report timing for block of 20 flows from reading dat to writing 1.wells for this block
    fprintf ( stdout, "Flow Block compute time for flow %d to %d: %.1f sec.\n",
        flow_block->begin(), flow, flow_block_timer.elapsed());

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    // my_img_set knows what buffer is associated with the absolute flow
    ptrMyImgSet->FinishFlow ( flow );
  }
}


bool ImageToWells::useFlowDataWriter()
{
  if (ptrWriteFlowData == NULL) return false; return (ptrWriteFlowData->getQueueSize() > 0);
}



