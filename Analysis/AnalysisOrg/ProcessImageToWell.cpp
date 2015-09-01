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

void WriteWashoutFile(BkgFitterTracker &bgFitter, const std::string &dirName) {
  std::string fileName = dirName + "/washouts.txt";
  std::ofstream o(fileName.c_str());
  o << "well\tflow" << std::endl;
  for (size_t i = 0; i < bgFitter.washout_flow.size(); i++) {
    o << i << '\t' << bgFitter.washout_flow[i]  << std::endl;
  }
  o.close();
}

static void DoThreadedSignalProcessing(
    OptArgs &opts, 
    CommandLineOpts &inception_state,
    const ComplexMask &from_beadfind_mask,  
    const char *chipType,
    const ImageSpecClass &my_image_spec, 
    SlicedPrequel &my_prequel_setup,
    const SeqListClass &my_keys, 
    const bool pass_tau,
		const BkgFitterTracker *bkg_fitter_tracker
  )
{
  // This is the main routine, more or less, as far as the background model goes.
  // Our job is to wait for the ImageLoader to load everything necessary,
  // and then apply the background model to flow blocks.

  // So... Let's start with the monitoring tools, for gathering statistics on these runs.
  MemUsage ( "StartingBackground" );
  time_t init_start;
  time ( &init_start );
  bool restart = not inception_state.bkg_control.signal_chunks.restart_from.empty();
  
  const std::string wellsFile = string(inception_state.sys_context.wellsFilePath) + "/" + 
                                       inception_state.sys_context.wellsFileName;

  std::cout << "Wells file name being opened: " << wellsFile << std::endl;
  const FlowBlockSequence & flow_block_sequence = 
    inception_state.bkg_control.signal_chunks.flow_block_sequence;

  // shut off gpu multi flow fit if starting flow is not in first 20
  if ( ! flow_block_sequence.HasFlowInFirstFlowBlock( inception_state.flow_context.startingFlow ) )
  {
    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
  }
 
  Json::Value json_params;
  bool regional_sampling_def = false;
  string sChipType = chipType;
  if(ChipIdDecoder::IsProtonChip())
  {
	regional_sampling_def = true;
  }
  bool regional_sampling = RetrieveParameterBool(opts, json_params, '-', "regional-sampling", regional_sampling_def);
  // shut off gpu multifit if regional sampling is not enabled
  if(!regional_sampling)
  {
    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
  }

  bool convert = RetrieveParameterBool(opts, json_params, '-', "wells-save-as-ushort", true);
  float lower = RetrieveParameterFloat(opts, json_params, '-', "wells-convert-low", -5.0);
  float upper = RetrieveParameterFloat(opts, json_params, '-', "wells-convert-high", 28.0);

  // Make a BkgFitterTracker object, and then either load its state from above,
  // or build it fresh.
  BkgFitterTracker GlobalFitter ( my_prequel_setup.num_regions );
  if( restart ){
    // Get the object that came from serialization.
    GlobalFitter = *bkg_fitter_tracker;
  }
  else {
    GlobalFitter.global_defaults.flow_global.SetFlowOrder ( inception_state.flow_context.flowOrder ); // @TODO: 2nd duplicated code instance
    // Build everything
    json_params["chipType"] = chipType;
    json_params["results_folder"] = inception_state.sys_context.GetResultsFolder();
    GlobalFitter.global_defaults.SetOpts(opts, json_params);	
	// >does not open wells file<
    fprintf(stdout, "Opening wells file %s ... ", wellsFile.c_str());
    RawWells preWells ( inception_state.sys_context.wellsFilePath, inception_state.sys_context.wellsFileName, convert, lower, upper );
    fprintf(stdout, "done\n");
    CreateWellsFileForWriting ( preWells,from_beadfind_mask.my_mask, inception_state, 
                              inception_state.flow_context.GetNumFlows(), 
                              my_image_spec.rows, my_image_spec.cols, chipType );
    // build trace tracking
    GlobalFitter.SetUpTraceTracking ( my_prequel_setup, inception_state, my_image_spec, 
                                      from_beadfind_mask, 
                                      flow_block_sequence.MaxFlowsInAnyFlowBlock() );
    GlobalFitter.AllocateRegionData(my_prequel_setup.region_list.size(), & inception_state );
  }

  // One way (fresh creation) or the other (serialization), we need to allocate scratch space,
  // now that the GlobalFitter has been built.
  GlobalFitter.AllocateSlicedChipScratchSpace( flow_block_sequence.MaxFlowsInAnyFlowBlock() );

  
     // plan (this happens whether we're from-disk or not):
  //GlobalFitter.PlanComputation ( inception_state.bkg_control);
  GlobalFitter.SetUpCpuPipelines(inception_state.bkg_control);
  GlobalFitter.SetUpGpuPipelines(inception_state.bkg_control);
  GlobalFitter.SpinnUpCpuThreads();

  // tweaking global defaults for bkg model if GPU is used
  if (GlobalFitter.useGpuAcceleration()) {
      GlobalFitter.global_defaults.signal_process_control.amp_guess_on_gpu = 
          (inception_state.bkg_control.gpuControl.gpuSingleFlowFit & 
          inception_state.bkg_control.gpuControl.gpuAmpGuess);
  }

  // do we have a wells file?
  ION_ASSERT( isFile(wellsFile.c_str()), "Wells file "+ wellsFile + " does not exist" );

  ChunkyWells rawWells ( inception_state.sys_context.wellsFilePath, 
                         inception_state.sys_context.wellsFileName,
                         inception_state.bkg_control.signal_chunks.save_wells_flow, 
                         inception_state.flow_context.startingFlow, 
                         inception_state.flow_context.endingFlow
                       );

  // Image Loading thread setup to grab flows in the background

  // ImageTracker constructed to load flows
  // must contact the GlobalFitter data that it will be associated with

  // from thin air each time:
  ImageTracker my_img_set ( inception_state.flow_context.getFlowSpan(),
                            inception_state.img_control.ignoreChecksumErrors,
                            inception_state.img_control.doSdat,
                            inception_state.img_control.total_timeout );
  my_img_set.SetUpImageLoaderInfo ( inception_state, from_beadfind_mask, my_image_spec,
                                    flow_block_sequence );
  my_img_set.DecideOnRawDatsToBufferForThisFlowBlock();
  my_img_set.FireUpThreads();

  master_fit_type_table *LevMarSparseMatrices = 0;
  
  // The number of flow blocks here is going to the hdf5 writer,
  // which needs to build structures containing all the flow blocks,
  // not just the ones that might be part of this run.
  int num_flow_blocks = flow_block_sequence.FlowBlockCount( 0, inception_state.flow_context.endingFlow );

  GlobalFitter.ThreadedInitialization ( rawWells, inception_state, from_beadfind_mask, 
                                        inception_state.sys_context.GetResultsFolder(), 
                                        my_image_spec,
                                        my_prequel_setup.smooth_t0_est,
                                        my_prequel_setup.region_list, 
                                        my_prequel_setup.region_timing, my_keys, restart,
                                        num_flow_blocks );

  // need to have initialized the regions for this
  GlobalFitter.SetRegionProcessOrder (inception_state);

  // init trace-output for the --bkg-debug-trace-sse/xyflow/rcflow options
  GlobalFitter.InitBeads_xyflow(inception_state);

  // Get the GPU ready, if we're using it.
  GlobalFitter.DetermineAndSetGPUAllocationAndKernelParams( inception_state.bkg_control, KEY_LEN, flow_block_sequence.MaxFlowsInAnyFlowBlock() );
  GlobalFitter.SpinnUpGpuThreads();
  //GlobalFitter.SpinUpGPUThreads(); //now is done within pipelinesetup

  float washoutThreshold = RetrieveParameterFloat(opts, json_params, '-', "bkg-washout-threshold", WASHOUT_THRESHOLD);
  GlobalFitter.setWashoutThreshold(washoutThreshold);

  int washoutFlowDetection = RetrieveParameterInt(opts, json_params, '-', "bkg-washout-flow-detection", WASHOUT_FLOW_DETECTION);
  GlobalFitter.setWashoutFlowDetection(washoutFlowDetection);

  MemUsage ( "AfterBgInitialization" );
  time_t init_end;
  time ( &init_end );

  fprintf ( stdout, "InitModel: %0.3lf sec.\n", difftime ( init_end,init_start ) );
  
  // JZ start flow data writer thread
  pthread_t flowDataWriterThread;
  SemQueue packQueue;
  SemQueue writeQueue;

  int saveQueueSize;
  if (inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution)
    saveQueueSize = 2;
  else
    saveQueueSize = RetrieveParameterInt(opts, json_params, '-', "wells-save-queue-size", 0);

  writeFlowDataFuncArg writerArg;
  if(saveQueueSize > 0) {
    unsigned int queueSize = (unsigned int)saveQueueSize;
    packQueue.init(queueSize);
    writeQueue.init(queueSize);
    size_t stepSize = rawWells.GetStepSize();
    size_t flowDepth = inception_state.bkg_control.signal_chunks.save_wells_flow;
    unsigned int spaceSize = my_image_spec.rows * my_image_spec.cols;
    unsigned int bufferSize = stepSize * stepSize * flowDepth;
    for(int item = 0; item < saveQueueSize; ++item) {
      ChunkFlowData* chunkData = new ChunkFlowData(spaceSize, flowDepth, bufferSize);
      packQueue.enQueue(chunkData);
    }

    writerArg.filePath = rawWells.GetHdf5FilePath();
    writerArg.numCols = my_image_spec.cols;
    writerArg.stepSize = stepSize;
    writerArg.saveAsUShort = convert;
    writerArg.packQueuePtr = &packQueue;
    writerArg.writeQueuePtr = &writeQueue;

    pthread_create(&flowDataWriterThread, NULL, WriteFlowDataFunc, &writerArg);
  }

  bool defaultSaveCopies = true;
  if(restart) {
	  defaultSaveCopies = false;
  }
  bool saveCopies = RetrieveParameterBool(opts, json_params, '-', "wells-save-number-copies", defaultSaveCopies);
  rawWells.SetSaveCopies(saveCopies);
  bool withCopies = RetrieveParameterBool(opts, json_params, '-', "wells-convert-with-copies", true);
  rawWells.SetConvertWithCopies(withCopies);

  // process all flows...
  // using actual flow values
  //GPUFlowByFlowPipelineInfo flowByFlowInfo;
  //pthread_t flowByFlowHandshakeThread;
  //bool handshakeCreated = false;
  Timer flow_block_timer;
  Timer signal_proc_timer;
  for ( int flow = inception_state.flow_context.startingFlow; 
            flow < (int)inception_state.flow_context.endingFlow; flow++ )
  {
    FlowBlockSequence::const_iterator flow_block = flow_block_sequence.BlockAtFlow( flow );

    if ( flow == flow_block->begin() || flow == inception_state.flow_context.startingFlow ) {
      flow_block_timer.restart();

      // Build some matrices to work with.

      if (!inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution || 
           (inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution && flow < inception_state.bkg_control.gpuControl.switchToFlowByFlowAt)) {
        LevMarSparseMatrices = new master_fit_type_table( GlobalFitter.global_defaults.flow_global, 
                                                        flow_block->begin(), max( KEY_LEN - flow_block->begin() , 0), 
                                                        flow_block->size() );
      }

    }

    // coordinate with the ImageLoader threads for this flow to be read in
    // WaitForFlowToLoad guarantees all flows up this one have been read in
    my_img_set.WaitForFlowToLoad ( flow );

    // ----- handle set up for processing this flow before we do anything needing
    bool last_flow = ( ( flow ) == ( inception_state.flow_context.GetNumFlows()- 1 ) ); // actually the literal >last< flow, not just the flow in a chunk, so we can handle not having a full chunk.

    // done with set up for anything this flow needs   
    signal_proc_timer.restart();

    // computation that modifies data
    // isolate this object so it can carry out actions in any order it chooses.

     if ( ! inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution || flow < inception_state.bkg_control.gpuControl.switchToFlowByFlowAt)
     {
        GlobalFitter.ExecuteFitForFlow ( flow, my_img_set, last_flow,
                                         max( KEY_LEN - flow_block->begin(), 0 ),
                                         LevMarSparseMatrices, & inception_state );
     }
     else {

       if(flow == inception_state.bkg_control.gpuControl.switchToFlowByFlowAt ){
          cout << "SignalProcessing: flow 20 reached, switching from old block of 20 flows to NEW flow by flow GPU pipeline!" <<endl;
          cout << "CUDA: cleaning up GPU pipeline and queuing system used for first 20 flows!" <<endl;
          GlobalFitter.UnSpinGpuThreads();
          cout << "CUDA: initiating flow by flow pipeline" << endl;

          if (inception_state.bkg_control.gpuControl.postFitHandshakeWorker) {
            cout << "Destroying legacy CPU bkgmodel workers after first 20 flows" << endl;
            GlobalFitter.UnSpinCpuThreads();
   
            // start separate GPU-CPU handshake thread if performing flow by flow GPU pipeline
            //setUpFlowByFlowHandshakeWorker(
            //    inception_state,
		// my_image_spec,
		// GlobalFitter,
		//&packQueue,
		//&writeQueue,
		//&rawWells,
		//flowByFlowInfo);

            // initiate thread here
      //      pthread_create(&flowByFlowHandshakeThread, NULL, flowByFlowHandshakeWorker, &flowByFlowInfo);
      //      handshakeCreated = true;
            cout << "Creating new CPU Handshake thread and worker threads for file writing." << endl;
            GlobalFitter.GpuQueueControl.setUpAndStartFlowByFlowHandshakeWorker(  inception_state,
                                                                                  my_image_spec,
                                                                                  &GlobalFitter.signal_proc_fitters,
                                                                                  &packQueue,
                                                                                  &writeQueue,
                                                                                  &rawWells );

          }
       }

      GlobalFitter.ExecuteGPUBlockLevelSignalProcessing(
          flow,
          flow_block->size(),
          my_img_set,
          last_flow,
          max( KEY_LEN - flow_block->begin(), 0 ),
          LevMarSparseMatrices,
          &inception_state,
          &(my_prequel_setup.smooth_t0_est));
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



    if (!GlobalFitter.GpuQueueControl.handshakeCreated()) {

      // Find the flow that's the last runnable flow in the "mixed_last_flow" block.
      int applyFlow = flow_block_sequence.BlockAtFlow(
          inception_state.bkg_control.polyclonal_filter.mixed_last_flow - 1 )->end() - 1;

      if ( flow == applyFlow and inception_state.bkg_control.polyclonal_filter.enable ){
        cout << "************ Applying Clonal Filter at " << applyFlow << endl;
          ApplyClonalFilter ( *from_beadfind_mask.my_mask,
                            inception_state.sys_context.GetResultsFolder(),
                            GlobalFitter.sliced_chip, inception_state.bkg_control.polyclonal_filter );
      }
      if (inception_state.bkg_control.pest_control.bkg_debug_files) {
        GlobalFitter.DumpBkgModelRegionInfo ( inception_state.sys_context.GetResultsFolder(),
                                            flow, last_flow, flow_block );
        GlobalFitter.DumpBkgModelBeadInfo( inception_state.sys_context.GetResultsFolder(),
                                         flow,last_flow, 
                                         inception_state.bkg_control.pest_control.debug_bead_only>0, flow_block );

      }


       //Here I should collect the sample for all the regions and n flows for regional fitting in the new pipeline
      //TODO: ad comandline parameter to turn sampel collection on or off and change number of history flows (currently hard coded to 20
      if(inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution){
        if(flow == (inception_state.bkg_control.gpuControl.switchToFlowByFlowAt -1) ){
          GlobalFitter.CollectSampleWellsForGPUBlockLevelSignalProcessing(
                    flow,
                    flow_block->size(),
                    my_img_set,
                    last_flow,
                    max( KEY_LEN - flow_block->begin(), 0 ),
                    LevMarSparseMatrices,
                    &inception_state,
                    &(my_prequel_setup.smooth_t0_est));
        }
      }



      // hdf5 dump of bead and regional parameters in bkgmodel
      if (inception_state.bkg_control.pest_control.bkg_debug_files)
        GlobalFitter.all_params_hdf.IncrementalWrite ( flow, last_flow, flow_block,
                                                     flow_block_sequence.FlowBlockIndex( flow ) );

      // Needed for 318 chips. Decide how many DATs to read ahead for every flow block
      // also report timing for block of 20 flows from reading dat to writing 1.wells for this block 
      if ( flow == flow_block->end() - 1 )
        my_img_set.DecideOnRawDatsToBufferForThisFlowBlock();

      // End of block cleanup.
      if ( (flow == flow_block->end() - 1 ) || last_flow) {
        // Make sure that we've written everything.
        if(saveQueueSize > 0) {
          rawWells.DoneUpThroughFlow( flow, &packQueue, &writeQueue );
        }
        else {
          rawWells.DoneUpThroughFlow( flow );
        }


        // Cleanup.
        delete LevMarSparseMatrices;
        LevMarSparseMatrices = 0;
      }
    } //post fit steps of bkgmodel in separate thread so as not to slow down GPU thread


    // report timing for block of 20 flows from reading dat to writing 1.wells for this block 
    fprintf ( stdout, "Flow Block compute time for flow %d to %d: %.1f sec.\n",
              flow_block->begin(), flow, flow_block_timer.elapsed());

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    // my_img_set knows what buffer is associated with the absolute flow
    my_img_set.FinishFlow ( flow );
  }


  if(saveCopies)
  {
    rawWells.WriteWellsCopies();
  }
  //if(saveMultiplier)
  //{
    //rawWells.WriteFlowMultiplier();
  //}

  if(saveQueueSize > 0) {
    pthread_join(flowDataWriterThread, NULL);
    packQueue.clear();
    writeQueue.clear();
  }

  // Need a commandline option to decide if we want post fit steps in separate background 
  // thread or not when running new flow by flow pipeline

  if (inception_state.bkg_control.gpuControl.postFitHandshakeWorker) {
    //if ( handshakeCreated) {
    if(GlobalFitter.GpuQueueControl.handshakeCreated()){
      GlobalFitter.GpuQueueControl.joinFlowByFlowHandshakeWorker();
      //  pthread_join(flowByFlowHandshakeThread, NULL);
    }
  }


  if ( not inception_state.bkg_control.signal_chunks.restart_next.empty() ){
    string filePath = inception_state.sys_context.analysisLocation + inception_state.bkg_control.signal_chunks.restart_next;
    ofstream outStream(filePath.c_str(), ios_base::trunc);
    assert(outStream.good());
    //boost::archive::text_oarchive outArchive(outStream);
    boost::archive::binary_oarchive outArchive(outStream);

    // get region associated objects on disk first

    time_t begin_save_time;
    time ( &begin_save_time );

    const ComplexMask *from_beadfind_mask_ptr = &from_beadfind_mask;
    BkgFitterTracker *GlobalFitter_ptr = &GlobalFitter;
    string git_hash = IonVersion::GetGitHash();
    
    outArchive
      << git_hash
      << my_prequel_setup
      << from_beadfind_mask_ptr
      << GlobalFitter_ptr;    
    outStream.close();

    time_t finish_save_time;
    time ( &finish_save_time );
    fprintf ( stdout, "Writing restart state to archive %s took %0.1f secs",
	      filePath.c_str(), difftime ( finish_save_time, begin_save_time ));
  }

  rawWells.Close();
  rawWells.GetWriteTimer().PrintMilliSeconds(std::cout, "Timer: Wells total writing time:");

  GlobalFitter.UnSpinGpuThreads();
  GlobalFitter.UnSpinCpuThreads();

  if ( inception_state.bkg_control.signal_chunks.updateMaskAfterBkgModel )
    from_beadfind_mask.pinnedInFlow->UpdateMaskWithPinned ( from_beadfind_mask.my_mask ); //update maskPtr

  from_beadfind_mask.pinnedInFlow->DumpSummaryPinsPerFlow ( inception_state.sys_context.GetResultsFolder() );
}


// Perform signal processing on a full proton block at once using GPU
/*void PerFlowSignalProcessingOnProtonBlockUsingGpu (
    CommandLineOpts &inception_state,                        
    const ComplexMask &from_beadfind_mask,  
    const char *chipType,
    const ImageSpecClass &my_image_spec, 
    SlicedPrequel &my_prequel_setup,
    const SeqListClass &my_keys, 
    const BkgFitterTracker *bkg_fitter_tracker
)
{
  MemUsage ( "StartingBackground" );
  time_t init_start;
  time ( &init_start );

  bool restart = not inception_state.bkg_control.signal_chunks.restart_from.empty();
  
  const std::string wellsFile = string(inception_state.sys_context.wellsFilePath) + "/" + 
                                       inception_state.sys_context.wellsFileName;

  const FlowBlockSequence & flow_block_sequence = 
    inception_state.bkg_control.signal_chunks.flow_block_sequence;


  // shut off gpu multi flow fit if starting flow is not in first 20
  if ( ! flow_block_sequence.HasFlowInFirstFlowBlock( inception_state.flow_context.startingFlow ) )
  {
    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
  }

  // shut off gpu multifit if regional sampling is not enabled
  if (!inception_state.bkg_control.regional_sampling)
  {
    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
  }


  // One way (fresh creation) or the other (serialization), we need to allocate scratch space,
  // now that the GlobalFitter has been built.
  BkgFitterTracker GlobalFitter(my_prequel_setup.num_regions);

  if( restart ){
    // Get the object that came from serialization.
    GlobalFitter = *bkg_fitter_tracker;
  }
  else {
    GlobalFitter.global_defaults.flow_global.SetFlowOrder ( inception_state.flow_context.flowOrder ); // @TODO: 2nd duplicated code instance
    // Build everything
    SetBkgModelGlobalDefaults ( GlobalFitter.global_defaults, inception_state.bkg_control,chipType,inception_state.sys_context.GetResultsFolder() );
    // >does not open wells file<
    fprintf(stdout, "Opening wells file %s ... ", wellsFile.c_str());
    RawWells preWells ( inception_state.sys_context.wellsFilePath, inception_state.sys_context.wellsFileName );
    fprintf(stdout, "done\n");
    CreateWellsFileForWriting ( preWells,from_beadfind_mask.my_mask, inception_state,
                              inception_state.flow_context.GetNumFlows(),
                              my_image_spec.rows, my_image_spec.cols, chipType );
    // build trace tracking
    GlobalFitter.SetUpTraceTracking ( my_prequel_setup, inception_state, my_image_spec,
                                      from_beadfind_mask,
                                      flow_block_sequence.MaxFlowsInAnyFlowBlock() );
    GlobalFitter.AllocateRegionData(my_prequel_setup.region_list.size());
  }

  GlobalFitter.AllocateSlicedChipScratchSpace( flow_block_sequence.MaxFlowsInAnyFlowBlock() );
  
  // plan (this happens whether we're from-disk or not):
  GlobalFitter.PlanComputation ( inception_state.bkg_control);

  // tweaking global defaults for bkg model if GPU is used
  if (GlobalFitter.IsGpuAccelerationUsed()) {
      GlobalFitter.global_defaults.signal_process_control.amp_guess_on_gpu =
          (inception_state.bkg_control.gpuControl.gpuSingleFlowFit &
          inception_state.bkg_control.gpuControl.gpuAmpGuess);
  }

  // do we have a wells file?
  ION_ASSERT( isFile(wellsFile.c_str()), "Wells file "+ wellsFile + " does not exist" );

  ChunkyWells rawWells ( inception_state.sys_context.wellsFilePath, 
                         inception_state.sys_context.wellsFileName,
                         inception_state.bkg_control.signal_chunks.save_wells_flow, 
                         inception_state.flow_context.startingFlow, 
                         inception_state.flow_context.endingFlow
                       );

  // Image Loading thread setup to grab flows in the background

  // ImageTracker constructed to load flows
  // must contact the GlobalFitter data that it will be associated with

  // from thin air each time:
  ImageTracker my_img_set ( inception_state.flow_context.getFlowSpan(),
                            inception_state.img_control.ignoreChecksumErrors,
                            inception_state.img_control.doSdat,
                            inception_state.img_control.total_timeout );
  my_img_set.SetUpImageLoaderInfo ( inception_state, from_beadfind_mask, my_image_spec,
                                    flow_block_sequence );
  my_img_set.DecideOnRawDatsToBufferForThisFlowBlock();
  my_img_set.FireUpThreads();

  // plan (this happens whether we're from-disk or not):
  int num_flow_blocks = flow_block_sequence.FlowBlockCount( 
                            inception_state.flow_context.startingFlow, 
                            inception_state.flow_context.endingFlow );

  GlobalFitter.ThreadedInitialization(
      rawWells, 
      inception_state, 
      from_beadfind_mask, 
      inception_state.sys_context.GetResultsFolder(), 
      my_image_spec,
      my_prequel_setup.smooth_t0_est,
      my_prequel_setup.region_list, 
      my_prequel_setup.region_timing, 
      my_keys, 
      restart,
      num_flow_blocks);

  // need to have initialized the regions for this
  GlobalFitter.SetRegionProcessOrder (inception_state);

  GlobalFitter.DetermineAndSetGPUAllocationAndKernelParams(
      inception_state.bkg_control, 
      KEY_LEN,
      flow_block_sequence.MaxFlowsInAnyFlowBlock());

  GlobalFitter.SpinUpGPUThreads();
      
  // process all flows...
  // using actual flow values
  master_fit_type_table *LevMarSparseMatrices = 0;
  Timer flow_block_timer;
  Timer signal_proc_timer;
  for ( int flow = inception_state.flow_context.startingFlow; 
            flow < (int)inception_state.flow_context.endingFlow; flow++ )
  {
    FlowBlockSequence::const_iterator flow_block = flow_block_sequence.BlockAtFlow( flow );

    if ( flow == flow_block->begin() || flow == inception_state.flow_context.startingFlow ) {
      flow_block_timer.restart();

      // Build some matrices to work with.
      LevMarSparseMatrices = new master_fit_type_table( GlobalFitter.global_defaults.flow_global, 
                             flow_block->begin(), max( KEY_LEN - flow_block->begin() ), 
                             flow_block->size() );
    }


    // coordinate with the ImageLoader threads for this flow to be read in
    // WaitForFlowToLoad guarantees all flows up this one have been read in
    my_img_set.WaitForFlowToLoad ( flow );

    // ----- handle set up for processing this flow before we do anything needing
    bool last_flow = ( ( flow ) == ( inception_state.flow_context.GetNumFlows()- 1 ) ); // actually the literal >last< flow, not just the flow in a chunk, so we can handle not having a full chunk.

    // done with set up for anything this flow needs   
    signal_proc_timer.restart();

    if(flow < 20){
      cout << "====> Flow " << flow << " using old pipeline! " <<endl;
      GlobalFitter.ExecuteFitForFlow ( flow, my_img_set, last_flow,
                                       max( KEY_LEN - flow_block->begin(), 0 ),
                                       LevMarSparseMatrices, & inception_state );


    }else{
      cout << "====> Flow " << flow << " using NEW pipeline! " <<endl;
    GlobalFitter.ExecuteGPUBlockLevelSignalProcessing(
        flow,
        flow_block->size(), 
        my_img_set,
        last_flow,
        max( KEY_LEN - flow_block->begin(), 0 ),
        LevMarSparseMatrices,
        &inception_state,
        &(my_prequel_setup.smooth_t0_est));
    }

    // Find the flow that's the last runnable flow in the "mixed_last_flow" block. 
    int applyFlow = flow_block_sequence.BlockAtFlow(
        inception_state.bkg_control.polyclonal_filter.mixed_last_flow - 1 )->end() - 1;

    if ( flow == applyFlow and inception_state.bkg_control.polyclonal_filter.enable )
      ApplyClonalFilter ( *from_beadfind_mask.my_mask, 
                          inception_state.sys_context.GetResultsFolder(), 
                          GlobalFitter.sliced_chip, inception_state.bkg_control.polyclonal_filter );

    // no more computation
    signal_proc_timer.elapsed(); 
    printf ( "SigProc: pure compute time for flow %d: %.1f sec.\n", 
             flow, signal_proc_timer.elapsed());


    // Needed for 318 chips. Decide how many DATs to read ahead for every flow block
    // also report timing for block of 20 flows from reading dat to writing 1.wells for this block
    if ( flow == flow_block->end() - 1 )
      my_img_set.DecideOnRawDatsToBufferForThisFlowBlock();

    // End of block cleanup.
    if ( (flow == flow_block->end() - 1 ) || last_flow) {
      // Make sure that we've written everything.
      rawWells.DoneUpThroughFlow( flow );

      // report timing for block of 20 flows from reading dat to writing 1.wells for this block 
      fprintf ( stdout, "Flow Block compute time for flow %d to %d: %.1f sec.\n",
              flow_block->begin(), flow, flow_block_timer.elapsed());

      // Cleanup.
      delete LevMarSparseMatrices;
      LevMarSparseMatrices = 0;
    }

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    // my_img_set knows what buffer is associated with the absolute flow
    my_img_set.FinishFlow ( flow );
  }

  if ( not inception_state.bkg_control.signal_chunks.restart_next.empty() ){
     string filePath = inception_state.sys_context.analysisLocation + inception_state.bkg_control.signal_chunks.restart_next;
     ofstream outStream(filePath.c_str(), ios_base::trunc);
     assert(outStream.good());
     //boost::archive::text_oarchive outArchive(outStream);
     boost::archive::binary_oarchive outArchive(outStream);

     // get region associated objects on disk first

     time_t begin_save_time;
     time ( &begin_save_time );

     const ComplexMask *from_beadfind_mask_ptr = &from_beadfind_mask;
     BkgFitterTracker *GlobalFitter_ptr = &GlobalFitter;
     string svn_rev = IonVersion::GetSvnRev();

     outArchive
       << svn_rev
       << my_prequel_setup
       << from_beadfind_mask_ptr
       << GlobalFitter_ptr;
     outStream.close();

     time_t finish_save_time;
     time ( &finish_save_time );
     fprintf ( stdout, "Writing restart state to archive %s took %0.1f secs",
         filePath.c_str(), difftime ( finish_save_time, begin_save_time ));
   }


  rawWells.Close();
  rawWells.GetWriteTimer().PrintMilliSeconds(std::cout, "Timer: Wells total writing time:");

  if ( inception_state.bkg_control.signal_chunks.updateMaskAfterBkgModel )
    from_beadfind_mask.pinnedInFlow->UpdateMaskWithPinned ( from_beadfind_mask.my_mask ); //update maskPtr

  from_beadfind_mask.pinnedInFlow->DumpSummaryPinsPerFlow ( inception_state.sys_context.GetResultsFolder() );

}
*/


void IsolatedSignalProcessing (
  SlicedPrequel    &my_prequel_setup,
  ImageSpecClass   &my_image_spec,
  Region           &wholeChip,
  OptArgs		   &opts,
  CommandLineOpts  &inception_state,
  string           &analysisLocation,
  SeqListClass     &my_keys,
  TrackProgress    &my_progress,
  ComplexMask      *complex_mask,
  BkgFitterTracker *bkg_fitter_tracker)
{
//@TODO: split function here as we immediately pick up the files

  ComplexMask from_beadfind_mask; //Todo: check if we can move this one level up so it is at the same level as the read in from the archive.
  if(!(inception_state.bkg_control.signal_chunks.restart_from.empty()))
  {
    from_beadfind_mask = *complex_mask;
  }
  else
    {
    // starting execution fresh
    from_beadfind_mask.InitMask();

    my_prequel_setup.LoadBeadFindForSignalProcessing ( true );

	Json::Value json1;
    vector<int> region_list;
	RetrieveParameterVectorInt(opts, json1, '-', "region-list", "", region_list);
	if(!region_list.empty())
	{
		my_prequel_setup.RestrictRegions(region_list);
	}

    if ( inception_state.bfd_control.beadMaskFile != NULL )
    {
      fprintf ( stdout, "overriding beadfind mask with %s\n", inception_state.bfd_control.beadMaskFile );
      from_beadfind_mask.my_mask->LoadMaskAndAbortOnFailure ( inception_state.bfd_control.beadMaskFile );
    }
    else
    {
      from_beadfind_mask.my_mask->LoadMaskAndAbortOnFailure ( my_prequel_setup.bfMaskFile.c_str() );
    }

    from_beadfind_mask.InitPinnedInFlow ( inception_state.flow_context.GetNumFlows() );
  }
  /********************************************************************
   *
   *  Background Modelling Process
   *
   *******************************************************************/
  // set up the directory for a new temporary wells file
  // clear out previous ones (can be bad with re-entrant processing!)
  ClearStaleWellsFile();
  inception_state.sys_context.MakeNewTmpWellsFile ( inception_state.sys_context.GetResultsFolder() );

  // feenableexcept(FE_DIVBYZERO | FE_INVALID); // | FE_OVERFLOW);

  // we might use some other model here
  if ( true )
  {

     const FlowBlockSequence & flow_block_sequence = 
         inception_state.bkg_control.signal_chunks.flow_block_sequence;


     // first 20 flows using traditional pipeline
     // rest of the flows using new GPU pipeline
     // Need to add support for 3-series but its assuemd that 3-series analysis 
     // will happen in one shot for all flows
     	 DoThreadedSignalProcessing ( 
     			opts, 
     			inception_state, 
     			from_beadfind_mask, 
     			ChipIdDecoder::GetChipType(), 
     			my_image_spec,
     			my_prequel_setup,
     			my_keys, 
     			inception_state.mod_control.passTau, 
     			bkg_fitter_tracker);
  } // end BKG_MODL

  // whatever model we run, copy the signals to permanent location
  inception_state.sys_context.CopyTmpWellFileToPermanent (  inception_state.sys_context.GetResultsFolder() );

  inception_state.sys_context.CleanupTmpWellsFile (); // remove local file(!) how will this work with re-entrant processing????

  // mask file may be updated by model processing
  // analysis.bfmask.bin is what BaseCaller expects
  string analysisMaskFile  = analysisLocation + "analysis.bfmask.bin";
  string analysisStatFile  = analysisLocation + "analysis.bfmask.stats";
  from_beadfind_mask.my_mask->UpdateBeadFindOutcomes ( wholeChip, analysisMaskFile.c_str(), !inception_state.bfd_control.SINGLEBF, 0, analysisStatFile.c_str() );

  my_progress.ReportState ( "Raw flowgrams complete" );
}



// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void RealImagesToWells (
  OptArgs &opts,
  CommandLineOpts& inception_state,
  SeqListClass&    my_keys,
  TrackProgress&   my_progress,
  ImageSpecClass&  my_image_spec,
  SlicedPrequel&   my_prequel_setup )
{
  Region            wholeChip(0, 0, my_image_spec.cols, my_image_spec.rows);
  ComplexMask*      complex_mask       = 0;
  BkgFitterTracker* bkg_fitter_tracker = 0;
   
  if(inception_state.bkg_control.signal_chunks.restart_from.empty())
  {
    // do separator if necessary: generate mask
    IsolatedBeadFind ( my_prequel_setup, my_image_spec, wholeChip, inception_state,
		       inception_state.sys_context.GetResultsFolder(), inception_state.sys_context.analysisLocation,  my_keys, my_progress );
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

   // boost::archive::text_iarchive in_archive(ifs);
    boost::archive::binary_iarchive in_archive(ifs);
    string saved_git_hash;
    in_archive >> saved_git_hash
	       >> my_prequel_setup
	       >> complex_mask
	       >> bkg_fitter_tracker;

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
 
  // do signal processing generate wells conditional on a separator
  IsolatedSignalProcessing ( my_prequel_setup, my_image_spec, wholeChip, opts, inception_state,
                             inception_state.sys_context.analysisLocation,  my_keys, my_progress, complex_mask, bkg_fitter_tracker );

  // @TODO cleanup causes crash when restarted, why?
  // if (complex_mask != NULL) delete complex_mask;
  // if (bkg_fitter_tracker != NULL) delete bkg_fitter_tracker;
}


