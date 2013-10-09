/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ProcessImageToWell.h"
#include "BkgFitterTracker.h"
#include "BkgModelHdf5.h"
#include "EmptyTraceTracker.h"
#include "GlobalDefaultsFromBkgControl.h"
#include "Image.h"
#include "ImageTransformer.h"
#include "IonErr.h"
#include "json/json.h"
#include "MaskFunctions.h"

#include "SeparatorInterface.h"
#include "TrackProgress.h"
#include "WellFileManipulation.h"
#include "ClonalFilter.h"
#include "ComplexMask.h"
#include "Serialization.h"

#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <fenv.h>

using namespace std;

void MakeDecisionOnGpuMultiFlowFit(CommandLineOpts &inception_state)
{
  // shut off gpu multi flow fit if starting flow is not in first 20
  if (inception_state.flow_context.startingFlow >= NUMFB)
  {
    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
  }

  // shut off gpu multifit if regional sampling is not enabled
  if (!inception_state.bkg_control.regional_sampling)
  {
    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
  }
}

//@TODO: have I mentioned how much I hate static variables recently?
void TinyInitializeUglyStaticForSignalProcessing ( GlobalDefaultsForBkgModel &global_defaults,CommandLineOpts &inception_state)
{
  int my_nuc_block[NUMFB];
  int flow_start = 0;
  // TODO: Objects should be isolated!!!!
  global_defaults.flow_global.GetFlowOrderBlock ( my_nuc_block,flow_start,flow_start+NUMFB );
  // @TODO these matrices never get updated for later blocks of nucs and should be updated during iterations
  InitializeLevMarSparseMatrices ( my_nuc_block );
}

void TinyDestroyUglyStaticForSignalProcessing()
{
  CleanupLevMarSparseMatrices();
}

void OutputRegionAvg(RegionalizedData *rdata, int reg, int maxWidth, ofstream &meanOut, ofstream &timeOut) {
  for (int flowIx = 0; flowIx < 10; flowIx++) {
    meanOut << reg << "\t" << rdata->region->row << "\t" << rdata->region->col << "\t" << flowIx;
    timeOut << reg << "\t" << rdata->region->row << "\t" << rdata->region->col << "\t" << flowIx;
    vector<double> mean(rdata->time_c.npts(), 0.0);
    int count = 0;
    for (int ibd = 0; ibd < rdata->my_trace.numLBeads; ibd++) {
      FG_BUFFER_TYPE *fgPtr = &rdata->my_trace.fg_buffers[rdata->my_trace.bead_flow_t*ibd+flowIx*rdata->time_c.npts()];
      for (int i = 0; i < rdata->time_c.npts(); i++) {
        mean[i] += fgPtr[i];
      }
      count++;
    }
    count = max(count, 1);
    int eIx = 0;
    int frames = 0;
    for (eIx = 0; eIx < rdata->time_c.npts(); eIx++) {
      frames += rdata->time_c.frames_per_point[eIx];
      meanOut << "\t" << mean[eIx]/count;
      timeOut << "\t" << frames;
    }
    for (; eIx < maxWidth; eIx++) {
      meanOut << "\t0";
      timeOut << "\t0";
    }
    meanOut << endl;
    timeOut << endl;
  }
}

void WriteSampleRegion(const std::string &results_folder, BkgFitterTracker &GlobalFitter, int flow, bool doWrite) {
  if(flow + 1 == NUMFB && doWrite) {
    string s = results_folder + "/vfrc-empties.txt";
    ofstream emptyFile(s.c_str());
    string sm = results_folder + "/vfrc-region-mean.txt";
    ofstream avgFile(sm.c_str());
    sm = results_folder + "/vfrc-region-time.txt";
    ofstream timeFile(sm.c_str());
    s = results_folder + "/vfrc-live.txt";
    ofstream liveFile(s.c_str());
    int reg = GlobalFitter.sliced_chip.size()/2;
    RegionalizedData *rdata = NULL;
    int maxWidth = 0;
    for (size_t i = 0; i < GlobalFitter.sliced_chip.size(); i++) {
      maxWidth = max(GlobalFitter.sliced_chip[i]->time_c.npts(), maxWidth);
    }
    for (size_t i = 0; i < GlobalFitter.sliced_chip.size(); i++) {
      OutputRegionAvg(GlobalFitter.sliced_chip[i],i, maxWidth, avgFile, timeFile);
    }
    avgFile.close();
    timeFile.close();
    rdata = GlobalFitter.sliced_chip[GlobalFitter.sliced_chip.size()/2];
    cout << "Region is: " << reg << "\t" << rdata->region->row << "\t" << rdata->region->col << endl;
    cout << "Tmid nuc is: " << rdata->t_mid_nuc_start << "\t" << rdata->sigma_start << endl;
    float bempty[2056];
    cout << "Tshift is: " << rdata->my_regions.rp.tshift << endl;
    
    rdata->emptytrace->GetShiftedBkg(rdata->my_regions.rp.tshift, rdata->time_c, bempty);
    cout << "Timing frames region 0: ";
    float deltaFrame = 0, deltaSec = 0;
    for (int eIx = 0; eIx < rdata->time_c.npts(); eIx++) {
      deltaFrame += rdata->time_c.deltaFrame[eIx];
      cout << "\t" << deltaFrame;
    }
    cout << endl;
    cout << "Timing seconds region 0: ";
    for (int eIx = 0; eIx < rdata->time_c.npts(); eIx++) {
      deltaSec += rdata->time_c.deltaFrameSeconds[eIx];
      cout << "\t" << deltaSec;
    }
    cout << endl;
    for (int fIx = 0; fIx < 10; fIx++) {
      emptyFile << fIx;
      for (int eIx = 0; eIx < rdata->time_c.npts(); eIx++) {
        emptyFile << '\t' << bempty[fIx * rdata->time_c.npts() + eIx];
      }
      emptyFile << endl;
    }
    for (int ibd = 0; ibd < rdata->my_trace.numLBeads; ibd++) {
      for (size_t flowIx = 0; flowIx < 10; flowIx++) {
        FG_BUFFER_TYPE *fgPtr = &rdata->my_trace.fg_buffers[rdata->my_trace.bead_flow_t*ibd+flowIx*rdata->time_c.npts()];
        int x = rdata->my_beads.params_nn[ibd].x + rdata->region->col;
        int y = rdata->my_beads.params_nn[ibd].y + rdata->region->row;
        liveFile << x << "\t" << y << "\t" << ibd << "\t" << flowIx;
        for (int i = 0; i < rdata->time_c.npts(); i++) {
          liveFile << "\t" << fgPtr[i];
        }
          liveFile << endl;
      }
    }
  }
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

void DoThreadedSignalProcessing ( CommandLineOpts &inception_state, ComplexMask &from_beadfind_mask,  char *chipType,
                                  ImageSpecClass &my_image_spec, SlicedPrequel &my_prequel_setup,SeqListClass &my_keys, bool pass_tau,
				  BkgFitterTracker *bkg_fitter_tracker)
{

  MemUsage ( "StartingBackground" );
  time_t init_start;
  time ( &init_start );

  bool restart = not inception_state.bkg_control.restart_from.empty();
  
  BkgFitterTracker GlobalFitter ( my_prequel_setup.num_regions );
  const std::string wellsFile = string(inception_state.sys_context.wellsFilePath) + "/" + inception_state.sys_context.wellsFileName;

  MakeDecisionOnGpuMultiFlowFit(inception_state);

  if( restart ){
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
    CreateWellsFileForWriting ( preWells,from_beadfind_mask.my_mask, inception_state, NUMFB,
                              inception_state.flow_context.GetNumFlows(), my_image_spec.rows, my_image_spec.cols, chipType );
    // build trace tracking
    GlobalFitter.SetUpTraceTracking ( my_prequel_setup, inception_state, my_image_spec, from_beadfind_mask );
    GlobalFitter.AllocateRegionData(my_prequel_setup.region_list.size());


  }
  
  TinyInitializeUglyStaticForSignalProcessing ( GlobalFitter.global_defaults , inception_state);

  // plan (this happens whether we're from-disk or not):
  GlobalFitter.PlanComputation ( inception_state.bkg_control );

  // tweaking global defaults for bkg model if GPU is used
  if (GlobalFitter.IsGpuAccelerationUsed()) {
      GlobalFitter.global_defaults.signal_process_control.amp_guess_on_gpu = 
          (inception_state.bkg_control.gpuControl.gpuSingleFlowFit & 
          inception_state.bkg_control.gpuControl.gpuAmpGuess);
  }

  // do we have a wells file?
  ION_ASSERT( isFile(wellsFile.c_str()), "Wells file "+ wellsFile + " does not exist" );

  RawWells rawWells ( inception_state.sys_context.wellsFilePath, inception_state.sys_context.wellsFileName );
  // plan (this happens whether we're from-disk or not):
  GlobalFitter.ThreadedInitialization ( GlobalFitter.global_defaults,
                                        rawWells, inception_state, from_beadfind_mask, inception_state.sys_context.GetResultsFolder(), my_image_spec,
					my_prequel_setup.smooth_t0_est,my_prequel_setup.region_list, my_prequel_setup.region_timing, my_keys, restart);

  MemUsage ( "AfterBgInitialization" );
  time_t init_end;
  time ( &init_end );

  fprintf ( stdout, "InitModel: %0.3lf sec.\n", difftime ( init_end,init_start ) );
  
  // Image Loading thread setup to grab flows in the background

  // ImageTracker constructed to load flows
  // must contact the GlobalFitter data that it will be associated with

  // from thin air each time:
  ImageTracker my_img_set ( inception_state.flow_context.getFlowSpan(),inception_state.img_control.ignoreChecksumErrors,inception_state.img_control.doSdat,inception_state.img_control.total_timeout );
  my_img_set.SetUpImageLoaderInfo ( inception_state, from_beadfind_mask, my_image_spec );
  my_img_set.DecideOnRawDatsToBufferForThisFlowBlock();
  my_img_set.FireUpThreads();


  // Now do threaded solving, going through all the flows

  GlobalFitter.SpinUp();
  // need to have initialized the regions for this
  GlobalFitter.SetRegionProcessOrder (inception_state);

  // init h5 for the bestRegion, this could be done in ThreadedInitialization(), and had to wait until setRegionProcessOrder
  GlobalFitter.InitBeads_BestRegion(inception_state);

  // init trace-output for the --bkg-debug-trace-sse/xyflow/rcflow options
  GlobalFitter.InitBeads_xyflow(inception_state);

  // determine maximum beads in a region for gpu memory allocations
  GlobalFitter.DetermineMaxLiveBeadsAndFramesAcrossAllRegionsForGpu();

  // ideally these are part of the rawWells object itself
  int write_well_flow_interval = inception_state.bkg_control.saveWellsFrequency*NUMFB; // goes with rawWells
  int flow_to_write_wells = -1000; // never happens unless we set it to happen
  
  // process all flows...
  // using actual flow values
  Timer flow_block_timer;
  Timer signal_proc_timer;
  SumTimer wellsWriteTimer;
  for ( int flow = inception_state.flow_context.startingFlow; flow < (int)inception_state.flow_context.endingFlow; flow++ )
  {
    if ((flow % NUMFB) == 0)
      flow_block_timer.restart();

    // coordinate with the ImageLoader threads for this flow to be read in
    // WaitForFlowToLoad guarantees all flows up this one have been read in
    my_img_set.WaitForFlowToLoad ( flow );

    // ----- handle set up for processing this flow before we do anything needing
    bool last_flow = ( ( flow ) == ( inception_state.flow_context.GetNumFlows()- 1 ) ); // actually the literal >last< flow, not just the flow in a chunk, so we can handle not having a full chunk.

    // always write intervals starting at wherever we are starting
    // logic here:  open wells file at startingFlow, tell at what flow we need to write things out.
    if (NeedToOpenWellChunk(flow-inception_state.flow_context.startingFlow, write_well_flow_interval))
    {
      // chunk size is flow interval unless we run out of things to do in this interval
      int chunk_depth = FigureChunkDepth(flow,inception_state.flow_context.endingFlow,write_well_flow_interval);
      OpenExistingWellsForOneChunk(rawWells,flow,chunk_depth); // start
      flow_to_write_wells = flow+chunk_depth-1; 
    }
    
    // done with set up for anything this flow needs   
    signal_proc_timer.restart();

    // computation that modifies data
    GlobalFitter.ExecuteFitForFlow ( flow,my_img_set,last_flow ); // isolate this object so it can carry out actions in any order it chooses.
    ApplyClonalFilter ( *from_beadfind_mask.my_mask, inception_state.sys_context.GetResultsFolder(), GlobalFitter.sliced_chip,inception_state.bkg_control.enableBkgModelClonalFilter, flow );

    // no more computation
   
    signal_proc_timer.elapsed(); 
    fprintf ( stdout, "SigProc: pure compute time for flow %d: %.1f sec.\n", flow, signal_proc_timer.elapsed());
    MemUsage ( "Memory_Flow: " + ToStr ( flow ) );
    if (inception_state.bkg_control.bkg_debug_files) {
      GlobalFitter.DumpBkgModelRegionInfo ( inception_state.sys_context.GetResultsFolder(),flow,last_flow );
      GlobalFitter.DumpBkgModelBeadInfo ( inception_state.sys_context.GetResultsFolder(),flow,last_flow, inception_state.bkg_control.debug_bead_only>0 );
    }

    // hdf5 dump of bead and regional parameters in bkgmodel
    if (inception_state.bkg_control.bkg_debug_files)
      GlobalFitter.all_params_hdf.IncrementalWrite (  flow,  last_flow );

    // done capturing parameters, close out this flow

    // logic here: wells file knows when it needs to write something out
    if (flow==flow_to_write_wells)
      WriteOneChunkAndClose(rawWells, wellsWriteTimer);

    // Needed for 318 chips. Decide how many DATs to read ahead for every block of NUMFB flows
    // also report timing for block of 20 flows from reading dat to writing 1.wells for this block 
    if ((flow % NUMFB) == (NUMFB - 1))
      my_img_set.DecideOnRawDatsToBufferForThisFlowBlock();

    // report timing for block of 20 flows from reading dat to writing 1.wells for this block 
    if (((flow % NUMFB) == (NUMFB - 1)) || last_flow)
      fprintf ( stdout, "Flow Block compute time for flow %d to %d: %.1f sec.\n",
              ((flow + 1) - NUMFB), flow, flow_block_timer.elapsed());

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    // my_img_set knows what buffer is associated with the absolute flow
    my_img_set.FinishFlow ( flow );

/*    // stop GPU thread computing doing fitting of first block of flows
    if (flow == (NUMFB - 1))
      GlobalFitter.UnSpinMultiFlowFitGpuThreads();*/
  }
  
  if ( not inception_state.bkg_control.restart_next.empty() ){
    string filePath = inception_state.sys_context.analysisLocation + inception_state.bkg_control.restart_next;
    ofstream outStream(filePath.c_str(), ios_base::trunc);
    assert(outStream.good());
    //boost::archive::text_oarchive outArchive(outStream);
    boost::archive::binary_oarchive outArchive(outStream);

    // get region associated objects on disk first

    time_t begin_save_time;
    time ( &begin_save_time );

    ComplexMask *from_beadfind_mask_ptr = &from_beadfind_mask;
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
  wellsWriteTimer.PrintMilliSeconds(std::cout, "Timer: Wells total writing time:");

  GlobalFitter.UnSpinGpuThreads ();

  TinyDestroyUglyStaticForSignalProcessing();

  if ( inception_state.bkg_control.updateMaskAfterBkgModel )
    from_beadfind_mask.pinnedInFlow->UpdateMaskWithPinned ( from_beadfind_mask.my_mask ); //update maskPtr

  from_beadfind_mask.pinnedInFlow->DumpSummaryPinsPerFlow ( inception_state.sys_context.GetResultsFolder() );
}


void IsolatedSignalProcessing (
  SlicedPrequel    &my_prequel_setup,
  ImageSpecClass   &my_image_spec,
  Region           &wholeChip,
  CommandLineOpts  &inception_state,
  string           &analysisLocation,
  SeqListClass     &my_keys,
  TrackProgress    &my_progress,
  ComplexMask      *complex_mask,
  BkgFitterTracker *bkg_fitter_tracker)
{
//@TODO: split function here as we immediately pick up the files
  ComplexMask from_beadfind_mask;
  if(!(inception_state.bkg_control.restart_from.empty()))
  {
    from_beadfind_mask = *complex_mask;
  }
  else
    {
    // starting execution fresh
    from_beadfind_mask.InitMask();

    my_prequel_setup.LoadBeadFindForSignalProcessing ( true );

    if (!inception_state.bkg_control.region_list.empty() ){
      my_prequel_setup.RestrictRegions(inception_state.bkg_control.region_list);
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
    DoThreadedSignalProcessing ( inception_state, from_beadfind_mask, ChipIdDecoder::GetChipType(), my_image_spec,
                                 my_prequel_setup,my_keys, inception_state.mod_control.passTau, bkg_fitter_tracker);

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
  CommandLineOpts& inception_state,
  SeqListClass&    my_keys,
  TrackProgress&   my_progress,
  ImageSpecClass&  my_image_spec,
  SlicedPrequel&   my_prequel_setup )
{
  Region            wholeChip(0, 0, my_image_spec.cols, my_image_spec.rows);
  ComplexMask*      complex_mask       = 0;
  BkgFitterTracker* bkg_fitter_tracker = 0;
   
  if(inception_state.bkg_control.restart_from.empty())
  {
    // do separator if necessary: generate mask
    IsolatedBeadFind ( my_prequel_setup, my_image_spec, wholeChip, inception_state,
		       inception_state.sys_context.GetResultsFolder(), inception_state.sys_context.analysisLocation,  my_keys, my_progress );
  }

  else
  {
    // restarting execution from known state, restore my_prequel_setup
    // no matter what I do seems like all restoration has to happen in this scope

    string filePath = inception_state.sys_context.analysisLocation + inception_state.bkg_control.restart_from;

    time_t begin_load_time;
    time ( &begin_load_time );

    ifstream ifs(filePath.c_str(), ifstream::in);
    assert(ifs.good());

    //boost::archive::text_iarchive in_archive(ifs);
    boost::archive::binary_iarchive in_archive(ifs);
    string saved_svn_rev;
    in_archive >> saved_svn_rev
	       >> my_prequel_setup
	       >> complex_mask
	       >> bkg_fitter_tracker;

    ifs.close();

    time_t finish_load_time;
    time ( &finish_load_time );
    fprintf ( stdout, "Loading restart state from archive %s took %0.1f sec\n",
	      filePath.c_str(), difftime ( finish_load_time, begin_load_time ));

    if ( inception_state.bkg_control.restart_check ){
      string svn_rev = IonVersion::GetSvnRev();
      ION_ASSERT( (saved_svn_rev.compare(svn_rev) == 0),
		  "This SVN rev " + svn_rev + " of Analysis does not match the SV rev " + saved_svn_rev + " where " + filePath + " was saved; disable this check by using --no-restart-check");
    }

    // set file locations
    my_prequel_setup.FileLocations ( inception_state.sys_context.analysisLocation );
  }

  // do signal processing generate wells conditional on a separator
  IsolatedSignalProcessing ( my_prequel_setup, my_image_spec, wholeChip, inception_state,
                             inception_state.sys_context.analysisLocation,  my_keys, my_progress, complex_mask, bkg_fitter_tracker );

  // @TODO cleanup causes crash when restarted, why?
  // if (complex_mask != NULL) delete complex_mask;
  // if (bkg_fitter_tracker != NULL) delete bkg_fitter_tracker;
}

