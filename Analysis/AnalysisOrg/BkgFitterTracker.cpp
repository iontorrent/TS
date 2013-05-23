/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgFitterTracker.h"
#include "BkgDataPointers.h"
#include "BkgModelHdf5.h"
#include "MaskSample.h"
#include "GpuMultiFlowFitControl.h"

void BkgFitterTracker::SetRegionProcessOrder ()
{

  analysis_compute_plan.region_order.resize (numFitters);
  int numBeads;
  int zeroRegions = 0;
  for (int i=0; i<numFitters; ++i)
  {
    numBeads = sliced_chip[i]->GetNumLiveBeads();
    if (numBeads == 0)
      zeroRegions++;

    analysis_compute_plan.region_order[i] = beadRegion (i, numBeads);
  }
  std::sort (analysis_compute_plan.region_order.begin(), analysis_compute_plan.region_order.end(), sortregionProcessOrderVector);

  int nonZeroRegions = numFitters - zeroRegions;

  printf ("Number of live bead regions: %d\n", nonZeroRegions);
  if (analysis_compute_plan.gpu_work_load != 0)
  {

    int gpuRegions = int (analysis_compute_plan.gpu_work_load * float (nonZeroRegions));
    if (gpuRegions > 0)
      analysis_compute_plan.lastRegionToProcess = gpuRegions;
  }
}


void BkgFitterTracker::UnSpinGpuThreads ()
{

  if (analysis_queue.GetGpuQueue())
  {
    WorkerInfoQueueItem item;
    item.finished = true;
    item.private_data = NULL;
    for (int i=0;i < analysis_compute_plan.numBkgWorkers_gpu;i++)
      analysis_queue.GetGpuQueue()->PutItem (item);
    analysis_queue.GetGpuQueue()->WaitTillDone();

    delete analysis_queue.GetGpuQueue();
    analysis_queue.SetGpuQueue(NULL);

  }
}
/*
void BkgFitterTracker::UnSpinMultiFlowFitGpuThreads ()
{

  if (analysis_queue.GetMultiFitGpuQueue())
  {
    WorkerInfoQueueItem item;
    item.finished = true;
    item.private_data = NULL;
    for (int i=0;i < analysis_compute_plan.numMultiFlowFitGpuWorkers;i++)
      analysis_queue.GetMultiFitGpuQueue()->PutItem (item);
    analysis_queue.GetMultiFitGpuQueue()->WaitTillDone();

    delete analysis_queue.GetMultiFitGpuQueue();
    printf("Deleting multi fit gpu queue\n");
    analysis_queue.SetMultiFitGpuQueue(NULL);
  }
}
*/


BkgFitterTracker::BkgFitterTracker (int numRegions)
{
  numFitters = numRegions;
  //signal_proc_fitters = new SignalProcessingMasterFitter * [numRegions];
  signal_proc_fitters.resize(numRegions);
  sliced_chip.resize(numRegions);
  bkinfo = NULL;
  all_emptytrace_track = NULL;
}

void BkgFitterTracker::AllocateRegionData(std::size_t numRegions)   
{
  sliced_chip.resize(numRegions);
  for(size_t i=0; i<numRegions; ++i)
    sliced_chip[i] = new RegionalizedData;
}

void BkgFitterTracker::DeleteFitters()
{
  for (int r = 0; r < numFitters; r++)
    delete signal_proc_fitters[r];

  signal_proc_fitters.clear();

  for (int r = 0; r < numFitters; r++)
    delete sliced_chip[r];

  sliced_chip.clear();

  delete all_emptytrace_track;
  all_emptytrace_track = NULL;

  if (bkinfo!=NULL)
    delete[] bkinfo;
}

BkgFitterTracker::~BkgFitterTracker()
{
  DeleteFitters();
  numFitters = 0;
}

void BkgFitterTracker::PlanComputation (BkgModelControlOpts &bkg_control)
{
  // how are we dividing the computation amongst resources available as directed by command line constraints

  PlanMyComputation (analysis_compute_plan,bkg_control);

  AllocateProcessorQueue (analysis_queue,analysis_compute_plan,numFitters);
}

void BkgFitterTracker::SetUpTraceTracking(SlicedPrequel &my_prequel_setup, CommandLineOpts &inception_state, ImageSpecClass &my_image_spec, ComplexMask &from_beadfind_mask)
{
  // because it is hypothetically possible that we track empties over a different region mesh than regular beads
  // this is set up beforehand, while deferring the beads to the region mesh used in signal processing
  all_emptytrace_track = new EmptyTraceTracker(my_prequel_setup.region_list, my_prequel_setup.region_timing, my_prequel_setup.smooth_t0_est, inception_state);
  all_emptytrace_track->Allocate(from_beadfind_mask.my_mask, my_image_spec);
  washout_flow.resize(from_beadfind_mask.my_mask->H() * from_beadfind_mask.my_mask->W(), -1);
}

void TrivialDebugGaussExp(string &outFile, std::vector<Region> &regions, std::vector<RegionTiming> &region_timing)
{
    char my_trivial_file[1024];
    sprintf(my_trivial_file, "%s/gauss_exp_sigma_tmid.txt",outFile.c_str());
    ofstream out (my_trivial_file);
  out << "row\tcol\tsigma.ge.est\ttmid.ge.est" << endl;
  unsigned int totalRegions = regions.size();
  for (unsigned int r = 0; r < totalRegions; r++)
  {
    out << regions[r].row << "\t" << regions[r].col << "\t" << region_timing[r].t_sigma << "\t" << region_timing[r].t_mid_nuc << endl;
  }
  out.close();
}

void BkgFitterTracker::InitCacheMath()
{
    // construct the shared math table
  poiss_cache.Allocate (MAX_POISSON_TABLE_COL,MAX_POISSON_TABLE_ROW,POISSON_TABLE_STEP);
  poiss_cache.GenerateValues(); // fill out my table
}

void BkgFitterTracker::ThreadedInitialization (RawWells &rawWells, CommandLineOpts &inception_state, ComplexMask &a_complex_mask, char *results_folder,
    ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, 
					       std::vector<Region> &regions,
					       std::vector<RegionTiming> &region_timing,SeqListClass &my_keys,
					       bool restart)
{
  // a debugging file if needed
  if (inception_state.bkg_control.bkg_debug_files) {
    all_params_hdf.Init ( inception_state.sys_context.results_folder,
                        inception_state.loc_context,  
                        my_image_spec, inception_state.flow_context.GetNumFlows(),
                        inception_state.bkg_control.bkgModelHdf5Debug );
  }
  
  int totalRegions = regions.size();
  // designate a set of reads that will be processed regardless of whether they pass filters
  set<int> randomLibSet;
  MaskSample<int> randomLib (*a_complex_mask.my_mask, MaskLib, inception_state.bkg_control.unfiltered_library_random_sample);
  randomLibSet.insert (randomLib.Sample().begin(), randomLib.Sample().end());

  InitCacheMath();
    
  if (inception_state.bkg_control.bkg_debug_files)
    TrivialDebugGaussExp(inception_state.sys_context.analysisLocation, regions,region_timing);

  ImageInitBkgWorkInfo *linfo = new ImageInitBkgWorkInfo[numFitters];
  for (int r = 0; r < numFitters; r++)
  {
    // load up the entire image, and do standard image processing (maybe all this 'standard' stuff could be on one call?)
    linfo[r].type = imageInitBkgModel;
    linfo[r].r = r;
    // data holders and fitters
    linfo[r].signal_proc_fitters = &signal_proc_fitters[0];
    linfo[r].sliced_chip = &sliced_chip[0];
    linfo[r].emptyTraceTracker = all_emptytrace_track;

    // context same for all regions 
    linfo[r].inception_state = &inception_state;  // why do global variables plague me so
    linfo[r].rows = my_image_spec.rows;
    linfo[r].cols = my_image_spec.cols;
    linfo[r].maxFrames = inception_state.img_control.maxFrames;
    linfo[r].uncompFrames = my_image_spec.uncompFrames;
    linfo[r].timestamps = my_image_spec.timestamps;
    
    linfo[r].nokey = inception_state.bkg_control.nokey;
    linfo[r].seqList = my_keys.seqList;
    linfo[r].numSeqListItems= my_keys.numSeqListItems;

    // global state across chip
    linfo[r].maskPtr = a_complex_mask.my_mask;
    linfo[r].pinnedInFlow = a_complex_mask.pinnedInFlow;
    linfo[r].global_defaults = &global_defaults;
    linfo[r].washout_flow = &washout_flow[0];
    // prequel data
    linfo[r].regions = &regions[0];
    linfo[r].numRegions = (int)totalRegions;
    //linfo[r].kic = keyIncorporation;
    linfo[r].t_mid_nuc = region_timing[r].t_mid_nuc;
    linfo[r].t0_frame = region_timing[r].t0_frame;
    linfo[r].t_sigma = region_timing[r].t_sigma;
    linfo[r].sep_t0_estimate = &smooth_t0_est;
    
    // fix tauB,tauE passing later
    linfo[r].math_poiss = &poiss_cache;
    linfo[r].sample = &randomLibSet;

    // i/o state
    linfo[r].results_folder = results_folder;
    linfo[r].rawWells = &rawWells;

    if (inception_state.bkg_control.bkgModelHdf5Debug)
    {
      linfo[r].ptrs = &all_params_hdf.ptrs;
    }
    else
    {
      linfo[r].ptrs = NULL;
    }
    linfo[r].restart = restart;


    // now put me on the queue
    analysis_queue.item.finished = false;
    analysis_queue.item.private_data = (void *) &linfo[r];
    analysis_queue.GetCpuQueue()->PutItem (analysis_queue.item);

  }
  // wait for all of the images to be loaded and initially processed
  analysis_queue.GetCpuQueue()->WaitTillDone();

  delete[] linfo;

  // set up for flow-by-flow fitting
  bkinfo = new BkgModelWorkInfo[numFitters];

}


// call the fitters for each region
void BkgFitterTracker::ExecuteFitForFlow (int flow, ImageTracker &my_img_set, bool last)
{

  int flow_buffer_for_flow = my_img_set.FlowBufferFromFlow(flow);
  for (int r = 0; r < numFitters; r++)
  {
    // these get free'd by the thread that processes them
    bkinfo[r].type = MULTI_FLOW_REGIONAL_FIT;
    bkinfo[r].bkgObj = signal_proc_fitters[analysis_compute_plan.region_order[r].first];
    bkinfo[r].flow = flow;
    bkinfo[r].sdat = NULL;
    bkinfo[r].img = NULL;
    bkinfo[r].doingSdat = my_img_set.doingSdat;
    if (bkinfo[r].doingSdat)
    {
      bkinfo[r].sdat = & (my_img_set.sdat[flow_buffer_for_flow]);
    }
    else
    {
      bkinfo[r].img = & (my_img_set.img[flow_buffer_for_flow]);
    }
    bkinfo[r].last = last;

    bkinfo[r].pq = &analysis_queue;
    analysis_queue.item.finished = false;
    analysis_queue.item.private_data = (void *) &bkinfo[r];
    AssignQueueForItem (analysis_queue,analysis_compute_plan);
  }

  WaitForRegionsToFinishProcessing (analysis_queue,analysis_compute_plan);
}

void BkgFitterTracker::SpinUp()
{
  SpinUpGPUThreads (analysis_queue,analysis_compute_plan);
}


void BkgFitterTracker::DumpBkgModelBeadParams (char *results_folder,  int flow, bool debug_bead_only)
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", results_folder, "BkgModelBeadData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);

  DumpBeadTitle (bkg_mod_bead_dbg);

  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->DumpExemplarBead (bkg_mod_bead_dbg,debug_bead_only);
  }
  fclose (bkg_mod_bead_dbg);
}

void BkgFitterTracker::DumpBkgModelBeadOffset (char *results_folder, int flow, bool debug_bead_only)
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", results_folder, "BkgModelBeadDcData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);



  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->DumpExemplarBeadDcOffset (bkg_mod_bead_dbg,debug_bead_only);
  }
  fclose (bkg_mod_bead_dbg);
}


void BkgFitterTracker::DumpBkgModelBeadInfo (char *results_folder,  int flow, bool last_flow, bool debug_bead_only)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if (CheckFlowForWrite (flow,last_flow))
  {
    DumpBkgModelBeadParams (results_folder, flow, debug_bead_only);
    DumpBkgModelBeadOffset (results_folder,  flow, debug_bead_only);
  }
}

void BkgFitterTracker::DumpBkgModelEmphasisTiming (char *results_folder, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_time_dbg = NULL;
  char *bkg_mod_time_name = (char *) malloc (512);
  snprintf (bkg_mod_time_name, 512, "%s/%s.%04d.%s", results_folder, "BkgModelEmphasisData",flow+1,"txt");
  fopen_s (&bkg_mod_time_dbg, bkg_mod_time_name, "wt");
  free (bkg_mod_time_name);

  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->DumpTimeAndEmphasisByRegion (bkg_mod_time_dbg);
  }
  fclose (bkg_mod_time_dbg);
}


void BkgFitterTracker::DumpBkgModelInitVals (char *results_folder, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_init_dbg = NULL;
  char *bkg_mod_init_name = (char *) malloc (512);
  snprintf (bkg_mod_init_name, 512, "%s/%s.%04d.%s", results_folder, "BkgModelInitVals",flow+1,"txt");
  fopen_s (&bkg_mod_init_dbg, bkg_mod_init_name, "wt");
  free (bkg_mod_init_name);

  for (int r = 0; r < numFitters; r++)
  {
    sliced_chip[r]->DumpInitValues (bkg_mod_init_dbg);

  }
  fclose (bkg_mod_init_dbg);
}

void BkgFitterTracker::DumpBkgModelDarkMatter (char *results_folder, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_dark_dbg = NULL;
  char *bkg_mod_dark_name = (char *) malloc (512);
  snprintf (bkg_mod_dark_name, 512, "%s/%s.%04d.%s", results_folder, "BkgModelDarkMatterData",flow+1,"txt");
  fopen_s (&bkg_mod_dark_dbg, bkg_mod_dark_name, "wt");
  free (bkg_mod_dark_name);

  signal_proc_fitters[0]->DumpDarkMatterTitle (bkg_mod_dark_dbg);

  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->DumpDarkMatter (bkg_mod_dark_dbg);

  }
  fclose (bkg_mod_dark_dbg);
}

void BkgFitterTracker::DumpBkgModelEmptyTrace (char *results_folder, int flow)
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_mt_dbg = NULL;
  char *bkg_mod_mt_name = (char *) malloc (512);
  snprintf (bkg_mod_mt_name, 512, "%s/%s.%04d.%s", results_folder, "BkgModelEmptyTraceData",flow+1,"txt");
  fopen_s (&bkg_mod_mt_dbg, bkg_mod_mt_name, "wt");
  free (bkg_mod_mt_name);

  for (int r = 0; r < numFitters; r++)
  {
    sliced_chip[r]->DumpEmptyTrace (bkg_mod_mt_dbg);
  }
  fclose (bkg_mod_mt_dbg);
}

void BkgFitterTracker::DumpBkgModelRegionParameters (char *results_folder,int flow)
{
  FILE *bkg_mod_reg_dbg = NULL;
  char *bkg_mod_reg_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_reg_dbg_fname, 512, "%s/%s.%04d.%s", results_folder, "BkgModelRegionData",flow+1,"txt");
  fopen_s (&bkg_mod_reg_dbg, bkg_mod_reg_dbg_fname, "wt");
  free (bkg_mod_reg_dbg_fname);

  struct reg_params rp;

  DumpRegionParamsTitle (bkg_mod_reg_dbg);

  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->GetRegParams (rp);
    //@TODO this routine should have no knowledge of internal representation of variables
    // Make this a routine to dump an informative line to a selected file from a regional parameter structure/class
    // especially as we use a very similar dumping line a lot in different places.
    // note: t0, rdr, and pdr evolve over time.  It would be nice to also capture how they changed throughout the analysis
    // this only captures the final value of each.
    DumpRegionParamsLine (bkg_mod_reg_dbg, signal_proc_fitters[r]->GetRegion()->row,signal_proc_fitters[r]->GetRegion()->col, rp);

  }
  fclose (bkg_mod_reg_dbg);
}

void BkgFitterTracker::DumpBkgModelRegionInfo (char *results_folder, int flow, bool last_flow)
{
  // get some regional data for the entire chip as debug
  // only do this every 20 flows as this is the block
  // should be triggered by bkgmodel
  if (CheckFlowForWrite (flow,last_flow))
  {
    DumpBkgModelRegionParameters (results_folder, flow);
    DumpBkgModelDarkMatter (results_folder,  flow);
    DumpBkgModelEmphasisTiming (results_folder, flow);
    DumpBkgModelEmptyTrace (results_folder,flow);
    DumpBkgModelInitVals (results_folder, flow);
  }
}


void BkgFitterTracker::DetermineMaxLiveBeadsAndFramesAcrossAllRegionsForGpu()
{
  int maxBeads = 0;
  int maxFrames = 0;
  for (int i=0; i<numFitters; ++i) {
    maxBeads = maxBeads < sliced_chip[i]->GetNumLiveBeads() ? 
                    sliced_chip[i]->GetNumLiveBeads() : maxBeads;
    maxFrames = maxFrames < sliced_chip[i]->GetNumFrames() ? 
                    sliced_chip[i]->GetNumFrames() : maxFrames;
  }
  
  if (maxBeads)
    GpuMultiFlowFitControl::SetMaxBeads(maxBeads);
 
  // it is always 0 right now since frames are actually set after the first flow is read in
  //if (maxFrames)
  //  GpuMultiFlowFitControl::SetMaxFrames(maxFrames);
}

