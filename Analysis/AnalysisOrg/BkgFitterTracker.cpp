/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgFitterTracker.h"
#include "BkgDataPointers.h"
#include "BkgModelHdf5.h"
#include "MaskSample.h"
#include "GpuMultiFlowFitControl.h"
#include "FlowSequence.h"


int BkgFitterTracker::findRegion(int row, int col)
{
	int reg = -1;
	for(int i=0; i<numFitters; ++i)
	{
		const Region *rp = sliced_chip[i]->get_region();
		int c = rp->col;
		int r = rp->row;
		int w = rp->w;
		int h = rp->h;
    if (col>=c && col<c+w && row>=r && row<r+h)
		{
		  reg = i;
		  break;
		}
	}
	return (reg);
}



void BkgFitterTracker::SetRegionProcessOrder (const CommandLineOpts &inception_state)
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
  printf("Number of live bead regions (nonZeroRegions): %d\n",nonZeroRegions);

  if (analysis_compute_plan.gpu_work_load != 0)
  {
    int gpuRegions = int (analysis_compute_plan.gpu_work_load * float (nonZeroRegions));
    if (gpuRegions > 0)
      analysis_compute_plan.lastRegionToProcess = gpuRegions;
  }
  // bestRegion is used for beads_bestRegion output to hdf5 file
  if (nonZeroRegions>0)
  {
      int r = inception_state.bkg_control.pest_control.bkgModelHdf5Debug_region_r;
      int c = inception_state.bkg_control.pest_control.bkgModelHdf5Debug_region_c;
      if (r >= 0 && c >= 0)
      {
      int reg =  findRegion(r,c);
	  //cout << "SetRegionProcessOrder... findRegion(" << x << "," << y << ") => bestRegion=" << reg << endl << flush;
      if (reg>=0)
          bestRegion = beadRegion(reg,sliced_chip[reg]->GetNumLiveBeads());
      else
          bestRegion = analysis_compute_plan.region_order[0];
      }
      else
          bestRegion = analysis_compute_plan.region_order[0];
      bestRegion_region = sliced_chip[bestRegion.first]->get_region();
      sliced_chip[bestRegion.first]->isBestRegion = true;
	  //cout << "SetRegionProcessOrder... bestRegion_region.row=" << bestRegion_region->row << " bestRegion_region.col=" << bestRegion_region->col << endl << flush;
  }
  else
  {
    bestRegion = beadRegion(0,0);
    bestRegion_region = NULL;
  }
  printf("BkgFitterTracker::SetRegionProcessOrder... bestRegion=(%d,%d)\n",
    bestRegion.first,bestRegion.second);

  // Now that we have a best region, we can init h5.
  InitBeads_BestRegion( inception_state );
}




void BkgFitterTracker::UnSpinGPUThreads ()
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


BkgFitterTracker::BkgFitterTracker (int numRegions) :
  numFitters( numRegions )
{
  //signal_proc_fitters = new SignalProcessingMasterFitter * [numRegions];
  signal_proc_fitters.resize(numRegions);
  sliced_chip.resize(numRegions);
  sliced_chip_extras.resize(numRegions);
  bkinfo = NULL;
  all_emptytrace_track = NULL;
  bestRegion_region = NULL;
}

void BkgFitterTracker::AllocateSlicedChipScratchSpace( int global_flow_max )
{
  size_t numRegions = sliced_chip.size();
  sliced_chip_extras.resize(numRegions);
  for(size_t i=0; i<numRegions; ++i) {
    sliced_chip_extras[i].allocate( global_flow_max );
  }
}

void BkgFitterTracker::AllocateRegionData(
    std::size_t numRegions, 
    const CommandLineOpts * inception_state 
  )
{
  sliced_chip.resize(numRegions);
  for(size_t i=0; i<numRegions; ++i) {
    sliced_chip[i] = new RegionalizedData( inception_state );
  }
}

void BkgFitterTracker::DeleteFitters()
{
  for (int r = 0; r < numFitters; r++)
    delete signal_proc_fitters[r];

  signal_proc_fitters.clear();

  for (int r = 0; r < numFitters; r++) {
    delete sliced_chip[r];
    sliced_chip_extras[r].free();
  }

  sliced_chip.clear();
  sliced_chip_extras.clear();
  
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

void BkgFitterTracker::PlanComputation (BkgModelControlOpts &bkg_control /*, int flow_max,
                                        master_fit_type_table *table*/)
{
  // how are we dividing the computation amongst resources available as directed by command line constraints

  PlanMyComputation (analysis_compute_plan,bkg_control/*, flow_max, table*/);

  AllocateProcessorQueue (analysis_queue,analysis_compute_plan,numFitters);
}

void BkgFitterTracker::SetUpTraceTracking(const SlicedPrequel &my_prequel_setup, const CommandLineOpts &inception_state, const ImageSpecClass &my_image_spec, const ComplexMask &from_beadfind_mask, int flow_max)
{
  // because it is hypothetically possible that we track empties over a different region mesh than regular beads
  // this is set up beforehand, while deferring the beads to the region mesh used in signal processing
  all_emptytrace_track = new EmptyTraceTracker(my_prequel_setup.region_list, my_prequel_setup.region_timing, my_prequel_setup.smooth_t0_est, inception_state);
  all_emptytrace_track->Allocate(from_beadfind_mask.my_mask, my_image_spec, flow_max);
  washout_flow.resize(from_beadfind_mask.my_mask->H() * from_beadfind_mask.my_mask->W(), -1);
}

static void TrivialDebugGaussExp(const string &outFile, const std::vector<Region> &regions, const std::vector<RegionTiming> &region_timing)
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


void BkgFitterTracker::InitBeads_BestRegion(const CommandLineOpts &inception_state)
{
    if (inception_state.bkg_control.pest_control.bkg_debug_files)
    {
        int nBeads_live = bestRegion.second;
        all_params_hdf.Init2(inception_state.bkg_control.pest_control.bkgModelHdf5Debug,nBeads_live,bestRegion_region);
    }
}


void BkgFitterTracker::InitBeads_xyflow(const CommandLineOpts &inception_state)
{
    if (inception_state.bkg_control.pest_control.bkg_debug_files && 
        inception_state.bkg_control.pest_control.bkgModel_xyflow_output)
    {
        bool readOk = false;
        int numFlows = inception_state.flow_context.GetNumFlows();
        switch(inception_state.bkg_control.pest_control.bkgModel_xyflow_fname_in_type)
        {
        case 1:
            readOk = inception_state.bkg_control.pest_control.read_file_sse(xyf_hash,numFlows);
            break;
        case 2:
            readOk = inception_state.bkg_control.pest_control.read_file_rcflow(xyf_hash,numFlows);
            break;
        case 3:
            readOk = inception_state.bkg_control.pest_control.read_file_xyflow(xyf_hash,numFlows);
            break;
        default:
            break;
        }
        if (! readOk) {
            std::cerr << "InitBeads_xyflow read error... " << inception_state.bkg_control.pest_control.bkgModel_xyflow_fname_in << std::endl << std::flush;
            exit(1);
        }
        // init xyflow
        all_params_hdf.InitBeads_xyflow(inception_state.bkg_control.pest_control.bkgModelHdf5Debug,xyf_hash);
    }
}


int BkgFitterTracker::getMaxFrames(
    const ImageSpecClass &my_image_spec, 
    const std::vector<RegionTiming> &region_timing
  )
{
  int max_frames = 0;
  for (size_t i = 0; i < sliced_chip.size(); i++) {
    TimeCompression time_c;
    int t0_frame = (int)(region_timing[i].t0_frame + VFC_T0_OFFSET + .5);
    time_c.choose_time = global_defaults.signal_process_control.choose_time;
    time_c.SetUpTime(my_image_spec.uncompFrames, t0_frame,
                   global_defaults.data_control.time_start_detail,
                   global_defaults.data_control.time_stop_detail,
                   global_defaults.data_control.time_left_avg);
    int nframes = time_c.npts();
    max_frames = max(max_frames, nframes);
  }
  return max_frames;
}


void BkgFitterTracker::ThreadedInitialization (
    RawWells &rawWells, 
    const CommandLineOpts &inception_state, 
    const ComplexMask &a_complex_mask, 
    const char *results_folder,
    const ImageSpecClass &my_image_spec, 
    const std::vector<float> &smooth_t0_est, 
    std::vector<Region> &regions,
		const std::vector<RegionTiming> &region_timing,
    const SeqListClass &my_keys,
    bool restart,
    int num_flow_blocks )
{
  int totalRegions = regions.size();
  // a debugging file if needed
  int max_frames = getMaxFrames(my_image_spec, region_timing);

  global_defaults.signal_process_control.set_max_frames(max_frames);
  if (inception_state.bkg_control.pest_control.bkg_debug_files) { 
    all_params_hdf.Init(  inception_state.sys_context.results_folder,
                          inception_state.loc_context,  
                          my_image_spec, inception_state.flow_context.GetNumFlows(),
                          inception_state.bkg_control.pest_control.bkgModelHdf5Debug, max_frames,
                          inception_state.bkg_control.signal_chunks.flow_block_sequence.MaxFlowsInAnyFlowBlock(), 
                          num_flow_blocks
                       );
  }
  
  // designate a set of reads that will be processed regardless of whether they pass filters
  set<int> randomLibSet;
  MaskSample<int> randomLib (*a_complex_mask.my_mask, MaskLib, inception_state.bkg_control.unfiltered_library_random_sample);
  randomLibSet.insert (randomLib.Sample().begin(), randomLib.Sample().end());

  InitCacheMath();
    
  if (inception_state.bkg_control.pest_control.bkg_debug_files)
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
    linfo[r].sliced_chip_extras = &sliced_chip_extras[0];
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

    if (inception_state.bkg_control.pest_control.bkgModelHdf5Debug)
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
  for (int r = 0; r < numFitters; r++)
    bkinfo[r].polyclonal_filter_opts = inception_state.bkg_control.polyclonal_filter;

}


// call the fitters for each region
void BkgFitterTracker::ExecuteFitForFlow (int flow, ImageTracker &my_img_set, bool last, 
  int flow_key, master_fit_type_table *table,
  const CommandLineOpts * inception_state )
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
    bkinfo[r].flow_key = flow_key;
    bkinfo[r].table = table;
    bkinfo[r].doingSdat = my_img_set.doingSdat;
    bkinfo[r].inception_state = inception_state;
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

void BkgFitterTracker::SpinUpGPUThreads()
{
  analysis_queue.SpinUpGPUThreads( analysis_compute_plan );
}


void BkgFitterTracker::DumpBkgModelBeadParams (char *results_folder,  int flow, bool debug_bead_only, int flow_max) const
{
  FILE *bkg_mod_bead_dbg = NULL;
  char *bkg_mod_bead_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_bead_dbg_fname, 512, "%s/%s.%04d.%s", results_folder, "BkgModelBeadData",flow+1,"txt");
  fopen_s (&bkg_mod_bead_dbg, bkg_mod_bead_dbg_fname, "wt");
  free (bkg_mod_bead_dbg_fname);

  BeadParams::DumpBeadTitle (bkg_mod_bead_dbg, flow_max);

  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->DumpExemplarBead (bkg_mod_bead_dbg,debug_bead_only, flow_max);
  }
  fclose (bkg_mod_bead_dbg);
}

void BkgFitterTracker::DumpBkgModelBeadOffset (char *results_folder, int flow, bool debug_bead_only) const
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


void BkgFitterTracker::DumpBkgModelBeadInfo (char *results_folder,  int flow, bool last_flow, 
    bool debug_bead_only, FlowBlockSequence::const_iterator flow_block) const
{
  // get some regional data for the entire chip as debug
  // should be triggered by bkgmodel
  if ( last_flow || flow + 1 == flow_block->end() )
  {
    DumpBkgModelBeadParams (results_folder, flow, debug_bead_only, flow_block->size() );
    DumpBkgModelBeadOffset (results_folder, flow, debug_bead_only);
  }
}

void BkgFitterTracker::DumpBkgModelEmphasisTiming (char *results_folder, int flow) const
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


void BkgFitterTracker::DumpBkgModelInitVals (char *results_folder, int flow) const
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

void BkgFitterTracker::DumpBkgModelDarkMatter (char *results_folder, int flow) const
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

void BkgFitterTracker::DumpBkgModelEmptyTrace (char *results_folder, int flow, int flow_max) const
{
  // dump the dark matter, which is a fitted compensation term
  FILE *bkg_mod_mt_dbg = NULL;
  char *bkg_mod_mt_name = (char *) malloc (512);
  snprintf (bkg_mod_mt_name, 512, "%s/%s.%04d.%s", results_folder, "BkgModelEmptyTraceData",flow+1,"txt");
  fopen_s (&bkg_mod_mt_dbg, bkg_mod_mt_name, "wt");
  free (bkg_mod_mt_name);

  for (int r = 0; r < numFitters; r++)
  {
    sliced_chip[r]->DumpEmptyTrace (bkg_mod_mt_dbg, flow_max);
  }
  fclose (bkg_mod_mt_dbg);
}

void BkgFitterTracker::DumpBkgModelRegionParameters (char *results_folder,int flow, int flow_max) const
{
  FILE *bkg_mod_reg_dbg = NULL;
  char *bkg_mod_reg_dbg_fname = (char *) malloc (512);
  snprintf (bkg_mod_reg_dbg_fname, 512, "%s/%s.%04d.%s", results_folder, "BkgModelRegionData",flow+1,"txt");
  fopen_s (&bkg_mod_reg_dbg, bkg_mod_reg_dbg_fname, "wt");
  free (bkg_mod_reg_dbg_fname);

  struct reg_params rp;

  rp.DumpRegionParamsTitle (bkg_mod_reg_dbg, flow_max);

  for (int r = 0; r < numFitters; r++)
  {
    signal_proc_fitters[r]->GetRegParams (rp);
    //@TODO this routine should have no knowledge of internal representation of variables
    // Make this a routine to dump an informative line to a selected file from a regional parameter structure/class
    // especially as we use a very similar dumping line a lot in different places.
    // note: t0, rdr, and pdr evolve over time.  It would be nice to also capture how they changed throughout the analysis
    // this only captures the final value of each.
    rp.DumpRegionParamsLine (bkg_mod_reg_dbg, signal_proc_fitters[r]->GetRegion()->row,signal_proc_fitters[r]->GetRegion()->col, flow_max);

  }
  fclose (bkg_mod_reg_dbg);
}

void BkgFitterTracker::DumpBkgModelRegionInfo (char *results_folder, int flow, bool last_flow, 
    FlowBlockSequence::const_iterator flow_block) const
{
  // get some regional data for the entire chip as debug
  // should be triggered by bkgmodel
  if ( last_flow || flow + 1 == flow_block->end() )
  {
    DumpBkgModelRegionParameters (results_folder, flow, flow_block->size() );
    DumpBkgModelDarkMatter (results_folder,  flow);
    DumpBkgModelEmphasisTiming (results_folder, flow);
    DumpBkgModelEmptyTrace (results_folder, flow, flow_block->size() );
    DumpBkgModelInitVals (results_folder, flow);
  }
}

void BkgFitterTracker::DetermineAndSetGPUAllocationAndKernelParams( 
    BkgModelControlOpts &bkg_control, 
    int global_max_flow_key,
    int global_max_flow_max
  )
{
  int maxBeads = 0;
  int maxFrames = global_defaults.signal_process_control.get_max_frames();
  if (maxFrames)
      GpuMultiFlowFitControl::SetMaxFrames(maxFrames);

  for (int i=0; i<numFitters; ++i) {
    maxBeads = maxBeads < sliced_chip[i]->GetNumLiveBeads() ?
        sliced_chip[i]->GetNumLiveBeads() : maxBeads;
  }
  if (maxBeads)
    GpuMultiFlowFitControl::SetMaxBeads(maxBeads);

  GpuMultiFlowFitControl::SetChemicalXtalkCorrectionForPGM(bkg_control.enable_trace_xtalk_correction);

  cout << "CUDA: worst case per region beads: "<< maxBeads << " frames: " << maxFrames << endl;
  configureKernelExecution(bkg_control.gpuControl, global_max_flow_key, global_max_flow_max);

}
