/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "EmptyTraceTracker.h"

// Each SignalProcessingMasterFitter object is associated with
// 1 Region and 1 block of flows 
// and needs access to an appropriate EmptyTrace object.
// This class provides the linkage between
// EmptyTrace objects and BkgModelFitter objects via the Region
// Acts a cache, so algorithmically assumes that an empty trace is constant
// across one or more regions per flow rather than per well/flow
// flow handling is implicitly done by the EmptyTrace object itself

EmptyTraceTracker::EmptyTraceTracker(
    const std::vector<Region> &_regions,
    const std::vector<RegionTiming> &_regionTiming, 
    const std::vector<float> &_sep_t0_est, 
    const CommandLineOpts& _inception_state
  )
  : regions(_regions), regionTiming(_regionTiming), sep_t0_est(_sep_t0_est), inception_state(_inception_state)
{
  assert (regions.size() == regionTiming.size());
  assert (regions.size() > 0);

  maxNumRegions = MaxNumRegions(regions);
  assert(maxNumRegions > 0);

  if ( &inception_state != NULL) { // inception_state not restored in a restart
    outlierDumpFile = inception_state.sys_context.analysisLocation + "/refOutliers.txt";
  }
}

EmptyTraceTracker::~EmptyTraceTracker()
{
 
  for (int r = 0; r < maxNumRegions; r++)
    if (emptyTracesForBMFitter[r] != NULL)
      delete emptyTracesForBMFitter[r];

  emptyTracesForBMFitter.clear();
}

// current algorithm is 1 empty trace per region per block of flows
// empty traces are re-used in subsequent blocks of flows
// regions are assumed uniform size arranged in a grid over the
// the image surface.  Grid need not be 100% populated by regions.
// regions are indexed set up in SetUpRegions
// Replace this function to try out other mappings of EmptyTraces to regions
// e.g., an EmptyTrace that spans surrounding regions for each region
// e.g., an EmptyTrace that spans a block of regions
// Note also that timing & t0_map in the EmptyTrace object is set by region
// and would have to be reconciled in a change as above

void EmptyTraceTracker::Allocate(const Mask *bfmask, const ImageSpecClass &imgSpec, int flow_block_size)
{
  // assumes regions are indexed over a small non-negative range
  imgFrames.resize(maxNumRegions);

  for (unsigned int i=0; i<regions.size(); i++){
    Region region = regions[i];
    imgFrames[region.index] = imgSpec.uncompFrames;
  }
  
  emptyTracesForBMFitter.resize(maxNumRegions);
  for (int i=0; i<maxNumRegions; i++)
    emptyTracesForBMFitter[i] = NULL;

  for (unsigned int i=0; i<regions.size(); i++){
    Region region = regions[i];

    // allocate only once, if changed need to deallocate
    assert ((region.index < maxNumRegions) & (region.index>=0));
    assert (emptyTracesForBMFitter[region.index] == NULL);

    // allocate empty traces, current algorithm is 1 per region
    // each region is also used by a BkgModelFitters
    EmptyTrace *emptyTrace = AllocateEmptyTrace(region, imgFrames[region.index], flow_block_size);

    emptyTrace->SetTrimWildTraceOptions(inception_state.bkg_control.trace_control.do_ref_trace_trim,
                                        inception_state.bkg_control.trace_control.span_inflator_min,
                                        inception_state.bkg_control.trace_control.span_inflator_mult,
                                        inception_state.bkg_control.trace_control.cutoff_quantile,
                                        global_defaults.data_control.nuc_flow_frame_width);
  
    emptyTrace->T0EstimateToMap(sep_t0_est, &region, bfmask);

    emptyTrace->CountReferenceTraces(region, bfmask);
    //fprintf(stdout, "Found %d reference traces starting at %d in region %d\n", cnt, ((EmptyTraceRecorder *)emptyTrace)->regionIndicesStartIndex, region.index);

    // assign the empty trace to the lookup vector BkgModelFitter can use
    // all regions must match an entry into emptyTracesForBMFitter so
    // every BkgModelFitter can find an EmptyTrace for its region.
    emptyTracesForBMFitter[region.index] = emptyTrace;
  }

  if (inception_state.bkg_control.trace_control.do_ref_trace_trim)
    InitializeDumpOutlierTracesFile();

}

void EmptyTraceTracker::SetEmptyTracesFromImageForRegion(
    Image &img, 
    const PinnedInFlow &pinnedInFlow, 
    int raw_flow, 
    const Mask *bfmask, 
    Region& region, 
    float t_mid_nuc_start,
    int flow_buffer_index
  )
{
  EmptyTrace *emptyTrace = NULL;

  // fprintf(stdout, "ETT: Setting Empty trace %lx in %lx[%d] for flow %d\n", (unsigned long)emptyTracesForBMFitter[region.index], (unsigned long)emptyTracesForBMFitter, region.index, flow);

  // set up timing for initial re-zeroing
  TimeCompression time_cp;
  time_cp.choose_time = global_defaults.signal_process_control.choose_time; // have to start out using the same compression as bkg model - this will become easier if we coordinate time tracker
  time_cp.SetUpTime(imgFrames[region.index],t_mid_nuc_start,global_defaults.data_control.time_start_detail,
        global_defaults.data_control.time_stop_detail,global_defaults.data_control.time_left_avg);
  float t_start = time_cp.time_start;
  

  emptyTrace = emptyTracesForBMFitter[region.index];
  emptyTrace->SetUsed(true);

  // make the emptyTrace aware of time in seconds
  emptyTrace->SetTime(time_cp.frames_per_second);

  // calculate average trace across all empty wells in this region for this flow
  emptyTrace->GenerateAverageEmptyTrace(&region, pinnedInFlow, bfmask, &img, flow_buffer_index,
                                        raw_flow);

  if (emptyTrace->nOutliers > 0)
    DumpOutlierTracesPerFlowPerRegion(raw_flow, region, emptyTrace->nOutliers, emptyTrace->nRef);

  // fill the buffer neg_bg_buffers_slope
  emptyTrace->RezeroReference(t_start, t_mid_nuc_start-MAGIC_OFFSET_FOR_EMPTY_TRACE, 
                              flow_buffer_index);
  emptyTrace->PrecomputeBackgroundSlopeForDeriv (flow_buffer_index);
 }


EmptyTrace *EmptyTraceTracker::AllocateEmptyTrace(Region &region, int nframes, int flow_block_size)
{
  EmptyTrace *emptyTrace;
  int ix = region.index;

  /*if (inception_state.bkg_control.replayBkgModelData)
    emptyTrace = new EmptyTraceReader(inception_state);
  else if (inception_state.bkg_control.recordBkgModelData)
    emptyTrace = new EmptyTraceRecorder(inception_state);
  else*/
    emptyTrace = new EmptyTrace(inception_state);

  emptyTrace->regionIndex = region.index;
  emptyTrace->Allocate(flow_block_size, nframes);

  emptyTracesForBMFitter[ix] = emptyTrace;
  return(emptyTrace);
}

EmptyTrace *EmptyTraceTracker::GetEmptyTrace(const Region &region)
{
  // only called after empty traces all initialized
  int r = region.index;
  assert( r < maxNumRegions );
  EmptyTrace *emptytrace = emptyTracesForBMFitter[r];
  assert( emptytrace != NULL );
  return(emptytrace);
}

int EmptyTraceTracker::MaxNumRegions(const std::vector<Region>& regions)
{
  int nmax = -1;

  for (unsigned int r=0; r < regions.size(); r++)
    nmax = (nmax > regions[r].index) ? nmax : regions[r].index;

  return (nmax+1);
}

void EmptyTraceTracker::InitializeDumpOutlierTracesFile()
{
  // initialize dump file
  FILE *fp = NULL;
  fopen_s (&fp, outlierDumpFile.c_str(), "w");
  if (fp) {
    fprintf(fp, "Flow\tRegionRow\tRegionCol\tNumOutliers\tNumRef\n");
    fclose (fp);
  }
  else {
    fprintf (stdout, "Could not open %s, err %s\n", outlierDumpFile.c_str(), strerror (errno));
  }
}

void EmptyTraceTracker::DumpOutlierTracesPerFlowPerRegion(int flow, Region& region, int nOutliers, int nRef)
{
  FILE *fp = NULL;

  fopen_s (&fp, outlierDumpFile.c_str(), "a");
  if (!fp) {
    fprintf (stdout, "Could not open %s, err %s\n", outlierDumpFile.c_str(), strerror (errno));
  }
  else {
    fprintf (fp, "%d\t%d\t%d\t%d\t%d\n", flow, region.row, region.col, nOutliers, nRef);
    fclose (fp);
  }
}
