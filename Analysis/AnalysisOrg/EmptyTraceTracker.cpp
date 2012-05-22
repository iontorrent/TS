/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "EmptyTraceTracker.h"

// Each BkgModelFitter object is associated with 1 Region and 1 block of flows 
// and needs access to an appropriate EmptyTrace object.
// This class provides the linkage between
// EmptyTrace objects and BkgModelFitter objects via the Region
// Acts a cache, so algorithmically assumes that an empty trace is constant
// across one or more regions per flow rather than per well/flow
// flow handling is implicitly done by the EmptyTrace object itself

EmptyTraceTracker::EmptyTraceTracker(Region *_regions, RegionTiming *_regionTiming, int nregions, std::vector<float> &_sep_t0_est, ImageSpecClass &imgSpec, CommandLineOpts& _clo)
  : numRegions(nregions), regions(_regions), regionTiming(_regionTiming), sep_t0_est(_sep_t0_est), clo(_clo)
{
  assert (numRegions > 0);

  maxNumRegions = MaxNumRegions(regions, numRegions);
  assert(maxNumRegions > 0);

  emptyTracesForBMFitter = new EmptyTrace * [maxNumRegions];
  for (int i=0; i<maxNumRegions; i++)
    emptyTracesForBMFitter[i] = NULL;

  imgFrames = imgSpec.uncompFrames;
  unallocated = new int [maxNumRegions];
  for (int i=0; i<maxNumRegions; i++)
    unallocated[i] = 1; //true
  
}

EmptyTraceTracker::~EmptyTraceTracker()
{
  for (int r = 0; r < numRegions; r++)
    if (emptyTracesForBMFitter[r] != NULL)
      delete emptyTracesForBMFitter[r];

  delete [] emptyTracesForBMFitter; // remove pointers
  delete [] unallocated;
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

void EmptyTraceTracker::SetEmptyTracesFromImage(Image &img, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask)
{
  for (int r=0; r<numRegions; r++){


    float t_mid_nuc_start = regionTiming[r].t_mid_nuc;

    SetEmptyTracesFromImageForRegion(img, pinnedInFlow, flow, bfmask, regions[r], t_mid_nuc_start);
  }
}

void EmptyTraceTracker::SetEmptyTracesFromImageForRegion(Image &img, PinnedInFlow &pinnedInFlow, int flow, Mask *bfmask, Region& region, float t_mid_nuc_start)
{
  EmptyTrace *emptyTrace = NULL;

  //fprintf(stdout, "ETT: Setting Empty traces for flow %d and region.index %d\n", flow, region.index);

  if (flow > 0)
    while ((int volatile *)unallocated[region.index]) // really read memory
      sleep(1);

  // set up timing for initial re-zeroing
  TimeCompression time_cp;
  time_cp.choose_time = global_defaults.choose_time; // have to start out using the same compression as bkg model - this will become easier if we coordinate time tracker
  time_cp.SetUpTime(imgFrames, t_mid_nuc_start, global_defaults.time_start_detail,
        global_defaults.time_stop_detail,global_defaults.time_left_avg);
  float t_start = time_cp.time_start;

  if ( flow == 0 )
  {
    // allocate only once, if changed need to deallocate
    assert (emptyTracesForBMFitter[region.index] == NULL);

    // allocate empty traces, current algorithm is 1 per region
    // each region is also used by a BkgModelFitters
    emptyTrace = AllocateEmptyTrace(region, imgFrames);

    emptyTrace->T0EstimateToMap(&sep_t0_est, &region, bfmask);

    // assign the empty trace to the lookup vector BkgModelFitter can use
    // all regions must match an entry into emptyTracesForBMFitter so
    // every BkgModelFitter can find an EmptyTrace for its region.
    emptyTracesForBMFitter[region.index] = emptyTrace;

    ((int volatile *)unallocated)[region.index] = 0;
  }
  
  emptyTrace = emptyTracesForBMFitter[region.index];
  emptyTrace->regionIndex = region.index; // for debugging
  // calculate average trace across all empty wells in this region for this flow
  emptyTrace->GenerateAverageEmptyTrace(&region, pinnedInFlow, bfmask, &img, flow);

  // fill the buffer neg_bg_buffers_slope
  emptyTrace->RezeroReference(t_start, t_mid_nuc_start-MAGIC_OFFSET_FOR_EMPTY_TRACE, flow);
  emptyTrace->PrecomputeBackgroundSlopeForDeriv (flow);
 }


EmptyTrace *EmptyTraceTracker::AllocateEmptyTrace(Region &region, int imgFrames)
{
  EmptyTrace *emptyTrace;
  int ix = region.index;

  if (clo.bkg_control.replayBkgModelData)
    emptyTrace = new EmptyTraceReader(clo);
  else if (clo.bkg_control.recordBkgModelData)
    emptyTrace = new EmptyTraceRecorder(clo);
  else
    emptyTrace = new EmptyTrace(clo);

  emptyTrace->Allocate(NUMFB, imgFrames);

  emptyTracesForBMFitter[ix] = emptyTrace;
  return(emptyTrace);
}

EmptyTrace *EmptyTraceTracker::GetEmptyTrace(Region &region)
{
  // only called after empty traces all initialized
  int r = region.index;
  assert( r < maxNumRegions );
  EmptyTrace *emptytrace = emptyTracesForBMFitter[r];
  assert( emptytrace != NULL );
  return(emptytrace);
}

int EmptyTraceTracker::MaxNumRegions(Region *regions, int nregions)
{
  int nmax = -1;

  for (int r=0; r<nregions; r++)
    nmax = (nmax > regions[r].index) ? nmax : regions[r].index;

  return (nmax+1);
}

