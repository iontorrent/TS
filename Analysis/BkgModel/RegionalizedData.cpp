/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RegionalizedData.h"
#include <iostream>
#include <fstream>
#include "IonErr.h"

using namespace std;

void RegionalizedData::DumpRegionParameters (float cur_avg_resid)
{ //lev_mar_fit->lm_state.avg_resid
  if (region != NULL)
    printf ("r(x,y) d(T,A,C,G) k(T,A,C,G)=(%d,%d) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f) %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n",
            region->col,region->row,
            my_regions.rp.d[0],my_regions.rp.d[1],my_regions.rp.d[2],my_regions.rp.d[3],
            my_regions.rp.krate[0],my_regions.rp.krate[1],my_regions.rp.krate[2],my_regions.rp.krate[3],
            my_regions.rp.kmax[0],my_regions.rp.kmax[1],my_regions.rp.kmax[2],my_regions.rp.kmax[3],
            cur_avg_resid,my_regions.rp.tshift,my_regions.rp.tau_R_m,my_regions.rp.tau_R_o,my_regions.rp.nuc_shape.sigma,GetTypicalMidNucTime (& (my_regions.rp.nuc_shape)));
  if (region != NULL)
    printf ("---(%d,%d) (%5.3f, %5.3f, %5.3f, %5.3f) (%5.3f, %5.3f, %5.3f, %5.3f)\n",
            region->col,region->row,
            my_regions.rp.nuc_shape.t_mid_nuc_delay[0],my_regions.rp.nuc_shape.t_mid_nuc_delay[1],my_regions.rp.nuc_shape.t_mid_nuc_delay[2],my_regions.rp.nuc_shape.t_mid_nuc_delay[3],
            my_regions.rp.nuc_shape.sigma_mult[0],my_regions.rp.nuc_shape.sigma_mult[1],my_regions.rp.nuc_shape.sigma_mult[2],my_regions.rp.nuc_shape.sigma_mult[3]);
}


void RegionalizedData::DumpTimeAndEmphasisByRegion (FILE *my_fp)
{
  // this will be a dumb file format
  // each line has x,y, type, hash, data per time point
  // dump timing
  int i = 0;
  fprintf (my_fp,"%d\t%d\t", region->col, region->row);
  fprintf (my_fp,"time\t%d\t",time_c.npts());

  for (i=0; i<time_c.npts(); i++)
    fprintf (my_fp,"%f\t",time_c.frameNumber[i]);
  for (; i<MAX_COMPRESSED_FRAMES; i++)
    fprintf (my_fp,"0.0\t");
  fprintf (my_fp,"\n");
  fprintf (my_fp,"%d\t%d\t", region->col, region->row);
  fprintf (my_fp,"frames_per_point\t%d\t",time_c.npts());
  for (i=0; i<time_c.npts(); i++)
    fprintf (my_fp,"%d\t", time_c.frames_per_point[i]);
  for (; i<MAX_COMPRESSED_FRAMES; i++)
    fprintf (my_fp,"0\t");
  fprintf (my_fp,"\n");
  // dump emphasis
  for (int el=0; el<emphasis_data.numEv; el++)
  {
    fprintf (my_fp,"%d\t%d\t", region->col, region->row);
    fprintf (my_fp,"em\t%d\t",el);
    for (i=0; i<time_c.npts(); i++)
      fprintf (my_fp,"%f\t",emphasis_data.EmphasisVectorByHomopolymer[el][i]);
    for (; i<MAX_COMPRESSED_FRAMES; i++)
      fprintf (my_fp,"0.0\t");
    fprintf (my_fp,"\n");
  }
}

void RegionalizedData::LimitedEmphasis( )
{
    my_beads.AssignEmphasisForAllBeads (0);
}

void RegionalizedData::AdaptiveEmphasis()
{
  my_beads.AssignEmphasisForAllBeads (emphasis_data.numEv-1);
}

void RegionalizedData::NoData()
{
  // I cannot believe I have to do this to make cppcheck happy
  // data branch
  region = NULL;
  emptyTraceTracker = NULL;
  emptytrace = NULL;

  doDcOffset = true;
  regionAndTimingMatchSdat = false;
  // timing start
  sigma_start = 0.0f;
  t_mid_nuc_start = 0.0f;
  // job holder flag
  fitters_applied = -1; // ready to add data
}


void RegionalizedData::SetTimeAndEmphasis (GlobalDefaultsForBkgModel &global_defaults, float tmid, float t0_offset)
{
  time_c.choose_time = global_defaults.signal_process_control.choose_time;
  time_c.SetUpTime (my_trace.imgFrames,t0_offset,global_defaults.data_control.time_start_detail, global_defaults.data_control.time_stop_detail, global_defaults.data_control.time_left_avg);
  time_c.t0 = t0_offset;
  // check the points that we need
  if (CENSOR_ZERO_EMPHASIS>0)
  {
    EmphasisClass trial_emphasis;

    // assuming that our t_mid_nuc estimation is decent
    // see what the emphasis functions needed for "detailed" results are
    // and assume the "coarse" blank emphasis function will work out fine.
    trial_emphasis.SetDefaultValues (global_defaults.data_control.emp,global_defaults.data_control.emphasis_ampl_default, global_defaults.data_control.emphasis_width_default);
    trial_emphasis.SetupEmphasisTiming (time_c.npts(), &time_c.frames_per_point[0],&time_c.frameNumber[0]);
    trial_emphasis.point_emphasis_by_compression = global_defaults.data_control.point_emphasis_by_compression;
    trial_emphasis.BuildCurrentEmphasisTable (t_mid_nuc_start, FINEXEMPHASIS);
//    int old_pts = time_c.npts;
    time_c.npts(trial_emphasis.ReportUnusedPoints (CENSOR_THRESHOLD, MIN_CENSOR)); // threshold the points for the number actually needed by emphasis

    // don't bother monitoring this now
    //printf ("Saved: %f = %d of %d\n", (1.0*time_c.npts) / (1.0*old_pts), time_c.npts, old_pts);
    // now give the emphasis data structure (and everything else) using the "used" number of points
  }
  emphasis_data.SetDefaultValues (global_defaults.data_control.emp,global_defaults.data_control.emphasis_ampl_default, global_defaults.data_control.emphasis_width_default);
  emphasis_data.SetupEmphasisTiming (time_c.npts(), &time_c.frames_per_point[0],&time_c.frameNumber[0]);
  emphasis_data.point_emphasis_by_compression = global_defaults.data_control.point_emphasis_by_compression;
  emphasis_data.BuildCurrentEmphasisTable (t_mid_nuc_start, FINEXEMPHASIS);
}


void RegionalizedData::SetupTimeAndBuffers (GlobalDefaultsForBkgModel &global_defaults,float sigma_guess,
    float t_mid_nuc_guess,
    float t0_offset)
{
  sigma_start = sigma_guess;
  t_mid_nuc_start = t_mid_nuc_guess;
  my_regions.InitRegionParams (t_mid_nuc_start,sigma_start, global_defaults);

  SetTimeAndEmphasis (global_defaults, t_mid_nuc_guess, t0_offset);

  AllocTraceBuffers();

  AllocFitBuffers();
}

void RegionalizedData::SetTshiftLimitsForSynchDat()
{
    // Reset the global parameters here. @todo - put this in with the rest of the parameters?
    // Need a configure() hook for first image if we're going to stick with the SignalProcessingMasterFitter doing
    // things implicitly.
    my_regions.rp_low.tshift    = -1.5f;
    my_regions.rp_high.tshift    = 3.5f;
    //    my_regions.rp.tshift    = 1.1f;
    my_regions.rp.tshift    = .4f;
    
}

void RegionalizedData::AllocFitBuffers()
{
  // so we need to make sure these structures match
  my_scratch.Allocate (time_c.npts(),1);
  my_regions.AllocScratch (time_c.npts());
}

void RegionalizedData::AllocTraceBuffers()
{
  // now do the traces set up for time compression
  my_trace.Allocate (my_flow.numfb,NUMFB*time_c.npts(),my_beads.numLBeads);
  my_trace.time_cp = &time_c; // point to the global time compression

}

void RegionalizedData::AddOneFlowToBuffer (GlobalDefaultsForBkgModel &global_defaults, int flow)
{
  // keep track of which flow # is in which buffer
  // also keep track of which nucleotide is associated with each flow
  my_flow.SetBuffFlow (flow);
  my_flow.SetFlowNdxMap (global_defaults.flow_global.GetNucNdx (flow));
  my_flow.SetDblTapMap (global_defaults.flow_global.IsDoubleTap (flow));

  // reset parameters for beads when we actually start fitting the beads
}

bool RegionalizedData::LoadOneFlow (SynchDat &data, GlobalDefaultsForBkgModel &global_defaults, int flow)
{
  doDcOffset = false;
  //  TraceChunk &chunk = data.GetItemByRowCol (get_region_row(), get_region_col());
  //  ION_ASSERT(chunk.mHeight == (size_t)region->h && chunk.mWidth == (size_t)region->w, "Wrong Region size.");
  AddOneFlowToBuffer (global_defaults,flow);
  UpdateTracesFromImage (data, flow);
  my_flow.Increment();
  //return (false);
  return (true);
}

bool RegionalizedData::LoadOneFlow (Image *img, GlobalDefaultsForBkgModel &global_defaults, int flow)
{
  doDcOffset = true;
  const RawImage *raw = img->GetImage();
  if (!raw)
  {
    fprintf (stderr, "ERROR: no image data\n");
    return true;
  }

  AddOneFlowToBuffer (global_defaults,flow);

  UpdateTracesFromImage (img, flow);

  my_flow.Increment();

  return (false);
}

void RegionalizedData::UpdateTracesFromImage (Image *img, int flow)
{
  my_trace.SetRawTrace(); // buffers treated as raw traces

  // populate bead traces from image file and
  // time-shift traces for uniform start times; compress traces to flows buffer
  my_trace.GenerateAllBeadTrace (region,my_beads,img, my_flow.flowBufferWritePos);
  // subtract mean signal in time before flow starts from traces in flows buffer

  float t_mid_nuc =  GetTypicalMidNucTime (&my_regions.rp.nuc_shape);
  float t_offset_beads = my_regions.rp.nuc_shape.sigma;
  my_trace.RezeroBeads (time_c.time_start, t_mid_nuc-t_offset_beads,
                        my_flow.flowBufferWritePos);

  // calculate average trace across all empty wells in a region for a flow
  // to FileLoadWorker at Image load time, should be able to go here
  // emptyTraceTracker->SetEmptyTracesFromImageForRegion(*img, global_state.pinnedInFlow, flow, global_state.bfmask, *region, t_mid_nuc);
  emptytrace = emptyTraceTracker->GetEmptyTrace (*region);

  // sanity check images are what we think
  assert (emptytrace->imgFrames == my_trace.imgFrames);
}

void RegionalizedData::UpdateTracesFromImage (SynchDat &chunk, int flow)
{
  ION_ASSERT(my_trace.time_cp->npts() <= (int)chunk.NumFrames(region->row, region->col), "Wrong number of frames.")
  my_trace.SetRawTrace(); // buffers treated as raw traces
  // populate bead traces from image file, just filling in data already saved
  my_trace.GenerateAllBeadTrace (region,my_beads,chunk, my_flow.flowBufferWritePos, regionAndTimingMatchSdat);

  // calculate average trace across all empty wells in a region for a flow
  // to FileLoadWorker at Image load time, should be able to go here
  emptytrace = emptyTraceTracker->GetEmptyTrace (*region);

  // sanity check images are what we think
  assert ( emptytrace->imgFrames == (int)chunk.NumFrames(region->row, region->col));
}

// Trivial fitters

// t_offset_beads = nuc_shape.sigma
// t_offset_empty = 4.0
void RegionalizedData::RezeroTraces (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum)
{
  if (doDcOffset)
  {
    // do these values make sense for offsets in RezeroBeads???
    emptytrace->RezeroReference (t_start, t_mid_nuc-t_offset_beads, fnum);
  }
  else
  {
   my_trace.RezeroBeads (t_start, t_mid_nuc - t_offset_beads, fnum);
    emptytrace->RezeroCompressedReference (my_trace.time_cp, t_start, t_mid_nuc - t_offset_beads, fnum);
  }
}

void RegionalizedData::RezeroTracesAllFlows (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty)
{
  //   my_trace.RezeroBeadsAllFlows (t_start, t_mid_nuc-t_offset_beads);
  if (doDcOffset)
  {
    emptytrace->RezeroReferenceAllFlows (t_start, t_mid_nuc - t_offset_empty);
  }
  else
  {
   my_trace.RezeroBeadsAllFlows (t_start, t_mid_nuc-t_offset_beads);
    emptytrace->RezeroCompressedReferenceAllFlows (my_trace.time_cp, t_start, t_mid_nuc - t_offset_beads);
    //emptytrace->RezeroCompressedReferenceAllFlows (&my_trace->time_cp);
  }
}

void RegionalizedData::RezeroByCurrentTiming()
{
  RezeroTracesAllFlows (time_c.time_start, GetTypicalMidNucTime (& (my_regions.rp.nuc_shape)), my_regions.rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE);
}

void RegionalizedData::PickRepresentativeHighQualityWells (float ssq_filter)
{
  my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (false); // retain fitted key values for snr purposes
  // use only high quality beads from now on when regional fitting
  if (ssq_filter>0.0f)
    my_beads.LowSSQRatioBeadsAreLowQuality (ssq_filter);
  my_beads.LowCopyBeadsAreLowQuality (my_beads.my_mean_copy_count);
  my_beads.KeyNormalizeReads (true); // force all beads to read the "true key" in the key flows
}

RegionalizedData::RegionalizedData()
{
  NoData();

}

RegionalizedData::~RegionalizedData()
{

}


void RegionalizedData::SetCrudeEmphasisVectors()
{
  // head off bad behaviour if this region was skipped during processing of previous block of flows.
  // not clear this is such a great way to solve this problem.
  if(my_beads.numLBeads != 0)
    emphasis_data.BuildCurrentEmphasisTable (GetTypicalMidNucTime (& (my_regions.rp.nuc_shape)), CRUDEXEMPHASIS); // why is this not per nuc?
}

void RegionalizedData::SetFineEmphasisVectors()
{
    emphasis_data.BuildCurrentEmphasisTable (GetTypicalMidNucTime (& (my_regions.rp.nuc_shape)), FINEXEMPHASIS);
}


void RegionalizedData::DumpEmptyTrace (FILE *my_fp)
{
  assert (emptyTraceTracker != NULL);   //sanity
  if (region!=NULL)
  {
    (emptyTraceTracker->GetEmptyTrace (*region))->DumpEmptyTrace (my_fp,region->col,region->row);
  }
}
