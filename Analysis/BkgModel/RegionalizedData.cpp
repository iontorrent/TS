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
  t0_frame = -1.0;
  // job holder flag
  fitters_applied = -1; // ready to add data
}


void RegionalizedData::SetTimeAndEmphasis (GlobalDefaultsForBkgModel &global_defaults, float tmid, float t0_offset)
{
  time_c.choose_time = global_defaults.signal_process_control.choose_time;
  time_c.SetUpTime (my_trace.imgFrames,t0_offset,global_defaults.data_control.time_start_detail, global_defaults.data_control.time_stop_detail, global_defaults.data_control.time_left_avg);
  time_c.t0 = t0_offset;

  // if fitting the exponential decay at the tail-end of the incorporation trace, we
  // need to keep more data points.
  if (global_defaults.signal_process_control.exp_tail_fit)
  {
     // primarily for thumbnail processing...we might need to ignore some of the long-time-history data points in the file
     // This is an annoying consequence of the software-derived thumbnail..where independent VFC data from each region
     // is combined into a single unified image all with a common time-domain.  As a result...most tiles in the thumbnail
     // have invalid data points, and it is harmful to include these in the processing.  Unfortunately...there is no simple
     // way to know from the image data which data points are valid, and which are not.
     // we make a blanket assumption here that data points beyond t0+60 frames are not useful.....this is probably 
     // mostly accurate.
     // The FPGA-based thumbnail does not have this issue
     float tend = t0_offset + 60.0f;
     int iend = 0;
     for (int i=0;i < time_c.npts();i++)
     {
        iend = i;
        if (time_c.frameNumber[i] >= tend)
           break;
     }

     if (iend < time_c.npts())
        time_c.npts(iend);
  }
  else
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
       //    trial_emphasis.BuildCurrentEmphasisTable (t_mid_nuc_start, FINEXEMPHASIS);
       trial_emphasis.BuildCurrentEmphasisTable (t0_offset, FINEXEMPHASIS);
   //    int old_pts = time_c.npts;
       time_c.npts(trial_emphasis.ReportUnusedPoints (CENSOR_THRESHOLD, MIN_CENSOR)); // threshold the points for the number actually needed by emphasis

       // don't bother monitoring this now
       //printf ("Saved: %f = %d of %d\n", (1.0*time_c.npts) / (1.0*old_pts), time_c.npts, old_pts);
       // now give the emphasis data structure (and everything else) using the "used" number of points
     }

  emphasis_data.SetDefaultValues (global_defaults.data_control.emp,global_defaults.data_control.emphasis_ampl_default, global_defaults.data_control.emphasis_width_default);
  emphasis_data.SetupEmphasisTiming (time_c.npts(), &time_c.frames_per_point[0],&time_c.frameNumber[0]);
  emphasis_data.point_emphasis_by_compression = global_defaults.data_control.point_emphasis_by_compression;
  //  emphasis_data.BuildCurrentEmphasisTable (t_mid_nuc_start, FINEXEMPHASIS);
  emphasis_data.BuildCurrentEmphasisTable (t0_offset, FINEXEMPHASIS);
}


void RegionalizedData::SetupTimeAndBuffers (GlobalDefaultsForBkgModel &global_defaults,float sigma_guess,
    float t_mid_nuc_guess,
    float t0_offset)
{
  sigma_start = sigma_guess;
  t_mid_nuc_start = t_mid_nuc_guess;
  t0_frame = t0_offset;
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
  float t_mid_nuc =  GetTypicalMidNucTime (&my_regions.rp.nuc_shape);
  float t_offset_beads = my_regions.rp.nuc_shape.sigma;
  my_trace.RezeroBeads(time_c.time_start, t_mid_nuc-t_offset_beads, my_flow.flowBufferWritePos);
  // calculate average trace across all empty wells in a region for a flow
  // to FileLoadWorker at Image load time, should be able to go here
  emptytrace = emptyTraceTracker->GetEmptyTrace (*region);
}

// Trivial fitters

// t_offset_beads = nuc_shape.sigma
// t_offset_empty = 4.0
void RegionalizedData::RezeroTraces (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum)
{
  emptytrace->RezeroReference (t_start, t_mid_nuc-t_offset_beads, fnum);
  my_trace.RezeroBeads (t_start, t_mid_nuc - t_offset_beads, fnum);
}

void RegionalizedData::RezeroTracesAllFlows (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty)
{
  my_trace.RezeroBeadsAllFlows (t_start, t_mid_nuc-t_offset_beads);
  emptytrace->RezeroReferenceAllFlows (t_start, t_mid_nuc - t_offset_empty);
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

/** Given uninitialized vectors key_zeromer, key_onemer, keyLen
 *  inputs to compute per bead signal: sc, t0_ix, t_end_ix
 *  modify key_zeromer to contain estimates of zeromer signal per bead
 *  and    key_onemer  to contain estimates of onemer signal per bead
 *  and    key_len     to contain the length of the key per bead
 *  (both of length numLBeads).
 */
void RegionalizedData::ZeromerAndOnemerAllBeads(Clonality& sc, size_t const t0_ix, size_t const t_end_ix, std::vector<float>& key_zeromer, std::vector<float>& key_onemer, std::vector<int>& keyLen)
{
  for (int ibd=0; ibd < my_beads.numLBeads; ibd++) {

    // find key for this bead
    keyLen[ibd] = 0;
    int key_id = my_beads.key_id[ibd];
    std::vector<int> key(NUMFB,-1);
    if (key_id >= 0)  // library or TF bead assumed to have key, go get it
      my_beads.SelectKeyFlowValuesFromBeadIdentity (&key[0], NULL, key_id, keyLen[ibd]);

    // find signal for this bead in its key flows
    std::vector<float> signal(keyLen[ibd], 0);
    for (int fnum=0; fnum < keyLen[ibd]; fnum++)
    {
      // approximate measure related to signal computed from trace
      // trace is assumed zero'd
      vector<float> trace(time_c.npts(), 0);
      my_trace.AccumulateSignal(&trace[0],ibd,fnum,time_c.npts());
      // signal[fnum] = sc.Incorporation(t0_ix, t_end_ix, trace);
      signal[fnum] = sc.Incorporation(t0_ix, t_end_ix, trace, fnum);
    }
    // estimate key_zeromer & key_onemer for this bead
    ZeromerAndOnemerOneBead(key, keyLen[ibd], signal, key_zeromer[ibd], key_onemer[ibd]);
  }
}

/** Given inputs key, keyLen and signal in the key flows,
 * return estimates of onemer & zeromer across the key flows
 * algorithm is "mean"
 */
void RegionalizedData::ZeromerAndOnemerOneBead(std::vector<int> const& key, int const keyLen, std::vector<float> const& signal, float& key_zeromer, float& key_onemer)
{
  double zeromer_sum = 0;
  int zero_count = 0;
  double onemer_sum = 0;
  int one_count = 0;

  if (keyLen == 0) {   // default if keyLen = 0
    key_zeromer = 0;
    key_onemer = 0;
    return;
  }

  for (int fnum=0; fnum < keyLen; fnum++)
  {
    // add this measure for any zeromer or a onemer for this bead
    if (key[fnum] == 0) {
      zero_count++;
      zeromer_sum += signal[fnum];
    }
    if (key[fnum] == 1) {
      one_count++;
      onemer_sum  += signal[fnum];
    }
    // check to see for presence of both 1-mer and 0-mer key flows
    if ( (zero_count > 0) && (one_count > 0) ){
      key_zeromer = zeromer_sum/zero_count;
      key_onemer = onemer_sum/one_count;
    }
  }
}

void RegionalizedData::CalculateFirstBlockClonalPenalty(float nuc_flow_frame_width, std::vector<float>& penalty, const int penalty_type)
{
  if (my_beads.ntarget >= my_beads.numLBeads) {
    // all beads will be sampled anyway
    penalty.assign(my_beads.numLBeads, numeric_limits<float>::infinity());
    fprintf(stdout, "region=%d ... valid = 0 out of %d\n", region->index, my_beads.numLBeads);
    return;
  }

  // MonoClonal Sampling scheme
  Clonality *sc = new Clonality();
  {
    std::vector<float> shifted_bkg(NUMFB*time_c.npts(), 0);
    (emptyTraceTracker->GetEmptyTrace (*region))->GetShiftedBkg(my_regions.rp.tshift, time_c, &shifted_bkg[0]);
    sc->SetShiftedBkg(shifted_bkg);
  }

  std::vector<int> flowCount(my_beads.numLBeads, 0);
  
  // start of the nuc rise in units of frames in time_c.t0;
  // ballpark end of the nuc rise in units of frames
  float t_end = time_c.t0 + nuc_flow_frame_width * 2/3;

  // index into part of trace that is important
  size_t t0_ix = time_c.SecondsToIndex(time_c.t0/time_c.frames_per_second);
  size_t t_end_ix = time_c.SecondsToIndex(t_end/time_c.frames_per_second);

  std::vector<float> key_onemer(my_beads.numLBeads, 0);
  std::vector<float> key_zeromer(my_beads.numLBeads, 0);
  std::vector<int> keyLen(my_beads.numLBeads, 0);

  ZeromerAndOnemerAllBeads(*sc, t0_ix, t_end_ix, key_zeromer, key_onemer, keyLen);

  for (int fnum=0; fnum < NUMFB; fnum++)
  {
    std::vector<float> signalInFlow(my_beads.numLBeads, 0);

    for (int ibd=0; ibd < my_beads.numLBeads; ibd++)
    {
      // approximate measure related to signal computed from trace
      // trace is assumed zero'd
      vector<float> trace(time_c.npts(), 0);
      my_trace.AccumulateSignal(&trace[0],ibd,fnum,time_c.npts());
      // signalInFlow[ibd] = sc->Incorporation(t0_ix, t_end_ix, trace);
      signalInFlow[ibd] = sc->Incorporation(t0_ix, t_end_ix, trace, fnum);
    }
    scprint(sc, "region=%d ; flow=%d ;\n", region->index, fnum);

    sc->NormalizeSignal(signalInFlow, key_zeromer, key_onemer);

    // increments penalty with penalties in this flow
    sc->AddClonalPenalty(signalInFlow, keyLen, fnum, flowCount, penalty);

    sc->flush();
  } // end flow iteration

  int valid = 0;
  for  (int ibd=0; ibd < my_beads.numLBeads; ibd++) {
    if ((flowCount[ibd] > 0) &&  std::isfinite(penalty[ibd]) )
    {
      penalty[ibd] /= flowCount[ibd];
      valid++;
    }
    else {
      // a bead has to have at least one working flow to be used
      penalty[ibd] = numeric_limits<float>::infinity();
    }
  }
  fprintf(stdout, "region=%d ... valid = %d out of %d\n", region->index, valid, my_beads.numLBeads);

  delete sc;
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
