/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RegionalizedData.h"

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
  // outputCount=0;
  isBestRegion = false;

  region = NULL;
  emptyTraceTracker = NULL;
  emptytrace = NULL;

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

    if (iend < time_c.npts()) {
      time_c.SetETFFrames(iend);
      time_c.UseETFCompression();
    }

    if (global_defaults.signal_process_control.recompress_tail_raw_trace)
      // Generate emphasis vector object for standard time compression
      std_time_comp_emphasis.SetUpEmphasis(global_defaults.data_control, time_c);
  }
  else
    // check the points that we need
    if (CENSOR_ZERO_EMPHASIS>0)
    {
      EmphasisClass trial_emphasis;

      // assuming that our t_mid_nuc estimation is decent
      // see what the emphasis functions needed for "detailed" results are
      // and assume the "coarse" blank emphasis function will work out fine.
      trial_emphasis.SetUpEmphasis(global_defaults.data_control,time_c);
      trial_emphasis.BuildCurrentEmphasisTable (t0_offset, FINEXEMPHASIS);
      time_c.SetStandardFrames(trial_emphasis.ReportUnusedPoints (CENSOR_THRESHOLD, MIN_CENSOR));
      time_c.UseStandardCompression();
    }

  emphasis_data.SetUpEmphasis(global_defaults.data_control, time_c);
  emphasis_data.BuildCurrentEmphasisTable (t0_offset, FINEXEMPHASIS);

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

void RegionalizedData::GenerateFineEmphasisForStdTimeCompression()
{
  std_time_comp_emphasis.BuildCurrentEmphasisTable (
        GetTypicalMidNucTime (& (my_regions.rp.nuc_shape)),
        FINEXEMPHASIS);
}

void RegionalizedData::SetUpEmphasisForStandardCompression(GlobalDefaultsForBkgModel &global_defaults)
{
}

void RegionalizedData::SetupTimeAndBuffers (
    GlobalDefaultsForBkgModel &global_defaults,float sigma_guess,
    float t_mid_nuc_guess,
    float t0_offset, int flow_block_size,
    int global_flow_max
    )
{
  sigma_start = sigma_guess;
  t_mid_nuc_start = t_mid_nuc_guess;
  t0_frame = t0_offset;
  my_regions.InitRegionParams (t_mid_nuc_start,sigma_start, global_defaults, flow_block_size);

  SetTimeAndEmphasis (global_defaults, t_mid_nuc_guess, t0_offset);

  AllocTraceBuffers( global_flow_max );

  AllocFitBuffers( global_flow_max );
}


void RegionalizedData::AllocFitBuffers( int flow_block_size )
{
  // so we need to make sure these structures match
  my_scratch.Allocate (time_c.npts(),1, flow_block_size);
  my_regions.AllocScratch (time_c.npts(), flow_block_size);
}

void RegionalizedData::AllocTraceBuffers( int flow_block_size )
{
  // now do the traces set up for time compression
  my_trace.Allocate (time_c.npts(),my_beads.numLBeads, flow_block_size);
  my_trace.time_cp = &time_c; // point to the global time compression

}

void RegionalizedData::AddOneFlowToBuffer (GlobalDefaultsForBkgModel &global_defaults, 
                                           FlowBufferInfo & my_flow, int flow)
{
  // keep track of which flow # is in which buffer
  // also keep track of which nucleotide is associated with each flow
  my_flow.SetFlowNdxMap (global_defaults.flow_global.GetNucNdx (flow));
  if (global_defaults.signal_process_control.double_tap_means_zero)
    my_flow.SetDblTapMap (global_defaults.flow_global.IsDoubleTap (flow));
  else
    my_flow.SetDblTapMap(1);  // double-tap is a zero multiplier on amplitude

  // reset parameters for beads when we actually start fitting the beads
}

bool RegionalizedData::LoadOneFlow (Image *img, GlobalDefaultsForBkgModel &global_defaults,  FlowBufferInfo & my_flow, int flow, int flow_block_size)
{
  const RawImage *raw = img->GetImage();
  if (!raw)
  {
    fprintf (stderr, "ERROR: no image data\n");
    return true;
  }

  AddOneFlowToBuffer (global_defaults,my_flow, flow);

  UpdateTracesFromImage (img, my_flow, flow, flow_block_size);

  my_flow.Increment();

  return (false);
}

//prototype GPU execution functions
// UpdateTracesFromImage had to be broken into two function, before and after GPUGenerateBeadTraces.
bool RegionalizedData::PrepareLoadOneFlowGPU (Image *img, 
                                              GlobalDefaultsForBkgModel &global_defaults, FlowBufferInfo & my_flow, int flow)
{
  const RawImage *raw = img->GetImage();
  if (!raw)
  {
    fprintf (stderr, "ERROR: no image data\n");
    return true;
  }

  AddOneFlowToBuffer (global_defaults, my_flow, flow);

  //UpdateTracesFromImage (img, flow);
  //break UpdateTracesFromImage into Pre-GenerateBeadTraces and Post-BeadTraces
  my_trace.SetRawTrace(); // buffers treated as raw traces

  // here we are setup for GPU execution
  return (false);
}

//Prototype GPU second half of UpdateTracesFromImage:
bool RegionalizedData::FinalizeLoadOneFlowGPU ( FlowBufferInfo & my_flow, int flow_block_size )
{
  //break UpdateTracesFromImage into Pre-GenerateBeadTraces and Post-BeadTraces
  float t_mid_nuc =  GetTypicalMidNucTime (&my_regions.rp.nuc_shape);
  float t_offset_beads = my_regions.rp.nuc_shape.sigma;
  my_trace.RezeroBeads (time_c.time_start, t_mid_nuc-t_offset_beads,
                        my_flow.flowBufferWritePos, flow_block_size);
  // calculate average trace across all empty wells in a region for a flow
  // to FileLoadWorker at Image load time, should be able to go here
  // emptyTraceTracker->SetEmptyTracesFromImageForRegion(*img, global_state.pinnedInFlow, flow, global_state.bfmask, *region, t_mid_nuc);
  emptytrace = emptyTraceTracker->GetEmptyTrace (*region);
  // sanity check images are what we think
  assert (emptytrace->imgFrames == my_trace.imgFrames);

  my_flow.Increment();

  return (false);
}


void RegionalizedData::UpdateTracesFromImage (Image *img, FlowBufferInfo &my_flow, int flow, int flow_block_size)
{
  my_trace.SetRawTrace(); // buffers treated as raw traces

  float t_mid_nuc =  GetTypicalMidNucTime (&my_regions.rp.nuc_shape);
  float t_offset_beads = my_regions.rp.nuc_shape.sigma;

#if 1

  // populate bead traces from image file and
  // time-shift traces for uniform start times; compress traces to flows buffer
  my_trace.GenerateAllBeadTrace (region,my_beads,img, my_flow.flowBufferWritePos, flow_block_size,time_c.time_start, t_mid_nuc-t_offset_beads);
  // subtract mean signal in time before flow starts from traces in flows buffer

//  now the RezeroBeads is done in GereateAllBeadTrace
//  my_trace.RezeroBeads (time_c.time_start, t_mid_nuc-t_offset_beads,
//                        my_flow.flowBufferWritePos, flow_block_size);


#else
  //Do it all at once.. generate bead trace and rezero like it is done in the new GPU pipeline
  my_trace.GenerateAllBeadTraceAnRezero(region,my_beads,img, my_flow.flowBufferWritePos, flow_block_size,
                                        time_c.time_start, t_mid_nuc-t_offset_beads);
#endif

  // calculate average trace across all empty wells in a region for a flow
  // to FileLoadWorker at Image load time, should be able to go here
  // emptyTraceTracker->SetEmptyTracesFromImageForRegion(*img, global_state.pinnedInFlow, flow, global_state.bfmask, *region, t_mid_nuc);
  emptytrace = emptyTraceTracker->GetEmptyTrace (*region);

  // sanity check images are what we think
  assert (emptytrace->imgFrames == my_trace.imgFrames);
}


// Trivial fitters

// t_offset_beads = nuc_shape.sigma
// t_offset_empty = 4.0
void RegionalizedData::RezeroTraces (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int flow_buffer_index, int flow_block_size)
{
  emptytrace->RezeroReference (t_start, t_mid_nuc-t_offset_empty, flow_buffer_index);
  my_trace.RezeroBeads (t_start, t_mid_nuc - t_offset_beads, flow_buffer_index, flow_block_size);
}

void RegionalizedData::RezeroTracesAllFlows (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int flow_block_size)
{
  my_trace.RezeroBeadsAllFlows (t_start, t_mid_nuc-t_offset_beads);
  emptytrace->RezeroReferenceAllFlows (t_start, t_mid_nuc - t_offset_empty, flow_block_size);
}

void RegionalizedData::RezeroByCurrentTiming(int flow_block_size)
{
  RezeroTracesAllFlows (time_c.time_start, GetTypicalMidNucTime (& (my_regions.rp.nuc_shape)), my_regions.rp.nuc_shape.sigma, MAGIC_OFFSET_FOR_EMPTY_TRACE, flow_block_size);
}

void RegionalizedData::PickRepresentativeHighQualityWells (float copy_stringency, int min_beads, int max_rank, bool revert_regional_sampling, int flow_block_size)
{
  if (my_beads.isSampled & !revert_regional_sampling){
    my_beads.my_mean_copy_count = my_beads.KeyNormalizeSampledReads ( true, flow_block_size );
    float stringent_filter = my_beads.my_mean_copy_count;
    int num_sampled = my_beads.NumberSampled(); // weird that I need to know this
    if (copy_stringency>0.0f){

      // make filter >stringent< as though the average bead were a certain copy count
      // this replicates a 'bug' that led to stringent filtering and slight performance changes
      stringent_filter = (num_sampled*my_beads.my_mean_copy_count + (my_beads.numLBeads-num_sampled)*copy_stringency)/my_beads.numLBeads;
    }

    // but if we wet the filter to be too stringent, we can lose all beads in a region
    // set a minimum number of beads to succeed with and we can move on
    // technically, I'm setting the minimum >rank< here
    float min_percentage = min_beads;
    min_percentage /= num_sampled;
    min_percentage = 1.0f-min_percentage;
    if (min_percentage<0.0f) min_percentage = 0.0f;

    float max_percentage = max_rank;
    max_percentage /= num_sampled;
    //max_percentage = 1.0f-max_percentage;
    if (max_percentage>1.0f) max_percentage = 1.0f; // maximum can't be more than final point
    if (max_percentage<min_percentage) max_percentage = min_percentage; // really ranks have to be in order to avoid trouble

    float robust_filter = my_beads.FindPercentageCopyCount(min_percentage); // sort is increasing, therefore our percentage must be "out of 1-f"
    float top_filter = my_beads.FindPercentageCopyCount(max_percentage); // increasing

    float low_filter = my_beads.FindPercentageCopyCount(0.25); // make sure
    float high_filter = my_beads.FindPercentageCopyCount(0.75); // make sure
    float final_filter = stringent_filter;
    if(stringent_filter>robust_filter)
      final_filter = robust_filter; // otherwise as long as we have the minimum number we're happy

    // filter: beads within a set of copy counts

    my_beads.LowCopyBeadsAreLowQuality (final_filter, top_filter);

    int test_sampled = my_beads.NumberSampled();
    printf("DEBUGFILTER: %d %d %d %d %f %f %f %f %f %f %f %f\n", region->index, test_sampled, num_sampled, my_beads.numLBeads,
           min_percentage, max_percentage, robust_filter, stringent_filter, final_filter, top_filter,
           low_filter, high_filter);

    // should check if not enough beads and rebuild filter if we fail

    my_beads.KeyNormalizeSampledReads (true, flow_block_size); // force all beads to read the "true key" in the key flows
  } else {
    my_beads.my_mean_copy_count = my_beads.KeyNormalizeReads (false, false, flow_block_size); // retain fitted key values for snr purposes
    float stringent_filter = my_beads.my_mean_copy_count;
    // if we have few enough beads in total that the average isn't going to give us min-beads, we're in trouble anyway so skip.
    // note that this diverges from the >sampled< path
    // because we are using 'above average' only instead of 'good enough' to yield enough beads when sampling
    float top_filter = my_beads.FindPercentageCopyCount(1.0f);
    my_beads.LowCopyBeadsAreLowQuality (stringent_filter, top_filter);
    my_beads.KeyNormalizeReads (true, false, flow_block_size); // force all beads to read the "true key" in the key flows
  }
}



/** Given uninitialized vectors key_zeromer, key_onemer, keyLen
 *  inputs to compute per bead signal: sc, t0_ix, t_end_ix
 *  modify key_zeromer to contain estimates of zeromer signal per bead
 *  and    key_onemer  to contain estimates of onemer signal per bead
 *  and    key_len     to contain the length of the key per bead
 *  (both of length numLBeads).
 */
void RegionalizedData::ZeromerAndOnemerAllBeads(Clonality& sc, size_t const t0_ix, 
                                                size_t const t_end_ix, std::vector<float>& key_zeromer, std::vector<float>& key_onemer,
                                                std::vector<int>& keyLen, int flow_block_size)
{
  for (int ibd=0; ibd < my_beads.numLBeads; ibd++) {

    // find key for this bead
    keyLen[ibd] = 0;
    int key_id = my_beads.key_id[ibd];
    std::vector<int> key(flow_block_size,-1);
    if (key_id >= 0)  // library or TF bead assumed to have key, go get it
      my_beads.SelectKeyFlowValuesFromBeadIdentity (&key[0], NULL, key_id, keyLen[ibd], flow_block_size);

    // find signal for this bead in its key flows
    std::vector<float> signal(keyLen[ibd], 0);
    for (int fnum=0; fnum < keyLen[ibd]; fnum++)
    {
      // approximate measure related to signal computed from trace
      // trace is assumed zero'd
      vector<float> trace(time_c.npts(), 0);
      my_trace.AccumulateSignal(&trace[0],ibd,fnum,time_c.npts(), flow_block_size);
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

void RegionalizedData::CalculateFirstBlockClonalPenalty(float nuc_flow_frame_width, std::vector<float>& penalty, const int penalty_type, int flow_block_size)
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
    std::vector<float> shifted_bkg(flow_block_size*time_c.npts(), 0);
    (emptyTraceTracker->GetEmptyTrace (*region))->GetShiftedBkg(my_regions.rp.tshift, time_c, &shifted_bkg[0], flow_block_size);
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

  ZeromerAndOnemerAllBeads(*sc, t0_ix, t_end_ix, key_zeromer, key_onemer, keyLen, flow_block_size);

  for (int fnum=0; fnum < flow_block_size; fnum++)
  {
    std::vector<float> signalInFlow(my_beads.numLBeads, 0);

    for (int ibd=0; ibd < my_beads.numLBeads; ibd++)
    {
      // approximate measure related to signal computed from trace
      // trace is assumed zero'd
      vector<float> trace(time_c.npts(), 0);
      my_trace.AccumulateSignal(&trace[0],ibd,fnum,time_c.npts(), flow_block_size);
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
  float pct = 100 * valid/float(my_beads.numLBeads);
  fprintf(stdout, "region=%d ... valid = %d/%d = %.2g%%\n", region->index, valid, my_beads.numLBeads,pct);

  delete sc;
}


RegionalizedData::RegionalizedData( const CommandLineOpts * inception_state ) :
  my_regions( inception_state )
{
  NoData();
  region_nSamples = inception_state->bkg_control.pest_control.bkgDebug_nSamples;
  sampleIndex_assigned = false;
}

RegionalizedData::RegionalizedData()
{
  NoData();
  sampleIndex_assigned = false;
}

RegionalizedData::~RegionalizedData()
{
}





void RegionalizedData::DumpEmptyTrace (FILE *my_fp, int flow_block_size)
{
  assert (emptyTraceTracker != NULL);   //sanity
  if (region!=NULL)
  {
    (emptyTraceTracker->GetEmptyTrace (*region))->
        DumpEmptyTrace (my_fp,region->col,region->row, flow_block_size);
  }
}



bool RegionalizedData::isRegionCenter(int ibd)
{
  //BeadParams *p= &my_beads.params_nn[ibd];
  //bool isCenter = (p->x == (region->w/2) && p->y==(region->h/2)) ? true: false;
  //if (isCenter) cout << "outputCount=" << ++outputCount << " ibd=" << ibd << " x=" << p->x << " y=" << p->y << " isCenter=" << isCenter<< " w=" << region->w << " h=" << region->h<< endl << flush;
  int nLiveBeads = GetNumLiveBeads();
  bool isCenter = (ibd == (nLiveBeads/2)) ? true:false;
  return (isCenter);
}


bool RegionalizedData::isLinePoint(int ibd, int nSamples)
{
  int nLiveBeads = GetNumLiveBeads();
  if (nSamples>0 && nLiveBeads>=nSamples) {
    int dx = nLiveBeads / (nSamples+1);
    if (dx<1)
      dx = 1;
    bool isSample = ((ibd+1)%dx==0) ? true:false;
    return (isSample);
  }
  else
    return (false);
}


bool RegionalizedData::isGridPoint(int ibd, int nSamples)
{
  int nLiveBeads = GetNumLiveBeads();
  if (nSamples>0 && nLiveBeads>=nSamples) {
    float bx = sqrt(nLiveBeads);
    float sx = sqrt(nSamples);
    int iy = ibd/bx;
    int ix = ibd - iy*bx;
    int dy = bx>=(sx+1) ? bx/(sx+1) : 1;
    return ((++iy%dy==0 && ++ix%dy==0) ? true:false) ;
  }
  else
    return (false);
}


int RegionalizedData::get_sampleIndex(int ibd) {
  try {
    return(regionSampleIndex[ibd]);
  }
  catch (...) {
    return (-1);
  }
}


int RegionalizedData::assign_sampleIndex(void)
{
  if (sampleIndex_assigned)
    return (nAssigned);
  if (betterUseEvenSamples())
    nAssigned = assign_sampleIndex_even();
  else
    nAssigned = assign_sampleIndex_even_random();
  //nAssigned = assign_sampleIndex_random(); // slower than assign_sampleIndex_even_random(), and may fall into infinite loop?
  //nAssigned = assign_sampleIndex_random_shuffle(); // slower than assign_sampleIndex_even_random(), and may not be reproducible
  return (nAssigned);
}


int RegionalizedData::assign_sampleIndex_even(void)
{
  nAssigned = 0;
  if (sampleIndex_assigned) {
    cout << "assign_sampleIndex_even...resetting regionSampleIndex" << endl << flush;
    //regionSampleIndex.erase ( regionSampleIndex.begin(), regionSampleIndex.end() );
    //regionSampleIndex.clear(); // onlcy changes contents to 0, not good??
    // what's the easiest way to clear the map? clear() still leaves it accessible, and won't work for us
    std::map<int,int> tmpSampleIndex; // does this work?? perhaps it never reaches here
    regionSampleIndex = tmpSampleIndex;
  }
  int nLiveBeads = GetNumLiveBeads();
  if (region_nSamples>nLiveBeads)
    region_nSamples = nLiveBeads;
  if (region_nSamples>0) {
    float scale = float(nLiveBeads) / region_nSamples;
    for (int n=0; n<region_nSamples; n++) {
      int ibd = n*scale;
      regionSampleIndex[ibd] = nAssigned++;
    }
  }
  sampleIndex_assigned = true; // must be true, otherwise this function could be called again and again in an infinite loop
  return (nAssigned);
}


int RegionalizedData::assign_sampleIndex_even_random(void)
{
  if (betterUseEvenSamples()) return (assign_sampleIndex_even()); // even() is good enough if region_nSamples>nBeads*fraction

  nAssigned = 0;
  if (sampleIndex_assigned) {
    cout << "assign_sampleIndex_even_random...resetting regionSampleIndex" << endl << flush;
    //regionSampleIndex.erase ( regionSampleIndex.begin(), regionSampleIndex.end() );
    //regionSampleIndex.clear(); // onlcy changes contents to 0, not good??
    // what's the easiest way to clear the map? clear() still leaves it accessible, and won't work for us
    std::map<int,int> tmpSampleIndex; // does this work?? perhaps it never reaches here
    regionSampleIndex = tmpSampleIndex;
  }
  int nLiveBeads = GetNumLiveBeads();
  if (region_nSamples>nLiveBeads)
    region_nSamples = nLiveBeads;
  if (region_nSamples>0 && nLiveBeads>=region_nSamples) {
    float scale = float(nLiveBeads) / region_nSamples;
    int range = int(scale);
    if (range>1) srand(42); //seed so that it is reproducible, same as in in Separator.cpp
    for (int n=0; n<region_nSamples; n++) {
      int ibd = n*scale;
      ibd += random_in_range(range); // 0 to range-1
      regionSampleIndex[ibd] = nAssigned++;
    }
  }
  sampleIndex_assigned = true; // must be true, otherwise this function could be called again and again in an infinite loop
  return (nAssigned);
}


int RegionalizedData::assign_sampleIndex_random(void)
{
  // please make assign_sampleIndex_random more robust in finding unique numbers before removing the following line
  if (betterUseEvenSamples()) return (assign_sampleIndex_even()); // avoid spending too much time in the while loop for unique random numbers

  nAssigned = 0;
  if (sampleIndex_assigned) {
    cout << "assign_sampleIndex_random...resetting regionSampleIndex" << endl << flush;
    //regionSampleIndex.erase ( regionSampleIndex.begin(), regionSampleIndex.end() );
    //regionSampleIndex.clear(); // onlcy changes contents to 0, not good??
    // what's the easiest way to clear the map? clear() still leaves it accessible, and won't work for us
    std::map<int,int> tmpSampleIndex; // does this work?? perhaps it never reaches here
    regionSampleIndex = tmpSampleIndex;
  }
  int nLiveBeads = GetNumLiveBeads();
  if (region_nSamples>nLiveBeads)
    region_nSamples = nLiveBeads;
  if (region_nSamples>0 && nLiveBeads>=region_nSamples) {
    srand(42); //seed so that it is reproducible, same as in in Separator.cpp
    while (nAssigned<region_nSamples) {
      int ibd = random_in_range(nLiveBeads);
      try {
        if (regionSampleIndex[ibd]<=0) // always 0??
          regionSampleIndex[ibd] = nAssigned++; // in case a negative (invalid) number is assigned, which shouldn't happen
        //cout << " ibd=" << ibd << " nAssigned=" << nAssigned << endl << flush;
      }
      catch (...) { // not assigned yet
        regionSampleIndex[ibd] = nAssigned++;
        //cout << " ibd=" << ibd << " nAssigned=" << nAssigned << endl << flush;
      }
    }
  }
  sampleIndex_assigned = true; // must be true, otherwise this function could be called again and again in an infinite loop
  return (nAssigned);
}


int RegionalizedData::assign_sampleIndex_random_shuffle(void) 
{
  nAssigned = 0;
  if (sampleIndex_assigned) {
    cout << "assign_sampleIndex_random_shuffle...resetting regionSampleIndex" << endl << flush;
    //regionSampleIndex.erase ( regionSampleIndex.begin(), regionSampleIndex.end() );
    //regionSampleIndex.clear(); // onlcy changes contents to 0, not good??
    // what's the easiest way to clear the map? clear() still leaves it accessible, and won't work for us
    std::map<int,int> tmpSampleIndex; // does this work?? perhaps it never reaches here
    regionSampleIndex = tmpSampleIndex;
  }
  int nLiveBeads = GetNumLiveBeads();
  if (region_nSamples>nLiveBeads)
    region_nSamples = nLiveBeads;
  if (region_nSamples>0 && nLiveBeads>=region_nSamples) {
    srand(42); //try but not sure this makes it reproducible, due to the std::random_shuffle() used in my_random_shuffle
    vector<int> beads;
    my_random_shuffle(beads, nLiveBeads); // shuffle to make the samples unique
    //cout << "beads.size()=" << beads.size() << " for nLiveBeads=" << nLiveBeads << endl << flush;
    assert((int)beads.size()==nLiveBeads);
    for (int n=0; n<region_nSamples; n++) {
      try {
        int ibd = beads[n];
        assert(ibd<nLiveBeads);
        regionSampleIndex[ibd] = nAssigned++;
        //cout << "n=" << n << " ibd=" << ibd << " nAssigned=" << nAssigned << endl << flush;
      }
      catch (...) {
        cerr << "error at n=" << n << endl << flush;
        throw 20;
      }
    }
  }
  sampleIndex_assigned = true; // must be true, otherwise this function could be called again and again in an infinite loop
  return (nAssigned);
}


void RegionalizedData::my_random_shuffle(vector<int>&cards, unsigned int nCards)
{
  if (nCards>0)
  {
    if (cards.size() != nCards)
    {
      cards.resize(nCards);
      for (unsigned int i=0; i<nCards; i++)
        cards[i] = i;
    }
    std::random_shuffle ( cards.begin(), cards.end());
  }
}


bool RegionalizedData::isRegionSample(int ibd)
{
  try {
    if (regionSampleIndex[ibd]>=0)
      return (true);
    else
      return (false);
  }
  catch (...) {
    return (false);
  }
}

void RegionalizedData::regParamsToJson(Json::Value &regparams_json)
{
  stringstream regId;
  regId << "row_" << region->row << "_col_" << region->col;
  Json::Value regP(Json::objectValue);
  my_regions.rp.ToJson(regP);
  regparams_json[regId.str()] = regP;
}

void RegionalizedData::LoadRestartRegParamsFromJson(const Json::Value &regparams_json) {
  stringstream regId;
  regId << "row_" << region->row << "_col_" << region->col;
  
  Json::Value::iterator it = regparams_json.begin();
  while(it != regparams_json.end()) {
    Json::Value p = (*it++);
    if (p.isMember(regId.str()))
      my_regions.rp.FromJson(p[regId.str()]);
  }
}
