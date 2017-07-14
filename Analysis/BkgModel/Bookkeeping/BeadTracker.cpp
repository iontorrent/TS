/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iomanip>
#include "BeadTracker.h"
#include "ClonalFilter/mixed.h"
#include "ImageLoader.h"
#include "MathUtil.h"
#include "LevMarState.h"
#include "GlobalWriter.h"

using namespace std;

// barcode tracking for initial training, now that barcodes are relevant and important


float BeadTracker::BarcodeNormalizeOneBead(int ibd, bool overwrite_barcode, int flow_block_size){
  float normalizer = 1.0f;
  BeadParams  *p  = &params_nn[ibd];
  bound_params *pl = &params_low;
  bound_params *ph = &params_high;

// handles unidentified beads as well
    normalizer = barcode_info.ComputeNormalizerBarcode(p->Ampl,barcode_info.barcode_id[ibd], flow_block_size,0);
  p->Copies *= normalizer;
  MultiplyVectorByScalar (p->Ampl,1.0f/normalizer,flow_block_size);
  if (overwrite_barcode)
    barcode_info.SetBarcodeFlows (p->Ampl,barcode_info.barcode_id[ibd]);

  p->ApplyLowerBound (pl, flow_block_size);
  p->ApplyUpperBound (ph, flow_block_size);
  p->ApplyAmplitudeDrivenKmultLimit(flow_block_size, 2.0f);
  return (p->Copies);
}


void BeadTracker::AssignBarcodeState(bool do_all, float basic_threshold, float tie_threshold, int flow_block_size, int flow_block_start){

  barcode_info.ResetCounts();
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (Sampled(ibd) or do_all){
      barcode_info.barcode_id[ibd]=barcode_info.ClassifyOneBead(params_nn[ibd].Ampl,basic_threshold,tie_threshold, flow_block_size, flow_block_start);
    }
  }
}

// bead parameter initial values and box constraints

BeadTracker::BeadTracker()
{
  my_mean_copy_count = 0.0; // don't know this yet
  DEBUG_BEAD = 0;
  numLBeads=0;
  numLBadKey=0;
  numSeqListItems = 0;
  max_emphasis = 0;
  regionindex = -1;
  isSampled = false;
  ntarget = 0;
  ignoreQuality = false;
  doAllBeads = false;

  // stop: should be own object
  stop_alpha = 1/12.0f;
  stop_maxvar = 9;
  stop_threshold = 0.075;

}

void BeadTracker::InitHighBeadParams()
{
  params_high.SetBeadStandardHigh ();
}

void BeadTracker::InitLowBeadParams(float AmplLowerLimit)
{
  params_low.SetBeadStandardLow (AmplLowerLimit);
}

void BeadTracker::InitModelBeadParams()
{
  params_nn.resize(numLBeads);
  all_status.resize(numLBeads);
  for (int i=0; i<numLBeads; i++)
  {
    // associate status with bead
    params_nn[i].my_state = &(all_status[i]);
  }
  for (int i=0;i<numLBeads;i++)
  {
    params_nn[i].SetBeadStandardValue ();
    params_nn[i].trace_ndx = i; // index into trace object
  }
}

void BeadTracker::DefineKeySequence (SequenceItem *seq_list, int number_Keys)
{
  seqList.clear();
  copy(seq_list, seq_list+number_Keys, back_inserter(seqList));
  numSeqListItems = number_Keys;
  key_id.resize(numLBeads); // indexing into this item
}

void BeadTracker::SelectDebugBead(int seed)
{
  // we do threaded initialization, so trusting in rand is not reproducible
  // fake "randomness" by using large numbers
  if (numLBeads>0)
    DEBUG_BEAD = (seed+SMALL_PRIME)*LARGE_PRIME % numLBeads;
  else
    DEBUG_BEAD = 0;
}

void BeadTracker::InitBeadParams(float AmplLowerLimit)
{
  // no spatial information here
  // just boot up parameters
  SelectDebugBead(regionindex+1);
  InitHighBeadParams();
  InitLowBeadParams(AmplLowerLimit);
  InitModelBeadParams();
}

void BeadTracker::IgnoreQuality ()
{ 
  ignoreQuality = true;
  if (ignoreQuality)
    for (size_t i=0; i<high_quality.size(); i++)
      high_quality[i] = true;
}


void BeadTracker::InitBeadList (Mask *bfMask, Region *region, bool nokey, SequenceItem *_seqList, 
    int _numSeq, const std::set<int>& sample, float AmplLowerLimit)
{
  MaskType processMask = MaskLive;
  if (nokey) {         // actually, we don't yet ignore the keys...
    doAllBeads = true;
    processMask = MaskAll; // see also ForceAll()
  }

  // Number of beads to process
  numLBeads =  bfMask->GetCount (processMask,*region);
  regionindex = region->index;

  InitBeadParams(AmplLowerLimit);
  DefineKeySequence (_seqList, _numSeq);
  high_quality.resize(numLBeads, true);
  sampled.resize(numLBeads, true);
  decay_ssq.resize(numLBeads, 5.0f);
  last_amplitude.resize(numLBeads, 0.5f);
  barcode_info.barcode_id.resize(numLBeads,-1); // no barcode assigned to any beaads

  // makes spatially organized data if needed
  BuildBeadMap (region,bfMask,processMask);
  // must be done after generating spatial data
  InitRandomSample (*bfMask, *region, sample);


}



//this function try to estimate a multiplication factor to determine the number
// of copies corresponding to a positive flow.
//input is an array of observed values for a specific bead over initial flows
//  keyval is an array of integers with 1 indicating incorporating flow
//  len is the length of flows to do the processes.
// effectively optimizing sum((y-a*x)^2) y = observed, x = keyval, a = normalizing constant
float ComputeNormalizerKeyFlows (const float *observed, const int *keyval, int len)
{
  float wsum = 0.0001f;
  float wdenom = 0.0001f; // avoid dividing by zero in the bad case
  for (int i=0; i<len; i++)
  {
    wsum += observed[i]*keyval[i];  // works for keyval = 0
    wdenom += keyval[i]*keyval[i];  // works for keyval >1
  }
  return (wsum/wdenom);
}



//this function gets the  observed values for the initial flows of the key in *observed
//for a number of flows = Keylen
// and then assume crudly that there are actually 3 active flows covering the key out of
//the initial 7 flow. it takes the mean of those flows and then fill the corresponding
//positions in keyval with 1 indicating corresponding flows triggered a 1'mer.
//The function will be deprecated soon.
void BeadTracker::SelectKeyFlowValues (int *keyval,float *observed, int keyLen)
{
  float mean_key=0.0f;
  for (int i=0;i<keyLen;i++)
    mean_key += observed[i];

  mean_key /= 3.0f;  // @TODO crude classifier assuming 3 live in first 7
  for (int i=0;i<keyLen;i++)
  {
    if (observed[i] > (mean_key/2.0f))
      keyval[i] = 1;
    else
      keyval[i] = 0;
  }
}

void BeadTracker::SelectKeyFlowValuesFromBeadIdentity (int *keyval, float *observed, int my_key_id, int &keyLen, int flow_block_size)
{
  if (my_key_id<0)
    SelectKeyFlowValues (keyval,observed,keyLen);
  else
  {
    // lookup key id in seqlist

    keyLen = seqList[my_key_id].usableKeyFlows; // return the correct number of flows
    if (keyLen>flow_block_size)
      keyLen = flow_block_size;  // might have a very long key(!)
    for (int i=0; i<keyLen; i++)
      keyval[i] = seqList[my_key_id].Ionogram[i];
  }
}

void SetKeyFlows (float *observed, int *keyval, int len)
{
  for (int i=0; i<len; i++)
    observed[i] = keyval[i];
}


float BeadTracker::KeyNormalizeOneBead (int ibd, bool overwrite_key, int flow_block_size)
{
  int   key_val[flow_block_size];
  float normalizer = 1.0f;
  BeadParams  *p  = &params_nn[ibd];
  bound_params *pl = &params_low;
  bound_params *ph = &params_high;

  bool oldkeynorm = false;
  int key_len;
  key_len = KEY_LEN; // default 7 flows in case we don't know anything
  if (oldkeynorm)
  {
    SelectKeyFlowValues (key_val, p->Ampl, key_len);
  }
  else
  {
    // update key_len from sequence identity if known
    SelectKeyFlowValuesFromBeadIdentity (key_val,p->Ampl, key_id[ibd], key_len, flow_block_size);
  }
  normalizer=ComputeNormalizerKeyFlows (p->Ampl,key_val,key_len);

  //cout <<ibd << " p-Copies before :" <<p->Copies ;
  p->Copies *= normalizer;
  // scale to right signal
  //cout <<" and after =" << p->Copies <<"\n";
  MultiplyVectorByScalar (p->Ampl,1.0f/normalizer,flow_block_size);

  // set to exact values in key flows, overriding real levels
  if (overwrite_key)
    SetKeyFlows (p->Ampl,key_val,key_len);

  p->ApplyLowerBound (pl, flow_block_size);
  p->ApplyUpperBound (ph, flow_block_size);
  p->ApplyAmplitudeDrivenKmultLimit(flow_block_size, 2.0f);
  return (p->Copies);
}


// Reads key identity from the separator and exploit the lookup table
// in particular, we need to handle 0,1,2 mers for possible signal types
// when using arbitrary keys
// we use this to set a good scale/copy number for the beads

float BeadTracker::KeyNormalizeReads(bool overwrite_key, bool sampled_only, int flow_block_size)
{

  // figure out what to expect in the key flows.
  // calculate key signal (S), set the key flows to exactly 0 and 1 mers, and
  // then scale everything else based on S.
  float meanCopyCount = 0.0;
  int goodBeadCount=0;
  // cout << "Number of live beads:" << numLBeads  <<"\n";
  for (int ibd=0;ibd < numLBeads;ibd++)
  {

    //  if(bfMask->Match( params_nn[ibd].x,params_nn[ibd].y, MaskLive  , MATCH_ANY)){
    meanCopyCount += KeyNormalizeOneBead (ibd,overwrite_key, flow_block_size);
    goodBeadCount++;
    // }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  meanCopyCount /= goodBeadCount;
  return (meanCopyCount);
}

float BeadTracker::KeyNormalizeSampledReads(bool overwrite_key, int flow_block_size)
{

  // figure out what to expect in the key flows.
  // calculate key signal (S), set the key flows to exactly 0 and 1 mers, and
  // then scale everything else based on S.
  float meanCopyCount = 0.0;
  int goodBeadCount=0;
  // cout << "Number of live beads:" << numLBeads  <<"\n";
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if( Sampled (ibd) ){
          meanCopyCount += KeyNormalizeOneBead (ibd,overwrite_key, flow_block_size);
          goodBeadCount++;
      }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  if (goodBeadCount > 0)
    meanCopyCount /= goodBeadCount;
  else
    meanCopyCount = 2.0f; // does this matter?

  return (meanCopyCount);
}

void BeadTracker::SetCopyCountOnUnSampledBeads(int flow_block_size){
  float high_copy_count = CopiesFromSampledReads ( );

  // whatever count I got is going to be on the high side because of filtering
  int final_samples = 0;
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (Sampled(ibd)){
      // do nothing because copy-count should be well-set already here
      //params_nn[ibd].Copies = high_copy_count; // guessing the mean of good wells is better than guessing nothing?
        final_samples++;
    } else {
      params_nn[ibd].Copies = high_copy_count; // guessing the mean of good wells is better than guessing nothing?
    }
  }
}

float BeadTracker::etbRFromSampledReads()
{
  float mean_etbR = 0.0;
  int goodBeadCount=0;
  // cout << "Number of live beads:" << numLBeads  <<"\n";
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if( Sampled (ibd) ){
          mean_etbR += params_nn[ibd].R;
          goodBeadCount++;
      }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  if (goodBeadCount > 0)
    mean_etbR /= goodBeadCount;
  else
    mean_etbR = 0.7f; // does this matter?

  return (mean_etbR);
}


float BeadTracker::CopiesFromReads()
{
  float mean_copy = 0.0;
  int goodBeadCount=0;
  // cout << "Number of live beads:" << numLBeads  <<"\n";
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    {
          mean_copy += params_nn[ibd].Copies;
          goodBeadCount++;
      }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  if (goodBeadCount > 0)
    mean_copy /= goodBeadCount;
  else
    mean_copy = 2.0f; // does this matter?

  return (mean_copy);
}

float BeadTracker::CopiesFromSampledReads()
{
  float mean_copy= 0.0;
  int goodBeadCount=0;
  // cout << "Number of live beads:" << numLBeads  <<"\n";
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if( Sampled (ibd) ){
          mean_copy += params_nn[ibd].Copies;
          goodBeadCount++;
      }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  if (goodBeadCount > 0)
    mean_copy /= goodBeadCount;
  else
    mean_copy = 2.0f; // does this matter?

  return (mean_copy);
}


float BeadTracker::etbRFromReads()
{
  float mean_etbR = 0.0;
  int goodBeadCount=0;
  // cout << "Number of live beads:" << numLBeads  <<"\n";
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    {
          mean_etbR += params_nn[ibd].R;
          goodBeadCount++;
      }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  if (goodBeadCount > 0)
    mean_etbR /= goodBeadCount;
  else
    mean_etbR = 0.7f; // does this matter?

  return (mean_etbR);
}
void BeadTracker::SetBufferingRatioOnUnSampledBeads(){
  float typical_etbR = etbRFromSampledReads ( );
  // whatever count I got is going to be on the high side because of filtering
  int final_samples = 0;
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (Sampled(ibd)){
      // do nothing because etbR should be well-set already here
      //params_nn[ibd].R = typical_etbR; // guessing the mean of good wells is better than guessing nothing?
      final_samples++;
    } else {
      params_nn[ibd].R = typical_etbR; // guessing the mean of good wells is better than guessing nothing?
    }
  }
}


void BeadTracker::LowCopyBeadsAreLowQuality (float min_copy_count, float max_copy_count)
{
  if (ignoreQuality)
    return;

  int cp_filter = 0;
  // only use the top amplitude signal beads for region-wide parameter fitting
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if ((params_nn[ibd].Copies < min_copy_count) or params_nn[ibd].Copies>max_copy_count)
    {
      high_quality[ibd] = false; // reject beads that don't match
      cp_filter++;
    }
  }
//  printf("Region %d fit exclude %d beads of %d for low copy count below %f\n", regionindex, cp_filter, numLBeads, mean_copy_count);
}

void BeadTracker::CorruptedBeadsAreLowQuality ()
{
  if (ignoreQuality)
    return;

  for (int ibd=0; ibd<numLBeads; ibd++)
  {
    if (params_nn[ibd].my_state->corrupt or params_nn[ibd].my_state->pinned)
      high_quality[ibd] = false; // reject beads that are sad in any way
  }
}

void BeadTracker::TypicalBeadParams(BeadParams *p)
{
  int count=1; // start with a "standard" bead in case we don't have much
  p->SetBeadStandardValue();
  for (int ibd=0; ibd<numLBeads; ibd++)
  {
    if (high_quality[ibd]) // should be a bunch!
    {
      p->Copies += params_nn[ibd].Copies;
      p->gain   += params_nn[ibd].gain;
      p->dmult  += params_nn[ibd].dmult;
      p->R      += params_nn[ibd].R;
      ++count;
    }
  }
  p->Copies/=count;
  p->gain /= count;
  p->dmult /= count;
  p->R /= count;
}


void BeadTracker::ResetFlowParams (int bufNum)
{
  for (int i=0;i<numLBeads;i++)
  {
    params_nn[i].Ampl[bufNum] = 0.001f;
    params_nn[i].kmult[bufNum] = 1.0f;
  }
}

void BeadTracker::ResetLocalBeadParams()
{
  for (int i=0; i<numLBeads; i++)
  {
    params_nn[i].SetStandardFlow();
  }
}

void BeadTracker::CompensateAmplitudeForEmptyWellNormalization(float *my_scale_buffer,
    int flow_block_size )
{
  // my scale buffer contains a rescaling of the data per flow per bead and is therefore a memory hog
  // we'll avoid using this if at all possible.
  for (int ibd=0;ibd <numLBeads;ibd++)
  {
    for (int fnum=0; fnum<flow_block_size; fnum++)
      params_nn[ibd].Ampl[fnum] *= my_scale_buffer[ibd*flow_block_size+fnum];
  }
}

//for all bead
void BeadTracker::AssignEmphasisForAllBeads (int _max_emphasis)
{
  max_emphasis = _max_emphasis;
}

void BeadTracker::AdjustForCopyNumber(vector<float>& ampl, const BeadParams& p, const vector<float>& copy_multiplier)
{
  size_t num_flows = ampl.size();
  for(size_t flow=0; flow<num_flows; ++flow)
    ampl[flow] = p.Ampl[flow] * p.Copies * copy_multiplier[flow];
}

void BeadTracker::ComputeKeyNorm(const vector<int>& keyIonogram, const vector<float>& copy_multiplier)
{
  vector<float> ampl(keyIonogram.size());
  for(int bead=0; bead<numLBeads; ++bead){
    BeadParams& p = params_nn[bead];
    AdjustForCopyNumber(ampl, p, copy_multiplier);
    p.my_state->key_norm = ComputeNormalizerKeyFlows(&ampl[0], &keyIonogram[0], keyIonogram.size());
  }
}

void BeadTracker::CheckKey(const vector<float>& copy_multiplier)
{
  // Flag beads with bad key:
  vector<float> ampl(seqList[LIBKEYNDX].usableKeyFlows);
  vector<float> nrm (seqList[LIBKEYNDX].usableKeyFlows);
  for (int bead=0; bead<numLBeads; ++bead)
  {
    BeadParams& p = params_nn[bead];
    AdjustForCopyNumber(ampl, p, copy_multiplier);
    transform (ampl.begin(), ampl.end(), nrm.begin(), bind2nd (divides<float>(),p.my_state->key_norm));
    if (not key_is_good (nrm.begin(), seqList[LIBKEYNDX].Ionogram, seqList[LIBKEYNDX].Ionogram+seqList[LIBKEYNDX].usableKeyFlows)){
      p.my_state->bad_read = true;
      ++numLBadKey;
    }
  }
}

void BeadTracker::UpdateClonalFilter (int flow, const vector<float>& copy_multiplier, const PolyclonalFilterOpts & opts, int flow_block_size, int flow_begin_real_index )
{
  // first block only:
  if(flow_begin_real_index == 0 && flow == flow_block_size-1){
    vector<int> keyIonogram(seqList[LIBKEYNDX].Ionogram, seqList[LIBKEYNDX].Ionogram+seqList[LIBKEYNDX].usableKeyFlows);
    ComputeKeyNorm(keyIonogram, copy_multiplier);
    CheckKey(copy_multiplier);
  }

  // all blocks used by clonal filter:
  // Figure out the real indices that bound this block.
  int block_begin = flow_begin_real_index;
  int block_end   = flow_begin_real_index + flow_block_size;

  // Does this block overlap the requested "mixed" range?
  if ( opts.mixed_first_flow < block_end and opts.mixed_last_flow >= block_begin )
    UpdatePPFSSQ (flow, copy_multiplier, opts, flow_block_size, flow_begin_real_index );

  // last block used by clonal filter:
  if ( opts.mixed_last_flow >= block_begin and opts.mixed_last_flow <= block_end )
    FinishClonalFilter( opts );
}

void BeadTracker::UpdatePPFSSQ (int flow, const vector<float>& copy_multiplier, 
    const PolyclonalFilterOpts & opts, int flow_block_size, int flow_begin_real_index )
{
  vector<float> ampl(flow_block_size);
  for (int bead=0; bead<numLBeads; ++bead)
  {
    BeadParams& p = params_nn[bead];
    AdjustForCopyNumber(ampl, p, copy_multiplier);
    transform (ampl.begin(), ampl.end(), ampl.begin(), bind2nd (divides<float>(),p.my_state->key_norm));

    // Update ppf and ssq:
    // We want to do this calculation on any flows in the current flow block that
    // are on [opts.mixed_first_flow and opts.mixed_last_flow).

    // Figure out the real indices that bound this block.
    int block_begin = flow_begin_real_index;
    int block_end   = flow_begin_real_index + flow_block_size;

    // Then get the real indices that we want to calculate.
    int calc_real_begin = max( opts.mixed_first_flow, block_begin );
    int calc_real_end   = min( opts.mixed_last_flow, block_end );

    // Finally, get flow numbers within the local block.
    int calc_begin = calc_real_begin - flow_begin_real_index;
    int calc_end   = calc_real_end   - flow_begin_real_index;

    for (int i=calc_begin; i<calc_end; ++i)
    {
      if (ampl[i] > mixed_pos_threshold())
        p.my_state->ppf += 1;
      float x = ampl[i] - round (ampl[i]);
      p.my_state->ssq += x * x;
    }
    // Flag beads with infinite signal:
    if (not all_finite (p.Ampl, p.Ampl+flow_block_size))
      p.my_state->bad_read = true;
  }
}

void BeadTracker::FinishClonalFilter( const PolyclonalFilterOpts & opts )
{
  for (int bead=0; bead<numLBeads; ++bead)
  {
    BeadParams& p = params_nn[bead];
    p.my_state->ppf /= (opts.mixed_last_flow - opts.mixed_first_flow);
  }
}
int BeadTracker::NumHighPPF() const
{
  int numHigh = 0;
  for (int bead=0; bead<numLBeads; ++bead)
  {
    const BeadParams& p = params_nn[bead];
    if (p.my_state->ppf > mixed_ppf_cutoff())
      ++numHigh;
  }
  return numHigh;
}

int BeadTracker::NumPolyclonal() const
{
  int numPolyclonal = 0;
  for (int bead=0; bead<numLBeads; ++bead)
  {
    const BeadParams& p = params_nn[bead];
    if (not p.my_state->clonal_read)
      ++numPolyclonal;
  }
  return numPolyclonal;
}

int BeadTracker::NumBadKey() const
{
  return numLBadKey;
}

int BeadTracker::NumHighQuality(){
  int num_high_quality = 0;
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (high_quality[ibd])
      num_high_quality++;
  }
  return(num_high_quality);
}

void BeadTracker::ZeroOutPins (Region *region, const Mask *bfmask, const PinnedInFlow& pinnedInFlow, int flow, int iFlowBuffer)

{
  // Zero out beads that pin in this flow or earlier flows

  for (int bead=0; bead<numLBeads; ++bead)
  {
    BeadParams& p = params_nn[bead];
    int ix = bfmask->ToIndex (p.y + region->row, p.x + region->col);

    // set bead Amplitude to zero in this flow so no basecalling occurs
    if (pinnedInFlow.IsPinned (flow, ix))
    {
      p.Ampl[iFlowBuffer] = 0.0f;
      ndx_map[p.y*region->w+p.x] = -1; // ignore this neighbor from now on to avoid contaminating xtalk
      p.my_state->pinned = true;
    }
  }
}

float BeadTracker::FindMeanDmult (bool skip_beads)
{
  float mean_dmult = 0.0f;
  float num_checked = 0.0001f;
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if (BeadIncluded(ibd, skip_beads))
    {
      mean_dmult += params_nn[ibd].dmult;
      num_checked += 1.0f;
    }
  }
  mean_dmult /= num_checked;
  return (mean_dmult);
}



void BeadTracker::RescaleDmult (float scale)
{
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    params_nn[ibd].dmult /= scale;
  }
}

void BeadTracker::RescaleDmultFromSample (float scale)
{
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if (Sampled(ibd))
      params_nn[ibd].dmult /= scale;
  }
}

float BeadTracker::CenterDmult (bool skip_beads)
{
  float mean_dmult = FindMeanDmult (skip_beads);
  RescaleDmult (mean_dmult);
  return (mean_dmult);
}

bool BeadTracker::UpdateSTOP(int ibd, int flow_block_size){
  // update this bead's "STOP" tracker
  bool tripped_flag = false;
  for (int fnum=0; fnum<flow_block_size; fnum++){
    float delta = params_nn[ibd].Ampl[fnum]-last_amplitude[ibd];
    last_amplitude[ibd] = params_nn[ibd].Ampl[fnum];
    delta *= delta; // was there a big difference in signals?

    if (delta>stop_maxvar) // large signals don't extend the decay window
      delta = stop_maxvar;
    decay_ssq[ibd] = (1.0f-stop_alpha)*decay_ssq[ibd]+stop_alpha*delta; // exponential tracker
    if (decay_ssq[ibd]<stop_threshold){  // this bead now officially low variability - likely indicates stopped producing useful signal
      if (high_quality[ibd]==true)
        tripped_flag = true;
      high_quality[ibd]=false; // stop using this bead for regional parameter fitting
    }
  }
  return(tripped_flag);
}

void BeadTracker::CenterKmult( float *mean_kmult, bool skip_beads, int *flow_ndx_map, int flow_block_size){
  float kmult_count[NUMNUC];
  for (int nnuc=0; nnuc<NUMNUC; nnuc++){
    mean_kmult[nnuc]=1.0f; // weak prior here
    kmult_count[nnuc] = 1.0f;
  }
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (BeadIncluded(ibd, skip_beads)){
      for (int fnum=0; fnum<flow_block_size; fnum++){
        int tnuc = flow_ndx_map[fnum];
        float wt = 0.001;
        if (params_nn[ibd].Ampl[fnum]>0.5) // if potentially accurate as fitted value = should be 'informative value' (copies?)
           wt = 1.0f;
        mean_kmult[tnuc] += wt*params_nn[ibd].kmult[fnum];
        kmult_count[tnuc] += wt;
      }
    }
  }
  for (int nnuc=0; nnuc<NUMNUC; nnuc++){
    mean_kmult[nnuc] /= kmult_count[nnuc];
  }
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (BeadIncluded(ibd, skip_beads)){
      for (int fnum=0; fnum<flow_block_size; fnum++){
        int tnuc = flow_ndx_map[fnum];
        params_nn[ibd].kmult[fnum] /=mean_kmult[tnuc];
      }
      params_nn[ibd].ApplyAmplitudeDrivenKmultLimit(flow_block_size, 2.0f);
    }
  }

}


float BeadTracker::CenterDmultFromSample ()
{
  float mean_dmult = FindMeanDmultFromSample ();
  RescaleDmultFromSample (mean_dmult);
  return (mean_dmult);
}

void BeadTracker::RescaleRatio (float scale)
{
  for (int ibd=0; ibd<numLBeads; ibd++)
  {
    params_nn[ibd].R /= scale;
    if (params_nn[ibd].R>1.0f)
      params_nn[ibd].R = 1.0f; // cannot go above 1.0
  }
}

float BeadTracker::FindMeanDmultFromSample ()
{
  float mean_dmult = 0.0f;
  int num_checked = 0;
  for (int ibd=0; ibd < numLBeads; ibd++)
  {
    // if this iteration is a region-wide parameter fit, then only process beads
    // in the sampled sub-group
    assert ( isSampled );
    if ( Sampled (ibd) ) {
      mean_dmult += params_nn[ibd].dmult;
      num_checked += 1;
    }
  }
  if (num_checked > 0)
    mean_dmult /= num_checked;
  else
    mean_dmult = 1.0f; // set mean_dmult to a sensible value, not 2.0!

  return (mean_dmult);
}

void BeadTracker::WriteCorruptedToMask (Region *region, Mask *bfmask, int16_t *washout_state, int flow)
{
  if (region!=NULL)
    for (int ibd=0; ibd<numLBeads ; ibd++)
      if (params_nn[ibd].my_state->corrupt)
      {
        ndx_map[params_nn[ibd].y*region->w+params_nn[ibd].x] = -1; // ignore this neighbor from now on to avoid contaminating xtalk
        bfmask->Set (params_nn[ibd].x+region->col,params_nn[ibd].y+region->row,MaskWashout);
        int well = params_nn[ibd].y+region->row * bfmask->W() + params_nn[ibd].x+region->col;
        if (washout_state != NULL && washout_state[well] < 0) {
          washout_state[well] = flow;
        }
      }
}

//
void BeadTracker::UpdateAllPhi(int flow, int flow_block_size){
  for (int ibd=0; ibd<numLBeads; ibd++){
    params_nn[ibd].UpdateCumulativeAvgIncorporation(flow, flow_block_size);
  }
}

void BeadTracker::BuildBeadMap (Region *region, Mask *bfmask, MaskType &process_mask)
{
  if (region!=NULL)
  {
    ndx_map.resize(region->w*region->h);
//    int chipwidth = bfmask->W();
    int nbd = 0;
    for (int y=0;y<region->h;y++)
    {
      for (int x=0;x<region->w;x++)
      {
        ndx_map[y*region->w+x] = -1;
// did I find the wells appropriately
        bool matchedbead = bfmask->Match (x+region->col,y+region->row,process_mask);
        bool validbead = true;
        // bool validbead = (mWellIdx.size()<1 | binary_search(mWellIdx.begin(),mWellIdx.end(),x+region->col+(y+region->row)*chipwidth));
        if (matchedbead & validbead)
        {
          params_nn[nbd].x = x;
          params_nn[nbd].y = y;
          // this is no longer needed because the data is shifted when read into the background model
          // so per-well treatment of t_mid_nuc is un-necessary

          ndx_map[y*region->w+x] = nbd;
          nbd++;
        }
      }
    }
    AssignKeyID (region,bfmask); // map beads to key id now that I know where they are on the mask
  }
}

void BeadTracker::InitBeadParamR (Mask *bfMask, Region *region, std::vector<float> *tauB,std:: vector<float> *tauE)
{
  if (region != NULL)
  {
    for (int y=0; y<region->h;y++)
    {
      for (int x=0; x<region->w;x++)
      {
        if (ndx_map[y*region->w+x] > -1)
        {
          int foundId = x + region->col + (y+region->row) *bfMask->W();
          if ( (*tauB) [foundId] != 0)
            params_nn[ndx_map[y*region->w+x]].R = (*tauE) [foundId]/ (*tauB) [foundId];
        }

      }
    }
  }
}

int BeadTracker::FindKeyID (Mask *bfMask, int ax, int ay)
{
  int retval = -1;
  for (int k = 0; k < numSeqListItems; k++)
  {
    if (bfMask->Match (ax, ay, seqList[k].type, MATCH_ANY))
    {
      retval = k;
    }
  }
  return (retval);
}

void BeadTracker::AssignKeyID (Region *region, Mask *bfMask)
{
  for (int ibd=0; ibd<numLBeads; ibd++)
  {
    int ax = params_nn[ibd].x + region->col;
    int ay = params_nn[ibd].y + region->row;
    key_id[ibd] = FindKeyID (bfMask,ax,ay);
  }
}


void BeadTracker::InitRandomSample (Mask& bf, Region& region, const std::set<int>& sample)
{
  int chipWidth = bf.W();
  for (int i=0; i<numLBeads; ++i)
  {
    struct BeadParams *p = &params_nn[i];
    int chipX   = region.col + p->x;
    int chipY   = region.row + p->y;
    int wellIdx = chipY * chipWidth + chipX;

    if (sample.count (wellIdx))
      p->my_state->random_samp = true;
  }
}

int BeadTracker::SystematicSampling(int sampling_rate, std::vector<size_t>::iterator first, std::vector<size_t>::iterator last)
{
  // Systematic Sampling scheme using bead order in BeadTracker
  if (first >= last)
    return 0;

  assert(sampling_rate >= 1);
  int nsample = 0;
  for (std::vector<size_t>::iterator ibd=first; ibd < last; ibd++)
  {
    if ((*ibd % sampling_rate) == 0) {
      sampled.at(*ibd) = true;
      nsample++;
    }
  }
  return nsample;
}

int BeadTracker::SetSampled(int numSamples)
{
  int sampling_rate = (numLBeads / numSamples) + 1;

  std::vector<size_t> ix(numLBeads, 0);
  for (size_t ibd=0; ibd < (size_t)numLBeads; ibd++) {
    sampled[ibd] = false;
    ix[ibd] = ibd;
  }
  int nsample = SystematicSampling(sampling_rate, ix.begin(), ix.end());
  isSampled = true;
  return nsample;
}

int BeadTracker::SetPseudoRandomSampled(int sample_level){
  for (int ibd=0; ibd<numLBeads; ibd++)
    sampled[ibd] = false;
  for (int i=0; i<sample_level; i++){
      int rbd = (i+SMALL_PRIME)*LARGE_PRIME % numLBeads;
      sampled[rbd] = true;
  }
  int nsample=0;
  for (int ibd=0; ibd<numLBeads; ibd++)
    if (sampled[ibd]) nsample++;
  isSampled = true;
  return(nsample);
}

/* choose a sample set of beads using penalty
 * order the live beads by penalty up to infinite penalty
 * choose every n-th bead starting from the 1st where n = sampling_rate
 * attempt to systematically choose beads to reach ntarget if not enough
 * beads to total ntarget reached using the sort
 * return the number of beads in the sample
 */
int BeadTracker::SetSampled(std::vector<float>& penalty, int sampling_rate)
{
  std::vector<size_t> ix;
  ionStats::sort_order(penalty.begin(), penalty.end(), ix);
  //ix.resize(numLBeads);
  //for (size_t i=0; i<(size_t)numLBeads; i++)
  //  ix[i] = i;

  int nsample = 0;  // total sample found
 
  for (int ibd=0; ibd < numLBeads; ibd++)
    sampled[ibd] = false;

  std::vector<size_t>::iterator itx = ix.begin();
  for (; itx != ix.end(); itx++)
  {
    if ( isinf (penalty[ *itx ]) )
      break;

    if ((*itx % sampling_rate) == 0) {

      if (nsample == ntarget)
	break;

      sampled[ *itx ] = true;
      nsample++;
    }
  }
  int nsample1 = nsample;
  int residual_sampling_rate = 0;
  if (nsample<ntarget && itx<ix.end() ){
    residual_sampling_rate = std::max(1.0, (double)(ix.end() - itx+ntarget-1)/(ntarget-nsample)); // (+ ntarget-1) guarantees a sampling rate that will never sample more than ntarget beads.
    nsample += SystematicSampling( residual_sampling_rate, itx, ix.end() );
  }
  if (false)
    fprintf(stdout, "Sampling scheme: clonal penalty: sampled %d beads with target %d at sampling rate %d selected from %d live beads with systematic sampling of %d at rate %d in (overall samples: %d ) region %d\n", nsample1, ntarget, sampling_rate, numLBeads, nsample-nsample1, residual_sampling_rate, nsample, regionindex);

  isSampled = true;  
  return nsample;
}

int BeadTracker::NumberSampled()
{
  int n=0;

  for (int ibd=0; ibd<numLBeads; ibd++)
    if (Sampled(ibd))
      n++;
  
  return(n);
}

float BeadTracker::FindPercentageCopyCount(float percentage){
  int num_sampled = NumberSampled();
  std::vector<float> copy_count;
      copy_count.resize(num_sampled);
  int kount =0;
  for (int ibd=0; ibd<numLBeads; ibd++){
    if (Sampled(ibd)){
      copy_count[kount] = params_nn[ibd].Copies;
      kount++;
    }
  }
  sort(copy_count.begin(), copy_count.end()); // standard
  int threshold = (num_sampled-1)*percentage; // rounding is fine here
  return(copy_count.at(threshold));
}

void BeadTracker::DumpBeads (FILE *my_fp, bool debug_only, int offset_col, int offset_row, int flow_block_size)
{
  if (numLBeads>0) // trap for dead regions
  {
    if (debug_only)
      params_nn[DEBUG_BEAD].DumpBeadProfile (my_fp,offset_col, offset_row, flow_block_size);
    else
      for (int ibd=0; ibd<numLBeads; ibd++)
        params_nn[ibd].DumpBeadProfile (my_fp, offset_col, offset_row, flow_block_size);
  }
}

void BeadTracker::CheckPointClassifier(FILE *my_fp, int id, int offset_col, int offset_row, int flow_block_size)
{
  if (numLBeads>0){
    params_nn[0].DumpBeadTitle(my_fp,flow_block_size);
    for (int ibd=0; ibd<numLBeads; ibd++){
      if (barcode_info.barcode_id[ibd]==id) // only matching beads
        params_nn[ibd].DumpBeadProfile (my_fp, offset_col, offset_row, flow_block_size);
    }
  }
}

void BeadTracker::DumpAllBeadsCSV (FILE *my_fp, int flow_block_size)
{
  for (int i=0;i<numLBeads;i++)
  {
    char sep_char = ',';
    //if (i == (numLBeads -1))
      sep_char = '\n';

    for (int j=0;j<flow_block_size;j++)
      fprintf (my_fp,"%10.5f,",params_nn[i].Ampl[j]);

    fprintf (my_fp,"%10.5f,%10.5f,%10.5f,%10.5f,%10.5f%c",params_nn[i].R,params_nn[i].dmult,params_nn[i].Copies,params_nn[i].gain,0.0,sep_char);
  }
}
