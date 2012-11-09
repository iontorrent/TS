/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iomanip>
#include "BeadTracker.h"
#include "mixed.h"
#include "ImageLoader.h"
#include "MathUtil.h"
#include "LevMarState.h"

using namespace std;

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
}

void BeadTracker::InitHighBeadParams()
{
  params_SetBeadStandardHigh (&params_high);
}

void BeadTracker::InitLowBeadParams(float AmplLowerLimit)
{
  params_SetBeadStandardLow (&params_low,AmplLowerLimit);
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
    params_SetBeadStandardValue (&params_nn[i]);
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

void BeadTracker::InitBeadList (Mask *bfMask, Region *region, SequenceItem *_seqList, int _numSeq, const std::set<int>& sample, float AmplLowerLimit)
{
  MaskType ProcessMask=MaskLive;

  // Number of beads to process
  numLBeads =  bfMask->GetCount (ProcessMask,*region);
  regionindex = region->index;

  InitBeadParams(AmplLowerLimit);
  DefineKeySequence (_seqList, _numSeq);
  high_quality.resize(numLBeads, true);
  sampled.resize(numLBeads, true);

  // makes spatially organized data if needed
  BuildBeadMap (region,bfMask,ProcessMask);
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


float ComputeSSQRatioKeyFlows (const float *observed, const int *keyval, int len)
{
  float wsum = 0.01f;
  float wdenom = 0.01f; // avoid dividing by zero in the bad case, smooth results to prevent wild values
  for (int i=0; i<len; i++)
  {
    float delta = observed[i]-keyval[i];
    
    wsum += keyval[i]*keyval[i];  // variation due to signal
    wdenom += delta*delta;  // variation due to noise
  }
  return ((wsum/wdenom));
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

void BeadTracker::SelectKeyFlowValuesFromBeadIdentity (int *keyval, float *observed, int my_key_id, int &keyLen)
{
  if (my_key_id<0)
    SelectKeyFlowValues (keyval,observed,keyLen);
  else
  {
    // lookup key id in seqlist

    keyLen = seqList[my_key_id].usableKeyFlows; // return the correct number of flows
    if (keyLen>NUMFB)
      keyLen = NUMFB;  // might have a very long key(!)
    for (int i=0; i<keyLen; i++)
      keyval[i] = seqList[my_key_id].Ionogram[i];
  }
}

void SetKeyFlows (float *observed, int *keyval, int len)
{
  for (int i=0; i<len; i++)
    observed[i] = keyval[i];
}

float BeadTracker::ComputeSSQRatioOneBead(int ibd)
{
  int key_val[NUMFB];
  struct bead_params *p = &params_nn[ibd];
  int key_len;
  key_len = KEY_LEN; 
  SelectKeyFlowValuesFromBeadIdentity (key_val,p->Ampl, key_id[ibd], key_len); // what should we see
  return(ComputeSSQRatioKeyFlows (p->Ampl, key_val, key_len)); // how much did we deviate from it?
}

float BeadTracker::KeyNormalizeOneBead (int ibd, bool overwrite_key)
{
  int   key_val[NUMFB];
  float normalizer = 1.0f;
  bead_params  *p  = &params_nn[ibd];
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
    SelectKeyFlowValuesFromBeadIdentity (key_val,p->Ampl, key_id[ibd], key_len);
  }
  normalizer=ComputeNormalizerKeyFlows (p->Ampl,key_val,key_len);

  //cout <<ibd << " p-Copies before :" <<p->Copies ;
  p->Copies *= normalizer;
  // scale to right signal
  //cout <<" and after =" << p->Copies <<"\n";
  MultiplyVectorByScalar (p->Ampl,1.0f/normalizer,NUMFB);

  // set to exact values in key flows, overriding real levels
  if (overwrite_key)
    SetKeyFlows (p->Ampl,key_val,key_len);

  params_ApplyLowerBound (p,pl);
  params_ApplyUpperBound (p,ph);
  return (p->Copies);
}


// Reads key identity from the separator and exploit the lookup table
// in particular, we need to handle 0,1,2 mers for possible signal types
// when using arbitrary keys
// we use this to set a good scale/copy number for the beads
float BeadTracker::KeyNormalizeReads(bool overwrite_key, bool sampled_only)
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
    meanCopyCount += KeyNormalizeOneBead (ibd,overwrite_key);
    goodBeadCount++;
    // }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  meanCopyCount /= goodBeadCount;
  return (meanCopyCount);
}

float BeadTracker::KeyNormalizeSampledReads(bool overwrite_key)
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
          meanCopyCount += KeyNormalizeOneBead (ibd,overwrite_key);
          goodBeadCount++;
      }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  meanCopyCount /= goodBeadCount;
  return (meanCopyCount);
}

void BeadTracker::LowSSQRatioBeadsAreLowQuality(float snr_threshold)
{
  int filter_total = 0;
  // only use the top snr beads for region-wide parameter fitting
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    float SNR = ComputeSSQRatioOneBead(ibd);
    if (SNR < snr_threshold)
    {
      high_quality[ibd] = false; // reject beads that don't match
      filter_total++;
    }
    //printf("LSSQ: %d %f\n",ibd, SNR);
  }
//  printf("Region %d fit exclude %d beads of %d for poor ssq ratio less than %f\n", regionindex, filter_total, numLBeads, snr_threshold);
}

//@TODO: these are actually part of beadtracker, move them
void BeadTracker::LowCopyBeadsAreLowQuality (float mean_copy_count)
{
  int cp_filter = 0;
  // only use the top amplitude signal beads for region-wide parameter fitting
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    if (params_nn[ibd].Copies < mean_copy_count)
    {
      high_quality[ibd] = false; // reject beads that don't match
      cp_filter++;
    }
  }
//  printf("Region %d fit exclude %d beads of %d for low copy count below %f\n", regionindex, cp_filter, numLBeads, mean_copy_count);
}

void BeadTracker::CorruptedBeadsAreLowQuality ()
{
  for (int ibd=0; ibd<numLBeads; ibd++)
  {
    if (params_nn[ibd].my_state->corrupt or params_nn[ibd].my_state->pinned)
      high_quality[ibd] = false; // reject beads that are sad in any way
  }
}

void BeadTracker::TypicalBeadParams(bead_params *p)
{
  int count=1; // start with a "standard" bead in case we don't have much
  params_SetBeadStandardValue(p);
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
    params_SetStandardFlow(&params_nn[i]);
  }
}

void BeadTracker::CompensateAmplitudeForEmptyWellNormalization(float *my_scale_buffer)
{
  // my scale buffer contains a rescaling of the data per flow per bead and is therefore a memory hog
  // we'll avoid using this if at all possible.
  for (int ibd=0;ibd <numLBeads;ibd++)
  {
    for (int fnum=0; fnum<NUMFB; fnum++)
      params_nn[ibd].Ampl[fnum] *= my_scale_buffer[ibd*NUMFB+fnum];
  }
}

//for all bead
void BeadTracker::AssignEmphasisForAllBeads (int _max_emphasis)
{
  max_emphasis = _max_emphasis;
}

void BeadTracker::AdjustForCopyNumber(vector<float>& ampl, const bead_params& p, const vector<float>& copy_multiplier)
{
  size_t num_flows = ampl.size();
  for(size_t flow=0; flow<num_flows; ++flow)
    ampl[flow] = p.Ampl[flow] * p.Copies * copy_multiplier[flow];
}

void BeadTracker::ComputeKeyNorm(const vector<int>& keyIonogram, const vector<float>& copy_multiplier)
{
  vector<float> ampl(keyIonogram.size());
  for(int bead=0; bead<numLBeads; ++bead){
    bead_params& p = params_nn[bead];
    AdjustForCopyNumber(ampl, p, copy_multiplier);
    p.my_state->key_norm = ComputeNormalizerKeyFlows(&ampl[0], &keyIonogram[0], keyIonogram.size());
  }
}

void BeadTracker::CheckKey(const vector<float>& copy_multiplier)
{
  // Flag beads with bad key:
  vector<float> ampl(seqList[1].usableKeyFlows);
  vector<float> nrm (seqList[1].usableKeyFlows);
  for (int bead=0; bead<numLBeads; ++bead)
  {
    bead_params& p = params_nn[bead];
    AdjustForCopyNumber(ampl, p, copy_multiplier);
    transform (ampl.begin(), ampl.end(), nrm.begin(), bind2nd (divides<float>(),p.my_state->key_norm));
    if (not key_is_good (nrm.begin(), seqList[1].Ionogram, seqList[1].Ionogram+seqList[1].usableKeyFlows)){
      p.my_state->bad_read = true;
      ++numLBadKey;
    }
  }
}

void BeadTracker::UpdateClonalFilter (int flow, const vector<float>& copy_multiplier)
{
  // first block only:
  if(flow == NUMFB-1){
    vector<int> keyIonogram(seqList[1].Ionogram, seqList[1].Ionogram+seqList[1].usableKeyFlows);
    ComputeKeyNorm(keyIonogram, copy_multiplier);
    CheckKey(copy_multiplier);
  }

  // all blocks used by clonal filter:
  int lastBlock = ceil(1.0 * mixed_last_flow() / NUMFB);
  int lastFlow  = lastBlock * NUMFB;
  if ((flow+1)%NUMFB==0 and flow<lastFlow)
    UpdatePPFSSQ (flow, copy_multiplier);

  // last block used by clonal filter:
  if (flow==lastFlow-1)
    FinishClonalFilter();
}

void BeadTracker::UpdatePPFSSQ (int flow, const vector<float>& copy_multiplier)
{
  vector<float> ampl(NUMFB);
  for (int bead=0; bead<numLBeads; ++bead)
  {
    bead_params& p = params_nn[bead];
    AdjustForCopyNumber(ampl, p, copy_multiplier);
    transform (ampl.begin(), ampl.end(), ampl.begin(), bind2nd (divides<float>(),p.my_state->key_norm));

    // Update ppf and ssq:
    assert(mixed_first_flow() < NUMFB);
    int first = max (flow+1-NUMFB, mixed_first_flow()) % NUMFB;
    int last  = min (flow, mixed_last_flow()-1) % NUMFB;
    for (int i=first; i<=last; ++i)
    {
      if (ampl[i] > mixed_pos_threshold())
        p.my_state->ppf += 1;
      float x = ampl[i] - round (ampl[i]);
      p.my_state->ssq += x * x;
    }
    // Flag beads with infinite signal:
    if (not all_finite (p.Ampl, p.Ampl+NUMFB))
      p.my_state->bad_read = true;
  }
}

void BeadTracker::FinishClonalFilter()
{
  for (int bead=0; bead<numLBeads; ++bead)
  {
    bead_params& p = params_nn[bead];
    p.my_state->ppf /= (mixed_last_flow() - mixed_first_flow());
  }
}
int BeadTracker::NumHighPPF() const
{
  int numHigh = 0;
  for (int bead=0; bead<numLBeads; ++bead)
  {
    const bead_params& p = params_nn[bead];
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
    const bead_params& p = params_nn[bead];
    if (not p.my_state->clonal_read)
      ++numPolyclonal;
  }
  return numPolyclonal;
}

int BeadTracker::NumBadKey() const
{
  return numLBadKey;
}

void BeadTracker::ZeroOutPins (Region *region, Mask *bfmask, PinnedInFlow& pinnedInFlow, int flow, int iFlowBuffer)

{
  // Zero out beads that pin in this flow or earlier flows
// Don't assume buffer structure
  //int iFlowBuffer = flow % NUMFB;

  for (int bead=0; bead<numLBeads; ++bead)
  {
    bead_params& p = params_nn[bead];
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

float BeadTracker::CenterDmult (bool skip_beads)
{
  // return ( CenterDmultFromSample (skip_beads) );
  float mean_dmult = FindMeanDmult (skip_beads);
  RescaleDmult (mean_dmult);
  return (mean_dmult);
}


float BeadTracker::CenterDmultFromSample (bool skip_beads)
{
  float mean_dmult = FindMeanDmultFromSample (skip_beads);
  RescaleDmult (mean_dmult);
  return (mean_dmult);
}

void BeadTracker::RescaleRatio (float scale)
{
  for (int ibd=0; ibd<numLBeads; ibd++)
  {
    params_nn[ibd].R /= scale;
  }
}

float BeadTracker::FindMeanDmultFromSample (bool skip_beads)
{
  float mean_dmult = 0.0f;
  float num_checked = 0.0001f;
  for (int ibd=0; ibd < numLBeads; ibd++)
  {
    // if this iteration is a region-wide parameter fit, then only process beads
    // in the selection sub-group
    if( !Sampled (ibd) )
      continue;

    if ( BeadIncluded(ibd, skip_beads) ) {
	mean_dmult += params_nn[ibd].dmult;
	num_checked += 1.0f;
      }
  }
  mean_dmult /= num_checked;
  return (mean_dmult);
}


void BeadTracker::WriteCorruptedToMask (Region *region, Mask *bfmask)
{
  if (region!=NULL)
    for (int ibd=0; ibd<numLBeads ; ibd++)
      if (params_nn[ibd].my_state->corrupt)
      {
        ndx_map[params_nn[ibd].y*region->w+params_nn[ibd].x] = -1; // ignore this neighbor from now on to avoid contaminating xtalk
        bfmask->Set (params_nn[ibd].x+region->col,params_nn[ibd].y+region->row,MaskWashout);
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
    struct bead_params *p = &params_nn[i];
    int chipX   = region.col + p->x;
    int chipY   = region.row + p->y;
    int wellIdx = chipY * chipWidth + chipX;

    if (sample.count (wellIdx))
      p->my_state->random_samp = true;
  }
}

void BeadTracker::SystematicSampling()
{
  // Systematic Sampling scheme
  int num_region_groups = (numLBeads / NUMBEADSPERGROUP) + 1;

  for (int ibd=0; ibd < numLBeads; ibd++)
  {
    sampled[ibd] = false;

    if ((ibd % num_region_groups) == 0) {
      sampled[ibd] = true;
    }
  }
}

void BeadTracker::SetSampled(){
  SystematicSampling();
  isSampled = true;
}

void BeadTracker::UpdateSampled( bool *well_completed){
  for (int ibd=0; ibd < numLBeads; ibd++) {
    if ( (high_quality[ibd] == false) || well_completed[ibd] ) {
      sampled[ibd] = false;
    }
  }
}

void BeadTracker::ExcludeFromSampled( int ibd ){
  // @TODO reconcile use of Sampled with MultiLevMar::ExcludeBead
  // Can we get rid of well_completed?
  //sampled[ibd] = false;
}

void BeadTracker::DumpBeads (FILE *my_fp, bool debug_only, int offset_col, int offset_row)
{
  if (numLBeads>0) // trap for dead regions
  {
    if (debug_only)
      DumpBeadProfile (&params_nn[DEBUG_BEAD],my_fp,offset_col, offset_row);
    else
      for (int ibd=0; ibd<numLBeads; ibd++)
        DumpBeadProfile (&params_nn[ibd],my_fp, offset_col, offset_row);
  }
}

void BeadTracker::DumpAllBeadsCSV (FILE *my_fp)
{
  for (int i=0;i<numLBeads;i++)
  {
    char sep_char = ',';
    if (i == (numLBeads -1))
      sep_char = '\n';

    for (int j=0;j<NUMFB;j++)
      fprintf (my_fp,"%10.5f,",params_nn[i].Ampl[j]);

    fprintf (my_fp,"%10.5f,%10.5f,%10.5f,%10.5f,%10.5f%c",params_nn[i].R,params_nn[i].dmult,params_nn[i].Copies,params_nn[i].gain,0.0,sep_char);
  }
}

/*void BeadTracker::DumpHits(int offset_col, int offset_row, int flow)
{
    char fname[100];
    sprintf(fname,"%d.%d.beads.%d.txt",offset_col,offset_row,flow);
    FILE *fp = fopen(fname,"wt");
    for (int i=0; i<numLBeads; i++){
      fprintf(fp, "%d",i);
      for (int j=0; j<NUMFB; j++)
      {
        fprintf(fp, "\t%d ",params_nn[i].my_state->hits_by_flow[j]);
        params_nn[i].my_state->hits_by_flow[j] = 0;
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
}*/
