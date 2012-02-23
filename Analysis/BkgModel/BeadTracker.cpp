/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iomanip>
#include "BeadTracker.h"
#include "mixed.h"

using namespace std;

// bead parameter initial values and box constraints

BeadTracker::BeadTracker()
{
  ndx_map = NULL;
  params_nn = NULL;
  params_low = NULL;
  params_high = NULL;
  seqList=NULL;
  DEBUG_BEAD = 0;
  numLBeads=0;
  numSeqListItems = 0;
  max_emphasis = 0;
  key_id = NULL;
}

void BeadTracker::InitHighBeadParams()
{
  params_high = new bound_params[numLBeads];
  memset (params_high,0,sizeof (bound_params[numLBeads]));
  for (int i=0;i<numLBeads;i++)
  {
    params_SetBeadStandardHigh (&params_high[i]);
  }
}

void BeadTracker::InitLowBeadParams()
{
  params_low = new bound_params[numLBeads];
  memset (params_low,0,sizeof (bound_params[numLBeads]));
  for (int i=0;i<numLBeads;i++)
  {
    params_SetBeadStandardLow (&params_low[i]);
  }
}

void BeadTracker::InitModelBeadParams()
{
  params_nn = new bead_params[numLBeads];
  memset (params_nn,0,sizeof (bead_params[numLBeads]));
  for (int i=0;i<numLBeads;i++)
  {
    params_SetBeadStandardValue (&params_nn[i]);
    params_nn[i].trace_ndx = i; // index into trace object
  }
}

void BeadTracker::DefineKeySequence (SequenceItem *seq_list, int number_Keys)
{
  this->seqList= seq_list;
  this->numSeqListItems=number_Keys;
  key_id = new int [numLBeads]; // indexing into this item
}

void BeadTracker::SelectDebugBead()
{
  // shouldn't have to bounds check this but I don't trust myself
  DEBUG_BEAD = (int) ( ( (double) rand() /RAND_MAX) *numLBeads);
  if (DEBUG_BEAD > numLBeads) DEBUG_BEAD =numLBeads-1; // within range
  if (DEBUG_BEAD < 0) DEBUG_BEAD = 0;

#ifdef DEBUG_BEAD_OVERRIDE
  DEBUG_BEAD = DEBUG_BEAD_OVERRIDE;
#endif
}

void BeadTracker::InitBeadParams()
{
  // no spatial information here
  // just boot up parameters
  SelectDebugBead();
  InitHighBeadParams();
  InitLowBeadParams();
  InitModelBeadParams();

}

void BeadTracker::InitBeadList (Mask *bfMask, Region *region, SequenceItem *_seqList, int _numSeq, const std::set<int>& sample)
{
  MaskType ProcessMask=MaskLive;

  // Number of beads to process
  numLBeads =  bfMask->GetCount (ProcessMask,*region);

  // Needs number of beads to initialize
  InitBeadParams();
  DefineKeySequence (_seqList, _numSeq);

  // makes spatially organized data if needed
  BuildBeadMap (region,bfMask,ProcessMask);
  // must be done after generating spatial data
  InitRandomSample (*bfMask, *region, sample);
}

// flow+1<NUMFB,0.02, 0.98
void BeadTracker::LimitBeadEvolution (bool first_block, float R_change_max, float copy_change_max)
{
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    // offset into my per-bead structure
    struct bead_params *p = &params_nn[ibd];
    struct bound_params *pl = &params_low[ibd];
    struct bound_params *ph = &params_high[ibd];

    if (first_block)
    {
      // assume first past through
      // limit the change in etbR
      pl->R = p->R - R_change_max;
      ph->R = p->R + R_change_max;
    }
    // limit the change in apparent number of molecules
    pl->Copies = p->Copies*copy_change_max;
    ph->Copies = p->Copies;
  }
}


//this function try to estimate a multiplication factor to determine the number
// of copies corresponding to a positive flow.
//input is an array of observed values for a specific bead over initial flows
//  keyval is an array of integers with 1 indicating incorporating flow
//  len is the length of flows to do the processes.
// effectively optimizing sum((y-a*x)^2) y = observed, x = keyval, a = normalizing constant
float ComputeNormalizerKeyFlows (const float *observed, const int *keyval, int len)
{
  float wsum = 0.0001;
  float wdenom = 0.0001; // avoid dividing by zero in the bad case
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
  float mean_key=0.0;
  for (int i=0;i<keyLen;i++)
    mean_key += observed[i];

  mean_key /= 3.0;  // @TODO crude classifier assuming 3 live in first 7
  for (int i=0;i<keyLen;i++)
  {
    if (observed[i] > (mean_key/2.0))
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

float BeadTracker::KeyNormalizeOneBead (int ibd)
{
  int key_val[NUMFB];
  float normalizer = 1.0;
  struct bead_params *p = &params_nn[ibd];
  struct bound_params *pl = &params_low[ibd];
  struct bound_params *ph = &params_high[ibd];

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
  MultiplyVectorByScalar (p->Ampl,1.0/normalizer,NUMFB);

  // set to exact values in key flows, overriding real levels
  SetKeyFlows (p->Ampl,key_val,key_len);

  params_ApplyLowerBound (p,pl);
  params_ApplyUpperBound (p,ph);
  return (p->Copies);
}


// Reads key identity from the separator and exploit the lookup table
// in particular, we need to handle 0,1,2 mers for possible signal types
// when using arbitrary keys
// we use this to set a good scale/copy number for the beads
float BeadTracker::KeyNormalizeReads()
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
    meanCopyCount += KeyNormalizeOneBead (ibd);
    goodBeadCount++;
    //   }
  }
  // cout << "Number of good beads:" << goodBeadCount  <<"\n";
  meanCopyCount /= goodBeadCount;
  return (meanCopyCount);
}





void BeadTracker::ResetFlowParams (int bufNum,int flow)
{
  (void) flow;
  for (int i=0;i<numLBeads;i++)
  {
    params_nn[i].Ampl[bufNum] = 0.001;
    params_nn[i].kmult[bufNum] = 1.0;
  }
}

void BeadTracker::Delete()
{
  if (ndx_map != NULL) delete [] ndx_map;
  if (key_id !=NULL) delete[] key_id;
  delete [] params_high;
  delete [] params_low;
  delete [] params_nn;
}

BeadTracker::~BeadTracker()
{
  Delete();
}

//for all bead
void BeadTracker::AssignEmphasisForAllBeads (int _max_emphasis)
{
  max_emphasis = _max_emphasis;
  for (int ibd=0;ibd <numLBeads;ibd++)
  {
    ComputeEmphasisOneBead(params_nn[ibd].WhichEmphasis, params_nn[ibd].Ampl, max_emphasis);
  }
}


void BeadTracker::CheckKey()
{
    // Flag beads with bad key:
    for(int bead=0; bead<numLBeads; ++bead){
        bead_params& p = params_nn[bead];
        if(not key_is_good(p.Ampl, seqList[1].Ionogram, seqList[1].Ionogram+seqList[1].usableKeyFlows))
            p.my_state.bad_read = true;
    }
}

void BeadTracker::UpdateClonalFilter()
{
  for (int bead=0; bead<numLBeads; ++bead)
  {
    bead_params& p = params_nn[bead];

    // Update ppf and ssq:
    for (int i=0; i<NUMFB; ++i)
    {
      float ampl = p.Ampl[i];
      if (ampl > mixed_pos_threshold())
        p.my_state.ppf += 1;
      float x = ampl - round (ampl);
      p.my_state.ssq += x * x;
    }

    // Flag beads with infinite signal:
    if (not all_finite (p.Ampl, p.Ampl+NUMFB))
      p.my_state.bad_read = true;
  }
}

void BeadTracker::FinishClonalFilter()
{
  for (int bead=0; bead<numLBeads; ++bead)
  {
    bead_params& p = params_nn[bead];
    p.my_state.ppf /= 80;
    if (p.my_state.ppf > mixed_ppf_cutoff())
      p.my_state.bad_read = true;
  }
}

float BeadTracker::FindMeanDmult()
{
  float mean_dmult = 0.0;

  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    mean_dmult += params_nn[ibd].dmult;
  }
  mean_dmult /= numLBeads;
  return (mean_dmult);
}

void BeadTracker::RescaleDmult (float scale)
{
  for (int ibd=0;ibd < numLBeads;ibd++)
  {
    params_nn[ibd].dmult /= scale;
  }
}

float BeadTracker::CenterDmult()
{
  float mean_dmult = FindMeanDmult();
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




void BeadTracker::WriteCorruptedToMask (Region *region, Mask *bfmask)
{
  if (region!=NULL)
    for (int ibd=0; ibd<numLBeads ; ibd++)
      if (params_nn[ibd].my_state.corrupt)
      {
        ndx_map[params_nn[ibd].y*region->w+params_nn[ibd].x] = -1; // ignore this neighbor from now on to avoid contaminating xtalk
        bfmask->Set (params_nn[ibd].x+region->col,params_nn[ibd].y+region->row,MaskWashout);
      }
}

void BeadTracker::BuildBeadMap (Region *region, Mask *bfmask, MaskType &process_mask)
{
  if (region!=NULL)
  {
    ndx_map = new int[region->w*region->h];
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
      p->my_state.random_samp = true;
  }
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
