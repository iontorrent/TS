/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "EmptyTrace.h"
#include <assert.h>
#include <math.h>

EmptyTrace::EmptyTrace(CommandLineOpts &clo)
{
  imgFrames = 0;
  // imgCols = 0;
  // imgRows = 0;

  neg_bg_buffers_slope = NULL;
  bg_buffers = NULL;
  bg_dc_offset = NULL;
  t0_map = NULL;
  numfb = 0;

  regionIndex = -1;

    referenceMask = MaskEmpty;
}

void EmptyTrace::Allocate(int _numfb, int _imgFrames)
{
  // bg_buffers and neg_bg_buffers_slope
  // are contiguous arrays organized as a trace per flow in a block of flows
  assert( _numfb > 0 );
  assert( _imgFrames > 0 );
  assert( bg_buffers == NULL);  //logic only checked for one-time allocation
  assert( neg_bg_buffers_slope == NULL );
  assert( bg_dc_offset == NULL );
  numfb = _numfb;

  imgFrames = _imgFrames;
  bg_buffers  = new float [numfb*imgFrames];
  neg_bg_buffers_slope  = new float [numfb*imgFrames];

  bg_dc_offset = new float [numfb];
  memset (bg_dc_offset,0,sizeof (float[numfb]));
}

EmptyTrace::~EmptyTrace()
{
  if (neg_bg_buffers_slope!=NULL) delete [] neg_bg_buffers_slope;
  if (bg_buffers!=NULL) delete [] bg_buffers;
  if (bg_dc_offset!=NULL) delete[] bg_dc_offset;
  if (t0_map != NULL) delete [] t0_map;
}


// Savitsky-Goulay filter coefficients for calculating the slope.  poly order = 2, span=+/- 2 points
const float EmptyTrace::bkg_sg_slope[BKG_SGSLOPE_LEN] = {-0.2,-0.1,0.0,0.1,0.2};

void EmptyTrace::SavitskyGolayComputeSlope (float *local_slope,float *source_val, int len)
{
  // compute slope using savitsky golay smoother
  // pad ends of sequence with repeat values
  int moff = (BKG_SGSLOPE_LEN-1) /2;
  int mmax = len-1;
  int j;
  for (int i=0; i< len; i++)
  {
    local_slope[i]=0;
    // WARNING: I compute negative slope here because it is used that way later
    for (int k=0; k<BKG_SGSLOPE_LEN; k++)
    {
      j = i+k-moff;
      if (j<0) j=0; // make sure we're in range!!!
      if (j>mmax) j=mmax; // make sure we're in range!!!
      local_slope[i] -= source_val[j]*bkg_sg_slope[k];
    }
  }
}


void EmptyTrace::PrecomputeBackgroundSlopeForDeriv (int flow)
{
  // calculate the slope of the background curve at every point
  int iFlowBuffer = flowToBuffer(flow);
  float *bsPtr = &neg_bg_buffers_slope[iFlowBuffer*imgFrames];
  float *bPtr  = &bg_buffers[iFlowBuffer*imgFrames];

  SavitskyGolayComputeSlope (bsPtr,bPtr,imgFrames);
}

// move the average empty trace in flow so it has zero mean
// for time points between t_start and t_end
// data in bg_buffers changes
void EmptyTrace::RezeroReference (float t_start, float t_end, int flow)
{
  int iFlowBuffer = flowToBuffer(flow);
  
  float *bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  /*
  char s[60000]; int n=0;
  n += sprintf(s, "Rezero\t%d\t%d\t%f\t%f\t%f", regionIndex, flow, t_start, t_end, bg_dc_offset[iFlowBuffer]);
  for (int pt = 0;pt < imgFrames;pt++)
    n += sprintf(&s[n], "\t%f", bPtr[pt]);
  */

  float dc_zero = ComputeDcOffsetEmpty(bPtr,t_start,t_end);
  
  for (int pt = 0;pt < imgFrames;pt++)
    bPtr[pt] -= dc_zero;
  
  bg_dc_offset[iFlowBuffer] += dc_zero; // track this

  /*
  n += sprintf(&s[n], "\t%f", bg_dc_offset[iFlowBuffer]);
  n += sprintf(&s[n], "\n");
  assert (n<60000);
  fprintf(stdout, "%s", s);
  */
}

void EmptyTrace::RezeroReferenceAllFlows(float t_start, float t_end)
{
  // re-zero the traces in all flows
  for (int fnum=0; fnum<numfb; fnum++)
  {
    RezeroReference (t_start, t_end, fnum);
  }
}

float EmptyTrace::ComputeDcOffsetEmpty(float *bPtr, float t_start, float t_end)
{
  float cnt = 0.0001f;
  float dc_zero = 0.000f;

  int above_t_start = (int)ceil(t_start);
  int below_t_end = (int)floor(t_end);

  for (int pt = above_t_start; pt <= below_t_end; pt++)
  {
    dc_zero += (float) (bPtr[pt]);
    cnt += 1.0f;
  }

  // include values surrounding t_start & t_end weighted by overhang
  if (above_t_start > 0){
    float overhang = (above_t_start-t_start);
    dc_zero = dc_zero + bPtr[above_t_start-1]*overhang;
    cnt += overhang;
  }

  if (below_t_end < (imgFrames-1)){
    float overhang = (t_end-below_t_end);
    dc_zero = dc_zero + bPtr[below_t_end+1]*(t_end-below_t_end);
    cnt += overhang;
  }
  dc_zero /= cnt;

  return(dc_zero);
}


void EmptyTrace::GetShiftedBkg (float tshift, TimeCompression &time_cp, float *bkg)
{
  ShiftMe (tshift, time_cp, bg_buffers, bkg);
}

void EmptyTrace::ShiftMe (float tshift, TimeCompression &time_cp, float *my_buff, float *out_buff)
{
  for (int fnum=0;fnum<numfb;fnum++)
  {
    float *fbkg = out_buff + fnum*time_cp.npts;
    float *bg = &my_buff[fnum*imgFrames];         // get ptr to start of neighbor background
    memset (fbkg,0,sizeof (float[time_cp.npts])); // on general principles
    for (int i=0;i < time_cp.npts;i++)
    {
      // get the frame number of this data point (might be fractional because this point could be
      // the average of several frames of data.  This number is the average time of all the averaged
      // data points
      float t=time_cp.frameNumber[i];
      float fn=t-tshift;
      if (fn < 0.0f) fn = 0.0f;
      if (fn > (imgFrames-2)) fn = imgFrames-2;
      int ifn= (int) fn;
      float frac = fn - ifn;

      fbkg[i] = ( (1-frac) *bg[ifn] + frac*bg[ifn+1]);
    }
  }
}

void EmptyTrace::GetShiftedSlope (float tshift, TimeCompression &time_cp, float *bkg)
{
  ShiftMe (tshift, time_cp, neg_bg_buffers_slope, bkg);
}

// dummy function, returns 0s
void EmptyTrace::FillEmptyTraceFromBuffer (short *bkg, int flow)
{
  int iFlowBuffer = flowToBuffer(flow);
  memset (&bg_buffers[iFlowBuffer*imgFrames],0,sizeof (float [imgFrames]));

  // copy background trace, linearize it from pH domain to [] domain
  float *bPtr = &bg_buffers[iFlowBuffer*imgFrames];
  int kount = 0;
  for (int frame=DEFAULT_FRAME_TSHIFT;frame<imgFrames;frame+=1)
  {
    bPtr[kount] += (bkg[frame]/FRAME_AVERAGE);
    kount++;
  }
  for (; kount<imgFrames; kount++)
    bPtr[kount] = bPtr[kount-1];
}

void EmptyTrace::AccumulateEmptyTrace (float *bPtr, float *tmp_shifted, float w)
{
  int kount = 0;
        // shift the background by DEFAULT_FRAME_TSHIFT frames automatically: must be non-negative
        // "tshift=DEFAULT_FRAME_TSHIFT compensates for this exactly"
  for (int frame=DEFAULT_FRAME_TSHIFT;frame<imgFrames;frame++)
  {
    bPtr[kount] += tmp_shifted[frame] * w;
    kount++;
  }
  // assume same value after shifting
  for (; kount<imgFrames; kount++)
  {
    bPtr[kount] = bPtr[kount-1];
  }
}

// Given a region, image and flow, average unpinned empty wells in this region
// using the timing in t0_map into an "average empty trace."
// Average is stored in bg_buffers
//
// WARNING!!! t0_map only matches on per-region basis.  If the EmptyTrace's idea
// of a region doesn't match BgkModel's idea of a region, the empty trace may not
// be what you think it should be

void EmptyTrace::GenerateAverageEmptyTrace (Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, Image *img, int flow)
{
  int iFlowBuffer = flowToBuffer(flow);
  // these are used by both the background and live-bead
  float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
  float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing

  bg_dc_offset[iFlowBuffer] = 0;  // zero out in each new block

  memset (&bg_buffers[iFlowBuffer*imgFrames],0,sizeof (float [imgFrames]));
  float *bPtr;

  float total_weight = 0.0001;
  bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  for (int ay=region->row;ay<region->row+region->h;ay++)
  {
    for (int ax=region->col;ax<region->col+region->w;ax++)
    {
      int ix = bfmask->ToIndex(ay, ax);
      bool isEmpty = bfmask->Match(ax,ay,referenceMask);
      bool isIgnoreOrAmbig = bfmask->Match(ax,ay,(MaskType)(MaskIgnore | MaskAmbiguous));
      bool isUnpinned = ! (pinnedInFlow.IsPinned(flow, ix) );
      if( isEmpty & isUnpinned & ~isIgnoreOrAmbig ) // valid empty well
      {

  TraceHelper::GetUncompressedTrace (tmp,img,ax,ay, imgFrames);
        // shift it to account for relative timing differences - "mean zero shift"
        // ax and ay are global coordinates and need to have the region location subtracted
        // off in order to index into the region-sized t0_map
        if (t0_map!=NULL)
    TraceHelper::SpecialShiftTrace (tmp,tmp_shifted,imgFrames,t0_map[ax-region->col+ (ay-region->row) *region->w]);
        else
          printf ("Alert: t0_map nonexistent\n");
        
        float w=1.0;
#ifdef LIVE_WELL_WEIGHT_BG
        w = 1.0f / (1.0f + (bfmask->GetNumLiveNeighbors (ay,ax) * 2.0f));
#endif
        total_weight += w;
        AccumulateEmptyTrace (bPtr,tmp_shifted,w);
      }
    }
  }
  // fprintf(stdout, "GenerateAverageEmptyTrace: iFlowBuffer=%d, region row=%d, region col=%d, total weight=%f, flow=%d\n", iFlowBuffer, region->row, region->col, total_weight, flow);

  for (int frame=0;frame < imgFrames;frame++)
  {
    bPtr[frame] = bPtr[frame] / total_weight;
  }
}

void EmptyTrace::T0EstimateToMap (std::vector<float> *sep_t0_est, Region *region, Mask *bfmask)
{
  assert (t0_map == NULL);  // no behavior defined for resetting t0_map
  int img_cols = bfmask->W(); //whole image
  if (sep_t0_est !=NULL)
  {
    t0_map = new float[region->w*region->h];
    float t0_mean =TraceHelper::ComputeT0Avg (region, bfmask, sep_t0_est, img_cols);
    TraceHelper::BuildT0Map (region, sep_t0_est,t0_mean, img_cols, t0_map);
  }
}



void EmptyTrace::DumpEmptyTrace (FILE *my_fp, int x, int y)
{
  // dump out the time-shifted empty trace value that gets computed
  for (int fnum=0; fnum<numfb; fnum++)
  {
    fprintf (my_fp, "%d\t%d\t%d\t%0.3f", x,y,fnum,bg_dc_offset[fnum]);
    for (int j=0; j<imgFrames; j++)
      fprintf (my_fp,"\t%0.3f", bg_buffers[fnum*imgFrames+j]);
    fprintf (my_fp,"\n");
  }
}

void EmptyTrace::Dump_bg_buffers(char *ss, int start, int len)
{
  TraceHelper::DumpBuffer(ss, bg_buffers, start, len);
}
