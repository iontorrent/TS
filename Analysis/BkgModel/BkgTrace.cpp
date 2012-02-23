/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgTrace.h"
#include <assert.h>

// Savitsky-Goulay filter coefficients for calculating the slope.  poly order = 2, span=+/- 2 points
const float BkgTrace::bkg_sg_slope[BKG_SGSLOPE_LEN] = {-0.2,-0.1,0.0,0.1,0.2};

void BkgTrace::SavitskyGolayComputeSlope (float *local_slope,float *source_val, int len)
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

BkgTrace::BkgTrace()
{
  neg_bg_buffers_slope = NULL;
  bg_buffers = NULL;
  fg_buffers =NULL;
  bg_dc_offset = NULL;
//  fg_dc_offset = NULL;
  imgFrames = 0;
  imgRows = 0;
  imgCols = 0;
  compFrames = 0;
  timestamps = NULL;
  time_cp = NULL;
  bead_flow_t = 0;
  t0_map = NULL;
  t0_mean = 0;
  numLBeads = 0;
}

void BkgTrace::Allocate (int numfb, int max_traces, int _numLBeads)
{
  bead_flow_t = max_traces;  // recompressed trace size * number of buffers
  numLBeads = _numLBeads;
  //buffers are 2D arrays where each column is single pixel's frames;
  bg_buffers  = new float [numfb*imgFrames];
  neg_bg_buffers_slope  = new float [numfb*imgFrames];
  fg_buffers  = new FG_BUFFER_TYPE [max_traces*numLBeads];
//  fg_dc_offset = new float [numLBeads*NUMFB];
//  memset (fg_dc_offset,0,sizeof (float[numLBeads*NUMFB]));
  bg_dc_offset = new float [NUMFB];
  memset (bg_dc_offset,0,sizeof (float[NUMFB]));
}

void BkgTrace::PrecomputeBackgroundSlopeForDeriv (int iFlowBuffer)
{
  // calculate the slope of the background curve at every point
  float *bsPtr = &neg_bg_buffers_slope[iFlowBuffer*imgFrames];
  float *bPtr  = &bg_buffers[iFlowBuffer*imgFrames];

  SavitskyGolayComputeSlope (bsPtr,bPtr,imgFrames);
}

BkgTrace::~BkgTrace()
{
  if (t0_map != NULL) delete [] t0_map;
  if (neg_bg_buffers_slope!=NULL) delete [] neg_bg_buffers_slope;
  if (bg_buffers!=NULL) delete [] bg_buffers;
  if (fg_buffers!=NULL) delete [] fg_buffers;
//  if (fg_dc_offset!=NULL) delete[] fg_dc_offset;
  if (bg_dc_offset!=NULL) delete[] bg_dc_offset;
  time_cp = NULL; // unlink
}

void BkgTrace::GetShiftedBkg (float tshift, float *bkg)
{
  ShiftMe (tshift,bg_buffers,bkg);
}

void BkgTrace::ShiftMe (float tshift,float *my_buff, float *out_buff)
{
  for (int fnum=0;fnum<NUMFB;fnum++)
  {
    float *fbkg = out_buff + fnum*time_cp->npts;
    float *bg = &my_buff[fnum*imgFrames];         // get ptr to start of neighbor background
    memset (fbkg,0,sizeof (float[time_cp->npts])); // on general principles
    for (int i=0;i < time_cp->npts;i++)
    {
      // get the frame number of this data point (might be fractional because this point could be
      // the average of several frames of data.  This number is the average time of all the averaged
      // data points
      float t=time_cp->frameNumber[i];
      float fn=t-tshift;
      if (fn < 0.0) fn = 0.0;
      if (fn > (imgFrames-2)) fn = imgFrames-2;
      int ifn= (int) fn;
      float frac = fn - ifn;

      fbkg[i] = ( (1-frac) *bg[ifn] + frac*bg[ifn+1]);
    }
  }
}

void BkgTrace::GetShiftedSlope (float tshift, float *bkg)
{
  ShiftMe (tshift,neg_bg_buffers_slope,bkg);
}

// t_offset_beads = nuc_shape.sigma
// t_offset_empty = 4.0
void BkgTrace::RezeroTraces (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum)
{
  RezeroBeads (t_start, t_mid_nuc-t_offset_beads,fnum); // do these values make sense for offsets???
  RezeroReference (t_start, t_mid_nuc-t_offset_empty,fnum);

}

void BkgTrace::RezeroTracesAllFlows (float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty)
{
  for (int fnum=0; fnum<NUMFB; fnum++)
    RezeroTraces (t_start, t_mid_nuc,t_offset_beads,t_offset_empty, fnum);
}

float BkgTrace::ComputeDcOffset(FG_BUFFER_TYPE *fgPtr,float t_start, float t_end)
{
  float dc_zero = 0.000;
  float cnt = 0.0001;
  int pt;
// TODO: is this really "rezero frames before pH step start?"
// this should be compatible with i_start from the nuc rise - which may change if we change the shape???
  for (pt = 0;time_cp->frameNumber[pt] < t_end;pt++)
  {
    if (time_cp->frameNumber[pt]>t_start)
    {
      dc_zero += (float) (fgPtr[pt]);
      cnt += 1.0; // should this be frames_per_point????
      //cnt += time_cp->frames_per_point[pt];  // this somehow makes it worse????
    }
  }

  dc_zero /= cnt;
  return(dc_zero);
}

void BkgTrace::RezeroOneBead (float t_start, float t_end, int fnum, int ibd)
{
  FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*ibd+fnum*time_cp->npts];

  float dc_zero = ComputeDcOffset(fgPtr, t_start, t_end);
  
  for (int pt = 0;pt < time_cp->npts;pt++)   // over real data
    fgPtr[pt] -= dc_zero;
  
//  fg_dc_offset[NUMFB*ibd+fnum] += dc_zero; // track this invisible variable
}


void BkgTrace::RezeroBeads (float t_start, float t_end, int fnum)
{
  // re-zero the traces
  for (int ibd = 0;ibd < numLBeads;ibd++)
  {
    RezeroOneBead (t_start,t_end,fnum, ibd);
  }
}

float BkgTrace::ComputeDcOffsetEmpty(float *bPtr, float t_start, float t_end)
{
    float cnt = 0.0001;
  float dc_zero = 0.000;
  for (int pt = 0;pt < t_end;pt++)
  {
    if (pt>t_start)
    {
      dc_zero += (float) (bPtr[pt]);
      cnt += 1.0;
    }
  }
  dc_zero /= cnt;
  return(dc_zero);
}

void BkgTrace::RezeroReference (float t_start, float t_end, int fnum)
{
  float *bPtr = &bg_buffers[fnum*imgFrames];

  float dc_zero = ComputeDcOffsetEmpty(bPtr,t_start,t_end);
  
  for (int pt = 0;pt < imgFrames;pt++)
    bPtr[pt] -= dc_zero;
  
  bg_dc_offset[fnum] += dc_zero; // track this
}

void BkgTrace::BuildT0Map (Region *region, std::vector<float> *sep_t0_est, float reg_t0_avg)
{
  t0_map = new float[region->w*region->h];

  for (int y=0;y<region->h;y++)
  {
    for (int x=0;x<region->w;x++)
    {
      // keep track of the offset for all wells so that we can shift empty well data while loading
      t0_map[y*region->w+x] = (*sep_t0_est) [x+region->col+ (y+region->row) * imgCols] - reg_t0_avg;
    }
  }
}


void BkgTrace::T0EstimateToMap (std::vector<float> *sep_t0_est, Region *region, Mask *bfmask)
{
  if (sep_t0_est !=NULL)
  {
    t0_mean =ComputeT0Avg (region,bfmask,sep_t0_est);
    BuildT0Map (region, sep_t0_est,t0_mean);
  }
}

// spatially organized time shifts
float BkgTrace::ComputeT0Avg (Region *region, Mask *bfmask, std::vector<float> *sep_t0_est)
{
  float reg_t0_avg = 0.0;
  int t0_avg_cnt = 0;
  for (int y=0;y<region->h;y++)
  {
    for (int x=0;x<region->w;x++)
    {
      if (!bfmask->Match (x+region->col,y+region->row, (MaskType) (MaskPinned | MaskIgnore | MaskExclude)))
      {
        reg_t0_avg += (*sep_t0_est) [x+region->col+ (y+region->row) * imgCols];
        t0_avg_cnt++;
      }
    }
  }
  reg_t0_avg /= t0_avg_cnt;
  return (reg_t0_avg);
}

// takes trc as a pointer to an input signal of length pts and shifts it in time by frame_offset, putting the result
// in trc_out.  If frame_offset is positive, the signal is shifted left (towards lower indicies in the array)
// If frame_offset is not an integer, linear interpolation is used to construct the output values
void ShiftTrace (float *trc,float *trc_out,int pts,float frame_offset)
{
  int pts_max = pts-1;
  for (int i=0;i < pts;i++)
  {
    float spt = (float) i+frame_offset;
    int left = (int) spt;
    int right = left+1;
    float frac = (float) right - spt;

    if (left < 0) left = 0;
    if (right < 0) right = 0;
    if (left > pts_max) left = pts_max;
    if (right > pts_max) right = pts_max;

    trc_out[i] = trc[left]*frac+trc[right]* (1-frac);
  }
}

// special cases
// time runs from 0:(pts-1)
// frame_offset never changes during this procedure
// so we can just update the left integer and keep the fraction identical
void SpecialShiftTrace (float *trc, float *trc_out, int pts, float frame_offset)
{
  int pts_max = pts-1;

  float spt = (float) frame_offset;
  int left = (int) spt;
  int right = left+1;
  float frac = (float) right - spt;
  float afrac = 1-frac;
  int c_left;
  int c_right;

  for (int i=0;i < pts;i++)
  {
    c_left = left;
    c_right = right;
    if (c_left < 0) c_left = 0;
    if (c_left > pts_max) c_left = pts_max;
    if (c_right < 0) c_right = 0;
    if (c_right > pts_max) c_right = pts_max;

    trc_out[i] = trc[c_left]*frac+trc[c_right]* afrac;
    left++;
    right++;
  }
}

void BkgTrace::FillEmptyTraceFromBuffer (short *bkg, int iFlowBuffer)
{
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

void BkgTrace::FillBeadTraceFromBuffer (short *img,int iFlowBuffer)
{
  //Populate the fg_buffers buffer with livebead only image data
  for (int nbd = 0;nbd < numLBeads;nbd++)
  {
    short *wdat = img+nbd*imgFrames;

    FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*nbd+iFlowBuffer*time_cp->npts];

    int npt = 0;
    for (int frame=0;npt < time_cp->npts;npt++)   // real data only
    {

      float avg = 0.0;
      for (int i=0;i<time_cp->frames_per_point[npt];i++)
      {
        float val = (float) wdat[frame+i];
        avg += val;
      }
      fgPtr[npt] = (FG_BUFFER_TYPE) (avg/time_cp->frames_per_point[npt]);
      frame+=time_cp->frames_per_point[npt];
    }
  }
}

void BkgTrace::AccumulateEmptyTrace (float *bPtr, float *tmp_shifted, float w)
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

void BkgTrace::GenerateAverageEmptyTrace (Region *region, short *emptyInFlow, Mask *bfmask, Image *img, int iFlowBuffer, int flow)
{
  // these are used by both the background and live-bead
  float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
  float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing

  memset (&bg_buffers[iFlowBuffer*imgFrames],0,sizeof (float [imgFrames]));
  float *bPtr;

  float total_weight = 0.0001;
  bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  assert (emptyInFlow != NULL);

  for (int ay=region->row;ay<region->row+region->h;ay++)
  {
    for (int ax=region->col;ax<region->col+region->w;ax++)
    {
      int ix = bfmask->ToIndex(ay, ax);
      if( (emptyInFlow[ix] < 0) | (emptyInFlow[ix] > flow)) // valid empty well
      {

        GetUncompressedTrace (tmp,img,ax,ay);
        // shift it to account for relative timing differences - "mean zero shift"
        // ax and ay are global coordinates and need to have the region location subtracted
        // off in order to index into the region-sized t0_map
        if (t0_map!=NULL)
          SpecialShiftTrace (tmp,tmp_shifted,imgFrames,t0_map[ax-region->col+ (ay-region->row) *region->w]);
        else
          printf ("Alert: t0_map nonexistent\n");
        
        float w=1.0;
#ifdef LIVE_WELL_WEIGHT_BG
        w = 1.0 / (1.0 + (bfmask->GetNumLiveNeighbors (ay,ax) * 2.0));
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

void BkgTrace::DumpBuffer(char *ss, float *buffer, int start, int len)
{
  // JGV
  char s[60000];
  int n = 0;
  n += sprintf(&s[n], "\nSBG, %s, %d, %d: ", ss, start, len);
  for (int ti=start; ti<start+len; ti++){
    n += sprintf(&s[n],  "%f ", buffer[ti]);
    if (n >= 60000 ){
      fprintf(stderr, "Oops, buffer overflow");
      return;
    }
    //n += sprintf(&s[n],  "%x ", *(int *)&(output[ti]));
  }
  n += sprintf(&s[n], "\n");
  assert( n<60000 );
  fprintf(stdout, "%s", s);
}

void BkgTrace::Dump_bg_buffers(char *ss, int start, int len)
{
  // JGV
  DumpBuffer(ss, bg_buffers, start, len);
}

void BkgTrace::DumpEmptyTrace (FILE *my_fp, int x, int y)
{
  // dump out the time-shifted empty trace value that gets computed
  for (int fnum=0; fnum<NUMFB; fnum++)
  {
    fprintf (my_fp, "%d\t%d\t%d\t%0.3f", x,y,fnum,bg_dc_offset[fnum]);
    for (int j=0; j<imgFrames; j++)
      fprintf (my_fp,"\t%0.3f", bg_buffers[fnum*imgFrames+j]);
    fprintf (my_fp,"\n");
  }
}

void BkgTrace::DumpABeadOffset (int a_bead, FILE *my_fp, int offset_col, int offset_row, bead_params *cur)
{
  fprintf (my_fp, "%d\t%d", cur->x+offset_col,cur->y+offset_row); // put back into absolute chip coordinates
  for (int fnum=0; fnum<NUMFB; fnum++)
  {
//    fprintf (my_fp,"\t%0.3f", fg_dc_offset[a_bead*NUMFB+fnum]);
  }
  fprintf (my_fp,"\n");
}


void BkgTrace::DumpBeadDcOffset (FILE *my_fp, bool debug_only, int DEBUG_BEAD, int offset_col, int offset_row, BeadTracker &my_beads)
{
  // either single bead or many beads
  if (numLBeads>0) // trap for dead regions
  {
    if (debug_only)
      DumpABeadOffset (DEBUG_BEAD,my_fp,offset_col, offset_row,&my_beads.params_nn[DEBUG_BEAD]);
    else
      for (int ibd=0; ibd<numLBeads; ibd++)
        DumpABeadOffset (ibd,my_fp, offset_col, offset_row, &my_beads.params_nn[ibd]);
  }
}

void BkgTrace::GetUncompressedTrace (float *tmp, Image *img, int absolute_x, int absolute_y)
{
  // uncompress the trace into a scratch buffer
  for (int frame=0;frame<imgFrames;frame++)
    tmp[frame] = (float) img->GetInterpolatedValue (frame,absolute_x,absolute_y);
}

void BkgTrace::RecompressTrace (FG_BUFFER_TYPE *fgPtr, float *tmp_shifted)
{
  int frame = 0;
  // do not shift real bead wells at all
  // compress them from the frames_per_point structure in time-compression
  for (int npt=0;npt < time_cp->npts;npt++)   // real data
  {
    float avg;
    avg=0.0;
    for (int i=0;i<time_cp->frames_per_point[npt];i++)
    {
      avg += tmp_shifted[frame+i];
    }

    fgPtr[npt] = (FG_BUFFER_TYPE) (avg/time_cp->frames_per_point[npt]);
    frame+=time_cp->frames_per_point[npt];
  }
}

void BkgTrace::GenerateAllBeadTrace (Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer)
{
  // these are used by both the background and live-bead
  float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
  float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing
  for (int nbd = 0;nbd < my_beads.numLBeads;nbd++) // is this the right iterator here?
  {
    int rx = my_beads.params_nn[nbd].x;  // should x,y be stored with traces instead?
    int ry = my_beads.params_nn[nbd].y;

    GetUncompressedTrace (tmp,img, rx+region->col, ry+region->row);

    // shift it by relative timing at this location
    // in this case x and y are local coordinates to the region, so they don't need to be offset
    // by the region location for indexing into t0_map
    if (t0_map!=NULL)
      SpecialShiftTrace (tmp,tmp_shifted,imgFrames,t0_map[rx+ry*region->w]);
    else
      printf ("Alert: t0_map nonexistent\n");

   // enter trace into fg_buffers at coordinates bead = nbd and flow = iFlowBuffer
    RecompressTrace (&fg_buffers[bead_flow_t*nbd+time_cp->npts*iFlowBuffer],tmp_shifted);
  }

}

void CopySignalForFits (float *signal_x, FG_BUFFER_TYPE *pfg, int len)
{
  for (int i=0; i<len; i++)
    signal_x[i] = (float) pfg[i];
}

void BkgTrace::FillSignalForBead(float *signal_x, int ibd)
{
  // Isolate signal extraction to trace routine
  // extract all flows for bead ibd
  CopySignalForFits (signal_x, &fg_buffers[bead_flow_t*ibd],bead_flow_t);
}

void BkgTrace::CopySignalForTrace(float *trace, int ntrace, int ibd,  int iFlowBuffer)
{
  // fg_buffers is a contiguous array organized as consecutive traces
  // in each flow buffer (total=bead_flow_t) for each bead in turn
  // |           bead 0                |           bead 1         |  ...
  // |       flow 0      | flow 1 | ...|
  // |v0 v1 (shorts) ... | 
  // |<-------    bead_flow_t  ------->|  this is the same for every bead
  // |<- time_cp->npts ->|                this is the same for every trace
  // copy the trace of flow iFlowBuffer and bead ibd into the float array 'trace'

  assert( ntrace == time_cp->npts ); // enforce whole trace for now
  if( iFlowBuffer*time_cp->npts >= bead_flow_t){
    assert(  iFlowBuffer*time_cp->npts < bead_flow_t);
  }
  FG_BUFFER_TYPE *pfg = &fg_buffers[ ibd*bead_flow_t + iFlowBuffer*time_cp->npts];
  for (int i=0; i < ntrace; i++){
    trace[i] = (float) pfg[i];
  }
}
