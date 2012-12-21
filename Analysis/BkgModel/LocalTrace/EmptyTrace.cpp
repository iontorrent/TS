/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <assert.h>
#include <math.h>
#include <iostream>
#include "EmptyTrace.h"
#include "BkgTrace.h"
#include "TraceClassifier.h"
#include "Stats.h"

EmptyTrace::EmptyTrace ( CommandLineOpts &clo )
{
  imgFrames = 0;
  // imgCols = 0;
  // imgRows = 0;

  neg_bg_buffers_slope = NULL;
  bg_buffers = NULL;
  bg_dc_offset = NULL;
  numfb = 0;
  t0_mean = 0;
  nRef = -1;
  secondsPerFrame = 0;
  regionIndex = -1;

  if ( clo.bkg_control.use_dud_and_empty_wells_as_reference )
    referenceMask = ( MaskType ) ( MaskReference | MaskDud );
  else
    referenceMask = MaskReference;

  do_ref_trace_trim = false;
  span_inflator_min = 0;
  span_inflator_mult = 0;
  cutoff_quantile = 0;
  nuc_flow_frame_width = 0;
  nOutliers = 0;
  trace_used = false;
}

EmptyTrace::EmptyTrace ()  // needed for serialization
{
  imgFrames = 0;
  neg_bg_buffers_slope = NULL;
  bg_buffers = NULL;
  bg_dc_offset = NULL;
  numfb = 0;
  t0_mean = 0;
  nRef = -1;
  secondsPerFrame = 0;
  regionIndex = -1;
  referenceMask = MaskReference;

  // emptyTrace outlier (wild trace) removal
  do_ref_trace_trim = false;
  span_inflator_min = 0;
  span_inflator_mult = 0;
  cutoff_quantile = 0;
  nuc_flow_frame_width = 0;
  nOutliers = 0;
  trace_used = false;
}

void EmptyTrace::Allocate ( int _numfb, int _imgFrames )
{
  // bg_buffers and neg_bg_buffers_slope
  // are contiguous arrays organized as a trace per flow in a block of flows
  assert ( _numfb > 0 );
  assert ( _imgFrames > 0 );
  assert ( bg_buffers == NULL );  //logic only checked for one-time allocation
  assert ( neg_bg_buffers_slope == NULL );
  assert ( bg_dc_offset == NULL );
  numfb = _numfb;

  imgFrames = _imgFrames;

  AllocateScratch();
}

void EmptyTrace::AllocateScratch()
{
  bg_buffers  = new float [numfb*imgFrames];
  neg_bg_buffers_slope  = new float [numfb*imgFrames];
  bg_dc_offset = new float [numfb];
  memset ( bg_dc_offset,0,sizeof ( float[numfb] ) );
}

EmptyTrace::~EmptyTrace()
{
  if ( neg_bg_buffers_slope!=NULL ) delete [] neg_bg_buffers_slope;
  if ( bg_buffers!=NULL ) delete [] bg_buffers;
  if ( bg_dc_offset!=NULL ) delete[] bg_dc_offset;
  t0_map.clear();
}


// Savitsky-Goulay filter coefficients for calculating the slope.  poly order = 2, span=+/- 2 points
const float EmptyTrace::bkg_sg_slope[BKG_SGSLOPE_LEN] = {-0.2,-0.1,0.0,0.1,0.2};

void EmptyTrace::SavitskyGolayComputeSlope ( float *local_slope,float *source_val, int len )
{
  // compute slope using savitsky golay smoother
  // pad ends of sequence with repeat values
  // assumes values placed at equally spaced points in units of frames
  int moff = ( BKG_SGSLOPE_LEN-1 ) /2;
  int mmax = len-1;
  int j;
  for ( int i=0; i< len; i++ )
    {
      local_slope[i]=0;
      // WARNING: I compute negative slope here because it is used that way later
      for ( int k=0; k<BKG_SGSLOPE_LEN; k++ )
        {
          j = i+k-moff;
          if ( j<0 ) j=0; // make sure we're in range!!!
          if ( j>mmax ) j=mmax; // make sure we're in range!!!
          local_slope[i] -= source_val[j]*bkg_sg_slope[k];
        }
    }
}


void EmptyTrace::PrecomputeBackgroundSlopeForDeriv ( int flow )
{
  // calculate the slope of the background curve at every point
  int iFlowBuffer = flowToBuffer ( flow );
  float *bsPtr = &neg_bg_buffers_slope[iFlowBuffer*imgFrames];
  float *bPtr  = &bg_buffers[iFlowBuffer*imgFrames];

  // expand into frames
  assert (secondsPerFrame > 0.0f);
  assert (imgFrames == (int)timePoints.size());

  int nFrames = (int)( (timePoints[timePoints.size()-1]+.0001)/secondsPerFrame);
  std::vector<float> frameTime(nFrames, 0);
  for (int i=0; i<nFrames; i++)
    frameTime[i] = (i+1.0f)*secondsPerFrame;

  std::vector<float> interpValues(nFrames, 0);
  TimeCompression::Interpolate(&timePoints[0], bPtr, imgFrames, &frameTime[0], &interpValues[0], nFrames);
			  
  //SavitskyGolayComputeSlope ( bsPtr,bPtr,imgFrames );
  std::vector<float> slopes(nFrames, 0);
  SavitskyGolayComputeSlope ( &slopes[0], &interpValues[0], nFrames );
  TimeCompression::Interpolate(&frameTime[0], &slopes[0], nFrames, &timePoints[0], bsPtr, imgFrames);

  /* // jgv
     char s[60000]; int n=0;
     n += sprintf(s, "SavitskyGolayComputeSlope\t%d :", regionIndex);
     for (int pt = 0;pt < imgFrames;pt++)
     n += sprintf(&s[n], "\t%f", bsPtr[pt]);

     n += sprintf(&s[n], "\nOrig Time: ");
     for (int i=0; i< timePoints.size(); i++)
     n += sprintf(&s[n], "\t%f", timePoints[i]);
     n += sprintf(&s[n], "\n\0");
     assert (n<60000);
     fprintf(stdout, "%s", s);
  */
}

// move the average empty trace in flow so it has zero mean
// for time points between t_start and t_end (units = frames, not seconds)
// data in bg_buffers changes
void EmptyTrace::RezeroReference ( float t_start, float t_end, int flow )
{
  if (t_start > t_end) {
    // code crashes unless t_start <= t_end
    // @TODO if this happens the entire region is bogus, abandon computation
    t_end = t_start;
  }

  int iFlowBuffer = flowToBuffer ( flow );

  float *bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  float dc_zero = ComputeDcOffsetEmpty ( bPtr,t_start,t_end );
  for ( int pt = 0;pt < imgFrames;pt++ )
    bPtr[pt] -= dc_zero;

  bg_dc_offset[iFlowBuffer] += dc_zero; // track this
}

void EmptyTrace::RezeroReferenceAllFlows ( float t_start, float t_end )
{
  // re-zero the traces in all flows
  for ( int fnum=0; fnum<numfb; fnum++ )
    {
      RezeroReference ( t_start, t_end, fnum );
    }
}

float EmptyTrace::ComputeDcOffsetEmpty ( float *bPtr, float t_start, float t_end )
{
  float cnt = 0.0001f;
  float dc_zero = 0.000f;

  int above_t_start = ( int ) ceil ( t_start );
  int below_t_end = ( int ) floor ( t_end );

  assert ( (0 <= above_t_start) && (above_t_start-1 < imgFrames) &&
	   (0 <= below_t_end+1) && (below_t_end < imgFrames) );

  for ( int pt = above_t_start; pt <= below_t_end; pt++ )
    {
      dc_zero += ( float ) ( bPtr[pt] );
      cnt += 1.0f;
    }

  // include values surrounding t_start & t_end weighted by overhang
  if ( above_t_start > 0 )
    {
      float overhang = ( above_t_start-t_start );
      dc_zero = dc_zero + bPtr[above_t_start-1]*overhang;
      cnt += overhang;
    }

  if ( below_t_end < ( imgFrames-1 ) )
    {
      float overhang = ( t_end-below_t_end );
      dc_zero = dc_zero + bPtr[below_t_end+1]* ( t_end-below_t_end );
      cnt += overhang;
    }
  dc_zero /= cnt;

  return ( dc_zero );
}

void EmptyTrace::GetShiftedBkg ( float tshift, TimeCompression &time_cp, float *bkg )
{
  ShiftMe ( tshift, time_cp, bg_buffers, bkg );
}

void EmptyTrace::ShiftMe (float tshift, TimeCompression &time_cp, float *my_buff, float *out_buff)
{
  for (int fnum=0;fnum<numfb;fnum++)
    {
      float *fbkg = out_buff + fnum*time_cp.npts();
      float *bg = &my_buff[fnum*imgFrames];         // get ptr to start of neighbor background
      memset (fbkg,0,sizeof (float[time_cp.npts()])); // on general principles
      // fprintf(stdout, "tshift %f\n", tshift);
      
      for (int i=0;i < time_cp.npts();i++){
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
          assert ( !isnan(fbkg[i]) );
      }
    }
}

void EmptyTrace::GetShiftedSlope ( float tshift, TimeCompression &time_cp, float *bkg )
{
  ShiftMe ( tshift, time_cp, neg_bg_buffers_slope, bkg );
}

// dummy function, returns 0s
void EmptyTrace::FillEmptyTraceFromBuffer ( short *bkg, int flow )
{
  int iFlowBuffer = flowToBuffer ( flow );
  memset ( &bg_buffers[iFlowBuffer*imgFrames],0,sizeof ( float [imgFrames] ) );

  // copy background trace, linearize it from pH domain to [] domain
  float *bPtr = &bg_buffers[iFlowBuffer*imgFrames];
  int kount = 0;
  for ( int frame=0;frame<imgFrames;frame+=1 )
    {
      bPtr[kount] += ( bkg[frame]/FRAME_AVERAGE );
      kount++;
    }
  for ( ; kount<imgFrames; kount++ )
    bPtr[kount] = bPtr[kount-1];
}

void EmptyTrace::AccumulateEmptyTrace ( float *bPtr, float *tmp_shifted, float w )
{
  int kount = 0;
  for ( int frame=0;frame<imgFrames;frame++ )
    {
      bPtr[kount] += tmp_shifted[frame] * w;
      kount++;
    }
}


void EmptyTrace::RemoveEmptyTrace ( float *bPtr, float *tmp_shifted, float w )
{
  int kount = 0;
  for ( int frame=0;frame<imgFrames;frame++ )
    {
      bPtr[kount] -= tmp_shifted[frame] * w;
      kount++;
    }
}

// Given a region, image and flow, average unpinned empty wells in this region
// using the timing in t0_map into an "average empty trace."
// Average is stored in bg_buffers
//
// WARNING!!! t0_map only matches on per-region basis.  If the EmptyTrace's idea
// of a region doesn't match BgkModel's idea of a region, the empty trace may not
// be what you think it should be

void EmptyTrace::GenerateAverageEmptyTrace ( Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, Image *img, int flow )
{
  int iFlowBuffer = flowToBuffer ( flow );
  // these are used by both the background and live-bead
  float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
  float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing

  bg_dc_offset[iFlowBuffer] = 0;  // zero out in each new block

  memset ( &bg_buffers[iFlowBuffer*imgFrames],0,sizeof ( float [imgFrames] ) );
  float *bPtr;

  float total_weight = 0.0001;
  bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  assert ( nRef >= 0 );

  int iWell = 0;
  int ix_t0 = 0, ix_t1 = 0, ix_t2 = 0;
  std::vector<float>valsAtT0, valsAtT1, valsAtT2;
  if ( do_ref_trace_trim )
    {
      valsAtT0.resize ( nRef,0 );
      valsAtT1.resize ( nRef,0 );
      valsAtT2.resize ( nRef,0 );
      ix_t0 = t0_mean;                            // start of nuc flow (approx)
      ix_t1 = ( t0_mean + nuc_flow_frame_width/2 ); // 1/2 way to the nuc_flow end
      ix_t1 = ( ix_t1 >= imgFrames ) ? ( imgFrames - 1 ) : ix_t1;
      ix_t2 = t0_mean + nuc_flow_frame_width;     // all the way to the nuc flow end
      ix_t2 = ( ix_t2 >= imgFrames ) ? ( imgFrames - 1 ) : ix_t2;
    }

  for ( int ay=region->row;ay<region->row+region->h;ay++ )
    {
      for ( int ax=region->col;ax<region->col+region->w;ax++ )
        {
          int ix = bfmask->ToIndex ( ay, ax );
          bool isStillUnpinned = ! ( pinnedInFlow.IsPinned ( flow, ix ) );
          if ( ReferenceWell ( ax,ay,bfmask ) & isStillUnpinned )   // valid reference well
            {

              TraceHelper::GetUncompressedTrace ( tmp,img,ax,ay, imgFrames );
              // shift it to account for relative timing differences - "mean zero shift"
              // ax and ay are global coordinates and need to have the region location subtracted
              // off in order to index into the region-sized t0_map
              if ( t0_map.size() > 0 )
                TraceHelper::SpecialShiftTrace ( tmp,tmp_shifted,imgFrames,t0_map[ax-region->col+ ( ay-region->row ) *region->w] );
              else
                printf ( "Alert in EmptyTrace: t0_map nonexistent\n" );

              float w=1.0;
#ifdef LIVE_WELL_WEIGHT_BG
              w = 1.0f / ( 1.0f + ( bfmask->GetNumLiveNeighbors ( ay,ax ) * 2.0f ) );
#endif
              total_weight += w;
              AccumulateEmptyTrace ( bPtr,tmp_shifted,w );
              if ( do_ref_trace_trim )
                {
                  valsAtT0[iWell] = tmp_shifted[ix_t0];
                  valsAtT1[iWell] = tmp_shifted[ix_t1];
                  valsAtT2[iWell] = tmp_shifted[ix_t2];
                  iWell++;
                }
            }
        }
    }
  // fprintf(stdout, "GenerateAverageEmptyTrace: iFlowBuffer=%d, region row=%d, region col=%d, total weight=%f, flow=%d\n", iFlowBuffer, region->row, region->col, total_weight, flow);

  //if ((regionIndex == 132) && (flow==19))
  //  fprintf(stdout, "Debug stop");
  float final_weight = total_weight;
  if ( do_ref_trace_trim )
    {
      SynchDat *sdat = NULL;
      final_weight = TrimWildTraces ( region, bPtr, valsAtT0, valsAtT1, valsAtT2, total_weight, bfmask, img, sdat );
    }

  // if ( final_weight != total_weight )
  //  fprintf (stdout, "Removed %f wild traces of %d from region %d in flow %d using t0_mean %f\n", total_weight - final_weight, nRef, regionIndex, flow, t0_mean);

  for ( int frame=0;frame < imgFrames;frame++ )
    {
      bPtr[frame] = bPtr[frame] / final_weight;
      assert ( !isnan (bPtr[frame]) );
    }
}

// given input time "seconds," return an index into time compressed trace
// nearest the time point of those with a lesser value,
// or 0 if it is less than all time points
int EmptyTrace::SecondsToIndex ( float seconds, std::vector<float>& time )
{
  // std::vector<float> cumTime(delta.size(), 0);
  // cumTime[0] = delta[0];
  // for (size_t i = 1; i < delta.size(); i++) {
  //   cumTime[i] = cumTime[i-1] + delta[i];
  // }

  if ( seconds < time[0] )
    return 0;
  std::vector<float>::iterator f = std::upper_bound ( time.begin(), time.end(), seconds );
  return ( f-time.begin() -1 );
}

// given uniform traces, i.e. non-sdat, set up timePoints
// generally treats the value v[i] as oberved at timePoints[i]
// maybe only approximately consistent with uniform data
void EmptyTrace::SetTime(float frames_per_second)
{
  assert (frames_per_second > 0);
  secondsPerFrame = 1.0f/frames_per_second;

  timePoints.resize(imgFrames,0);
  for (size_t i=0; i<timePoints.size(); i++) {
    timePoints[i] = (i+1.0f)*secondsPerFrame;  // units of seconds
  }
}  

// given traces, set up timePoints
// assumes that the corresponding value v[i] matches the midpoint
// of the interval [0, timePoints[0]] if i=0;
// otherwise is    [timePoints[i-1], timePoints[i]], if i>0
void EmptyTrace::SetTimeFromSdatChunk(Region& region, SynchDat &sdat) {
  TraceChunk &chunk = sdat.mChunks.GetItemByRowCol(region.row, region.col);
  int numUncompFrames = sdat.GetOrigUncompFrames();
  int baseFrameRate = sdat.GetBaseFrameRate();
  // copy the timing over
  // needed for interpolation
  timePoints.clear();
  timePoints.resize(numUncompFrames);
  for (int i = 0; i < numUncompFrames; i++) {
    timePoints[i] = round(i * baseFrameRate / 1000.0f);
  }
  secondsPerFrame = chunk.FramesToSeconds(1.0f);
}

// sdat has compressed data by regions which may not match the regions
// being passed in here.  Each sdat region is organized with an
// associated timing vector that matches the frame data to the timing data
// If reference traces in this region span multiple sdat regions then their
// timing may be different and will be interpolated to match the timing of
// the EmptyTrace
void EmptyTrace::GenerateAverageEmptyTrace ( Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, SynchDat &sdat, int flow )
{
  int iFlowBuffer = flowToBuffer ( flow );
  // these are used by both the background and live-bead
  bg_dc_offset[iFlowBuffer] = 0;  // zero out in each new block

  memset ( &bg_buffers[iFlowBuffer*imgFrames],0,sizeof ( float [imgFrames] ) );
  float *bPtr;
  std::vector<float> tmp ( imgFrames, 0 );

  float total_weight = 0.0001f;
  bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  assert ( nRef >= 0 );

  int ix_t0 = 0, ix_t1 = 0, ix_t2 = 0;
  std::vector<float>valsAtT0, valsAtT1, valsAtT2, samplingTimes;
  if ( do_ref_trace_trim )
    {
      valsAtT0.resize ( nRef,0 );
      valsAtT1.resize ( nRef,0 );
      valsAtT2.resize ( nRef,0 );
      ix_t0 = sampleIndex[0]; // t0
      ix_t1 = sampleIndex[1]; // 1/2 way to the nuc_flow end
      ix_t2 = sampleIndex[2];   //  nuc flow end
    }
  TraceChunk &chunk = sdat.mChunks.GetItemByRowCol(region->row, region->col);
  for ( int ay=region->row;ay<region->row+region->h;ay++ )
    {
      for ( int ax=region->col;ax<region->col+region->w;ax++ )
        {

          int ix = bfmask->ToIndex ( ay, ax );
          bool isStillUnpinned = ! ( pinnedInFlow.IsPinned ( flow, ix ) );
          if ( ReferenceWell ( ax,ay,bfmask ) & isStillUnpinned ) // valid empty well
            {
              float w=1.0f;
#ifdef LIVE_WELL_WEIGHT_BG
              w = 1.0f / ( 1.0f + ( bfmask->GetNumLiveNeighbors ( ay,ax ) * 2.0f ) );
#endif
              total_weight += w;
              for ( size_t f = 0; f < ( size_t ) chunk.mDepth; f++ ) {
                tmp[f] += w * sdat.At ( ay, ax, f );
              }
            }
        }
    }
  // fprintf(stdout, "GenerateAverageEmptyTrace: iFlowBuffer=%d, region row=%d, region col=%d, total weight=%f, flow=%d\n", iFlowBuffer, region->row, region->col, total_weight, flow);
  float final_weight = total_weight;
  if ( do_ref_trace_trim )
    {
      Image *img = NULL;
      final_weight = TrimWildTraces ( region, bPtr, valsAtT0, valsAtT1, valsAtT2, total_weight, bfmask, img, &sdat );
    }

  for ( int frame=0;frame < (int)chunk.mDepth;frame++ )
    {
      tmp[frame] = tmp[frame] / final_weight;
      assert ( !isnan (tmp[frame]) );
    }
  int numUncompFrames = sdat.GetOrigUncompFrames();//GetOrigChipFrames();
  int baseFrameRate = sdat.GetBaseFrameRate();
  
  for (int i = 0; i < numUncompFrames; i++) {
    bPtr[i] = SynchDat::InterpolateValue(&tmp[0], chunk.mTimePoints, (i * baseFrameRate) / 1000.0f);
  }

}

void EmptyTrace::GenerateAverageEmptyTraceUncomp ( TimeCompression &time_cp, Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, SynchDat &sdat, int flow )
{
  //  doingSdat = true;
  int iFlowBuffer = flowToBuffer ( flow );
  // these are used by both the background and live-bead
  bg_dc_offset[iFlowBuffer] = 0;  // zero out in each new block
  memset ( &bg_buffers[iFlowBuffer*imgFrames],0,sizeof ( float [imgFrames] ) );
  float *bPtr;
  std::vector<float> tmp ( imgFrames, 0 );

  float total_weight = 0.0001f;
  bPtr = &bg_buffers[iFlowBuffer*imgFrames];

  assert ( nRef >= 0 );

  int iWell = 0;
  int ix_t0 = 0, ix_t1 = 0, ix_t2 = 0;
  std::vector<float>valsAtT0, valsAtT1, valsAtT2, samplingTimes;
  if ( do_ref_trace_trim )
    {
      valsAtT0.resize ( nRef,0 );
      valsAtT1.resize ( nRef,0 );
      valsAtT2.resize ( nRef,0 );
      ix_t0 = sampleIndex[0]; // t0
      ix_t1 = sampleIndex[1]; // 1/2 way to the nuc_flow end
      ix_t2 = sampleIndex[2];   //  nuc flow end
    }

  for ( int ay=region->row;ay<region->row+region->h;ay++ )
    {
      for ( int ax=region->col;ax<region->col+region->w;ax++ )
        {

          int ix = bfmask->ToIndex ( ay, ax );
          bool isStillUnpinned = ! ( pinnedInFlow.IsPinned ( flow, ix ) );
          if ( ReferenceWell ( ax,ay,bfmask ) & isStillUnpinned ) // valid empty well
            {
              float w=1.0f;
#ifdef LIVE_WELL_WEIGHT_BG
              w = 1.0f / ( 1.0f + ( bfmask->GetNumLiveNeighbors ( ay,ax ) * 2.0f ) );
#endif
              total_weight += w;
              for ( size_t f = 0; f < ( size_t ) imgFrames; f++ ) {
                tmp[f] = sdat.GetTimeVal ( ay, ax, f/time_cp.frames_per_second );
              }
        
              AccumulateEmptyTrace ( bPtr, &tmp[0], w );
              if ( do_ref_trace_trim )
                {
                  valsAtT0[iWell] = tmp[ix_t0];
                  valsAtT1[iWell] = tmp[ix_t1];
                  valsAtT2[iWell] = tmp[ix_t2];
                  iWell++;
                }
            }
        }
    }
  // fprintf(stdout, "GenerateAverageEmptyTrace: iFlowBuffer=%d, region row=%d, region col=%d, total weight=%f, flow=%d\n", iFlowBuffer, region->row, region->col, total_weight, flow);
  float final_weight = total_weight;
  if ( do_ref_trace_trim )
    {
      Image *img = NULL;
      final_weight = TrimWildTraces ( region, bPtr, valsAtT0, valsAtT1, valsAtT2, total_weight, bfmask, img, &sdat );
    }

  for ( int frame=0;frame < imgFrames;frame++ )
    {
      bPtr[frame] = bPtr[frame] / final_weight;
      assert ( !isnan (bPtr[frame]) );
    }
}

void EmptyTrace::T0EstimateToMap ( std::vector<float>& sep_t0_est, Region *region, Mask *bfmask )
{
  assert ( t0_map.size() == 0 );  // no behavior defined for resetting t0_map
  int img_cols = bfmask->W(); //whole image
  if ( sep_t0_est.size() > 0 )
    {
      t0_map.resize ( region->w*region->h );
      t0_mean =TraceHelper::ComputeT0Avg ( region, bfmask, sep_t0_est, img_cols );
      TraceHelper::BuildT0Map ( region, sep_t0_est,t0_mean, img_cols, t0_map );
    }
}



void EmptyTrace::DumpEmptyTrace ( FILE *my_fp, int x, int y )
{
  // dump out the time-shifted empty trace value that gets computed
  // if this region is unused then the buffers are not initialized
  float value_to_dump = -1.0f;
  for ( int fnum=0; fnum<numfb; fnum++ )
    {
      if ( GetUsed() )
	value_to_dump = bg_dc_offset[fnum];
      fprintf ( my_fp, "%d\t%d\t%d\t%0.3f", x,y,fnum, value_to_dump);

      for ( int j=0; j<imgFrames; j++ ) {
	if (GetUsed() )
	  value_to_dump = bg_buffers[fnum*imgFrames+j];
        fprintf ( my_fp,"\t%0.3f", value_to_dump);
      }
      fprintf ( my_fp,"\n" );
    }
}

void EmptyTrace::Dump_bg_buffers ( char *ss, int start, int len )
{
  TraceHelper::DumpBuffer ( ss, bg_buffers, start, len );
}


int EmptyTrace::CountReferenceTraces ( Region& region, Mask *bfmask )
{
  int count = 0;
  for ( int ay=region.row; ay<region.row + region.h; ay++ )
    for ( int ax=region.col; ax < region.col + region.w; ax++ )

      if ( ReferenceWell ( ax, ay, bfmask ) )
        count++;

  nRef = count;

  regionIndices.resize ( nRef );
  int cnt = 0;
  for ( int ay=region.row;ay<region.row+region.h;ay++ )
    for ( int ax=region.col;ax<region.col+region.w;ax++ )

      if ( ReferenceWell ( ax, ay, bfmask ) )
        {
          int ix = bfmask->ToIndex ( ay, ax );
          regionIndices[cnt]=ix;
          cnt++;
        }
  assert ( cnt == nRef );
  return nRef;
}

void EmptyTrace::SetTrimWildTraceOptions ( bool _do_ref_trace_trim,
                                           float _span_inflator_min,
                                           float _span_inflator_mult,
                                           float _cutoff_quantile,
                                           float _nuc_flow_frame_width )
{
  do_ref_trace_trim = _do_ref_trace_trim;
  span_inflator_min = _span_inflator_min;
  span_inflator_mult = _span_inflator_mult;
  cutoff_quantile = _cutoff_quantile;
  nuc_flow_frame_width = _nuc_flow_frame_width;
}

float EmptyTrace::TrimWildTraces ( Region *region, float *bPtr,
                                   std::vector<float>& valsAtT0,
                                   std::vector<float>& valsAtT1,
                                   std::vector<float>& valsAtT2, float total_weight,
                                   Mask *bfmask, Image *img, SynchDat *sdat )
{
  // find anything really wild and trim it out
  // by adjusting values in the float array bPtr
  // returns the new weight to calculate the mean
  // algorithm is to zero with respect to frame = t0 across all traces
  // and then look at the zero-ed traces at t1 and t2 for
  // any trace that is more than span_inflator*min distance of the quantiles
  // [cutoff_quantile 1-cutoff_quantile] from the median
  // t0 should be close to where the trace values start to rise
  // do the same for frame = t2 which should be close to where the
  // traces max out, and frame = t1 which is midway
  // [t0 t2] roughly bracket the incorporation frames

  nOutliers = 0;
  int min_count = 3;
  if ( nRef < min_count )  // don't do anything
    return ( total_weight );

  float wt = total_weight;
  float w = 1.0f;

  float span_inflator = span_inflator_min + span_inflator_mult/sqrt ( ( float ) nRef );
  float cutoff_q = cutoff_quantile;
  int low = ( nRef * cutoff_q ); // at least 1 element outside range
  int hi  = nRef - low -1;
  assert ( low < hi ); // in case we change parameters

  // zero with respect to t0
  std::vector<float>::iterator it0 = valsAtT0.begin();
  std::vector<float>::iterator it1 = valsAtT1.begin();
  std::vector<float>::iterator it2 = valsAtT2.begin();

  for ( ; it0 !=valsAtT0.end() && it1 !=valsAtT1.end() && it2 !=valsAtT2.end(); ++it0, ++it1,++it2 )
    {
      *it1 = *it1 - *it0;
      *it2 = *it2 - *it0;
    }

  // find cutoffs at values in valsAtT1
  std::vector<float> v1 ( valsAtT1 );
  std::nth_element ( v1.begin(), v1.begin() +low, v1.end() );
  float v1_low = v1[low];
  float med1 = ionStats::median ( v1 );
  std::nth_element ( v1.begin(), v1.begin() +hi, v1.end() );
  float v1_hi = v1[hi];
  float m1 = ( ( med1 - v1_low ) < ( v1_hi -med1 ) ) ? ( med1 - v1_low ) : ( v1_hi -med1 );
  float span1 = m1*span_inflator;
  float v1_cutoff_low = med1 - span1;
  float v1_cutoff_hi  = med1 + span1;

  // find cutoffs at values in valsAtT2
  std::vector<float> v2 ( valsAtT2 );
  std::nth_element ( v2.begin(), v2.begin() +low, v2.end() );
  float v2_low = v2[low];
  float med2 = ionStats::median ( v2 );
  std::nth_element ( v2.begin(), v2.begin() +hi, v2.end() );
  float v2_hi = v2[hi];
  float m2 = ( ( med2 - v2_low ) < ( v2_hi -med2 ) ) ? ( med2 - v2_low ) : ( v2_hi -med2 );
  float span2 = m2*span_inflator;
  float v2_cutoff_low = med2 - span2;
  float v2_cutoff_hi  = med2 + span2;

  it1 = valsAtT1.begin();
  it2 = valsAtT2.begin();
  int i = 0;

  float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
  std::vector<float> tmp_shifted ( imgFrames,0 ); // scratch space used to time-shift data before averaging/re-compressing

  // find traces outside the cutoffs and remove them
  for ( ; it1 !=valsAtT1.end() && it2 !=valsAtT2.end(); ++it1,++it2,++i )
    {
      if ( ( *it1 < v1_cutoff_low ) || ( *it1 > v1_cutoff_hi ) ||
           ( *it2 < v2_cutoff_low ) || ( *it2 > v2_cutoff_hi ) )
        {
          int ax;
          int ay;
          bfmask->IndexToRowCol ( regionIndices[i], ay, ax );
          assert ( regionIndices[i] == bfmask->ToIndex ( ay, ax ) );
          
          if ( sdat == NULL )  {
            TraceHelper::GetUncompressedTrace ( tmp,img,ax,ay, imgFrames );
            if ( t0_map.size() > 0 )
              TraceHelper::SpecialShiftTrace ( tmp,&tmp_shifted[0],imgFrames,t0_map[ax-region->col+ ( ay-region->row ) *region->w] );
          }
          else {
            //        if ( regionAndTimingMatchesSdat )
            if(true) {
              for ( size_t f = 0; f < ( size_t ) imgFrames; f++ ) {
                tmp_shifted[f] = sdat->At ( ay, ax, f );
              }
            }
            else {
              sdat->InterpolatedAt ( ay, ax, timePoints, tmp_shifted );
            }
          }
          // remove the wild trace
          RemoveEmptyTrace ( bPtr, &tmp_shifted[0], w );
          wt -= w;
          nOutliers++;
        }
    }
  return ( ( float ) wt );
}




