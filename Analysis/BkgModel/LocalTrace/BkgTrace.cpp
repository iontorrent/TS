/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgTrace.h"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "IonErr.h"
using namespace std;
//#define DEBUG_BKTRC 1

BkgTrace::BkgTrace()
{
    fg_buffers =NULL;
    bead_trace_raw = NULL;
    bead_trace_bkg_corrected = NULL;
    fg_dc_offset = NULL;
    imgFrames = 0;
    imgRows = 0;
    imgCols = 0;
    compFrames = 0;
    time_cp = NULL;
    bead_flow_t = 0;
    numLBeads = 0;
    bead_scale_by_flow = NULL;
    restart = false;
}

void BkgTrace::Allocate (int numfb, int max_traces, int _numLBeads)
{
    bead_flow_t = max_traces;  // recompressed trace size * number of buffers
    numLBeads = _numLBeads;

    AllocateScratch();
}

void BkgTrace::AllocateScratch ()
{
    //buffers are 2D arrays where each column is single pixel's frames;
    fg_buffers  = new FG_BUFFER_TYPE [bead_flow_t*numLBeads];
    // DO NOT ALLOCATE new buffers for bkg corrected data at this time
    
    fg_dc_offset = new float [numLBeads*NUMFB];
    memset (fg_dc_offset,0,sizeof (float[numLBeads*NUMFB]));

    // hack for annoying empty traces
    // multiplicative correction per bead per flow
    bead_scale_by_flow = new float[numLBeads*NUMFB];
    
    // must have the buffers allocated here
    
    SetRawTrace(); // indicate that we are using uncorrected data by default.
}  

bool BkgTrace::NeedsAllocation()
{
  if (fg_buffers == NULL){
    // fg_buffers == NULL if and only if this object not reloaded from disk
    assert ( !restart );
    return (true);
  }
  return ( false );
}

void BkgTrace::SetRawTrace()
{
  bead_trace_raw = fg_buffers;
  bead_trace_bkg_corrected = NULL;
}

void BkgTrace::SetBkgCorrectTrace()
{
  bead_trace_raw = NULL;
  bead_trace_bkg_corrected = fg_buffers;
}

bool BkgTrace::AlreadyAdjusted()
{
  return(bead_trace_raw==NULL);
}

BkgTrace::~BkgTrace()
{
  bead_trace_raw = NULL;
  bead_trace_bkg_corrected = NULL; // unlink
  
  t0_map.clear();
  if (fg_buffers!=NULL) delete [] fg_buffers;
  if (fg_dc_offset!=NULL) delete[] fg_dc_offset;
  if (bead_scale_by_flow!=NULL) delete[] bead_scale_by_flow;

  time_cp = NULL; // unlink
}


float BkgTrace::ComputeDcOffset(FG_BUFFER_TYPE *fgPtr,float t_start, float t_end)
{
    float dc_zero = 0.000f;
    float cnt = 0.0001f;
    int pt;
    int pt1 = 0;
    int pt2 = 0;
// TODO: is this really "rezero frames before pH step start?"
// this should be compatible with i_start from the nuc rise - which may change if we change the shape???
    for (pt = 0;time_cp->frameNumber[pt] < t_end;pt++)
    {
        pt2 = pt+1;
        if (time_cp->frameNumber[pt]>t_start)
        {
            if (pt1 == 0)
                pt1 = pt; // set to first point above t_start

            dc_zero += (float) (fgPtr[pt]);
            cnt += 1.0f; // should this be frames_per_point????
            //cnt += time_cp->frames_per_point[pt];  // this somehow makes it worse????
        }
    }

    // include values surrounding t_start & t_end weighted by overhang
    if (pt1 > 0) {
        // timecp->frameNumber[pt1-1] < t_start <= timecp->frameNumber[pt1]
        // normalize to a fraction in the spirit of "this somehow makes it worse"
      float den = (time_cp->frameNumber[pt1]-time_cp->frameNumber[pt1-1]);
      if ( den > 0 ) {
        float overhang = (time_cp->frameNumber[pt1] - t_start)/den;
        dc_zero = dc_zero + fgPtr[pt1-1]*overhang;
        cnt += overhang;
      }
    }

    if ( (pt2 < time_cp->npts()) && (pt2>0) ) {
      // timecp->frameNumber[pt2-1] <= t_end < timecp->frameNumber[pt2]
      // normalize to a fraction in the spirit of "this somehow makes it worse
      float den = (time_cp->frameNumber[pt2]-time_cp->frameNumber[pt2-1]);
      if ( den > 0 ) {
	float overhang = (t_end - time_cp->frameNumber[pt2-1])/den;
	dc_zero = dc_zero + fgPtr[pt2]*overhang;
	cnt += overhang;
      }
    }

    dc_zero /= cnt;
    return(dc_zero);
}

float BkgTrace::GetBeadDCoffset(int ibd, int iFlowBuffer)
{
  return(fg_dc_offset[ibd*NUMFB+iFlowBuffer]);
}


void BkgTrace::DumpFlows(std::ostream &out) {
  for (int ibd = 0; ibd < numLBeads; ibd++) {
    for (size_t flow = 0; flow < NUMFB; flow++) {
      FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*ibd+flow*time_cp->npts()];
      out << ibd << "\t" << flow;
      for (int i = 0; i < time_cp->npts(); i++) {
        out << "\t" << fgPtr[i];
      }
      out << endl;
    }
  }
    
}

void BkgTrace::RezeroOneBead (float t_start, float t_end, int fnum, int ibd)
{
	FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*ibd+fnum*time_cp->npts()];

    float dc_zero = ComputeDcOffset(fgPtr, t_start, t_end);
    //    float dc_zero = (fgPtr[0] + fgPtr[1] + fgPtr[2] + fgPtr[3] + fgPtr[4] + fgPtr[5])/6.0f;
    for (int pt = 0;pt < time_cp->npts();pt++)   // over real data
        fgPtr[pt] -= dc_zero;

    fg_dc_offset[NUMFB*ibd+fnum] += dc_zero; // track this invisible variable
}


void BkgTrace::RezeroBeads (float t_start, float t_end, int fnum)
{
    // re-zero the traces
    for (int ibd = 0;ibd < numLBeads;ibd++)
    {
        RezeroOneBead (t_start,t_end,fnum, ibd);
    }
}

void BkgTrace::RezeroBeadsAllFlows (float t_start, float t_end)
{
    // re-zero the traces in all flows
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
        RezeroBeads (t_start,t_end,fnum);
    }
}

void TraceHelper::BuildT0Map (Region *region, std::vector<float>& sep_t0_est, float reg_t0_avg, int img_cols, std::vector<float>& output)
{
	float op;
	for (int y=0;y<region->h;y++)
    {
        for (int x=0;x<region->w;x++)
        {
            // keep track of the offset for all wells so that we can shift empty well data while loading
		   op = sep_t0_est[x+region->col+ (y+region->row) * img_cols] - reg_t0_avg;
		   if(op < 0)
			   op = 0.0f;
		   if(op > 50)
			   op = 50.0f;
		  output[y*region->w+x] = op;
        }
    }
}

// spatially organized time shifts
float TraceHelper::ComputeT0Avg (Region *region, Mask *bfmask, std::vector<float>& sep_t0_est, int img_cols)
{
    float reg_t0_avg = 0.0f;
    int t0_avg_cnt = 0;
    for (int ay=region->row; ay < region->row+region->h; ay++)
    {
        for (int ax=region->col; ax < region->col+region->w; ax++)
        {
            // uncomment if T0Estimate is ever re-calculated once images start to load
            // int ix = bfmask->ToIndex(ay, ax);
            // bool isUnpinned = (pinnedInFlow[ix] < 0) | (pinnedInFlow[ix] > flow);
            bool isUnpinned = true;
            if (isUnpinned && (!bfmask->Match (ax,ay, (MaskType) (MaskPinned | MaskIgnore | MaskExclude))))
            {
                reg_t0_avg += sep_t0_est[ax + ay * img_cols];
                t0_avg_cnt++;
            }
        }
    }
    if ( t0_avg_cnt > 0 ) reg_t0_avg /= t0_avg_cnt;
    return (reg_t0_avg);
}


void BkgTrace::T0EstimateToMap (std::vector<float>&  sep_t0_est, Region *region, Mask *bfmask)
{
  assert (t0_map.size() == 0);  // no behavior defined for resetting t0_map
  if (sep_t0_est.size() > 0)
    {
      t0_map.resize(region->w*region->h);
        float t0_mean =TraceHelper::ComputeT0Avg (region, bfmask,sep_t0_est, imgCols);
        TraceHelper::BuildT0Map (region, sep_t0_est,t0_mean, imgCols, t0_map);
    }
}


// takes trc as a pointer to an input signal of length pts and shifts it in time by frame_offset, putting the result
// in trc_out.  If frame_offset is positive, the signal is shifted left (towards lower indicies in the array)
// If frame_offset is not an integer, linear interpolation is used to construct the output values
void TraceHelper::ShiftTrace (float *trc,float *trc_out,int pts,float frame_offset)
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
void TraceHelper::SpecialShiftTrace (float *trc, float *trc_out, int pts, float frame_offset/*, int print*/)
{
    int pts_max = pts-1;

    float spt = (float) frame_offset;
    int left = (int) spt;
    int right = left+1;
    float frac = (float) right - spt;
    float afrac = 1-frac;
    int c_left;
    int c_right;
#ifdef DEBUG_BKTRC
    if(print)
    	printf("OLD: (%.2f %.2f/%.2f)",frame_offset,frac,afrac);
#endif
    for (int i=0;i < pts;i++)
    {
        c_left = left;
        c_right = right;
        if (c_left < 0) c_left = 0;
        if (c_left > pts_max) c_left = pts_max;
        if (c_right < 0) c_right = 0;
        if (c_right > pts_max) c_right = pts_max;

        trc_out[i] = trc[c_left]*frac+trc[c_right]* afrac;
#ifdef DEBUG_BKTRC
        if(print)
        	printf(" %.2f(%.2f %.2f %.2f)",trc_out[i],trc[c_left],trc[c_right],frac);
#endif
        left++;
        right++;
    }
}

void BkgTrace::FillBeadTraceFromBuffer (short *img,int iFlowBuffer)
{
    //Populate the fg_buffers buffer with livebead only image data
    for (int nbd = 0;nbd < numLBeads;nbd++)
    {
        short *wdat = img+nbd*imgFrames;

        FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*nbd+iFlowBuffer*time_cp->npts()];

        int npt = 0;
        for (int frame=0;npt < time_cp->npts();npt++)   // real data only
        {

            float avg = 0.0f;
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


void TraceHelper::DumpBuffer(char *ss, float *buffer, int start, int len)
{
    char s[60000];
    int n = 0;
    n += sprintf(&s[n], "\nSBG, %s, %d, %d: ", ss, start, len);
    for (int ti=start; ti<start+len; ti++) {
        n += sprintf(&s[n],  "%f ", buffer[ti]);
        assert(n<60000);
    }
    n += sprintf(&s[n], "\n");
    assert( n<60000 );
    fprintf(stdout, "%s", s);
}

void BkgTrace::DumpABeadOffset (int a_bead, FILE *my_fp, int offset_col, int offset_row, bead_params *cur)
{
    fprintf (my_fp, "%d\t%d", cur->x+offset_col,cur->y+offset_row); // put back into absolute chip coordinates
    for (int fnum=0; fnum<NUMFB; fnum++)
    {
      fprintf (my_fp,"\t%0.3f", fg_dc_offset[a_bead*NUMFB+fnum]);
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



void TraceHelper::GetUncompressedTrace (float *tmp, Image *img, int absolute_x, int absolute_y, int img_frames)
{
    // uncompress the trace into a scratch buffer
  //for (int frame=0;frame<img_frames;frame++)
  //  tmp[frame] = (float) img->GetInterpolatedValue (frame,absolute_x,absolute_y);
      img->GetUncompressedTrace(tmp,img_frames, absolute_x,absolute_y);
}

void BkgTrace::RecompressTrace (FG_BUFFER_TYPE *fgPtr, float *tmp_shifted/*, int print*/)
{
    int i;
    int frame = 0;
    int npt,nptsu,nptsc=time_cp->npts();
    float avg;

    // do not shift real bead wells at all
    // compress them from the frames_per_point structure in time-compression
#ifdef DEBUG_BKTRC
	if(print)
		printf("OLD: 0(%d)= ",frame);
#endif
	for (npt=0;npt < nptsc;npt++)   // real data
    {
        avg=0.0;
        nptsu = time_cp->frames_per_point[npt];
        for (i=0;i<nptsu;i++)
        {
            avg += tmp_shifted[frame+i];
#ifdef DEBUG_BKTRC
            if(print)
            	printf(" %.2f",tmp_shifted[frame+i]);
#endif
        }

        fgPtr[npt] = (FG_BUFFER_TYPE) (avg/nptsu);
        frame+=time_cp->frames_per_point[npt];
#ifdef DEBUG_BKTRC
        if(print)
        	printf(" = (%u)\nOLD: %d(%d)= ",fgPtr[npt],npt+1,frame);
#endif
        }
#ifdef DEBUG_BKTRC
    if(print)
    	printf("\n");
#endif
}

// Keep empty scale associated with trace/image data - not a per bead data object
// this lets us not allocate it if we're not using this hack.
void BkgTrace::KeepEmptyScale(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer)
{
  float ewamp =1.0f;
    bool ewscale_correct = img->isEmptyWellAmplitudeAvailable();

    if (ewscale_correct)
        ewamp = img->GetEmptyWellAmplitudeRegionAverage(region);
    for (int nbd = 0;nbd < my_beads.numLBeads;nbd++) // is this the right iterator here?
    {
        int rx = my_beads.params_nn[nbd].x;  // should x,y be stored with traces instead?
        int ry = my_beads.params_nn[nbd].y;

        if (ewscale_correct)
        {
            bead_scale_by_flow[nbd*NUMFB+iFlowBuffer] = img->getEmptyWellAmplitude(ry+region->row,rx+region->col) / ewamp;
//            my_beads.params_nn[nbd].AScale[iFlowBuffer] = img->getEmptyWellAmplitude(ry+region->row,rx+region->col) / ewamp;
        }
        else
        {
            bead_scale_by_flow[nbd*NUMFB+iFlowBuffer] = 1.0f;  // shouldn't even allocate if we're not doing image rescaling
//            my_beads.params_nn[nbd].AScale[iFlowBuffer] = 1.0f;
        }
    }
}

// Keep empty scale associated with trace/image data - not a per bead data object
// this lets us not allocate it if we're not using this hack.
void BkgTrace::KeepEmptyScale(Region *region, BeadTracker &my_beads, SynchDat &chunk, int iFlowBuffer)
{
  float ewamp =1.0f;
  for (int nbd = 0;nbd < my_beads.numLBeads;nbd++) // is this the right iterator here?
    {
      bead_scale_by_flow[nbd*NUMFB+iFlowBuffer] = ewamp;  // shouldn't even allocate if we're not doing image rescaling
    }
}


typedef float vecf_t __attribute__ ((vector_size (BKTRC_VEC_SIZE_B)));
typedef union{
	vecf_t V;
	float A[BKTRC_VEC_SIZE];
}vecf_u;

void BkgTrace::LoadImgWOffset(const RawImage *raw, int16_t *out[BKTRC_VEC_SIZE], std::vector<int> &compFrms, int nfrms, int l_coord[BKTRC_VEC_SIZE], float t0Shift/*, int print*/)
{
	int i;
	int t0ShiftWhole;
	float multT;
	float t0ShiftFrac;
	int my_frame = 0,compFrm,curFrms,curCompFrms;
	vecf_u prev;
	vecf_u next;
	vecf_u tmpAdder;
	vecf_u mult;
	vecf_u curCompFrmsV;

	int interf,lastInterf=-1;
	int16_t lastVal[BKTRC_VEC_SIZE];
	int f_coord[BKTRC_VEC_SIZE];

	if(t0Shift < 0)
		t0Shift = 0;
	if(t0Shift > (raw->uncompFrames-2))
		t0Shift = (raw->uncompFrames-2);
	t0ShiftWhole=(int)t0Shift;
	t0ShiftFrac = t0Shift - (float)t0ShiftWhole;
	// first, skip t0ShiftWhole input frames
	my_frame = raw->interpolatedFrames[t0ShiftWhole]-1;
	compFrm = 0;
	tmpAdder.V=(vecf_t){0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	curFrms=0;
	curCompFrms=compFrms[compFrm];

#ifdef DEBUG_BKTRC
	if(print)
		printf("NEW: T0=%.2f %d(%d)= ",t0Shift,compFrm,my_frame);
#endif

	while ((my_frame < raw->uncompFrames) && (compFrm < nfrms))
	{
	  interf= raw->interpolatedFrames[my_frame];

	  if(interf != lastInterf)
	  {
		  for(i=0;i<BKTRC_VEC_SIZE;i++)
		  {
			  f_coord[i] = l_coord[i]+raw->frameStride*interf;
			  next.A[i] = raw->image[f_coord[i]];
		  }
		  if(interf > 0)
		  {
			  for(i=0;i<BKTRC_VEC_SIZE;i++)
				  prev.A[i] = raw->image[f_coord[i]-raw->frameStride];
		  }
		  else
		  {
			  prev.V = next.V;
		  }
	  }

	  // interpolate
	  multT=raw->interpolatedMult[my_frame] - (t0ShiftFrac/raw->interpolatedDiv[my_frame]);
	  mult.V = (vecf_t){multT,multT,multT,multT,multT,multT,multT,multT,};
	  tmpAdder.V += ( (prev.V)-(next.V) ) * (mult.V) + (next.V);
#ifdef DEBUG_BKTRC
	  if(print)
		  printf(" %.2f",( (prev.A[0])-(next.A[0]) ) * (mult.A[0]) + (next.A[0]));
//		  printf(" %.2f(%.2f %.2f %.2f)",( (prev.A[0])-(next.A[0]) ) * (mult.A[0]) + (next.A[0]),prev.A[0],next.A[0],mult.A[0]);
#endif


	  if(++curFrms >= curCompFrms)
	  {
		  curCompFrmsV.V = (vecf_t){curCompFrms,curCompFrms,curCompFrms,curCompFrms,curCompFrms,curCompFrms,curCompFrms,curCompFrms,};
		  tmpAdder.V /= curCompFrmsV.V;
		  for(i=0;i<BKTRC_VEC_SIZE;i++)
			  out[i][compFrm] = (int16_t)(tmpAdder.A[i]);
		  compFrm++;
		  curCompFrms = compFrms[compFrm];
		  curFrms=0;
#ifdef DEBUG_BKTRC
          if(print)
        	  printf(" = (%f)\nNEW: %d(%d)= ",tmpAdder.A[0],compFrm,my_frame+1);
#endif
          tmpAdder.V = (vecf_t){0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	  }
	  my_frame++;
	}
	if(compFrm > 0 && compFrm < nfrms)
	{
		for(i=0;i<BKTRC_VEC_SIZE;i++)
			lastVal[i] = out[i][compFrm-1];
		for(;compFrm < nfrms;compFrm++)
		{
			for(i=0;i<BKTRC_VEC_SIZE;i++)
				out[i][compFrm] = lastVal[i];
		}
	}
#ifdef DEBUG_BKTRC
	if(print)
		printf("\n");
#endif
}

// Given a region, image and flow, read trace data for beads in my_beads
// for this region using the timing in t0_map from Image img
// trace data is stored in fg_buffers
void BkgTrace::GenerateAllBeadTrace (Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer)
{
    // these are used by both the background and live-bead
	int i;
	int nbdx;
	int l_coord[BKTRC_VEC_SIZE];
    int rx[BKTRC_VEC_SIZE],rxh[BKTRC_VEC_SIZE],ry[BKTRC_VEC_SIZE],ryh[BKTRC_VEC_SIZE];
    int npts=time_cp->npts();
    FG_BUFFER_TYPE *fgPtr[BKTRC_VEC_SIZE];
    float localT0;
    const RawImage *raw = img->GetImage();

#ifdef DEBUG_BKTRC
    FG_BUFFER_TYPE *fgPtrCopy[BKTRC_VEC_SIZE];
    FG_BUFFER_TYPE *fg_buffers_copy  = new FG_BUFFER_TYPE [bead_flow_t*numLBeads];
    memcpy(fg_buffers_copy,fg_buffers,bead_flow_t*numLBeads*sizeof(FG_BUFFER_TYPE));
#endif

    for (int nbd = 0;nbd < my_beads.numLBeads;nbd+=BKTRC_VEC_SIZE) // is this the right iterator here?
    {
    	localT0=0.0f;
    	for(i=0;i<BKTRC_VEC_SIZE;i++)
    	{
    		if ((nbd+i) < my_beads.numLBeads)
    			nbdx = nbd+i;
    		else
    			nbdx = my_beads.numLBeads-1;

    		rx[i] = my_beads.params_nn[nbdx].x;  // should x,y be stored with traces instead?
            rxh[i] = rx[i] + region->col;
            ry[i] = my_beads.params_nn[nbdx].y;
            ryh[i] = ry[i] + region->row;
            l_coord[i] = ryh[i]*raw->cols+rxh[i];
            fgPtr[i] = &fg_buffers[bead_flow_t*nbdx+npts*iFlowBuffer];
#ifdef DEBUG_BKTRC
            fgPtrCopy[i] = &fg_buffers_copy[bead_flow_t*nbdx+npts*iFlowBuffer];
#endif
            localT0 += t0_map[rx[i]+ry[i]*region->w];
    	}
    	localT0 /= BKTRC_VEC_SIZE;

#ifdef DEBUG_BKTRC
		for (i = 0; i < BKTRC_VEC_SIZE; i++)
		{
			float tmp[imgFrames]; // scratch space used to hold un-frame-compressed data before shifting it
			float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing
			//    			int16_t tmp_fg_buffers[npts];


			img->GetUncompressedTrace(tmp, imgFrames, rxh[i], ryh[i]);
			TraceHelper::SpecialShiftTrace(tmp, tmp_shifted, imgFrames,
					localT0/*t0_map[rx[i] + ry[i] * region->w]*/);
			RecompressTrace(fgPtrCopy[i], tmp_shifted);
		}
#endif


#if 1
    	LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0);
#endif

#ifdef DEBUG_BKTRC
#if 1
        for(i=0;i<BKTRC_VEC_SIZE;i++)
        {
			float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
			float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing
			int16_t tmp_fg_buffers[npts];

			img->GetUncompressedTrace(tmp,imgFrames, rxh[i],ryh[i]);
			// shift it by relative timing at this location
			// in this case x and y are local coordinates to the region, so they don't need to be offset
			// by the region location for indexing into t0_map
			if (t0_map.size() > 0)
				TraceHelper::SpecialShiftTrace (tmp,tmp_shifted,imgFrames,localT0);
			else
				printf ("Alert in BkgTrace: t0_map nonexistent\n");
			// enter trace into fg_buffers at coordinates bead = nbd and flow = iFlowBuffer
			RecompressTrace (tmp_fg_buffers,tmp_shifted);

			int different=0;
			for(int j=0;j<npts;j++)
			{
				if(fgPtr[i][j] > (tmp_fg_buffers[j]+1) ||
				   fgPtr[i][j] < (tmp_fg_buffers[j]-1))
				{
					different=j;
					break;
				}
			}
			if(i == 0 && different != 0)
			{

				int len = different+1;
				if(len < 30)
					len = 30;
				if(len > npts)
					len = npts;
				printf("Not the Same %d len=%d x=%d(%d) y=%d(%d)\n",different,npts,rx[i],rxh[i],ry[i],ryh[i]);
				printf("  old: ");
				for(int j=0;j<len;j++)
					printf(" %d",tmp_fg_buffers[j]);
				printf("\n  new: ");
				for(int j=0;j<len;j++)
					printf(" %d",fgPtr[i][j]);
				printf("\n");

				printf("Image Compression mult: ");
				for(int j=0;j<raw->frames;j++)
					printf(" %f",raw->interpolatedDiv[j]);
				printf("\n");
				printf("Image Compression div: ");
				for(int j=0;j<raw->frames;j++)
					printf(" %f",raw->interpolatedMult[j]);
				printf("\n");
				printf("Image Compression interpolation: ");
				for(int j=0;j<raw->frames;j++)
					printf(" %d",raw->interpolatedFrames[j]);
				printf("\n");
				printf("Bkg Compression: ");
				for(int j=0;j<time_cp->npts();j++)
					printf(" %f",(float)time_cp->frames_per_point[j]);
				printf("\n");
		    	LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0,1);
//				TraceHelper::SpecialShiftTrace (tmp,tmp_shifted,imgFrames,t0_map[rx[0]+ry[0]*region->w],1);
				RecompressTrace (tmp_fg_buffers,tmp_shifted,1);

			}
        }
#else
        for(int j=0;j<bead_flow_t*numLBeads;j++)
        {
        	if(fg_buffers[j] > (fg_buffers_copy[j]+1) ||
        		fg_buffers[j] < (fg_buffers_copy[j]-1))
        		printf("fg_buffers[%d] != %d %d\n",j,fg_buffers[j],fg_buffers_copy[j]);
        }
#endif
#endif
    }

#ifdef DEBUG_BKTRC
    delete [] fg_buffers_copy;
#endif
    KeepEmptyScale(region, my_beads,img, iFlowBuffer);

}

void BkgTrace::GenerateAllBeadTrace (Region *region, BeadTracker &my_beads, SynchDat &sdat, int iFlowBuffer, bool matchSdat)
{
  //  ION_ASSERT(chunk.NumFrames(my_beads.params_nn[0].x,my_beads.params_nn[0].y) >= (size_t)time_cp->npts(), "Wrong data points.");
  //  ION_ASSERT(chunk.NumFrames(region->row, region->col) >= (size_t)time_cp->npts(), "Wrong data points.");
  assert(matchSdat);
  TraceChunk &chunk = sdat.mChunks.GetItemByRowCol(region->row, region->col);
  assert(chunk.mRowStart == (size_t)region->row && chunk.mColStart == (size_t)region->col && 
	 chunk.mHeight == (size_t)region->h && chunk.mWidth == (size_t)region->w);
    for (int nbd = 0;nbd < my_beads.numLBeads;nbd++) // is this the right iterator here?
      {
        int rx = my_beads.params_nn[nbd].x;  // should x,y be stored with traces instead?
        int ry = my_beads.params_nn[nbd].y;

        FG_BUFFER_TYPE *fg = &fg_buffers[bead_flow_t*nbd+time_cp->npts()*iFlowBuffer];
        FG_BUFFER_TYPE *current = fg;
        if (matchSdat) {
	  int16_t *p = &chunk.mData[0] + ry * chunk.mWidth + rx;
	  for (int i = 0; i < time_cp->npts(); i++) {
	    *current++ = *p;
	    p += chunk.mFrameStep;
	  }
	}
	else {
	  std::vector<float> tmp(time_cp->npts(), 0);
	  sdat.InterpolatedAt(ry + region->row, rx + region->col, time_cp->mTimePoints, tmp);
	  for (int i = 0; i < time_cp->npts(); i++) {
	    fg[i] = tmp[i];
	  }
	}
        // float offset = BkgTrace::ComputeDcOffset(fg, *time_cp, 0.0f, time_cp->t0-2);
        // for (int i = 0; i < time_cp->npts(); i++) {
        //   fg[i] = round(fg[i] - offset);
        // }
      }
    KeepEmptyScale(region, my_beads, sdat, iFlowBuffer);
}


// accumulates a single flow of a single bead's data into the caller's buffer
void BkgTrace::AccumulateSignal (float *signal_x, int ibd, int fnum, int len)
{
  FG_BUFFER_TYPE *pfg = &fg_buffers[bead_flow_t*ibd+fnum*time_cp->npts()];

  for (int i=0; i<len; i++)
    signal_x[i] += (float) pfg[i];
}

void BkgTrace::WriteBackSignalForBead(float *signal_x, int ibd, int fnum)
{
  int fnum_start = 0;
  int len = bead_flow_t;

  if (fnum != -1)
  {
    fnum_start = fnum;
    len = time_cp->npts();
  }

  FG_BUFFER_TYPE *pfg = &fg_buffers[bead_flow_t*ibd+fnum_start*time_cp->npts()];
  for (int i=0; i < len;i++) {
    if (isfinite(signal_x[i])) {
      pfg[i] = (FG_BUFFER_TYPE)(signal_x[i] + 0.5);
    }
    else {
      ION_ABORT("Not finite.");
    }
  }
}

void CopySignalForFits (float *signal_x, FG_BUFFER_TYPE *pfg, int len)
{
    for (int i=0; i<len; i++)
        signal_x[i] = (float) pfg[i];
}

//@WARNING: Please do not overload functions for single flows and multiple flows as this
// causes bugs that are difficult to trace down.
void BkgTrace::SingleFlowFillSignalForBead(float *signal_x, int ibd, int fnum)
{

  CopySignalForFits (signal_x, &fg_buffers[bead_flow_t*ibd+fnum],time_cp->npts());
}

void BkgTrace::MultiFlowFillSignalForBead(float *signal_x, int ibd)
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

  assert( ntrace == time_cp->npts() ); // enforce whole trace for now
  if ( iFlowBuffer*time_cp->npts() >= bead_flow_t) {
    assert(  iFlowBuffer*time_cp->npts() < bead_flow_t);
  }
  FG_BUFFER_TYPE *pfg = &fg_buffers[ ibd*bead_flow_t + iFlowBuffer*time_cp->npts()];
  for (int i=0; i < ntrace; i++) {
    trace[i] = (float) pfg[i];
  }
}
