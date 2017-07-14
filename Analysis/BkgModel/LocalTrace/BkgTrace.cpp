/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgTrace.h"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "Vecs.h"
#include "IonErr.h"
using namespace std;
//#define DEBUG_BKTRC 1

#define WITH_SUBTRACT 1

double GenAllBtrc_time=0;
double ReZero_time=0;

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
    npts = 0;
    numLBeads = 0;
    bead_scale_by_flow = NULL;
    restart = false;
    allocated_flow_block_size = 0;
}

void BkgTrace::Allocate (int _npts, int _numLBeads, int flow_block_size)
{
    npts = _npts;  // recompressed trace size * number of buffers
    numLBeads = _numLBeads;

    AllocateScratch(flow_block_size);
}

void BkgTrace::AllocateScratch ( int flow_block_size )
{
    allocated_flow_block_size = flow_block_size;

    //buffers are 2D arrays where each column is single pixel's frames;
    fg_buffers  = new FG_BUFFER_TYPE [npts*flow_block_size*numLBeads];
    // DO NOT ALLOCATE new buffers for bkg corrected data at this time
    
    fg_dc_offset = new float [numLBeads*flow_block_size];
    memset (fg_dc_offset,0,sizeof (float[numLBeads*flow_block_size]));

    // hack for annoying empty traces
    // multiplicative correction per bead per flow
    bead_scale_by_flow = new float[numLBeads*flow_block_size];
    
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

void BkgTrace::ComputeDcOffset_params(float t_start, float t_end,
			int &pt1, int &pt2, float &cnt, float &overhang_start, float &overhang_end)
{
	cnt = 0.0001f;
	int pt;
	pt1 = -1;
	pt2 = 0;
	overhang_start = 0.0f;
	overhang_end = 0.0f;

// TODO: is this really "rezero frames before pH step start?"
// this should be compatible with i_start from the nuc rise - which may change if we change the shape???
	for (pt = 0; time_cp->frameNumber[pt] < t_end; pt++)
	{
		pt2 = pt + 1;
		if (time_cp->frameNumber[pt] > t_start)
		{
			if (pt1 == -1)
				pt1 = pt; // set to first point above t_start

			cnt += 1.0f; // should this be frames_per_point????
		}
	}

	if (pt1 < 0)
		pt1 = 0; // make sure we don't index incorrectly


	// include values surrounding t_start & t_end weighted by overhang
	else
	{
		// This part is really broken.  Fixing it makes things worse??
		//   the fraction overhang_start is > 1

		int ohpt1 = pt1 ? pt1 : 1;
		float den = (time_cp->frameNumber[ohpt1]
				- time_cp->frameNumber[ohpt1 - 1]);
		if (den > 0)
		{
			overhang_start = (time_cp->frameNumber[ohpt1] - t_start) / den;
			cnt += overhang_start;
		}
	}

	if ((pt2 < time_cp->npts()) && (pt2 > 0))
	{
		// timecp->frameNumber[pt2-1] <= t_end < timecp->frameNumber[pt2]
		// normalize to a fraction in the spirit of "this somehow makes it worse
		float den = (time_cp->frameNumber[pt2] - time_cp->frameNumber[pt2 - 1]);
		if (den > 0)
		{
			overhang_end = (t_end - time_cp->frameNumber[pt2 - 1]) / den;
			cnt += overhang_end;
		}
	}
}



//on the fly dc offset calculation during generate bead traces when trace is uncompressed
void BkgTrace::RezeroUncompressedTraceV(void *vPtrToV8F,  float t_start, float t_end)//, int ibd, int fnum)
{
  if (t_start > t_end) {
    // code crashes unless t_start <= t_end
    t_end = t_start;
  }

  //passed as void pointer to remove need for vector type awareness in header
  v8f_u * bPtr = (v8f_u *)vPtrToV8F;

  float cnt = 0.0001f;
  v8f_u tmp;  //tmp.V = LD_VEC8F(value);
  v8f_u dc_zero;
  dc_zero.V= LD_VEC8F(0.0f);
  int above_t_start = ( int ) ceil ( t_start );
  int below_t_end = ( int ) floor ( t_end );

  assert ( (0 <= above_t_start) && (above_t_start-1 < imgFrames) &&
     (0 <= below_t_end+1) && (below_t_end < imgFrames) );

  for ( int pt = above_t_start; pt <= below_t_end; pt++ )
    {
      dc_zero.V = dc_zero.V + bPtr[pt].V ;
      cnt += 1.0f;
    }

  // include values surrounding t_start & t_end weighted by overhang
  if ( above_t_start > 0 )
    {
      float overhang = above_t_start-t_start;
      tmp.V = LD_VEC8F(overhang);
      dc_zero.V = dc_zero.V + bPtr[above_t_start-1].V * tmp.V;
      cnt += overhang;
    }

  if ( below_t_end < ( imgFrames-1 ) )
    {
      float overhang = ( t_end-below_t_end );
      tmp.V = LD_VEC8F(overhang);
      dc_zero.V = dc_zero.V + bPtr[below_t_end+1].V * tmp.V;
      cnt += overhang;
    }

  tmp.V = LD_VEC8F(cnt);
  dc_zero.V = dc_zero.V / tmp.V;


  for ( int pt = 0;pt < imgFrames;pt++ )
    bPtr[pt].V = bPtr[pt].V - dc_zero.V;

  //fg_dc_offset[allocated_flow_block_size*ibd+fnum] += dc_zero;

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
  return(fg_dc_offset[ibd*allocated_flow_block_size+iFlowBuffer]);
}


void BkgTrace::DumpFlows(std::ostream &out) {
  for (int ibd = 0; ibd < numLBeads; ibd++) {
    for (int flow = 0; flow < allocated_flow_block_size; flow++) {
      FG_BUFFER_TYPE *fgPtr = &fg_buffers[npts*allocated_flow_block_size*ibd+flow*time_cp->npts()];
      out << ibd << "\t" << flow;
      for (int i = 0; i < time_cp->npts(); i++) {
        out << "\t" << fgPtr[i];
      }
      out << endl;
    }
  }
    
}

void BkgTrace::RezeroOneBead (float t_start, float t_end, int fnum, int ibd, int flow_block_size)
{
	FG_BUFFER_TYPE *fgPtr = &fg_buffers[npts*flow_block_size*ibd+fnum*time_cp->npts()];

    float dc_zero = ComputeDcOffset(fgPtr, t_start, t_end);
    //    float dc_zero = (fgPtr[0] + fgPtr[1] + fgPtr[2] + fgPtr[3] + fgPtr[4] + fgPtr[5])/6.0f;
    for (int pt = 0;pt < time_cp->npts();pt++)   // over real data
        fgPtr[pt] -= dc_zero;

    fg_dc_offset[allocated_flow_block_size*ibd+fnum] += dc_zero; // track this invisible variable
}


void BkgTrace::RezeroBeads (float t_start, float t_end, int fnum, int flow_block_size)
{
#if 0
    for (int ibd = 0;ibd < numLBeads;ibd++)
    	RezeroOneBead(t_start,t_end,fnum,ibd);
#else
    // re-zero the traces
    //
    // Identical in output to above function, just extracts logic
    //   from the inner loop to make it faster
	int pt;
	int start_pt=0;
	int end_pt=0;
	float cnt;
    float dc_zero=0.0f;// = ComputeDcOffset(fgPtr, t_start, t_end);
    float overhang_start=0.0f;
    float overhang_end=0.0f;
    int overhang_start_pt=1;
    int overhang_end_pt=1;
    FG_BUFFER_TYPE dc_zero_s;
    int nPts = time_cp->npts();
//    Timer tmr;

    ComputeDcOffset_params(t_start, t_end, start_pt, end_pt, cnt,
    		overhang_start, overhang_end);

    if(start_pt > 0)
    	overhang_start_pt = start_pt-1;
    else
    	overhang_start_pt = 0;

    if(end_pt > 0 && end_pt < nPts)
    	overhang_end_pt = end_pt;
    else
    	overhang_end_pt=0;

//    printf("%s: t_start=%f/%f start_pt=(%d/%d)/(%d/%d) cnt=%f/%f oh=%f/%f \n",
//    		__FUNCTION__,t_start,t_end,start_pt,end_pt,dstart_pt,dend_pt,cnt,dcnt,overhang_start,overhang_end);
	FG_BUFFER_TYPE *fgPtr = &fg_buffers[fnum*nPts];
    for (int ibd = 0;ibd < numLBeads;ibd++)
    {
        dc_zero=0;

        for (pt = start_pt; pt < end_pt; pt++)
			dc_zero += (fgPtr[pt]);

        // add end interpolation parts
        dc_zero += overhang_start*(fgPtr[overhang_start_pt]);
        dc_zero += overhang_end  *(fgPtr[overhang_end_pt]);

        // make it into an average
		dc_zero /= cnt;

#if 0
//DEBUG
		float dc_zero2 = ComputeDcOffset(fgPtr, t_start, t_end);
		if(dc_zero != dc_zero2)
			printf("%s: %f != %f\n",__FUNCTION__,dc_zero,dc_zero2);
//DEBUG
#endif

		// now, subtract the dc offset from all the points
		dc_zero_s = dc_zero;
        for (int pt = 0;pt < nPts;pt++)   // over real data
            fgPtr[pt] -= dc_zero_s;

        fgPtr += npts * flow_block_size;
    }

//    ReZero_time += tmr.elapsed();
#endif
}

void BkgTrace::RezeroBeadsAllFlows (float t_start, float t_end)
{
    // re-zero the traces in all flows
    for (int fnum=0; fnum<allocated_flow_block_size; fnum++)
    {
        RezeroBeads (t_start,t_end,fnum, allocated_flow_block_size);
    }
}

//T0Map can now contrain negative t0 values for faster traces
void TraceHelper::BuildT0Map (const Region *region, const std::vector<float>& sep_t0_est, float reg_t0_avg, int img_cols, std::vector<float>& output)
{
	float op;
	for (int y=0;y<region->h;y++)
    {
        for (int x=0;x<region->w;x++)
        {
            // keep track of the offset for all wells so that we can shift empty well data while loading
		  op = sep_t0_est[x+region->col+ (y+region->row) * img_cols] - reg_t0_avg;
		  output[y*region->w+x] = op;
        }
    }
}

// spatially organized time shifts
float TraceHelper::ComputeT0Avg (const Region *region, const Mask *bfmask, const std::vector<float>& sep_t0_est, int img_cols)
{
    double reg_t0_avg = 0.0f; //changed to double not to lose precision for larg regions
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
    return (float)(reg_t0_avg);
}


void BkgTrace::T0EstimateToMap (const std::vector<float>&  sep_t0_est, Region *region, Mask *bfmask)
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
        int left = floor(spt); //allow for negative shift by using floor instead of int truncation
        int right = left+1;
        float frac = (float) right - spt;
        float afrac = 1- frac;

        if (left < 0) left = 0;
        if (right < 0) right = 0;
        if (left > pts_max) left = pts_max;
        if (right > pts_max) right = pts_max;

        trc_out[i] = trc[left]*frac+trc[right]* afrac;
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
    int left = floor(spt); //allow for negative shift by using floor instead of int truncation
    int right = left+1;
    float frac, afrac;
    frac = (float) right - spt;
    afrac = 1-frac;

    int c_left, c_right;

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

// bi-directional shift trace
// takes trc as a pointer to an input signal of length pts and shifts it in time by frame_offset, putting the result
// in trc_out.  If frame_offset is positive, the signal is shifted left (towards lower indices in the array)
// if the frame_offset is negative it is shifted right towards higher indices.
// If frame_offset is not an integer, linear interpolation is used to construct the output values
void TraceHelper::ShiftTraceBiDirect (float *trc, float *trc_out, int pts, float frame_offset)
{

  float shift = -frame_offset; //shift tract in opposite direction of to correctframe_offset

  if(shift != 0){

      int shiftwhole = (int)shift;  //Truncate to get number of whole frames to shift.

      int nearFrame = -shiftwhole;   //determine the closer of the two frames to interpolate in-between
      int farFrame = nearFrame + ((shift < 0)?(1):(-1));  //determine the frame further away. interpolate between near and far

      float farFrac =  abs(shift-(float)shiftwhole);  //determine fraction of far frame
      float nearFrac = 1.0f - farFrac;  //and fraction of near frame used for interpolation

  //  cout << "nearFrame "<< nearFrame <<" nearFrac "<< nearFrac <<" farFrame " << farFrame <<" farFrac "<<farFrac <<endl;

      int lastframe = pts-1;  // useful input frames range from 0 to frames-1

      for(int i=0; i<pts; i++){

        int nframe = nearFrame;
        int fframe = farFrame;

        if(nframe < 0 || fframe < 0) // if  near- or far-Frame below lower boundary both are set
          nframe = fframe = 0;

        if(nframe > lastframe || fframe > lastframe) //handle right boundary, use last frame for left and right when right is out of bounds
          nframe = fframe = lastframe;

        trc_out[i] =trc[nframe]*nearFrac + trc[fframe]*farFrac;

        nearFrame++;
        farFrame++;

      }
    }else{
      for(int i=0; i<pts; i++){
        trc_out[i] = trc[i];
      }
    }
}

void TraceHelper::ShiftTraceBiDirect_vec (void *trc_v8f_u, void *trc_out_v8f_u, int pts, float frame_offset)
{

  v8f_u * trc = (v8f_u*)trc_v8f_u;
  v8f_u * trc_out = (v8f_u*)trc_out_v8f_u;


  float shift = -frame_offset; //shift tract in opposite direction of to correctframe_offset

  if(shift != 0){

      int lastframe = pts-1;  // useful input frames range from 0 to frames-1

      int shiftwhole = (int)shift;  //Truncate to get number of whole frames to shift.

      int nearFrame = -shiftwhole;   //determine the closer of the two frames to interpolate in-between
      int farFrame = nearFrame + ((shift < 0)?(1):(-1));  //determine the frame further away. interpolate between near and far

      float far =  abs(shift-(float)shiftwhole);  //determine fraction of far frame
      float near = 1.0f - far;  //and fraction of near frame used for interpolation
      v8f_u farFrac;
      v8f_u nearFrac;
      farFrac.V = LD_VEC8F(far);
      nearFrac.V = LD_VEC8F(near);

  //  cout << "nearFrame "<< nearFrame <<" nearFrac "<< nearFrac <<" farFrame " << farFrame <<" farFrac "<<farFrac <<endl;

      for(int i=0; i<pts; i++){

        int nframe = nearFrame;
        int fframe = farFrame;

        if(nframe < 0 || fframe < 0) // if  near- or far-Frame below lower boundary both are set
          nframe = fframe = 0;

        if(nframe > lastframe || fframe > lastframe) //handle right boundary, use last frame for left and right when right is out of bounds
          nframe = fframe = lastframe;

        trc_out[i].V =trc[nframe].V*nearFrac.V + trc[fframe].V*farFrac.V;

        nearFrame++;
        farFrame++;

      }
    }else{
      for(int i=0; i<pts; i++){
        trc_out[i].V = trc[i].V;
      }
    }
}

void BkgTrace::FillBeadTraceFromBuffer (short *img,int iFlowBuffer, int flow_block_size)
{
    //Populate the fg_buffers buffer with livebead only image data
    for (int nbd = 0;nbd < numLBeads;nbd++)
    {
        short *wdat = img+nbd*imgFrames;

        FG_BUFFER_TYPE *fgPtr = &fg_buffers[npts*flow_block_size*nbd+iFlowBuffer*time_cp->npts()];

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

void BkgTrace::DumpABeadOffset (int a_bead, FILE *my_fp, int offset_col, int offset_row, BeadParams *cur)
{
    fprintf (my_fp, "%d\t%d", cur->x+offset_col,cur->y+offset_row); // put back into absolute chip coordinates
    for (int fnum=0; fnum<allocated_flow_block_size; fnum++)
    {
      fprintf (my_fp,"\t%0.3f", fg_dc_offset[a_bead*allocated_flow_block_size+fnum]);
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
            bead_scale_by_flow[nbd*allocated_flow_block_size+iFlowBuffer] = img->getEmptyWellAmplitude(ry+region->row,rx+region->col) / ewamp;
//            my_beads.params_nn[nbd].AScale[iFlowBuffer] = img->getEmptyWellAmplitude(ry+region->row,rx+region->col) / ewamp;
        }
        else
        {
            bead_scale_by_flow[nbd*allocated_flow_block_size+iFlowBuffer] = 1.0f;  // shouldn't even allocate if we're not doing image rescaling
//            my_beads.params_nn[nbd].AScale[iFlowBuffer] = 1.0f;
        }
    }
}


void BkgTrace::LoadImgWOffset(const RawImage *raw, int16_t *out[VEC8_SIZE], std::vector<int> &compFrms, int nfrms, int l_coord[VEC8_SIZE], float t0Shift/*, int print*/)
{
	int i;
	int t0ShiftWhole;
	float multT;
	float t0ShiftFrac;
	int my_frame = 0,compFrm,curFrms,curCompFrms;
	v8f_u prev;
	v8f_u next;
	v8f_u tmpAdder;
	v8f_u mult;
	v8f_u curCompFrmsV;

	int interf,lastInterf=-1;
	int16_t lastVal[VEC8_SIZE];
	int f_coord[VEC8_SIZE];

	//allow for negative t0Shift (faster traces)
        if(t0Shift < 0-(raw->uncompFrames-2))
          t0Shift = 0-(raw->uncompFrames-2);
	if(t0Shift > (raw->uncompFrames-2))
          t0Shift = (raw->uncompFrames-2);

  //by using floor() instead of (int) here
  //we now can allow for negative t0Shifts  
	t0ShiftWhole=floor(t0Shift); 
	t0ShiftFrac = t0Shift - (float)t0ShiftWhole;

	// skip t0ShiftWhole input frames,
	// if T0Shift whole < 0 start at frame 0;
	int StartAtFrame = (t0ShiftWhole < 0)?(0):(t0ShiftWhole);

	my_frame = raw->interpolatedFrames[StartAtFrame]-1;
	compFrm = 0;
	tmpAdder.V=LD_VEC8F(0.0f);
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
		  for(i=0;i<VEC8_SIZE;i++)
		  {
			  f_coord[i] = l_coord[i]+raw->frameStride*interf;
			  next.A[i] = raw->image[f_coord[i]];
		  }
		  if(interf > 0)
		  {
			  for(i=0;i<VEC8_SIZE;i++)
				  prev.A[i] = raw->image[f_coord[i]-raw->frameStride];
		  }
		  else
		  {
			  prev.V = next.V;
		  }
	  }

	  // interpolate
	  multT=raw->interpolatedMult[my_frame] - (t0ShiftFrac/raw->interpolatedDiv[my_frame]);
	  mult.V = LD_VEC8F(multT);
	  tmpAdder.V += ( (prev.V)-(next.V) ) * (mult.V) + (next.V);
#ifdef DEBUG_BKTRC
	  if(print)
		  printf(" %.2f",( (prev.A[0])-(next.A[0]) ) * (mult.A[0]) + (next.A[0]));
//		  printf(" %.2f(%.2f %.2f %.2f)",( (prev.A[0])-(next.A[0]) ) * (mult.A[0]) + (next.A[0]),prev.A[0],next.A[0],mult.A[0]);
#endif


	  if(++curFrms >= curCompFrms)
	  {
		  curCompFrmsV.V = LD_VEC8F((float)curCompFrms);
//		  tmpAdder.V *= LD_VEC8F(2.0f);
		  tmpAdder.V /= curCompFrmsV.V;
		  for(i=0;i<VEC8_SIZE;i++)
			  out[i][compFrm] = (int16_t)(tmpAdder.A[i]);
		  compFrm++;
		  curCompFrms = compFrms[compFrm];
		  curFrms=0;
#ifdef DEBUG_BKTRC
          if(print)
        	  printf(" = (%f)\nNEW: %d(%d)= ",tmpAdder.A[0],compFrm,my_frame+1);
#endif
          tmpAdder.V = LD_VEC8F(0.0f);
	  }

	  //reuse my_frame while not compensated for negative t0 shifts
    //T0ShiftWhole will be < 0 for negative t0 
	  if(t0ShiftWhole < 0)
	    t0ShiftWhole++;
	  else
	    my_frame++;

	}
	if(compFrm > 0 && compFrm < nfrms)
	{
		for(i=0;i<VEC8_SIZE;i++)
			lastVal[i] = out[i][compFrm-1];
		for(;compFrm < nfrms;compFrm++)
		{
			for(i=0;i<VEC8_SIZE;i++)
				out[i][compFrm] = lastVal[i];
		}
	}
#ifdef DEBUG_BKTRC
	if(print)
		printf("\n");
#endif
}


void BkgTrace::LoadImgWRezeroOffset(const RawImage *raw, int16_t *out[VEC8_SIZE],
                                    std::vector<int> &compFrms, int nfrms,
                                    int l_coord[VEC8_SIZE], float t0Shift,
                                    float t_start, float t_end)
{
 // static int debugoutput = 0;
  int i;
  int t0ShiftWhole;
  float multT;
  float t0ShiftFrac;
  int my_frame = 0,compFrm,curFrms,curCompFrms;
  v8f_u prev;
  v8f_u next;
  v8f_u dcOffset;
  v8f_u tmpAdder;
  v8f_u mult;
  v8f_u curCompFrmsV;
  v8f_u * uncompTrace = NULL;
  v8f_u * tmpTrace = NULL;
  int interf,lastInterf=-1;
  int16_t lastVal[VEC8_SIZE];
  int f_coord[VEC8_SIZE];

  //allow for negative t0Shift (faster traces)
        if(t0Shift < 0-(raw->uncompFrames-2))
          t0Shift = 0-(raw->uncompFrames-2);
  if(t0Shift > (raw->uncompFrames-2))
          t0Shift = (raw->uncompFrames-2);

  uncompTrace = new v8f_u[raw->uncompFrames+2];
  tmpTrace = uncompTrace + 2;
  //by using floor() instead of (int) here
  //we now can allow for negative t0Shifts
  //t0ShiftWhole=floor(t0Shift);
  //t0ShiftFrac = t0Shift - (float)t0ShiftWhole;

  // skip t0ShiftWhole input frames,
  // if T0Shift whole < 0 start at frame 0;
  //int StartAtFrame = (t0ShiftWhole < 0)?(0):(t0ShiftWhole);

  //my_frame = raw->interpolatedFrames[StartAtFrame]-1;
  compFrm = 0;
  tmpAdder.V=LD_VEC8F(0.0f);
  curFrms=0;
  curCompFrms=compFrms[compFrm];

#ifdef DEBUG_BKTRC
  if(print)
    printf("NEW: T0=%.2f %d(%d)= ",t0Shift,compFrm,my_frame);
#endif

  //uncompress
  for( int f=0; f<imgFrames; f++)
  {
    interf= raw->interpolatedFrames[f];

    if(interf != lastInterf)
    {
      for(i=0;i<VEC8_SIZE;i++)
      {
        f_coord[i] = l_coord[i]+raw->frameStride*interf;
        next.A[i] = raw->image[f_coord[i]];
      }
      if(interf > 0)
      {
        for(i=0;i<VEC8_SIZE;i++)
          prev.A[i] = raw->image[f_coord[i]-raw->frameStride];
      }
      else
      {
        prev.V = next.V;
      }
    }

    // interpolate
    multT=raw->interpolatedMult[f]; // - (t0ShiftFrac/raw->interpolatedDiv[f]);
    mult.V = LD_VEC8F(multT);
    tmpTrace[f].V =  ( (prev.V)-(next.V) ) * (mult.V) + (next.V);
  }

/*  if(DEBUGcnt == 0 && debugoutput < 32){
  for(int bi=0; bi < VEC8_SIZE; bi ++){
    for(int frnum = 0; frnum < imgFrames; frnum++ ){
      printf("%f,", uncompTrace[frnum].A[bi]);
    }
    printf("\n");
    debugoutput++;
  }
  }
  */

  TraceHelper::ShiftTraceBiDirect_vec ((void*)tmpTrace, (void *)uncompTrace, imgFrames, t0Shift);
  RezeroUncompressedTraceV( (void*)uncompTrace, t_start, t_end);

  //recompress
  my_frame = 0;
  while ((my_frame < raw->uncompFrames) && (compFrm < nfrms))
    {
      tmpAdder.V = tmpAdder.V +  uncompTrace[my_frame].V;

#ifdef DEBUG_BKTRC
    if(print)
      printf(" %.2f",( (prev.A[0])-(next.A[0]) ) * (mult.A[0]) + (next.A[0]));
//      printf(" %.2f(%.2f %.2f %.2f)",( (prev.A[0])-(next.A[0]) ) * (mult.A[0]) + (next.A[0]),prev.A[0],next.A[0],mult.A[0]);
#endif

    if(++curFrms >= curCompFrms)
    {
      curCompFrmsV.V = LD_VEC8F((float)curCompFrms);
//      tmpAdder.V *= LD_VEC8F(2.0f);
      tmpAdder.V = tmpAdder.V / curCompFrmsV.V;
      for(i=0;i<VEC8_SIZE;i++)
        out[i][compFrm] = (int16_t)(tmpAdder.A[i]);
      compFrm++;
      curCompFrms = compFrms[compFrm];
      curFrms=0;
#ifdef DEBUG_BKTRC
          if(print)
            printf(" = (%f)\nNEW: %d(%d)= ",tmpAdder.A[0],compFrm,my_frame+1);
#endif
          tmpAdder.V = LD_VEC8F(0.0f);
    }

    //reuse my_frame while not compensated for negative t0 shifts
    //T0ShiftWhole will be < 0 for negative t0
    //if(t0ShiftWhole < 0)
     // t0ShiftWhole++;
    //else
      my_frame++;

  }


  if(compFrm > 0 && compFrm < nfrms)
  {
    for(i=0;i<VEC8_SIZE;i++)
      lastVal[i] = out[i][compFrm-1];
    for(;compFrm < nfrms;compFrm++)
    {
      for(i=0;i<VEC8_SIZE;i++)
        out[i][compFrm] = lastVal[i];
    }
  }

/*
  if(DEBUGcnt == 0 && debugoutput < 32){
    for(int bi=0; bi < VEC8_SIZE; bi ++){
      for(int frnum = 0; frnum < nfrms; frnum++ ){
        printf("%hd,", out[bi][frnum]);
      }
      printf("\n");
      debugoutput++;
    }
    }
*/


  delete[] uncompTrace;

#ifdef DEBUG_BKTRC
  if(print)
    printf("\n");
#endif
}

//#define BEADTRACE_CHKBOTH_DBG 1
void BkgTrace::GenerateAllBeadTrace (Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer,
		int flow_block_size, float t_start, float t_end)
{
#ifndef BEADTRACE_CHKBOTH_DBG
//	Timer t;
//	t.restart();
	if(/*0 && */(((region->w % VEC8_SIZE) == 0) &&
	   ((img->GetImage()->cols % VEC8_SIZE) == 0))) {// check that pointers are alligned too.
	   GenerateAllBeadTrace_vec(region,my_beads,img,iFlowBuffer,fg_buffers, flow_block_size, t_start, t_end);
#ifndef WITH_SUBTRACT
	   RezeroBeads (t_start, t_end, iFlowBuffer,flow_block_size);
#endif
	}else{
	   GenerateAllBeadTrace_nonvec(region,my_beads,img,iFlowBuffer,fg_buffers, flow_block_size);
	   RezeroBeads (t_start, t_end, iFlowBuffer,flow_block_size);
   }

//	cout << "Generate Bead Trace: " << t.elapsed() << "s" << endl;

#else
    FG_BUFFER_TYPE *fg_buffers2 = new FG_BUFFER_TYPE [npts*flow_block_size*numLBeads];

    GenerateAllBeadTrace_nonvec(region,my_beads,img,iFlowBuffer,fg_buffers);
    GenerateAllBeadTrace_vec(region,my_beads,img,iFlowBuffer,fg_buffers2);


#define DBG_ROW 448
#define DBG_COL 864
	// check the results...
    if(region->row == DBG_ROW && region->col == DBG_COL /*&& iFlowBuffer == 19*/)
    {
	  for (int ibd = 0; ibd < numLBeads; ibd++) {
//	    for (size_t flow = 0; flow < allocated_flow_block_size; flow++)
		  {
		      FG_BUFFER_TYPE *fgPtr1 = &fg_buffers[npts*flow_block_size*ibd+iFlowBuffer*time_cp->npts()];
		      FG_BUFFER_TYPE *fgPtr2 = &fg_buffers2[npts*flow_block_size*ibd+iFlowBuffer*time_cp->npts()];

		      for (int i = 0; i < time_cp->npts(); i++) {
		    	  if(fgPtr1[i] > (fgPtr2[i]+5) || fgPtr1[i] < (fgPtr2[i]-5))
		    		  printf(" %d/%d/%d: (%d/%d) 1)%d 2)%d \n",ibd,iFlowBuffer,i,
		    				  my_beads.params_nn[ibd].y,my_beads.params_nn[ibd].x,fgPtr1[i],fgPtr2[i]);
		      }
	    }
	  }

	}
    delete [] fg_buffers2;
#endif
}

// Given a region, image and flow, read trace data for beads in my_beads
// for this region using the timing in t0_map from Image img
// trace data is stored in fg_buffers
void BkgTrace::GenerateAllBeadTrace_nonvec (Region *region, BeadTracker &my_beads, Image *img,
		int iFlowBuffer, FG_BUFFER_TYPE *fgb, int flow_block_size)
{
    // these are used by both the background and live-bead
	int i;
	int nbdx;
	int l_coord[VEC8_SIZE];
    int rx[VEC8_SIZE],rxh[VEC8_SIZE],ry[VEC8_SIZE],ryh[VEC8_SIZE];
    int npts=time_cp->npts();
    FG_BUFFER_TYPE *fgPtr[VEC8_SIZE];
    float localT0=0.0f;
    const RawImage *raw = img->GetImage();


    for (int nbd = 0;nbd < my_beads.numLBeads;nbd+=VEC8_SIZE) // is this the right iterator here?
    {
    	localT0=0.0f;
    	for(i=0;i<VEC8_SIZE;i++)
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
            fgPtr[i] = &fgb[npts*flow_block_size*nbdx+npts*iFlowBuffer];
            localT0 += t0_map[rx[i]+ry[i]*region->w];
    	}
    	localT0 /= VEC8_SIZE;


    	LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0);

    }

    KeepEmptyScale(region, my_beads,img, iFlowBuffer);
#ifdef BEADTRACE_DBG
//#if 1
        if(region->row == DBG_ROW && region->col == DBG_COL /*&& iFlowBuffer == 19*/)
        {
    	for (int ibd = 0; ibd < /*numLBeads*/2; ibd++)
    	{
    		for (size_t flow = 0; flow < allocated_flow_block_size; flow++)
    		{
    			FG_BUFFER_TYPE *fgPtr = &fgb[npts * flow_block_size * ibd
    					+ flow * time_cp->npts()];
    			printf("NV %d/%d(%d/%d %lf): %d/%d ",ibd,flow,my_beads.params_nn[ibd].y,my_beads.params_nn[ibd].x,t0_map[0],region->row,region->col);
    			for (int i = 0; i < time_cp->npts(); i++)
    			{
    				printf(" %d",fgPtr[i]);
    			}
    			printf("\n");
    		}
    	}
        }
    #endif

}
// Given a region, image and flow, read trace data for beads in my_beads
// for this region using the timing in t0_map from Image img
// trace data is stored in fg_buffers
void BkgTrace::GenerateAllBeadTraceAnRezero (Region *region, BeadTracker &my_beads, Image *img,
    int iFlowBuffer, int flow_block_size, float t_start, float t_end)
{
    // these are used by both the background and live-bead

  int i;
  int nbdx;
  int l_coord[VEC8_SIZE];
    int rx[VEC8_SIZE],rxh[VEC8_SIZE],ry[VEC8_SIZE],ryh[VEC8_SIZE];
    int npts=time_cp->npts();
    FG_BUFFER_TYPE *fgPtr[VEC8_SIZE];
    float localT0=0.0f;
    const RawImage *raw = img->GetImage();

    for (int nbd = 0;nbd < my_beads.numLBeads;nbd+=VEC8_SIZE) // is this the right iterator here?
    {
      localT0=0.0f;
      for(i=0;i<VEC8_SIZE;i++)
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
            fgPtr[i] = &fg_buffers[npts*flow_block_size*nbdx+npts*iFlowBuffer];
            localT0 += t0_map[rx[i]+ry[i]*region->w];
      }
      localT0 /= VEC8_SIZE;

      LoadImgWRezeroOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0, t_start, t_end);
      //LoadImgWOffset(raw, fgPtr, time_cp->frames_per_point, npts, l_coord, localT0);

    }


    KeepEmptyScale(region, my_beads,img, iFlowBuffer);
#ifdef BEADTRACE_DBG
//#if 1
        if(region->row == DBG_ROW && region->col == DBG_COL /*&& iFlowBuffer == 19*/)
        {
      for (int ibd = 0; ibd < /*numLBeads*/2; ibd++)
      {
        for (size_t flow = 0; flow < allocated_flow_block_size; flow++)
        {
          FG_BUFFER_TYPE *fgPtr = &fgb[npts * flow_block_size * ibd
              + flow * time_cp->npts()];
          printf("NV %d/%d(%d/%d %lf): %d/%d ",ibd,flow,my_beads.params_nn[ibd].y,my_beads.params_nn[ibd].x,t0_map[0],region->row,region->col);
          for (int i = 0; i < time_cp->npts(); i++)
          {
            printf(" %d",fgPtr[i]);
          }
          printf("\n");
        }
      }
        }
    #endif

}








void BkgTrace::GenerateAllBeadTrace_vec (Region *region, BeadTracker &my_beads,
		Image *img, int iFlowBuffer, FG_BUFFER_TYPE *fgb, int flow_block_size,
		float t_start, float t_end)
{
    // these are used by both the background and live-bead
	uint k;
	int nbdx=0;
    int npts=time_cp->npts();
    FG_BUFFER_TYPE *fgPtr  __attribute__ ((aligned(16)));
	int16_t *sptr  __attribute__ ((aligned(16)));
	int16_t *imgPtr  __attribute__ ((aligned(16)));
    const RawImage *raw = img->GetImage();
    float localT0;
    int x,y;
	int t0ShiftWhole=0;
	float t0ShiftFrac=0;
	int my_frame = 0,compFrm,curFrms,curCompFrms;

#define MY_VEC_SIZE 8
#define MY_VECF v8f_u
#define MY_VECS v8s_u


	MY_VECF prev;
	MY_VECF next;
	MY_VECF dummy={};
	MY_VECF tmpAdder;
#ifdef WITH_SUBTRACT
	MY_VECF tmpTrace[npts];
	MY_VECS tmpTraceS[npts];
#endif

	int frameStride=raw->rows*raw->cols;
	int interf,lastInterf=-1;
	int storeIdx[MY_VEC_SIZE];
    Timer tmr;

#ifdef WITH_SUBTRACT
	int start_pt=0;
	int end_pt=0;
	float dc_cnt;
    float overhang_start=0.0f;
    float overhang_end=0.0f;
    int overhang_start_pt=1;
    int overhang_end_pt=1;

    ComputeDcOffset_params(t_start, t_end, start_pt, end_pt, dc_cnt,
    		overhang_start, overhang_end);

    if(start_pt > 0)
    	overhang_start_pt = start_pt-1;
    else
    	overhang_start_pt = 0;

    if(end_pt > 0 && end_pt < npts)
    	overhang_end_pt = end_pt;
    else
    	overhang_end_pt=0;
	MY_VECF ohsv;
	MY_VECF ohev;
	ohsv.V = dummy.V + overhang_start;
    ohev.V = dummy.V + overhang_end;
#endif

	fgPtr = &fgb[npts * flow_block_size*nbdx+npts*iFlowBuffer];
    for (y = 0; y < region->h; y++)
    {
    	imgPtr = &raw->image[(y+region->row)*raw->cols+region->col];
        for (x = 0;x < region->w && nbdx < numLBeads; x+=MY_VEC_SIZE,imgPtr+=MY_VEC_SIZE)
        {
    		int incr=0;
    		int nbdxCopy=nbdx;
        	float localT0Cnt=0.0f;
        	localT0=0.0f;
			for(k=0;k<MY_VEC_SIZE;k++)
				storeIdx[k]=-1; // initialize to not used..

			for(k=0;k<MY_VEC_SIZE && nbdx < my_beads.numLBeads;k++)
			{
				if ((my_beads.params_nn[nbdx].x == (x+(int)k)) &&
					(my_beads.params_nn[nbdx].y == y))
				{
					localT0 += t0_map[y*region->w+(x+k)]; //BUG!! t0_map is a 2D map containing all beads not only live beads.
					localT0Cnt += 1.0f;
					storeIdx[k]=incr++;
					nbdx++;
				}else{
//	   		        printf("%d/%d: %f\n",y,x+k,t0_map[y*region->w+(x+k)]);
				}
			}
    		if(nbdxCopy == nbdx)
    			continue; // there are no live beads in this chunk

    		localT0 /= localT0Cnt;

		    //allow for negative t0shift, faster traces
    		if(localT0 < 0-(raw->uncompFrames-2))
                  localT0 = 0-(raw->uncompFrames-2);
    		if(localT0 > (raw->uncompFrames-2))
    			localT0 = (raw->uncompFrames-2);

            //by using floor() instead of (int) here
            //we now can allow for negative t0Shifts
    		t0ShiftWhole=floor(localT0);
    		t0ShiftFrac = localT0 - (float)t0ShiftWhole;

    		// if T0Shift whole < 0 start at frame 0;
    		int StartAtFrame = (t0ShiftWhole < 0)?(0):(t0ShiftWhole);

    		my_frame = raw->interpolatedFrames[StartAtFrame]-1;
    		compFrm = 0;
			tmpAdder.V=0.0f+dummy.V;
			prev.V=0.0f+dummy.V;
    		curFrms=0;
    		curCompFrms=time_cp->frames_per_point[compFrm];


    		interf= raw->interpolatedFrames[my_frame];
			sptr = &imgPtr[interf*frameStride];

			LD_VS_VF(sptr, next);

			while ((my_frame < raw->uncompFrames) && (compFrm < npts)) {
				interf= raw->interpolatedFrames[my_frame];

				if(interf != lastInterf) {
					sptr = &imgPtr[interf*frameStride];
					prev.V = next.V;
					LD_VS_VF(sptr, next);
				}

				// interpolate
				MY_VECF mult;
				float tmpMult = (raw->interpolatedMult[my_frame]
														- (t0ShiftFrac/raw->interpolatedDiv[my_frame]));
				mult.V = dummy.V + tmpMult;
				//BC_VEC(mult,&tmpMult);
				tmpAdder.V += ((prev.V)-(next.V) ) * (mult.V) + (next.V);
				if(++curFrms >= curCompFrms) {
					tmpAdder.V = tmpAdder.V / ((float )curCompFrms);

#ifdef WITH_SUBTRACT
					tmpTrace[compFrm].V = tmpAdder.V;
#else
    			  MY_VECS svalV;
    			  CVT_VF_VS(svalV,tmpAdder);

    			  for(k=0;k<MY_VEC_SIZE;k++)
    			  {
    				  if(storeIdx[k] >= 0)
    					  fgPtr[storeIdx[k]*npts*flow_block_size+compFrm] = svalV.A[k];
    			  }
#endif
					compFrm++;
					curCompFrms = time_cp->frames_per_point[compFrm];
					curFrms=0;
					tmpAdder.V = dummy.V + 0.0f;
				}
				//reuse current my_frame while not compensated for negative t0 shift
				if(t0ShiftWhole < 0)
					t0ShiftWhole++;
				else
					my_frame++;

			}
    		if(compFrm > 0 && compFrm < npts)
    		{
    			for(;compFrm < npts;compFrm++)
    			{
#ifdef WITH_SUBTRACT
   					tmpTrace[compFrm].V = tmpTrace[compFrm-1].V;
#else
    				for(k=0;k<MY_VEC_SIZE;k++)
    				{
      				if(storeIdx[k] >= 0)
      					fgPtr[storeIdx[k]*npts*flow_block_size + compFrm] = 
                    fgPtr[storeIdx[k]*npts*flow_block_size + compFrm-1];

    				}

#endif
    			}
			}

#ifdef WITH_SUBTRACT
    		// do the trace zeroing...
    		{
    			MY_VECF dc_zero;
				dc_zero.V=dummy.V + 0.0f;

    	        for (int pt = start_pt; pt < end_pt; pt++){
       				dc_zero.V += (tmpTrace[pt].V);
    	        }

    	        // add end interpolation parts
				dc_zero.V += ohsv.V*(tmpTrace[overhang_start_pt].V);
				dc_zero.V += ohev.V*(tmpTrace[overhang_end_pt].V);

    	        // make it into an average
				dc_zero.V /= dummy.V + dc_cnt;

    			// now, subtract the dc offset from all the points
    	        for (int pt = 0;pt < npts;pt++){   // over real data
   					tmpTrace[pt].V -= dc_zero.V;
   					CVT_VF_VS(tmpTraceS[pt],tmpTrace[pt]);
    	        }
    		}
		// now, turn it back into short int's and save to fg_buffer
			for (k = 0; k < MY_VEC_SIZE; k++) {
				if (storeIdx[k] >= 0){
					int stidx=storeIdx[k] * npts * flow_block_size;

					for (int frm=0; frm < compFrm; frm++) {
						fgPtr[stidx + frm] = tmpTraceS[frm].V[k];
//						fgPtr[stidx + frm] = tmpTrace[frm].V[k];
					}
				}
			}
#endif

			for(k=0;k<MY_VEC_SIZE;k++)
			{
				  if(storeIdx[k] >= 0)
					  fgPtr += npts * flow_block_size; // advance one bead
			}
		}
    	
    }

/*
    cout << "CPU fgBuffer: (" << region->col << " " << region->row << ")"<< endl;
    for(int b= (216*224)/2 - 16; b<(216*224)/2; b++){
      fgPtr = &fgb[npts * flow_block_size*b+npts*iFlowBuffer];
      cout << b << " ";
      for(int f = 0; f< npts; f++){
        cout << fgPtr[f] << " ";
      }
      cout << endl;
    }
*/

    GenAllBtrc_time += tmr.elapsed();
    KeepEmptyScale(region, my_beads,img, iFlowBuffer);

#ifdef BEADTRACE_DBG
//#if 1
    if(region->row == DBG_ROW && region->col == DBG_COL /*&& iFlowBuffer == 19*/)
    {
	for (int ibd = 0; ibd < /*numLBeads*/2; ibd++)
	{
		for (size_t flow = 0; flow < allocated_flow_block_size; flow++)
		{
			FG_BUFFER_TYPE *fgPtr = &fgb[npts * flow_block_size * ibd
					+ flow * time_cp->npts()];
			printf("V %d/%d(%d/%d %lf): %d/%d ",ibd,flow,my_beads.params_nn[ibd].y,my_beads.params_nn[ibd].x,t0_map[0],region->row,region->col);
			for (int i = 0; i < time_cp->npts(); i++)
			{
				printf(" %d",fgPtr[i]);
			}
			printf("\n");
		}
	}
    }
#endif
}


// accumulates a single flow of a single bead's data into the caller's buffer
void BkgTrace::AccumulateSignal (float *signal_x, int ibd, int fnum, int len, int flow_block_size)
{
  FG_BUFFER_TYPE *pfg = &fg_buffers[npts*flow_block_size*ibd+fnum*time_cp->npts()];

  for (int i=0; i<len; i++)
    signal_x[i] += (float) pfg[i];
}

void BkgTrace::WriteBackSignalForBead(float *signal_x, int ibd, int fnum, int flow_block_size)
{
  int fnum_start = 0;
  int len = npts * flow_block_size;

  if (fnum != -1)
  {
    fnum_start = fnum;
    len = time_cp->npts();
  }

  FG_BUFFER_TYPE *pfg = &fg_buffers[npts * flow_block_size *ibd+fnum_start*time_cp->npts()];
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
void BkgTrace::SingleFlowFillSignalForBead(float *signal_x, int ibd, int fnum, int flow_block_size)
{

  CopySignalForFits (signal_x, &fg_buffers[npts * flow_block_size *ibd+fnum],time_cp->npts());
}

void BkgTrace::MultiFlowFillSignalForBead(float *signal_x, int ibd, int flow_block_size) const
{
  // Isolate signal extraction to trace routine
  // extract all flows for bead ibd
  CopySignalForFits (signal_x, &fg_buffers[npts * flow_block_size*ibd],npts * flow_block_size);
}

void BkgTrace::CopySignalForTrace(float *trace, int ntrace, int ibd,  int iFlowBuffer, int flow_block_size)
{
  // fg_buffers is a contiguous array organized as consecutive traces
  // in each flow buffer (total=npts*flow_block_size) for each bead in turn
  // |           bead 0                |           bead 1         |  ...
  // |       flow 0      | flow 1 | ...|
  // |v0 v1 (shorts) ... |
  // |<-------    npts*flow_block_size  ------->|  this is the same for every bead
  // |<- time_cp->npts ->|                this is the same for every trace
  // copy the trace of flow iFlowBuffer and bead ibd into the float array 'trace'

  assert( ntrace == time_cp->npts() ); // enforce whole trace for now
  if ( iFlowBuffer*time_cp->npts() >= npts*flow_block_size) {
    assert(  iFlowBuffer*time_cp->npts() < npts*flow_block_size);
  }
  FG_BUFFER_TYPE *pfg = &fg_buffers[ ibd*npts * flow_block_size + iFlowBuffer*time_cp->npts()];
  for (int i=0; i < ntrace; i++) {
    trace[i] = (float) pfg[i];
  }
}
