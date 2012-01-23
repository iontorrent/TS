/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BkgTrace.h"


// Savitsky-Goulay filter coefficients for calculating the slope.  poly order = 2, span=+/- 2 points
const float BkgTrace::bkg_sg_slope[BKG_SGSLOPE_LEN] = {-0.2,-0.1,0.0,0.1,0.2};

void BkgTrace::SavitskyGolayComputeSlope(float *local_slope,float *source_val, int len)
{
    // compute slope using savitsky golay smoother
    // pad ends of sequence with repeat values
    int moff = (BKG_SGSLOPE_LEN-1)/2;
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

void BkgTrace::Allocate(int numfb, int max_traces, int _numLBeads)
{
    bead_flow_t = max_traces;
    numLBeads = _numLBeads;
    //buffers are 2D arrays where each column is single pixel's frames;
    bg_buffers  = new float [numfb*imgFrames];
    neg_bg_buffers_slope  = new float [numfb*imgFrames];
    fg_buffers  = new FG_BUFFER_TYPE [max_traces*numLBeads];
}

void BkgTrace::PrecomputeBackgroundSlopeForDeriv(int iFlowBuffer)
{
    // calculate the slope of the background curve at every point
    float *bsPtr = &neg_bg_buffers_slope[iFlowBuffer*imgFrames];
    float *bPtr  = &bg_buffers[iFlowBuffer*imgFrames];

    SavitskyGolayComputeSlope(bsPtr,bPtr,imgFrames);
}

BkgTrace::~BkgTrace()
{
     if (t0_map != NULL) delete [] t0_map;
   if (neg_bg_buffers_slope!=NULL) delete [] neg_bg_buffers_slope;
    if (bg_buffers!=NULL) delete [] bg_buffers;
    if (fg_buffers!=NULL) delete [] fg_buffers;
    time_cp = NULL; // unlink
}

void BkgTrace::GetShiftedBkg(float tshift, float *bkg)
{
  ShiftMe(tshift,bg_buffers,bkg);
}

void BkgTrace::ShiftMe(float tshift,float *my_buff, float *out_buff)
{
    for (int fnum=0;fnum<NUMFB;fnum++)
    {
        float *fbkg = out_buff + fnum*time_cp->npts;
        float *bg = &my_buff[fnum*imgFrames];         // get ptr to start of neighbor background
        memset(fbkg,0,sizeof(float[time_cp->npts])); // on general principles
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

            fbkg[i] = ((1-frac) *bg[ifn] + frac*bg[ifn+1]);
        }
    }
}

void BkgTrace::GetShiftedSlope(float tshift, float *bkg)
{
  ShiftMe(tshift,neg_bg_buffers_slope,bkg);
}


void BkgTrace::RezeroBeads(float t_mid_nuc, float t_offset, int fnum)
{
    // re-zero the traces
    for (int ibd = 0;ibd < numLBeads;ibd++)
    {
        FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*ibd+fnum*time_cp->npts];

        float tzero = 0.0;
        float cnt = 0.0;
        int pt;
// TODO: is this really "rezero frames before pH step start?"
// this should be compatible with i_start from the nuc rise - which may change if we change the shape???
        for (pt = 0;time_cp->frameNumber[pt] < t_mid_nuc-t_offset;pt++)
        {
            tzero += (float)(fgPtr[pt]);
            cnt += 1.0;
        }

        if (cnt != 0)
            tzero /= cnt;
        for (pt = 0;pt < time_cp->npts;pt++)   // over real data
            fgPtr[pt] -= tzero;
    }

}

void BkgTrace::RezeroReference(float t_mid_nuc, float t_offset, int fnum)
{
    float *bPtr = &bg_buffers[fnum*imgFrames];
    int cnt = 0;
    float tzero = 0.0;
    for (int pt = 0;pt < t_mid_nuc-t_offset;pt++)
    {
        tzero += (float)(bPtr[pt]);
        cnt += 1.0;
    }

    if (cnt != 0)
        tzero /= cnt;
    for (int pt = 0;pt < imgFrames;pt++)
        bPtr[pt] -= tzero;
}

void BkgTrace::BuildT0Map(Region *region, std::vector<float> *sep_t0_est, float reg_t0_avg)
{
    t0_map = new float[region->w*region->h];

    for (int y=0;y<region->h;y++)
    {
        for (int x=0;x<region->w;x++)
        {
            // keep track of the offset for all wells so that we can shift empty well data while loading
            t0_map[y*region->w+x] = (*sep_t0_est)[x+region->col+ (y+region->row) * imgCols] - reg_t0_avg;
        }
    }
}


void BkgTrace::T0EstimateToMap(std::vector<float> *sep_t0_est, Region *region, Mask *bfmask)
{
    if (sep_t0_est !=NULL)
    {
        t0_mean =ComputeT0Avg(region,bfmask,sep_t0_est);
        BuildT0Map(region, sep_t0_est,t0_mean);
    }
}
        
// spatially organized time shifts
float BkgTrace::ComputeT0Avg(Region *region, Mask *bfmask, std::vector<float> *sep_t0_est)
{
    float reg_t0_avg = 0.0;
    int t0_avg_cnt = 0;
    for (int y=0;y<region->h;y++)
    {
        for (int x=0;x<region->w;x++)
        {
            if (!bfmask->Match(x+region->col,y+region->row, (MaskType)(MaskPinned | MaskIgnore | MaskExclude)))
            {
                reg_t0_avg += (*sep_t0_est)[x+region->col+ (y+region->row) * imgCols];
                t0_avg_cnt++;
            }
        }
    }
    reg_t0_avg /= t0_avg_cnt;
    return(reg_t0_avg);
}        

// takes trc as a pointer to an input signal of length pts and shifts it in time by frame_offset, putting the result
// in trc_out.  If frame_offset is positive, the signal is shifted left (towards lower indicies in the array)
// If frame_offset is not an integer, linear interpolation is used to construct the output values
void ShiftTrace(float *trc,float *trc_out,int pts,float frame_offset)
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

void BkgTrace::FillEmptyTraceFromBuffer(short *bkg, int iFlowBuffer)
{
    memset(&bg_buffers[iFlowBuffer*imgFrames],0,sizeof(float [imgFrames]));

    // copy background trace, linearize it from pH domain to [] domain
    float *bPtr = &bg_buffers[iFlowBuffer*imgFrames];
    for (int frame=DEFAULT_FRAME_TSHIFT;frame<imgFrames;frame+=1)
    {
        *bPtr += (bkg[frame]/FRAME_AVERAGE);
        bPtr++;
    }

    *bPtr = * (bPtr-1);
    bPtr++;
    *bPtr = * (bPtr-1);
    bPtr++;
    *bPtr = * (bPtr-1);
    bPtr++;
}

void BkgTrace::FillBeadTraceFromBuffer(short *img,int iFlowBuffer)
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
            *fgPtr = (FG_BUFFER_TYPE) (avg/time_cp->frames_per_point[npt]);
            fgPtr++;
            frame+=time_cp->frames_per_point[npt];
        }
    }
}

void BkgTrace::GenerateAverageEmptyTrace(Region *region, Mask *pinnedmask, Mask *bfmask, Image *img, int iFlowBuffer)
{
    // these are used by both the background and live-bead
    float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
    float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing

    int numAvg = 0;
    memset(&bg_buffers[iFlowBuffer*imgFrames],0,sizeof(float [imgFrames]));
    float *bPtr;

    double weight = 0;

    for (int y=region->row;y<region->row+region->h;y++)
    {
        for (int x=region->col;x<region->col+region->w;x++)
        {
            bPtr = &bg_buffers[iFlowBuffer*imgFrames];
            if (pinnedmask->Match(x,y,MaskEmpty))
            {
                numAvg++;
                double w = 1.0 / (1.0 + (bfmask->GetNumLiveNeighbors(y,x) * 2.0));
                weight += w;
                // uncompress the trace into a scratch buffer
                for (int frame=0;frame<imgFrames;frame++)
                    tmp[frame] = ((float) img->GetInterpolatedValue(frame,x,y));

                // shift it to account for relative timing differences - "mean zero shift"
                // x and y are global coordinates and need to have the region location subtracted
                // off in order to index into the region-sized t0_map
                if (t0_map!=NULL)
                    ShiftTrace(tmp,tmp_shifted,imgFrames,t0_map[x-region->col+ (y-region->row) *region->w]);
                else
                    printf("Alert: t0_map nonexistent\n");
                // shift the background by 3 frames automatically:  
                // "tshift=3 compensates for this exactly"
                for (int frame=DEFAULT_FRAME_TSHIFT;frame<imgFrames;frame++)
                {
#ifdef LIVE_WELL_WEIGHT_BG
                    *bPtr += tmp_shifted[frame] * w;
#else
                    *bPtr += tmp_shifted[frame];
#endif
                    bPtr++;
                }
                *bPtr = (* (bPtr-1)) ;
                bPtr++;
                *bPtr = (* (bPtr-1)) ;
                bPtr++;
                *bPtr = (* (bPtr-1)) ;
                bPtr++;
            }
        }
    }

    bPtr = &bg_buffers[iFlowBuffer*imgFrames];
    // in case we didn't have ANY empty wells...just leave all zeros in the background trace.
    // this won't be right...but at least we won't put NaN values into the wells file
    // we might want to punt on this subregion and not waste any time on it at all if this
    // happens

    if (numAvg != 0 && weight != 0) {
        for (int frame=0;frame < imgFrames;frame++) {
#ifdef LIVE_WELL_WEIGHT_BG
            bPtr[frame] = bPtr[frame] / weight;
#else
            bPtr[frame] = bPtr[frame] / numAvg;
#endif
        }
    }
}


void BkgTrace::GenerateAllBeadTrace(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer)
{
    //Populate the fg_buffers buffer with livebead only image data
    int nbd=0;

    // these are used by both the background and live-bead
    float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
    float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing
    nbd=0;
    for (nbd = 0;nbd < my_beads.numLBeads;nbd++) // is this the right iterator here?
    {
        int x = my_beads.params_nn[nbd].x;  // should x,y be stored with traces instead?
        int y = my_beads.params_nn[nbd].y;

        FG_BUFFER_TYPE *fgPtr = &fg_buffers[bead_flow_t*nbd+time_cp->npts*iFlowBuffer];

        // uncompress the trace into a scratch buffer
        for (int frame=0;frame<imgFrames;frame++)
            tmp[frame] = (float) img->GetInterpolatedValue(frame,x+region->col,y+region->row);

        // shift it by relative timing at this location
        // in this case x and y are local coordinates to the region, so they don't need to be offset
        // by the region location for indexing into t0_map
        if (t0_map!=NULL)
            ShiftTrace(tmp,tmp_shifted,imgFrames,t0_map[x+y*region->w]);
        else
            printf("Alert: t0_map nonexistent\n");

        int npt = 0;
        // do not shift real bead wells at all
        for (int frame=0;npt < time_cp->npts;npt++)   // real data
        {
            float avg;
            avg=0.0;
            for (int i=0;i<time_cp->frames_per_point[npt];i++)
            {
                float val = tmp_shifted[frame+i];
                avg += val;
            }

            *fgPtr = (FG_BUFFER_TYPE) (avg/time_cp->frames_per_point[npt]);
            fgPtr++;
            frame+=time_cp->frames_per_point[npt];
        }
    }

}

// t_offset_beads = nuc_shape.sigma
// t_offset_empty = 4.0
void BkgTrace::RezeroTraces(float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum)
{
    RezeroBeads(t_mid_nuc,t_offset_beads,fnum); // do these values make sense for offsets???
    RezeroReference(t_mid_nuc,t_offset_empty,fnum);

}

void BkgTrace::RezeroTracesAllFlows(float t_mid_nuc, float t_offset_beads, float t_offset_empty)
{
    for (int fnum=0; fnum<NUMFB; fnum++)
        RezeroTraces(t_mid_nuc,t_offset_beads,t_offset_empty, fnum);
}

