/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGTRACE_H
#define BKGTRACE_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <float.h>
#include <string.h>
#include "TimeCompression.h"
#include "Region.h"
#include "Mask.h"
#include "Image.h"
#include "BeadTracker.h"

class BkgTrace{
public:
    // basic info about the images we are processing
    int  imgRows;
    int  imgCols;
    int  imgFrames;
    int     compFrames;
    int     *timestamps;
//        int     rawImgFrames;    // buffers to hold the average background signal and foreground data for each live bead well
    float *bg_buffers;
    float *neg_bg_buffers_slope;
    FG_BUFFER_TYPE *fg_buffers;
    int bead_flow_t; // important size information for fg_buffers
    int numLBeads; // other size information for fg_buffers;
    
    // used to shift traces as they are read in
    float   *t0_map;
    float   t0_mean;
    
    TimeCompression *time_cp; // point to time compression for now
    
    // kernel used to smooth and measure the slope of the background signal
    // this >never< changes, so is fine as a static const variable
#define BKG_SGSLOPE_LEN 5
    static const float bkg_sg_slope[BKG_SGSLOPE_LEN];
    BkgTrace();
    ~BkgTrace();
    void    SavitskyGolayComputeSlope(float *local_slope,float *source_val, int len);
    void Allocate(int numfb, int max_traces, int _numLBeads);
    void    PrecomputeBackgroundSlopeForDeriv(int iFlowBuffer);
    void    GetShiftedBkg(float tshift,float *bkg);
    void    ShiftMe(float tshift,float *my_buff, float *out_buff);
    void    GetShiftedSlope(float tshift, float *bkg);
    void    RezeroBeads(float t_mid_nuc, float t_offset, int fnum);
    void    RezeroReference(float t_mid_nuc, float t_offset, int fnum);
    void    RezeroTraces(float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum);
    void    RezeroTracesAllFlows(float t_mid_nuc, float t_offset_beads, float t_offset_empty);
    float   ComputeT0Avg(Region *region, Mask *bfmask, std::vector<float> *sep_t0_est);
    void    BuildT0Map(Region *region, std::vector<float> *sep_t0_est, float reg_t0_avg);
    void    GenerateAverageEmptyTrace(Region *region, Mask *pinnedmask, Mask *bfmask, Image *img, int iFlowBuffer);
    void  GenerateAllBeadTrace(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer);
    void   FillEmptyTraceFromBuffer(short *bkg, int iFlowBuffer);
    void   FillBeadTraceFromBuffer(short *img,int iFlowBuffer);
    void   T0EstimateToMap(std::vector<float> *sep_t0_est, Region *region, Mask *bfmask);
};

void    ShiftTrace(float *trc,float *trc_out,int pts,float frame_offset);

#endif // BKGTRACE_H