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
//    float *fg_dc_offset; // per bead per flow dc offset - expensive to track
    float *bg_dc_offset; // per empty trace per flow dc offset
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
    void    RezeroBeads(float t_start, float t_end, int fnum);
    void    RezeroReference(float t_start, float t_end, int fnum);
    void    RezeroTraces(float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty, int fnum);
    void    RezeroOneBead(float t_start, float t_end, int fnum, int ibd);
    void    RezeroTracesAllFlows(float t_start, float t_mid_nuc, float t_offset_beads, float t_offset_empty);
    float   ComputeT0Avg(Region *region, Mask *bfmask, std::vector<float> *sep_t0_est);
    void    BuildT0Map(Region *region, std::vector<float> *sep_t0_est, float reg_t0_avg);
    void    GenerateAverageEmptyTrace(Region *region, short *emptyInFlow, Mask *bfmask, Image *img, int iFlowBuffer, int flow);
    void  GenerateAllBeadTrace(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer);
    void   FillEmptyTraceFromBuffer(short *bkg, int iFlowBuffer);
    void   FillBeadTraceFromBuffer(short *img,int iFlowBuffer);
    void   T0EstimateToMap(std::vector<float> *sep_t0_est, Region *region, Mask *bfmask);
    void   DumpEmptyTrace(FILE *my_fp, int x, int y); // collect everything
    void   DumpBeadDcOffset(FILE *my_fp, bool debug_only, int DEBUG_BEAD, int x, int y,BeadTracker &my_beads);
    void    DumpABeadOffset(int a_bead, FILE *my_fp, int offset_col, int offset_row, bead_params *cur);
    void    GetUncompressedTrace(float *tmp, Image *img, int absolute_x, int absolute_y);
    void    RecompressTrace(FG_BUFFER_TYPE *fgPtr, float *tmp_shifted);
    void    AccumulateEmptyTrace(float *bPtr, float *tmp_shifted, float w);
    float   ComputeDcOffset(FG_BUFFER_TYPE *fgPtr,float t_start, float t_end);
    float   ComputeDcOffsetEmpty(float *bPtr, float t_start, float t_end);
    void    FillSignalForBead(float *signal_x, int ibd);
    void CopySignalForTrace(float *trace, int ntrace, int ibd,  int iFlowBuffer);
    void DumpBuffer(char *ss, float *buffer, int start, int len); //JGV
    void Dump_bg_buffers(char *ss, int start, int len); //JGV
};

void    ShiftTrace(float *trc,float *trc_out,int pts,float frame_offset);

void CopySignalForFits(float *signal_x, FG_BUFFER_TYPE *pfg, int len);

#endif // BKGTRACE_H
