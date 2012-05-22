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
    FG_BUFFER_TYPE *fg_buffers;
    // hack to indicate type of buffer we're using
    FG_BUFFER_TYPE *bead_trace_raw;
    FG_BUFFER_TYPE *bead_trace_bkg_corrected;
    
    float *fg_dc_offset; // per bead per flow dc offset - expensive to track
    int bead_flow_t; // important size information for fg_buffers
    int numLBeads; // other size information for fg_buffers;
    
    // used to shift traces as they are read in
    float   *t0_map;


    
    TimeCompression *time_cp; // point to time compression for now

    BkgTrace();
    virtual ~BkgTrace();
    void Allocate(int numfb, int max_traces, int _numLBeads);
    void    RezeroBeads(float t_start, float t_end, int fnum);

    void    RezeroOneBead(float t_start, float t_end, int fnum, int ibd);
    void    RezeroBeadsAllFlows (float t_start, float t_end);
    void  GenerateAllBeadTrace(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer);
    void   FillBeadTraceFromBuffer(short *img,int iFlowBuffer);

    void   DumpEmptyTrace(FILE *my_fp, int x, int y); // collect everything
    void   DumpBeadDcOffset(FILE *my_fp, bool debug_only, int DEBUG_BEAD, int x, int y,BeadTracker &my_beads);
    void    DumpABeadOffset(int a_bead, FILE *my_fp, int offset_col, int offset_row, bead_params *cur);
    void    RecompressTrace(FG_BUFFER_TYPE *fgPtr, float *tmp_shifted);

    float   ComputeDcOffset(FG_BUFFER_TYPE *fgPtr,float t_start, float t_end);

    void SetRawTrace();
    void SetBkgCorrectTrace();
    bool AlreadyAdjusted();
    void AccumulateSignal (float *signal_x, int ibd, int fnum, int len);
    void WriteBackSignalForBead(float *signal_x, int ibd, int fnum=-1);

        void SingleFlowFillSignalForBead(float *signal_x, int ibd, int fnum);
    void MultiFlowFillSignalForBead(float *signal_x, int ibd);
    void CopySignalForTrace(float *trace, int ntrace, int ibd,  int iFlowBuffer);
    void T0EstimateToMap(std::vector<float> *sep_t0_est, Region *region, Mask *bfmask);
    void SetImageParams(int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps)
    {
      imgRows=_rows;
      imgCols=_cols;
      imgFrames=_uncompFrames;
      compFrames=_frames;
      timestamps=_timestamps;
//            rawImgFrames=_frames;
    };
};

// functionality used here and in EmptyTrace class which currently uses the same
// scheme for handling time decompression and compression
namespace TraceHelper{
  void ShiftTrace(float *trc,float *trc_out,int pts,float frame_offset);
  void GetUncompressedTrace(float *tmp, Image *img, int absolute_x, int absolute_y, int img_frames);
  void SpecialShiftTrace (float *trc, float *trc_out, int pts, float frame_offset);
  float ComputeT0Avg(Region *region, Mask *bfmask, std::vector<float> *sep_t0_est, int img_cols);
  void BuildT0Map (Region *region, std::vector<float> *sep_t0_est, float reg_t0_avg, int img_cols, float *output);
  void DumpBuffer(char *ss, float *buffer, int start, int len);
}



#endif // BKGTRACE_H
