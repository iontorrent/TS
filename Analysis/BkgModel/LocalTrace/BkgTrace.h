/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGTRACE_H
#define BKGTRACE_H

#include "BkgMagicDefines.h"
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <iostream>
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
//        int     rawImgFrames;    // buffers to hold the average background signal and foreground data for each live bead well
    FG_BUFFER_TYPE *fg_buffers;
    // hack to indicate type of buffer we're using
    FG_BUFFER_TYPE *bead_trace_raw;
    FG_BUFFER_TYPE *bead_trace_bkg_corrected;
    
    float *fg_dc_offset; // per bead per flow dc offset - expensive to track
    int npts; // recompressed trace size. important size information for fg_buffers
    int numLBeads; // other size information for fg_buffers;
    
    // used to shift traces as they are read in
    std::vector<float> t0_map;

    float *bead_scale_by_flow;
    
    TimeCompression *time_cp; // point to time compression for now

    BkgTrace();
    virtual ~BkgTrace();
  void DumpFlows(std::ostream &out);
  bool NeedsAllocation();
    void Allocate(int npts, int _numLBeads, int flow_block_size);
    void    RezeroBeads(float t_start, float t_end, int fnum, int flow_block_size);

    void    RezeroOneBead(float t_start, float t_end, int fnum, int ibd, int flow_block_size);
    void    RezeroBeadsAllFlows (float t_start, float t_end);
    void    RezeroUncompressedTraceV(void *Ptr,  float t_start, float t_end);
    void  GenerateAllBeadTrace(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer, int flow_block_size,float t_start, float t_end);
    void  GenerateAllBeadTrace_nonvec(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer, FG_BUFFER_TYPE *fgb, int flow_block_size);
    void  GenerateAllBeadTrace_vec(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer, FG_BUFFER_TYPE *fgb, int flow_block_size, float t_start, float t_end);
    void  GenerateAllBeadTraceAnRezero(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer, int flow_block_size, float t_start, float t_end);

#define BKTRC_VEC_SIZE 8
#define BKTRC_VEC_SIZE_B 32

    void    LoadImgWOffset(const RawImage *raw, int16_t *out[BKTRC_VEC_SIZE], std::vector<int> &compFrms,
    		int nfrms, int l_coord[BKTRC_VEC_SIZE], float t0Shift/*, int print=0*/);
    void LoadImgWRezeroOffset(const RawImage *raw, int16_t *out[BKTRC_VEC_SIZE], std::vector<int> &compFrms,
        int nfrms, int l_coord[BKTRC_VEC_SIZE], float t0Shift, float t_start, float t_end);

    void  KeepEmptyScale(Region *region, BeadTracker &my_beads, Image *img, int iFlowBuffer);
    void   FillBeadTraceFromBuffer(short *img,int iFlowBuffer, int flow_block_size);
    void   DumpEmptyTrace(FILE *my_fp, int x, int y); // collect everything
    void   DumpBeadDcOffset(FILE *my_fp, bool debug_only, int DEBUG_BEAD, int x, int y,BeadTracker &my_beads);
    void    DumpABeadOffset(int a_bead, FILE *my_fp, int offset_col, int offset_row, BeadParams *cur);
    void    RecompressTrace(FG_BUFFER_TYPE *fgPtr, float *tmp_shifted);

    float   ComputeDcOffset(FG_BUFFER_TYPE *fgPtr,float t_start, float t_end);
    void    ComputeDcOffset_params(float t_start, float t_end,
    			int &pt1, int &pt2, float &cnt, float &overhang_start, float &overhang_end);
    template<typename T>
      float ComputeDcOffset(const T *fgPtr, TimeCompression &tc, float t_start, float t_end);
    float  GetBeadDCoffset(int ibd, int iFlowBuffer);

    void SetRawTrace();
    void SetBkgCorrectTrace();
    void AccumulateSignal (float *signal_x, int ibd, int fnum, int len, int flow_block_size);
    void WriteBackSignalForBead(float *signal_x, int ibd, int fnum /*=-1*/, int flow_block_size);
    void SingleFlowFillSignalForBead(float *signal_x, int ibd, int fnum, int flow_block_size);
    void MultiFlowFillSignalForBead(float *signal_x, int ibd, int flow_block_size) const;
    void CopySignalForTrace(float *trace, int ntrace, int ibd,  int iFlowBuffer, int flow_block_size);
    void T0EstimateToMap(const std::vector<float>& sep_t0_est, Region *region, Mask *bfmask);
    void SetImageParams(int _rows, int _cols, int _frames, int _uncompFrames, int *_timestamps)
    {
      imgRows=_rows;
      imgCols=_cols;
      imgFrames=_uncompFrames;
      compFrames=_frames;
    };

 private:
    bool restart;
    void AllocateScratch( int flow_block_size );
    int allocated_flow_block_size;

    // Serialization section
    friend class boost::serialization::access;
    template<typename Archive>
      void save(Archive& ar, const unsigned version) const
      {
	//fprintf(stdout, "Serialization: save BkgTrace ... ");
        ar &
	  imgRows &
	  imgCols &
	  imgFrames &
	  compFrames &
	  // fg_buffers handled in AllocateScratch
	  // bead_trace_raw handled in AllocateScratch
	  // bead_trace_bkg_corrected handled in AllocateScratch
	  // fg_dc_offset handled in AllocateScratch
	  npts &
	  numLBeads &
	  t0_map &
	  // bead_scale_by_flow handled in AllocateScratch
	  time_cp &
    allocated_flow_block_size;
	// fprintf(stdout, "done BkgTrace\n");
    }
    template<typename Archive>
      void load(Archive& ar, const unsigned version)
      {
        // fprintf(stdout, "Serialization: load BkgTrace ... ");
        ar &
	  imgRows &
	  imgCols &
	  imgFrames &
	  compFrames &
	  // fg_buffers handled in AllocateScratch
	  // bead_trace_raw handled in AllocateScratch
	  // bead_trace_bkg_corrected handled in AllocateScratch
	  // fg_dc_offset handled in AllocateScratch
	  npts &
	  numLBeads &
	  t0_map &
	  // bead_scale_by_flow handled in AllocateScratch
	  time_cp &
    allocated_flow_block_size;
	
	restart = true;
	AllocateScratch( allocated_flow_block_size );
	// fprintf(stdout, "done BkgTrace\n");
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

};

template<typename T>
float BkgTrace::ComputeDcOffset(const T *fgPtr, TimeCompression &tc, float t_start, float t_end)
{
    float dc_zero = 0.000f;
    float cnt = 0.0001f;
    int pt;
    int pt1 = 0;
    int pt2 = 0;
// TODO: is this really "rezero frames before pH step start?"
// this should be compatible with i_start from the nuc rise - which may change if we change the shape???
    for (pt = 0;tc.frameNumber[pt] < t_end;pt++)
    {
        pt2 = pt+1;
        if (tc.frameNumber[pt]>t_start)
        {
            if (pt1 == 0)
                pt1 = pt; // set to first point above t_start

            dc_zero += (float) (fgPtr[pt]);
            cnt += 1.0f; // should this be frames_per_point????
            //cnt += tc.frames_per_point[pt];  // this somehow makes it worse????
        }
    }

    // include values surrounding t_start & t_end weighted by overhang
    if (pt1 > 0) {
      // timecp->frameNumber[pt1-1] < t_start <= timecp->frameNumber[pt1]
      // normalize to a fraction in the spirit of "this somehow makes it worse"
      float den = (tc.frameNumber[pt1]- tc.frameNumber[pt1-1]);
      if ( den > 0 ) {
        float overhang = (tc.frameNumber[pt1] - t_start)/den;
        dc_zero = dc_zero + fgPtr[pt1-1]*overhang;
        cnt += overhang;
      }
    }

    if ( (pt2 < tc.npts()) && (pt2>0) ) {
      // timecp->frameNumber[pt2-1] <= t_end < timecp->frameNumber[pt2]
      // normalize to a fraction in the spirit of "this somehow makes it worse"
      float den = (tc.frameNumber[pt2]-tc.frameNumber[pt2-1]);
      if ( den > 0 ) {
	float overhang = (t_end - tc.frameNumber[pt2-1])/den;
	dc_zero = dc_zero + fgPtr[pt2]*overhang;
	cnt += overhang;
      }
    }

    dc_zero /= cnt;
    return(dc_zero);
}

// functionality used here and in EmptyTrace class which currently uses the same
// scheme for handling time decompression and compression
namespace TraceHelper{
  void ShiftTrace(float *trc,float *trc_out,int pts,float frame_offset);
  void SpecialShiftTrace (float *trc, float *trc_out, int pts, float frame_offset/*, int print=0*/);
  void ShiftTraceBiDirect (float *trc, float *trc_out, int pts, float frame_offset);
  void ShiftTraceBiDirect_vec (void *trc_v8f_u, void *trc_out_v8f_u, int pts, float frame_offset);
  void GetUncompressedTrace(float *tmp, Image *img, int absolute_x, int absolute_y, int img_frames);
  float ComputeT0Avg(const Region *region, const Mask *bfmask, const std::vector<float>& sep_t0_est, int img_cols);
  void BuildT0Map (const Region *region, const std::vector<float>& sep_t0_est, float reg_t0_avg, int img_cols, std::vector<float>& output);
  void DumpBuffer(char *ss, float *buffer, int start, int len);
}



#endif // BKGTRACE_H
