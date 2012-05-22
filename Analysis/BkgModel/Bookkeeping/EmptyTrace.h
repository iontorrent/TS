/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EMPTYTRACE_H
#define EMPTYTRACE_H

#include "BkgTrace.h"
#include "PinnedInFlow.h"
#include "Mask.h"
#include "CommandLineOpts.h"

class EmptyTrace
{
 public:
  EmptyTrace(CommandLineOpts &clo);
  virtual ~EmptyTrace();
  virtual void GenerateAverageEmptyTrace(Region *region, PinnedInFlow &pinnedInFlow, Mask *bfmask, Image *img, int flow);  
  virtual void  Allocate(int numfb, int _imgFrames);
  void  PrecomputeBackgroundSlopeForDeriv(int iFlowBuffer);
  void  FillEmptyTraceFromBuffer(short *bkg, int flow);   
  void  RezeroReference(float t_start, float t_end, int fnum);
  void  RezeroReferenceAllFlows(float t_start, float t_end);
  void  GetShiftedBkg(float tshift,TimeCompression &time_cp, float *bkg);
  void  GetShiftedSlope(float tshift, TimeCompression &time_cp, float *bkg);
  void TimingSetUp(int t_mid_nuc_start, int time_start_detail, int time_stop_detail, int time_left_avg);
  void  T0EstimateToMap(std::vector<float> *sep_t0_est, Region *region, Mask *bfmask);

  void Dump_bg_buffers(char *ss, int start, int len); //JGV
  void DumpEmptyTrace(FILE *fp, int x, int y);

  int imgFrames;

  float *bg_buffers;  // should be private, exposed in the HDF5 dump
  // float * get_bg_buffers(int flowBufferReadPos) { return &bg_buffers[flowBufferReadPos*imgFrames]; }
  float *bg_dc_offset;

  int regionIndex;

 protected:
  int numfb; // number of flow buffers 
  float *neg_bg_buffers_slope;
  float *t0_map;  //@TODO: timing map across all regions
  MaskType referenceMask;

//@TODO: this is not a safe way of accessing buffers - need flow buffer object reference
  int flowToBuffer(int flow){
    return (flow % numfb);  // mapping of flow to bg_buffer entry
  };

  // kernel used to smooth and measure the slope of the background signal
  // this >never< changes, so is fine as a static const variable
#define BKG_SGSLOPE_LEN 5
  static const float bkg_sg_slope[BKG_SGSLOPE_LEN];
  void  SavitskyGolayComputeSlope(float *local_slope,float *source_val, int len);

  float ComputeDcOffsetEmpty(float *bPtr, float t_start, float t_end);
  void  AccumulateEmptyTrace(float *bPtr, float *tmp_shifted, float w);
  void  ShiftMe(float tshift, TimeCompression &time_cp, float *my_buff, float *out_buff);

private:
  EmptyTrace();   // do not use

};

#endif // EMPTYTRACE_H
