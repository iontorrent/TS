/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DEBUGWRITER_H
#define DEBUGWRITER_H

#include <stdio.h>
#include "Region.h"
#include "BeadParams.h"
#include "BeadTracker.h"
#include "RegionParams.h"
#include "RegionTracker.h"

class SignalProcessingMasterFitter; // forward definition

class debug_collection
{
  public:
  // debug output files
  FILE        *data_dbg_file;
  FILE        *trace_dbg_file;
  FILE        *grid_dbg_file;
  FILE        *iter_dbg_file;
  FILE        *region_trace_file;
  FILE        *region_only_trace_file;
  FILE        *region_1mer_trace_file;
  FILE        *region_0mer_trace_file;

  debug_collection();
  ~debug_collection();
  void DebugFileClose();
  void DebugFileOpen(std::string& dirName, Region *region);
  void DebugBeadIteration (bead_params &eval_params, reg_params &eval_rp, int iter, float residual,RegionTracker *my_regions);
  void DebugIterations(BeadTracker &my_beads, RegionTracker *my_regions);

  void DumpRegionTrace (SignalProcessingMasterFitter &bkg);
  // used for convenience in dumping region trace
  void    MultiFlowComputeTotalSignalTrace (SignalProcessingMasterFitter &bkg, float *fval,struct bead_params *p,struct reg_params *reg_p,float *sbg=NULL);
};

#endif // DEBUGWRITER_H
