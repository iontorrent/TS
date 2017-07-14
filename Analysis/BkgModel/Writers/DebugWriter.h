/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DEBUGWRITER_H
#define DEBUGWRITER_H

#include <stdio.h>
#include "Region.h"
#include "BeadParams.h"
#include "BeadTracker.h"
#include "RegionParams.h"
#include "RegionTracker.h"
#include "BkgModel/Fitters/Complex/BkgFitMatrixPacker.h"
#include "hdf5.h"
#include <armadillo>

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
  void DebugBeadIteration (BeadParams &eval_params, reg_params &eval_rp, int iter, float residual,RegionTracker *my_regions);
  void DebugIterations(BeadTracker &my_beads, RegionTracker *my_regions, int flow_block_size);

  void DumpRegionTrace (SignalProcessingMasterFitter &bkg, int flow_block_size, int flow_block_start);
  // used for convenience in dumping region trace
  void MultiFlowComputeTotalSignalTrace (SignalProcessingMasterFitter &bkg, float *fval,struct BeadParams *p,struct reg_params *reg_p,float *sbg /*=NULL*/,
                                         int flow_block_size, int flow_block_start);
};

#define IF_OPTIMIZER_DEBUG( D, X ) { if( D->bkg_control.pest_control.bkgModelHdf5Debug > 3 ) {X;} }
class DebugSaver{

private:
    static hid_t hdf_file_id;


public:
    DebugSaver()  {} //: hdf_file_id(-1)
    ~DebugSaver();
    void DebugFileOpen(std::string& dirName);
    void WriteData(const BkgFitMatrixPacker* reg_fit, reg_params &rp, int flow, const Region* region, const std::vector<string> derivativeNames, int nbeads);
};


#endif // DEBUGWRITER_H
