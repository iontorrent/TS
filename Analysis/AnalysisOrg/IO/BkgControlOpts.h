/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGCONTROLOPTS_H
#define BKGCONTROLOPTS_H

#include "stdlib.h"
#include "stdio.h"
#include <unistd.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "Region.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "SpecialDataTypes.h"
#include "SeqList.h"
#include "GpuControlOpts.h"
#include "DebugMe.h"
#include "ClonalFilter/polyclonal_filter.h"
#include "FlowSequence.h"
#include "OptBase.h"

#define REGIONAL_SAMPLING_SYSTEMATIC -1
#define REGIONAL_SAMPLING_CLONAL_KEY_NORMALIZED 1
#define REGIONAL_SAMPLING_PSEUDORANDOM 2

class SignalProcessingBlockControl{
public:
  bool restart;  // do we need restarting
  std::string restart_from;  // file to read restart info from
  std::string restart_next;  // file to write restart info to
  bool restart_check;   // if set, only restart with the same build number
  int save_wells_flow;        // New parameter, which defaults to saveWellsFrequency * 20.
  int wellsCompression;  // compression level to use in hdf5 for wells data, 3 by default 0 for no compression
  int numCpuThreads;
  bool updateMaskAfterBkgModel;
  FlowBlockSequence   flow_block_sequence;    // Every 20 flows, 0:15,15:1, etc.

  SignalProcessingBlockControl();
  void PrintHelp();
  void SetOpts(OptArgs &opts, Json::Value& json_params);
};

class TraceControl{
public:
  // emptyTrace outlier (wild trace) removal
  bool do_ref_trace_trim;
  float span_inflator_min;
  float span_inflator_mult;
  float cutoff_quantile;

  bool empty_well_normalization;
  bool use_dud_and_empty_wells_as_reference;

  TraceControl();
  void PrintHelp();
  void SetOpts(OptArgs &opts, Json::Value& json_params);
};

// What does the bkg-model section of the software need to know?
class BkgModelControlOpts{
  public:
    SignalProcessingBlockControl signal_chunks;
    GpuControlOpts gpuControl;
    DebugMe pest_control;
    TraceControl trace_control;

    PolyclonalFilterOpts polyclonal_filter;

    int emphasize_by_compression;
    bool enable_trace_xtalk_correction;   

	// how many wells to force processing on
    int unfiltered_library_random_sample;

    std::string region_list;  // CSV string of regions to use, eg "0,1,2,4"

    bool nokey; // keyless background model calling

    float washout_threshold;
    int washout_flow_detection;

    // Options for controlling regional double exponential smoothing.
    struct {
      float alpha;
      float gamma;
    } regional_smoothing;

   std::string restartRegParamsFile;

    void DefaultBkgModelControl(void);
	void PrintHelp();
    void SetOpts(OptArgs &opts, Json::Value& json_params, int num_flows);
};

#endif // BKGCONTROLOPTS_H
