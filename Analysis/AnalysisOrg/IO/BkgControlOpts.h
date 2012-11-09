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
#include <string>
#include "Region.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "SpecialDataTypes.h"
#include "SeqList.h"
#include "ChipIdDecoder.h"


// What does the bkg-model section of the software need to know?
class BkgModelControlOpts{
  public:
    int bkgModelHdf5Debug;
    float bkg_model_emphasis_width;
    float bkg_model_emphasis_amplitude;
    float dntp_uM[4];
    float AmplLowerLimit;
    int bkgModelMaxIter;
    char *gopt;
    char *xtalk;
    float krate[4];
    float kmax[4];
    float diff_rate[4];
    int no_rdr_fit_first_20_flows;
    int fitting_taue;
    int var_kmult_only;
    int generic_test_flag;
    int emphasize_by_compression;
    bool enableXtalkCorrection;
    bool enable_dark_matter;
    bool enableBkgModelClonalFilter;
    bool updateMaskAfterBkgModel;
    bool prefilter_beads;

    int fit_alternate;
    float damp_kmult; // dampen kmult variation
    float kmult_low_limit;
    float kmult_hi_limit;
    float krate_adj_threshold; // three parameters controlling the range of optimization available

    float ssq_filter;

    int recordBkgModelData;  // whether to record bkg data to a hdf 5 file
    int replayBkgModelData;  // whether to replay bkg data from a hdf5 file

    bool restart;  // do we need restarting
    std::string restart_from;  // file to read restart info from
    std::string restart_next;  // file to write restart info to
    bool restart_check;   // if set, only restart with the same build number

    int bkgDebugParam;
    // temporary: dump debugging information for all beads, not just one
    int debug_bead_only;
    // commandline options for GPU for background model computation
    float gpuWorkLoad;
    int gpuMultiFlowFit;
    int gpuSingleFlowFit;
    int numCpuThreads;

    int vectorize;
    
    // only the row and col fields are used to specify location of debug regions
    std::vector<Region> BkgTraceDebugRegions;
    
    int readaheadDat;
    int saveWellsFrequency;

    bool useProjectionSearchForSingleFlowFit;
    int choose_time;

    // options that control changes for proton analysis (some may also be applicable to 3-series analysis)
    bool proton_dot_wells_post_correction;
    bool empty_well_normalization;
    bool use_dud_and_empty_wells_as_reference;
    int  single_flow_fit_max_retry;
    bool per_flow_t_mid_nuc_tracking;
    bool regional_sampling;

    // how many wells to force processing on
    int unfiltered_library_random_sample;


    void DefaultBkgModelControl(void);

    // emptyTrace outlier (wild trace) removal
    bool do_ref_trace_trim;
    float span_inflator_min;
    float span_inflator_mult;
    float cutoff_quantile;

    std::string region_list;  // CSV string of regions to use, eg "0,1,2,4"

};

#endif // BKGCONTROLOPTS_H
