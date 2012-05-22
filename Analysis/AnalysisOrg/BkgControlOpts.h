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
#include "ChipIdDecoder.h"


// What does the bkg-model section of the software need to know?
class BkgModelControlOpts{
  public:
    int bkgModelHdf5Debug;
    float bkg_model_emphasis_width;
    float bkg_model_emphasis_amplitude;
    float dntp_uM;
    float AmplLowerLimit;
    int bkgModelMaxIter;
    char *gopt;
    char *xtalk;
    float krate[4];
    float kmax[4];
    float diff_rate[4];
    int no_rdr_fit_first_20_flows;
    int var_kmult_only;
    int generic_test_flag;
    bool enableXtalkCorrection;
    bool enableBkgModelClonalFilter;
    bool updateMaskAfterBkgModel;

    float damp_kmult; // dampen kmult variation
    float kmult_low_limit;
    float kmult_hi_limit;
    float krate_adj_threshold; // three parameters controlling the range of optimization available

    float ssq_filter;

    int recordBkgModelData;  // whether to record bkg data to a hdf 5 file
    int replayBkgModelData;  // whether to replay bkg data from a hdf5 file
    
    int bkgDebugParam;
    // temporary: dump debugging information for all beads, not just one
    int debug_bead_only;
    // commandline options for GPU for background model computation
    float gpuWorkLoad;
    int useBothCpuAndGpu;
    int numGpuThreads;
    int numCpuThreads;

    int vectorize;
    
    // only the row and col fields are used to specify location of debug regions
    std::vector<Region> BkgTraceDebugRegions;
    
    int readaheadDat;
    int saveWellsFrequency;

    bool useProjectionSearchForSingleFlowFit;
    int choose_time;



    void DefaultBkgModelControl(void);

};

#endif // BKGCONTROLOPTS_H
