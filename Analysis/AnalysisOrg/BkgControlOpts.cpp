/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgControlOpts.h"



void BkgModelControlOpts::DefaultBkgModelControl()
{
  bkgModelHdf5Debug = 0;
    bkg_model_emphasis_width = 32.0;
    bkg_model_emphasis_amplitude = 4.0;
    dntp_uM = 50.0;
    AmplLowerLimit = 0.001;
    bkgModelMaxIter = 17;
    gopt = NULL; // NULL enables per-chip optimizations now by default, other options like "disable" would use the old hard-coded defaults, and can be changed via cmd-line to any optimized file
    xtalk = "disable";
    //xtalk= NULL;
    for (int i=0;i<4;i++)
    {
        krate[i] = -1.0;
        diff_rate[i] = -1.0;
        kmax[i] = -1.0;
    }
    no_rdr_fit_first_20_flows = 0;
    var_kmult_only = 0;
    generic_test_flag = 0;
    BkgTraceDebugRegions.clear();
    bkgDebugParam = 0;
    enableXtalkCorrection = true;
    enableBkgModelClonalFilter = true;
    updateMaskAfterBkgModel = true;

    // options for replay
    replayBkgModelData = false;
    recordBkgModelData = false;
      
    damp_kmult = 0;
    kmult_hi_limit = 1.75;
    kmult_low_limit = 0.65;
    krate_adj_threshold = 2.0;

    ssq_filter = 0.0f; // no filtering
    // how to do computation
    //@TODO: get the command line specification of vectorization to actually work
    
    vectorize = 1;
    //vectorize = 0;
    gpuWorkLoad = 1.0;
    useBothCpuAndGpu = 1;
    numGpuThreads = 2;
    numCpuThreads = 0;
    readaheadDat = 0;
    saveWellsFrequency = 3;
    useProjectionSearchForSingleFlowFit = false;
    choose_time = 0; // default standard time compression

    // diagnostics
    debug_bead_only = 1;  // only debug bead
}
