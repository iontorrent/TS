/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <iostream>
#include <sstream>
#include "BkgControlOpts.h"


void BkgModelControlOpts::DefaultBkgModelControl()
{
  bkgModelHdf5Debug = 0;
    bkg_model_emphasis_width = 32.0;
    bkg_model_emphasis_amplitude = 4.0;
    for (int i_nuc=0; i_nuc<4; i_nuc++) dntp_uM[i_nuc] = -1.0f;
    AmplLowerLimit = 0.001;
    bkgModelMaxIter = 17;
    gopt = "default"; // "default" enables per-chip optimizations; other options: "disable" use the old hard-coded defaults, "opt" used only during optimization, and path to any optimized param file would load the file.
    xtalk = "disable";
    //xtalk= NULL;
    for (int i=0;i<4;i++)
    {
        krate[i] = -1.0;
        diff_rate[i] = -1.0;
        kmax[i] = -1.0;
    }
    no_rdr_fit_first_20_flows = 0;
    fitting_taue = 0;
    var_kmult_only = 0;
    generic_test_flag = 0;
    fit_alternate = 0;
    emphasize_by_compression=1; // by default turned to the old method
    BkgTraceDebugRegions.clear();
    bkgDebugParam = 0;

    enableXtalkCorrection = true;
    enable_dark_matter = true;
    enableBkgModelClonalFilter = true;
    updateMaskAfterBkgModel = true;

    // options for replay
    replayBkgModelData = false;
    recordBkgModelData = false;

    restart = false;
    restart_from = "";
    restart_next = "";
    restart_check = true;

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
    gpuMultiFlowFit = 1;
    gpuSingleFlowFit = 1;
    numCpuThreads = 0;
    readaheadDat = 0;
    saveWellsFrequency = 3;
    useProjectionSearchForSingleFlowFit = false;
    choose_time = 0; // default standard time compression

    use_dud_and_empty_wells_as_reference = false;
    proton_dot_wells_post_correction = false;
    empty_well_normalization = false;
    single_flow_fit_max_retry = 0;
    per_flow_t_mid_nuc_tracking = false;
    regional_sampling = false;
    prefilter_beads = false;

    unfiltered_library_random_sample = 100000;

    // diagnostics
    debug_bead_only = 1;  // only debug bead

    // emptyTrace outlier (wild trace) removal
    do_ref_trace_trim = false;
    span_inflator_min = 10;
    span_inflator_mult = 10;
    cutoff_quantile = .2;

    region_list = "";
}
