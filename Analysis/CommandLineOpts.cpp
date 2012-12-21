/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string.h>
#include <stdio.h>
#include <getopt.h> // for getopt_long
#include <stdlib.h> //EXIT_FAILURE
#include <ctype.h>  //tolower
#include <libgen.h> //dirname, basename
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "CommandLineOpts.h"
#include "IonErr.h"

using namespace std;

void readPerFlowScaleFile(vector<float> &perFlowScaleVal, char *perFlowScaleFile);
void readBasecallSubsetFile(char *basecallSubsetFile, set< pair <unsigned short,unsigned short> > &basecallSubset);


void CommandLineOpts::PrintHelp() {
    fprintf (stdout, "\n");
    fprintf (stdout, "Usage:\n");
    fprintf (stdout, "\tAnalysis [options][data_directory]\n");
    fprintf (stdout, "\tOptions:\n");
    fprintf (stdout, "\t\tSee man page for Analysis for complete list of options\n");
    fprintf (stdout, "\n");
    exit (EXIT_FAILURE);
}

void ModuleControlOpts::DefaultControl()
{
    USE_BKGMODEL = 1;
    BEADFIND_ONLY = false;
    USE_RAWWELLS = 0;
    WELLS_FILE_ONLY = false;	// when true, stop procesing at 1.wells file
}

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
    enableBkgModelClonalFilter = false;
    relaxKrateConstraint = 0;
    damp_kmult = 0;
    // how to do computation
    vectorize = 1;
    gpuWorkLoad = 1.0;
    numGpuThreads = 2;
    numCpuThreads = 0;
    readaheadDat = 0;
    saveWellsFrequency = 3;
    filterBubbles = 0;
    // diagnostics
    debug_bead_only = 1;  // only debug bead
}

void CafieControlOpts::DefaultCAFIEControl()
{
    singleCoreCafie = false;
    libPhaseEstimator = "nel-mead-treephaser";
//    libPhaseEstimator = "nel-mead-adaptive-treephaser";
    cfiedrRegionsX = 13, cfiedrRegionsY = 12;
    cfiedrRegionSizeX = 0, cfiedrRegionSizeY = 0;
    blockSizeX = 0, blockSizeY = 0;
    LibcfOverride = 0.0;
    LibieOverride = 0.0;
    LibdrOverride = 0.0;
    basecaller = "treephaser-swan";
    doCafieResidual = 0;
    numCafieSolveFlows = 0;
    // Options related to doing basecalling on just a subset of wells
    basecallSubsetFile = NULL;
}

void BeadfindControlOpts::DefaultBeadfindControl()
{
    //maxNumKeyFlows = 0;
    //minNumKeyFlows = 99;
    bfMinLiveRatio = .0001;
    bfMinLiveLibSnr = 4;
    bfMinLiveTfSnr = 4;
    bfTfFilterQuantile = 1;
    bfLibFilterQuantile = 1;
    skipBeadfindSdRecover = 0;
    beadfindThumbnail = 0;
    beadfindLagOneFilt = 0;
    beadMaskFile = NULL;
    maskFileCategorized = 0;
    sprintf (bfFileBase, "beadfind_post_0003.dat");
    sprintf (preRunbfFileBase, "beadfind_pre_0003.dat");
    BF_ADVANCED = true;
    SINGLEBF = true;
    noduds = 0;
    beadfindType = "differential";
}

void FilterControlOpts::DefaultFilterControl()
{
    nUnfilteredLib = 100000;
    unfilteredLibDir = strdup("unfiltered");
    beadSummaryFile = strdup("beadSummary.unfiltered.txt");
    KEYPASSFILTER = true;
    
    minReadLength = 8;
    // Options related to filtering reads by percentage of positive flows
    percentPositiveFlowsFilterTraining = 0;  // Should the ppf filter be used when initially estimating CAFIE params?
    percentPositiveFlowsFilterCalling = 1;   // Should the ppf filter be used when writing the SFF?
    percentPositiveFlowsFilterTFs = 0;       // If the ppf filter is on, should it be applied to TFs?
    // Options related to filtering reads by putative clonality
    clonalFilterTraining = 0;  // Should the clonality filter be used when initially estimating CAFIE params?
    clonalFilterSolving  = 1;  // Should the clonality filter be used when solving library reads?
    // Options related to filtering reads by CAFIE residuals
    cafieResFilterTraining = 0;  // Should the cafieRes filter be used when initially estimating CAFIE params?
    cafieResFilterCalling = 1;   // Should the cafieRes filter be used when writing the SFF?
    cafieResFilterTFs = 0;       // If the cafie residual filter is on, should it be applied to TFs?
    cafieResMaxFlow = 60;
    cafieResMinFlow = 12;
    cafieResMaxValue = 0.06;
    cafieResMaxValueOverride = false;
    cafieResMaxValueByFlowOrder[string("TACG")] = 0.06;  // regular flow order
    cafieResMaxValueByFlowOrder[string("TACGTACGTCTGAGCATCGATCGATGTACAGC")] = 0.08;  // xdb flow order
}

void SystemContext::DefaultSystemContext()
{
     dat_source_directory = NULL;
    wells_output_directory = NULL;
    basecaller_output_directory = NULL;
    
    strcpy (runId, "");
    
   sprintf (wellsFileName, "1.wells");
    strcpy (tmpWellsFile, "");
    LOCAL_WELLS_FILE = true;
    strcpy (wellsFilePath, "");
    wellStatFile=NULL;
    wellsFormat = "hdf5";
    NO_SUBDIR = 0;  // when set to true, no experiment subdirectory is created for output files.
}

void SpatialContext::DefaultSpatialContext()
{
    numRegions = 0;
    /* enables sub chip analysis */
    chipRegion.row=0;
    chipRegion.col=0;
    chipRegion.w=0;
    chipRegion.h=0;
    cols = 0;
    rows = 0;
    regionXOrigin = 624;
    regionYOrigin = 125;
    regionXSize = 50;
    regionYSize = 50;
    cropRegions = NULL;
    numCropRegions = 0;
    // some raw image processing (like cross talk correction in the Image class) needs the absolute coordinates of the
    // pixels in the image.  This is easy for a standard data set, but for a cropped data set the origin of the data is
    // unknown.  These allow the user to specify the location of the cropped region so that these parts of analysis
    // will work as designed.
    cropped_region_x_offset = 0;
    cropped_region_y_offset = 0;

    // datasets that are divided into blocks; each block has an offset from the chip's origin:
    chip_offset_x = -1;
    chip_offset_y = -1;
    chip_len_x = 0;
    chip_len_y = 0;
}

void ImageControlOpts::DefaultImageOpts(){
    maxFrames = 0;    // Set later from the first raw image header.
    totalFrames = 0;
    NNinnerx = 1;
    NNinnery = 1;
    NNouterx = 12;
    NNoutery = 8;
    ignoreChecksumErrors = 0;
    hilowPixFilter = 0;   // default is disabled
    flowTimeOffset = 1000;

    // image diagnostics
    outputPinnedWells = 0;
}

void KeyContext::DefaultKeys()
{
    libKey = strdup ("TCAG");
    tfKey = strdup ("ATCG");
    minNumKeyFlows = 99;
    maxNumKeyFlows = 0;
}

void FlowContext::DefaultFlowFormula()
{
    flowOrder = strdup ("TACG");
    numFlowsPerCycle = strlen (flowOrder);
    flowOrderOverride = false;
    flowOrderIndex = NULL;
    numTotalFlows = 0;
   flowLimitSet = 0;
}

void ObsoleteOpts::Defaults()
{
    NUC_TRACE_CORRECT = 0;
    USE_PINNED = false;
    lowerIntegralBound = 0;    // Frame 15...(used to be 20, Added a little more at the start for bkgModel)
    upperIntegralBound = 3049;    // Frame 60
    minPeakThreshold = 20;
    neighborSubtract = 0;
}


CommandLineOpts::CommandLineOpts(int argc, char *argv[])
{
    //Constructor
    if (argc == 1) {
        PrintHelp();
    }
    // helper pointer for loading options
    sPtr = NULL;
    /*---   options variables       ---*/

    // overall program flow control what to do?
    mod_control.DefaultControl();
    
    // controls for individual modules - how we are to analyze
    cfe_control.DefaultCAFIEControl();
    bkg_control.DefaultBkgModelControl();
    bfd_control.DefaultBeadfindControl();
    flt_control.DefaultFilterControl();
    img_control.DefaultImageOpts();

    // obsolete
    no_control.Defaults();

    // contexts for program operation - what the state of the world is
    sys_context.DefaultSystemContext();
    loc_context.DefaultSpatialContext();
    flow_context.DefaultFlowFormula();
    key_context.DefaultKeys();


    /*---   end options variables   ---*/

    /*---   other variables ---*/
    fpLog = NULL;

    /*---   Parse command line options  ---*/
    numArgs = argc;
    argvCopy = (char **) malloc (sizeof(char *) * argc);
    for (int i=0;i<argc;i++)
        argvCopy[i] = strdup (argv[i]);
    GetOpts (argc, argv);

}

FilterControlOpts::~FilterControlOpts()
{
    if (unfilteredLibDir)
        free (unfilteredLibDir);
    if (beadSummaryFile)
        free (beadSummaryFile);
}

SystemContext::~SystemContext()
{
    if (experimentName)
        free (experimentName);
    if (wells_output_directory)
        free (wells_output_directory);
    if (dat_source_directory)
        free (dat_source_directory);
    if (basecaller_output_directory)
        free (basecaller_output_directory);
}

CafieControlOpts::~CafieControlOpts()
{

    if (basecallSubsetFile)
        free(basecallSubsetFile);
}

KeyContext::~KeyContext()
{
    if (libKey)
        free (libKey);
    if (tfKey)
        free (tfKey);
}

BeadfindControlOpts::~BeadfindControlOpts()
{
    if (beadMaskFile)
        free (beadMaskFile);
}

FlowContext::~FlowContext()
{
    if (flowOrder)
        free (flowOrder);
}

SpatialContext::~SpatialContext()
{
    if (cropRegions)
        free(cropRegions);
}

CommandLineOpts::~CommandLineOpts()
{
    //Destructor

}

/*
 *  Use getopts to parse command line
 */
void CommandLineOpts::GetOpts (int argc, char *argv[])
{
    //DEBUG
    //fprintf (stdout, "Number of arguments: %d\n", argc);
    //for (int i=0;i<argc;i++)
    //    fprintf (stdout, "%s\n", argv[i]);
    //

    int c;
    long input;
    int option_index = 0;
    
    //@TODO this structure needs sorting or replacing badly!!!
    
    static struct option long_options[] =
    {
        {"no-subdir",               no_argument,        &sys_context.NO_SUBDIR,         1},
        {"output-dir",     required_argument,  NULL,       0},
        {"basecaller-output-dir",   required_argument,  NULL,       0},
        {"beadfindFile",            required_argument,  NULL,               'b'},
        {"beadfindfile",            required_argument,  NULL,               'b'},
        {"cycles",                  required_argument,  NULL,               'c'}, //Deprecated; use flowlimit
        {"frames",                  required_argument,  NULL,               'f'},
        {"help",                    no_argument,    NULL,               'h'},
        {"integral-bounds",         required_argument,  NULL,               'i'},
        {"keypass-filter",          required_argument,  NULL,               'k'},
        {"peak-threshold",          required_argument,  NULL,               'p'},
        {"version",                 no_argument,        NULL,               'v'},
        {"nuc-correct",             no_argument,        &no_control.NUC_TRACE_CORRECT, 1},
        {"basecallSubsetFile",      required_argument,  NULL,   0},
        {"phred-score-version",     required_argument,  NULL,   0},
        {"phred-table-file",        required_argument,  NULL,       0},
        {"min-read-length",         required_argument,        NULL,                         0},
        {"cr-filter-train",                          required_argument,        NULL,                                0},
        {"cr-filter",                                required_argument,        NULL,                                0},
        {"cr-filter-tf",                             required_argument,        NULL,                                0},
        {"cafie-residual-filter-max-flow",           required_argument,        NULL,                          0},
        {"cafie-residual-filter-min-flow",           required_argument,        NULL,                          0},
        {"cafie-residual-filter-max-value",          required_argument,        NULL,                          0},
        {"num-cafie-solve-flows",                    required_argument,        NULL,                                0},
        {"ppf-filter-train",                         required_argument,        NULL,                                0},
        {"ppf-filter",                               required_argument,        NULL,                                0},
        {"ppf-filter-tf",                            required_argument,        NULL,                                0},
        {"clonal-filter-train",                      required_argument,        NULL,                            0},
        {"clonal-filter-solve",                      required_argument,        NULL,                            0},
        {"cfiedr-regions",          required_argument,  NULL,               'R'},
        {"cfiedr-regions-size",     required_argument,  NULL,               'S'},
        {"block-size",              required_argument,  NULL,               'U'},
        {"region-size",             required_argument,  NULL,             0},
        {"from-wells",              required_argument,  NULL,               0},
        {"flowtimeoffset",          required_argument,  NULL,               0},
        {"flowlimit",       required_argument,  NULL,       0},
        {"Libcf-ie-dr",       required_argument,  NULL,               0},
        {"libcf-ie-dr",       required_argument,  NULL,               0},
        {"nnMask",          required_argument,  NULL,       0},
        {"nnmask",          required_argument,  NULL,       0},
        {"nnMaskWH",        required_argument,  NULL,       0},
        {"nnmaskwh",        required_argument,  NULL,       0},
        {"libraryKey",        required_argument,  NULL,       0},
        {"librarykey",        required_argument,  NULL,       0},
        {"tfKey",         required_argument,  NULL,       0},
        {"tfkey",         required_argument,  NULL,       0},
        {"forceNN",         no_argument,    &no_control.neighborSubtract,  1},
        {"forcenn",         no_argument,    &no_control.neighborSubtract,  1},
        {"singleCoreCafie",     no_argument,    &cfe_control.singleCoreCafie, 1},
        {"singlecorecafie",     no_argument,    &cfe_control.singleCoreCafie, 1},
        {"analysis-mode",     required_argument,  NULL,       0},
        {"use-pinned",        no_argument,    &no_control.USE_PINNED,    1},
        {"well-stat-file",      required_argument,  NULL,           0},
        {"basecaller",          required_argument,  NULL,       0},
        {"phase-estimator",         required_argument,  NULL,               0},
        {"ignore-checksum-errors",      no_argument,  NULL,           0},
        {"ignore-checksum-errors-1frame",   no_argument,  NULL,           0},
        {"flow-order",        required_argument,  NULL,           0},
        
        {"bfold",         no_argument,    &bfd_control.BF_ADVANCED,   0},
        {"bfonly",          no_argument,    &mod_control.BEADFIND_ONLY,   1},
        {"noduds",          no_argument,    &bfd_control.noduds,   1},
        {"beadmask-categorized",          no_argument,    &bfd_control.maskFileCategorized,   1},
        
        {"local-wells-file",    no_argument,    &sys_context.LOCAL_WELLS_FILE,  1},
        {"no-local-wells-file",    no_argument,    &sys_context.LOCAL_WELLS_FILE,  0},
        
        {"use-beadmask",      required_argument,  NULL,       0},
        
        {"bkg-debug-param",     required_argument,    NULL,   0},
        {"xtalk-correction",required_argument,     NULL,  0},
        {"clonal-filter-bkgmodel",required_argument,     NULL,  0},
        {"bkg-relax-krate-constraint",required_argument,     NULL,  0},
        {"bkg-damp-kmult",required_argument,     NULL,  0},
        {"bkg-h5-debug",           required_argument,  NULL,               0},
        {"bkg-emphasis",            required_argument,  NULL,               0},
        {"dntp-uM",                 required_argument,  NULL,               0},
        {"bkg-ampl-lower-limit",    required_argument,  NULL,               0},
        {"bkg-effort-level",        required_argument,  NULL,               0},
        {"gopt",                    required_argument,  NULL,               0},
        {"xtalk",                   required_argument,  NULL,               0},
        {"krate",                   required_argument,  NULL,               0},
        {"kmax",                    required_argument,  NULL,               0},
        {"diffusion-rate",          required_argument,  NULL,               0},
        {"cropped",                 required_argument,  NULL,       0},
        {"analysis-region",         required_argument,  NULL,       0},
        {"cafie-residuals",         no_argument,        &cfe_control.doCafieResidual,   1},
        {"n-unfiltered-lib",        no_argument,        &flt_control.nUnfilteredLib,    1},
        {"bead-washout",            no_argument,    NULL,       0},
        {"hilowfilter",             required_argument,  NULL,       0},
        {"numcputhreads",           required_argument,  NULL,       0},
        {"numgputhreads",           required_argument,  NULL,       0},
        {"wells-format",            required_argument,  NULL,               0},
        {"beadfind-type",           required_argument,  NULL,               0},
        {"beadfind-basis",          required_argument,  NULL,               0},
        {"beadfind-dat",            required_argument,  NULL,               0},
        {"beadfind-bgdat",          required_argument,  NULL,               0},
        {"beadfind-minlive",        required_argument,  NULL,               0},
        {"beadfind-minlivesnr",     required_argument,  NULL,               0},
        {"beadfind-min-lib-snr",    required_argument,  NULL,               0},
        {"beadfind-min-tf-snr",     required_argument,  NULL,               0},
        {"beadfind-lib-filt",    required_argument,  NULL,               0},
        {"beadfind-tf-filt",     required_argument,  NULL,               0},
        {"beadfind-skip-sd-recover",required_argument,  NULL,               0},
        {"beadfind-thumbnail",      required_argument,  NULL,               0},
        {"beadfind-lagone-filt",    required_argument,  NULL,               0},
        {"flag-bubbles",            no_argument,        &bkg_control.filterBubbles,     1},
        {"gpuWorkLoad",             required_argument,  NULL,             0},
        {"save-wells-freq",         required_argument,  NULL,             0},
        {"vectorize",               no_argument,        &bkg_control.vectorize,         1},
        {"bkg-dbg-trace",           required_argument,  NULL,               0},
        {"limit-rdr-fit",           no_argument,        &bkg_control.no_rdr_fit_first_20_flows,     1},
        {"var-kmult-only",          no_argument,        &bkg_control.var_kmult_only, 1},
        {"generic-test-flag",          no_argument,        &bkg_control.generic_test_flag, 1},
        {"debug-all-beads",         no_argument,    &bkg_control.debug_bead_only,              0}, // turn off what is turned on
        {"cropped-region-origin",   required_argument,  NULL,               0},
        {"output-pinned-wells",     no_argument,    &img_control.outputPinnedWells,   0},
        {"readaheadDat",            required_argument,  NULL,             0},
        {NULL,                      0,                  NULL,               0}
    };

    while ( (c = getopt_long (argc, argv, "b:c:f:hi:k:m:p:R:v", long_options, &option_index)) != -1 )
    {

        switch (c)
        {
        case (0):
        {
            char *lOption = strdup (long_options[option_index].name);
            ToLower (lOption);


            if (long_options[option_index].flag != 0)
                break;


            // module control:  what are we doing overall?
            
            if (strcmp (lOption, "analysis-mode") == 0) {
                ToLower(optarg);
                if (strcmp (optarg,"bkgmodel") == 0) {
                    mod_control.USE_BKGMODEL = 1;
                }
                else if (strcmp (optarg,"bfonly") == 0) {
                    mod_control.BEADFIND_ONLY = 1;
                }
                else {
                    fprintf (stderr, "Option Error: %s=%s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "from-wells") == 0) {
                if (isFile(optarg)) {
                    mod_control.USE_RAWWELLS = 1;
                    strncpy (sys_context.wellsFileName, basename(optarg), 256);
                    strncpy (sys_context.wellsFilePath, dirname(optarg), 256);
                    sys_context.dat_source_directory = "./";  //fake source directory
                }
                else {
                    fprintf (stderr, "Invalid file specified: %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
            }

            // end module control ------------------------------------------------------------------
            
            // cafie/basecaller control  -----------------------------------------------------------------------
             if (strcmp (lOption, "phred-table-file") == 0) {
                if (isFile(optarg)) {
                    cfe_control.phredTableFile = string( optarg );
                }
                else {
                    fprintf (stderr, "Invalid file specified for phred table: %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
            }

            if (strcmp (lOption, "basecallSubsetFile") == 0) {
                cfe_control.basecallSubsetFile = strdup(optarg);
                readBasecallSubsetFile(cfe_control.basecallSubsetFile,cfe_control.basecallSubset);
            }

           if (strcmp (lOption, "num-cafie-solve-flows") == 0) {
                int stat = sscanf (optarg, "%d", &cfe_control.numCafieSolveFlows);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (cfe_control.numCafieSolveFlows < 0) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "libcf-ie-dr") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%lf,%lf,%lf", &cfe_control.LibcfOverride, &cfe_control.LibieOverride, &cfe_control.LibdrOverride);
                    if (stat != 3) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                    cfe_control.libPhaseEstimator = "override";
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp(lOption, "basecaller") == 0) {
                cfe_control.basecaller = optarg;
            }
            if (strcmp(lOption, "phase-estimator") == 0) {
                cfe_control.libPhaseEstimator = optarg;
            }

            // end cafie control -------------------------------------------------------

            // filter control -------------------------------------------------------------

            
            if (strcmp (lOption, "ppf-filter-train") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.percentPositiveFlowsFilterTraining = 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.percentPositiveFlowsFilterTraining = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "ppf-filter") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.percentPositiveFlowsFilterCalling = 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.percentPositiveFlowsFilterCalling = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "ppf-filter-tf") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.percentPositiveFlowsFilterTFs = 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.percentPositiveFlowsFilterTFs = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "clonal-filter-train") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.clonalFilterTraining = 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.clonalFilterTraining = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "clonal-filter-solve") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.clonalFilterSolving = 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.clonalFilterSolving = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "min-read-length") == 0) {
                int stat = sscanf (optarg, "%d", &flt_control.minReadLength);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (flt_control.minReadLength < 1) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
           if (strcmp (lOption, "cr-filter-train") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.cafieResFilterTraining= 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.cafieResFilterTraining= 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cr-filter") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.cafieResFilterCalling= 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.cafieResFilterCalling= 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cr-filter-tf") == 0) {
                if (!strcmp(optarg,"off")) {
                    flt_control.cafieResFilterTFs = 0;
                } else if (!strcmp(optarg,"on")) {
                    flt_control.cafieResFilterTFs = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-residual-filter-max-flow") == 0) {
                int stat = sscanf (optarg, "%d", &flt_control.cafieResMaxFlow);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (flt_control.cafieResMaxFlow < 1) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-residual-filter-min-flow") == 0) {
                int stat = sscanf (optarg, "%d", &flt_control.cafieResMinFlow);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (flt_control.cafieResMinFlow < 0) {
                    fprintf (stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-residual-filter-max-value") == 0) {
                flt_control.cafieResMaxValueOverride = true;
                int stat = sscanf (optarg, "%lf", &flt_control.cafieResMaxValue);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (flt_control.cafieResMaxValue <= 0) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }

            // end filter control options -----------------------------------------------------





            // Image control options ---------------------------------------------------

            
            if (strcmp(lOption, "ignore-checksum-errors") == 0) {
                img_control.ignoreChecksumErrors |= 0x01;
            }
            if (strcmp(lOption, "ignore-checksum-errors-1frame") == 0) {
                img_control.ignoreChecksumErrors |= 0x02;
            }
             if (strcmp (lOption, "output-pinned-wells") == 0)
            {
                img_control.outputPinnedWells = 1;
            }
           if (strcmp (lOption, "flowtimeoffset") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    img_control.flowTimeOffset = (int) input;
                }
            }
            if (strcmp (lOption, "nnmask") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int inner = 1, outer = 3;
                    int stat = sscanf (optarg, "%d,%d", &inner, &outer);
                    if (stat != 2) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                    img_control.NNinnerx = inner;
                    img_control.NNinnery = inner;
                    img_control.NNouterx = outer;
                    img_control.NNoutery = outer;
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "nnmaskwh") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%d,%d,%d,%d", &img_control.NNinnerx, &img_control.NNinnery, &img_control.NNouterx, &img_control.NNoutery);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "hilowfilter") == 0) {
                ToLower(optarg);
                if (strcmp (optarg, "true") == 0 ||
                        strcmp (optarg, "on") == 0 ||
                        atoi(optarg) == 1)
                {
                    img_control.hilowPixFilter = 1;
                }
                else
                {
                    img_control.hilowPixFilter = 0;
                }
            }

            // end image control options -----------------------------------------------------------

            // flow entry and manipulation --------------------------------------------
            if (strcmp (lOption, "flow-order") == 0) {
                if (flow_context.flowOrder)
                    free(flow_context.flowOrder);
                flow_context.flowOrder = strdup(optarg);
                flow_context.numFlowsPerCycle = strlen(flow_context.flowOrder);
                flow_context.flowOrderOverride = true;
            }

            if (strcmp (long_options[option_index].name, "flowlimit") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    flow_context.flowLimitSet = (unsigned int) input;
                }
            }

            // end flow entry & manipulation ------------------------------------------------

            // keys - only two types for now --------------------------------------------
            if (strcmp (lOption, "librarykey") == 0) {
                key_context.libKey = (char *) malloc (strlen(optarg)+1);
                strcpy (key_context.libKey, optarg);
                ToUpper (key_context.libKey);
            }
            if (strcmp (lOption, "tfkey") == 0) {
                key_context.tfKey = (char *) malloc (strlen(optarg)+1);
                strcpy (key_context.tfKey, optarg);
                ToUpper (key_context.tfKey);
            }
            // end keys ------------------------------------------------------------------

            // Spatial reasoning about the chip, cropped area, etc -----------------------------------

            if (strcmp (lOption, "region-size") == 0) {
                sPtr = strchr(optarg,'x');
                if (sPtr) {
                    int stat = sscanf (optarg, "%dx%d", &loc_context.regionXSize, &loc_context.regionYSize);
                    if (stat != 2) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "cropped") == 0) {
                if (optarg) {
                    loc_context.numCropRegions++;
                    loc_context.cropRegions = (Region *) realloc (loc_context.cropRegions, sizeof(Region) * loc_context.numCropRegions);
                    int stat = sscanf (optarg, "%d,%d,%d,%d",
                                       &loc_context.cropRegions[loc_context.numCropRegions-1].col,
                                       &loc_context.cropRegions[loc_context.numCropRegions-1].row,
                                       &loc_context.cropRegions[loc_context.numCropRegions-1].w,
                                       &loc_context.cropRegions[loc_context.numCropRegions-1].h);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "analysis-region") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%d,%d,%d,%d",
                                       &loc_context.chipRegion.col,
                                       &loc_context.chipRegion.row,
                                       &loc_context.chipRegion.w,
                                       &loc_context.chipRegion.h);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "cropped-region-origin") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%d,%d",
                                       &loc_context.cropped_region_x_offset,
                                       &loc_context.cropped_region_y_offset);
                    if (stat != 2) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }

            // end spatial context about the chip ------------------------------------------------

            // System context: file manipulation, directories and names -----------------------------------


            if (strcmp (lOption, "well-stat-file") == 0) {
                if (sys_context.wellStatFile)
                    free(sys_context.wellStatFile);
                sys_context.wellStatFile = strdup(optarg);
            }

            if (strcmp (long_options[option_index].name, "wells-format") == 0) {
                sys_context.wellsFormat = optarg;
                if (sys_context.wellsFormat != "legacy" && sys_context.wellsFormat != "hdf5") {
                    fprintf (stderr, "*Error* - Illegal option to --wells-format: %s, valid options are 'legacy' or 'hdf5'\n",
                             sys_context.wellsFormat.c_str());
                    exit (EXIT_FAILURE);
                }
            }

            if (strcmp (lOption, "output-dir") == 0) {
                sys_context.wells_output_directory = strdup (optarg);
            }
            
            if (strcmp (lOption, "basecaller-output-dir") == 0) {
                sys_context.basecaller_output_directory = strdup (optarg);
            }
            
            // End system context files -------------------------------------------------------------


            // All beadfind options in this section, please ---------------------------------
            
            if (strcmp (long_options[option_index].name, "beadfind-type") == 0) {
                bfd_control.beadfindType = optarg;
                if (bfd_control.beadfindType != "differential" && bfd_control.beadfindType != "original") {
                    fprintf (stderr, "*Error* - Illegal option to --beadfind-type: %s, valid options are 'differential' or 'original'\n",
                             bfd_control.beadfindType.c_str());
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp(lOption, "use-beadmask") == 0) {
                bfd_control.beadMaskFile = strdup (optarg);
            }
            if (strcmp(lOption, "beadmask-categorized") == 0) {
                bfd_control.maskFileCategorized = 1;
            }
            if (strcmp (long_options[option_index].name, "beadfind-basis") == 0) {
                bfd_control.bfType = optarg;
                if (bfd_control.bfType != "signal" && bfd_control.bfType != "buffer") {
                    fprintf (stderr, "*Error* - Illegal option to --beadfind-basis: %s, valid options are 'signal' or 'buffer'\n",
                             bfd_control.bfType.c_str());
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "beadfind-dat") == 0) {
                bfd_control.bfDat = optarg;
            }
            if (strcmp (long_options[option_index].name, "beadfind-bgdat") == 0) {
                bfd_control.bfBgDat = optarg;
            }
            if (strcmp(long_options[option_index].name, "beadfind-minlive") == 0) {
                bfd_control.bfMinLiveRatio = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-minlivesnr") == 0 ||
                    strcmp(long_options[option_index].name, "beadfind-min-lib-snr") == 0) {
                bfd_control.bfMinLiveLibSnr = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-min-tf-snr") == 0) {
                bfd_control.bfMinLiveTfSnr = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-lib-filt") == 0) {
                bfd_control.bfLibFilterQuantile = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-tf-filt") == 0) {
                bfd_control.bfTfFilterQuantile = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-skip-sd-recover") == 0) {
                bfd_control.skipBeadfindSdRecover = atoi (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-thumbnail") == 0) {
                bfd_control.beadfindThumbnail = atoi (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-lagone-filt") == 0) {
                bfd_control.beadfindLagOneFilt = atoi (optarg);
            }
            if (strcmp (long_options[option_index].name, "bead-washout") == 0) {
                bfd_control.SINGLEBF = false;
            }

            // End of beadfind control ---------------------------------


            // All bkg_control options in this section, please ------------------------
            if (strcmp (lOption, "save-wells-freq") == 0) {
              bkg_control.saveWellsFrequency = atoi(optarg);
              fprintf (stdout, "Saving wells every %d blocks.\n", bkg_control.saveWellsFrequency);
              if (bkg_control.saveWellsFrequency < 1 || bkg_control.saveWellsFrequency > 100) {
                fprintf (stderr, "Option Error, must be between 1 and 100: %s %s\n", long_options[option_index].name,optarg);
                exit (EXIT_FAILURE);
              }

            }            
           if (strcmp (lOption, "clonal-filter-bkgmodel") == 0) {
                if (!strcmp(optarg,"off")) {
                    bkg_control.enableBkgModelClonalFilter = false;
                } else if (!strcmp(optarg,"on")) {
                    bkg_control.enableBkgModelClonalFilter = true;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
             if (strcmp (lOption, "xtalk-correction") == 0) {
                if (!strcmp(optarg,"off")) {
                    bkg_control.enableXtalkCorrection = false;
                } else if (!strcmp(optarg,"on")) {
                    bkg_control.enableXtalkCorrection = true;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp(lOption, "flag-bubbles") == 0) {
                bkg_control.filterBubbles = 1;
            }
            if (strcmp (long_options[option_index].name, "bkg-debug-param") == 0) {
              bkg_control.bkgModelHdf5Debug = atoi(optarg);
            }
            if (strcmp (lOption, "bkg-relax-krate-constraint") == 0) {
                int stat = sscanf (optarg, "%d", &bkg_control.relaxKrateConstraint);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (bkg_control.relaxKrateConstraint < 0) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "bkg-damp-kmult") == 0) {
                int stat = sscanf (optarg, "%f", &bkg_control.damp_kmult);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (bkg_control.damp_kmult < 0) {
                    fprintf (stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }

            if (strcmp (long_options[option_index].name, "bkg-emphasis") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%f,%f", &bkg_control.bkg_model_emphasis_width, &bkg_control.bkg_model_emphasis_amplitude);
                    if (stat != 2) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "dntp-uM") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%f", &bkg_control.dntp_uM);
                    if (stat != 1) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "bkg-ampl-lower-limit") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%f", &bkg_control.AmplLowerLimit);
                    if (stat != 1) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }           if (strcmp (long_options[option_index].name, "bkg-effort-level") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%d", &bkg_control.bkgModelMaxIter);
                    if (stat != 1) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }

                    if (bkg_control.bkgModelMaxIter < 5)
                    {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "gopt") == 0) {
                bkg_control.gopt = optarg;
                if (strcmp(bkg_control.gopt, "disable") == 0 || strcmp(bkg_control.gopt, "opt") == 0);
                else
                {
                    FILE *gopt_file = fopen(bkg_control.gopt,"r");
                    if (gopt_file != NULL)
                        fclose(gopt_file);
                    else {
                        fprintf (stderr, "Option Error: %s cannot open file %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }
                }
            }
            if (strcmp (long_options[option_index].name, "xtalk") == 0) {
                bkg_control.xtalk = optarg;
                if (strcmp(bkg_control.xtalk, "disable") == 0 || strcmp(bkg_control.xtalk, "opt") == 0);
                else
                {
                    FILE *tmp_file = fopen(bkg_control.xtalk,"r");
                    if (tmp_file != NULL)
                        fclose(tmp_file);
                    else {
                        fprintf (stderr, "Option Error: %s cannot open file %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }
                }
            }

            if (strcmp (long_options[option_index].name, "krate") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%f,%f,%f,%f", &bkg_control.krate[0],&bkg_control.krate[1],&bkg_control.krate[2],&bkg_control.krate[3]);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }

                    for (int i=0;i < 3;i++)
                    {
                        if ((bkg_control.krate[i] < 0.01) || (bkg_control.krate[i] > 100.0))
                        {
                            fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                            exit (1);
                        }
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (1);
                }
            }
            if (strcmp (long_options[option_index].name, "kmax") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%f,%f,%f,%f", &bkg_control.kmax[0],&bkg_control.kmax[1],&bkg_control.kmax[2],&bkg_control.kmax[3]);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }

                    for (int i=0;i < 3;i++)
                    {
                        if ((bkg_control.kmax[i] < 0.01) || (bkg_control.kmax[i] > 100.0))
                        {
                            fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                            exit (1);
                        }
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (1);
                }
            }
            if (strcmp (long_options[option_index].name, "diffusion-rate") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%f,%f,%f,%f", &bkg_control.diff_rate[0],&bkg_control.diff_rate[1],&bkg_control.diff_rate[2],&bkg_control.diff_rate[3]);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }

                    for (int i=0;i < 3;i++)
                    {
                        if ((bkg_control.diff_rate[i] < 0.01) || (bkg_control.diff_rate[i] > 1000.0))
                        {
                            fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                            exit (1);
                        }
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (1);
                }
            }            if (strcmp (lOption, "gpuworkload") == 0) {
                int stat = sscanf (optarg, "%f", &bkg_control.gpuWorkLoad);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if ((bkg_control.gpuWorkLoad > 1) || (bkg_control.gpuWorkLoad < 0)) {
                    fprintf (stderr, "Option Error: %s must specify a value between 0 and 1 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "numgputhreads") == 0) {
                int stat = sscanf (optarg, "%d", &bkg_control.numGpuThreads);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (bkg_control.numGpuThreads <= 0 || bkg_control.numGpuThreads >=5 ) {
                    fprintf (stderr, "Option Error: %s must specify a value between 1 and 4 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "numcputhreads") == 0) {
                int stat = sscanf (optarg, "%d", &bkg_control.numCpuThreads);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (bkg_control.numCpuThreads <= 0) {
                    fprintf (stderr, "Option Error: %s must specify a value greater than 0 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "readaheaddat") == 0) {
                int stat = sscanf (optarg, "%d", &bkg_control.readaheadDat);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (bkg_control.readaheadDat <= 0) {
                    fprintf (stderr, "Option Error: %s must specify a value greater than 0 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
           if (strcmp (lOption, "bkg-dbg-trace") == 0) {
                sPtr = strchr(optarg,'x');
                if (sPtr) {
                    Region dbg_reg;

                    int stat = sscanf (optarg, "%dx%d", &dbg_reg.col, &dbg_reg.row);
                    if (stat != 2) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }

                    bkg_control.BkgTraceDebugRegions.push_back(dbg_reg);
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }

            // end bkg model control----------


            free (lOption);

            break;
        }
        /*  End processing long options */

        case 'b':   //beadfind file name
            /*
            **  When this is set, we override the find-washouts default by
            **  setting the preRun filename to NULL.
            */
            snprintf (bfd_control.preRunbfFileBase, 256, "%s", optarg);
            //sprintf (preRunbfFileBase, "");
            bfd_control.bfFileBase[0] = '\0';
            bfd_control.SINGLEBF = true;
            break;
        case 'c':
            fprintf (stderr,"\n* * * * * * * * * * * * * * * * * * * * * * * * * *\n");
            fprintf (stderr, "The --cycles, -c keyword has been deprecated.\n"
                     "Use the --flowlimit keyword instead.\n");
            fprintf (stderr,"* * * * * * * * * * * * * * * * * * * * * * * * * *\n\n");
            exit (EXIT_FAILURE);
            break;
        case 'f':   // maximum frames
            if (validIn (optarg, &input)) {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            else {
                img_control.maxFrames = (int) input;
            }
            break;
        case 'h': // command help
            PrintHelp();
            break;
        case 'i':
            // integral bounds are "25:60"
            sPtr = strchr(optarg,':');
            if (sPtr) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    no_control.lowerIntegralBound = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    no_control.upperIntegralBound = (int) input;
                }
            }
            else {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            break;
        case 'k':   // enable/disable keypass filter
            for (unsigned int i = 0; i < strlen (optarg); i++)
                optarg[i] = tolower(optarg[i]);
            if (strcmp (optarg, "off") == 0) {
                flt_control.KEYPASSFILTER = false;
            }
            else if (strcmp (optarg, "on") == 0) {
                flt_control.KEYPASSFILTER = true;
            }
            else {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            break;
        case 'p':   //minimum peak threshold
            if (validIn (optarg, &input)) {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            else {
                no_control.minPeakThreshold = (int) input;
            }
            break;
        case 'v':   //version
            fprintf (stdout, "%s", IonVersion::GetFullVersion("Analysis").c_str());
            exit (EXIT_SUCCESS);
            break;
        case 'R': // use cfiedr in regions
            sPtr = strchr(optarg,'x');
            if (sPtr) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfe_control.cfiedrRegionsX = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfe_control.cfiedrRegionsY = (int) input;
                }
            }
            else {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            break;
        case 'U': // specify block size
            sPtr = strchr(optarg,'x');
            if (sPtr) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfe_control.blockSizeX = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfe_control.blockSizeY = (int) input;
                }
            }
            else {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            break;
        case 'S': // specify cfiedr region size
            sPtr = strchr(optarg,'x');
            if (sPtr) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfe_control.cfiedrRegionSizeX = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfe_control.cfiedrRegionSizeY = (int) input;
                }
            }
            else {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            break;
        case '?':
            /* getopt_long already printed an error message.*/
            exit (EXIT_FAILURE);
            break;
        default:
            fprintf (stderr, "What have we here? (%c)\n", c);
            exit (EXIT_FAILURE);
        }
    }

    // Pick up any non-option arguments (ie, source directory)
    for (c = optind; c < argc; c++)
    {
        sys_context.dat_source_directory = argv[c];
        break; //cause we only expect one non-option argument
    }

    // @TODO: this is all post-processing after options are established

    sys_context.GenerateContext(mod_control.USE_RAWWELLS); // find our directories
    flow_context.DetectFlowFormula(sys_context,mod_control.USE_RAWWELLS);  // if we didn't set it, search for it
    flt_control.RecognizeFlow(flow_context.flowOrder);  // must be after flow order is established, of course
    loc_context.FindDimensionsByType(sys_context.dat_source_directory);
    cfe_control.EchoDerivedChipParams(loc_context.chip_len_x,loc_context.chip_len_y); // must be after dimensions
}

void SpatialContext::FindDimensionsByType(char *dat_source_directory)
{
      char *chipType = GetChipId(dat_source_directory);
    ChipIdDecoder::SetGlobalChipId(chipType);  // @TODO: bad coding style, function side effect setting global variable
    int dims[2];
    GetChipDim(chipType, dims, dat_source_directory);  // @what if we're doing from wells and there are no dats?
    chip_len_x = dims[0];
    chip_len_y = dims[1];
}


void FilterControlOpts::RecognizeFlow(char *flowFormula)
{
    // Set some options that depend of flow order (unless explicitly set in command line)
    map<string,double>::iterator it;
    // cafieResMaxValue
    it = cafieResMaxValueByFlowOrder.find(string(flowFormula));
    if (!cafieResMaxValueOverride && it != cafieResMaxValueByFlowOrder.end()) {
        cafieResMaxValue = it->second;
    }

    // Test some dependencies between options
    if (cafieResMinFlow >= cafieResMaxFlow) {
        fprintf (stderr, "value of --cafie-residual-filter-min-flow must be strictly less than that of --cafie-residual-filter-max-flow.\n");
        exit (EXIT_FAILURE);
    }
}


void SystemContext::GenerateContext(int from_wells)
{
      if (!dat_source_directory) {
        dat_source_directory = (char *) malloc (2);
        snprintf (dat_source_directory, 1, ".");  // assume current directory if not provided as an argument
    }

    // Test for a valid data source directory
    // Exception: if this is a re-analysis from wells file, then we can skip this test.
    if (isDir(dat_source_directory) == false && (from_wells == 0)) {
        fprintf (stderr, "'%s' is not a directory.  Exiting.\n", dat_source_directory);
        exit (EXIT_FAILURE);
    }

    // standard output directory
    if (!wells_output_directory) {
        experimentName = (char*) malloc (3);
        strcpy (experimentName, "./");
    } else {
        if (NO_SUBDIR) {
            experimentName = strdup (wells_output_directory);
        } else {
            experimentName = experimentDir (dat_source_directory, wells_output_directory);    // subDir is created with this name
        }

    }

    if (!basecaller_output_directory) {
        basecaller_output_directory = strdup(experimentName);  // why is this duplicated?
    }

}

void FlowContext::DetectFlowFormula(SystemContext &sys_context, int from_wells)
{
// @TODO: obviously needs to be refactored into flow routine
// expand flow formula = flowOrder into appropriate number of flows
    //Determine total number of flows in experiment or previous analysis
    if (from_wells == 0) {
        numTotalFlows = GetTotalFlows(sys_context.dat_source_directory);
        assert (numTotalFlows > 0);
    }
    else {
        // Get total flows from processParams.txt
        numTotalFlows = atoi (GetProcessParam(sys_context.wellsFilePath, "numFlows"));
        assert (numTotalFlows > 0);
    }

    //If flow order was not specified on command line,
    //set it here from info from explog.txt or processParams.txt
    if (!flowOrderOverride) {
        if (flowOrder)
            free (flowOrder);
        // Get flow order from the explog.txt file
        if (from_wells == 0) {
            flowOrder = GetPGMFlowOrder (sys_context.dat_source_directory);
            assert (flowOrder != NULL);
            numFlowsPerCycle = strlen (flowOrder);
            assert (numFlowsPerCycle > 0);
        }
        // Get flow order from the processParams.txt file
        else
        {
            flowOrder = GetProcessParam (sys_context.wellsFilePath, "flowOrder");
            assert (flowOrder != NULL);
            numFlowsPerCycle = strlen (flowOrder);
            assert (numFlowsPerCycle > 0);
        }
    }

    // Adjust number of flows according to any command line options which may have been used
    // to limit these values
    if (flowLimitSet) {
        //support user specified number of flows
        numTotalFlows = (flowLimitSet < numTotalFlows ? flowLimitSet: numTotalFlows);
        assert (numTotalFlows > 0);
    }
}

void CafieControlOpts::EchoDerivedChipParams(int chip_len_x, int chip_len_y)
{
    //@TODO: isolate to cfe_control
    //overwrite cafie region size (13x12)
    if ((cfiedrRegionSizeX != 0) && (cfiedrRegionSizeY != 0) && (blockSizeX != 0) && (blockSizeY != 0)) {
        std::cout << "INFO: blockSizeX: " << blockSizeX << " ,blockSizeY: " << blockSizeY << std::endl;
        cfiedrRegionsX = blockSizeX /cfiedrRegionSizeX;
        cfiedrRegionsY = blockSizeY / cfiedrRegionSizeY;
        std::cout << "INFO: cfiedrRegionsX: " << cfiedrRegionsX << " ,cfiedrRegionsY: " << cfiedrRegionsY << std::endl;
    }

    //print debug information
    if ((blockSizeX == 0) && (blockSizeY == 0)) {
        unsigned short cafieYinc =
            ceil(chip_len_y / (double) cfiedrRegionSizeY);
        unsigned short cafieXinc =
            ceil(chip_len_x / (double) cfiedrRegionSizeX);
        std::cout << "DEBUG: precalculated values: cafieXinc: " << cafieXinc << " ,cafieYinc: " << cafieYinc << std::endl;
    }
}

void CommandLineOpts::WriteProcessParameters ()
{
    //  Dump the processing parameters to a file
    fprintf (fpLog, "Command line = ");
    for (int i = 0; i < numArgs; i++)
        fprintf (fpLog, "%s ", argvCopy[i]);
    fprintf (fpLog, "\n");
    fprintf (fpLog, "dataDirectory = %s\n", sys_context.dat_source_directory);
    fprintf (fpLog, "runId = %s\n", sys_context.runId);
    fprintf (fpLog, "flowOrder = %s\n", flow_context.flowOrder);
    fprintf (fpLog, "washFlow = %d\n", GetWashFlow());
    fprintf (fpLog, "libraryKey = %s\n", key_context.libKey);
    fprintf (fpLog, "tfKey = %s\n", key_context.tfKey);
    fprintf (fpLog, "minNumKeyFlows = %d\n", key_context.minNumKeyFlows);
    fprintf (fpLog, "maxNumKeyFlows = %d\n", key_context.maxNumKeyFlows);
    fprintf (fpLog, "numFlows = %d\n", flow_context.numTotalFlows);
    fprintf (fpLog, "cyclesProcessed = %d\n", flow_context.numTotalFlows/4); // @TODO: may conflict with PGM now
    fprintf (fpLog, "framesProcessed = %d\n", img_control.maxFrames);
    fprintf (fpLog, "framesInData = %d\n", img_control.totalFrames);
    //fprintf (fpLog, "framesPerSecond = %f\n", img.GetFPS());
    fprintf (fpLog, "minPeakThreshold = %d\n", no_control.minPeakThreshold);
    fprintf (fpLog, "lowerIntegrationTime = %d\n", no_control.lowerIntegralBound);
    fprintf (fpLog, "upperIntegrationTime = %d\n", no_control.upperIntegralBound);
    fprintf (fpLog, "bkgModelUsed = %s\n", (mod_control.USE_BKGMODEL ? "true":"false"));
    fprintf (fpLog, "nucTraceCorrectionUsed = %s\n", (no_control.NUC_TRACE_CORRECT ? "true":"false"));
    fprintf (fpLog, "cafieResFilterTrainingUsed = %s\n", (flt_control.cafieResFilterTraining ? "true":"false"));
    fprintf (fpLog, "cafieResFilterCallingUsed = %s\n", (flt_control.cafieResFilterCalling ? "true":"false"));
    fprintf (fpLog, "cafieResFilterTFsUsed = %s\n", (flt_control.cafieResFilterTFs ? "true":"false"));
    fprintf (fpLog, "cafieResMaxFlow = %d\n", flt_control.cafieResMaxFlow);
    fprintf (fpLog, "cafieResMinFlow = %d\n", flt_control.cafieResMinFlow);
    fprintf (fpLog, "cafieResMaxValue = %lf\n", flt_control.cafieResMaxValue);
    fprintf (fpLog, "basecaller = %s\n", cfe_control.basecaller.c_str());
    fprintf (fpLog, "percentPositiveFlowsFilterCallingUsed = %s\n", (flt_control.percentPositiveFlowsFilterCalling ? "true":"false"));
    fprintf (fpLog, "percentPositiveFlowsFilterTrainingUsed = %s\n", (flt_control.percentPositiveFlowsFilterTraining ? "true":"false"));
    fprintf (fpLog, "percentPositiveFlowsFilterTFsUsed = %s\n", (flt_control.percentPositiveFlowsFilterTFs ? "true":"false"));
    fprintf (fpLog, "clonalFilterTraining = %s\n", (flt_control.clonalFilterTraining ? "true":"false"));
    fprintf (fpLog, "clonalFilterSolving = %s\n", (flt_control.clonalFilterSolving ? "true":"false"));
    if (cfe_control.libPhaseEstimator == "override") {
        fprintf (fpLog, "Libcf-ie-dr values used = %0.5lf %0.5lf %0.5lf\n", cfe_control.LibcfOverride, cfe_control.LibieOverride, cfe_control.LibdrOverride);
    }
    fprintf (fpLog, "nearest-neighborParameters = Inner: (%d,%d) Outer: (%d,%d)\n", img_control.NNinnerx, img_control.NNinnery, img_control.NNouterx, img_control.NNoutery);
    
    fprintf (fpLog, "Advanced beadfind = %s\n", bfd_control.BF_ADVANCED ? "enabled":"disabled");
    fprintf (fpLog, "cfiedroopRegions = %d (%dx%d)\n", cfe_control.cfiedrRegionsX * cfe_control.cfiedrRegionsY,cfe_control.cfiedrRegionsX,cfe_control.cfiedrRegionsY);
    fprintf (fpLog, "cfiedroopRegion dimensions = %dx%d\n", (int) ceil(loc_context.cols/(double)cfe_control.cfiedrRegionsX), (int) ceil(loc_context.rows/(double)cfe_control.cfiedrRegionsY));
    fprintf (fpLog, "numCafieSolveFlows = %d\n", cfe_control.numCafieSolveFlows);
    fprintf (fpLog, "use pinned wells = %s\n", no_control.USE_PINNED ? "true":"false");
    fprintf (fpLog, "use exclusion mask = %s\n", loc_context.exclusionMaskSet ? "true":"false");
    fprintf (fpLog, "Version = %s\n", IonVersion::GetVersion().c_str());
    fprintf (fpLog, "Build = %s\n", IonVersion::GetBuildNum().c_str());
    fprintf (fpLog, "SvnRev = %s\n", IonVersion::GetSvnRev().c_str());
    
    fprintf (fpLog, "Chip = %d,%d\n", loc_context.chip_len_x,loc_context.chip_len_y);
    fprintf (fpLog, "Block = %d,%d,%d,%d\n", loc_context.chip_offset_x,loc_context.chip_offset_y,loc_context.cols,loc_context.rows);
    for (int q=0;q<loc_context.numCropRegions;q++)
        fprintf (fpLog, "Cropped Region = %d,%d,%d,%d\n", loc_context.cropRegions[q].col,loc_context.cropRegions[q].row,loc_context.cropRegions[q].w,loc_context.cropRegions[q].h);

    fprintf (fpLog, "Analysis Region = %d,%d,%d,%d\n", loc_context.chipRegion.col,loc_context.chipRegion.row,loc_context.chipRegion.col+loc_context.chipRegion.w,loc_context.chipRegion.row+loc_context.chipRegion.h);
    fprintf (fpLog, "numRegions = %d\n", loc_context.numRegions);
    fprintf (fpLog, "regionRows = %d\nregionCols = %d\n", loc_context.regionsY, loc_context.regionsX);
    fprintf (fpLog, "regionSize = %dx%d\n", loc_context.regionXSize, loc_context.regionYSize);
    //fprintf (fpLog, "\tRow Column Height Width\n");
    //for (int i=0;i<numRegions;i++)
    //  fprintf (fpLog, "[%3d] %5d %5d %5d %5d\n", i, regions[i].row, regions[i].col,regions[i].h,regions[i].w);
    fflush (NULL);
}

FILE * CommandLineOpts::InitFPLog ()
{
    char file[] = "processParameters.txt";
    char *fileName = (char *) malloc (strlen (sys_context.experimentName) + strlen (file) + 2);
    sprintf (fileName, "%s/%s", sys_context.experimentName, file);
    fopen_s(&fpLog, fileName, "wb");
    if (!fpLog) {
        perror (fileName);
        exit (errno);
    }
    free (fileName);
    fileName = NULL;
    fprintf(fpLog, "[global]\n");
    return (fpLog);
}

//
//  Create a name for the results of the analysis
//  Use the raw data directory name.  If it is not in standard format, use it in its entirety
//  Raw dir names are R_YYYY_MM_DD_hh_mm_ss_XXX_description
//  Results directory ("experiment" directory) will be 'description'_username_YY_MM_DD_seconds-in-day
//
char *SystemContext::experimentDir (char *rawdataDir, char *dirOut)
{
    char *expDir = NULL;
    char *timeStamp = NULL;
    char *sPtr = NULL;
    time_t now;
    struct tm  *tm = NULL;

    // Strip a trailing slash
    if (dirOut[strlen(dirOut)-1] == '/')
        dirOut[strlen(dirOut)-1] = '\0';

    //  Another algorithm counts forward through the date portion 6 underscores
    sPtr = rawdataDir;
    for (int i = 0; i < 7; i++) {
        sPtr = strchr (sPtr, '_');
        if (!sPtr) {
            sPtr = "analysis";
            break;
        }
        sPtr++;
    }
    if (sPtr[strlen(sPtr)-1] == '/')    // Strip a trailing slash too
        sPtr[strlen(sPtr)-1] = '\0';

    // Generate a timestamp string
    time ( &now );
    tm = localtime (&now);
    timeStamp = (char *) malloc (sizeof (char) * 18);
    snprintf (timeStamp, 18, "_%d_%02d_%02d_%d",1900 + tm->tm_year, tm->tm_mon+1, tm->tm_mday, 3600 * tm->tm_hour + 60 * tm->tm_min + tm->tm_sec);

    int strSize = strlen (dirOut) + strlen (timeStamp) + strlen (sPtr)+2;
    expDir = (char *) malloc (sizeof(char) * strSize);
    if (expDir != NULL)
        snprintf (expDir, strSize, "%s/%s%s", dirOut,sPtr,timeStamp);


    free (timeStamp);
    return (expDir);
}

void readBasecallSubsetFile(char *basecallSubsetFile, set< pair <unsigned short,unsigned short> > &basecallSubset) {
    ifstream inFile;
    inFile.open(basecallSubsetFile);
    if (inFile.fail())
        ION_ABORT("Unable to open basecallSubset file for read: " + string(basecallSubsetFile));

    vector <unsigned short> data;
    if (inFile.good()) {
        string line;
        getline(inFile,line);
        char delim = '\t';

        // Parse the line
        size_t current = 0;
        size_t next = 0;
        while (current < line.length()) {
            next = line.find(delim, current);
            if (next == string::npos) {
                next = line.length();
            }
            string entry = line.substr(current, next-current);
            istringstream i(entry);
            unsigned short value;
            char c;
            if (!(i >> value) || (i.get(c))) {
                ION_ABORT("Problem converting entry \"" + entry + "\" from file " + string(basecallSubsetFile) + " to unsigned short");
            } else {
                data.push_back(value);
                if (data.size()==2) {
                    pair< unsigned short, unsigned short> thisPair;
                    thisPair.first  = data[0];
                    thisPair.second = data[1];
                    basecallSubset.insert(thisPair);
                    data.erase(data.begin(),data.begin()+2);
                }
            }
            current = next + 1;
        }
    }
    if (data.size() > 0)
        ION_WARN("expected an even number of entries in basecallSubset file " + string(basecallSubsetFile));
}
