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

void CommandLineOpts::DefaultBkgModelControl()
{
     USE_BKGMODEL = 1;
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
    numCpuThreads = numCores();
    readaheadDat = 0;
}

void CommandLineOpts::DefaultCAFIEControl()
{
     NormalizeZeros = false; // if USE_BKGMODEL true
    singleCoreCafie = false;
    //bool tryAllReads = true; // this determines what TF's we track to determine cf/ie/dr - normally would be set to false but TF's with lib key are being handled right now - risk that we include non-TF's as part of the cf/ie/dr calcs
    tryAllReads = false; // this determines what TF's we track to determine cf/ie/dr - normally would be set to false but TF's with lib key are being handled right now - risk that we include non-TF's as part of the cf/ie/dr calcs
     minTFScore = 0.85; // if we can't score this TF with 85% confidence, its unknown
   minTFFlows = 36; // 8 flows for key, plus at least one more cycle, or we ignore this TF
    libPhaseEstimator = "nel-mead-treephaser";
//    libPhaseEstimator = "nel-mead-adaptive-treephaser";
    TF_CAFIE_CORRECTION = 1;
    TFoverride = NULL;
    cfiedrRegionsX = 13, cfiedrRegionsY = 12;
    cfiedrRegionSizeX = 0, cfiedrRegionSizeY = 0;
    blockSizeX = 0, blockSizeY = 0;
    cafieFlowMax = 80; // reasonable number of flows to fit while staying out of the adapter area on TF's A, B, C, D
    minTFCount = 1000;
    LibcfOverride = 0.0;
    LibieOverride = 0.0;
    LibdrOverride = 0.0;
    TFcfOverride = 0.0;
    TFieOverride = 0.0;
    TFdrOverride = 0.0;
    initial_cf = 0.008;
    initial_ie = 0.008;
    initial_dr = 0.000;
    libKey = strdup ("TCAG");
    tfKey = strdup ("ATCG");
    dotFixDebug = false;
    basecaller = "treephaser-swan";
    regionCafieDebugFile=NULL;
    wantDotFixes = true;
    doCafieResidual = 0;
    skiptfbasecalling = 0;
    alternateTFMode = 0; // MGD hack - needs to be 0 for default processing!
    numCFIEDRFitPasses = 1;
    useCafieHPIgnoreList = false; // defaults to false - just ignore all HP's when estimating, gets set to true when hp ignore list passed in
    cafieHPIgnoreList = NULL; // list of HP's to ignore
    numCafieHPIgnoreList = 0;
    cafieFitIgnoreLowQual = false; // set to true to enable the CFIEDRFit code to ignore low quality ideal vector base calls
    perFlowScale = false;
    perFlowScaleFile = NULL;
    perFlowScaleVal.resize(0);
    numFlowsToFitCafie1 = 60; // num flows to cafie fit pass 1
    numFlowsIncrement = 20; // multiple pass increment
    cfiedrKeepPercent = 0.8; // default of 1.0 is mean, removal of outliers = 0.6;
    numCafieSolveFlows = 0;
    droopMode = strdup ("estimate");
    hpScaleFactor = 1.0;
    SCALED_SOLVE2 = 1;
    NONLINEAR_HP_SCALE = 0;
    usePass1Droop = false; // when set to true, we calculate droop as an independent param estimate, then just solve cf & ie
    wantPerWellCafie = 0;
}

void CommandLineOpts::DefaultBeadfindControl()
{
    maxNumKeyFlows = 0;
    minNumKeyFlows = 99;
    bfMinLiveRatio = .0001;
    bfMinLiveLibSnr = 3.5;
    bfTfFilterQuantile = 1.25;
    bfLibFilterQuantile = .5;
    bfUseProj = 1;
    skipBeadfindSdRecover = 0;
    beadfindThumbnail = 0;
    beadfindLagOneFilt = 0;
    beadMaskFile = NULL;
    maskFileCategorized = 0;
    sprintf (bfFileBase, "beadfind_post_0003.dat");
    sprintf (preRunbfFileBase, "beadfind_pre_0003.dat");
    BEADFIND_ONLY = false;
    noduds = 0;
    beadfindType = "differential";
}

void CommandLineOpts::DefaultFilterControl()
{
    nUnfilteredLib = 100000;
    unfilteredLibDir = strdup("unfiltered");
    beadSummaryFile = strdup("beadSummary.unfiltered.txt");
     KEYPASSFILTER = true;
   // Options related to filtering reads by percentage of positive flows
    percentPositiveFlowsFilterTraining = 0;  // Should the ppf filter be used when initially estimating CAFIE params?
    percentPositiveFlowsFilterCalling = 1;   // Should the ppf filter be used when writing the SFF?
    percentPositiveFlowsFilterTFs = 0;       // If the ppf filter is on, should it be applied to TFs?
    percentPositiveFlowsMaxFlow = 60;
    percentPositiveFlowsMinFlow = 0;
    percentPositiveFlowsMaxValue = 0.6;
    percentPositiveFlowsMaxValueOverride = false;
    percentPositiveFlowsMaxValueByFlowOrder[string("TACG")] = 0.60;  // regular flow order
    percentPositiveFlowsMaxValueByFlowOrder[string("TACGTACGTCTGAGCATCGATCGATGTACAGC")] = 0.60;  // xdb flow order
    percentPositiveFlowsMinValue = 0.4;
    percentPositiveFlowsMinValueOverride = false;
    percentPositiveFlowsMinValueByFlowOrder[string("TACG")] = 0.40;  // regular flow order
    percentPositiveFlowsMinValueByFlowOrder[string("TACGTACGTCTGAGCATCGATCGATGTACAGC")] = 0.30;  // xdb flow order
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

void CommandLineOpts::DefaultWellControl()
{
    sprintf (wellsFileName, "1.wells");
    strcpy (tmpWellsFile, "");
    LOCAL_WELLS_FILE = true;
    strcpy (wellsFilePath, "");
    wellStatFile=NULL;
    wellsFormat = "hdf5";
}

CommandLineOpts::CommandLineOpts(int argc, char *argv[])
{
    //Constructor
    if (argc == 1) {
        PrintHelp();
    }
    /*---   options variables       ---*/
    dirExt = NULL;
    sprintf (dirOut, "./");
    OUTPUTDIR_OVERRIDE = NULL;

    DefaultCAFIEControl();
    DefaultBkgModelControl();
    DefaultBeadfindControl();
    DefaultFilterControl();
    DefaultWellControl();
    
    numRegions = 0;
    sequenceAllLib = 0;
    minReadLength = 8;
    lowerIntegralBound = 0;    // Frame 15...(used to be 20, Added a little more at the start for bkgModel)
    upperIntegralBound = 3049;    // Frame 60
    minPeakThreshold = 20;
    maxFrames = 0;    // Set later from the first raw image header.
    totalFrames = 0;
    sPtr = NULL;
   NUC_TRACE_CORRECT = 0;
    NO_SUBDIR = 0;  // when set to true, no experiment subdirectory is created for output files.
    // int     minSeqBases = 14; // if TF doesn't have this many bases, its ignored
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
    USE_RAWWELLS = 0;
    flowTimeOffset = 1000;
    strcpy (runId, "");
    NNinnerx = 1;
    NNinnery = 1;
    NNouterx = 12;
    NNoutery = 8;
    flowOrder = strdup ("TACG");
    numFlowsPerCycle = strlen (flowOrder);
    flowOrderOverride = false;
    neighborSubtract = 0;
    numGroupsPerRegion = 1;
    SINGLEBF = true;
    USE_PINNED = false;
    BF_ADVANCED = true;
    flowOrderIndex = NULL;
    hilowPixFilter = 0;   // default is disabled
    ignoreChecksumErrors = 0;
    // Options related to doing basecalling on just a subset of wells
    basecallSubsetFile = NULL;
    // Options related to per-flow scaling
    flowLimitSet = 0;

    filterBubbles = 0;
    outputPinnedWells = 0;

    
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

    /*---   end options variables   ---*/

    /*---   other variables ---*/
    fpLog = NULL;

    /*---   Parse command line options  ---*/
    numArgs = argc;
    argvCopy = (char **) malloc (sizeof(char *) * argc);
    for (int i=0;i<argc;i++)
        argvCopy[i] = strdup (argv[i]);
    GetOpts (argc, argv);

    if (OUTPUTDIR_OVERRIDE) {
        experimentName = strdup (OUTPUTDIR_OVERRIDE);
    } else {
        if (NO_SUBDIR) {
            experimentName = (char *) malloc (3);
            strcpy (experimentName, "./");
        } else {
            experimentName = experimentDir (dirExt, dirOut);    // subDir is created with this name
        }
    }
}

CommandLineOpts::~CommandLineOpts()
{
    //Destructor
    if (experimentName)
        free (experimentName);
    if (OUTPUTDIR_OVERRIDE)
        free (OUTPUTDIR_OVERRIDE);
    if (unfilteredLibDir)
        free (unfilteredLibDir);
    if (beadSummaryFile)
        free (beadSummaryFile);
    if (dirExt)
        free (dirExt);
    if (beadMaskFile)
        free (beadMaskFile);
    if (libKey)
        free (libKey);
    if (tfKey)
        free (tfKey);
    if (flowOrder)
        free (flowOrder);
    if (basecallSubsetFile)
        free(basecallSubsetFile);
    if (perFlowScaleFile)
        free(perFlowScaleFile);
    if (droopMode)
        free (droopMode);
    if (cropRegions)
        free(cropRegions);
    if (cafieHPIgnoreList)
        delete [] cafieHPIgnoreList;
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
    static struct option long_options[] =
    {
        {"no-subdir",               no_argument,        &NO_SUBDIR,         1},
        {"output-dir",        required_argument,  NULL,       0},
        {"TF",                      required_argument,  NULL,               0},
        {"beadfindFile",            required_argument,  NULL,               'b'},
        {"beadfindfile",            required_argument,  NULL,               'b'},
        {"cycles",                  required_argument,  NULL,               'c'}, //Deprecated; use flowlimit
        {"frames",                  required_argument,  NULL,               'f'},
        {"help",                    no_argument,    NULL,               'h'},
        {"integral-bounds",         required_argument,  NULL,               'i'},
        {"keypass-filter",          required_argument,  NULL,               'k'},
        {"peak-threshold",          required_argument,  NULL,               'p'},
        {"version",                 no_argument,        NULL,               'v'},
        {"nuc-correct",             no_argument,        &NUC_TRACE_CORRECT, 1},
        {"basecallSubsetFile",      required_argument,  NULL,   0},
        {"per-flow-scale-file",     required_argument,  NULL,   0},
        {"phred-score-version",     required_argument,  NULL,   0},
        {"phred-table-file",        required_argument,  NULL,       0},
        {"nuc-mults",               required_argument,  NULL,       0},
        {"scaled-solve2",           required_argument,  NULL,       0},
        {"sequence-all-lib",        required_argument,  NULL,       0},
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
        {"percent-positive-flow-filter-max-flow",    required_argument,        NULL,                          0},
        {"percent-positive-flow-filter-min-flow",    required_argument,        NULL,                          0},
        {"percent-positive-flow-filter-max-value",   required_argument,        NULL,                          0},
        {"percent-positive-flow-filter-min-value",   required_argument,        NULL,                          0},
        {"clonal-filter-train",                      required_argument,        NULL,                            0},
        {"clonal-filter-solve",                      required_argument,        NULL,                            0},
        {"cafie-lowqualfilter",      required_argument,  NULL,            0},
        {"nonlinear-hp-scale",      required_argument,  NULL,           0},
        {"cfiedr-regions",          required_argument,  NULL,               'R'},
        {"cfiedr-regions-size",     required_argument,  NULL,               'S'},
        {"block-size",              required_argument,  NULL,               'U'},
        {"region-cafie-auto",       no_argument,        NULL,               'A'},
        {"region-cafie-tf",         no_argument,        NULL,               'T'},
        {"Separate-Droop",          no_argument,        NULL,               'D'},
        {"separate-droop",          no_argument,        NULL,               'D'},
        {"Per-Num-Norm",            no_argument,        NULL,               'N'},
        {"per-num-norm",            no_argument,        NULL,               'N'},
        {"region-size",             required_argument,  NULL,             0},
        {"minTF",                   required_argument,  NULL,             0},
        {"mintf",                   required_argument,  NULL,             0},
        {"numGroupsPerRegion",      required_argument,  NULL,             0},
        {"numgroupsperregion",      required_argument,  NULL,             0},
        {"minTFFlows",              required_argument,  NULL,             0},
        {"mintfflows",              required_argument,  NULL,             0},
        {"alternateTFMode",         required_argument,  NULL,             0},
        {"from-wells",              required_argument,  NULL,               0},
        {"flowtimeoffset",          required_argument,  NULL,               0},
        {"flowlimit",       required_argument,  NULL,       0},
        {"cafieflowmax",            required_argument,  NULL,               0},
        {"Libcf-ie-dr",       required_argument,  NULL,               0},
        {"libcf-ie-dr",       required_argument,  NULL,               0},
        {"TFcf-ie-dr",        required_argument,  NULL,               0},
        {"tfcf-ie-dr",        required_argument,  NULL,               0},
        {"initial-cfiedr",        required_argument,  NULL,               0},
        {"try-all-tfs",       no_argument,  NULL,               0},
        {"ignore-hps",        required_argument,  NULL,               0},
        {"nnMask",          required_argument,  NULL,       0},
        {"nnmask",          required_argument,  NULL,       0},
        {"nnMaskWH",        required_argument,  NULL,       0},
        {"nnmaskwh",        required_argument,  NULL,       0},
        {"libraryKey",        required_argument,  NULL,       0},
        {"librarykey",        required_argument,  NULL,       0},
        {"tfKey",         required_argument,  NULL,       0},
        {"tfkey",         required_argument,  NULL,       0},
        {"forceNN",         no_argument,    &neighborSubtract,  1},
        {"forcenn",         no_argument,    &neighborSubtract,  1},
        {"singleCoreCafie",     no_argument,    &singleCoreCafie, 1},
        {"singlecorecafie",     no_argument,    &singleCoreCafie, 1},
        {"analysis-mode",     required_argument,  NULL,       0},
        {"perWell-cafie",     no_argument,    &wantPerWellCafie,  1},
        {"perwell-cafie",     no_argument,    &wantPerWellCafie,  1},
        {"droop-mode",          required_argument,  NULL,       0},
        {"use-pinned",        no_argument,    &USE_PINNED,    1},
        {"well-stat-file",      required_argument,  NULL,           0},
        {"basecaller",          required_argument,  NULL,       0},
        {"cafie-solver",      no_argument,  NULL,           0},
        {"phase-estimator",         required_argument,  NULL,               0},
        {"dot-fix-debug",     no_argument,  NULL,           0},
        {"ignore-checksum-errors",      no_argument,  NULL,           0},
        {"ignore-checksum-errors-1frame",   no_argument,  NULL,           0},
        {"region-cafie-debug-file",   required_argument,  NULL,           0},
        {"flow-order",        required_argument,  NULL,           0},
        {"region-cafie-file",   no_argument,    NULL,           'F'},
        {"bfold",         no_argument,    &BF_ADVANCED,   0},
        {"bfonly",          no_argument,    &BEADFIND_ONLY,   1},
        {"noduds",          no_argument,    &noduds,   1},
        {"local-wells-file",    no_argument,    &LOCAL_WELLS_FILE,  1},
        {"use-beadmask",      required_argument,  NULL,       0},
        {"beadmask-categorized",          no_argument,    &maskFileCategorized,   1},
        {"bkg-debug-param",     no_argument,    &bkgDebugParam,   1},
        {"xtalk-correction",required_argument,     NULL,  0},
        {"clonal-filter-bkgmodel",required_argument,     NULL,  0},
        {"bkg-relax-krate-constraint",required_argument,     NULL,  0},
        {"bkg-damp-kmult",required_argument,     NULL,  0},
        {"mintfscore",        required_argument,  NULL,       0},
        {"bkg-emphasis",            required_argument,  NULL,               0},
        {"dntp-uM",                 required_argument,  NULL,               0},
        {"bkg-ampl-lower-limit",    required_argument,  NULL,               0},
        {"bkg-effort-level",        required_argument,  NULL,               0},
        {"gopt",                    required_argument,  NULL,               0},
        {"xtalk",                    required_argument,  NULL,               0},
        {"krate",                   required_argument,  NULL,               0},
        {"kmax",                    required_argument,  NULL,               0},
        {"diffusion-rate",          required_argument,  NULL,               0},
        {"cropped",         required_argument,  NULL,       0},
        {"analysis-region",     required_argument,  NULL,       0},
        {"cafie-residuals",         no_argument,        &doCafieResidual,   1},
        {"n-unfiltered-lib",        no_argument,        &nUnfilteredLib,    1},
        {"num-cfiedr-fit-passes", required_argument,  NULL,       0},
        {"cfiedr-keep-percent",   required_argument,  NULL,       0},
        {"skip-tf-basecalling",   no_argument,    &skiptfbasecalling, 1},
        {"num-flows-to-fit",    required_argument,  NULL,       0},
        {"num-flows-increment",   required_argument,  NULL,       0},
        {"bead-washout",      no_argument,    NULL,       0},
        {"hilowfilter",       required_argument,  NULL,       0},
        {"wells-format",           required_argument,  NULL,               0},
        {"beadfind-type",           required_argument,  NULL,               0},
        {"beadfind-basis",          required_argument,  NULL,               0},
        {"beadfind-dat",            required_argument,  NULL,               0},
        {"beadfind-bgdat",          required_argument,  NULL,               0},
        {"beadfind-minlive",        required_argument,  NULL,               0},
        {"beadfind-minlivesnr",     required_argument,  NULL,               0},
        {"beadfind-tf-filter-quantile",     required_argument,  NULL,               0},
        {"beadfind-lib-filter-quantile",     required_argument,  NULL,               0},
        {"beadfind-use-proj",     required_argument,  NULL,               0},
        {"beadfind-minlivesnr",     required_argument,  NULL,               0},
        {"beadfind-minlivesnr",     required_argument,  NULL,               0},
        {"beadfind-skip-sd-recover",required_argument,  NULL,               0},
        {"beadfind-thumbnail",required_argument,  NULL,               0},
        {"beadfind-lagone-filt",required_argument,  NULL,               0},
        {"flag-bubbles",            no_argument,        &filterBubbles,     1},
        {"gpuWorkLoad",             required_argument,  NULL,             0},
        {"numGpuThreads",           required_argument,  NULL,             0},
        {"numCpuThreads",           required_argument,  NULL,             0},
        {"vectorize",               no_argument,        &vectorize,         1},
        {"bkg-dbg-trace",           required_argument,  NULL,               0},
        {"limit-rdr-fit",           no_argument,        &no_rdr_fit_first_20_flows,     1},
        {"cropped-region-origin",   required_argument,  NULL,               0},
        {"output-pinned-wells",      no_argument,    &outputPinnedWells,   0},
        {"readaheadDat",           required_argument,  NULL,             0},
        {NULL,                      0,                  NULL,               0}
    };

    while ( (c = getopt_long (argc, argv, "Ab:c:f:Fhi:k:m:p:R:Dv", long_options, &option_index)) != -1 )
    {

        switch (c)
        {
        case (0):
        {
            char *lOption = strdup (long_options[option_index].name);
            ToLower (lOption);

            if (strcmp (lOption, "output-pinned-wells") == 0)
            {
                outputPinnedWells = 1;
            }

            if (long_options[option_index].flag != 0)
                break;
            if (strcmp (lOption, "tf") == 0) {
                TFoverride = optarg;
            }
            if (strcmp (lOption, "nonlinear-hp-scale") == 0) {
                NONLINEAR_HP_SCALE=1;
                int stat = sscanf (optarg, "%lf", &hpScaleFactor);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "analysis-mode") == 0) {
                ToLower(optarg);
                if (strcmp (optarg,"bkgmodel") == 0) {
                    USE_BKGMODEL = 1;
                }
                else if (strcmp (optarg,"bfonly") == 0) {
                    BEADFIND_ONLY = 1;
                }
                else {
                    fprintf (stderr, "Option Error: %s=%s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "num-cafie-solve-flows") == 0) {
                int stat = sscanf (optarg, "%d", &numCafieSolveFlows);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (numCafieSolveFlows < 0) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "ppf-filter-train") == 0) {
                if (!strcmp(optarg,"off")) {
                    percentPositiveFlowsFilterTraining = 0;
                } else if (!strcmp(optarg,"on")) {
                    percentPositiveFlowsFilterTraining = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "ppf-filter") == 0) {
                if (!strcmp(optarg,"off")) {
                    percentPositiveFlowsFilterCalling = 0;
                } else if (!strcmp(optarg,"on")) {
                    percentPositiveFlowsFilterCalling = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "ppf-filter-tf") == 0) {
                if (!strcmp(optarg,"off")) {
                    percentPositiveFlowsFilterTFs = 0;
                } else if (!strcmp(optarg,"on")) {
                    percentPositiveFlowsFilterTFs = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "percent-positive-flow-filter-max-flow") == 0) {
                int stat = sscanf (optarg, "%d", &percentPositiveFlowsMaxFlow);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (percentPositiveFlowsMaxFlow < 1) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "percent-positive-flow-filter-min-flow") == 0) {
                int stat = sscanf (optarg, "%d", &percentPositiveFlowsMinFlow);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (percentPositiveFlowsMinFlow < 0) {
                    fprintf (stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            } if (strcmp (lOption, "percent-positive-flow-filter-max-value") == 0) {
                percentPositiveFlowsMaxValueOverride = true;
                int stat = sscanf (optarg, "%lf", &percentPositiveFlowsMaxValue);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if ((percentPositiveFlowsMaxValue > 1) || (percentPositiveFlowsMaxValue < 0)) {
                    fprintf (stderr, "Option Error: %s must specify a value between 0 and 1 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "percent-positive-flow-filter-min-value") == 0) {
                percentPositiveFlowsMinValueOverride = true;
                int stat = sscanf (optarg, "%lf", &percentPositiveFlowsMinValue);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if ((percentPositiveFlowsMinValue > 1) || (percentPositiveFlowsMinValue < 0)) {
                    fprintf (stderr, "Option Error: %s must specify a value between 0 and 1 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "clonal-filter-train") == 0) {
                if (!strcmp(optarg,"off")) {
                    clonalFilterTraining = 0;
                } else if (!strcmp(optarg,"on")) {
                    clonalFilterTraining = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "clonal-filter-solve") == 0) {
                if (!strcmp(optarg,"off")) {
                    clonalFilterSolving = 0;
                } else if (!strcmp(optarg,"on")) {
                    clonalFilterSolving = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-lowqualfilter") == 0) {
                if (!strcmp(optarg,"off")) {
                    cafieFitIgnoreLowQual = false;
                } else if (!strcmp(optarg,"on")) {
                    cafieFitIgnoreLowQual = true;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "min-read-length") == 0) {
                int stat = sscanf (optarg, "%d", &minReadLength);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (minReadLength < 1) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "xtalk-correction") == 0) {
                if (!strcmp(optarg,"off")) {
                    enableXtalkCorrection = false;
                } else if (!strcmp(optarg,"on")) {
                    enableXtalkCorrection = true;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "clonal-filter-bkgmodel") == 0) {
                if (!strcmp(optarg,"off")) {
                    enableBkgModelClonalFilter = false;
                } else if (!strcmp(optarg,"on")) {
                    enableBkgModelClonalFilter = true;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "bkg-relax-krate-constraint") == 0) {
                int stat = sscanf (optarg, "%d", &relaxKrateConstraint);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (relaxKrateConstraint < 0) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "bkg-damp-kmult") == 0) {
                int stat = sscanf (optarg, "%f", &damp_kmult);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (damp_kmult < 0) {
                    fprintf (stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }            if (strcmp (lOption, "cr-filter-train") == 0) {
                if (!strcmp(optarg,"off")) {
                    cafieResFilterTraining= 0;
                } else if (!strcmp(optarg,"on")) {
                    cafieResFilterTraining= 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cr-filter") == 0) {
                if (!strcmp(optarg,"off")) {
                    cafieResFilterCalling= 0;
                } else if (!strcmp(optarg,"on")) {
                    cafieResFilterCalling= 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cr-filter-tf") == 0) {
                if (!strcmp(optarg,"off")) {
                    cafieResFilterTFs = 0;
                } else if (!strcmp(optarg,"on")) {
                    cafieResFilterTFs = 1;
                } else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-residual-filter-max-flow") == 0) {
                int stat = sscanf (optarg, "%d", &cafieResMaxFlow);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (cafieResMaxFlow < 1) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-residual-filter-min-flow") == 0) {
                int stat = sscanf (optarg, "%d", &cafieResMinFlow);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (cafieResMinFlow < 0) {
                    fprintf (stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cafie-residual-filter-max-value") == 0) {
                cafieResMaxValueOverride = true;
                int stat = sscanf (optarg, "%lf", &cafieResMaxValue);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (cafieResMaxValue <= 0) {
                    fprintf (stderr, "Option Error: %s must specify a positive value (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "basecallsubsetfile") == 0) {
                basecallSubsetFile = strdup(optarg);
                readBasecallSubsetFile(basecallSubsetFile,basecallSubset);
            }
            if (strcmp (lOption, "per-flow-scale-file") == 0) {
                perFlowScaleFile = strdup(optarg);
                perFlowScale = true;
                readPerFlowScaleFile(perFlowScaleVal, perFlowScaleFile);
            }

            if (strcmp (lOption, "phred-table-file") == 0) {
                if (isFile(optarg)) {
                    phredTableFile = string( optarg );
                }
                else {
                    fprintf (stderr, "Invalid file specified for phred table: %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
            }

            if (strcmp (lOption, "region-size") == 0) {
                sPtr = strchr(optarg,'x');
                if (sPtr) {
                    int stat = sscanf (optarg, "%dx%d", &regionXSize, &regionYSize);
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
            if (strcmp (lOption, "mintfflows") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    minTFFlows = (int) input;
                }
            }

            if (strcmp (lOption, "alternateTFMode") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else {
                    alternateTFMode = (int) input;
                    cafieFlowMax = 120; // by default, give us a bump on how many flows we use to determine what TF we have
                }
            }

            if (strcmp (lOption, "from-wells") == 0) {
                if (isFile(optarg)) {
                    USE_RAWWELLS = 1;
                    NormalizeZeros = false;
                    strncpy (wellsFileName, basename(optarg), 256);
                    strncpy (wellsFilePath, dirname(optarg), 256);
                    dirExt = "./";  //fake source directory
                }
                else {
                    fprintf (stderr, "Invalid file specified: %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp(lOption, "sequence-all-lib") == 0) {
                for (unsigned int i = 0; i < strlen(optarg); i++)
                    optarg[i] = tolower(optarg[i]);
                if (strcmp(optarg, "off") == 0) {
                    sequenceAllLib = 0;
                } else if (strcmp(optarg, "on") == 0) {
                    sequenceAllLib = 1;
                } else {
                    fprintf(stderr, "Option Error: %c %s\n", c, optarg);
                    exit(EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "scaled-solve2") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    SCALED_SOLVE2 = (int) input;
                }
            }
            if (strcmp (lOption, "flowtimeoffset") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    flowTimeOffset = (int) input;
                }
            }
            if (strcmp (lOption, "cafieflowmax") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cafieFlowMax = (int) input;
                }
            }
            if (strcmp (lOption, "mintf") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    minTFCount = (int) input;
                }
            }
            if (strcmp (lOption, "numgroupsperregion") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    numGroupsPerRegion = (int) input;
                }
            }
            if (strcmp (lOption, "libcf-ie-dr") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%lf,%lf,%lf", &LibcfOverride, &LibieOverride, &LibdrOverride);
                    if (stat != 3) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                    libPhaseEstimator = "override";
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "tfcf-ie-dr") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%lf,%lf,%lf", &TFcfOverride, &TFieOverride, &TFdrOverride);
                    if (stat != 3) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                    TF_CAFIE_CORRECTION = 0;
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "initial-cfiedr") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%lf,%lf,%lf", &initial_cf, &initial_ie, &initial_dr);
                    if (stat != 3) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "cfiedr-keep-percent") == 0) {
                int stat = sscanf (optarg, "%lf", &cfiedrKeepPercent);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "ignore-hps") == 0) {
                // count how many HP entries are in our list
                numCafieHPIgnoreList = 0;
                int len = strlen(optarg);
                if (len > 0)
                    numCafieHPIgnoreList = 1;
                for (int k=0;k<len;k++) {
                    if (optarg[k] == ',')
                        numCafieHPIgnoreList++;
                }
                if (numCafieHPIgnoreList > 0) {
                    cafieHPIgnoreList = new int[numCafieHPIgnoreList];
                    int nextHP = 0;
                    char *ptr = optarg;
                    do {
                        sscanf(ptr, "%d", &cafieHPIgnoreList[nextHP]);
                        ptr = strchr(ptr, ',');
                        if (ptr) ptr++; // skip the comma so we can read the next int
                        nextHP++;
                    } while (ptr);
                    useCafieHPIgnoreList = true;
                    assert(nextHP == numCafieHPIgnoreList); // MGD - intel compiler not liking: && "Number of ints in HP list did not match our expected number?");
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "well-stat-file") == 0) {
                if (wellStatFile)
                    free(wellStatFile);
                wellStatFile = strdup(optarg);
            }
            if (strcmp(lOption, "basecaller") == 0) {
                basecaller = optarg;
            }
            if (strcmp(lOption, "cafie-solver") == 0) {
                basecaller = "cafie-solver";
            }
            if (strcmp(lOption, "phase-estimator") == 0) {
                libPhaseEstimator = optarg;
            }
            if (strcmp(lOption, "dot-fix-debug") == 0) {
                dotFixDebug = true;
            }
            if (strcmp(lOption, "try-all-tfs") == 0) {
                tryAllReads = true;
            }
            if (strcmp(lOption, "ignore-checksum-errors") == 0) {
                ignoreChecksumErrors |= 0x01;
            }
            if (strcmp(lOption, "ignore-checksum-errors-1frame") == 0) {
                ignoreChecksumErrors |= 0x02;
            }
            if (strcmp(lOption, "droop-mode") == 0) {
                if (droopMode)
                    free(droopMode);
                droopMode = strdup(optarg);
            }

            if (strcmp(lOption, "region-cafie-debug-file") == 0) {
                if (regionCafieDebugFile)
                    free(regionCafieDebugFile);
                regionCafieDebugFile = strdup(optarg);
            }
            if (strcmp (lOption, "flow-order") == 0) {
                if (flowOrder)
                    free(flowOrder);
                flowOrder = strdup(optarg);
                numFlowsPerCycle = strlen(flowOrder);
                flowOrderOverride = true;
                // if (numFlowsPerCycle > 4)
                // wantDotFixes = false; // MGD - for now, the CafieSolver lacks the ability to be smart about detecting valid 'dots' in the face of arbitrary flow orders
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
                    NNinnerx = inner;
                    NNinnery = inner;
                    NNouterx = outer;
                    NNoutery = outer;
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "nnmaskwh") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%d,%d,%d,%d", &NNinnerx, &NNinnery, &NNouterx, &NNoutery);
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
            if (strcmp (lOption, "num-cfiedr-fit-passes") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    numCFIEDRFitPasses = (int) input;
                }
            }
            if (strcmp (lOption, "librarykey") == 0) {
                libKey = (char *) malloc (strlen(optarg)+1);
                strcpy (libKey, optarg);
                ToUpper (libKey);
            }
            if (strcmp (lOption, "tfkey") == 0) {
                tfKey = (char *) malloc (strlen(optarg)+1);
                strcpy (tfKey, optarg);
                ToUpper (tfKey);
            }
            if (strcmp(lOption, "use-beadmask") == 0) {
                beadMaskFile = strdup (optarg);
            }
            if (strcmp(lOption, "beadmask-categorized") == 0) {
                maskFileCategorized = 1;
            }
            if (strcmp(lOption, "flag-bubbles") == 0) {
                filterBubbles = 1;
            }
            if (strcmp(lOption, "mintfscore") == 0) {
                minTFScore = atof (optarg);
            }
            if (strcmp (long_options[option_index].name, "bkg-emphasis") == 0) {
                sPtr = strchr(optarg,',');
                if (sPtr) {
                    int stat = sscanf (optarg, "%f,%f", &bkg_model_emphasis_width, &bkg_model_emphasis_amplitude);
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
                    int stat = sscanf (optarg, "%f", &dntp_uM);
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
                    int stat = sscanf (optarg, "%f", &AmplLowerLimit);
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
            if (strcmp (long_options[option_index].name, "bkg-effort-level") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%d", &bkgModelMaxIter);
                    if (stat != 1) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (EXIT_FAILURE);
                    }

                    if (bkgModelMaxIter < 5)
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
                gopt = optarg;
                if (strcmp(gopt, "disable") == 0 || strcmp(gopt, "opt") == 0);
                else
                {
                    FILE *gopt_file = fopen(gopt,"r");
                    if (gopt_file != NULL)
                        fclose(gopt_file);
                    else {
                        fprintf (stderr, "Option Error: %s cannot open file %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }
                }
            }
            if (strcmp (long_options[option_index].name, "xtalk") == 0) {
                xtalk = optarg;
                if (strcmp(xtalk, "disable") == 0 || strcmp(xtalk, "opt") == 0);
                else
                {
                    FILE *tmp_file = fopen(xtalk,"r");
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
                    int stat = sscanf (optarg, "%f,%f,%f,%f", &krate[0],&krate[1],&krate[2],&krate[3]);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }

                    for (int i=0;i < 3;i++)
                    {
                        if ((krate[i] < 0.01) || (krate[i] > 100.0))
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
                    int stat = sscanf (optarg, "%f,%f,%f,%f", &kmax[0],&kmax[1],&kmax[2],&kmax[3]);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }

                    for (int i=0;i < 3;i++)
                    {
                        if ((kmax[i] < 0.01) || (kmax[i] > 100.0))
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
                    int stat = sscanf (optarg, "%f,%f,%f,%f", &diff_rate[0],&diff_rate[1],&diff_rate[2],&diff_rate[3]);
                    if (stat != 4) {
                        fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                        exit (1);
                    }

                    for (int i=0;i < 3;i++)
                    {
                        if ((diff_rate[i] < 0.01) || (diff_rate[i] > 1000.0))
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
            if (strcmp (long_options[option_index].name, "cropped") == 0) {
                if (optarg) {
                    numCropRegions++;
                    cropRegions = (Region *) realloc (cropRegions, sizeof(Region) * numCropRegions);
                    int stat = sscanf (optarg, "%d,%d,%d,%d",
                                       &cropRegions[numCropRegions-1].col,
                                       &cropRegions[numCropRegions-1].row,
                                       &cropRegions[numCropRegions-1].w,
                                       &cropRegions[numCropRegions-1].h);
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
                                       &chipRegion.col,
                                       &chipRegion.row,
                                       &chipRegion.w,
                                       &chipRegion.h);
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
            if (strcmp (long_options[option_index].name, "beadfind-type") == 0) {
                beadfindType = optarg;
                if (beadfindType != "differential" && beadfindType != "original") {
                    fprintf (stderr, "*Error* - Illegal option to --beadfind-type: %s, valid options are 'differential' or 'original'\n",
                             beadfindType.c_str());
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "wells-format") == 0) {
                wellsFormat = optarg;
                if (wellsFormat != "legacy" && wellsFormat != "hdf5") {
                    fprintf (stderr, "*Error* - Illegal option to --wells-format: %s, valid options are 'legacy' or 'hdf5'\n",
                             wellsFormat.c_str());
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "beadfind-basis") == 0) {
                bfType = optarg;
                if (bfType != "signal" && bfType != "buffer") {
                    fprintf (stderr, "*Error* - Illegal option to --beadfind-basis: %s, valid options are 'signal' or 'buffer'\n",
                             bfType.c_str());
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (long_options[option_index].name, "beadfind-dat") == 0) {
                bfDat = optarg;
            }
            if (strcmp (long_options[option_index].name, "beadfind-bgdat") == 0) {
                bfBgDat = optarg;
            }
            if (strcmp(long_options[option_index].name, "beadfind-minlive") == 0) {
                bfMinLiveRatio = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-minlivesnr") == 0) {
                bfMinLiveLibSnr = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-tf-filter-quantile") == 0) {
                bfTfFilterQuantile = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-lib-filter-quantile") == 0) {
                bfLibFilterQuantile = atof (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-use-proj") == 0) {
                bfUseProj = atoi (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-skip-sd-recover") == 0) {
                skipBeadfindSdRecover = atoi (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-thumbnail") == 0) {
	      beadfindThumbnail = atoi (optarg);
            }
            if (strcmp(long_options[option_index].name, "beadfind-lagone-filt") == 0) {
	      beadfindLagOneFilt = atoi (optarg);
            }
            if (strcmp (long_options[option_index].name, "num-flows-to-fit") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    numFlowsToFitCafie1 = (int) input;
                }
            }
            if (strcmp (long_options[option_index].name, "num-flows-increment") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    numFlowsIncrement = (int) input;
                }
            }
            if (strcmp (long_options[option_index].name, "bead-washout") == 0) {
                SINGLEBF = false;
            }
            if (strcmp (long_options[option_index].name, "hilowfilter") == 0) {
                ToLower(optarg);
                if (strcmp (optarg, "true") == 0 ||
                        strcmp (optarg, "on") == 0 ||
                        atoi(optarg) == 1)
                {
                    hilowPixFilter = 1;
                }
                else
                {
                    hilowPixFilter = 0;
                }
            }
            if (strcmp (lOption, "gpuworkload") == 0) {
                int stat = sscanf (optarg, "%f", &gpuWorkLoad);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if ((gpuWorkLoad > 1) || (gpuWorkLoad < 0)) {
                    fprintf (stderr, "Option Error: %s must specify a value between 0 and 1 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "numgputhreads") == 0) {
                int stat = sscanf (optarg, "%d", &numGpuThreads);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (numGpuThreads <= 0 || numGpuThreads >=5 ) {
                    fprintf (stderr, "Option Error: %s must specify a value between 1 and 4 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "numcputhreads") == 0) {
                int stat = sscanf (optarg, "%d", &numCpuThreads);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (numCpuThreads <= 0) {
                    fprintf (stderr, "Option Error: %s must specify a value greater than 0 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }
            if (strcmp (lOption, "readaheaddat") == 0) {
                int stat = sscanf (optarg, "%d", &readaheadDat);
                if (stat != 1) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } else if (readaheadDat <= 0) {
                    fprintf (stderr, "Option Error: %s must specify a value greater than 0 (%s invalid).\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                } 
            }


            if (strcmp (long_options[option_index].name, "flowlimit") == 0) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    flowLimitSet = (unsigned int) input;
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

                    BkgTraceDebugRegions.push_back(dbg_reg);
                }
                else {
                    fprintf (stderr, "Option Error: %s %s\n", long_options[option_index].name,optarg);
                    exit (EXIT_FAILURE);
                }
            }

            if (strcmp (long_options[option_index].name, "cropped-region-origin") == 0) {
                if (optarg) {
                    int stat = sscanf (optarg, "%d,%d",
                                       &cropped_region_x_offset,
                                       &cropped_region_y_offset);
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

            if (strcmp (lOption, "output-dir") == 0) {
                OUTPUTDIR_OVERRIDE = strdup (optarg);
            }
            free (lOption);

            break;
        }
        /*  End processing long options */

        case 'b':   //beadfind file name
            /*
            **  When this is set, we override the find-washouts default by
            **  setting the preRun filename to NULL.
            */
            snprintf (preRunbfFileBase, 256, "%s", optarg);
            //sprintf (preRunbfFileBase, "");
            bfFileBase[0] = '\0';
            SINGLEBF = true;
            break;
        case 'c':
            fprintf (stderr,"\n* * * * * * * * * * * * * * * * * * * * * * * * * *\n");
            fprintf (stderr, "The --cycles, -c keyword has been deprecated.\n"
                     "Use the --flowlimit keyword instead.\n");
            fprintf (stderr,"* * * * * * * * * * * * * * * * * * * * * * * * * *\n\n");
            exit (EXIT_FAILURE);
            break;
        case 'F': // use cf/ie/dr from file
            libPhaseEstimator = "from-file";
            cfiedrRegionsX = 13;
            cfiedrRegionsY = 12;
            break;
        case 'f':   // maximum frames
            if (validIn (optarg, &input)) {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            else {
                maxFrames = (int) input;
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
                    lowerIntegralBound = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    upperIntegralBound = (int) input;
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
                KEYPASSFILTER = false;
            }
            else if (strcmp (optarg, "on") == 0) {
                KEYPASSFILTER = true;
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
                minPeakThreshold = (int) input;
            }
            break;
        case 'v':   //version
            fprintf (stdout, "%s", IonVersion::GetFullVersion("Analysis").c_str());
            exit (EXIT_SUCCESS);
            break;
        case 'A': // auto calculate cafie for library reads per region
            libPhaseEstimator = "lev-mar-cafiesolver";
            // if regions are still 1x1, assume they were not set, and provide a decent default for a 314 chip of 13x12
            cfiedrRegionsX = 13;
            cfiedrRegionsY = 12;
            // MGD note - this is just not the right way to do things, would much rather have the cafie region size be the input, and it can calculate the number of regions needed - this way it can scale to other chip types... and when the chip density/mm changes, we would then also want to scale so that region sizes roughly match a physical dimension
            break;
        case 'R': // use cfiedr in regions
            sPtr = strchr(optarg,'x');
            if (sPtr) {
                if (validIn (optarg, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfiedrRegionsX = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfiedrRegionsY = (int) input;
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
                    blockSizeX = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    blockSizeY = (int) input;
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
                    cfiedrRegionSizeX = (int) input;
                }
                if (validIn (++sPtr, &input)) {
                    fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                    exit (EXIT_FAILURE);
                }
                else {
                    cfiedrRegionSizeY = (int) input;
                }
            }
            else {
                fprintf (stderr, "Option Error: %c %s\n", c,optarg);
                exit (EXIT_FAILURE);
            }
            break;
        case 'D': // Separate first-pass to calculate droop
            usePass1Droop = true;
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
        dirExt = argv[c];
        break; //cause we only expect one non-option argument
    }

    if (!dirExt) {
        dirExt = (char *) malloc (2);
        snprintf (dirExt, 1, ".");  // assume current directory if not provided as an argument
    }

    // Test for a valid data source directory
    // Exception: if this is a re-analysis from wells file, then we can skip this test.
    if (isDir(dirExt) == false && (USE_RAWWELLS == 0)) {
        fprintf (stderr, "'%s' is not a directory.  Exiting.\n", dirExt);
        exit (EXIT_FAILURE);
    }

    //Determine total number of flows in experiment or previous analysis
    if (USE_RAWWELLS == 0) {
        numTotalFlows = GetTotalFlows(dirExt);
        assert (numTotalFlows > 0);
    }
    else {
        // Get total flows from processParams.txt
        numTotalFlows = atoi (GetProcessParam(wellsFilePath, "numFlows"));
        assert (numTotalFlows > 0);
    }

    //If flow order was not specified on command line,
    //set it here from info from explog.txt or processParams.txt
    if (!flowOrderOverride) {
        if (flowOrder)
            free (flowOrder);
        // Get flow order from the explog.txt file
        if (USE_RAWWELLS == 0) {
            flowOrder = GetPGMFlowOrder (dirExt);
            assert (flowOrder != NULL);
            numFlowsPerCycle = strlen (flowOrder);
            assert (numFlowsPerCycle > 0);
        }
        // Get flow order from the processParams.txt file
        else
        {
            flowOrder = GetProcessParam (wellsFilePath, "flowOrder");
            assert (flowOrder != NULL);
            numFlowsPerCycle = strlen (flowOrder);
            assert (numFlowsPerCycle > 0);
        }
    }

    // Adjust number of flows according to any command line options which may have been used
    // to limit these values
    if (flowLimitSet) {
        //support user specified number of flows
        numTotalFlows = (flowLimitSet < numTotalFlows ? flowLimitSet:numTotalFlows);
        assert (numTotalFlows > 0);
    }

    // Set some options that depend of flow order (unless explicitly set in command line)
    map<string,double>::iterator it;
    // cafieResMaxValue
    it = cafieResMaxValueByFlowOrder.find(string(flowOrder));
    if (!cafieResMaxValueOverride && it != cafieResMaxValueByFlowOrder.end()) {
        cafieResMaxValue = it->second;
    }
    // percentPositiveFlowsMaxValue
    it = percentPositiveFlowsMaxValueByFlowOrder.find(string(flowOrder));
    if (!percentPositiveFlowsMaxValueOverride && it != percentPositiveFlowsMaxValueByFlowOrder.end()) {
        percentPositiveFlowsMaxValue = it->second;
    }
    // percentPositiveFlowsMinValue
    it = percentPositiveFlowsMinValueByFlowOrder.find(string(flowOrder));
    if (!percentPositiveFlowsMinValueOverride && it != percentPositiveFlowsMinValueByFlowOrder.end()) {
        percentPositiveFlowsMinValue = it->second;
    }

    // Test some dependencies between options
    if (cafieResMinFlow >= cafieResMaxFlow) {
        fprintf (stderr, "value of --cafie-residual-filter-min-flow must be strictly less than that of --cafie-residual-filter-max-flow.\n");
        exit (EXIT_FAILURE);
    }
    if (percentPositiveFlowsMinFlow >= percentPositiveFlowsMaxFlow) {
        fprintf (stderr, "value of --percent-positive-flow-filter-min-flow must be strictly less than that of --percent-positive-flow-filter-max-flow.\n");
        exit (EXIT_FAILURE);
    }
    if (percentPositiveFlowsMinValue >= percentPositiveFlowsMaxValue) {
        fprintf (stderr, "value of --percent-positive-flow-filter-min-value must be strictly less than that of --percent-positive-flow-filter-max-value.\n");
        exit (EXIT_FAILURE);
    }
    if (perFlowScale) {
        if (perFlowScaleVal.size() < numTotalFlows) {
            fprintf (stderr, "%d per-flow scaling factors supplied for %d flows to analyze - exiting.\n",(int)perFlowScaleVal.size(),numTotalFlows);
            exit (EXIT_FAILURE);
        } else if (perFlowScaleVal.size() > numTotalFlows) {
            fprintf (stderr, "%d per-flow scaling factors supplied for %d flows to analyze - OK, but perhaps not what was intended?\n",(int)perFlowScaleVal.size(),numTotalFlows);
        }
    }
    
    //ChipType properties
    char *chipType = GetChipId(dirExt);
    ChipIdDecoder::SetGlobalChipId(chipType);
    int dims[2];
    GetChipDim(chipType, dims);
    chip_len_x = dims[0];
    chip_len_y = dims[1];

    //overwrite cafie region size (13x12)
    if ((cfiedrRegionSizeX != 0) && (cfiedrRegionSizeY != 0) && (blockSizeX != 0) && (blockSizeY != 0)) {
        std::cout << "INFO: blockSizeX: " << blockSizeX << " ,blockSizeY: " << blockSizeY << std::endl;
        cfiedrRegionsX = blockSizeX / cfiedrRegionSizeX;
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
    fprintf (fpLog, "dataDirectory = %s\n", dirExt);
    fprintf (fpLog, "runId = %s\n", runId);
    fprintf (fpLog, "flowOrder = %s\n", flowOrder);
    fprintf (fpLog, "wantDotFixes = %s\n", (wantDotFixes?"true":"false"));
    fprintf (fpLog, "washFlow = %d\n", GetWashFlow());
    fprintf (fpLog, "libraryKey = %s\n", libKey);
    fprintf (fpLog, "tfKey = %s\n", tfKey);
    fprintf (fpLog, "minNumKeyFlows = %d\n", minNumKeyFlows);
    fprintf (fpLog, "maxNumKeyFlows = %d\n", maxNumKeyFlows);
    fprintf (fpLog, "numFlows = %d\n", numTotalFlows);
    fprintf (fpLog, "cyclesProcessed = %d\n", numTotalFlows/4);
    fprintf (fpLog, "framesProcessed = %d\n", maxFrames);
    fprintf (fpLog, "framesInData = %d\n", totalFrames);
    //fprintf (fpLog, "framesPerSecond = %f\n", img.GetFPS());
    fprintf (fpLog, "minPeakThreshold = %d\n", minPeakThreshold);
    fprintf (fpLog, "lowerIntegrationTime = %d\n", lowerIntegralBound);
    fprintf (fpLog, "upperIntegrationTime = %d\n", upperIntegralBound);
    fprintf (fpLog, "bkgModelUsed = %s\n", (USE_BKGMODEL ? "true":"false"));
    fprintf (fpLog, "nucTraceCorrectionUsed = %s\n", (NUC_TRACE_CORRECT ? "true":"false"));
    fprintf (fpLog, "scaledSolve2Used = %s\n", (SCALED_SOLVE2 ? "true":"false"));
    fprintf (fpLog, "cafieResFilterTrainingUsed = %s\n", (cafieResFilterTraining ? "true":"false"));
    fprintf (fpLog, "cafieResFilterCallingUsed = %s\n", (cafieResFilterCalling ? "true":"false"));
    fprintf (fpLog, "cafieResFilterTFsUsed = %s\n", (cafieResFilterTFs ? "true":"false"));
    fprintf (fpLog, "sequenceAllLibUsed = %s\n", (sequenceAllLib ? "true":"false"));
    fprintf (fpLog, "cafieResMaxFlow = %d\n", cafieResMaxFlow);
    fprintf (fpLog, "cafieResMinFlow = %d\n", cafieResMinFlow);
    fprintf (fpLog, "cafieResMaxValue = %lf\n", cafieResMaxValue);
    fprintf (fpLog, "numFlowsToFit = %d\n", numFlowsToFitCafie1);
    fprintf (fpLog, "numFlowsIncrement = %d\n", numFlowsIncrement);
    fprintf (fpLog, "numCFIEDRFitPasses = %d\n", numCFIEDRFitPasses);
    fprintf (fpLog, "droopMode = %s\n", droopMode);
    fprintf (fpLog, "basecaller = %s\n", basecaller.c_str());
    fprintf (fpLog, "percentPositiveFlowsFilterCallingUsed = %s\n", (percentPositiveFlowsFilterCalling ? "true":"false"));
    fprintf (fpLog, "percentPositiveFlowsFilterTrainingUsed = %s\n", (percentPositiveFlowsFilterTraining ? "true":"false"));
    fprintf (fpLog, "percentPositiveFlowsFilterTFsUsed = %s\n", (percentPositiveFlowsFilterTFs ? "true":"false"));
    fprintf (fpLog, "percentPositiveFlowsMaxFlow = %d\n", percentPositiveFlowsMaxFlow);
    fprintf (fpLog, "percentPositiveFlowsMinFlow = %d\n", percentPositiveFlowsMinFlow);
    fprintf (fpLog, "percentPositiveFlowsMaxValue = %lf\n", percentPositiveFlowsMaxValue);
    fprintf (fpLog, "percentPositiveFlowsMinValue = %lf\n", percentPositiveFlowsMinValue);
    fprintf (fpLog, "clonalFilterTraining = %s\n", (clonalFilterTraining ? "true":"false"));
    fprintf (fpLog, "clonalFilterSolving = %s\n", (clonalFilterSolving ? "true":"false"));
    fprintf (fpLog, "hpScaleFactorUsed = %lf\n",hpScaleFactor);
    if (TF_CAFIE_CORRECTION == 0) {
        fprintf (fpLog, "TFcf-ie-dr values used = %0.5lf %0.5lf %0.5lf\n", TFcfOverride, TFieOverride, TFdrOverride);
    }
    if (libPhaseEstimator == "override") {
        fprintf (fpLog, "Libcf-ie-dr values used = %0.5lf %0.5lf %0.5lf\n", LibcfOverride, LibieOverride, LibdrOverride);
    }
    fprintf (fpLog, "nearest-neighborParameters = Inner: (%d,%d) Outer: (%d,%d)\n", NNinnerx, NNinnery, NNouterx, NNoutery);
    fprintf (fpLog, "Advanced beadfind = %s\n", BF_ADVANCED ? "enabled":"disabled");
    fprintf (fpLog, "minTFFlows = %d\n", minTFFlows);
    fprintf (fpLog, "minTFScore = %0.2lf\n", minTFScore);
    fprintf (fpLog, "cfiedroopRegions = %d (%dx%d)\n", cfiedrRegionsX * cfiedrRegionsY,cfiedrRegionsX,cfiedrRegionsY);
    fprintf (fpLog, "cfiedroopRegion dimensions = %dx%d\n", (int) ceil(cols/(double)cfiedrRegionsX), (int) ceil(rows/(double)cfiedrRegionsY));
    fprintf (fpLog, "numCafieSolveFlows = %d\n", numCafieSolveFlows);
    fprintf (fpLog, "use pinned wells = %s\n", USE_PINNED ? "true":"false");
    fprintf (fpLog, "use exclusion mask = %s\n", exclusionMaskSet ? "true":"false");
    fprintf (fpLog, "Version = %s\n", IonVersion::GetVersion().c_str());
    fprintf (fpLog, "Build = %s\n", IonVersion::GetBuildNum().c_str());
    fprintf (fpLog, "SvnRev = %s\n", IonVersion::GetSvnRev().c_str());
    fprintf (fpLog, "Chip = %d,%d\n", chip_len_x,chip_len_y);
    fprintf (fpLog, "Block = %d,%d,%d,%d\n", chip_offset_x,chip_offset_y,cols,rows);
    for (int q=0;q<numCropRegions;q++)
        fprintf (fpLog, "Cropped Region = %d,%d,%d,%d\n", cropRegions[q].col,cropRegions[q].row,cropRegions[q].w,cropRegions[q].h);
    fprintf (fpLog, "Analysis Region = %d,%d,%d,%d\n", chipRegion.col,chipRegion.row,chipRegion.col+chipRegion.w,chipRegion.row+chipRegion.h);
    fprintf (fpLog, "numRegions = %d\n", numRegions);
    fprintf (fpLog, "regionRows = %d\nregionCols = %d\n", regionsY, regionsX);
    fprintf (fpLog, "regionSize = %dx%d\n", regionXSize, regionYSize);
    //fprintf (fpLog, "\tRow Column Height Width\n");
    //for (int i=0;i<numRegions;i++)
    //  fprintf (fpLog, "[%3d] %5d %5d %5d %5d\n", i, regions[i].row, regions[i].col,regions[i].h,regions[i].w);
    fflush (NULL);
}

FILE * CommandLineOpts::InitFPLog ()
{
    char file[] = "processParameters.txt";
    char *fileName = (char *) malloc (strlen (experimentName) + strlen (file) + 2);
    sprintf (fileName, "%s/%s", experimentName, file);
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
char *CommandLineOpts::experimentDir (char *rawdataDir, char *dirOut)
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

void readPerFlowScaleFile(vector<float> &perFlowScaleVal, char *perFlowScaleFile) {
    char *buf = (char*) calloc(1,PER_FLOW_SCALE_MAX_LINE_LEN * sizeof(char));
    size_t bufSize = PER_FLOW_SCALE_MAX_LINE_LEN;
    FILE *fp = fopen(perFlowScaleFile, "r");
    if (fp) {
        int bytes_read = getline(&buf,&bufSize,fp);
        if (bytes_read == -1) {
            fprintf (stderr, "Failed to read first line of file %s\n", perFlowScaleFile);
            exit (EXIT_FAILURE);
        } else {
            int nFlow=0;
            int stat = sscanf(buf, "%d", &nFlow);
            if (stat != 1) {
                fprintf (stderr, "Invalid entry on first line of file %s\n", perFlowScaleFile);
                exit (EXIT_FAILURE);
            } else if (nFlow <= 0) {
                fprintf (stderr, "First entry should be a positive integer in %s\n", perFlowScaleFile);
                exit (EXIT_FAILURE);
            } else {
                perFlowScaleVal.resize(nFlow);
                for (int i=0; i<nFlow; i++) {
                    bytes_read = getline(&buf,&bufSize,fp);
                    if (bytes_read == -1) {
                        fprintf (stderr, "Failed to read line %d of file %s\n", (i+1), perFlowScaleFile);
                        exit (EXIT_FAILURE);
                    } else {
                        float val=0;
                        stat = sscanf(buf, "%f", &val);
                        if (stat != 1) {
                            fprintf (stderr, "Invalid entry on line %d of file %s\n", (i+1), perFlowScaleFile);
                            exit (EXIT_FAILURE);
                        } else if (val <= 0) {
                            fprintf (stderr, "Entry should be a positive float in line %d of %s\n", (i+1), perFlowScaleFile);
                        } else {
                            perFlowScaleVal[i] = val;
                        }
                    }
                }
            }
        }
        fclose(fp);
    } else {
        fprintf (stderr, "Failed to open per-flow scale file %s\n", perFlowScaleFile);
        exit (EXIT_FAILURE);
    }
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
