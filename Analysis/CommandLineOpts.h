/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef COMMANDLINEOPTS_H
#define COMMANDLINEOPTS_H

#include <vector>
#include <string>
#include <map>
#include <set>
#include "Region.h"
#include "IonVersion.h"
#include "Utils.h"

#define PER_FLOW_SCALE_MAX_LINE_LEN 1024

class CommandLineOpts {
public:
    CommandLineOpts(int argc, char *argv[]);
    ~CommandLineOpts();
    
    void DefaultBkgModelControl();
    void DefaultCAFIEControl();
    void DefaultBeadfindControl();
    void DefaultFilterControl();
    void DefaultWellControl();

    void GetOpts(int argc, char *argv[]);
    void WriteProcessParameters();
    FILE *InitFPLog();
    char *GetExperimentName() {
        return (experimentName);
    }
    int GetWashFlow() {
        int hasWashFlow = HasWashFlow(dirExt);
        return (hasWashFlow < 0 ? 0 : hasWashFlow);
    }
    void PrintHelp();
    int GetNumFlows() {
        return (numTotalFlows);
    }

    /*---   options variables       ---*/
    char *dirExt;
    char *OUTPUTDIR_OVERRIDE;
    char dirOut[MAX_PATH_LENGTH];
    char *beadMaskFile;
    int maskFileCategorized;
    char bfFileBase[MAX_PATH_LENGTH];
    char preRunbfFileBase[MAX_PATH_LENGTH];
    char wellsFileName[MAX_PATH_LENGTH];
    char tmpWellsFile[MAX_PATH_LENGTH];
    int numRegions;
    int numCafieSolveFlows;
    int lowerIntegralBound; // Frame 15...(used to be 20, Added a little more at the start for bkgModel)
    int upperIntegralBound; // Frame 60
    int minPeakThreshold;
    int totalFrames;
    int maxFrames; // Set later from the first raw image header.
    char *sPtr;
    bool KEYPASSFILTER;

    //bool tryAllReads = true; // this determines what TF's we track to determine cf/ie/dr - normally would be set to false but TF's with lib key are being handled right now - risk that we include non-TF's as part of the cf/ie/dr calcs
    bool tryAllReads; // this determines what TF's we track to determine cf/ie/dr - normally would be set to false but TF's with lib key are being handled right now - risk that we include non-TF's as part of the cf/ie/dr calcs
    int NUC_TRACE_CORRECT;
    std::string libPhaseEstimator;
    int TF_CAFIE_CORRECTION;
    char *TFoverride;
    int cfiedrRegionsX, cfiedrRegionsY;
    int cfiedrRegionSizeX, cfiedrRegionSizeY;
    int blockSizeX, blockSizeY;
    bool usePass1Droop; // when set to true, we calculate droop as an independent param estimate, then just solve cf & ie
    int NO_SUBDIR; // when set to true, no experiment subdirectory is created for output files.
    double minTFScore; // if we can't score this TF with 85% confidence, its unknown
    // int     minSeqBases = 14; // if TF doesn't have this many bases, its ignored
    int minTFFlows; // 8 flows for key, plus at least one more cycle, or we ignore this TF
    int alternateTFMode; // better tuning for TF processing
    int cols;
    int rows;
    int regionXOrigin;
    int regionYOrigin;
    int regionXSize;
    int regionYSize;
    int regionsX;
    int regionsY;
    Region *cropRegions;
    int numCropRegions;
    int USE_RAWWELLS;
    int flowTimeOffset;
    int cafieFlowMax;
    int minTFCount;
    double LibcfOverride;
    double LibieOverride;
    double LibdrOverride;
    double TFcfOverride;
    double TFieOverride;
    double TFdrOverride;
    double initial_cf;
    double initial_ie;
    double initial_dr;
    char runId[6];
    int NNinnerx;
    int NNinnery;
    int NNouterx;
    int NNoutery;
    char *libKey;
    char *tfKey;
    char *flowOrder;
    bool flowOrderOverride;
    int neighborSubtract;
    int numGroupsPerRegion;
    bool USE_BKGMODEL;
    int BEADFIND_ONLY;
    int noduds;
    bool SINGLEBF;
    bool NormalizeZeros;
    int singleCoreCafie;
    int USE_PINNED;
    int BF_ADVANCED;
    int LOCAL_WELLS_FILE;
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
    int *flowOrderIndex;
    int SCALED_SOLVE2;
    int NONLINEAR_HP_SCALE;
    int wantPerWellCafie;
    char *droopMode;
    double hpScaleFactor;
    char wellsFilePath[MAX_PATH_LENGTH];
    char *wellStatFile;
    bool dotFixDebug;
    std::string basecaller;
    char *regionCafieDebugFile;
    bool wantDotFixes;
    int doCafieResidual;
    int nUnfilteredLib;
    char *unfilteredLibDir;
    char *beadSummaryFile;
    char *experimentName;
    int maxNumKeyFlows;
    int minNumKeyFlows;
    bool exclusionMaskSet;
    int skiptfbasecalling;
    int numCFIEDRFitPasses;
    int sequenceAllLib;
    int minReadLength;
    int hilowPixFilter;
    bool useCafieHPIgnoreList; // defaults to false - just ignore all HP's when estimating
    int *cafieHPIgnoreList; // list of HP's to ignore
    int numCafieHPIgnoreList;
    bool cafieFitIgnoreLowQual;
    int ignoreChecksumErrors; // set to true to force corrupt checksum files to load anyway - beware!
    // Options related to filtering reads by percentage of positive flows
    int percentPositiveFlowsFilterTraining;
    int percentPositiveFlowsFilterCalling;
    int percentPositiveFlowsFilterTFs;
    int percentPositiveFlowsMaxFlow;
    int percentPositiveFlowsMinFlow;
    double percentPositiveFlowsMaxValue;
    bool percentPositiveFlowsMaxValueOverride;
    std::map<std::string,double> percentPositiveFlowsMaxValueByFlowOrder; // For holding flow-specific values.
    double percentPositiveFlowsMinValue;
    bool percentPositiveFlowsMinValueOverride;
    std::map<std::string,double> percentPositiveFlowsMinValueByFlowOrder; // For holding flow-specific values.
    // Options related to filtering reads by putative clonality
    int clonalFilterTraining;
    int clonalFilterSolving;
    // Options related to filtering reads by CAFIE residuals
    int cafieResFilterTraining;
    int cafieResFilterCalling;
    int cafieResFilterTFs;
    int cafieResMaxFlow;
    int cafieResMinFlow;
    double cafieResMaxValue;
    bool cafieResMaxValueOverride; // Will be true if the value is explicitly set on command line
    std::map<std::string,double> cafieResMaxValueByFlowOrder; // For holding flow-specific values.
    // Options related to doing basecalling on just a subset of wells
    char *basecallSubsetFile;
    std::set< std::pair <unsigned short,unsigned short> > basecallSubset;
    // Options related to per-flow scaling
    bool perFlowScale;
    char *perFlowScaleFile;
    std::vector<float> perFlowScaleVal;
    int numFlowsToFitCafie1; // num flows to cafie fit pass 1
    int numFlowsIncrement; // multiple pass increment
    double cfiedrKeepPercent; // 0.0 = median, 1.0 = mean, 0.6 = truncated mean
    std::string phredTableFile;
    unsigned int numFlowsPerCycle;
    std::string wellsFormat;
    std::string beadfindType;
    int filterBubbles;
    int bkgDebugParam;
    bool enableXtalkCorrection;
    bool enableBkgModelClonalFilter;
    int relaxKrateConstraint;
    float damp_kmult; // dampen kmult variation
    std::string bfType; // signal or buffer
    std::string bfDat;
    std::string bfBgDat;
    double bfMinLiveRatio;
    double bfMinLiveLibSnr;
    double bfTfFilterQuantile;
    double bfLibFilterQuantile;
    int bfUseProj;

    int skipBeadfindSdRecover;
    int beadfindThumbnail; // Is this a thumbnail chip where we need to skip smoothing across regions?
    int beadfindLagOneFilt;
    // commandline options for GPU for background model computation
    float gpuWorkLoad;
    int numGpuThreads;
    int numCpuThreads;

    int vectorize;
    int outputPinnedWells;

    int cropped_region_x_offset;
    int cropped_region_y_offset;
    int chip_offset_x;
    int chip_offset_y;
    int chip_len_x;
    int chip_len_y;
    int readaheadDat;

    // only the row and col fields are used to specify location of debug regions
    std::vector<Region> BkgTraceDebugRegions;

    struct Region GetChipRegion() {
        return chipRegion;
    }

    /*---   end options variables   ---*/
    FILE *fpLog;

protected:
    unsigned int numTotalFlows;
private:
    int numArgs;
    unsigned int flowLimitSet;
    char **argvCopy;
    char *experimentDir(char *rawdataDir, char *dirOut);
    struct Region chipRegion;
};

#endif // COMMANDLINEOPTS_H
