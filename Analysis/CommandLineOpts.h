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

// define overall program flow
class ModuleControlOpts{
  public:
    int BEADFIND_ONLY;
    bool USE_BKGMODEL;
    int USE_RAWWELLS;
    bool WELLS_FILE_ONLY;

    void DefaultControl();
};

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
    int relaxKrateConstraint;
    float damp_kmult; // dampen kmult variation
    int bkgDebugParam;
    // temporary: dump debugging information for all beads, not just one
   int debug_bead_only;
    // commandline options for GPU for background model computation
    float gpuWorkLoad;
    int numGpuThreads;
    int numCpuThreads;

    int vectorize;
    // only the row and col fields are used to specify location of debug regions
    std::vector<Region> BkgTraceDebugRegions;
    int readaheadDat;
    int saveWellsFrequency;
    int filterBubbles;
    
    void DefaultBkgModelControl();

};

class BeadfindControlOpts{
  public:
    double bfMinLiveRatio;
    double bfMinLiveLibSnr;
    double bfMinLiveTfSnr;
    double bfTfFilterQuantile;
    double bfLibFilterQuantile;
    int skipBeadfindSdRecover;
    int beadfindThumbnail; // Is this a thumbnail chip where we need to skip smoothing across regions?
    int beadfindLagOneFilt;
    char *beadMaskFile;
    int maskFileCategorized;
    char bfFileBase[MAX_PATH_LENGTH];
    char preRunbfFileBase[MAX_PATH_LENGTH];
    int noduds;
    std::string beadfindType;
    std::string bfType; // signal or buffer
    std::string bfDat;
    std::string bfBgDat;
    bool SINGLEBF;
    int BF_ADVANCED;
    
    void DefaultBeadfindControl();
    ~BeadfindControlOpts();
};


class FilterControlOpts{
  public:
    int percentPositiveFlowsFilterTraining;
    int percentPositiveFlowsFilterCalling;
    int percentPositiveFlowsFilterTFs;
    bool KEYPASSFILTER;
    // Options related to filtering reads by percentage of positive flows
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
    // too short!
    int minReadLength;

    // unfiltered summary
    int nUnfilteredLib;
    char *unfilteredLibDir;
    char *beadSummaryFile;
    
    void DefaultFilterControl();
    void RecognizeFlow(char *flowFormula);
    ~FilterControlOpts();
};


class CafieControlOpts{
  public:
    int singleCoreCafie;
    double LibcfOverride;
    double LibieOverride;
    double LibdrOverride;
    std::string libPhaseEstimator;
    std::string basecaller;
    int cfiedrRegionsX, cfiedrRegionsY;
    int cfiedrRegionSizeX, cfiedrRegionSizeY;
    int blockSizeX, blockSizeY;
    int numCafieSolveFlows;

    int doCafieResidual;
    char *basecallSubsetFile;
    std::set< std::pair <unsigned short,unsigned short> > basecallSubset;

    // should this be in system context?
    std::string phredTableFile;
    
    void DefaultCAFIEControl();
    void EchoDerivedChipParams(int chip_len_x, int chip_len_y);
    ~CafieControlOpts();
};

// handles file i/o and naming conventions
class SystemContext{
  public:
     char *dat_source_directory;
    char *wells_output_directory;
    char *basecaller_output_directory;
    
    char wellsFileName[MAX_PATH_LENGTH];
    char tmpWellsFile[MAX_PATH_LENGTH];
    char runId[6];
    char wellsFilePath[MAX_PATH_LENGTH];
    char *wellStatFile;
    char *experimentName;
    int NO_SUBDIR; // when set to true, no experiment subdirectory is created for output files.
    int LOCAL_WELLS_FILE;
    std::string wellsFormat;

    char *experimentDir(char *rawdataDir, char *dirOut);
    void DefaultSystemContext();
    void GenerateContext(int from_wells);
    ~SystemContext();
};

// cropping, displacing, locating items on the chip
// some of this is probably part of bkgmodel controls (analysis regions)
// some of this is also probably part of Image tracking 
class SpatialContext{
  public:
    int numRegions;
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
     int cropped_region_x_offset;
    int cropped_region_y_offset;
    int chip_offset_x;
    int chip_offset_y;
    int chip_len_x;
    int chip_len_y;
     struct Region chipRegion;

     // do we have regions that are known to be excluded
   bool exclusionMaskSet;

   void FindDimensionsByType(char *dat_source_directory);
     void DefaultSpatialContext();
     ~SpatialContext();
};

// control options on loading dat files
class ImageControlOpts{
  public:
     int totalFrames;
    int maxFrames; // Set later from the first raw image header.
      int NNinnerx;
    int NNinnery;
    int NNouterx;
    int NNoutery;
     int hilowPixFilter;
    int ignoreChecksumErrors; // set to true to force corrupt checksum files to load anyway - beware!
    int flowTimeOffset;
    // do diagnostics?
   int outputPinnedWells;

  void DefaultImageOpts();
};

class KeyContext{
  public:
     char *libKey;
    char *tfKey;
    int maxNumKeyFlows;
    int minNumKeyFlows;
    
    void DefaultKeys();
    ~KeyContext();
};

// track the flow formula which gets translated at least 4 separate times in the code into the actual flows done by the PGM
// obvious candidate for centralized code
class FlowContext{
  public:
      char *flowOrder;
    bool flowOrderOverride;
     int *flowOrderIndex;  // obviously this contains the nuc type per flow for all flows
   unsigned int numFlowsPerCycle;
    unsigned int flowLimitSet;
    unsigned int numTotalFlows;

   void DefaultFlowFormula();
   void DetectFlowFormula(SystemContext &sys_context, int from_wells); // if not specified, go find it out
   ~FlowContext();
};

class ObsoleteOpts{
  public:
    int NUC_TRACE_CORRECT;
    int USE_PINNED;
    int lowerIntegralBound; // Frame 15...(used to be 20, Added a little more at the start for bkgModel)
    int upperIntegralBound; // Frame 60
    int minPeakThreshold;
    int neighborSubtract;

    void Defaults();
};



class CommandLineOpts {
public:
    CommandLineOpts(int argc, char *argv[]);
    ~CommandLineOpts();

    void GetOpts(int argc, char *argv[]);
    void WriteProcessParameters();
    FILE *InitFPLog();
    char *GetExperimentName() {
        return (sys_context.experimentName);
    }
    int GetWashFlow() {
        int hasWashFlow = HasWashFlow(sys_context.dat_source_directory);
        return (hasWashFlow < 0 ? 0 : hasWashFlow);
    }
    void PrintHelp();
    int GetNumFlows() {
        return (flow_context.numTotalFlows);
    }
    struct Region GetChipRegion() {
        return loc_context.chipRegion;
    }

    /*---   options variables       ---*/
    // how the overall program flow will go
     ModuleControlOpts mod_control;
    // what context describes the local system environment
    SystemContext sys_context;
    
    // What does each module need to know?
    BkgModelControlOpts bkg_control;
    BeadfindControlOpts bfd_control;
    FilterControlOpts flt_control;
    CafieControlOpts cfe_control;
    ImageControlOpts img_control;
    
    // these appear obsolete and useless
    ObsoleteOpts no_control;

    // what context describes the chip, the flow order used, and the keys
    // note these are three separate semantic entities
    SpatialContext loc_context;
    KeyContext key_context;
    FlowContext flow_context;

   /*---   end options variables   ---*/
    FILE *fpLog;

protected:
private:
    int numArgs;
    char **argvCopy;
    char *sPtr; // only used internally, I believe
};

#endif // COMMANDLINEOPTS_H
