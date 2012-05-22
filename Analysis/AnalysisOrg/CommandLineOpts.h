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
#include "SystemContext.h"
#include "BkgControlOpts.h"
#include "BeadfindControlOpts.h"
#include "FilterControlOpts.h"
#include "BaseCallerControlOpts.h"
#include "SpatialContext.h"
#include "ImageControlOpts.h"

#define PER_FLOW_SCALE_MAX_LINE_LEN 1024

// define overall program flow
class ModuleControlOpts{
  public:
    int BEADFIND_ONLY;
    bool USE_BKGMODEL;
    int USE_RAWWELLS;
    int WELLS_FILE_ONLY;
    // controls information flow to bkg model from beadfind, so belongs here
    bool passTau;

    void DefaultControl();
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
