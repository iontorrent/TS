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
#include "HandleExpLog.h"
#include "SystemContext.h"
#include "BkgControlOpts.h"
#include "BeadfindControlOpts.h"
#include "KeyContext.h"
#include "FlowContext.h"
#include "SpatialContext.h"
#include "ImageControlOpts.h"
#include "FlowSequence.h"
#include "OptBase.h"

#define PER_FLOW_SCALE_MAX_LINE_LEN 1024

enum ValidTypes {
    VT_BOOL = 0
    , VT_INT
    , VT_FLOAT
    , VT_DOUBLE
    , VT_STRING
    , VT_VECTOR_INT
    , VT_VECTOR_FLOAT
    , VT_VECTOR_DOUBLE
    // Add new enums here...
    , VT_VECTOR_UNKOWN
};

class ValidateOpts
{
public:
	ValidateOpts();
	void Validate(const int argc, char *argv[]);

private:
	map<string, ValidTypes> m_opts;
};

// define overall program flow
class ModuleControlOpts{
  public:
    bool BEADFIND_ONLY;

    // controls information flow to bkg model from beadfind, so belongs here
    bool passTau;
    bool reusePriorBeadfind;  // true: reuse data from prior beadfind & skip new beadfind step; false: run beadfind

	ModuleControlOpts()
	{
		BEADFIND_ONLY = false;
		passTau = true;
		reusePriorBeadfind = false;
	}	
	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);
};

class ObsoleteOpts{
  public:
    int NUC_TRACE_CORRECT;
    bool USE_PINNED;
    int neighborSubtract;

	ObsoleteOpts()
	{
		NUC_TRACE_CORRECT = 0;
		USE_PINNED = false;
		neighborSubtract = 0;
	}

	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);
};


class CommandLineOpts {
public:
	CommandLineOpts() {}
    ~CommandLineOpts() {}

	void SetUpProcessing(); 
	void SetSysContextLocations();
    void SetFlowContext(string explog_path);

    // CHIP DEFAULTS WHICH SHOULD BE CONFIGURATION FILES
	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);

    /*---   options variables       ---*/
    // how the overall program flow will go
    ModuleControlOpts mod_control;

    // what context describes the local system environment
    SystemContext sys_context;
    
    // What does each module need to know?
    BkgModelControlOpts bkg_control;
    BeadfindControlOpts bfd_control;


    ImageControlOpts img_control;
    
    // these appear obsolete and useless
    ObsoleteOpts no_control;

    // what context describes the chip, the flow order used, and the keys
    // note these are three separate semantic entities
    SpatialContext loc_context;
    KeyContext key_context;
    FlowContext flow_context;

   /*---   end options variables   ---*/


protected:
private: 
    int numArgs;
    char **argvCopy;

};

#endif // COMMANDLINEOPTS_H
