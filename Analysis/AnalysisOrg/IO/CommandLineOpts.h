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

#define PER_FLOW_SCALE_MAX_LINE_LEN 1024

// define overall program flow
class ModuleControlOpts{
  public:
    int BEADFIND_ONLY;
    bool USE_BKGMODEL;

    // controls information flow to bkg model from beadfind, so belongs here
    bool passTau;
    bool reusePriorBeadfind;  // true: reuse data from prior beadfind & skip new beadfind step; false: run beadfind

    void DefaultControl();
};




class ObsoleteOpts{
  public:
    int NUC_TRACE_CORRECT;
    int USE_PINNED;

    int neighborSubtract;

    void Defaults();
};

// please hope that this does not grow to enormous size
struct RadioButtonOptions{
  bool use_dud_reference_set;
  bool empty_well_normalization_set ;
  bool single_flow_fit_max_retry_set ;
  bool gain_correct_images_set ;
  bool per_flow_t_mid_nuc_tracking_set ;
  bool regional_sampling_set ;
  bool use_proton_correction_set;
  bool amplitude_lower_limit_set;
  bool clonal_solve_bkg_set;
  bool col_flicker_correct_set;
  RadioButtonOptions(){
    use_dud_reference_set = false;
    empty_well_normalization_set = false;
    single_flow_fit_max_retry_set = false;
    gain_correct_images_set = false;
    col_flicker_correct_set = false;
    per_flow_t_mid_nuc_tracking_set = false;
    regional_sampling_set = false;
    use_proton_correction_set = false;
    amplitude_lower_limit_set = false;
    clonal_solve_bkg_set = false;
  };
};

class CommandLineOpts {
public:
    CommandLineOpts(int argc, char *argv[]);
    ~CommandLineOpts();


    void GetOpts(int argc, char *argv[]); 
    std::string GetCmdLine();
    void SetUpProcessing(); 
    void SetSysContextLocations();
    void SetFlowContext(char *explog_path);
    void SetProtonDefault();

    void PrintHelp();


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
    void SetAnyLongBeadFindOption(char *lOption, const char *original_name);
    void SetAnyLongSignalProcessingOption(char *lOption, const char *original_name);
    void SetAnyLongImageProcessingOption(char *lOption, const char *original_name);
    void SetAnyLongSpatialContextOption(char *lOption, const char *original_name);
    void SetLongKeyOption(char *lOption, const char *original_name);
    void SetSystemContextOption(char *lOption, const char *original_name);
    void SetFlowContextOption(char *lOption, const char *original_name);
    void SetModuleControlOption(char *lOption, const char *original_name);

    void PickUpSourceDirectory(int argc, char *argv[] );
    void SetGlobalChipID(char *dat_source_directory);  // only do this >once<
        
    int numArgs;
    char **argvCopy;
    char *sPtr; // only used internally, I believe
   // control of how everything got set
   // never use these for anything outside this class
   RadioButtonOptions radio_buttons;
};

#endif // COMMANDLINEOPTS_H
