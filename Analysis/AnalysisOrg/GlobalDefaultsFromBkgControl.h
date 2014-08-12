/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GLOBALDEFAULTSFROMBKGCONTROL_H
#define GLOBALDEFAULTSFROMBKGCONTROL_H

#include "BkgControlOpts.h"
#include "SignalProcessingMasterFitter.h"

// Bad coding style, but it is what we do
// set global defaults (static, effectively a global variable) for the bkg model
// from the control options
void SetBkgModelGlobalDefaults(GlobalDefaultsForBkgModel &global_defaults, BkgModelControlOpts &bkg_control, const char *chipType,char *results_folder);


#endif // GLOBALDEFAULTSFROMBKGCONTROL_H
