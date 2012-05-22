/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GLOBALDEFAULTSFROMBKGCONTROL_H
#define GLOBALDEFAULTSFROMBKGCONTROL_H

#include "BkgControlOpts.h"
#include "BkgModel.h"

// Bad coding style, but it is what we do
// set global defaults (static, effectively a global variable) for the bkg model
// from the control options
void SetBkgModelGlobalDefaults(BkgModelControlOpts &bkg_control, char *chipType,char *experimentName);


#endif // GLOBALDEFAULTSFROMBKGCONTROL_H