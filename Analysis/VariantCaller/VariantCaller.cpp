/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantCaller.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#include "VariantCaller.h"

#include <errno.h>

#include "MiscUtil.h"

#include "IonVersion.h"


int main(int argc, char* argv[]) {

  printf("tvc %s-%s (%s) - Torrent Variant Caller\n\n",
         IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());

  ExtendParameters parameters(argc, argv);

  LiveFiles active_files;
  active_files.ActivateFiles(parameters);
 
  InputStructures global_context;

  (parameters.program_flow.DEBUG > 0) ? global_context.DEBUG = 1 : global_context.DEBUG = 0;
  global_context.BringUpReferenceData(parameters);

  ThreadedVariantCaller(active_files.outVCFFile, active_files.filterVCFFile, active_files.consensusFile, global_context, &parameters);

  active_files.ShutDown();

  return 0;
}
