/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantCaller.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#include "VariantCaller.h"

#include <errno.h>

#include "MiscUtil.h"

#include "IonVersion.h"


void TheSilenceOfTheArmadillos(ofstream &null_ostream)
{
    // Disable armadillo warning messages.
  arma::set_stream_err1(null_ostream);
  arma::set_stream_err2(null_ostream);
}

int main(int argc, char* argv[]) {

  printf("tvc %s-%s (%s) - Torrent Variant Caller\n\n",
         IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());

  // stolen from "Analysis" to silence error messages from Armadillo library
  ofstream null_ostream("/dev/null"); // must stay live for entire scope, or crash when writing
  TheSilenceOfTheArmadillos(null_ostream);


  ExtendParameters parameters(argc, argv);

  LiveFiles active_files;
  active_files.ActivateFiles(parameters);
 
  InputStructures global_context;

  (parameters.program_flow.DEBUG > 0) ? global_context.DEBUG = 1 : global_context.DEBUG = 0;
  global_context.BringUpReferenceData(parameters);

  ThreadedVariantCaller(active_files.outVCFFile, active_files.filterVCFFile, global_context, &parameters);

  active_files.ShutDown();

  return 0;
}
