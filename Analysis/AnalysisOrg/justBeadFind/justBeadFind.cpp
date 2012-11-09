/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// Analysis Pipeline
// (c) 2009
// $Rev: 29830 $
//  $Date: 2012-04-24 13:57:04 -0400 (Tue, 24 Apr 2012) $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <armadillo>

#include "CommandLineOpts.h"
#include "Mask.h"
#include "Region.h"
#include "SeqList.h"
#include "TrackProgress.h"
#include "SlicedPrequel.h"
#include "SeparatorInterface.h"
#include "SetUpForProcessing.h"

#include "IonErr.h"
#include "RegionAnalysis.h"
#include "ImageSpecClass.h"
#include "MaskFunctions.h"

#include "dbgmem.h"


void DumpStartingStateOfProgram (int argc, char *argv[], TrackProgress &my_progress)
{
  char myHostName[128] = { 0 };
  gethostname (myHostName, 128);
  fprintf (stdout, "\n");
  fprintf (stdout, "Hostname = %s\n", myHostName);
  fprintf (stdout, "Start Time = %s", ctime (&my_progress.analysis_start_time));
  fprintf (stdout, "Version = %s-%s (%s) (%s)\n",
           IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
           IonVersion::GetSvnRev().c_str(), IonVersion::GetBuildNum().c_str());
  fprintf (stdout, "Command line = ");
  for (int i = 0; i < argc; i++)
    fprintf (stdout, "%s ", argv[i]);
  fprintf (stdout, "\n");
  fflush (NULL);
}

#ifdef _DEBUG
void memstatus (void)
{
  memdump();
  dbgmemClose();
}

#endif /* _DEBUG */


void TheSilenceOfTheArmadillos(ofstream &null_ostream)
{
    // Disable armadillo warning messages.
  arma::set_stream_err1(null_ostream);
  arma::set_stream_err2(null_ostream);
}


/*************************************************************************************************
 *************************************************************************************************
 *
 *  Start of Main Function
 *
 *************************************************************************************************
 ************************************************************************************************/
int main (int argc, char *argv[])
{

  init_salute();
#ifdef _DEBUG
  atexit (memstatus);
  dbgmemInit();
#endif /* _DEBUG */
  ofstream null_ostream("/dev/null"); // must stay live for entire scope, or crash when writing
  TheSilenceOfTheArmadillos(null_ostream);
  
  TrackProgress my_progress;  
  DumpStartingStateOfProgram (argc,argv,my_progress);

  CommandLineOpts inception_state (argc, argv);
  SeqListClass my_keys;
  ImageSpecClass my_image_spec;
  SlicedPrequel my_prequel_setup;  

  SetUpOrLoadInitialState(inception_state, my_keys, my_progress, my_image_spec, my_prequel_setup);

  // Start logging process parameters & timing now that we have somewhere to log
  my_progress.InitFPLog(inception_state);

  // Write processParameters.parse file now that processing is about to begin
  my_progress.WriteProcessParameters(inception_state);
  
  // Do separator
  Region wholeChip(0, 0, my_image_spec.cols, my_image_spec.rows);
  IsolatedBeadFind( my_prequel_setup, my_image_spec, wholeChip, inception_state,
        inception_state.sys_context.GetResultsFolder(), inception_state.sys_context.analysisLocation,  my_keys, my_progress);

  exit (EXIT_SUCCESS);
}


