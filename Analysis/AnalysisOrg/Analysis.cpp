/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// Analysis Pipeline
// (c) 2009
// $Rev: 43200 $
//  $Date: 2012-09-24 10:33:48 -0700 (Mon, 24 Sep 2012) $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <limits.h>
#include <signal.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>


#include "Image.h"
#include "Region.h"
#include "Mask.h"


#include "RawWells.h"

//#include "fstrcmp.h"
#include "LinuxCompat.h"
#include "Utils.h"
#include "SeqList.h"

#include "SampleStats.h"
#include "Stats.h"
#include "CommandLineOpts.h"

#include "ReservoirSample.h"
#include "IonErr.h"
#include "RegionAnalysis.h"
#include "ChipIdDecoder.h"
#include "SetUpForProcessing.h"
#include "TrackProgress.h"
#include "ImageSpecClass.h"
#include "ProcessImageToWell.h"
#include "StackUnwind.h"

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

  InitStackUnwind(inception_state.sys_context.stackDumpFile);

  SetUpOrLoadInitialState(inception_state, my_keys, my_progress, my_image_spec, my_prequel_setup);
  
  // Start logging process parameters & timing now that we have somewhere to log
  my_progress.InitFPLog(inception_state);

  // Write processParameters.parse file now that processing is about to begin
  my_progress.WriteProcessParameters(inception_state);

  // Do background model
  RealImagesToWells ( inception_state, my_keys, my_progress, my_image_spec,
		      my_prequel_setup);

  my_progress.ReportState ("Analysis (wells file only) Complete");

  exit (EXIT_SUCCESS);
}



