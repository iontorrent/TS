/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// Analysis Pipeline
// (c) 2009
// $Rev: 23473 $
//  $Date: 2012-01-26 19:07:26 -0500 (Thu, 26 Jan 2012) $
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

#include "gsl/gsl_fit.h"

//#include "cudaWrapper.h"
#include "Image.h"
#include "Region.h"
#include "Mask.h"
#include "Filter.h"

#include "RawWells.h"

#include "fstrcmp.h"
#include "LinuxCompat.h"
#include "Utils.h"
#include "SeqList.h"
//#include "WorkerInfoQueue.h"
#include "SampleStats.h"
#include "Stats.h"
#include "CommandLineOpts.h"
#include "file-io/ion_util.h"
#include "ReservoirSample.h"
#include "IonErr.h"
#include "RegionAnalysis.h"
#include "ChipIdDecoder.h"
#include "PinnedWellReporter.h"
#include "TrackProgress.h"
#include "ImageSpecClass.h"
#include "ProcessImageToWell.h"
#include "BaseCaller.h"

#include "dbgmem.h"

// Uncomment below to enable Train-on-Live but process all beads
#define ALLBEADS


// un-comment the following line to output RawWells files that contain some of the fitted parameters
// of the background model
//#define BKG_DEBUG_PARAMETER_OUTPUT


void DumpStartingStateOfProgram(int argc, char *argv[], TrackProgress &my_progress)
{
  char myHostName[128] = { 0 };
  gethostname(myHostName, 128);
  fprintf(stdout, "\n");
  fprintf(stdout, "Hostname = %s\n", myHostName);
  fprintf(stdout, "Start Time = %s", ctime(&my_progress.analysis_start_time));
  fprintf(stdout, "Version = %s-%s (%s) (%s)\n",
          IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
          IonVersion::GetSvnRev().c_str(), IonVersion::GetBuildNum().c_str());
  fprintf(stdout, "Command line = ");
  for (int i = 0; i < argc; i++)
    fprintf(stdout, "%s ", argv[i]);
  fprintf(stdout, "\n");
  fflush(NULL);
}

#ifdef _DEBUG
void memstatus(void)
{
  memdump();
  dbgmemClose();
}

#endif /* _DEBUG */



void CreateResultsFolder(char *experimentName)
{
  // Create results folder
  if (mkdir(experimentName, 0777))
  {
    if (errno == EEXIST)
    {
      //already exists? well okay...
    }
    else
    {
      perror(experimentName);
      exit(EXIT_FAILURE);
    }
  }
}

void SetUpAnalysisLocation(CommandLineOpts &clo, char *experimentName, string &analysisLocation)
{
  // Start processing;
  char *analysisPath = NULL;
  char *tmpStr = NULL;
  char *tmpPath = strdup(experimentName);
  tmpStr = realpath(dirname(tmpPath), NULL);
  free(tmpPath);
  analysisPath = (char *) malloc(strlen(tmpStr) + strlen(experimentName) + 2);
  sprintf(analysisPath, "%s/%s", tmpStr, experimentName);
  fprintf(stdout, "Analysis results = %s\n\n", analysisPath);

  analysisLocation = analysisPath;
  char *analysisDir = NULL;
  if (clo.sys_context.NO_SUBDIR)
  {
    analysisDir = strdup(basename(tmpStr));
  }
  else
  {
    analysisDir = strdup(basename(analysisPath));
  }

  //  Create a run identifier from output results directory string
  ion_run_to_readname(clo.sys_context.runId, analysisDir, strlen(analysisDir));

  free(analysisDir);
  free(analysisPath);
  free(tmpStr);
}


void ExportSubRegionSpecsToMask(CommandLineOpts &clo)
{
  // Default analysis mode sets values to 0 and whole-chip processing proceeds.
  // otherwise, command line override (--analysis-region) can define a subchip region.
  Mask::chipSubRegion.row = (clo.GetChipRegion()).row;
  Mask::chipSubRegion.col = (clo.GetChipRegion()).col;
  Mask::chipSubRegion.h = (clo.GetChipRegion()).h;
  Mask::chipSubRegion.w = (clo.GetChipRegion()).w;
}



void InitPinnedWellReporterSystem(CommandLineOpts &clo)
{
  // Enable or disable the PinnedWellReporter system.
  bool bEnablePWR = false;
  if (0 != clo.img_control.outputPinnedWells)
    bEnablePWR = true;
  PWR::PinnedWellReporter::Instance(bEnablePWR);
}



/*************************************************************************************************
 *************************************************************************************************
 *
 *  Start of Main Function
 *
 *************************************************************************************************
 ************************************************************************************************/
int main(int argc, char *argv[])
{

  init_salute();
#ifdef _DEBUG
  atexit(memstatus);
  dbgmemInit();
#endif /* _DEBUG */

  TrackProgress my_progress;
  DumpStartingStateOfProgram(argc,argv,my_progress);

  CommandLineOpts clo(argc, argv);

  InitPinnedWellReporterSystem(clo);

  // Directory to which results files will be written
  char *experimentName = NULL;
  experimentName = strdup(clo.GetExperimentName());

  CreateResultsFolder(experimentName);
  CreateResultsFolder(clo.sys_context.basecaller_output_directory);

  string analysisLocation;
  SetUpAnalysisLocation(clo,experimentName,analysisLocation);

  // Start logging process parameters & timing now that we have somewhere to log to
  my_progress.fpLog = clo.InitFPLog();

  // create a raw wells file object
  // for new analysis, this is a file to be created; for reprocessing, this is the wells file to be read.
  // create the new wells file on a local partition.
  ClearStaleWellsFile();
  MakeNewTmpWellsFile(clo.sys_context, experimentName);

  RawWells rawWells(clo.sys_context.wellsFilePath, clo.sys_context.wellsFileName);

  int well_rows, well_cols; // dimension of wells file - found out from images if we use them - why is this separate from the rawWells object?

  // structure our flows & special key sequences we look for
  int numFlows = clo.GetNumFlows();

  SeqListClass my_keys;
  my_keys.StdInitialize(clo.flow_context.flowOrder,clo.key_context.libKey, clo.key_context.tfKey,my_progress.fpLog); // 8th duplicated flow processing code
  //@TODO: these parameters are just for reporting purposes???
  // they appear to be ignored everywhere
  my_keys.UpdateMinFlows(clo.key_context.minNumKeyFlows);
  my_keys.UpdateMaxFlows(clo.key_context.maxNumKeyFlows);

  // GENERATE FUNCTIONAL WELLS FILE & BEADFIND FROM IMAGES OR PREVIOUS PROCESS

  Region wholeChip;
  //Create empty Mask object
  ExportSubRegionSpecsToMask(clo);
  Mask bfmask(1, 1);
  Mask *maskPtr = &bfmask;

  GetFromImagesToWells(rawWells, maskPtr, clo, experimentName, analysisLocation, numFlows, my_keys,my_progress, wholeChip, well_rows,well_cols);

  if (!clo.mod_control.USE_RAWWELLS & clo.mod_control.WELLS_FILE_ONLY){
    // stop after generating the functional wells file
    UpdateBeadFindOutcomes(maskPtr, wholeChip, experimentName, clo, clo.mod_control.USE_RAWWELLS);
    my_progress.ReportState("Analysis (wells file only) Complete");
    CleanupTmpWellsFile(clo);

    free(experimentName);
    exit(EXIT_SUCCESS);
  }

  // Update progress bar status file: img proc complete/sig proc started
  updateProgress(IMAGE_TO_SIGNAL);

  // ==============================================
  //                BASE CALLING
  // ==============================================
  // no images below this point

  // operating from a wells file generated above
  GenerateBasesFromWells(clo, rawWells, maskPtr, my_keys.seqList, well_rows, well_cols, experimentName, my_progress);

  UpdateBeadFindOutcomes(maskPtr, wholeChip, experimentName, clo, clo.mod_control.USE_RAWWELLS);
  my_progress.ReportState("Analysis Complete");

  CleanupTmpWellsFile(clo);

  free(experimentName);
  exit(EXIT_SUCCESS);
}


