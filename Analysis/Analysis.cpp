/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// Analysis Pipeline
// (c) 2009
// $Rev: 20450 $
//  $Date: 2011-11-30 13:08:01 -0800 (Wed, 30 Nov 2011) $
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
//#include "WorkerInfoQueue.h"
#include "SampleStats.h"
#include "Stats.h"
#include "CommandLineOpts.h"
#include "Flow.h"
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
  if (clo.NO_SUBDIR)
  {
    analysisDir = strdup(basename(tmpStr));
  }
  else
  {
    analysisDir = strdup(basename(analysisPath));
  }

  //  Create a run identifier from output results directory string
  ion_run_to_readname(clo.runId, analysisDir, strlen(analysisDir));

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

void InitializeSeqList(SequenceItem *seqList, int numSeqListItems, CommandLineOpts &clo, FILE *fpLog, Flow *flw)
{
  seqList[0].seq = clo.tfKey;
  seqList[1].seq = clo.libKey;
  //const int numSeqListItems = sizeof(seqList) / sizeof(SequenceItem);
  // Calculate number of key flows & Ionogram
  //  TFs tempTF(flw->GetFlowOrder()); // MGD note - would like GenerateIonogram to be in the utils lib soon
  for (int i = 0; i < numSeqListItems; i++)
  {
    int zeroMerCount = 0;
    int oneMerCount = 0;
    seqList[i].len = strlen(seqList[i].seq);
    //    seqList[i].numKeyFlows = tempTF.GenerateIonogram(seqList[i].seq,
    //        seqList[i].len, seqList[i].Ionogram);
    seqList[i].numKeyFlows = seqToFlow(seqList[i].seq, seqList[i].len,
                                       seqList[i].Ionogram, 64, flw->GetFlowOrder(), strlen(flw->GetFlowOrder()));

    seqList[i].usableKeyFlows = seqList[i].numKeyFlows - 1;
    // and calculate for the given flow order, what nucs are 1-mers and which are 0-mers
    // requirement is that we get at least one flow for each nuc that has a 0 and a 1
    // it just might take lots of flows for a given key
    int flow;
    for (flow = 0; flow < seqList[i].numKeyFlows; flow++)
    {
      seqList[i].onemers[flow] = -1;
      seqList[i].zeromers[flow] = -1;
    }
    for (flow = 0; flow < seqList[i].numKeyFlows; flow++)
    {
      if (seqList[i].Ionogram[flow] == 1)
      {
        // for now just mark the first occurance of any nuc hit
        if (seqList[i].onemers[flw->GetNuc(flow)] == -1)
        {
          oneMerCount++;
          seqList[i].onemers[flw->GetNuc(flow)] = flow;
        }
      }
      else
      {
        // for now just mark the first occurance of any nuc hit
        if (seqList[i].zeromers[flw->GetNuc(flow)] == -1)
        {
          zeroMerCount++;
          seqList[i].zeromers[flw->GetNuc(flow)] = flow;
        }
      }
    }
    if (oneMerCount <= 1 || zeroMerCount <= 1)
    {
      fprintf(
        fpLog,
        "Key: '%s' with flow order: '%s' does not have at least 2 0mers and 2 1mers.\n",
        seqList[i].seq, flw->GetFlowOrder());
      fprintf(
        stderr,
        "Key: '%s' with flow order: '%s' does not have at least 2 0mers and 2 1mers.\n",
        seqList[i].seq, flw->GetFlowOrder());
      exit(EXIT_FAILURE);
    }
    if (seqList[i].numKeyFlows > clo.maxNumKeyFlows)
      clo.maxNumKeyFlows = seqList[i].numKeyFlows;
    if ((seqList[i].numKeyFlows - 1) < clo.minNumKeyFlows)
      clo.minNumKeyFlows = seqList[i].numKeyFlows - 1;
  }
}



void InitPinnedWellReporterSystem(CommandLineOpts &clo)
{
  // Enable or disable the PinnedWellReporter system.
  bool bEnablePWR = false;
  if (0 != clo.outputPinnedWells)
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

  string analysisLocation;
  SetUpAnalysisLocation(clo,experimentName,analysisLocation);

  // Start logging process parameters & timing now that we have somewhere to log to
  my_progress.fpLog = clo.InitFPLog();

  // create a raw wells file object
  // for new analysis, this is a file to be created; for reprocessing, this is the wells file to be read.
  // create the new wells file on a local partition.
  ClearStaleWellsFile();
  MakeNewTmpWellsFile(clo, experimentName);

  RawWells rawWells(clo.wellsFilePath, clo.wellsFileName);

  int well_rows, well_cols; // dimension of wells file - found out from images if we use them - why is this separate from the rawWells object?

  // structure our flows & special key sequences we look for
  int numFlows = clo.GetNumFlows();

  Flow *flw;
  flw = new Flow(clo.flowOrder);
  SequenceItem seqList[] = { { MaskTF, "ATCG", 0, 0, 0, {0}, {0}, {0} }, { MaskLib, "TCAG", 0, 0, 0, {0}, {0}, {0} } };
  int numSeqListItems = 2;
  InitializeSeqList(seqList,numSeqListItems,clo,my_progress.fpLog,flw);

  // GENERATE FUNCTIONAL WELLS FILE & BEADFIND FROM IMAGES OR PREVIOUS PROCESS

  Region wholeChip;
  //Create empty Mask object
  ExportSubRegionSpecsToMask(clo);
  Mask bfmask(1, 1);
  Mask *maskPtr = &bfmask;

  GetFromImagesToWells(rawWells, maskPtr, clo, experimentName, analysisLocation,flw,numFlows, seqList,numSeqListItems,my_progress, wholeChip, well_rows,well_cols);

  // Update progress bar status file: img proc complete/sig proc started
  updateProgress(IMAGE_TO_SIGNAL);

  // ==============================================
  //                BASE CALLING
  // ==============================================
  // no images below this point

  // operating from a wells file generated above
  GenerateBasesFromWells(clo, rawWells, flw, maskPtr, seqList, well_rows, well_cols,
                         experimentName, my_progress);

  UpdateBeadFindOutcomes(maskPtr, wholeChip, experimentName, clo, clo.USE_RAWWELLS);
  my_progress.ReportState("Analysis Complete");

  CleanupTmpWellsFile(clo);

  free(experimentName);
  if (flw!=NULL) delete flw;
  exit(EXIT_SUCCESS);
}


