/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SYSTEMCONTEXT_H
#define SYSTEMCONTEXT_H

#include "stdlib.h"
#include "stdio.h"
#include <unistd.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "Region.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "SpecialDataTypes.h"
#include "SeqList.h"

// handles file i/o and naming conventions
// use this when a routine needs to know the directories we're operating within
// or where to put temporary files
// or whatever...

// the idea is to actually isolate the logic so it can be comprehended and used without being spread
//  in a dozen locations across the code.
class SystemContext{
  public:
     char *dat_source_directory;
    char *wells_output_directory;
    char *basecaller_output_directory;

    char wellsFileName[MAX_PATH_LENGTH];
    char tmpWellsFile[MAX_PATH_LENGTH];
    char runId[6];
    char wellsFilePath[MAX_PATH_LENGTH];
    char *wellStatFile;
    char *experimentName;
    int NO_SUBDIR; // when set to true, no experiment subdirectory is created for output files.
    int LOCAL_WELLS_FILE;
    std::string wellsFormat;

    char *experimentDir(char *rawdataDir, char *dirOut);
    void DefaultSystemContext();
    void GenerateContext(int from_wells);
    // why we >pass< experimentName to this routine is a big mystery...
    // expeerimentName is not the experimentName, it's a directory
    void SetUpAnalysisLocation(char *experimentName, std::string &analysisLocation);
    void MakeSymbolicLinkToOldDirectory (char *experimentName);
    void CopyFilesForReportGeneration (char *experimentName, SeqListClass &my_keys);
    void MakeNewTmpWellsFile( char *experimentName);
    void CleanupTmpWellsFile(bool USE_RAWWELLS);
    void CopyTmpWellFileToPermanent(bool USE_RAWWELLS, char *experimentName);
    void CopyBasecallerOutput(char *dirname);
    ~SystemContext();
};

void CreateResultsFolder(char *experimentName);
// how exactly do these know the directory context???
void  ClearStaleWellsFile (void);
void  ClearStaleSFFFiles (void);

#endif // SYSTEMCONTEXT_H