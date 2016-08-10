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
#include "OptBase.h"

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
    std::string analysisLocation; // full path in results_folder / terminated

    char wellsFileName[MAX_PATH_LENGTH];
    char tmpWellsFile[MAX_PATH_LENGTH];
    char runId[6];
    char wellsFilePath[MAX_PATH_LENGTH];
	std::string wellStatFile;
	std::string stackDumpFile;
    char *results_folder;
    bool NO_SUBDIR; // when set to true, no experiment subdirectory is created for output files.
    bool LOCAL_WELLS_FILE;
    std::string wellsFormat;
    std::string explog_path;
    std::string explog_final_path;

    bool well_convert;
    float well_upper;
    float well_lower;
    int wells_save_queue_size;
    bool wells_save_number_copies;
    bool wells_convert_with_copies;
    std::vector<int> region_list;

    //const char *GetResultsFolder();    
    char *GetResultsFolder() const;    
    char *experimentDir(char *rawdataDir, char *dirOut);
    void DefaultSystemContext();
    void GenerateContext();
    void FindExpLogPath();
    void WaitForExpLogFinalPath();
    void SetUpAnalysisLocation();
    void MakeSymbolicLinkToOldDirectory (char *experimentName);
    void CopyFilesForReportGeneration (char *experimentName, SeqListClass &my_keys);
    void MakeNewTmpWellsFile( char *experimentName);

    // check explog_final.txt if there are DataCollect exclude regions
    bool CheckDatacollectExcludeRegions(int beginX, int endX, int chipSizeX, int beginY, int endY, int chipSizeY);

    void CleanupTmpWellsFile();
    void CopyTmpWellFileToPermanent( char *experimentName);

    // what does this object need from the command line?
	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);

    ~SystemContext();
};


// how exactly do these know the directory context???
void  ClearStaleWellsFile (void);

#endif // SYSTEMCONTEXT_H
