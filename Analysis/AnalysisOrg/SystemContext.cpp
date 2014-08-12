/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <iostream>
#include <assert.h>

#include "SystemContext.h"
#include "dirent.h"
#include "HandleExpLog.h"
#include "IonErr.h"

using namespace std;


void SystemContext::DefaultSystemContext()
{
  dat_source_directory = NULL;
  wells_output_directory = NULL;
  results_folder = NULL;
  analysisLocation = "";

  strcpy (runId, "");

  sprintf (wellsFileName, "1.wells");
  strcpy (tmpWellsFile, "");
  LOCAL_WELLS_FILE = true;
  strcpy (wellsFilePath, "");
  wellStatFile="";
  stackDumpFile="";
  //wellsFormat = "hdf5";
  wellsFormat.assign("hdf5");
  NO_SUBDIR = false;  // when set to true, no experiment subdirectory is created for output files.

  explog_path = "";
}

//const char *SystemContext::GetResultsFolder()
char *SystemContext::GetResultsFolder() const
{
  return(results_folder);
}

void SystemContext::GenerateContext ()
{
  if (!dat_source_directory)
  {
    dat_source_directory = (char *) malloc (2);
    snprintf (dat_source_directory, 1, ".");  // assume current directory if not provided as an argument
  }

  // Test for a valid data source directory
  if (isDir (dat_source_directory) == false )
  {
    fprintf (stderr, "'%s' is not a directory.  Exiting.\n", dat_source_directory);
    exit (EXIT_FAILURE);
  }

  // standard output directory
  if (!wells_output_directory)
  {
    if (results_folder) { free (results_folder); }
    results_folder = (char*) malloc (3);
    strcpy (results_folder, "./");
  }
  else   // --output-dir specified, so wells_output_directory an input arg
  {
	  cout << "wells_output_directory = " << wells_output_directory << endl;
    if (NO_SUBDIR)   // --no-subdir specified
    {
      // wells_output_directory is a directory spec.
      // make fully qualified to avoid SetUpAnalysis munging
      if (strlen (wells_output_directory) == 1)
        assert (wells_output_directory[0] != '/');   // root not allowed
      char *tmpPath = strdup (wells_output_directory);
      char *real_path = realpath (dirname (tmpPath), NULL);
      if (real_path == NULL){
        std::string ss = tmpPath;  // dirname overwrites tmpPath
        ss = ss + ": directory not found";
        ION_ASSERT ((real_path != NULL), ss.c_str());
      }
      char *tmpBase = strdup (wells_output_directory);
      char *base_name = basename (tmpBase);
      int strSz = strlen (real_path) + strlen (base_name) + 2;
      if (results_folder) { free (results_folder); }
      results_folder = (char *) malloc (sizeof (char) * strSz);
      snprintf (results_folder,strSz,"%s/%s",real_path,base_name);
      free (tmpPath);
      free (tmpBase);
      free (real_path);
    }
    else   // put wells_output_directory+time_stamp in dat_source_directory
    {
      if (results_folder) { free (results_folder); }
      results_folder = experimentDir (dat_source_directory, wells_output_directory);
    }
  }

}

SystemContext::~SystemContext()
{
  if (results_folder){
    free (results_folder);
    results_folder = NULL;
  }
  if (wells_output_directory){
    free (wells_output_directory);
    wells_output_directory = NULL;
  }
  if (dat_source_directory)
    free (dat_source_directory);
  /*if (explog_path)
    free (explog_path);*/
}


// utility function
void SystemContext::MakeSymbolicLinkToOldDirectory (char *results_folder)
{
  // Create symbolic link to bfmask.bin and 1.wells in new subdirectory: links are for disc space usage reasons
  char *oldpath = NULL;
  int sz = strlen (wellsFilePath) + strlen (wellsFileName) + 2;
  oldpath = (char *) malloc (sz);
  snprintf (oldpath, sz, "%s/%s", wellsFilePath, wellsFileName);
  char *fullPath = realpath (oldpath, NULL);

  char *newpath = NULL;
  sz = strlen (results_folder) + strlen (wellsFileName) + 2;
  newpath = (char *) malloc (sz);
  snprintf (newpath, sz, "%s/%s", results_folder, wellsFileName);

  int ret = symlink (fullPath, newpath);
  if (ret)
  {
    perror (oldpath);
  }
  free (oldpath);
  free (newpath);
  free (fullPath);
}



void SystemContext::MakeNewTmpWellsFile (char *results_folder)
{
  if (wellsFilePath[0] == '\0')
  {
    if (LOCAL_WELLS_FILE)
    {
      char fTemplate[256] = { 0 };
      //Utils:ClearStaleWellsFile() is sensitive to temp well filename format
      sprintf (fTemplate, "/tmp/well_%d_XXXXXX", getpid());
      int tmpFH = mkstemp (fTemplate);
      if (tmpFH == 0)
        exit (EXIT_FAILURE);
      close (tmpFH);

      strcpy (tmpWellsFile, fTemplate);
      strcpy (wellsFilePath, "/tmp");
      strcpy (wellsFileName, basename (fTemplate));
    }
    else
    {
      strcpy (wellsFilePath, results_folder);
    }
  }
  printf("wells_file_path: %s\n", wellsFilePath);
  printf("wells_file_name: %s\n", wellsFileName);
}


// fill the new directory with files needed for report generation
void SystemContext::CopyFilesForReportGeneration (char *results_folder, SeqListClass &my_keys)
{
  //--- Copy files needed for report generation ---
  //--- Copy bfmask.stats ---
  int sz;
  char *newpath = NULL;
  char *oldpath = NULL;
  sz = strlen (wellsFilePath) + strlen ("bfmask.stats") + 2;
  oldpath = (char *) malloc (sz);
  snprintf (oldpath, sz, "%s/%s", wellsFilePath, "bfmask.stats");
  sz = strlen (results_folder) + strlen ("bfmask.stats") + 2;
  newpath = (char *) malloc (sz);
  snprintf (newpath, sz, "%s/%s", results_folder, "bfmask.stats");
  fprintf (stderr, "%s\n%s\n", oldpath, newpath);
  if( CopyFile (oldpath, newpath) ) ExitCode::UpdateExitCode(EXIT_FAILURE);
  free (oldpath);
  free (newpath);
  //--- Copy avgNukeTrace_ATCG.txt and avgNukeTrace_TCAG.txt
  //@TODO:  Is this really compatible with 3 keys?
  for (int q = 0; q < my_keys.numSeqListItems; q++)
  {
    char *filename;
    filename = (char *) malloc (strlen ("avgNukeTrace_") + strlen (
                                  my_keys.seqList[q].seq.c_str()) + 5);
    sprintf (filename, "avgNukeTrace_%s.txt", my_keys.seqList[q].seq.c_str());

    sz = strlen (wellsFilePath) + strlen (filename) + 2;
    oldpath = (char *) malloc (sz);
    snprintf (oldpath, sz, "%s/%s", wellsFilePath, filename);

    sz = strlen (results_folder) + strlen (filename) + 2;
    newpath = (char *) malloc (sz);
    snprintf (newpath, sz, "%s/%s",results_folder, filename);

    if( CopyFile (oldpath, newpath) ) ExitCode::UpdateExitCode(EXIT_FAILURE);
    free (oldpath);
    free (newpath);
    free (filename);
  }
}


//
//  Create a name for the results of the analysis
//  Use the raw data directory name.  If it is not in standard format, use it in its entirety
//  Raw dir names are R_YYYY_MM_DD_hh_mm_ss_XXX_description
//  Results directory ("experiment" directory) will be 'description'_username_YY_MM_DD_seconds-in-day
//
char *SystemContext::experimentDir (char *rawdataDir, char *dirOut)
{
  char *expDir = NULL;
  char *timeStamp = NULL;
  char *sPtr = NULL;
  time_t now;
  struct tm  *tm = NULL;

  // Strip a trailing slash
  if (dirOut[strlen (dirOut)-1] == '/')
    dirOut[strlen (dirOut)-1] = '\0';

  //  Another algorithm counts forward through the date portion 6 underscores
  sPtr = rawdataDir;
  for (int i = 0; i < 7; i++)
  {
    sPtr = strchr (sPtr, '_');
    if (!sPtr)
    {
      sPtr = "analysis";
      break;
    }
    sPtr++;
  }
  if (sPtr[strlen (sPtr)-1] == '/')   // Strip a trailing slash too
    sPtr[strlen (sPtr)-1] = '\0';

  // Generate a timestamp string
  time (&now);
  tm = localtime (&now);
  timeStamp = (char *) malloc (sizeof (char) * 18);
  snprintf (timeStamp, 18, "_%d_%02d_%02d_%d",1900 + tm->tm_year, tm->tm_mon+1, tm->tm_mday, 3600 * tm->tm_hour + 60 * tm->tm_min + tm->tm_sec);

  int strSize = strlen (dirOut) + strlen (timeStamp) + strlen (sPtr) +2;
  expDir = (char *) malloc (sizeof (char) * strSize);
  if (expDir != NULL)
    snprintf (expDir, strSize, "%s/%s%s", dirOut,sPtr,timeStamp);

  free (timeStamp);

  cout << "SystemContext::experimentDir... dat_source_directory=" << dat_source_directory << ", wells_output_directory=" << wells_output_directory << ", expDir=" << expDir << endl;
  return (expDir);
}


void SystemContext::SetUpAnalysisLocation()
{
  char *path = strdup(results_folder);

  //get full path
  char *real_path = realpath (path, NULL);
  if (real_path == NULL) {
    cout << "Couldn't set up real output path from " << path << ". Using working directory." << endl;
    real_path = realpath("./", NULL);
  }

  analysisLocation = string(real_path);
  // file locations
  if (!analysisLocation.empty() && *analysisLocation.rbegin() != '/')
    analysisLocation = analysisLocation+"/";

  char *bName = basename(real_path);
  ion_run_to_readname (runId, bName, strlen (bName)); // Create a run identifier from output results directory string
  cout << "SystemContext::SetUpAnalysisLocation... experimentName=" << results_folder << endl;
  cout << "SystemContext::SetUpAnalysisLocation... analysisLocation  =" << analysisLocation << endl << endl;
  cout << "SystemContext::SetUpAnalysisLocation... baseName      =" << bName << endl;
  cout << "SystemContext::SetUpAnalysisLocation... runId         =" << runId << endl;

  free(path);
  free(real_path);
}

// make sure we have explog file if not set cmd-line
void SystemContext::FindExpLogPath()
{
  if (explog_path.length() == 0){
    explog_path = MakeExpLogPathFromDatDir(dat_source_directory);
    if (explog_path.length() == 0)
    {
      fprintf (stderr, "Unable to find explog file.  Exiting.\n");
      exit (EXIT_FAILURE);
    }
  }
}

void SystemContext::CleanupTmpWellsFile ()
{
  //Cleanup
  //Copy wells file from temporary, local file to permanent; remove temp file
  //Copy temp wells file moved to pre-cafie code.
  if (LOCAL_WELLS_FILE)
  {
    unlink (tmpWellsFile);
  }
}

void SystemContext::CopyTmpWellFileToPermanent ( char *results_folder)
{
  // defaults moved here because never changed

  static char *wellfileIndex = "1";
  static char *wellfileExt = "wells";

  if (LOCAL_WELLS_FILE)
  {
    char wellFileName[MAX_PATH_LENGTH];
    sprintf (wellFileName, "%s/%s.%s", results_folder, wellfileIndex, wellfileExt);
    if( CopyFile (tmpWellsFile, wellFileName) )
      ExitCode::UpdateExitCode(EXIT_FAILURE);
  }
}


/*
 *  Remove temporary 1.wells files leftover from previous Analysis
 */
void  ClearStaleWellsFile (void)
{
  DIR *dirfd;
  struct dirent *dirent;
  dirfd = opendir ("/tmp");
  while ( (dirent = readdir (dirfd)) != NULL)
  {
    if (! (strstr (dirent->d_name, "well_")))
      continue;

    int pid;
    DIR *tmpdir = NULL;
    char name[MAX_PATH_LENGTH];
    char trash[MAX_PATH_LENGTH];
    sscanf (dirent->d_name,"well_%d_%s", &pid, trash);
    sprintf (name, "/proc/%d", pid);
    if ( (tmpdir = opendir (name)))
    {
      closedir (tmpdir);
    }
    else
    {
      char creamEntry[MAX_PATH_LENGTH];
      sprintf (creamEntry, "/tmp/%s", dirent->d_name);
      unlink (creamEntry);
    }
  }
  closedir (dirfd);
}

void SystemContext::PrintHelp()
{
	printf ("     SystemContext\n");
    printf ("     --no-subdir             BOOL              no subdir [false]\n");
    printf ("     --local-wells-file      BOOL              use local wells file [false]\n");
    printf ("     --well-stat-file        FILE              well stat file name []\n");
    printf ("     --stack-dump-file       FILE              stack dump file name []\n");
	printf ("     --wells-format          STRING            wells format [hdf5]\n");
    printf ("     --output-dir            DIRECTORY         wells output directory []\n");
    printf ("     --explog-path           DIRECTORY         explog output directory []\n");
    printf ("     --dat-source-directory  DIRECTORY         dat source input directory, if there is no such option the last argument of command line must be dat source input directory []\n");
    printf ("\n");
}

void SystemContext::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	LOCAL_WELLS_FILE = RetrieveParameterBool(opts, json_params, '-', "local-wells-file", false);
	wellStatFile = RetrieveParameterString(opts, json_params, '-', "well-stat-file", "");
	stackDumpFile = RetrieveParameterString(opts, json_params, '-', "stack-dump-file", "");
	wellsFormat = RetrieveParameterString(opts, json_params, '-', "wells-format", "hdf5");
	string s0 = RetrieveParameterString(opts, json_params, '-', "output-dir", "");
	wells_output_directory = strdup(s0.c_str());
	explog_path = RetrieveParameterString(opts, json_params, '-', "explog-path", "");
	NO_SUBDIR = RetrieveParameterBool(opts, json_params, '-', "no-subdir", true);
	//jz moved from CommandLineOpts::PickUpSourceDirectory
	string s = RetrieveParameterString(opts, json_params, '-', "dat-source-directory", "");
	if(s.length() > 0)
	{
		if (dat_source_directory)
		{
			free (dat_source_directory);
			dat_source_directory = NULL;
		}

		dat_source_directory = (char *) malloc (s.length() + 1);
		sprintf (dat_source_directory, "%s", s.c_str());  
	}
}