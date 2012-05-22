/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <iostream>
#include <assert.h>

#include "SystemContext.h"
#include "dirent.h"

using namespace std;

void SystemContext::DefaultSystemContext()
{
  dat_source_directory = NULL;
  wells_output_directory = NULL;
  basecaller_output_directory = NULL;

  strcpy (runId, "");

  sprintf (wellsFileName, "1.wells");
  strcpy (tmpWellsFile, "");
  LOCAL_WELLS_FILE = true;
  strcpy (wellsFilePath, "");
  wellStatFile=NULL;
  wellsFormat = "hdf5";
  NO_SUBDIR = 0;  // when set to true, no experiment subdirectory is created for output files.
}

void SystemContext::CopyBasecallerOutput (char *dirname)
{
  if (basecaller_output_directory)
    free (basecaller_output_directory);
  basecaller_output_directory = strdup (dirname);
}

void SystemContext::GenerateContext (int from_wells)
{
  if (!dat_source_directory)
  {
    dat_source_directory = (char *) malloc (2);
    snprintf (dat_source_directory, 1, ".");  // assume current directory if not provided as an argument
  }

  // Test for a valid data source directory
  // Exception: if this is a re-analysis from wells file, then we can skip this test.
  if (isDir (dat_source_directory) == false && (from_wells == 0))
  {
    fprintf (stderr, "'%s' is not a directory.  Exiting.\n", dat_source_directory);
    exit (EXIT_FAILURE);
  }

  // standard output directory
  if (!wells_output_directory)
  {
    experimentName = (char*) malloc (3);
    strcpy (experimentName, "./");
  }
  else   // --output-dir specified, so wells_output_directory an input arg
  {
    if (NO_SUBDIR)   // --no-subdir specified
    {
      // wells_output_directory is a directory spec.
      // make fully qualified to avoid SetUpAnalysis munging
      if (strlen (wells_output_directory) == 1)
        assert (wells_output_directory[0] != '/');   // root not allowed
      char *tmpPath = strdup (wells_output_directory);
      char *real_path = realpath (dirname (tmpPath), NULL);
      char *tmpBase = strdup (wells_output_directory);
      char *base_name = basename (tmpBase);
      int strSz = strlen (real_path) + strlen (base_name) + 2;
      experimentName = (char *) malloc (sizeof (char) * strSz);
      snprintf (experimentName,strSz,"%s/%s",real_path,base_name);
      free (tmpPath);
      free (tmpBase);
      free (real_path);
    }
    else   // put wells_output_directory+time_stamp in dat_source_directory
    {
      experimentName = experimentDir (dat_source_directory, wells_output_directory);
    }
  }

  if (!basecaller_output_directory)
  {
    basecaller_output_directory = strdup (experimentName); // why is this duplicated?
  }

}

SystemContext::~SystemContext()
{
  if (experimentName)
    free (experimentName);
  if (wells_output_directory)
    free (wells_output_directory);
  if (dat_source_directory)
    free (dat_source_directory);
  if (basecaller_output_directory)
    free (basecaller_output_directory);
}


// utility function
void SystemContext::MakeSymbolicLinkToOldDirectory (char *experimentName)
{
// Create symbolic link to bfmask.bin and 1.wells in new subdirectory: links are for disc space usage reasons
  char *oldpath = NULL;
  int sz = strlen (wellsFilePath) + strlen (wellsFileName) + 2;
  oldpath = (char *) malloc (sz);
  snprintf (oldpath, sz, "%s/%s", wellsFilePath, wellsFileName);
  char *fullPath = realpath (oldpath, NULL);

  char *newpath = NULL;
  sz = strlen (experimentName) + strlen (wellsFileName) + 2;
  newpath = (char *) malloc (sz);
  snprintf (newpath, sz, "%s/%s", experimentName, wellsFileName);

  int ret = symlink (fullPath, newpath);
  if (ret)
  {
    perror (oldpath);
  }
  free (oldpath);
  free (newpath);
  free (fullPath);
}



void SystemContext::MakeNewTmpWellsFile (char *experimentName)
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
      strcpy (wellsFilePath, experimentName);
    }
  }
}


// fill the new directory with files needed for report generation
void SystemContext::CopyFilesForReportGeneration (char *experimentName, SeqListClass &my_keys)
{
//--- Copy files needed for report generation ---
//--- Copy bfmask.stats ---
  int sz;
  char *newpath = NULL;
  char *oldpath = NULL;
  sz = strlen (wellsFilePath) + strlen ("bfmask.stats") + 2;
  oldpath = (char *) malloc (sz);
  snprintf (oldpath, sz, "%s/%s", wellsFilePath, "bfmask.stats");
  sz = strlen (experimentName) + strlen ("bfmask.stats") + 2;
  newpath = (char *) malloc (sz);
  snprintf (newpath, sz, "%s/%s", experimentName, "bfmask.stats");
  fprintf (stderr, "%s\n%s\n", oldpath, newpath);
  CopyFile (oldpath, newpath);
  free (oldpath);
  free (newpath);
//--- Copy avgNukeTrace_ATCG.txt and avgNukeTrace_TCAG.txt
//@TODO:  Is this really compatible with 3 keys?
  for (int q = 0; q < my_keys.numSeqListItems; q++)
  {
    char *filename;
    filename = (char *) malloc (strlen ("avgNukeTrace_") + strlen (
                                  my_keys.seqList[q].seq) + 5);
    sprintf (filename, "avgNukeTrace_%s.txt", my_keys.seqList[q].seq);

    sz = strlen (wellsFilePath) + strlen (filename) + 2;
    oldpath = (char *) malloc (sz);
    snprintf (oldpath, sz, "%s/%s", wellsFilePath, filename);

    sz = strlen (experimentName) + strlen (filename) + 2;
    newpath = (char *) malloc (sz);
    snprintf (newpath, sz, "%s/%s", experimentName, filename);

    CopyFile (oldpath, newpath);
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


void SystemContext::SetUpAnalysisLocation (char *experimentName, std::string &analysisLocation)
{
  // cout << "SystemContext::SetUpAnalysisLocation... experimentName=" << experimentName << endl;
  // 1. output analysisLocation
  char *tmpPath = strdup (experimentName);
  char *tmpStr = realpath (dirname (tmpPath), NULL); // side-effect of dirname(): changes tmpPath, don't use experimentName directly here!!
  string realDir (tmpStr);
  string expName (experimentName); // use the tmp expName to get the analysisPath
  // remove the starting "." or "./" in expName
  if (expName.substr (0,2).compare ("./") == 0)
    expName.replace (0,2,""); // remove "./"
  else if (expName.substr (0,1).compare (".") == 0)
    expName.replace (0,1,""); // remove "."

  // check to see if expName contains the realPath already
  //if (wells_output_directory) expName = analPath;   // overwrite experimentName
  size_t found = expName.find (realDir);
  string analysisPath = (found!=string::npos) ? expName : realDir + expName; // expName used only when it does not contain the realDir
  analysisLocation = analysisPath;

  // 2. output runId (member variable)
  char *analysisDir = NO_SUBDIR ? strdup (tmpStr) : strdup (analysisPath.c_str());
  char *bName = basename(analysisDir);
  ion_run_to_readname (runId, bName, strlen (bName)); // Create a run identifier from output results directory string
  cout << "SystemContext::SetUpAnalysisLocation... experimentName=" << experimentName << endl;
  cout << "SystemContext::SetUpAnalysisLocation... tmpStr        =" << tmpStr << endl;
  cout << "SystemContext::SetUpAnalysisLocation... realPath      =" << realDir << endl;
  cout << "SystemContext::SetUpAnalysisLocation... expName       =" << expName << endl;
  cout << "SystemContext::SetUpAnalysisLocation... analysisDir   =" << analysisDir << endl;
  cout << "SystemContext::SetUpAnalysisLocation... analysisPath  =" << analysisPath << endl;
  cout << "SystemContext::SetUpAnalysisLocation... baseName      =" << bName << endl;
  cout << "SystemContext::SetUpAnalysisLocation... runId         =" << runId << endl << endl;

  /*
    char *analysisPath = (char *) malloc (strlen (tmpStr) + strlen (experimentName) + 2);
    sprintf (analysisPath, "%s/%s", tmpStr, experimentName);
    fprintf (stdout, "Analysis results = %s\n\n", analysisPath);

    char *analysisDir = NO_SUBDIR ? strdup(basename(tmpStr)) : strdup(basename(analysisPath));
    ion_run_to_readname (runId, analysisDir, strlen (analysisDir)); // Create a run identifier from output results directory string
    analysisLocation = analysisPath;

    cout << "SystemContext::SetUpAnalysisLocation... tmpStr=" << tmpStr << ", experimentName=" << experimentName << ", analysisDir=" << analysisDir <<", analysisPath=" << analysisPath << endl;
  free (analysisPath);
  */
  free (tmpPath);
  free (tmpStr);
  free (analysisDir);
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

/*
 *  Remove temporary 1.wells files leftover from previous Analysis
 */
void  ClearStaleSFFFiles (void)
{
  DIR *dirfd;
  struct dirent *dirent;
  const char *files[] = {"_rawlib.sff","_rawtf.sff"};
  for (int i = 0;i < 2; i++)
  {
    dirfd = opendir ("/tmp");
    while ( (dirent = readdir (dirfd)) != NULL)
    {
      if (! (strstr (dirent->d_name, files[i])))
        continue;

      int pid;
      DIR *tmpdir = NULL;
      char name[MAX_PATH_LENGTH];
      char trash[MAX_PATH_LENGTH];
      sscanf (dirent->d_name,"%d_%s", &pid, trash);
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
}


void CreateResultsFolder (char *experimentName)
{
  // Create results folder
  if (mkdir (experimentName, 0777))
  {
    if (errno == EEXIST)
    {
      //already exists? well okay...
    }
    else
    {
      perror (experimentName);
      exit (EXIT_FAILURE);
    }
  }
}

void SystemContext::CleanupTmpWellsFile (bool USE_RAWWELLS)
{
  //Cleanup
  //Copy wells file from temporary, local file to permanent; remove temp file
  //Copy temp wells file moved to pre-cafie code.
  if (LOCAL_WELLS_FILE && !USE_RAWWELLS)
  {
    unlink (tmpWellsFile);
  }
}

void SystemContext::CopyTmpWellFileToPermanent (bool USE_RAWWELLS, char *experimentName)
{
  // defaults moved here because never changed

  static char *wellfileIndex = "1";
  static char *wellfileExt = "wells";

  if (LOCAL_WELLS_FILE && !USE_RAWWELLS)
  {
    char wellFileName[MAX_PATH_LENGTH];
    sprintf (wellFileName, "%s/%s.%s", experimentName, wellfileIndex, wellfileExt);
    CopyFile (tmpWellsFile, wellFileName);
  }
}
