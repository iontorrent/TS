/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Utils.h"
#ifndef ALIGNSTATS_IGNORE
#include "IonVersion.h"
#endif
#include <cstdio>
#include <sys/stat.h>
#include <libgen.h>
#include <limits.h>
#include <errno.h>
#include <cstring>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h> // getpid
#include <cmath>   // isnan

using namespace std;

char *readline (FILE *fp)
{
  char* line = (char*) calloc (1, sizeof (char));
  char c;
  int len = 0;
  while ( (c = fgetc (fp)) != EOF && c != '\n')
  {
    line = (char *) realloc (line, sizeof (char) * (len + 2));
    line[len++] = c;
    line[len] = '\0';
  }
  return (line);
}

//void CreateResultsFolder (char *experimentName)
void CreateResultsFolder (const char *experimentName)
{
  // Create results folder
  if (mkdir (experimentName, 0777))
  {
    if (errno == EEXIST)
    {
      //need to check whether it is directory indeed
      if(!isDir(experimentName)){
          printf("%s is not a directory!\n", experimentName);
          exit(EXIT_FAILURE);
      }
    }
    else
    {
      perror (experimentName);
      exit (EXIT_FAILURE);
    }
  }
}

//
//  Tests a string for containing a valid filesystem path
//
bool isDir (const char *path)
{
  struct stat x;
  if (stat (path, &x) != 0)
    return false;
  return (S_ISDIR (x.st_mode) ? true:false);
}
//
//  Tests a string for containing a valid filesystem file
//
bool isFile (const char *path)
{
  struct stat x;
  if (stat (path, &x) != 0)
    return false;
  return (S_ISREG (x.st_mode) ? true:false);
}


//
//Converts initial portion of a string to a long integer, with error checking
//
bool validIn (char *inStr, long *value)
{
  if (inStr == NULL)
  {
    fprintf (stderr, "input string is null\n");
    return EXIT_FAILURE;
  }

  char *endPtr = NULL;

  errno = 0;
  *value = strtol (inStr, &endPtr, 10);
  if ( (errno == ERANGE && (*value == LONG_MAX || *value == LONG_MIN))
       || (errno != 0 && *value == 0))
  {
    perror ("strtol");
    return (EXIT_FAILURE);
  }

  if (endPtr == inStr)
  {
    fprintf (stderr, "No digits were found in %s\n", inStr);
    return (EXIT_FAILURE);
  }
  //fprintf (stdout, "Converted to %ld\n", *value);
  return EXIT_SUCCESS;
}

// determine if a string is a valid number (of any type; int, float, etc...)
bool isNumeric (char const* str, int numberBase)
{
  std::istringstream iss (str);

  if (numberBase == 10)
  {
    double doubleSink;
    iss >> doubleSink;
  }
  else if (numberBase == 8 || numberBase == 16)
  {
    int intSink;
    iss >> ( (numberBase == 8) ? oct : hex) >> intSink;
  }
  else
    return false;

  // was any input successfully consumed/converted?
  if (!iss)
    return false;

  // was all the input successfully consumed/converted?
  return (iss.rdbuf()->in_avail() == 0);
}

// convert a string to a floating point value
double ToDouble (char const* str)
{
  std::istringstream iss (str);
  double doubleSink;

  iss >> doubleSink;
  return doubleSink;
}

int numCores ()
{
#if 1
  // returns physical cpu cores
  int return_num_cores = 1;
  int cores = 0;
  int processors = 0;
  int n = 0; //number elements read
  FILE *fp = NULL;

  // if the grep finds nothing, then this returns a NULL fp...

  // Number of cores
  fp = popen ("cat /proc/cpuinfo | grep \"cpu cores\" | uniq  | awk '{ print $4 }'", "r");
  if (fp != NULL)
  {
    n = fscanf (fp, "%d", &cores);
    if (n != 1)
      cores=1;
  }
  pclose (fp);

  // Number of processors
  fp = popen ("cat /proc/cpuinfo | grep \"physical\\ id\" | sort | uniq | wc -l", "r");
  if (fp != NULL)
  {
    n = fscanf (fp, "%d", &processors);
    if (n != 1)
      processors=1;
  }
  pclose (fp);


  if (cores==0 || processors==0)
  {
    fp = popen ("grep \"processor\" /proc/cpuinfo | wc -l","r");
    if (fp != NULL)
    {
      n = fscanf (fp, "%d", &return_num_cores);
      if (n != 1)
        return_num_cores=1;
    }
    pclose (fp);
  }
  else
    return_num_cores = cores * processors;

  /* Hack: Some VMs report 0 cores */
  return_num_cores  = (return_num_cores > 0 ? return_num_cores:1);

  return return_num_cores;
#else
  // returns virtual cpu cores
  return (sysconf (_SC_NPROCESSORS_ONLN));
#endif
}

//
// Convert all chars in a string to upper case
//
void ToUpper (char *in)
{
  for (int i=0;in[i];i++)
    in[i] = toupper (in[i]);
}//
// Convert all chars in a string to lower case
//
void ToLower (char *in)
{
  for (int i=0;in[i];i++)
    in[i] = tolower (in[i]);
}
//
//  Returns path of executable
//  Not portable - uses the proc filesystem
void get_exe_name (char * buffer)
{
  char linkname[64] = {0};
  pid_t pid;
  unsigned long offset = 0;

  pid = getpid();
  snprintf (&linkname[0], sizeof (linkname), "/proc/%i/exe", pid);

  if (readlink (&linkname[0], buffer, PATH_MAX) == -1)
    offset = 0;
  else
  {
    offset = strlen (buffer);
    while (offset && buffer[offset - 1] != '/') --offset;
    if (offset && buffer[offset - 1] == '/') --offset;
  }

  buffer[offset] = 0;
}
//
//  Return fully qualified path to some configuration files
// Search order:
//  $HOME directory
//  $ION_CONFIG directory
//  Relative to executable's directory ../config/
//  Absolute path: /opt/ion/config/
//  Absolute path: /opt/ion/alignTools/
//
char *GetIonConfigFile (const char filename[])
{
  char *string = NULL;

  fprintf (stdout, "# DEBUG: Looking for '%s'\n", filename);
  // Search for TF config file:
  //  Current working directory
  size_t bufsize = 512;
  char buf[bufsize];
  assert (getcwd (buf, bufsize));
  string = (char *) malloc (strlen (filename) + bufsize + 2);
  sprintf (string, "%s/%s", buf, filename);
  if (isFile (string))
  {
    return (string);
  }
  else
  {
    free (string);
    string = NULL;
  }

  // Search for config file in $HOME:
  //  $HOME
  char *HOME = getenv ("HOME");
  if (HOME)
  {
    string = (char *) malloc (strlen (filename) + strlen (HOME) + 2);
    sprintf (string, "%s/%s", HOME, filename);
    if (isFile (string))
    {
      fprintf (stdout, "Found ... %s\n", string);
      return (string);
    }
    else
    {
      free (string);
      string = NULL;
    }
    //free (HOME);
  }

  // Search for config file in $ION_CONFIG:
  //  Installation environment variable
  char *ION_CONFIG = getenv ("ION_CONFIG");
  if (ION_CONFIG)
  {
    string = (char *) malloc (strlen (filename) + strlen (ION_CONFIG) + 2);
    sprintf (string, "%s/%s", ION_CONFIG, filename);
    if (isFile (string))
    {
      fprintf(stdout, "Found ... %s\n", string);
      return (string);
    }
    else
    {
      fprintf(stdout,"%s is not a file \n", string);
      free (string);
      string = NULL;
    }
  }

  // Search for config file:
  //  Last ditch effort: Installation location.  Get location of binary, then up one dir and down into config
  char INSTALL[PATH_MAX] = {0};
  get_exe_name (INSTALL);
  // executable is always in bin so we specifically strip that off.
  char *sPtr = NULL;
  sPtr = strrchr (INSTALL, '/');
  if (sPtr)
    *sPtr = '\0';
  string = (char *) malloc (strlen (filename) + strlen (INSTALL) + strlen ("config") + 3);
  sprintf (string, "%s/config/%s", INSTALL, filename);
  if (isFile (string))
  {
    fprintf(stdout, "Found ... %s\n", string);
    return (string);
  }
  else
  {
    free (string);
    string = NULL;
  }

  // Ultimate last ditch: hardcoded path
  string = (char *) malloc (strlen (filename) + strlen ("/opt/ion/config") + 2);
  sprintf (string, "/opt/ion/config/%s", filename);
  if (isFile (string))
  {
    fprintf(stdout, "Found ... %s\n", string);
    return (string);
  }
  else
  {
    free (string);
    string = NULL;
  }

  // (YALDE): Yet another Ultimate last ditch: hardcoded path
  string = (char *) malloc (strlen (filename) + strlen ("/opt/ion/alignTools") + 2);
  sprintf (string, "/opt/ion/alignTools/%s", filename);
  if (isFile (string))
  {
    fprintf(stdout, "Found ... %s\n", string);
    return (string);
  }
  else
  {
    free (string);
    string = NULL;
  }

  fprintf (stderr, "Cannot find Ion Config file: %s\n", filename);
  return (NULL);
}

//Copy a file
bool CopyFile (char *filefrom, char *fileto)
{
//#define printTime
#ifdef printTime
  time_t startTime;
  time_t endTime;
  time (&startTime);
#endif

  int size = 4096;
  char cmd[size];
  int alloc = snprintf (cmd, size-1, "cp %s %s",filefrom, fileto);
  //int alloc = snprintf (cmd, size-1, "cp %s %s && chmod a+rw %s &",filefrom, fileto, fileto);
  if (alloc < 0 || alloc > size-1)
  {
    fprintf (stderr, "CopyFile could not execute system copy command:\n");
    fprintf (stderr, "Copy file: %s\n", filefrom);
    fprintf (stderr, "To: %s\n", fileto);
    return (1);
  }

  int status;
  status = system (cmd);
  if (WIFEXITED (status))
  {
    if (WEXITSTATUS (status))
    {
      // error encountered
      fprintf (stderr, "From: %s\n", filefrom);
      fprintf (stderr, "To: %s\n", fileto);
      fprintf (stderr, "Command: %s\n", cmd);
      fprintf (stderr, "system copy command returned status %d\n", WEXITSTATUS (status));
      return (1);
    }
  }

  /* When 1.wells get copied, they have permissions of 600 and we like 666 */
  //snprintf (cmd, size-1, "chmod a+rw %s", fileto);
  /* Changing permissions to not allow others write*/
  snprintf (cmd, size-1, "chmod u=rw,g=rw,o=r %s", fileto);
  status = system (cmd);
  if (WIFEXITED (status))
  {
    if (WEXITSTATUS (status))
    {
      // error encountered
      fprintf (stderr, "chmod command returned status %d\n", WEXITSTATUS (status));
      return (1);
    }
  }


#ifdef printTime
  time (&endTime);
  struct stat buf;
  lstat (filefrom, &buf);
  fprintf (stdout, "Copy Time: %0.1lf sec. (%ld bytes)\n", difftime (endTime, startTime), (long int) buf.st_size);
  fprintf (stderr, "Copy file: %s\n", filefrom);
  fprintf (stderr, "To: %s\n", fileto);
#endif

  return 0;
}

int GetNumLines (char *filename)
{
  int cnt = 0;
  FILE *fp = fopen (filename, "rb");
  if (!fp)
  {
    perror (filename);
    return (-1);
  }
  while (!feof (fp))
  {
    if (fgetc (fp) == '\n')
      cnt++;
  }
  fclose (fp);
  return (cnt);
}

void Trim (char *buf)
{
  int len = strlen (buf);
  while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == '\n'))
    len--;
  buf[len] = 0;
}
//
// Opens processParameters.txt file and reads the argument for the given
// keyword
char * GetProcessParam (const char *filePath, const char *pattern)
{
  FILE *fp = NULL;
  char *fileName = NULL;
  char *arg = NULL;
  char *keyword = NULL;
  char *argument = NULL;
  char buf[16384];
  char *sPtr = NULL;

  fileName = (char *) malloc (strlen (filePath) +strlen ("/processParameters.txt") +1);
  sprintf (fileName, "%s/%s", filePath, "processParameters.txt");

  fp = fopen (fileName, "rb");
  if (!fp)
  {
    perror (fileName);
    free (fileName);
    return (NULL);
  }

  free (fileName);

  while (fgets (buf, sizeof (buf), fp))
  {
    Trim (buf);
    if ( (sPtr = strchr (buf, '=')))
    {
      //allocate plenty of space for each component of the entry. and initialize
      keyword  = (char *) calloc (1,strlen (buf));
      argument = (char *) calloc (1,strlen (buf));

      //separate the components at the '=' char, remove whitespace
      char *aPtr = sPtr+1;
      while (isspace (*aPtr)) aPtr++;
      strncpy (argument, aPtr, strlen (buf)-1);
      char *end = aPtr + strlen (aPtr) - 1;
      while (end > aPtr && isspace (*end)) end--;

      *sPtr = '\0';
      strncpy (keyword, buf, strlen (buf)-1);
      end = keyword + strlen (keyword) - 1;
      while (end > keyword && isspace (*end)) end--;

      //select the desired keyword.  note: whitespace would be a problem
      //if we searched for exact match.
      // note that this is a latent bug, as strstr will find the pattern >anywhere< in the line
      // if the keyword doesn't match, that's a problem with >the input file<, not with the parser
      if (strstr (keyword, pattern))
      {
        arg = (char *) malloc (strlen (argument) +1);
        strcpy (arg, argument);
        free (keyword);
        free (argument);
        break;
      }
      free (keyword);
      free (argument);
    }
  }

  fclose (fp);

  return (arg);
}
/*
 * For given width and height chip, and input region index and number of regions, return
 *  Region structure for unique region specified.
 */
#ifndef ALIGNSTATS_IGNORE
void defineSubRegion (int rows, int cols, int runIndex, int regionIdx, Region *cropRegions)
{

  // regionIdx is number of regions to create.
  // runIndex is which region to return.
  switch (regionIdx)
  {
    case 4:
    {
      int xinc = cols/2;
      int yinc = rows/2;
   std::vector<Region> regions(regionIdx);
      int i;
      int x;
      int y;
      for (i = 0, x=0;x<cols;x+=xinc)
      {
        for (y=0;y<rows;y+=yinc)
        {
          regions[i].col = x;
          regions[i].row = y;
          regions[i].w = xinc;
          regions[i].h = yinc;
          if (regions[i].col + regions[i].w > cols) // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
            regions[i].w = cols - regions[i].col; // but better to be safe!
          if (regions[i].row + regions[i].h > rows)
            regions[i].h = rows - regions[i].row;
          i++;
        }
      }

      cropRegions->col = regions[runIndex-1].col;
      cropRegions->row = regions[runIndex-1].row;
      cropRegions->w = regions[runIndex-1].w;
      cropRegions->h = regions[runIndex-1].h;

      break;
    }
    case 9:
      //break;
    case 16:
      //break;
    default:
      fprintf (stderr, "Unsupported region divisor: %d\n", regionIdx);
      break;
  }

}
#endif
bool IsValid (const double *vals, int numVals)
{
  int i;
  for (i=0;i<numVals;i++)
  {
    if (std::isnan (vals[i]))
      return false;
  }
  return true;
}

void FillInDirName (const string &path, string &dir, string &file)
{
  size_t slashPos = path.rfind ('/');
  if (slashPos != string::npos)
  {
    dir = path.substr (0, slashPos);
    file = path.substr (slashPos + 1, (path.length() - (slashPos+1)));
  }
  else
  {
    dir = ".";
    file = path;
  }
}

#ifndef ALIGNSTATS_IGNORE
void init_salute()
{
  char banner[256];
  sprintf (banner, "/usr/bin/figlet -m0 Analysis %s 2>/dev/null", IonVersion::GetVersion().c_str());
  if (system (banner))
  {
    // figlet did not execute;
    fprintf (stdout, "%s\n", IonVersion::GetVersion().c_str());
  }
}
#endif


std::string get_time_iso_string(time_t time)
{
  char time_buffer[1024];
  strftime(time_buffer, 1024, "%Y-%m-%dT%H:%M:%S", localtime(&time));
  return std::string(time_buffer);
}

int count_char (std::string s, char c)
{

  size_t pos = 0;
  int tot = 0;
  string tmp = s;
  while (pos!=string::npos)
  {
    tmp = tmp.substr (pos+1);

    pos = tmp.find (c);
    if (pos != string::npos)
    {
      tot++;
    }

  }
  if (tot > 0)
  {
    return tot;
  }
  else
  {
    return 0;
  }



}


string get_file_extension (const string& s)
{

  size_t i = s.rfind ('.', s.length());
  if (i != string::npos)
  {
    return (s.substr (i+1, s.length() - i));
  }

  return ("");
}


void split (const string& s, char c, vector<string>& v)
{
  v.clear();
  if (s != "") {
    string::size_type i = 0;
    string::size_type j = s.find (c);
    if (j == string::npos)
    {
      v.push_back (s);
    }
    else
    {
      while (j != string::npos)
      {
        v.push_back (s.substr (i, j-i));
        i = ++j;
        j = s.find (c,j);

        if (j == string::npos)
        {
          v.push_back (s.substr (i, s.length()));
        }
      }
    }
  }
}

void uintVectorToString (vector<unsigned int> &v, string &s, string &nullStr, char delim)
{
  if (v.size() > 0)
  {
    std::stringstream converter0;
    converter0 << v[0];
    s = converter0.str();
    for (unsigned int i=1; i<v.size(); i++)
    {
      std::stringstream converter1;
      converter1 << v[i];
      s += delim + converter1.str();
    }
  }
  else
  {
    s = nullStr;
  }
}

int seqToFlow (const char *seq, int seqLen, int *ionogram, int ionogramLen, char *flowOrder, int flowOrderLen)
{
  int flows = 0;
  int bases = 0;
  while (flows < ionogramLen && bases < seqLen)
  {
    ionogram[flows] = 0;
    while ( (bases < seqLen) && (flowOrder[flows%flowOrderLen] == seq[bases]))
    {
      ionogram[flows]++;
      bases++;
    }
    flows++;
  }
  return flows;
}

#ifndef ALIGNSTATS_IGNORE
void flowToSeq (string &seq, hpLen_vec_t &hpLen, string &flowOrder)
{
  unsigned int cycleLen = flowOrder.size();
  seq.clear();
  if (cycleLen > 0)
  {
    for (unsigned int iFlow=0; iFlow < hpLen.size(); iFlow++)
    {
      char thisNuc = flowOrder[iFlow % cycleLen];
      for (char iNuc=0; iNuc < hpLen[iFlow]; iNuc++)
      {
        seq += thisNuc;
      }
    }
  }
}
//
// Returns pointer to string containing path to explog.txt file
// Can be in given raw data directory, or parent of given directory if its a gridded dataset
//
char * MakeExpLogPathFromDatDir (const char *dir)
{
  //first try the given directory - default behavior for monogrid data
  char filename[] = {"explog.txt"};
  char *filepath = NULL;
  filepath = (char *) malloc (sizeof (char) * (strlen (filename) + strlen (dir) + 2));
  sprintf (filepath, "%s/%s", dir, filename);
  if (isFile (filepath))
    return filepath;
  if (filepath)
    free (filepath);
  filepath = NULL;
  //second try the parent directory
  char *parent = NULL;
  parent = strdup (dir);
  char *parent2 = dirname (parent);
  filepath = (char *) malloc (sizeof (char) * (strlen (filename) + strlen (parent2) + 2));
  sprintf (filepath, "%s/%s", parent2, filename);
  if (parent)
    free (parent);
  //if (parent2)
  //free (parent2);
  
  if (isFile (filepath))
  {
    return filepath;
  }
  // third try:
  char filename_thumbnail[] = {"explog_final.txt"};
  if (filepath) free (filepath);
  filepath = (char *) malloc (sizeof (char) * (strlen (filename_thumbnail) + strlen (dir) + 2));
  sprintf (filepath, "%s/%s", dir, filename_thumbnail);

  if (isFile (filepath))
  {
    return filepath;
  }
  free(filepath);
  return NULL;
}


//
// Returns pointer to string containing path to explog_final.txt file
// Can be in given raw data directory, or parent of given directory if its a gridded dataset
//
char * MakeExpLogFinalPathFromDatDir (const char *dir)
{
  //first try the given directory - default behavior for monogrid data
  char filename[] = {"explog_final.txt"};
  char *filepath = NULL;
  filepath = (char *) malloc (sizeof (char) * (strlen (filename) + strlen (dir) + 2));
  sprintf (filepath, "%s/%s", dir, filename);
  if (isFile (filepath))
    return filepath;
  if (filepath)
    free (filepath);
  filepath = NULL;
  //second try the parent directory
  char *parent = NULL;
  parent = strdup (dir);
  char *parent2 = dirname (parent);
  filepath = (char *) malloc (sizeof (char) * (strlen (filename) + strlen (parent2) + 2));
  sprintf (filepath, "%s/%s", parent2, filename);
  if (parent)
    free (parent);
  if (isFile (filepath))
    return filepath;
  if (filepath)
    free(filepath);
  return NULL;
}


std::string GetMemUsage()
{
  pid_t pid =  getpid();
  string name = "/proc/" + ToStr (pid) + "/statm";
  std::ifstream file;
  file.open (name.c_str() , ifstream::in);
  string line;
  string usage;
  vector<string> words;
  if (getline (file, line))
  {
    split (line,' ',words);
  }
  if (words.size() < 3)
  {
    usage = "unknown";
  }
  else
  {
    size_t virt = atoi (words[0].c_str()) * 4 * 1024 / 1048576;
    size_t resident = atoi (words[1].c_str()) * 4 * 1024 / 1048576;
    usage = "Virtual: " + ToStr (virt) + "MB Resident: " + ToStr (resident) + "MB";
  }
  file.close();
  return usage;
}


void MemoryUsage (const std::string &s)
{
  std::cout << "MEM USAGE: " << s << " - " << GetMemUsage() << std::endl;
}

void MemUsage (const std::string &s)
{
  MemoryUsage (s);
}

void TrimString (std::string &str)
{
  std::string whitespaces (" \t\f\v\n\r");
  size_t found = str.find_last_not_of (whitespaces);
  if (found != std::string::npos)
    str.erase (found+1);
  else
    str.clear();
  found = str.find_first_not_of (whitespaces);
  if (found != std::string::npos)
    str.erase (0,found);
  else
    str.clear();
}

int totalMemOnTorrentServer()
{
  const int totalMem = 48*1024*1024; // defaults to T7500
  FILE *fp = NULL;
  int mem;
  fp = popen ("cat /proc/meminfo | grep \"MemTotal:\" | awk '{ print $2 }'", "r");

  // if the grep finds nothing, then this returns a NULL fp...
  if (fp == NULL)
    return totalMem;

  int n = fscanf (fp, "%d", &mem);
  if (n != 1)
    mem = totalMem;

  pclose (fp);

  return mem;
}


int GetSystemMemInBuffers()
{
  FILE *fp = NULL;
  int mem = 0;
  fp = popen ("cat /proc/meminfo | grep \"Buffers:\" | awk '{ print $2 }'", "r");

  // if the grep finds nothing, then this returns a NULL fp...
  if (fp == NULL)
    return mem;

  int n = fscanf (fp, "%d", &mem);

  if (n != 1)
    mem = 0;

  pclose (fp);

  return mem;
}

int GetCachedSystemMem()
{
  FILE *fp = NULL;
  int mem = 0;
  fp = popen ("cat /proc/meminfo | grep \"Cached:\" | awk '{ print $2 }'", "r");

  // if the grep finds nothing, then this returns a NULL fp...
  if (fp == NULL)
    return mem;

  int n = fscanf (fp, "%d", &mem);

  if (n != 1)
    mem = 0;

  pclose (fp);

  return mem;
}

int GetFreeSystemMem()
{
  FILE *fp = NULL;
  int mem = 0;
  fp = popen ("cat /proc/meminfo | grep \"MemFree:\" | awk '{ print $2 }'", "r");

  // if the grep finds nothing, then this returns a NULL fp...
  if (fp == NULL)
    return mem;

  int n = fscanf (fp, "%d", &mem);
 
  if (n != 1)
    mem = 0;

  pclose (fp);

  return mem;
}


int GetAbsoluteFreeSystemMemoryInKB()
{
  int freeMem = 0;

  freeMem = GetFreeSystemMem() + GetCachedSystemMem() + GetSystemMemInBuffers();

  return freeMem;
}

// -----------------


void expandBaseSymbol(char nuc, std::vector<bool>& nuc_ensemble)
{
  nuc_ensemble.assign(4, false);
  nuc = toupper(nuc);

  switch(nuc) {
      case 'A': nuc_ensemble[0] = true; break;
      case 'C': nuc_ensemble[1] = true; break;
      case 'G': nuc_ensemble[2] = true; break;
      case 'T': nuc_ensemble[3] = true; break;
      case 'U': nuc_ensemble[3] = true; break;

      case 'W': nuc_ensemble[0] = true; nuc_ensemble[3] = true; break;
      case 'S': nuc_ensemble[1] = true; nuc_ensemble[2] = true; break;
      case 'M': nuc_ensemble[0] = true; nuc_ensemble[1] = true; break;
      case 'K': nuc_ensemble[2] = true; nuc_ensemble[3] = true; break;
      case 'R': nuc_ensemble[0] = true; nuc_ensemble[2] = true; break;
      case 'Y': nuc_ensemble[1] = true; nuc_ensemble[3] = true; break;

      case 'B': nuc_ensemble[1] = true; nuc_ensemble[2] = true; nuc_ensemble[3] = true; break;
      case 'D': nuc_ensemble[0] = true; nuc_ensemble[2] = true; nuc_ensemble[3] = true; break;
      case 'H': nuc_ensemble[0] = true; nuc_ensemble[1] = true; nuc_ensemble[3] = true; break;
      case 'I': nuc_ensemble[0] = true; nuc_ensemble[1] = true; nuc_ensemble[3] = true; break;
      case 'V': nuc_ensemble[0] = true; nuc_ensemble[1] = true; nuc_ensemble[2] = true; break;

      case 'N': nuc_ensemble.assign(4, true); break;
  }
}

char contractNucSymbol(const std::vector<bool>& nuc_ensemble)
{
  if (nuc_ensemble.size() != 4)
    return 'Z';

  unsigned int weight = 0;
  for (unsigned int i=0; i<4; ++i)
    if (nuc_ensemble[i])
      ++weight;

  if (weight == 0)
    return 'Z';

  else if (weight==4)
    return 'N';

  else if (weight == 3){
    if (not nuc_ensemble[0])
      return 'B';
    else if (not nuc_ensemble[1])
      return 'D';
    else if (not nuc_ensemble[2])
      return 'H';
    else // if (not nuc_ensemble[3])
      return 'V';
  }

  else if (weight == 2){
    if (nuc_ensemble[0] and nuc_ensemble[3])
      return 'W';
    else if (nuc_ensemble[1] and nuc_ensemble[2])
      return 'S';
    else if (nuc_ensemble[0] and nuc_ensemble[1])
      return 'M';
    else if (nuc_ensemble[2] and nuc_ensemble[3])
      return 'K';
    else if (nuc_ensemble[0] and nuc_ensemble[2])
      return 'R';
    else //if (nuc_ensemble[1] and nuc_ensemble[3])
      return 'Y';
  }

  else if (nuc_ensemble[0])
    return 'A';
  else if (nuc_ensemble[1])
    return 'C';
  else if (nuc_ensemble[2])
    return 'G';
  else // if (nuc_ensemble[3])
    return 'T';
}


bool isBaseMatch(char nuc1, char nuc2)
{
  std::vector<bool> ensemble1, ensemble2;
  expandBaseSymbol(nuc1, ensemble1);
  expandBaseSymbol(nuc2, ensemble2);

  /* / ---- XXX
  cerr << "[";
  for (unsigned int i=0; i<ensemble1.size(); ++i){
    cerr << ensemble1[i];
  }
  cerr << "] [";
  for (unsigned int i=0; i<ensemble2.size(); ++i){
    cerr << ensemble2[i];
  }
  cerr << "] ";
  // ------- XXX //*/


  for (unsigned int i=0; i<ensemble2.size(); ++i){
    if (ensemble1[i] and ensemble2[i])
      return true;
  }
  return false;
}


char getMatchSymbol(char nuc1, char nuc2)
{
  std::vector<bool> ensemble1, ensemble2;
  expandBaseSymbol(nuc1, ensemble1);
  expandBaseSymbol(nuc2, ensemble2);

  for (unsigned int i=0; i<ensemble1.size(); ++i){
    ensemble1[i] = ensemble1[i] and ensemble2[i];
  }

  return contractNucSymbol(ensemble1);
}

#endif
