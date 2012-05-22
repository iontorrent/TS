/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <sys/time.h>
#include "Image.h"
#include "SpecialDataTypes.h"
#include "LinuxCompat.h"

// Transition states used in Analysis main function for progress tracking
#define WELL_TO_IMAGE 1
#define IMAGE_TO_SIGNAL 2
#define MAX_PATH_LENGTH 2048
// Marco to determine size of static array on stack
#define ArraySize(x) (sizeof(x)/sizeof((x)[0]))

char    *GetExpLogParameter (const char *filename, const char *paramName);
void GetExpLogParameters (const char *filename, const char *paramName,
                          std::vector<std::string> &values);
char * getExpLogPath (const char *dir);
bool    isDir (const char *path);
bool    isFile (const char *path);
bool    IsValid (const double *vals, int numVals);
bool    isNumeric (char const* input, int numberBase = 10);
int     HasWashFlow (char *datapath);
short   GetPinHigh();
short   GetPinLow();
bool    validIn (char *inStr, long *value);
int     numCores ();
void  ToUpper (char *str);
void  ToLower (char *str);
double  ToDouble (char const* str);
void  get_exe_name (char * buffer);
char  *GetIonConfigFile (const char filename[]);
bool  CopyFile (char *, char *);

int   GetCycles (char *dir);
int   GetTotalFlows (char *dir);
char  *GetChipId (const char *dir);
int   GetNumLines (char *filename);
char  *GetProcessParam (const char *, const char *);
void  defineSubRegion (int rows, int cols, int runIndex, int regionIdx, Region *cropRegions);
// Break a path to an indiviual file into the usual directory and file used for Analysis and libraries
void FillInDirName (const std::string &path, std::string &dir, std::string &file);
void  init_salute();
bool  updateProgress (int transition);
char  *GetPGMFlowOrder (char *path);

int seqToFlow (const char *seq, int seqLen, int *ionogram, int ionogramLen, char *flowOrder, int flowOrderLen);
void flowToSeq (std::string &seq, hpLen_vec_t &hpLen, std::string &flowOrder);
void GetChipDim (const char *type, int dims[2], const char *);
std::string GetMemUsage();
void MemoryUsage (const std::string &s);
void MemUsage (const std::string &s);
int totalMemOnTorrentServer();

//string utils
int     count_char (std::string s, char c);
std::string get_file_extension (const std::string& s);
void    split (const std::string& s, char c, std::vector<std::string>& v);
void uintVectorToString (std::vector<unsigned int> &v, std::string &s, std::string &nullStr, char delim);
/** Trim off any whitespace from either end of string. */
void TrimString (std::string &str);
bool isInternalServer();

template <class T>
std::vector<T> char2Vec (const char *s, char delim='*')
{
  std::stringstream str (std::stringstream::in | std::stringstream::out);
  std::vector<T> vec;
  while (s != NULL && *s != '\0')
  {
    while (*s != '\0')
    {
      str.put (*s);
      s++;
      if (*s == delim || delim == '*')
      {
        if (*s == delim)
        {
          s++;
        }
        break;
      }
    }
    str << " ";
    T x;
    str >> x;
    vec.push_back (x);
  }
  return vec;
}
/** Slow - only use for error messages etc. */
template <class T>
std::string ToStr (T t)
{
  std::ostringstream oss;
  oss << t;
  return oss.str();
}



//
// Utility timer class
//

class Timer
{
  public:
    Timer()
    {
      restart();
    }
    void restart()
    {
      gettimeofday (&start_time, NULL);
    }
    double elapsed()
    {
      gettimeofday (&end_time, NULL);
      return (end_time.tv_sec - start_time.tv_sec + static_cast<double> (end_time.tv_usec - start_time.tv_usec) / (1000000.0));
    }
  private:
    timeval start_time;
    timeval end_time;
};

class ClockTimer
{

  public:
    ClockTimer()
    {
      StartTimer();
    }
    void StartTimer()
    {
      gettimeofday (&st, NULL);
    }

    size_t GetSeconds()
    {
      struct timeval et;
      gettimeofday (&et, NULL);
      return ( ( (et.tv_sec*1.0e6+et.tv_usec) - (st.tv_sec * 1.0e6 + st.tv_usec)) /1.0e6);
    }

    size_t GetMicroSec()
    {
      struct timeval et;
      gettimeofday (&et, NULL);
      return ( ( (et.tv_sec*1.0e6+et.tv_usec) - (st.tv_sec * 1.0e6 + st.tv_usec)));
    }

    size_t GetMinutes()
    {
      return GetSeconds() / 60;
    }

    void PrintSeconds (std::ostream &out, const std::string &prefix)
    {
      out << prefix << " " << GetSeconds() << " seconds." << std::endl;
    }
    void PrintMicroSeconds (std::ostream &out, const std::string &prefix)
    {
      out << prefix << " " << GetMicroSec() / 1e6 << " seconds." << std::endl;
    }

    void PrintMilliSeconds (std::ostream &out, const std::string &prefix)
    {
      out << prefix << " " << GetMicroSec() / 1e3 << " milli seconds." << std::endl;
    }

  private:
    struct timeval st;
};


#endif // UTILS_H
