/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <cstdio>
#include <time.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <sys/time.h>

#ifndef ALIGNSTATS_IGNORE
#include "SpecialDataTypes.h"
#include "LinuxCompat.h"
#endif

// Transition states used in Analysis main function for progress tracking
#define WELL_TO_IMAGE 1
#define IMAGE_TO_SIGNAL 2
#define MAX_PATH_LENGTH 2048

// Marco to determine size of static array on stack
#define ArraySize(x) (sizeof(x)/sizeof((x)[0]))

// Macro to prohibit auto generation of copy constructor and assignment operator
#undef  ION_DISABLE_COPY_ASSIGN
#define ION_DISABLE_COPY_ASSIGN(Class)			\
  Class(const Class &);\
  Class &operator=(const Class &);

#define FREEZ(ptr) { if (*ptr) { free(*ptr); *ptr = NULL; } }

//void CreateResultsFolder(char *experimentName);
void CreateResultsFolder(const char *experimentName);
bool    isDir (const char *path);
bool    isFile (const char *path);
bool    IsValid (const double *vals, int numVals);
bool    isNumeric (char const* input, int numberBase = 10);
bool    validIn (char *inStr, long *value);
int     numCores ();
void  ToUpper (char *str);
void  ToLower (char *str);
double  ToDouble (char const* str);
void  get_exe_name (char * buffer);
char  *GetIonConfigFile (const char filename[]);
bool  CopyFile (char *, char *);
int   GetNumLines (char *filename);

std::string get_time_iso_string(time_t time);

char  *GetProcessParam (const char *, const char *);
#ifndef ALIGNSTATS_IGNORE
void  defineSubRegion (int rows, int cols, int runIndex, int regionIdx, Region *cropRegions);
#endif
// Break a path to an indiviual file into the usual directory and file used for Analysis and libraries
void FillInDirName (const std::string &path, std::string &dir, std::string &file);
void  init_salute();


int seqToFlow (const char *seq, int seqLen, int *ionogram, int ionogramLen, char *flowOrder, int flowOrderLen);
#ifndef ALIGNSTATS_IGNORE
void flowToSeq (std::string &seq, hpLen_vec_t &hpLen, std::string &flowOrder);
#endif
void GetChipDim (const char *type, int dims[2], const char *);
std::string GetMemUsage();
void MemoryUsage (const std::string &s);
void MemUsage (const std::string &s);
int totalMemOnTorrentServer();
int GetAbsoluteFreeSystemMemoryInKB();
int GetFreeSystemMem();
int GetCachedSystemMem();
int GetSystemMemInBuffers();

// --- Mapping of base ambiguity symbols

void expandBaseSymbol(char nuc, std::vector<bool>& nuc_ensemble);
char contractNucSymbol(const std::vector<bool>& nuc_ensemble);
bool isBaseMatch(char nuc1, char nuc2);
char getMatchSymbol(char nuc1, char nuc2);

//string utils
int     count_char (std::string s, char c);
std::string get_file_extension (const std::string& s);
void    split (const std::string& s, char c, std::vector<std::string>& v);
void uintVectorToString (std::vector<unsigned int> &v, std::string &s, std::string &nullStr, char delim);
/** Trim off any whitespace from either end of string. */
void TrimString (std::string &str);
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
      st_last = st;
      usec = usec_last = 0;
    }
    
    void UpdateLast(const struct timeval &t) { st_last = t; }

    void CalcMicroSec() {
      struct timeval et;
      gettimeofday (&et, NULL);
      usec = (et.tv_sec*1.0e6+et.tv_usec) - (st.tv_sec * 1.0e6 + st.tv_usec);
      usec_last = (et.tv_sec*1.0e6+et.tv_usec) - (st_last.tv_sec * 1.0e6 + st_last.tv_usec);
      UpdateLast(et);
    }

    size_t GetSeconds() {
      CalcMicroSec();
      return usec / 1e6;
    }

    size_t GetMicroSec() {
      CalcMicroSec();
      return usec;
    }

    size_t GetMinutes() { return GetSeconds() / 60; }

    void PrintSeconds (std::ostream &out, const std::string &prefix) {
      out << prefix << " " << GetSeconds() << " seconds." << std::endl;
    }

    void PrintMicroSecondsUpdate(FILE *out, const std::string &prefix) {
      CalcMicroSec();
      fprintf(out, "%s since_last: %.2f seconds total: %.2f seconds\n", prefix.c_str(), usec_last/1e6, usec/1e6);
    }

    void USecUp(FILE *out, const std::string &prefix) {
      CalcMicroSec();
      fprintf(out, "%s since_last: %.2f seconds total: %.2f seconds\n", prefix.c_str(), usec_last/1e6, usec/1e6);
    }

    void PrintMicroSeconds (std::ostream &out, const std::string &prefix) {
      CalcMicroSec();
      out << prefix << " " << usec / 1e6 << " seconds." << std::endl;
    }

    void PrintMilliSeconds (std::ostream &out, const std::string &prefix)  {
      CalcMicroSec();
      out << prefix << " " << usec / 1e3 << " milli seconds." << std::endl;
    }

  private:
    size_t usec;
    size_t usec_last;
    struct timeval st;
    struct timeval st_last;
};

/** For timing repetitive jobs. */
class SumTimer {

public:
  SumTimer() { StartTimer(); }
  void StartTimer() { mUsecTotal = 0.0; mTimer.StartTimer(); mCalls=0; }
  void StartInterval() { mTimer.StartTimer(); }
  void EndInterval() { mUsecTotal += mTimer.GetMicroSec(); mCalls++;}
  double GetTotalUsec() const { return mUsecTotal; }
  size_t GetCalls() const { return mCalls; }
  void PrintSeconds (std::ostream &out, const std::string &prefix) const { 
    out << prefix << " " << GetTotalUsec() / 1e6 << " seconds in " << mCalls << " intervals." << std::endl; 
  }
  void PrintMilliSeconds (std::ostream &out, const std::string &prefix) const { 
    out << prefix << " " << GetTotalUsec() / 1e3 << " milli seconds in " << mCalls << " intervals." << std::endl;
  }
  
private:
  size_t mCalls;
  double mUsecTotal; // microseconds
  ClockTimer mTimer;
};

/**
 * Utility class for storing key value pairs. 
 */ 
class Info {

public:

  /** Get the value associated with a particular key, return false if key not present. */
  bool GetValue(const std::string &key, std::string &value) const;
  
  /** 
   * Set the value associated with a particular key. Newer values for
   * same key overwrite previos values. */
  bool SetValue(const std::string &key, const std::string &value);

  /** Get the key and the value associated at index. */
  bool GetEntry(int index, std::string &key, std::string &value) const;

  /** Get the total count of key value pairs valid for GetEntry() */
  int GetCount() const;

  /** Entry exists. */
  bool KeyExists(const std::string &key) const;
    
  
  /** Empty out the keys, value pairs. */
  void Clear();

private:   
  std::vector<std::string> mKeys;
  std::vector<std::string> mValues;
  std::map<std::string, size_t> mMap;
};

/** calculate median using partial sort which runs in N * log(N/2) time. */
template <typename T>
T fast_median(T* start, size_t num_elements) {
  T *middle, *end;
  end = start + num_elements;
  middle = start + (num_elements >> 1);
  std::partial_sort(start, middle, end);
  // Odd elements
  if (num_elements % 2 == 1) {
    return (*middle);
  }
  // else even, average middle two values
  return (*middle + *(middle -1)) /2.0f;
}

#endif // UTILS_H
