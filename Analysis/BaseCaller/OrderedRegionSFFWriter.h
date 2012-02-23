/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ORDEREDREGIONSFFWRITER_H
#define ORDEREDREGIONSFFWRITER_H

#include <string>
#include <vector>
#include <deque>

#include "SpecialDataTypes.h"
#include "file-io/sff_definitions.h"

using namespace std;

class SFFWriterWell {
public:
  SFFWriterWell() { numBases = 0; clipQualLeft=clipQualRight=clipAdapterLeft=clipAdapterRight=0; }
  ~SFFWriterWell() {}

  void  moveTo(SFFWriterWell &w);
  void  copyTo(SFFWriterWell &w);

  string            name;
  weight_vec_t      flowIonogram;
  vector<uint8_t>   baseFlowIndex;
  vector<char>      baseCalls;
  vector<uint8_t>   baseQVs;
  int               numBases;
  int32_t           clipQualLeft;
  int32_t           clipQualRight;
  int32_t           clipAdapterLeft;
  int32_t           clipAdapterRight;
};



class OrderedRegionSFFWriter {
public:
  OrderedRegionSFFWriter();
  ~OrderedRegionSFFWriter();

  void OpenForWrite(const char *experimentName, const char *sffFileName, int numRegions, int numFlows,
      const char *flowChars, const char *keySequence);

  void WriteRegion(int iRegion, deque<SFFWriterWell> &regionWells);

  void Close();

  int NumReads() { return numReads; }

private:

  void PhysicalWriteRegion(int iRegion);

  int                           numReads;
  int                           numFlows;
  int                           numRegions;

  // Queue for reads to be written out
  int                           numRegionsWritten;
  vector<bool>                  isRegionReady;
  vector<deque<SFFWriterWell> > regionDropbox;
  pthread_mutex_t               *dropboxWriteMutex;
  pthread_mutex_t               *sffWriteMutex;

  // Stuff related to actual SFF writing
  sff_file_t                    *sff_file;
  sff_t                         *sff;
};



#endif // ORDEREDREGIONSFFWRITER_H
