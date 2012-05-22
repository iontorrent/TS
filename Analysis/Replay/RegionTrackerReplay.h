/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONTRACKERREPLAY_H
#define REGIONTRACKERREPLAY_H

#include "RegionTracker.h"
#include "CommandLineOpts.h"
#include "H5Replay.h"

// *********************************************************************
// reader specific class

class RegionTrackerReader : public RegionTracker {

 public:
  RegionTrackerReader (CommandLineOpts &_clo, int _regionindex);
  ~RegionTrackerReader();

  void Read(int flow);

 private:
  std::vector<int> flowToIndex;
  std::vector<int> flowToBlockId;
  int reg_params_length;
  int numFlows;
  
  CommandLineOpts &clo;
  int regionindex;

  H5ReplayReader *reader_rp;
  H5ReplayReader *reader_mm;

  void Init();
  void ReadFlowIndex();
  void ReadRegParams(int flowindex);
  void ReadMissingMass(int flowindex);

  RegionTrackerReader (); // don't use
};

// *********************************************************************
// recorder specific class

class RegionTrackerRecorder : public RegionTracker {

 public:
  RegionTrackerRecorder(CommandLineOpts &clo, int _regionindex);
  ~RegionTrackerRecorder();

  void Write(int flow);

 private:
  std::vector<int> flowToIndex;
  std::vector<int> flowToBlockId;
  int reg_params_length;
  int numFlows;
  
  CommandLineOpts &clo;
  int regionindex;

  H5ReplayRecorder *recorder_rp;
  H5ReplayRecorder *recorder_mm;

  void Init();
  void WriteFlowInfo();
  void WriteRegParams(int block_id);
  void WriteMissingMass(int block_id);

  RegionTrackerRecorder(); // don't use
};


#endif // REGIONTRACKERREPLAY_H
