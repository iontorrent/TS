/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PINNEDINFLOWREPLAY_H
#define PINNEDINFLOWREPLAY_H

#include "PinnedInFlow.h"
#include "CommandLineOpts.h"
#include "H5Replay.h"

// *********************************************************************
// reader specific class

class PinnedInFlowReader : public PinnedInFlow {

 public:
  PinnedInFlowReader (Mask *maskPtr, int numFlows, CommandLineOpts &clo);
  ~PinnedInFlowReader();

  void Initialize (Mask *maskPtr);
  int Update(int flow, Image *img);

 private:

  H5ReplayReader *reader_1;
  H5ReplayReader *reader_2;

  void InitializePinnedInFlow (Mask *maskPtr);
  void InitializePinsPerFlow ();

  int mW;             // Image Width
  int mH;             // Image height

  PinnedInFlowReader (); // don't use
};

// *********************************************************************
// recorder specific class

class PinnedInFlowRecorder : public PinnedInFlow {

 public:
  PinnedInFlowRecorder(Mask *maskPtr, int numFlows, CommandLineOpts &clo);
  ~PinnedInFlowRecorder();

 private:

  H5ReplayRecorder *recorder_1;
  H5ReplayRecorder *recorder_2;

  int mW;             // Image Width
  int mH;             // Image height

  void CreateDataset_1();
  void CreateDataset_2();
  void WriteBuffer_1();
  void WriteBuffer_2();

  PinnedInFlowRecorder(); // don't use
};







#endif // PINNEDINFLOWREPLAY_H
