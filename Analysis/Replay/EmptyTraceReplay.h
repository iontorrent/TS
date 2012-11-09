/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EMPTYTRACEREPLAY_H
#define EMPTYTRACEREPLAY_H

#include "EmptyTrace.h"
#include "CommandLineOpts.h"
#include "H5Replay.h"
#include "IonErr.h"

// *********************************************************************
// reader specific class

class EmptyTraceReader : public EmptyTrace {

 public:
  EmptyTraceReader (CommandLineOpts &clo);
  ~EmptyTraceReader();
  void GenerateAverageEmptyTrace(Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, Image *img, int flow);
  void GenerateAverageEmptyTrace (Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, SynchDat &sdat, int flow) { ION_ABORT("Not supported");}
  void  Allocate(int numfb, int _imgFrames);

 private:
  void ReadEmptyTraceBuffer(int flow);
  void FillBuffer(int iFlowBuffer, int regionIndex);

  int bufferCount;
  pthread_mutex_t *read_mutex;

  H5ReplayReader *reader;

  EmptyTraceReader (); // don't use
};

// *********************************************************************
// recorder specific class

class EmptyTraceRecorder : public EmptyTrace {

 public:
  EmptyTraceRecorder(CommandLineOpts &clo);
  ~EmptyTraceRecorder();
  void GenerateAverageEmptyTrace(Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, Image *img, int flow);
  void GenerateAverageEmptyTrace (Region *region, PinnedInFlow& pinnedInFlow, Mask *bfmask, SynchDat &sdat, int flow){ ION_ABORT("Not supported"); }
  void  Allocate(int numfb, int _imgFrames);

 private:
  void WriteEmptyTraceBuffer(int flow);
  void WriteBuffer(int flow, int regionindex);

  int bufferCount;
  pthread_mutex_t *write_mutex;

  H5ReplayRecorder *recorder;

  EmptyTraceRecorder(); // don't use
};

#endif // EMPTYTRACEREPLAY_H
