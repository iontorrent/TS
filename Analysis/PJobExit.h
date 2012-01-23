/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PJOBEXIT_H
#define PJOBEXIT_H

#include "PJob.h"

/** 
 * Job that just kills the threads.
 */ 
class PJobExit : public PJob {

 public:
  /** Exit the thread. */
  void Run() { }
  void TearDown() {Exit(); }
};

#endif // PJOBEXIT_H
