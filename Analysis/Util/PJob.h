/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PJOB_H
#define PJOB_H

#include <pthread.h>

/** 
 * Abstract class for encapsulating work that needs to be done and run
 * by a thread.
 * 
 * @todo - should jobs get a payload of void * or just be initialized
 * with everything they need?
 */ 
class PJob {

 public:
  PJob() {};
  virtual ~PJob() {}
  /** Do any initialization before running */
  virtual void SetUp() {}
  /** Process work. */
  virtual void Run() = 0;
  /** Cleanup any resources. */
  virtual void TearDown() {}
  /** Signal to kill worker. */
  virtual bool IsEnd() { return false; }
  /** Exit this pthread (killing thread) */
  void Exit() {
    pthread_exit(NULL);
  }
};

#endif // PJOB_H

