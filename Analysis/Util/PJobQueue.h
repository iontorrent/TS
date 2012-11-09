/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PJOBQUEUE_H
#define PJOBQUEUE_H

#include <vector>
#include <pthread.h>
#include "WorkerInfoQueue.h"
#include "PJob.h"
#include "PJobExit.h"
#include "PJobQueue.h"

/**
 * Manages a queue of posix threads that will run jobs adhering to the
 * PJob.h interface.
 */
class PJobQueue {
public:

  /** Basic constructor. */
  PJobQueue() { 
    mWorkQueue = NULL;
  }

  /** Intitialize a queue and begin specified number of threads. */
  PJobQueue(int nThreads, int queueSize) {
    Init(nThreads, queueSize);
  }

  /** Destructor, kill all existing threads and clean up memory. */
  ~PJobQueue() {
    // @todo - some check to make sure we didn't hang?
    ExitThreads();
    delete mWorkQueue;
  }

  /** Start up the specified number of threads and create queue of specified size. */
  void Init(int nThreads, int queueSize) {
    assert(nThreads > 0 && queueSize > 0);
    // @todo - What happens if you don't fill the queue or you try to overfill?
    mWorkQueue = new WorkerInfoQueue(queueSize);
    mWorkers.resize(nThreads,0);
    for (int i = 0; i < nThreads; i++) {
      int t = pthread_create(&mWorkers[i], NULL, PJobQueue::PJobQueueWorker, mWorkQueue);
      pthread_detach(mWorkers[i]);
      assert(t == 0);
    }
  }
  
  size_t NumThreads() { return mWorkers.size(); }

  /** Run a job encapsulated by the PJob Interface. */
  static void *PJobQueueWorker(void *arg) {
    WorkerInfoQueue *queue = static_cast<WorkerInfoQueue *>(arg);
    assert(queue);
    int done = false;
    while(!done) {
      // Query for the next job to run
      WorkerInfoQueueItem item = queue->GetItem();
      // @todo Is this really necessary? Do we ever get an item that is finished?
      if (item.finished == true) {
	queue->DecrementDone();
	continue;
      }
      PJob &job = *((PJob *)(item.private_data));
      // @todo - handle errors or failures? 
      job.SetUp();
      job.Run();
      item.finished = true;
      bool isEnd =  job.IsEnd();
      job.TearDown();
      queue->DecrementDone();
      if (isEnd) {
        pthread_exit(NULL);
      }
    }
    return 0;
  }

  /** 
   * Put the specified job in the queue to be run. Note that
   * the memory for the job is owned elsewhere and no cleanup will
   * be done by this function. Don't cleanup memory until sucessful
   * call to WaitUntilDone()
   */
  void AddJob(PJob &job) {
    assert(mWorkers.size() > 0);
    WorkerInfoQueueItem item;
    item.finished = false;
    item.private_data = (void *)&job;
    mWorkQueue->PutItem(item);
  }
  
  /** Wait for current jobs in queue to run. */
  void WaitUntilDone() {
    mWorkQueue->WaitTillDone();
  }

 private:

  /** Assign each thread a job that will kill it. */
  void ExitThreads() {
    std::vector<PJobExit> jobs(mWorkers.size());
    for (size_t i = 0; i < mWorkers.size(); i++) {
      AddJob(jobs[i]);
    }
    WaitUntilDone();
    for(size_t i = 0; i < mWorkers.size(); i++) {
      //pthread_join(mWorkers[i], NULL);
      //pthread_detach(mWorkers[i]);
    }
    mWorkers.resize(0);
  }

  WorkerInfoQueue *mWorkQueue;       ///< Queue to manage the threads
  std::vector<pthread_t> mWorkers; ///< Ids of the posix threads
};

#endif // PJOBQUEUE_H

