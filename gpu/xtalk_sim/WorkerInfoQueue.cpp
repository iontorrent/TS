/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "WorkerInfoQueue.h"

// create a queue w/ that can hold the specified number of items
WorkerInfoQueue::WorkerInfoQueue(int _depth)
{
  depth = _depth;
  rdndx = 0;
  wrndx = 0;
  num = 0;
  not_done_cnt = 0;
  qlist = new WorkerInfoQueueItem[depth];
  
  pthread_mutex_init(&lock, NULL);
  pthread_cond_init(&rdcond,NULL);
  pthread_cond_init(&wrcond,NULL);
  pthread_cond_init(&donecond,NULL);
}

// put a new item on the queue.  this will block if the queue is full
void WorkerInfoQueue::PutItem(WorkerInfoQueueItem &new_item)
{
  // obtain the lock
  pthread_mutex_lock(&lock);
  
  // wait for someone to signal new free space
  // fprintf(stdout, "WorkerInfoQueue::PutItem: %d\n", num);
  while (num >= depth)
    pthread_cond_wait(&wrcond,&lock);
  
  qlist[wrndx] = new_item;
  if (++wrndx >= depth) wrndx = 0;
  num++;
  not_done_cnt++;
  
  // signal readers to check for the new item
  pthread_cond_signal(&rdcond);
  
  // give up the lock
  pthread_mutex_unlock(&lock);
}

// remove an item from the queue.  this will block if the queue is empty
WorkerInfoQueueItem WorkerInfoQueue::GetItem(void)
{
  WorkerInfoQueueItem item;
  
  // obtain the lock
  pthread_mutex_lock(&lock);
  
  // wait for someone to signal a new item
  // fprintf(stdout, "WorkerInfoQueue::GetItem: %d\n", num);
  while (num == 0)
    pthread_cond_wait(&rdcond,&lock);
  
  item = qlist[rdndx];
  if (++rdndx >= depth) rdndx = 0;
  num--;
  
  // signal writers that more free space is available
  pthread_cond_signal(&wrcond);
  
  // give up the lock
  pthread_mutex_unlock(&lock);
  
  return(item);
}

// try to remove an item from the queue.  this will return item with empty data if the queue is empty
WorkerInfoQueueItem WorkerInfoQueue::TryGetItem(void)
{
  WorkerInfoQueueItem item;

  // obtain the lock
  pthread_mutex_lock(&lock);

  // wait for someone to signal a new item
  if (num == 0) {
	  pthread_mutex_unlock(&lock);
	  item.private_data = NULL;
	  return item;
  }

  item = qlist[rdndx];
  if (++rdndx >= depth) rdndx = 0;
  num--;

  // signal writers that more free space is available
  pthread_cond_signal(&wrcond);

  // give up the lock
  pthread_mutex_unlock(&lock);

  return(item);
}


// NOTE: just because the q is empty...doesn't mean the workers are done with the
// last item they pulled off.  Worker's decrement the 'not done' count whenever they
// finish a work item.  This waits till all the work items have been completed
void WorkerInfoQueue::WaitTillDone(void)
{
  // obtain the lock
  pthread_mutex_lock(&lock);
  
  // wait for someone to signal new free space
  while (not_done_cnt > 0)
    pthread_cond_wait(&donecond,&lock);
  
  // give up the lock
  pthread_mutex_unlock(&lock);
}

// Allows workers to indicate that they have completed a task
void WorkerInfoQueue::DecrementDone(void)
{
  // obtain the lock
  pthread_mutex_lock(&lock);
  
  // decrement done count
  not_done_cnt--;
  
  // if everything is done, signal the condition change
  if (not_done_cnt == 0)
    pthread_cond_signal(&donecond);
  
  // give up the lock
  pthread_mutex_unlock(&lock);
}

WorkerInfoQueue::~WorkerInfoQueue()
{
  pthread_cond_destroy(&donecond);
  pthread_cond_destroy(&wrcond);
  pthread_cond_destroy(&rdcond);
  pthread_mutex_destroy(&lock);
  delete [] qlist;
}


// create a queue w/ that can hold the specified number of items
DynamicWorkQueueGpuCpu::DynamicWorkQueueGpuCpu(int _depth)
{
  start = false;
  depth = _depth;
  gpuRdIdx = 0;
  cpuRdIdx = depth - 1;
  wrIdx = 0;
  not_done_cnt = 0;
  qlist = new WorkerInfoQueueItem[depth];
  
  pthread_mutex_init(&lock, NULL);
  pthread_cond_init(&startcond,NULL);
  pthread_cond_init(&donecond,NULL);
}


WorkerInfoQueueItem DynamicWorkQueueGpuCpu::GetGpuItem() {
  
  WorkerInfoQueueItem item;
  
  pthread_mutex_lock(&lock);
  //printf("Acquiring lock gpu\n");    
  
  
  while (!start)
    pthread_cond_wait(&startcond, &lock);
  
  if (gpuRdIdx == cpuRdIdx)
    start = false;
  
  //printf("Getting Gpu Item, GpuIdx: %d CpuIdx: %d, start: %d\n", gpuRdIdx, cpuRdIdx, start); 
  item = qlist[gpuRdIdx++];
  
  //printf("Releasing lock gpu\n");    
  pthread_mutex_unlock(&lock);         
  
  return item;
}

WorkerInfoQueueItem DynamicWorkQueueGpuCpu::GetCpuItem() {
  
  WorkerInfoQueueItem item;
  
  pthread_mutex_lock(&lock);
  
  //printf("Acquiring lock cpu\n");    
  
  
  while (!start)
    pthread_cond_wait(&startcond, &lock);
  
  if (cpuRdIdx == gpuRdIdx)
    start = false;
  
  //printf("Getting Cpu Item, GpuIdx: %d CpuIdx: %d, start: %d\n", gpuRdIdx, cpuRdIdx, start); 
  item = qlist[cpuRdIdx--];
  
  //printf("Releasing lock cpu\n");    
  pthread_mutex_unlock(&lock);         
  
  return item;
}

void DynamicWorkQueueGpuCpu::PutItem(WorkerInfoQueueItem &new_item) {
  
  
  qlist[wrIdx++] = new_item;
  not_done_cnt++;
  
  //printf("Putting Item\n"); 
  if (wrIdx == depth) {
    pthread_mutex_lock(&lock);
    //printf("Signalling to start\n");
    start = true;
    pthread_cond_broadcast(&startcond); 
    pthread_mutex_unlock(&lock);
  }
  
}


// NOTE: just because the q is empty...doesn't mean the workers are done with the
// last item they pulled off.  Worker's decrement the 'not done' count whenever they
// finish a work item.  This waits till all the work items have been completed
void DynamicWorkQueueGpuCpu::WaitTillDone(void)
{
  // obtain the lock
  pthread_mutex_lock(&lock);
  
  // wait for someone to signal new free space
  while (not_done_cnt > 0)
    pthread_cond_wait(&donecond,&lock);
  
  // give up the lock
  pthread_mutex_unlock(&lock);
}

// Allows workers to indicate that they have completed a task
void DynamicWorkQueueGpuCpu::DecrementDone(void)
{
  // obtain the lock
  pthread_mutex_lock(&lock);
  
  // decrement done count
  not_done_cnt--;
  
  // if everything is done, signal the condition change
  if (not_done_cnt == 0) {
    start = false;
    pthread_cond_signal(&donecond);
  }
  
  // give up the lock
  pthread_mutex_unlock(&lock);
}

void DynamicWorkQueueGpuCpu::ResetIndices() {
  pthread_mutex_lock(&lock);
  wrIdx = 0;
  gpuRdIdx = 0;
  cpuRdIdx = depth - 1;
  pthread_mutex_unlock(&lock);
}

int DynamicWorkQueueGpuCpu::getGpuReadIndex() {
  return gpuRdIdx;
}

DynamicWorkQueueGpuCpu::~DynamicWorkQueueGpuCpu()
{
  pthread_cond_destroy(&donecond);
  pthread_cond_destroy(&startcond);
  pthread_mutex_destroy(&lock);
  delete [] qlist;
}


