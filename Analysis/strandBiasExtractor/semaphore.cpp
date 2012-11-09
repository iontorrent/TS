/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

/* I N C L U D E S ***********************************************************/

#include "semaphore.h"
#include <pthread.h>

using namespace ion;

void ion::init_semaphore(semaphore_t *semaphore, int count)
{
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
  pthread_mutex_init(&(semaphore->mutex), &attr);
  
  pthread_cond_init(&(semaphore->cond), 0);
  semaphore->count = count;
  return;
}

void ion::delete_semaphore(semaphore_t *semaphore)
{
  pthread_mutex_destroy(&(semaphore->mutex));
  pthread_cond_destroy(&(semaphore->cond));
  return;
}

void ion::down(semaphore_t *semaphore)
{
  int retVal = 0;
  //fprintf(stdout, "trying to obtain lock mutex  in down %d \n", retVal);
  retVal = pthread_mutex_lock(&(semaphore->mutex));
  if (retVal)
	fprintf(stdout, "obtained lock on mutex in down %d \n", retVal);
 
  semaphore->count--;
 //if(semaphore->count < 0)
  //{
  //  pthread_mutex_unlock(&(semaphore->mutex));
 //   pthread_cond_wait(&(semaphore->cond), &(semaphore->cond_m));
  //  pthread_mutex_unlock(&(semaphore->cond_m));
  //}
 // else
 // {
    retVal = pthread_mutex_unlock(&(semaphore->mutex));
	if (retVal)
		fprintf(stdout, "mutex returned error in unlock down %d \n", retVal);
    
	pthread_cond_signal(&(semaphore->cond));
 // }
	//fprintf(stdout, "realeased lock from down \n");
  return;
}

void ion::up(semaphore_t *semaphore)
{
	int retVal = 0;
	
   
  //fprintf(stdout, "trying to obtain lock on mutex in up \n");
  retVal = pthread_mutex_lock(&(semaphore->mutex));
  if (retVal)
	fprintf(stdout, "retval for lock mutex in up %d \n", retVal);
  semaphore->count++;
  retVal = pthread_mutex_unlock(&(semaphore->mutex));
  pthread_cond_signal(&(semaphore->cond));
  if (retVal)
	fprintf(stdout, "released lock in up %d \n", retVal);
  return;
}
