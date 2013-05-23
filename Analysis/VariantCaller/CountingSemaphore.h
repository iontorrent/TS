/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef COUNTINGSEMAPHORE_H
#define COUNTINGSEMAPHORE_H

/* I N C L U D E S ***********************************************************/
#ifndef COUNTING_SEM
#define COUNTING_SEM

#include <pthread.h>
#include <stdio.h>

namespace ion {
typedef struct
{
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int count;
} semaphore_t;


void init_semaphore(semaphore_t *semaphore, int count);
void delete_semaphore(semaphore_t *semaphore);
void up(semaphore_t *semaphore);
void down(semaphore_t *semaphore);
}
#endif // COUNTING_SEM
#endif // COUNTINGSEMAPHORE_H
