/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IONIMAGESEM_H
#define IONIMAGESEM_H
#include <pthread.h>

typedef struct ion_semaphore {
    pthread_mutex_t lock;
    pthread_cond_t nonzero;
    unsigned count;
    pthread_t owner;
    pthread_t WriterLock;
} ion_semaphore_t;

class IonImageSem {
 public:
  static void Take(int fast=0);
  static void Give();
  static void LockWriter();
  static void UnLockWriter();

 protected:
  static ion_semaphore_t * Create(const char *semaphore_name);
  static ion_semaphore_t * Open(const char *semaphore_name);
  static void              Init(void);
  static ion_semaphore_t *Ion_Image_SemPtr;
  static pthread_once_t  IonImageSemOnceControl;
};

#endif // IONIMAGESEM_H
