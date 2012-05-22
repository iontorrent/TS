/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef MUTEXUTILS_H
#define MUTEXUTILS_H

#include <pthread.h>

namespace ION
{
/**
 * struct ScopedMutex
 *      This struct holds a pointer to a mutex that is locked
 *      in the constructor and automatically unlocked in the
 *      destructor when this object leaves scope.
 * 
 * @param pMutex is the mutex to be locked then unlocked
 */
struct ScopedMutex
{
    ScopedMutex( pthread_mutex_t *pMutex )
        : _pMutex(pMutex) { pthread_mutex_lock( _pMutex ); }
    ~ScopedMutex() { pthread_mutex_unlock( _pMutex ); }
private:
    pthread_mutex_t *_pMutex;
};
// END struct ScopedMutex

}
// END namespace ION

#endif // MUTEXUTILS_H

