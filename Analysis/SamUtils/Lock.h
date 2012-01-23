/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *  Lock.h
 *  SamUtils_mt_branch
 *
 *  Created by Michael Lyons on 2/22/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#ifndef LOCK_H
#define LOCK_H

#include <pthread.h>
/**
 An interface to encapsulate a synchronization mechanism in a concurrent computation environment.  
 
 */
class Lockable {
public:
	/**
	 Grants mutual exclusion to caller.  
	 */
    virtual void lock() = 0;
	/**
	 Returns the mutual exclusion to the program
	 */
    virtual void unlock() = 0;
	
    virtual ~Lockable() { };
};

/**
 An implementation of a lock that is extremely simple to use.  
 Given some level of scope, the lock will be obtained once that scope is reached, and 
 released once this object goes out of scope.  
 
 {//start scope
	ScopedLock(Lockable);
	// all code from here on is in a mutually exclusive code block until
	// ScopedLock goes out of scope
 }//end scope
 
 */
class ScopedLock {
    Lockable *lockable;
	
public:
	/**
	 Requires a Lockable implementation in order to to construct.
	 */
    explicit ScopedLock(Lockable *lockable): lockable(lockable) {
        lockable->lock();
    }
	
    virtual ~ScopedLock() {
        lockable->unlock();
    }
};

/**
 Class which wraps pthread_mutex.  Provides an OO interface to this C structure
 This is meant to be a C++ replacement for a pthread_mutex
 */
class NativeMutex: public Lockable {
	
    pthread_mutex_t mutx;	/**< a pthread_mutex */
    friend class NativeCond; /**< let's be friends */
	
public:
	/**
	 Initialze the mutex.  Immediately ready to use after construction
	 */
    NativeMutex() {
        pthread_mutex_init(&mutx, NULL);
    }
	
	/**
	 Attempts to get mutual exclusion.  This functionn call will block if
	 the mutex is already locked
	 */
    virtual void lock() {
        pthread_mutex_lock(&mutx);
    }
	
	/**
	 Attemps to release mutual exclusion.  
	 */
    virtual void unlock() {
        pthread_mutex_unlock(&mutx);
    };
	
    virtual ~NativeMutex() {
        pthread_mutex_destroy(&mutx);
    }
};

/**
 Class which wraps pthread_cond_t.  Provides an OO interface to this C structure.
 This is meant to be a C++ replacement for a pthread_cond_t
 */
class NativeCond {
    pthread_cond_t cond; /**< the actual conditional */
    
public:
	/**
	 Initializes the conditional.  Ready to use after construction.
	 */
    NativeCond() {
        pthread_cond_init(&cond, NULL);
    }
	
	/**
	 Wait for a broadcast or signal
	 @param NativeMutex* mutex		a NativeMutex
	 */
    void wait(NativeMutex *mutex) {
        pthread_cond_wait(&cond, &(mutex->mutx));
    }
	
	/**
	 Broadcast to all threads waiting on this condition to wake up
	 */
    void broadcast() {
        pthread_cond_broadcast(&cond);
    }
	
	/**
	 Signal a thread waiting on this condition to wake up
	 */
    void signal() {
        pthread_cond_signal(&cond);
    }
    
    virtual ~NativeCond() {
        pthread_cond_destroy(&cond);
    }
    
};

#endif // LOCK_H

