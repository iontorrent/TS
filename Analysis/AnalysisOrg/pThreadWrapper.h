/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * pThreadWrapper.h
 *
 *  Created on: Jul 15, 2015
 *      Author: Jakob Siegel
 */

#ifndef PTHREADWRAPPER_H_
#define PTHREADWRAPPER_H_


#include <pthread.h>



class pThreadWrapper
{

  pthread_t _thread;
  static void * InternalThreadEntryFunc(void * This) {((pThreadWrapper *)This)->InternalThreadFunction(); return NULL;}

public:

  pThreadWrapper(){
    /*initialize _thread with parent thread so we know that no child thread was created yet */
    _thread = pthread_self();
  }

  /*should probably force a join if _thread != pthread_self() but only provides warning output for now*/
  virtual ~pThreadWrapper()
  {
    if( ! pthread_equal( pthread_self(),_thread ))
      cout << "WARNING: pThreadWrapper is getting destroyed while created pthread is neither detached or joined!"<< endl;
  }

   /* Returns true if the internal thread already was created or successfully started or is already, false if there was an error starting the thread */
   bool StartInternalThread()
   {
      if(hasChildThread()) return true;
      if ((pthread_create(&_thread, NULL, InternalThreadEntryFunc, this) == 0)) return true;
      else _thread = pthread_self();
      return false;
   }

   /* Will not return until the internal thread has exited. */
   void JoinInternalThread()
   {
     if(hasChildThread()){ /* only try to perform join if _thread was created as a new thread */
         (void) pthread_join(_thread, NULL);
         _thread = pthread_self(); //mark as child less
     }
   }


   bool hasChildThread()
   {
     return  (!pthread_equal( pthread_self(),_thread ));
   }

   /* USING DETACH WITH A THREAD WRAPPER CALSS VOIDS ANY BENEFITS FROM WRAPPING THE THREADS IN A CLASS AND IS NOT RECOMMENDED SINCE
    * A DETACHED THREAD SHOULD NOT USE RESOURCES HELD IN THE DERIVED WRAPPER CLASS OTHERWISE CORRECTNESS CANOT BE GUARANTEED IF
    * WRAPPER CLASS OBJECT GETS DESTROYED BEFORE THE DETACHED THREAD FINISHES.
    */
   //void detachThread()
   //{
   //  if( pthread_equal( pthread_self(),_thread )) /* only detach if _thread was created as a new thread */
   //  {
   //      (void) pthread_detach(_thread); /*CAUTION: the thread might still rely on resources only available in the derived class! */
   //  }
   //  _thread = pthread_self(); /* thread is detached and we have no handle to it anymore. */
   //}


protected:
   /* Implement this method in your subclass with the code you want your thread to run. */
   virtual void InternalThreadFunction() = 0;

};


#endif /* PTHREADWRAPPER_H_ */
