/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include "IonImageSem.h"
#ifdef BB_DC
#include "logging.h"
#define ISMLOG DTRACE
#else
#define ISMLOG printf
#endif

pthread_once_t  IonImageSem::IonImageSemOnceControl = PTHREAD_ONCE_INIT;
ion_semaphore_t *IonImageSem::Ion_Image_SemPtr=NULL;

void IonImageSem::Init()
{
	if(IonImageSem::Ion_Image_SemPtr == NULL) // shouldn't need this check, but....
	{
		IonImageSem::Ion_Image_SemPtr = IonImageSem::Open("/tmp/IonImageSemV2");
	}
}

ion_semaphore_t * IonImageSem::Create(const char *semaphore_name)
{
	int fd;
	ion_semaphore_t *semap;

    fd = open(semaphore_name, O_RDWR | O_CREAT /*| O_EXCL*/, 0666);
    if (fd < 0){
    	ISMLOG("%s: failed, %s\n",__FUNCTION__,strerror(errno));
        return (NULL);
    }
    if( ftruncate(fd, sizeof(ion_semaphore_t)))
    {}
#ifdef USE_ION_PTHREAD_SEM
    pthread_mutexattr_t psharedm;
    pthread_condattr_t psharedc;
    (void) pthread_mutexattr_init(&psharedm);
//    int robustness=0;
//    (void) pthread_mutexattr_getrobust_np(&psharedm,&robustness);
//	DTRACEP("%s: robust: %d\n",__FUNCTION__,robustness);
    (void) pthread_mutexattr_setpshared(&psharedm, PTHREAD_PROCESS_SHARED);
    (void)pthread_mutexattr_setrobust_np(&psharedm, PTHREAD_MUTEX_ROBUST_NP);
    (void) pthread_condattr_init(&psharedc);
    (void) pthread_condattr_setpshared(&psharedc,
        PTHREAD_PROCESS_SHARED);
#endif
    semap = (ion_semaphore_t *) mmap(NULL, sizeof(ion_semaphore_t),
            PROT_READ | PROT_WRITE, MAP_SHARED,
            fd, 0);
    close (fd);
    if(semap)
    {
    	memset(semap,0,sizeof(ion_semaphore_t));
#ifdef USE_ION_PTHREAD_SEM
    	(void) pthread_mutex_init(&semap->lock, &psharedm);
#endif
    //    (void) pthread_cond_init(&semap->nonzero, &psharedc);
    }
    return (semap);
}


ion_semaphore_t *IonImageSem::Open(const char *semaphore_name)
{
    int fd;
    ion_semaphore_t *semap;
    ISMLOG("%s: trying to open file: '%s'\n", __FUNCTION__, semaphore_name);
    fd = open(semaphore_name, O_RDWR, 0666);
    if (fd < 0){
    	ISMLOG("%s: Creating sema, %s\n",__FUNCTION__,strerror(errno));
    	semap = Create(semaphore_name);
    }
    else
    {
      semap = (ion_semaphore_t *) mmap(NULL, sizeof(ion_semaphore_t),
				       PROT_READ | PROT_WRITE, MAP_SHARED,
				       fd, 0);
      close (fd);
      if(semap == NULL)
	semap = Create(semaphore_name);

#ifdef USE_ION_PTHREAD_SEM
      int lock_rc;
      if((lock_rc = pthread_mutex_lock(&semap->lock)) != 0)
	{
			munmap(semap,sizeof(ion_semaphore_t));
			semap = ion_semaphore_create(semaphore_name);
	}
      else
	{
	  pthread_mutex_unlock(&semap->lock);
	}
#endif
    }
    
    return (semap);
}

void IonImageSem::LockWriter()
{
    pthread_once(&IonImageSem::IonImageSemOnceControl, IonImageSem::Init);

	if(Ion_Image_SemPtr)
	{
		if(Ion_Image_SemPtr->WriterLock == 0){
			ISMLOG("%s: writer locking semaphore\n",__FUNCTION__);
		}
		Ion_Image_SemPtr->WriterLock = pthread_self();
	}
}

void IonImageSem::UnLockWriter()
{
    pthread_once(&IonImageSem::IonImageSemOnceControl, IonImageSem::Init);

    if(Ion_Image_SemPtr)
	{
		if(Ion_Image_SemPtr->WriterLock != 0){
			ISMLOG("%s: writer unlocking semaphore\n",__FUNCTION__);
		}
		Ion_Image_SemPtr->WriterLock = 0;
	}
}


void IonImageSem::Take(int fast)
{
    pthread_once(&IonImageSem::IonImageSemOnceControl, IonImageSem::Init);

    ion_semaphore_t *sem = Ion_Image_SemPtr;
	uint32_t msecs_waited=0;
	uint32_t its_the_same_cnt=0;
	pthread_t last_owner=0;
	unsigned last_cnt=0;
	static unsigned myLocalCnt=0;

	if(sem)
	{
#ifdef USE_ION_PTHREAD_SEM
	  int lock_rc;
		if((lock_rc = pthread_mutex_lock(&ssem->lock)) != 0)
		  ISMLOG("problems taking image sem %s\n",strerror(lock_rc));
#else
         uint32_t i;
         uint32_t sleepUsecs = 10000;
         uint32_t same_limit=(30*(1000000/sleepUsecs)); // if the same reader holds for more than 30 seconds, something is wrong...
         uint32_t limit=30*60*(1000000/sleepUsecs);


         if(fast){
                 sleepUsecs /= 5;
                 limit *= 5;
                 same_limit *= 5;
         }

         for(i=0;i<limit;i++)
           {
             if(sem->WriterLock == 0 && sem->owner == 0)
               {
                 break;
               }
             else if(sem->WriterLock == 0 && (sem->owner == last_owner && sem->count == last_cnt)){
				 its_the_same_cnt++;
				 if(its_the_same_cnt > same_limit){
					 break;
				 }
			 }else{
				 its_the_same_cnt=0;
				 last_owner = sem->owner;
				 last_cnt = sem->count;
			 }
             usleep(sleepUsecs);
           }
		if(i == limit || its_the_same_cnt > same_limit){
		  ISMLOG("issues taking semaphore writer=%lx owner=%lx\n",sem->WriterLock,sem->owner);
		}
		
		msecs_waited = i*sleepUsecs/1000;
		if(msecs_waited > 0){
		  ISMLOG("waited %d msecs for file semaphore\n",msecs_waited);
		}
		sem->owner = pthread_self();
		sem->count = myLocalCnt++;
#endif
	}
	else
	{
		ISMLOG("Image Sem is NULL\n");
	}
}

void IonImageSem::Give() {
    pthread_once(&IonImageSem::IonImageSemOnceControl, IonImageSem::Init);


    ion_semaphore_t *sem = Ion_Image_SemPtr;
  if(sem)
    {
#ifdef USE_ION_PTHREAD_SEM
      pthread_mutex_unlock(&sem->lock);
#else
      if(sem->owner == pthread_self())
	{
	  sem->owner = 0;
	}
#endif
    }
}
