/* $Id: AeOSLocal.h,v 1.10.2.1 2009/12/01 18:07:58 hfeng Exp $ */

#ifndef _AE_OS_LOCAL_H_
#define _AE_OS_LOCAL_H_

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include <unistd.h>
#include <ctype.h>
#include <fcntl.h>
#include <stdarg.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <dirent.h>

#if defined(sun)
#include <sys/filio.h>
#endif

#if defined(HAVE_PTHREADS)
#include <pthread.h>

typedef pthread_mutex_t AeMutex;

#if !defined(PTHREAD_MUTEX_RECURSIVE) && defined(PTHREAD_MUTEX_RECURSIVE_NP)
#define PTHREAD_MUTEX_RECURSIVE PTHREAD_MUTEX_RECURSIVE_NP
#endif

#else

typedef struct _AeMutex AeMutex;
struct _AeMutex
{
    int dummy;
};
#endif /* defined(HAVE_PTHREADS) */

/* buffer size for sockets. change the value as you need */
#define AE_BUFFER_CHUNK 4096

typedef int  AeSocket;
typedef int  AeSocketFD;
typedef fd_set AeFDArray;
typedef int  AeFileHandle;
typedef DIR* AeDirHandle;

#if (defined __GNUC__ || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L))
typedef long long AeInt64;
typedef unsigned long long AeUInt64;
#define AE_INT64_FORMAT_MODIFIER "ll"
#define AeStrToInt64 strtoll
#endif

#define AeSocketGetFD(x)    *(x)
#define AeFDZero            FD_ZERO
#define AeFDSet             FD_SET
#define AeFDIsSet           FD_ISSET
#define AeFileInvalidHandle	-1

void *AeAlloc(int size);
void AeFree(void* memptr);
void *AeCalloc(int BlockCount,int BlockSize);
void *AeRealloc(void* memptr,int NewSize);

#define PATH_CHAR	'/'
#define PATH_STR	"/"

/* provide compatibility with [broken] uc-libc for uClinux */
#if defined(m68k) && defined(__pic__) && defined(__linux__) && !defined(__UCLIBC__)
#define __UC_LIBC__

extern int snprintf(char *str, size_t size, const char *format, ...);
extern int vsnprintf(char *str, size_t size, const char *format, va_list ap);
extern int vsprintf(char *str, const char *format, va_list ap);
extern int sscanf(const char *str, const char *format, ...);

extern void abort(void);

extern int gethostname(char *name, size_t len);

#endif /* defined(m68k) && defined(__pic__) && defined(__linux__) && !defined(__UCLIBC__) */

#endif
