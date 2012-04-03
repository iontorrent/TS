/* $Id: AeOSLocal.h,v 1.5.2.1 2009/12/01 18:07:58 hfeng Exp $ */

#ifndef _AE_OS_LOCAL_H_
#define _AE_OS_LOCAL_H_

#ifndef _WIN32
#define _WIN32
#endif

#define _WIN32_WINNT 0x0400
#if(_MSC_VER < 1300)
    #include <windows.h>
    #include <ws2tcpip.h>
#else
	#define _CRT_SECURE_NO_DEPRECATE  // re:  Secure Template Overloads see MSDN
    #define _USE_32BIT_TIME_T       // otherwise time_t is 64 bits in .2005
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include <ws2tcpip.h>
#include <malloc.h>
#include <sys/timeb.h>
#include <io.h>
#include <limits.h>

#if _INTEGRAL_MAX_BITS >= 64
typedef __int64 AeInt64;
typedef unsigned __int64 AeUInt64;
#define AE_INT64_FORMAT_MODIFIER "I64"
#define AeStrToInt64(s, e, r) _atoi64(s)
#endif

/* buffer size for sockets. change the value as you need */
#define AE_BUFFER_CHUNK 4096

typedef SOCKET AeSocket;
typedef SOCKET AeSocketFD;
typedef fd_set AeFDArray;
typedef HANDLE AeFileHandle;
typedef HANDLE AeDirHandle;
typedef HANDLE AeMutex;

#define AeSocketGetFD(x)    *(x)
#define AeFDZero            FD_ZERO
#define AeFDSet             FD_SET
#define AeFDIsSet           FD_ISSET
#define AeFileInvalidHandle INVALID_HANDLE_VALUE

void *AeAlloc(int size);
void AeFree(void* memptr);
void *AeCalloc(int BlockCount,int BlockSize);
void *AeRealloc(void* memptr,int NewSize);

#define snprintf                    _snprintf
#define strcasecmp(s1, s2)          _stricmp(s1, s2)
#define strncasecmp(s1, s2, n)      _strnicmp(s1, s2, n)

#define PATH_CHAR   '\\'
#define PATH_STR    "\\"

#endif
