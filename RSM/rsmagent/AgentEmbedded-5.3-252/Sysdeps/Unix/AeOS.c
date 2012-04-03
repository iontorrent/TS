/* $Id: AeOS.c,v 1.12.2.2 2009/12/01 18:08:47 hfeng Exp $ */

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"

static AeError UnixTranslateError(int iError);
static void UnixSignalHandlerStub(int iSignal);
static void _AeLogEx(AeLogRecordType iType, AeUInt32 iCategory, AeChar *pFormat, va_list *pArgs);

/* 
 * intermediate function to handle the following warning about the use of %c:
 * warning: `%c' yields only last 2 digits of year in some locales
 */
#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 3) && (__GNUC_PATCHLEVEL__== 1)
	static size_t my_strftime(char *s, size_t max, const char  *fmt,  const struct tm *tm);
#endif

/******************************************************************************/
void AeSRand()
{
    srand(time(NULL));
}

/******************************************************************************/
double AeRand()
{
    return rand();
}

void *AeAlloc(int size)
{
    void* p = malloc(size);
    return p;
}

void AeFree(void* memptr)
{
    free(memptr);
}

void *AeCalloc(int BlockCount,int BlockSize)
{
    return calloc(BlockCount, BlockSize);
}

void *AeRealloc(void* memptr,int NewSize)
{
    return realloc(memptr, NewSize);
}


/******************************************************************************/
AeError AeNetInitialize(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    // essentially ignore SIGPIPE
    sa.sa_handler = UnixSignalHandlerStub;
    sigaction(SIGPIPE, &sa, NULL);
	
    return AeEOK;
}

/******************************************************************************/
AeError AeNetGetSocket(AeSocket *pSock, AeBool bStream)
{
    int iType;
    
    if (bStream)
        iType = SOCK_STREAM;
    else
        iType = SOCK_DGRAM;

    *pSock = socket(AF_INET, iType, 0);
    if ((*pSock) == -1)
        return UnixTranslateError(errno);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetConnect(AeSocket *pSock, AeNetAddress *pAddress)
{
    struct sockaddr_in inAddr;
    int rc;

    inAddr.sin_family = AF_INET;
    inAddr.sin_addr.s_addr = pAddress->iAddress;
    inAddr.sin_port = pAddress->iPort;

    rc = connect(*pSock, (struct sockaddr *) &inAddr, sizeof(struct sockaddr_in));
    if (rc != 0)
        return UnixTranslateError(errno);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetDisconnect(AeSocket *pSock)
{
    if (*pSock != -1)
    {
        close(*pSock);
        *pSock = -1;
    }

    return AeEOK;
}

/******************************************************************************/
AeError AeNetSend(AeSocket *pSock, AePointer pData, AeInt32 iLength, AeInt32 *piSent)
{
    int rc;

    rc = send(*pSock, pData, iLength, 0);
    if (rc == -1)
    {
        *piSent = 0;

        return UnixTranslateError(errno);
    }

    *piSent = rc;

    return AeEOK;
}

/******************************************************************************/
AeError AeNetReceive(AeSocket *pSock, AePointer pData, AeInt32 iLength, AeInt32 *piReceived)
{
    int rc;

    rc = recv(*pSock, pData, iLength, 0);
    if (rc == -1)
    {
        *piReceived = 0;

        return UnixTranslateError(errno);
    }

    *piReceived = rc;

    return AeEOK;
}

/******************************************************************************/
AeError AeSelect(AeInt iMaxFD, AeFDArray *pReadFDs, AeFDArray *pWriteFDs, AeFDArray *pExceptFDs, AeTimeValue *pTimeOut)
{
    struct timeval tv;

    tv.tv_sec = pTimeOut->iSec;
    tv.tv_usec = pTimeOut->iMicroSec;

    return select(iMaxFD, pReadFDs, pWriteFDs, pExceptFDs, &tv);
}

/******************************************************************************/
AeError AeNetSetBlocking(AeSocket *pSock, AeBool bBlocking)
{
    unsigned long iCommand;
    int rc;

    if (bBlocking)
        iCommand = 0;
    else
        iCommand = 1;

    rc = ioctl(*pSock, FIONBIO, &iCommand);
    if (rc != 0)
        return UnixTranslateError(errno);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetSetNoDelay(AeSocket *pSock, AeBool bNoDelay)
{
    int iOn = 1;
    int rc;

    rc = setsockopt(*pSock, IPPROTO_TCP, TCP_NODELAY, (char *) &iOn, sizeof(iOn));
    if (rc != 0)
        return UnixTranslateError(errno);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetSetSendBufferSize(AeSocket *pSock, AeInt32 iSize)
{
    int rc;

    rc = setsockopt(*pSock, SOL_SOCKET, SO_SNDBUF, &iSize, sizeof(iSize));
    if (rc != 0)
        return UnixTranslateError(errno);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetGetLastError(void)
{
    return UnixTranslateError(errno);
}

/******************************************************************************/
AeError AeNetGetPendingError(AeSocket *pSock)
{
#ifndef __UC_LIBC__
    socklen_t iLength;
#else
    int iLength;
#endif
    int iError, rc;

    iError = 0;
    iLength = sizeof(iError);
    rc = getsockopt(*pSock, SOL_SOCKET, SO_ERROR, &iError, &iLength);
    if (rc != 0)
        return AeNetGetLastError();

    return UnixTranslateError(iError);
}

/******************************************************************************/
AeError AeNetResolve(AeChar *pHostname, AeUInt32 *piAddress)
{
    struct hostent *pHost;

    pHost = gethostbyname(pHostname);
    if (!pHost)
        return UnixTranslateError(errno);

    *piAddress = *((AeUInt32 *) pHost->h_addr);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetHostName(AeChar *pHostname, AeInt iLength)
{
	/* host name on axis device is this format: axis-00408c9016c8 and NTLM proxy does not like it */
#ifdef AXIS_LINUX
	strcpy(pHostname, "axis");
#else
    int rc;

    rc = gethostname(pHostname, iLength);
    if (rc != 0)
        return UnixTranslateError(errno);
#endif
    return AeEOK;
}

/******************************************************************************/
AeUInt32 AeNetInetAddr(AeChar *pHost)
{
    return inet_addr(pHost);
}

/******************************************************************************/
AeUInt32 AeNetHToNL(AeUInt32 iNumber)
{
    return htonl(iNumber);
}

/******************************************************************************/
AeUInt16 AeNetHToNS(AeUInt16 iNumber)
{
    return htons(iNumber);
}

/******************************************************************************/
void AeMutexInitialize(AeMutex *pMutex)
{
#if defined(_REENTRANT) && defined(HAVE_PTHREADS)
    pthread_mutexattr_t attr;

    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(pMutex, &attr);
    pthread_mutexattr_destroy(&attr);
#endif
}

/******************************************************************************/
void AeMutexDestroy(AeMutex *pMutex)
{
#if defined(_REENTRANT) && defined(HAVE_PTHREADS)
    pthread_mutex_destroy(pMutex);
#endif
}

/******************************************************************************/
void AeMutexLock(AeMutex *pMutex)
{
#if defined(_REENTRANT) && defined(HAVE_PTHREADS)
    pthread_mutex_lock(pMutex);
#endif
}

/******************************************************************************/
void AeMutexUnlock(AeMutex *pMutex)
{
#if defined(_REENTRANT) && defined(HAVE_PTHREADS)
    pthread_mutex_unlock(pMutex);
#endif
}

/******************************************************************************/
void AeGetCurrentTime(AeTimeValue *pTime)
{
    gettimeofday((struct timeval *) pTime, NULL);
}

/******************************************************************************/
AeError AeLogOpen(void)
{
    return AeEOK;
}

/******************************************************************************/
void AeLogClose(void)
{
}

/******************************************************************************/
void AeLog(AeLogRecordType iType, AeChar *pFormat, ...)
{
    va_list vaArgs;

    va_start(vaArgs, pFormat);
    _AeLogEx(iType, AE_LOG_CATEGORY_NONE, pFormat, &vaArgs);
    va_end(vaArgs);
}

/******************************************************************************/
void AeLogEx(AeLogRecordType iType, AeUInt32 iCategory, AeChar *pFormat, ...)
{
    va_list vaArgs;

    va_start(vaArgs, pFormat);
    _AeLogEx(iType, iCategory, pFormat, &vaArgs);
    va_end(vaArgs);
}

/******************************************************************************/
static void _AeLogEx(AeLogRecordType iType, AeUInt32 iCategory, AeChar *pFormat, va_list *pArgs)
{
    time_t iTime;
    char *pFinalFormat, *pEvent, pTimeStr[64], pCategoryStr[192];
    int iFinalFormatSize;

    time(&iTime);
	#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 3) && (__GNUC_PATCHLEVEL__== 1)
		my_strftime(pTimeStr, sizeof(pTimeStr), "%c", localtime(&iTime));
	#else
		strftime(pTimeStr, sizeof(pTimeStr), "%c", localtime(&iTime));
	#endif

    switch (iType)
    {
        case AeLogTrace:
            pEvent = "TRACE  ";
            break;
        case AeLogDebug:
            pEvent = "DEBUG  ";
            break;
        case AeLogInfo:
            pEvent = "INFO   ";
            break;
        case AeLogWarning:
            pEvent = "WARNING";
            break;
        case AeLogError:
            pEvent = "ERROR  ";
            break;
        default:
            return;
    }

    AeGetLogCategoryString(iCategory, pCategoryStr, sizeof(pCategoryStr));
    
    iFinalFormatSize = strlen(pTimeStr) + 1 + strlen(pEvent) + 1 + strlen(pCategoryStr) + 3 + strlen(pFormat) + 2;
    pFinalFormat = AeAlloc(iFinalFormatSize);
    if (!pFinalFormat)
        return;

    strcpy(pFinalFormat, pTimeStr);
    strcat(pFinalFormat, " ");
    strcat(pFinalFormat, pEvent);
    strcat(pFinalFormat, "[");
    strcat(pFinalFormat, pCategoryStr);
    strcat(pFinalFormat, "]: ");
    strcat(pFinalFormat, pFormat);
    strcat(pFinalFormat, "\n");

    vfprintf(stdout, pFinalFormat, *pArgs);

    AeFree(pFinalFormat);
}


/******************************************************************************/
#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 3) && (__GNUC_PATCHLEVEL__== 1)
static size_t my_strftime(char *s, size_t max, const char  *fmt,  const struct tm *tm)
{
	return strftime(s, max, fmt, tm);
}
#endif


/******************************************************************************/
static AeError UnixTranslateError(int iError)
{
    switch (iError)
    {
        case 0:
            return AeEOK;

        case EWOULDBLOCK:
        case EINPROGRESS:
            return AeENetWouldBlock;

        case ECONNREFUSED:
            return AeENetConnRefused;

        case ECONNRESET:
            return AeENetConnReset;

        case EPIPE:
        case ECONNABORTED:
            return AeENetConnAborted;

        case ENOTCONN:
            return AeENetNotConn;

        case ENETUNREACH:
            return AeENetNetUnreach;

        case EHOSTUNREACH:
            return AeENetHostUnreach;

        default:
            return AeENetGeneral;
    }
}

/******************************************************************************/
static void UnixSignalHandlerStub(int iSignal)
{
}

/******************************************************************************/
AeFileHandle AeFileOpen(AeChar *pName, AeUInt32 iFlags)
{
    AeUInt32 iOpenFlags = 0;

    if (iFlags & AE_OPEN_READ_ONLY)
        iOpenFlags = O_RDONLY;

    if (iFlags & AE_OPEN_WRITE_ONLY)
        iOpenFlags = O_WRONLY;

    if (iFlags & AE_OPEN_READ_WRITE)
        iOpenFlags = O_RDWR;

    if (iFlags & AE_OPEN_CREATE)
        iOpenFlags |= O_CREAT;

    if (iFlags & AE_OPEN_TRUNCATE)
        iOpenFlags |= O_TRUNC;

#ifdef ENABLE_LARGEFILE64
    iOpenFlags |= O_LARGEFILE;
#endif

    return open(pName, iOpenFlags, 0644);
}

/******************************************************************************/
static int g_unixWhenceValues[] = { SEEK_SET, SEEK_CUR, SEEK_END };
#define AE_UNIX_WHENCE(x) (g_unixWhenceValues[(x) - 1])

/******************************************************************************/
#ifndef ENABLE_LARGEFILE64
AeInt32	AeFileSeek(AeFileHandle file, AeInt32 iOffset, AeInt32 iWhence)
{
    return lseek(file, iOffset, AE_UNIX_WHENCE(iWhence));
}
#else
AeInt64 AeFileSeek64(AeFileHandle file, AeInt64 iOffset, AeInt32 iWhence)
{
    return lseek64(file, iOffset, AE_UNIX_WHENCE(iWhence));
}
#endif

/******************************************************************************/
AeInt32 AeFileRead(AeFileHandle file, void *pBuf, AeInt32 iSize)
{
	return read(file, pBuf, iSize);
}

/******************************************************************************/
AeInt32	AeFileWrite(AeFileHandle file, void *pBuf, AeInt32 iSize)
{
	return write(file, pBuf, iSize);
}

/******************************************************************************/
AeInt32	AeFileClose(AeFileHandle file)
{
	return close(file);
}

/******************************************************************************/
AeBool AeFileDelete(AeChar *pName)
{
    if (unlink(pName) == 0)
        return AeTrue;

    return AeFalse;
}

/******************************************************************************/
AeBool AeFileExist(AeChar *pName)
{
    struct stat sbuf;

    if (stat(pName, &sbuf) != -1)
        return AeTrue;

    return AeFalse;
}

/******************************************************************************/
#ifndef ENABLE_LARGEFILE64
AeInt32	AeFileGetSize(AeChar *pName)
{
    struct stat statBuf;
   
    if (stat(pName, &statBuf) != 0)
        return -1;

    return statBuf.st_size;
}
#else
AeInt64	AeFileGetSize64(AeChar *pName)
{
    struct stat64 statBuf;
   
    if (stat64(pName, &statBuf) != 0)
        return -1;

    return statBuf.st_size;
}
#endif

/******************************************************************************/
AeBool AeMakeDir(AeChar *pName)
{
    if (mkdir(pName, 0755) == -1 && errno != EEXIST)
        return AeFalse;

    return AeTrue;
}

/******************************************************************************/
void AeSleep(AeTimeValue *pTime)
{
    unsigned long iMicroseconds = AE_TIME_VALUE_MICROSECONDS(*pTime);
    usleep(iMicroseconds);
}

/* provide compatibility with [broken] uc-libc for uClinux */
#ifdef __UC_LIBC__

/******************************************************************************/
/* this function is dangerous: target buffer size limit is not respected */
int snprintf(char *str, size_t size, const char *format, ...)
{
    va_list vaArgs;
    int rc;

    va_start(vaArgs, format);
    rc = vsprintf(str, format, vaArgs);
    va_end(vaArgs);

    return rc;
}

/******************************************************************************/
/* this function is dangerous: target buffer size limit is not respected */
int vsnprintf(char *str, size_t size, const char *format, va_list ap)
{
    return vsprintf(str, format, ap);
}

/******************************************************************************/
void abort(void)
{
    _exit(1);
}

/******************************************************************************/
void *realloc(void *ptr, size_t size)
{
    if (ptr == NULL)
    {
        return (void *) malloc(size);
    }
    if (size <= 0)
    {
        free(ptr);
        ptr = NULL;
    }
    else
    {
        char *orig = ptr;
        ptr = (void *) malloc(size);
        memcpy(ptr, orig, size);
        free(orig);
    }
    return ptr;
}

#endif /* __UC_LIBC__ */
