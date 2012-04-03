/* $Id: AeOS.c,v 1.24.2.1 2009/12/01 18:08:47 hfeng Exp $ */

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"

static AeError Win32TranslateError(int iError);
static void _AeLogEx(AeLogRecordType iType, AeUInt32 iCategory, AeChar *pFormat, va_list *pArgs);

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
    static AeBool bInitialized = AeFalse;
    
    if (!bInitialized)
    {
        WSADATA wsaData;
        int rc;

        memset(&wsaData, 0, sizeof(wsaData));
        rc = WSAStartup(MAKEWORD(1, 1), &wsaData);
        if (rc != 0)
            return Win32TranslateError(WSAGetLastError());
    }

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
    if ((*pSock) == INVALID_SOCKET)
        return Win32TranslateError(WSAGetLastError());

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
        return Win32TranslateError(WSAGetLastError());

    return AeEOK;
}

/******************************************************************************/
AeError AeNetDisconnect(AeSocket *pSock)
{
    if (*pSock != INVALID_SOCKET)
    {
        closesocket(*pSock);
        *pSock = INVALID_SOCKET;
    }

    return AeEOK;
}

/******************************************************************************/
AeError AeNetSend(AeSocket *pSock, AePointer pData, AeInt32 iLength, AeInt32 *piSent)
{
    int rc;

    rc = send(*pSock, pData, iLength, 0);
    if (rc == SOCKET_ERROR)
    {
        *piSent = 0;

        return Win32TranslateError(WSAGetLastError());
    }

    *piSent = rc;

    return AeEOK;
}

/******************************************************************************/
AeError AeNetReceive(AeSocket *pSock, AePointer pData, AeInt32 iLength, AeInt32 *piReceived)
{
    int rc;

    rc = recv(*pSock, pData, iLength, 0);
    if (rc == SOCKET_ERROR)
    {
        *piReceived = 0;

        return Win32TranslateError(WSAGetLastError());
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
    u_long iCommand;
    int rc;

    if (bBlocking)
        iCommand = 0;
    else
        iCommand = 1;

    rc = ioctlsocket(*pSock, FIONBIO, &iCommand);
    if (rc != 0)
        return Win32TranslateError(WSAGetLastError());

    return AeEOK;
}

/******************************************************************************/
AeError AeNetSetNoDelay(AeSocket *pSock, AeBool bNoDelay)
{
    int iOn = 1;
    int rc;

    rc = setsockopt(*pSock, IPPROTO_TCP, TCP_NODELAY, (char *) &iOn, sizeof(iOn));
    if (rc != 0)
        return Win32TranslateError(WSAGetLastError());

    return AeEOK;
}

/******************************************************************************/
AeError AeNetSetSendBufferSize(AeSocket *pSock, AeInt32 iSize)
{
    int rc;

    rc = setsockopt(*pSock, SOL_SOCKET, SO_SNDBUF, (char *) &iSize, sizeof(iSize));
    if (rc != 0)
        return Win32TranslateError(WSAGetLastError());

    return AeEOK;
}

/******************************************************************************/
AeError AeNetGetLastError(void)
{
    return Win32TranslateError(WSAGetLastError());
}

/******************************************************************************/
AeError AeNetGetPendingError(AeSocket *pSock)
{
    int iError, iLength, rc;

    iError = 0;
    iLength = sizeof(iError);
    rc = getsockopt(*pSock, SOL_SOCKET, SO_ERROR, (char *) &iError, &iLength);
    if (rc != 0)
        return AeNetGetLastError();

    return Win32TranslateError(iError);
}

/******************************************************************************/
AeError AeNetResolve(AeChar *pHostname, AeUInt32 *piAddress)
{
    struct hostent *pHost;

    pHost = gethostbyname(pHostname);
    if (!pHost)
        return Win32TranslateError(WSAGetLastError());

    *piAddress = *((AeUInt32 *) pHost->h_addr);

    return AeEOK;
}

/******************************************************************************/
AeError AeNetHostName(AeChar *pHostname, AeInt iLength)
{
    int rc;

    rc = gethostname(pHostname, iLength);
    if (rc != 0)
        return Win32TranslateError(WSAGetLastError());

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
    *pMutex = CreateMutex(NULL, FALSE, NULL);
}

/******************************************************************************/
void AeMutexDestroy(AeMutex *pMutex)
{
    CloseHandle(*pMutex);
}

/******************************************************************************/
void AeMutexLock(AeMutex *pMutex)
{
    WaitForSingleObject(*pMutex, INFINITE);
}

/******************************************************************************/
void AeMutexUnlock(AeMutex *pMutex)
{
    ReleaseMutex(*pMutex);
}

/******************************************************************************/
void AeGetCurrentTime(AeTimeValue *pTime)
{
    struct _timeb timeBuf;

    _ftime(&timeBuf);
    pTime->iSec = timeBuf.time;
    pTime->iMicroSec = timeBuf.millitm * 1000;
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
    strftime(pTimeStr, sizeof(pTimeStr), "%c", localtime(&iTime));

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
static AeError Win32TranslateError(int iError)
{
    switch (iError)
    {
        case 0:
            return AeEOK;

        case WSAEWOULDBLOCK:
        case WSAEINPROGRESS:
            return AeENetWouldBlock;

        case WSAECONNREFUSED:
            return AeENetConnRefused;

        case WSAECONNRESET:
            return AeENetConnReset;

        case WSAECONNABORTED:
            return AeENetConnAborted;

        case WSAENOTCONN:
            return AeENetNotConn;

        case WSAENETUNREACH:
            return AeENetNetUnreach;

        case WSAETIMEDOUT:
        case WSAEHOSTUNREACH:
            return AeENetHostUnreach;

        default:
            return AeENetGeneral;
    }
}


/******************************************************************************/
AeFileHandle AeFileOpen(AeChar *pName, AeUInt32 iFlags)
{
    AeUInt32 iReadWriteFlags = 0;
    AeUInt32 iCreationFlags  = OPEN_EXISTING;

    if (iFlags & AE_OPEN_READ_ONLY)
        iReadWriteFlags = GENERIC_READ;

    if (iFlags & AE_OPEN_WRITE_ONLY)
        iReadWriteFlags = GENERIC_WRITE;

    if (iFlags & AE_OPEN_READ_WRITE)
        iReadWriteFlags = GENERIC_READ | GENERIC_WRITE;

    if (iFlags & AE_OPEN_CREATE)
        iCreationFlags = OPEN_ALWAYS;

	/* This overwrites the above */
    if (iFlags & AE_OPEN_TRUNCATE)
        iCreationFlags = CREATE_ALWAYS;

	/* return valid HANDLE or INVALID_HANDLE_VALUE */
	return CreateFile(pName, iReadWriteFlags , FILE_SHARE_WRITE | FILE_SHARE_READ, 
        NULL, iCreationFlags, FILE_ATTRIBUTE_NORMAL, 0);
}

/******************************************************************************/
static DWORD g_win32WhenceValues[] = { FILE_BEGIN, FILE_CURRENT, FILE_END };
#define AE_WIN32_WHENCE(x) (g_win32WhenceValues[(x) - 1])

/******************************************************************************/
#ifndef ENABLE_LARGEFILE64
AeInt32	AeFileSeek(AeFileHandle file, AeInt32 iOffset, AeInt32 iWhence)
{
    return SetFilePointer(file, iOffset, NULL, AE_WIN32_WHENCE(iWhence));
}
#else
AeInt64 AeFileSeek64(AeFileHandle file, AeInt64 iOffset, AeInt32 iWhence)
{
    LONG iLo, iHi, *pHi;
    DWORD rc;
    AeInt64 iResult;

    if (iOffset >= LONG_MIN && iOffset <= LONG_MAX)
    {
        iLo = (LONG) iOffset;
        pHi = NULL;
    }
    else
    {
        iLo = (LONG) (iOffset & 0xffffffff);
        iHi = (LONG) (iOffset >> 32);
        pHi = &iHi;
    }
    
    rc = SetFilePointer(file, iLo, pHi, AE_WIN32_WHENCE(iWhence));
    if (rc == (DWORD) -1)
        return -1;
    
    iResult = ((AeInt64) rc) & 0xffffffff;
    if (pHi)
        iResult |= ((AeInt64) *pHi) << 32;
    
    return iResult;
}
#endif

/******************************************************************************/
AeInt32 AeFileRead(AeFileHandle file, void *pBuf, AeInt32 iSize)
{
    AeInt32 iSizeRead = 0;

    if (ReadFile(file, pBuf, iSize, &iSizeRead, NULL) == FALSE)
        return -1;

    return iSizeRead;
}

/******************************************************************************/
AeInt32	AeFileWrite(AeFileHandle file, void *pBuf, AeInt32 iSize)
{
    AeInt32 iSizeWritten = 0;

    if (WriteFile(file, pBuf, iSize, &iSizeWritten, NULL) == FALSE)
        return -1;

    return iSizeWritten;
}

/******************************************************************************/
AeInt32	AeFileClose(AeFileHandle file)
{
    if (CloseHandle(file) == TRUE)
        return 0;

    return -1;
}

/******************************************************************************/
AeBool AeFileDelete(AeChar *pName)
{
    if (DeleteFile(pName) == TRUE)
        return AeTrue;
    
    return AeFalse;
}

/******************************************************************************/
AeBool AeFileExist(AeChar *pName)
{
    if (GetFileAttributes(pName) != (DWORD) -1)
        return AeTrue;

    return AeFalse;
}

/******************************************************************************/
#ifndef ENABLE_LARGEFILE64
AeInt32	AeFileGetSize(AeChar *pName)
{
    WIN32_FILE_ATTRIBUTE_DATA attrData;

    if (GetFileAttributesEx(pName, GetFileExInfoStandard, &attrData) == FALSE)
        return -1;

    return (AeInt32) attrData.nFileSizeLow;
}
#else
AeInt64	AeFileGetSize64(AeChar *pName)
{
    WIN32_FILE_ATTRIBUTE_DATA attrData;
    AeInt64 iResult;

    if (GetFileAttributesEx(pName, GetFileExInfoStandard, &attrData) == FALSE)
        return -1;

    iResult = (((AeInt64) attrData.nFileSizeHigh) << 32) |
        (((AeInt64) attrData.nFileSizeLow) & 0xffffffff);

    return iResult;
}
#endif

/******************************************************************************/
AeBool AeMakeDir(AeChar *pName)
{
    if (!CreateDirectory(pName, NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        return AeFalse;

    return AeTrue;
}

/******************************************************************************/
void AeSleep(AeTimeValue *pTime)
{
    DWORD iMilliseconds = AE_TIME_VALUE_MILLISECONDS(*pTime);
    Sleep(iMilliseconds);
}
