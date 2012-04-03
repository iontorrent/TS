/* $Id: AeOS.h,v 1.20.2.1 2009/12/01 18:05:01 hfeng Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeOS.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  System-dependent function declarations
 *
 **************************************************************************/
#ifndef _AE_OS_H_
#define _AE_OS_H_

/* macros to convert between little endian and host byte order */
#ifdef AE_BIG_ENDIAN
#define AeNetHToLEL(x) \
     ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) | \
      (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))
#define AeNetHToLES(x) \
     ((((x) & 0xff00) >> 8) | (((x) & 0x00ff) <<  8))
#else
#define AeNetHToLEL(x) (x)
#define AeNetHToLES(x) (x)
#endif /* AE_BIG_ENDIAN */
#define AeNetLEToHL AeNetHToLEL
#define AeNetLEToHS AeNetHToLES

typedef struct _AeNetAddress AeNetAddress;
typedef enum _AeLogRecordType AeLogRecordType;
typedef AeLogRecordType AeLogLevel;

/* IP address and port number in the network byte order. */
struct _AeNetAddress
{
    AeUInt32    iAddress;
    AeUInt16    iPort;
};

/* Log message type */
enum _AeLogRecordType
{
    AeLogNone,
    AeLogError,
    AeLogWarning,
    AeLogInfo,
    AeLogDebug,
    AeLogTrace
};

/* Log message category values (may be OR-ed) */
#define AE_LOG_CATEGORY_NONE            0
#define AE_LOG_CATEGORY_NETWORK         0x00000001
#define AE_LOG_CATEGORY_SERVER_STATUS   0x00000002
#define AE_LOG_CATEGORY_DATA_QUEUE      0x00000004
#define AE_LOG_CATEGORY_REMOTE_SESSION  0x00000008
#define AE_LOG_CATEGORY_FILE_TRANSFER   0x00000010
#define AE_LOG_CATEGORY_FILE_UPLOAD     0x00000020
#define AE_LOG_CATEGORY_FILE_DOWNLOAD   0x00000040

#ifdef __cplusplus
extern "C" {
#endif

/* Initializes network environment. */
AeError     AeNetInitialize(void);

/* Creates socket object. bStream specifies whether the socket is
   connection-oriented (true) or not (false). */
AeError     AeNetGetSocket(AeSocket *pSock, AeBool bStream);

/* Connects socket (pSock) to remote service specified by network address
   (pAddress). */
AeError     AeNetConnect(AeSocket *pSock, AeNetAddress *pAddress);

/* Disconnects connected socket (pSock) */
AeError     AeNetDisconnect(AeSocket *pSock);

/* Sends iLength bytes from buffer (pData) over the socket pSock. The socket
   must be connected before calling. The actual number of bytes sent is stored in
   an integer pointed to by piSent. */
AeError     AeNetSend(AeSocket *pSock, AePointer pData, AeInt32 iLength, AeInt32 *piSent);

/* Receives up to iLength bytes into the buffer (pData) from the socket pSock. The
   socket must be connected before calling. The actual number of bytes received is
   stored in an integer pointed to by piReceived. */
AeError     AeNetReceive(AeSocket *pSock, AePointer pData, AeInt32 iLength, AeInt32 *piReceived);

/* Waits for a number of file descriptors to change status. Three independent
   sets of descriptors are watched. Those listed in pReadFDs will be watched to
   see if characters become available for reading (receiving). Those in
   pWriteFDs will be watched to see if a write (send) will not block, and those
   in pExceptFDs will be watched for exceptions. On return, the sets are
   modified in place to indicate which descriptors actually changed status.
   iMaxFD is the highest numbered descriptor in any of three sets, plus 1.
   pTimeOut is an upper bound on the amount of time elapsed before the function
   returns. It may be zero, causing AeSelect() to return immediately. If
   pTimeOut is NULL (no timeout), AeSelect() can block indefinitely. */
AeError     AeSelect(AeInt iMaxFD, AeFDArray *pReadFDs, AeFDArray *pWriteFDs, AeFDArray *pExceptFDs, AeTimeValue *pTimeOut);

/* Sets blocking/non-blocking mode (bBlocking is true/false) for socket pSock. */
AeError     AeNetSetBlocking(AeSocket *pSock, AeBool bBlocking);

/* Disables/enables Nagle's algorithm (bNoDelay is true/false) for socket pSock. */
AeError     AeNetSetNoDelay(AeSocket *pSock, AeBool bNoDelay);

/* Sets maximum send buffer size for socket pSock to iSize. */
AeError     AeNetSetSendBufferSize(AeSocket *pSock, AeInt32 iSize);

/* Returns last error returned by the OS. */
AeError     AeNetGetLastError(void);

/* Returns a pending error for socket pSock. */
AeError     AeNetGetPendingError(AeSocket *pSock);

/* Resolves the host name specified by pHostname to the 32-bit IP address (pointed to by
   piAddress) in the network byte order. */
AeError     AeNetResolve(AeChar *pHostname, AeUInt32 *piAddress);

/* Retrieves the host name of the local machine to a buffer pointed to by pHostname. The
   size of the buffer is specified by iLength. */
AeError     AeNetHostName(AeChar *pHostname, AeInt iLength);

/* Returns the 32-bit IP address in the network byte order based on a string
   (pHost) containing dotted IP address. */
AeUInt32    AeNetInetAddr(AeChar *pHost);

/* Converts a 32-bit unsigned integer to the network byte order. */
AeUInt32    AeNetHToNL(AeUInt32 iNumber);

/* Converts a 16-bit unsigned integer to the network byte order. */
AeUInt16    AeNetHToNS(AeUInt16 iNumber);


/* Initializes the mutex object. */
void        AeMutexInitialize(AeMutex *pMutex);

/* Destroys the mutex object. */
void        AeMutexDestroy(AeMutex *pMutex);

/* Locks the mutex object. */
void        AeMutexLock(AeMutex *pMutex);

/* Unlocks the mutex object. */
void        AeMutexUnlock(AeMutex *pMutex);


/* Retrieves the current date/time with a microsecond resolution. */
void        AeGetCurrentTime(AeTimeValue *pTime);


/* Opens (initilizes) logging facility. */
AeError     AeLogOpen(void);

/* Closes (shuts down) logging facility. */
void        AeLogClose(void);

/* Logs a message of the specified type. */
void        AeLog(AeLogRecordType iType, AeChar *pFormat, ...);

/* Logs a message of the specified type and category. */
void        AeLogEx(AeLogRecordType iType, AeUInt32 iCategory, AeChar *pFormat, ...);

/* Some platforms don't have srand() or rand() */
void        AeSRand();
double      AeRand();

/* file operations support */
#define AE_OPEN_READ_ONLY	0x0001		/* opens the file for reading */
#define AE_OPEN_WRITE_ONLY	0x0002		/* opens the file for writing */
#define AE_OPEN_READ_WRITE	0x0004		/* opens the file for reading and writing */

#define AE_OPEN_CREATE		0x0100		/* opens or creates the file, preserve existing content */
#define AE_OPEN_TRUNCATE	0x0200		/* opens or creates the file, truncates the content if it exists*/

#define AE_SEEK_SET		1				/* Beginning + iOffset */
#define AE_SEEK_CUR		2				/* Current + iOffset */
#define AE_SEEK_END		3				/* End + iOffset */

/* Open the named file in the specified mode, returns handle or NULL*/
AeFileHandle	AeFileOpen(AeChar* name, AeUInt32 openFlags);

/* Sets the file position */
#ifndef ENABLE_LARGEFILE64
AeInt32			AeFileSeek(AeFileHandle file, AeInt32 iOffset, AeInt32 iWhence);
#else
AeInt64			AeFileSeek64(AeFileHandle file, AeInt64 iOffset, AeInt32 iWhence);
#endif

/* Reads the contents from the file, returns the bytes read */
AeInt32			AeFileRead(AeFileHandle file, void *pBuf, AeInt32 iSize);

/* Writes the contents to the file, returns the bytes written */
AeInt32			AeFileWrite(AeFileHandle file, void *pBuf, AeInt32 iSize);

/* Closes the file */
AeInt32			AeFileClose(AeFileHandle file);

/* Removes the file */
AeBool			AeFileDelete(AeChar *pName);

/* Checks whether the file exists */
AeBool			AeFileExist(AeChar *pName);

/* Returns the size of the file */
#ifndef ENABLE_LARGEFILE64
AeInt32			AeFileGetSize(AeChar *pName);
#else
AeInt64			AeFileGetSize64(AeChar *pName);
#endif

/* Creates the directory. */
AeBool          AeMakeDir(AeChar *pName);

/* Tells application to sleep for specified amount of time (used when
   AeDRMSetYieldOnIdle is set to false) */
void            AeSleep(AeTimeValue *pTime);

#ifdef __cplusplus
}
#endif

#endif
