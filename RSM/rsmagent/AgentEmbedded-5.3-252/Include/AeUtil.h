/* $Id: AeUtil.h,v 1.1.2.3 2009/05/29 14:44:18 hfeng Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2009 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeUtil.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Axeda Agent Embedded utility functions
 *
 **************************************************************************/


#ifndef _AE_UTIL_H_
#define _AE_UTIL_H_

/* Functions to log debug message. agentembedded no longer uses this function */
#define AE_LOG (*g_pLogFunc)

/* Functions to log debug message. agentembedded now uses this function to log any messages */
/* Use AeSetLogExFunc() call to install application log functions. see AeInterface.h */
#define AE_LOG_EX (*g_pLogExFunc)

/* max-min utility macros */
#define AE_MIN(x, y) ((x) < (y) ? (x) : (y))
#define AE_MAX(x, y) ((x) > (y) ? (x) : (y))

/* agentembedded intermal macros */
#ifndef ENABLE_LARGEFILE64
#define AE_LONGINT_FORMAT_MODIFIER "l"
#else
#define AE_LONGINT_FORMAT_MODIFIER AE_INT64_FORMAT_MODIFIER
#endif

/* some structures to handle filters */
typedef struct _AeFilterHandler AeFilterHandler;
typedef struct _AeFileFilterHandler AeFileFilterHandler;
typedef struct _AeURL AeURL;

/* structure to handle filters */
struct _AeFilterHandler
{
    void    *(*pOpen)(AeBool bEncode);
    void    (*pClose)(void *pFilter);
    void    (*pReset)(void *pFilter);
    AeInt32 (*pWrite)(void *pFilter, AeChar *pData, AeInt32 iSize);
    AeInt32 (*pRead)(void *pFilter, AeChar *pData, AeInt32 iSize);
    AeBool  (*pIsEnd)(void *pFilter);
    AeInt32 (*pGetError)(void *pFilter);
    AeChar  *(*pGetErrorString)(AeInt32 iError);
};

/* structure to handle file filters */
struct _AeFileFilterHandler
{
    void    (*pReset)(void *pFilter);
    void    (*pDestroy)(void *pFilter);
    AeBool  (*pIsEndOfStream)(void *pFilter);
    AeBool  (*pIsEndOfFile)(void *pFilter);
    AeInt32 (*pWriteRaw)(void *pFilter, AeChar *pData, AeInt32 iSize);
    AeBool  (*pReadFile)(void *pFilter, AeFileStat **ppFile);
    AeInt32 (*pReadFileData)(void *pFilter, AeChar **ppData);
    AeInt32 (*pReadRaw)(void *pFilter, AeChar *pData, AeInt32 iSize);
    AeBool  (*pWriteFile)(void *pFilter, AeFileStat *pFile);
    AeInt32 (*pWriteFileData)(void *pFilter, AeChar *pData, AeInt32 iSize);
    AeInt32 (*pGetError)(void *pFilter);
    AeChar  *(*pGetErrorString)(void *pFilter, AeInt32 iError);
};

/* structure to URL */
struct _AeURL
{
    AeBool      bSecure;
    AeChar      *pHost;
    AeUInt16    iPort;
    AeChar      *pAbsPath;
    AeChar      *pUser;
    AeChar      *pPassword;
    AeBuffer    *pBuffer;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Sets a string value by allocating memory and sets the pointer to *ppString */
/* If pValue is NULL, it just frees the *ppString memory */
/* It is recommended that *ppString points to NULL when calling into this function */
/* Applications can free this pointer by calling AeFree(*ppString) or AeSetString(ppString, NULL, -1) */
void    AeSetString(AeChar **ppString, AeChar *pValue, AeInt iLength);

/* Removes the specified character from the end of string, as many occurrences as there are there. */
void    AeTrimRight(AeChar *pString, AeChar cCh);

/* Copies the src AeFileStat to dst AeFileStat. Use AeFileStatDestroy() to free AeFileStat pointers */
void    AeFileStatCopy(AeFileStat *pDst, AeFileStat *pSrc);

/* Frees memory of AeFileStat pointer and nested memory inside */
void    AeFileStatDestroy(AeFileStat *pFile);

/* Base64 encode the input. Returns the encoded string in pOutput. Also returns the pOutput size */
/* Applcations are responsibable to alloc enough memory into pOutput ((iLength + 2) / 3 * 4 + 1) */
AeInt32 AeBase64Encode(AeChar *pInput, AeInt32 iLength, AeChar *pOutput);

/* Base64 decode the input. Returns the decoded string in pOutput. Also returns the pOutput size */
/* Applcations are responsibable to alloc enough memory into pOutput (iLength) */
AeInt32 AeBase64Decode(AeChar *pInput, AeInt32 iLength, AeChar *pOutput);

/* Convert the bytes' numerical value into string representations. Examples: 0XFF to "ff", 0X1A to "1a", etc. */
/* pBin is array of input bytes, iLength is number of bytes, pHex is converted output */
/* Applcations are responsibable to alloc enough memory into pHex (iLength * 2 + 1) */
void    AeBinToHex(AeUInt8 *pBin, AeInt32 iLength, AeChar *pHex);

/* Convert the time to string format. time is seconds since beginning of 1970 */
/* Use AeGetCurrentTime() to get the current system time. See AeOS.h */
/* Applcations are responsibable to alloc enough memory into pOutput (64) */
void    AeGetISO8601Time(AeTimeValue *pTime, AeChar *pOutput, AeInt iMaxLen);

/* Creates a new AeURL. AeURL deals with protocol communications (like http) to some remote host */
/* Normally, applications can use AeWebRequest to perform these tasks. No need to deal with AeURL directly */
/* See AeWebRequest.h for details on how to use AeWebRequest */
AeURL   *AeURLNew(void);

/* Destroys the AeURL and all its nested memory objects */
void    AeURLDestroy(AeURL *pURL);

/* Sets secure comminication, using SSL */
#define AeURLSetSecure(u, x) ((u)->bSecure = (x))

/* Sets the remote host */
void    AeURLSetHost(AeURL *pURL, AeChar *pHost);

/* Sets the remote host port */
#define AeURLSetPort(u, x) ((u)->iPort = (x))

/* Sets the absolute path of the remote host site */
void    AeURLSetAbsPath(AeURL *pURL, AeChar *pAbsPath);

/* Sets the user for the remote site */
void    AeURLSetUser(AeURL *pURL, AeChar *pUser);

/* Sets the password for the remote site */
void    AeURLSetPassword(AeURL *pURL, AeChar *pPassword);

/* Sets the URL for the remote site. */
/* pString can be something like: http://192.168.1.1:80@user:pass/abs-path */
/* AeURLSet will pick them apart and set to each member accordingly */
AeBool  AeURLSet(AeURL *pURL, AeChar *pString);

/* Gets the URL for the remote site. This is opposite of AeURLGet(). It constructs the whole from various members */
AeChar  *AeURLGet(AeURL *pURL);

/* Coverts the path delimiter according to the current platform. that is '/' or '\' */
AeBool AeConvertPath(AeChar *pPath);

/* Recursively creates the specified directory chain */
AeBool AeMakeDirHierarchy(AeChar *pPath);

#ifdef __cplusplus
}
#endif

#endif
