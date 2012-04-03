/* $Id: AeWebRequest.h,v 1.5.2.1 2009/06/16 20:52:27 hfeng Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeWebRequest.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Declarations for HTTP request manipulation
 *
 **************************************************************************/

#ifndef _AE_WEB_REQUEST_H_
#define _AE_WEB_REQUEST_H_

#define MIME_DEFAULT        "application/octet-stream"

#define HTTP_VERSION_10     "1.0"
#define HTTP_VERSION_11     "1.1"

#define HTTP_DEFAULT_PORT   80
#define HTTPS_DEFAULT_PORT  443

#define HTTP_METHOD_OPTIONS "OPTIONS"
#define HTTP_METHOD_GET     "GET"
#define HTTP_METHOD_HEAD    "HEAD"
#define HTTP_METHOD_POST    "POST"
#define HTTP_METHOD_PUT     "PUT"
#define HTTP_METHOD_DELETE  "DELETE"
#define HTTP_METHOD_CONNECT "CONNECT"

#define HTTP_STATUS_CONTINUE        100
#define HTTP_STATUS_OK              200
#define HTTP_STATUS_UNAUTHORIZED    401
#define HTTP_STATUS_PROXYAUTHREQ    407
#define HTTP_STATUS_REQUEST_TIMEOUT 408
#define HTTP_SERVICE_UNAVAILABLE    503 

typedef struct _AeWebProxyInfo AeWebProxyInfo;
typedef struct _AeWebRequest AeWebRequest;

/* HTTP Proxy structure */
struct _AeWebProxyInfo
{
    AeWebProxyProtocol  iProto;
    AeChar              *pHost;
    AeUInt16            iPort;
    AeChar              *pUser;
    AeChar              *pPassword;
};

/* HTTP request structure (should not be filled directly). */
struct _AeWebRequest
{
    AePointer   pURL;
    AeChar      *pVersion;
    AeChar      *pMethod;
    AeChar      *pHost;
    AeUInt16    iPort;
    AeChar      *pAbsPath;
    AeChar      *pUser;
    AeChar      *pPassword;
    AeChar      *pContentType;
    AeChar      *pEntityData;
    AeInt32     iEntitySize;
    AeBool      bPersistent;
    AeBool      bStrict;
    AeBool      bSecure;
    AeTimeValue timeOut;
    AePointer   pRequestHeaders;
    AePointer   pResponseHeaders;
    AeInt       iResponseStatus;
    AeError     iError;
    AePointer   pUserData;
	AeWebProxyInfo*   pProxyInfo;

    void        (*pOnError)(AeWebRequest *pRequest, AeError iError);
    AeBool      (*pOnResponse)(AeWebRequest *pRequest, AeInt iStatusCode);
    AeBool      (*pOnEntity)(AeWebRequest *pRequest, AeInt32 iDataOffset, AeChar *pData, AeInt32 iSize);
    void        (*pOnCompleted)(AeWebRequest *pRequest);
};

#ifdef __cplusplus
extern "C" {
#endif

/* Creates new HTTP request object. */
AeWebRequest    *AeWebRequestNew(void);

/* Destroys HTTP request object. */
void            AeWebRequestDestroy(AeWebRequest *pRequest);

/* Parses the specified string as URL and modifies corresponding properties of
   the request (secure, username, password, hostname, port, absolute path). */
AeBool          AeWebRequestSetURL(AeWebRequest *pRequest, AeChar *pString);

/* Composes a URL string based on the corresponding properties of the request
   (secure, username, password, hostname, port, absolute path). */
AeChar          *AeWebRequestGetURL(AeWebRequest *pRequest);

/* Sets HTTP version. */
void            AeWebRequestSetVersion(AeWebRequest *pRequest, AeChar *pVersion);

/* Sets HTTP request method. */
void            AeWebRequestSetMethod(AeWebRequest *pRequest, AeChar *pMethod);

/* Sets origin server hostname or IP address. */
void            AeWebRequestSetHost(AeWebRequest *pRequest, AeChar *pHost);

/* Sets origin server port number. */
#define         AeWebRequestSetPort(r, x) ((r)->iPort = (x))

/* Sets absolute path. */
void            AeWebRequestSetAbsPath(AeWebRequest *pRequest, AeChar *pAbsPath);

/* Sets username for the origin server authentication. */
void            AeWebRequestSetUser(AeWebRequest *pRequest, AeChar *pUser);

/* Sets password for the origin server authentication. */
void            AeWebRequestSetPassword(AeWebRequest *pRequest, AeChar *pPassword);

/* Sets MIME content type of the request body. */
void            AeWebRequestSetContentType(AeWebRequest *pRequest, AeChar *pContentType);

/* Sets pointer to a buffer for the request body. */
#define         AeWebRequestSetEntityData(r, x) ((r)->pEntityData = (x))

/* Sets size of the request body. */
#define         AeWebRequestSetEntitySize(r, x) ((r)->iEntitySize = (x))

/* Enables/disables persistent connection. */
#define         AeWebRequestSetPersistent(r, x) ((r)->bPersistent = (x))

/* Enables/disables strict mode (in strict mode the request is processed in
   strict accordance with the specified HTTP protocol version). */
#define         AeWebRequestSetStrict(r, x) ((r)->bStrict = (x))

/* Enables/disables secure request (SSL). */
#define         AeWebRequestSetSecure(r, x) ((r)->bSecure = (x))

/* Sets timeout for communications triggered by the request. */
#define         AeWebRequestSetTimeOut(r, x) ((r)->timeOut = *(x))

/* Sets user data. This may be used in the callbacks. */
#define         AeWebRequestSetUserData(r, x) ((r)->pUserData = (x))

/* Sets pProxyInfo to be used with this WebRequest */
#define         AeWebRequestSetProxyInfo(r, x) ((r)->pProxyInfo = (x))

/* Adds custom request header (name - value pair). */
void            AeWebRequestSetRequestHeader(AeWebRequest *pRequest, AeChar *pName, AeChar *pValue);

/* Adds custom response header (name - value pair). */
void            AeWebRequestSetResponseHeader(AeWebRequest *pRequest, AeChar *pName, AeChar *pValue);

/* Retrieves response header value by name. */
AeChar          *AeWebRequestGetResponseHeader(AeWebRequest *pRequest, AeChar *pName);

/* Retrieves the first response header (name - value pair). This function also
   returns a pointer, which may be used in calls to AeWebRequestGetNextResponseHeader() */
AePointer       AeWebRequestGetFirstResponseHeader(AeWebRequest *pRequest, AeChar **ppName, AeChar **ppValue);

/* Retrieves next response header (name - value pair). This function also
   returns a pointer, which may be used in the subsequent calls. */
AePointer       AeWebRequestGetNextResponseHeader(AeWebRequest *pRequest, AePointer pPosition, AeChar **ppName, AeChar **ppValue);

/* Installs callback invoked on asynchronous communication error.
   Callback prototype:
   void (*)(AeWebRequest *pRequest, AeError iError);
   iError indicates the error code. */
#define         AeWebRequestSetOnError(r, x) ((r)->pOnError = (x))

/* Installs callback invoked on response header receipt.
   Callback prototype:
   AeBool (*)(AeWebRequest *pRequest, AeInt iStatusCode);
   iStatusCode indicates numeric HTTP status. The callback function should
   return false, if it is required to abort the request, or true otherwise. */
#define         AeWebRequestSetOnResponse(r, x) ((r)->pOnResponse = (x))

/* Installs callback invoked on receipt of a chunk of the response body.
   Callback prototype:
   AeBool (*)(AeWebRequest *pRequest, AeInt32 iDataOffset, AeChar *pData, AeInt32 iSize);
   pData points to the chunk buffer, iSize indicates size of the chunk, and
   iDataOffset specifies offset of the chunk from the beginning of the response
   body. The callback function should return false, if it is required to abort
   the request, or true otherwise. */
#define         AeWebRequestSetOnEntity(r, x) ((r)->pOnEntity = (x))

/* Installs callback invoked on successful request completetion.
   Callback prototype:
   void (*)(AeWebRequest *pRequest);
   */
#define         AeWebRequestSetOnCompleted(r, x) ((r)->pOnCompleted = (x))

/* Resets request to the initial state */
void            AeWebRequestClearStatus(AeWebRequest *pRequest);

#ifdef __cplusplus
}
#endif

#endif
