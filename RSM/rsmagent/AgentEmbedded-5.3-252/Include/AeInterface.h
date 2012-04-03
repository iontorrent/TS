/* $Id: AeInterface.h,v 1.21 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Systems. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeInterface.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Interface function declarations
 *
 **************************************************************************/
#ifndef _AE_INTERFACE_H_
#define _AE_INTERFACE_H_

#include "AeWebRequest.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initializes Agent Embedded. This function must be called before any other
   call to Agent Embedded. 
*/
AeError     AeInitialize(void);

/* Shuts down Axeda Agent Embedded. This function should be called 
   to release system resources when the application stops using Agent Embedded. 
*/ 
void        AeShutdown(void);

/* Installs a custom log function (default: AeLog). When pLogFunc is NULL, the
   default log function is set. This function is kept for backwards
   compatibility only. Agent Embedded does not use this type of log function
   any more. See AeSetLogExFunc(). 
*/
void        AeSetLogFunc(void (*pLogFunc)(AeLogRecordType iType, AeChar *pFormat, ...));

/* Installs a custom extended log function (default: AeLogEx). When pLogExFunc
   is NULL, the default log function is set. 
*/
void        AeSetLogExFunc(void (*pLogExFunc)(AeLogRecordType iType, AeUInt32 iCategory, AeChar *pFormat, ...));

/* DRM: Configures the maximum memory size for the internal data queue. The data queue
        is used to accumulate device data submitted through AeDRMPostXXX() functions
        before it is delivered to the specified Enterprise server. iSize parameter 
        specifies the maximum memory size of the data queue, in bytes. 
*/
void        AeDRMSetQueueSize(AeInt32 iSize);

/* DRM: Configures the retry period for communication failures (default: 30
        seconds). If a failure is detected during communication with the Enterprise server,
        Agent Embedded stops exchanging data with the server for the specified period
        of time (pointed to by pPeriod) and then retries. 
*/
void        AeDRMSetRetryPeriod(AeTimeValue *pPeriod);

/* DRM: Configures the timestamp mode. By default Agent Embedded uses the system time of
        the machine where it is running to generate timestamps for the messages sent to Enterprise      
        server (Local timestamp mode). To override this mode, set Server timestamp
        mode by passing AeTrue to this function. In Server timestamp mode,
        the Enterprise server uses its own system time to process the messages. 
*/
void        AeDRMSetTimeStampMode(AeBool bServerMode);

/* DRM: Configures the maximum log level (default: AeLogInfo). Agent Embedded logs
        messages through the calls to AeLog(). Only messages with an iType less than or equal
        to the specified iLevel are logged. 
*/
void        AeDRMSetLogLevel(AeLogLevel iLevel);

/* DRM: Enables/disables debug messages (default: disabled).
        AeDRMSetDebug(AeTrue) is an equivalent of AeDRMSetLogLevel(AeLogDebug);
        AeDRMSetDebug(AeFalse) is an equivalent of AeDRMSetLogLevel(AeLogNone); 
*/
void        AeDRMSetDebug(AeBool bDebug);

/* DRM: Specifies whether AeDRMExecute() should return once it finds no pending
        tasks to carry out. The default behavior is to stay inside the function until
        the specified amount of time elapses. AeDRMExecute() calls AeSleep() to consume
        unused time. When AeDRMExecute() is configured to return on entering the idle
        state, the application is responsible for avoiding a possible tight loop by
        inserting brief delays between calls to AeDRMExecute(). 
*/
void        AeDRMSetYieldOnIdle(AeBool bYieldOnIdle);

/* DRM: Installs the callback invoked on an asynchronous communication error. When a
        communication error occurs while Agent Embedded is performing pending tasks in
        AeDRMExecute(), the callback installed by this function is invoked. The
        callback function accepts the id of the Enterprise server that caused the error
        (iServerId), and the error code (iError). 
*/
void        AeDRMSetOnWebError(void (*pCallback)(AeInt32 iServerId, AeError iError));

/* DRM: Installs the callback invoked on device registration. The callback
        installed by this function is invoked when a device is successfully
        registered while in AeDRMExecute(). The callback function accepts the id 
        of the registered device (iDeviceId). 
*/
void        AeDRMSetOnDeviceRegistered(void (*pCallback)(AeInt32 iDeviceId));

/* DRM: Installs the callback invoked on a change in queue status. Whenever the status
        of the internal data queue is changed among "Empty", "Non-Empty" and
        "Full", the callback installed by this function is invoked. The callback
        function accepts the new queue status (iNewStatus). 
*/
void        AeDRMSetOnQueueStatus(void (*pCallback)(AeDRMQueueStatus iNewStatus));

/* DRM: Installs the callback invoked on receipt of a generic SOAP method. When a SOAP
        method received from Enterprise server is not known (not handled
        by any of callbacks installed through AeDRMSetOnCommandXXX()), the callback
        installed by this function is invoked. The callback function accepts the id of 
        the target device (iDeviceId), a SOAP method handle (pMethod) and a pointer to the
        structure for invocation results (pStatus). The received SOAP method may be
        processed by the application using the functions in the AeDRMSOAPXXX() family. 
*/
void        AeDRMSetOnSOAPMethod(void (*pCallback)(AeInt32 iDeviceId, AeHandle pMethod, AeDRMSOAPCommandStatus *pStatus));

/* DRM: Similar to AeDRMSetOnSOAPMethod(), but the callback additionally
        accepts the id of the Enterprise server (iServerId) from which the method
        originated, as well as the method id structure (pSOAPId). The extra
        parameters may be used to submit invocation results asynchronously,
        i.e., using AeDRMPostSOAPCommandStatus(). To indicate that the method is
        executed asynchronously, the callback must return
        AE_DRM_SOAP_COMMAND_STATUS_DEFERRED in pStatus->iStatusCode. 
*/
void        AeDRMSetOnSOAPMethodEx(void (*pCallback)(AeInt32 iDeviceId, AeInt32 iServerId, AeHandle pMethod, AeDRMSOAPCommandId *pSOAPId, AeDRMSOAPCommandStatus *pStatus));

/* DRM: Installs the callback invoked on receipt of the "Set Tag" ("DynamicData.SetTag")
        SOAP method. (A "Tag" is a data item.) The callback function accepts the id of the target   
        device (iDeviceId), the name and new value of the data item (pDataItem), and a pointer to the 
        structure for invocation results (pStatus). 
*/
void        AeDRMSetOnCommandSetTag(void (*pCallback)(AeInt32 iDeviceId, AeDRMDataItem *pDataItem, AeDRMSOAPCommandStatus *pStatus));

/* DRM: Installs the callback invoked on the receipt of the "Set Time" ("EEnterpriseProxy.St") 
        SOAP method. The callback function accepts the id of the target device (iDeviceId),
        a new time value (pTime), a new time zone offset (piTZOffset: minutes from GMT),
        and a pointer to the structure for invocation results (pStatus). pTime or
        piTZOffset may be NULL if the value is not changed. 
*/
void        AeDRMSetOnCommandSetTime(void (*pCallback)(AeInt32 iDeviceId, AeTimeValue *pTime, AeInt32 *piTZOffset, AeDRMSOAPCommandStatus *pStatus));

/* DRM: Installs the callback invoked on receipt of the "Restart" ("EEnterpriseProxy.Rs") SOAP
        method. The callback function accepts the id of the target device (iDeviceId), a
        flag indicating whether hard or soft restart is requested (bHard), and
        a pointer to the structure for invocation results (pStatus). 
*/
void        AeDRMSetOnCommandRestart(void (*pCallback)(AeInt32 iDeviceId, AeBool bHard, AeDRMSOAPCommandStatus *pStatus));

/* DRM: Installs the callback invoked on receipt of the "Set Ping Rate" ("EEnterpriseProxy.Pu")
        SOAP method. The callback is also invoked by AgentEmbedded when the ping rate is restored
        to the original value after the period for a temporary change to the ping rate elapses. 
        The callback function accepts the id of the Enterprise server (iServerId) and the new ping 
        rate (pPingRate). pPingRate->pDuration specifies the time period during which the indicated
        ping rate will be effective. When the period is unspecified, pPingRate->pDuration is NULL. 
*/
void        AeDRMSetOnPingRateUpdate(void (*pCallback)(AeInt32 iServerId, AeDRMPingRate *pPingRate));

/* DRM: Installs the callback invoked at the start of a download of a series
        of one or more files. The callback function accepts the id of the target 
        device (iDeviceId). The callback must return AeTrue if the user application
        is going to perform custom processing on the downloaded files. In this
        case, Agent Embedded will pass incoming file data to the application
        through the callback installed with AeDRMSetOnFileDownloadData(). The
        callback must return AeFalse if Agent Embedded should process
        downloaded files internally (i.e., store them in the local file
        system). The callback may optionally fill the output parameter
        (ppUserData) with a pointer to application's structure associated
        with the file download. Agent Embedded will use this pointer in
        subsequent invocations of callbacks installed with
        AeDRMSetOnFileDownloadData() and AeDRMSetOnFileDownloadEnd(). 
*/
void        AeDRMSetOnFileDownloadBegin(AeBool (*pCallback)(AeInt32 iDeviceId, AePointer *ppUserData));

/* DRM: Installs the callback invoked when Agent Embedded receives a
        portion of a downloaded file. The callback function accepts the id
        of the target device (iDeviceId), a file descriptor (pFile), a block
        of received file data (pData and iSize), and the pointer previously 
        returned by the application through the callback installed with
        AeDRMSetOnFileDownloadBegin(). When the callback is invoked with a
        NULL block (pData and iSize are zero), this indicates the last
        block of the file described by pFile. The callback indicates data
        processing status by returning AeTrue in the case of success, or
        AeFalse otherwise. 
*/
#ifndef ENABLE_LARGEFILE64
void        AeDRMSetOnFileDownloadData(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileStat *pFile, AeChar *pData, AeInt32 iSize, AePointer pUserData));
#else
void        AeDRMSetOnFileDownloadData64(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileStat *pFile, AeChar *pData, AeInt32 iSize, AePointer pUserData));
#define     AeDRMSetOnFileDownloadData AeDRMSetOnFileDownloadData64
#endif

/* DRM: Installs the callback invoked when Agent Embedded either
        encounters an error during the file download or receives all
        files in the series successfully. The error or success of the
        download is indicated by the boolean bOK parameter. The callback
        function also accepts the id of the target device (iDeviceId) and 
        the pointer previously returned by the application through the callback
        installed with AeDRMSetOnFileDownloadBegin(). 
*/
void        AeDRMSetOnFileDownloadEnd(void (*pCallback)(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData));

/* DRM: Installs the callback invoked on receipt of the "File Upload" 
        SOAP method. The callback function accepts the id of the target 
        device (iDeviceId) and a NULL-terminated array of the uploaded file 
        specifications (ppUploads). The callback must return AeTrue if the 
        application is going to provide data for uploaded files. In this
        case, Agent Embedded will request the content of the files to be uploaded
        from the application via the callback installed with AeDRMSetOnFileUploadData().
        The callback must return AeFalse if Agent Embedded should process the
        files internally (i.e., retrieve them from the local file system). 
        The callback may optionally fill the output parameter (ppUserData) 
        with a pointer to the application's structure associated
        with the file upload. Agent Embedded will use this pointer in
        subsequent invocations of callbacks installed with
        AeDRMSetOnFileUploadData(), AeDRMSetOnFileUploadEnd() and
        AeDRMSetOnFileTransferEvent(). 
*/
#ifndef ENABLE_LARGEFILE64
void        AeDRMSetOnFileUploadBegin(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData));
#else
void        AeDRMSetOnFileUploadBegin64(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData));
#define     AeDRMSetOnFileUploadBegin AeDRMSetOnFileUploadBegin64
#endif

/* DRM: Similar to AeDRMSetOnFileUploadBegin(), but the callback accepts
        additional upload parameters via pParam. 
*/
#ifndef ENABLE_LARGEFILE64
void        AeDRMSetOnFileUploadBeginEx(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AeFileUploadExecuteParam *pParam, AePointer *ppUserData));
#else
void        AeDRMSetOnFileUploadBeginEx64(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AeFileUploadExecuteParam *pParam, AePointer *ppUserData));
#define     AeDRMSetOnFileUploadBeginEx AeDRMSetOnFileUploadBeginEx64
#endif

/* DRM: Installs the callback invoked when Agent Embedded requires more
        file data for the upload. The callback function accepts the id of the 
        target device (iDeviceId) and the pointer previously returned by 
        the application through the callback installed with
        AeDRMSetOnFileUploadBegin(): pUserData. The application is responsible
        for filling ppFile with a pointer to an appropriate file descriptor;
        ppData and piSize must be filled with pointers to a buffer and to the
        buffer size, respectively. The buffer should contain the next block of
        data to be included in the upload. The end of the upload should be
        indicated by filling ppFile with a NULL pointer. Note that the file
        descriptor and the buffer returned by the application must remain intact
        at least until this callback (or the one installed with
        AeDRMSetOnFileUploadEnd(), or with AeDRMSetOnFileTransferEvent()) is
        invoked again for the same upload. The application is responsible
        for memory management of these objects.  The callback indicates data
        availability status by returning AeTrue in the case of success, or
        AeFalse otherwise. A NULL block (both *ppData and *piSize are zero)
        returned by the application will cause Agent Embedded to defer file
        transfer processing until the next call to AeDRMExecute(), at which time
        the callback will be invoked again. 
*/
#ifndef ENABLE_LARGEFILE64
void        AeDRMSetOnFileUploadData(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData));
#else
void        AeDRMSetOnFileUploadData64(AeBool (*pCallback)(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData));
#define     AeDRMSetOnFileUploadData AeDRMSetOnFileUploadData64
#endif

/* DRM: Installs the callback invoked when Agent Embedded either
        encounters an error during the file upload or uploads all files   
        successfully. The error or success of the upload is indicated by
        the Boolean bOK parameter. The callback function also accepts the id
        of the target device (iDeviceId) and the pointer previously returned by
        the application through the callback installed with
        AeDRMSetOnFileUploadBegin(). 
*/
void        AeDRMSetOnFileUploadEnd(void (*pCallback)(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData));

/* DRM: Installs the callback invoked when Agent Embedded encounters an event
        during the file transfer. The event type is indicated in iEvent. The
        callback function also accepts the id of the target device (iDeviceId) and the
        pointer previously returned by the application through the callback
        installed with AeDRMSetOnFileDownloadBegin() or
        AeDRMSetOnFileUploadBegin(). 
*/
void        AeDRMSetOnFileTransferEvent(void (*pCallback)(AeInt32 iDeviceId, AeFileTransferEvent iEvent, AePointer pUserData));

/* DRM: Adds the specified device to the configuration. iType specifies the type of the
        device (master or managed). Model number and serial number of the device are
        specified with pModelNumber and pSerialNumber. Assigned device id is an
        output parameter (piId). 
*/
AeError     AeDRMAddDevice(AeDRMDeviceType iType, AeChar *pModelNumber, AeChar *pSerialNumber, AeInt32 *piId);

/* DRM: Adds the destination Enterprise server to the configuration. The server is
        specified by 

        iType:        the configuration type: primary, additional or backup
        pURL:         the URL. Specifying "https://" at the beginning of the URL enables secure
                      communication through SSL. 
        pOwner:       the name of target database
        pPingRate     the ping rate (the interval between attempts to contact the Enterprise server)
   
        The assigned server id is an output parameter (piId). 
*/
AeError     AeDRMAddServer(AeDRMServerConfigType iType, AeChar *pURL, AeChar *pOwner, AeTimeValue *pPingRate, AeInt32 *piId);

/* DRM: Adds a remote session to Agent Embedded.
        iDeviceId:    device id
        pName:        name of the session (unique among all sessions)
        pDescription: description of the session
        pType:        type of session, supported types are "telnet",
                      "browser", "auto" or "manual"
        pServer:      destination server host for the session, for
	                  example, telnet server or web server
        iPort:        destination server port number, for example, 23
                      (telnet) or 80 (HTTP).
*/
AeError     AeDRMAddRemoteSession(AeInt32 iDeviceId, AeChar *pName, AeChar *pDescription, AeChar *pType, AeChar *pServer, AeUInt16 iPort);

/* DRM: Submits the specified data item for delivery. The data item (pDataItem) will be posted to
        the Enterprise server (iServerId) based on priority iPriority. The source device is specified 
        by iDeviceId. 
*/
AeError     AeDRMPostDataItem(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeDRMDataItem *pDataItem);

/* DRM: Submits the specified alarm for delivery. The alarm (pAlarm) will be posted to the
        Enterprise server (iServerId) based on priority iPriority. The source device is specified
        by iDeviceId. 
*/
AeError     AeDRMPostAlarm(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeDRMAlarm *pAlarm);

/* DRM: Submits the specified event for delivery. The event (pEvent) will be posted to the 
        Enterprise server (iServerId) based on priority iPriority. The source device is 
        specified by iDeviceId. 
*/
AeError     AeDRMPostEvent(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeDRMEvent *pEvent);

/* DRM: Submits the specified e-mail for delivery. The e-mail (pEmail) will be posted to the
        Enterprise server (iServerId) based on priority iPriority. The source device is 
        specified by iDeviceId. 
*/
AeError     AeDRMPostEmail(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeDRMEmail *pEmail);

/* DRM: Submits the specified file upload request for delivery. The file upload request
        described by NULL-terminated array of uploaded file specifications (ppUploads) 
        will be posted to the Enterprise server (iServerId) based on priority iPriority. 
        The source device is specified by iDeviceId. 
*/
#ifndef ENABLE_LARGEFILE64
AeError     AeDRMPostFileUploadRequest(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeFileUploadSpec **ppUploads);
#else
AeError     AeDRMPostFileUploadRequest64(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeFileUploadSpec **ppUploads);
#define     AeDRMPostFileUploadRequest AeDRMPostFileUploadRequest64
#endif

/* DRM: Submits the specified SOAP method processing status for delivery. The status
        (pStatus) for command identified by pSOAPId will be posted to the Enterprise server 
        (iServerId) based on priority iPriority. The source device is specified by iDeviceId. 
*/
AeError     AeDRMPostSOAPCommandStatus(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeDRMSOAPCommandId *pSOAPId, AeDRMSOAPCommandStatus *pStatus);

/* DRM: Similar to AeDRMPostFileUploadRequest(), but specifies additional upload
        parameters in pParam. 
*/
#ifndef ENABLE_LARGEFILE64
AeError     AeDRMPostFileUploadRequestEx(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeFileUploadSpec **ppUploads, AeFileUploadRequestParam *pParam);
#else
AeError     AeDRMPostFileUploadRequestEx64(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeFileUploadSpec **ppUploads, AeFileUploadRequestParam *pParam);
#define     AeDRMPostFileUploadRequestEx AeDRMPostFileUploadRequestEx64
#endif

/* DRM: Submits opaque data for delivery. The data (pData) will be posted to the Enterprise
        server (iServerId) based on priority iPriority. The source device is specified by
        iDeviceId. The application is responsible for formatting XML data correctly. 
*/
AeError     AeDRMPostOpaque(AeInt32 iDeviceId, AeInt32 iServerId, AeDRMQueuePriority iPriority, AeChar *pData);

/* DRM: Performs pending tasks: if pTimeLimit is NULL, the operation is synchronous, 
        i.e., the function blocks until all of the pending tasks are completed.
        If pTimeLimit is non-NULL, the function returns when either pTimeLimit 
        elapses or the tasks are completed, whichever comes first. 
*/
AeError     AeDRMExecute(AeTimeValue *pTimeLimit);

/* DRM: Disables or enables device processing. At any time the application 
        can enable/disable (bEnable) the processing for device iDeviceId. 
*/
AeError     AeDRMSetDeviceStatus(AeInt32 iDeviceId, AeBool bEnable);

/* Web layer: Configures the HTTP version: HTTP/1.1 or HTTP/1.0 (default: HTTP/1.1).
              The specified HTTP version (iVersion) will be used for communication with
              the specified Enterprise server (iServerId). 
*/
AeError     AeWebSetVersion(AeInt32 iServerId, AeWebVersion iVersion);

/* Web layer: Enables/disables HTTP/1.1 persistent connection usage (default:
              disabled). Persistent connection will be enabled/disabled (bPersistent)
              when communicating with the specified Enterprise server (iServerId). 
*/
AeError     AeWebSetPersistent(AeInt32 iServerId, AeBool bPersistent);

/* Web layer: Configures the communication timeout (default: 30 seconds). When in
              AeDRMExecute() no activity is detected within amount of time specified by
              pTimeOut in communication with the specified Enterprise server (iServerId),
              the connection is considered timed out, and the corresponding synchronous error 
              is reported. 
*/
AeError     AeWebSetTimeout(AeInt32 iServerId, AeTimeValue *pTimeOut);

/* Web layer: Configures the proxy server. Proxy server is specified by the protocol
              (iProto: SOCKS or HTTP), host name (pHost), and port number (iPort).
              If authentication is required for the proxy, username (pUser) and password
              (pPassword) should be non-NULL. 
*/
AeError     AeWebSetProxy(AeWebProxyProtocol iProto, AeChar *pHost, AeUInt16 iPort, AeChar *pUser, AeChar *pPassword);

/* Web layer: Configures SSL. The encryption level (low, medium or high) is
              specified by iLevel. If bServerAuth is true, Agent Embedded will try to
              validate the server certificate. In this case, pCACertFile should specify
              a trusted CA chain file. 
*/
AeError     AeWebSetSSL(AeWebCryptoLevel iLevel, AeBool bServerAuth, AeChar *pCACertFile);

/* Web layer: Synchronously executes a raw HTTP request. The HTTP request is
              specified by pRequest. 
*/
AeError     AeWebSyncExecute(AeWebRequest *pRequest);

/* Web layer: Asynchronously executes a raw HTTP request. The HTTP request is
              specified by pRequest. This function should be called by the application in
              a loop until the completion of the request is indicated (Boolean pointed to
              by pbComplete is set to true). When the function is called the first time for
              request, the handle pointed to by ppHandle must be NULL. AeWebAsyncExecute() 
              fills out the handle when it returns. To complete the request, subsequent calls
              should pass this returned handle. Each call to AeWebAsyncExecute() may be
              limited by specifying the amount of time in pTimeLimit. If pTimeLimit is
              non-NULL, the function returns when either pTimeLimit elapses or the request
              is completed, whichever comes first. 
*/
AeError     AeWebAsyncExecute(AeHandle *ppHandle, AeWebRequest *pRequest, AeTimeValue *pTimeLimit, AeBool *pbComplete);

/* Returns textual error description. The function returns a pointer to a static
   buffer containing textual description of an error corresponding to iError. 
*/
AeChar      *AeGetErrorString(AeError iError);

/* Stores log category description in provided character buffer (pBuffer),
   which should be at least iMaxLength long.
*/
void        AeGetLogCategoryString(AeUInt32 iCategory, AeChar *pBuffer, AeInt32 iMaxLength);

#ifdef __cplusplus
}
#endif

#endif
