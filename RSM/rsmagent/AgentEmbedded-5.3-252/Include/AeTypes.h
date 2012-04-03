/* $Id: AeTypes.h,v 1.8 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeTypes.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Axeda Agent Embedded types and macros
 *
 **************************************************************************/

#ifndef _AE_TYPES_H_
#define _AE_TYPES_H_

typedef char            AeChar;
typedef unsigned char   AeUChar;
typedef int             AeInt;
typedef unsigned int    AeUInt;
typedef AeInt           AeBool;
typedef float           AeFloat;
typedef double          AeDouble;
typedef void            *AePointer;
typedef AePointer       AeHandle;

typedef signed char     AeInt8;
typedef unsigned char   AeUInt8;

typedef signed short    AeInt16;
typedef unsigned short  AeUInt16;
typedef signed long     AeInt32;
typedef unsigned long   AeUInt32;

#define AeTrue          1
#define AeFalse         0

typedef struct _AeTimeValue AeTimeValue;
typedef enum _AeWebVersion AeWebVersion;
typedef enum _AeWebProxyProtocol AeWebProxyProtocol;
typedef enum _AeWebCryptoLevel AeWebCryptoLevel;
typedef enum _AeDRMDeviceType AeDRMDeviceType;
typedef enum _AeDRMServerConfigType AeDRMServerConfigType;
typedef enum _AeDRMQueuePriority AeDRMQueuePriority;
typedef enum _AeDRMQueueStatus AeDRMQueueStatus;
typedef enum _AeDRMDataType AeDRMDataType;
typedef enum _AeDRMDataQuality AeDRMDataQuality;
typedef struct _AeDRMDataValue AeDRMDataValue;
typedef struct _AeDRMDataItem AeDRMDataItem;
typedef struct _AeDRMAlarm AeDRMAlarm;
typedef struct _AeDRMEvent AeDRMEvent;
typedef struct _AeDRMEmail AeDRMEmail;
typedef struct _AeDRMSOAPCommandStatus AeDRMSOAPCommandStatus;
typedef struct _AeDRMSOAPCommandId AeDRMSOAPCommandId;
typedef struct _AeDRMPingRate AeDRMPingRate;

typedef enum _AeFileType AeFileType;
typedef enum _AeFileCompression AeFileCompression;
typedef enum _AeFileTransferEvent AeFileTransferEvent;
typedef struct _AeFileStat AeFileStat;
typedef struct _AeFileUploadSpec AeFileUploadSpec;
typedef struct _AeFileUploadRequestParam AeFileUploadRequestParam;
typedef struct _AeFileUploadExecuteParam AeFileUploadExecuteParam;

/* Date/time representation structure. */
struct _AeTimeValue
{
    AeInt32     iSec;               /* seconds since 00:00:00 Jan 2 1970 UTC */
    AeInt32     iMicroSec;          /* microseconds (1/1000000th sec) */
};

/* macros for manipulation with AeTimeValue */
#define AE_TIME_VALUE_SUBTRACT(r, x, y)\
    do {\
        if ((x).iSec >= (y).iSec)\
        {\
            (r).iSec = (x).iSec - (y).iSec;\
            if ((x).iMicroSec >= (y).iMicroSec)\
                (r).iMicroSec = (x).iMicroSec - (y).iMicroSec;\
            else\
            {\
                if ((r).iSec > 0)\
                {\
                    (r).iSec -= 1;\
                    (r).iMicroSec = 1000000L - ((y).iMicroSec - (x).iMicroSec);\
                }\
                else\
                    (r).iMicroSec = 0;\
            }\
        }\
        else\
        {\
            (r).iSec = 0;\
            (r).iMicroSec = 0;\
        }\
    } while (0);

#define AE_TIME_VALUE_ADD(r, x, y)\
    do {\
        (r).iSec = (x).iSec + (y).iSec;\
        (r).iMicroSec = (x).iMicroSec + (y).iMicroSec;\
        if ((r).iMicroSec > 1000000L)\
        {\
            (r).iSec += 1;\
            (r).iMicroSec -= 1000000L;\
        }\
    } while (0);

#define AE_TIME_VALUE_GREATER_THAN(x, y) \
    ((x).iSec > (y).iSec || \
     ((x).iSec == (y).iSec && \
      (x).iMicroSec > (y).iMicroSec))

#define AE_TIME_VALUE_LESS_THAN(x, y) \
    ((x).iSec < (y).iSec || \
     ((x).iSec == (y).iSec && \
      (x).iMicroSec < (y).iMicroSec))

#define AE_TIME_VALUE_IS_NULL(x) \
    ((x).iSec == 0 && \
     (x).iMicroSec == 0)

/* Use these macros only with small time intervals (not with the system
 * clock values!). Overflow is possible. */
#define AE_TIME_VALUE_MILLISECONDS(t) ((t).iSec * 1000 + (t).iMicroSec / 1000)
#define AE_TIME_VALUE_MICROSECONDS(t) ((t).iSec * 1000000 + (t).iMicroSec)

/* HTTP version. */
enum _AeWebVersion
{
    AeWebHTTP10,                    /* HTTP/1.0 */
    AeWebHTTP11                     /* HTTP/1.1 */
};

/* Proxy server protocol. */
enum _AeWebProxyProtocol
{
    AeWebProxyProtoNone,            /* no proxy */
    AeWebProxyProtoSOCKS,           /* SOCKS proxy */
    AeWebProxyProtoHTTP             /* HTTP proxy */
};

/* SSL encryption level. */
enum _AeWebCryptoLevel
{
    AeWebCryptoNone,                /* no encryption (SSL disabled) */
    AeWebCryptoLow,                 /* low-strength cipher */
    AeWebCryptoMedium,              /* medium-strength cipher */
    AeWebCryptoHigh                 /* high-strength cipher */
};

/* Device type. */
enum _AeDRMDeviceType
{
    AeDRMDeviceMaster,              /* master device */
    AeDRMDeviceManaged              /* managed device (Gateway only) */
};

/* Configuration type for the Enterprise server - Primary, Backup, Additional.*/
enum _AeDRMServerConfigType
{
    AeDRMServerConfigPrimary,       /* primary Enterprise server */
    AeDRMServerConfigAdditional,    /* additional Enterprise server */
    AeDRMServerConfigBackup         /* backup Enterprise server */
};

/* Priority of item sent to the data queue - could be a file, data item value, alarm, event.*/
enum _AeDRMQueuePriority
{
    AeDRMQueuePriorityLow,          /* low priority */
    AeDRMQueuePriorityNormal,       /* normal priority */
    AeDRMQueuePriorityUrgent        /* high priority */
};

/* Status of the data queue. */
enum _AeDRMQueueStatus
{
    AeDRMQueueEmpty,                /* the queue is empty */
    AeDRMQueueNonEmpty,             /* the queue is non-empty */
    AeDRMQueueFull                  /* the queue is full */
};

/* Type of data item value. */
enum _AeDRMDataType
{
    AeDRMDataAnalog,                /* analog */
    AeDRMDataDigital,               /* digital (0/1) */
    AeDRMDataString                 /* string */
};

/* Quality of the data item value. */
enum _AeDRMDataQuality
{
    AeDRMDataGood,                  /* good */
    AeDRMDataBad,                   /* bad */
    AeDRMDataUncertain              /* uncertain (undefined) */
};

/* Structure for a data item value. */
struct _AeDRMDataValue
{
    AeDRMDataType       iType;      /* type */
    AeDRMDataQuality    iQuality;   /* quality */
    AeTimeValue         timeStamp;  /* timestamp */
    union
    {
#ifndef __UC_LIBC__
        AeDouble    dAnalog;        /* analog value */
#else
        AeFloat     dAnalog;        /* analog value */
#endif
        AeBool      bDigital;       /* digital value */
        AeChar      *pString;       /* string value */
    } data;
};

/* Structure for a data item. */
struct _AeDRMDataItem
{
    AeChar          *pName;         /* name */
    AeDRMDataValue  value;          /* value */
};

/* Structure for an Alarm description. */
struct _AeDRMAlarm
{
    AeChar          *pName;         /* name */
    AeChar          *pDescription;  /* textual description */
    AeChar          *pCondition;    /* condition: LoLo, Lo, Hi, HiHi */
    AeTimeValue     timeStamp;      /* timestamp */
    AeInt           iSeverity;      /* severity */
    AeBool          bActive;        /* is the alarm active? */
    AeBool          bAck;           /* has the alarm been acknowledged? */
    AeDRMDataItem   *pDataItem;     /* associated data item (may be NULL) */
};

/* Structure of an Event description. */
struct _AeDRMEvent
{
    AeChar      *pName;             /* name */
    AeChar      *pMessage;          /* textual message */
    AeTimeValue timeStamp;          /* timestamp */
    AeInt       iSeverity;          /* severity */
};

/* Structure for an E-mail description. */
struct _AeDRMEmail
{
    AeChar  *pRecipients;           /* comma-separated list of recipients */
    AeChar  *pSenderName;           /* sender name */
    AeChar  *pSenderAddress;        /* sender e-mail address */
    AeChar  *pSubject;              /* subject */
    AeChar  *pServerHost;           /* SMTP server host name or IP address */
    AeChar  *pContentType;          /* MIME content type */
    AeChar  *pBody;                 /* message body */
};

/* Structure for SOAP command execution status (result). */
struct _AeDRMSOAPCommandStatus
{
    AeUInt32    iStatusCode;        /* numeric status code */
    AeChar      *pStatusReason;     /* textual description */
};

/* Structure for SOAP command identification. */
struct _AeDRMSOAPCommandId
{
    AeChar  *pCmdId;                /* command id */
    AeChar  *pUserId;               /* id of user who submitted command */
};

/* Structure for Ping Rate description. */
struct _AeDRMPingRate
{
    AeTimeValue rate;               /* ping rate value */
    AeTimeValue *pDuration;         /* time period when rate effective */
};

/* "standard" SOAP command status values. */
#define AE_DRM_SOAP_COMMAND_STATUS_SUCCESS             0
#define AE_DRM_SOAP_COMMAND_STATUS_FAILED              0xC0000002L
#define AE_DRM_SOAP_COMMAND_STATUS_INVALID_PARAMS      0xC0000003L
#define AE_DRM_SOAP_COMMAND_STATUS_NOT_IMPLEMENTED     0xC0000009L
#define AE_DRM_SOAP_COMMAND_STATUS_DEFERRED            0xF0000001L

/* File types. */
enum _AeFileType
{
    AeFileTypeUnknown,
    AeFileTypeRegular,
    AeFileTypeDirectory
};

/* Type of compression for file transfer (None disables compression.)*/
enum _AeFileCompression
{
    AeFileCompressionNone,
    AeFileCompressionTarGzip
};

/* Structure for File description. 
   The structure is used to communicate file attributes from Agent Embedded
   to the application during file download. It is also used to communicate 
   the attributes from the application to Agent Embedded during file upload. 
   When ENABLE_LARGEFILE64 is defined, the file size limitions are as follows.
   Compressed transfers are limited to 8 GB. Uncompressed transfers have a
   theoretical limit of 9223372036854775807 bytes. Your C compiler must support
   64-bit integer to use AeInt64. In addition, if files are transferred to or
   from a real file system, the file system and the OS must support 64-bit-based
   file operations.*/
struct _AeFileStat
{
    AeChar     *pName;
    AeFileType iType;
#ifndef ENABLE_LARGEFILE64
    AeInt32    iSize;
#else
    AeInt64    iSize;
#endif
    AeInt32    iMTime;                /* date and time that file was last modified*/

};

/* File transfer event. */
enum _AeFileTransferEvent
{
    AeFileTransferCancelled,
    AeFileTransferPaused,
    AeFileTransferPreempted,
    AeFileTransferReactivated
};

/* File upload specification structure.
   The application uses this structure to request a file upload (one
   structure for each file). In this case, Agent Embedded ignores 
   the iPosition field. This structure is also passed from Agent
   Embedded to the application (one structure per file) when the upload
   command is received from the Enterprise server. In this case, if it
   intends to provide file data, the application must start with the file 
   position indicated by iPosition. */
struct _AeFileUploadSpec
{
    AeChar   *pName;
    AeBool   bDelete;
#ifndef ENABLE_LARGEFILE64
    AeInt32  iPosition;
#else
    AeInt64  iPosition;
#endif
};

/* File upload request parameters.
   This structure is used to communicate additional upload parameters from
   the application to Agent Embedded when the former requests a file upload.
   The parameters apply to the entire upload operation rather than to
   individual files. When iCompression specifies AeFileCompressionNone,
   only one file can be requested in the upload. pId specifies a string
   that is to be passed back to the application when the upload is
   executed, i.e., when the upload command is received from the Enterprise server.
   pHint specifies a string to be associated with the upload for
   customized processing on the server side. iPriority specifies the
   priority of associated file transfer, which is to be compared with
   priorities of other pending file transfers. iCompression specifies
   the compression type for the upload. iMask specifies which parameters
   were initialized. */
struct _AeFileUploadRequestParam
{
    AeUInt32            iMask;
    AeChar              *pId;
    AeChar              *pHint;
    AeInt               iPriority;
    AeFileCompression   iCompression;
};

#define AE_FILE_UPLOAD_REQUEST_PARAM_ID             0x00000001
#define AE_FILE_UPLOAD_REQUEST_PARAM_HINT           0x00000002
#define AE_FILE_UPLOAD_REQUEST_PARAM_PRIORITY       0x00000004
#define AE_FILE_UPLOAD_REQUEST_PARAM_COMPRESSION    0x00000008

/* File upload execution parameters.
   This structure is used to communicate additional upload parameters from
   Agent Embedded to the application when the upload is executed, i.e.
   when the upload command is received from the Enterprise server. pId specifies
   the same string that has been passed from the application to the server
   when upload was requested (pId is not set if the upload was initiated
   by the server). iMask specifies which parameters were initialized. */
struct _AeFileUploadExecuteParam
{
    AeUInt32    iMask;
    AeChar      *pId;
};

#define AE_FILE_UPLOAD_EXECUTE_PARAM_ID 0x00000001

#endif
