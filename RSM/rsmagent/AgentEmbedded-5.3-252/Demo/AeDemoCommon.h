/* $Id: AeDemoCommon.h,v 1.6 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation.  All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDemoCommon.c
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Common declarations for demo applications
 *
 **************************************************************************/

#ifndef _AE_DEMO_COMMON_H_
#define _AE_DEMO_COMMON_H_

#define DEFAULT_OWNER "drm-data_source"

typedef struct _AeDemoConfig AeDemoConfig;

struct _AeDemoConfig
{
    AeChar              *pOwner;
    AeBool              bSecure;
    AeBool              bDebug;
    AeWebProxyProtocol  iProxyProto;
    AeChar              *pProxyHost;
    AeChar              *pProxyUser;
    AeChar              *pProxyPassword;
    AeInt16             iProxyPort;

    AeChar              *pRemoteSessionName;
    AeChar              *pRemoteSessionType;
    AeChar              *pRemoteSessionHost;
    AeInt16             iRemoteSessionPort;

	AeChar				*pUploadFileName;
	AeBool				initiateUploadOnly;
	AeInt64				iSimSize;
	AeChar				*pFileHintName;
	AeChar				*pFileClientID;
	AeInt16				iFilePriority;


	AeChar				*pUploadFileName2;
	AeInt64				iSimSize2;
	AeChar				*pFileHintName2;
	AeChar				*pFileClientID2;
	AeChar				iFilePriority2;
	AeBool				compress;

	AeBool				exitAfterFirstFile;
	AeUInt32			ipingRate;
	AeUInt32			interDataDelay;

	AeChar				*pSerialNumber;
	AeChar				*pModelNumber;

	AeBool				bSendAlarm;
	AeBool				bAlarmAck;
	AeBool				bPersist;


};

#ifdef __cplusplus
extern "C" {
#endif

void    AeDemoUsage(char *pProgName, char *pNonOpts);
AeBool  AeDemoProcessOptions(AeDemoConfig *pConfig, int *pargc, char *argv[], int iNonOptCount);
AeBool  AeDemoApplyConfig(AeDemoConfig *pConfig);

#ifdef __cplusplus
}
#endif

#endif
