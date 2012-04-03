/* $Id: AeDemoCommon.c,v 1.10 2008/05/21 18:24:36 dkhodos Exp $ */

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
 *  Description:  Common module for demo applications
 *
 **************************************************************************/

#ifdef WIN32
#if(_MSC_VER >= 1300)
	#define _CRT_SECURE_NO_DEPRECATE  // re:  Secure Template Overloads see MSDN
    #define _USE_32BIT_TIME_T       // otherwise time_t is 64 bits in .2005
#endif
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"
#include "AeDemoCommon.h"

/******************************************************************************/
void AeDemoUsage(char *pProgName, char *pNonOpts)
{
    fprintf(stderr, "Usage: %s [-o owner] [-s] [-g] [{-ph|-ps} proxy-host [-pp proxy-port] [-pu proxy-user] [-pw proxy-password]] [-rsname remote-session-name [-rstype remote-session-type] -rshost remote-session-host -rsport remote-session-port] %s\n",
        pProgName, pNonOpts ? pNonOpts : "");
}

/******************************************************************************/
AeBool AeDemoProcessOptions(AeDemoConfig *pConfig, int *pargc, char *argv[], int iNonOptCount)
{
    int i;

    /* set defaults */
    pConfig->pOwner = DEFAULT_OWNER;
    pConfig->bSecure = AeFalse;
    pConfig->bDebug = AeFalse;
    pConfig->iProxyProto = AeWebProxyProtoNone;
    pConfig->pProxyHost = NULL;
    pConfig->iProxyPort = 0;
    pConfig->pProxyUser = NULL;
    pConfig->pProxyPassword = NULL;

    pConfig->pRemoteSessionName = NULL;
    pConfig->pRemoteSessionType = "auto";
    pConfig->pRemoteSessionHost = NULL;
    pConfig->iRemoteSessionPort = 0;

	pConfig->pUploadFileName = NULL;
	pConfig->initiateUploadOnly = AeFalse;
	pConfig->iSimSize = 0;
	pConfig->pFileHintName = NULL;
	pConfig->pFileClientID = NULL;
	pConfig->iFilePriority = 5;
	pConfig->interDataDelay = 0;
	pConfig->compress = AeFalse;

	pConfig->pSerialNumber = NULL;
	pConfig->pModelNumber = NULL;
	
	pConfig->exitAfterFirstFile = AeTrue;

	pConfig->bSendAlarm = AeFalse;
	pConfig->bAlarmAck = AeFalse;
	pConfig->bPersist = AeTrue;


    /* process options/arguments */
    for (i = 1; i < *pargc; i++)
    {
        if (!strcmp(argv[i], "-o") && (++i < *pargc))
            pConfig->pOwner = argv[i];
        else if (!strcmp(argv[i], "-s"))
            pConfig->bSecure = AeTrue;
        else if (!strcmp(argv[i], "-g"))
            pConfig->bDebug = AeTrue;
        else if (!strcmp(argv[i], "-ph") && (++i < *pargc))
        {
            pConfig->pProxyHost = argv[i];
            pConfig->iProxyProto = AeWebProxyProtoHTTP;
            pConfig->iProxyPort = 80;
        }
        else if (!strcmp(argv[i], "-ps") && (++i < *pargc))
        {
            pConfig->pProxyHost = argv[i];
            pConfig->iProxyProto = AeWebProxyProtoSOCKS;
            pConfig->iProxyPort = 1080;
        }
        else if (!strcmp(argv[i], "-pp") && (++i < *pargc))
            pConfig->iProxyPort = (AeInt16) strtol(argv[i], NULL, 10);
        else if (!strcmp(argv[i], "-pu") && (++i < *pargc))
            pConfig->pProxyUser = argv[i];
        else if (!strcmp(argv[i], "-pw") && (++i < *pargc))
            pConfig->pProxyPassword = argv[i];

        else if (!strcmp(argv[i], "-rsname") && (++i < *pargc))
            pConfig->pRemoteSessionName = argv[i];
        else if (!strcmp(argv[i], "-rstype") && (++i < *pargc))
            pConfig->pRemoteSessionType = argv[i];
        else if (!strcmp(argv[i], "-rshost") && (++i < *pargc))
            pConfig->pRemoteSessionHost = argv[i];
        else if (!strcmp(argv[i], "-rsport") && (++i < *pargc))
            pConfig->iRemoteSessionPort = (AeInt16) strtol(argv[i], NULL, 10);
        else if (!strcmp(argv[i], "-f") && (++i < *pargc))
            pConfig->pUploadFileName = argv[i];
        else if (!strcmp(argv[i], "-initonly"))
			pConfig->initiateUploadOnly = AeTrue;
		else if (!strcmp(argv[i], "-sim") && (++i < *pargc))
            pConfig->iSimSize = (AeUInt32) strtoul(argv[i], NULL, 10);
		else if (!strcmp(argv[i], "-hint") && (++i < *pargc))
			pConfig->pFileHintName = argv[i];
		else if (!strcmp(argv[i], "-clientid") && (++i < *pargc))
			pConfig->pFileClientID = argv[i];
		else if (!strcmp(argv[i], "-pri") && (++i < *pargc))
			pConfig->iFilePriority = (AeInt16) strtol(argv[i], NULL, 10);
		else if (!strcmp(argv[i], "-stay")) 
			pConfig->exitAfterFirstFile = AeFalse;
		else if (!strcmp(argv[i], "-pingrate") && (++i < *pargc))
			pConfig->ipingRate = (AeUInt32) strtol(argv[i], NULL, 10);
		else if (!strcmp(argv[i], "-delay") && (++i < *pargc))
			pConfig->interDataDelay = (AeUInt32) strtol(argv[i], NULL, 10);
		else if (!strcmp(argv[i], "-compress"))
			pConfig->compress = AeTrue;
		else if (!strcmp(argv[i], "-sn") && (++i < *pargc))
			pConfig->pSerialNumber = argv[i];
		else if (!strcmp(argv[i], "-mn") && (++i < *pargc))
			pConfig->pModelNumber = argv[i];
		else if (!strcmp(argv[i], "-sendalarm"))
			pConfig->bSendAlarm = AeTrue;
		else if (!strcmp(argv[i], "-alarmack"))
			pConfig->bAlarmAck = AeTrue;
		else if (!strcmp(argv[i], "-persistconnection") && (++i < *pargc))
			pConfig->bPersist = (AeUInt32) strtol(argv[i], NULL, 10);
		else if (argv[i][0] != '-')
            break;
    }

    /* need minimum number of non-options */
    if (*pargc - i < iNonOptCount)
        return AeFalse;

    *pargc = i;

    return AeTrue;
}

/******************************************************************************/
AeBool AeDemoApplyConfig(AeDemoConfig *pConfig)
{
    AeError rc;

    /* configure SSL */
    if (pConfig->bSecure)
    {
        /* configure SSL parameters: encryption level is medium (128 bit),
           disable server certificate validation */
        rc = AeWebSetSSL(AeWebCryptoMedium, AeFalse, NULL);
        if (rc != AeEOK)
        {
            fprintf(stderr, "Failed to set SSL parameters (%s)\n", AeGetErrorString(rc));
            return AeFalse;
        }
    }

    /* configure proxy server */
    if (pConfig->iProxyProto != AeWebProxyProtoNone)
    {
        rc = AeWebSetProxy(pConfig->iProxyProto, pConfig->pProxyHost, pConfig->iProxyPort,
            pConfig->pProxyUser, pConfig->pProxyPassword);
        if (rc != AeEOK)
        {
            fprintf(stderr, "Failed to configure proxy (%s)\n", AeGetErrorString(rc));
            return AeFalse;
        }
    }

    /* enable/disable debug messages */
    AeDRMSetLogLevel(pConfig->bDebug ? AeLogDebug : AeLogInfo);

    return AeTrue;
}
