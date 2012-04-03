/* $Id: AeDemo5.c,v 1.8 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation.  All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDemo5.c
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  Demo application: custom file upload example
 *
 **************************************************************************/
#if (WIN32) && (_MSC_VER > 1300)
	#define _CRT_SECURE_NO_DEPRECATE  // re:  Secure Template Overloads see MSDN
    #define _USE_32BIT_TIME_T       // otherwise time_t is 64 bits in .2005
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#ifdef WIN32
#include <conio.h>
#else
#include <unistd.h>
#endif
#include <errno.h>

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"
#include "AeDemoCommon.h"

#if (WIN32) && (_MSC_VER < 1300) && (_DEBUG)
    #define _CRTDBG_MAP_ALLOC
    #include <crtdbg.h>
#endif

/* Predefine the model/serial number. */
#define MODEL_NUMBER    "DemoModel1"
#define SERIAL_NUMBER   "DemoDevice1"

/* Define rate in seconds. */
#define PING_RATE       5

/* uncomment the following to initiate upload from the application */
#define INITIATE_UPLOAD 1
#define UPLOAD_FILE_NAME "c:\\temp\\file.dat"

typedef struct _AeDemoUpload AeDemoUpload;

struct _AeDemoUpload
{
    AeFileUploadSpec **ppUploads;
    AeInt32          iUploadIdx;
    AeFileStat       curFileStat;
    AeFileHandle     iCurFileHandle;
    AeChar           pBuffer[BUFSIZ];
};

/* function prototypes  */
static AeBool OnFileUploadBegin(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData);
static AeBool OnFileUploadData(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData);
static void OnFileUploadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData);

/******************************************************************************
 * Synopsis:
 *
 *     AeDemo4 [-o owner] [-s] [-g] [{-ph|-ps} proxy-host [-pp proxy-port]
 *             [-pu proxy-user] [-pw proxy-password]] [-o owner] url
 *
 * url:
 *        Use the following syntax:
 *            http://<server-name>:<port>/eMessage (non-secure communications) or
 *            https://<server-name>:<port>/eMessage  (secure communications)
 *          where <server-name> is replaced with the Enterprise server name.
 *            and <port> is replaced with the port number.  (Use 443 for SSL).
 *        For example, http://drm.axeda.com/eMessage.
 *
 * Options:
 *
 *     -o owner
 *         Use specified owner (database name).
 *     -s
 *         Use secure communication (SSL).
 *     -g
 *         Enable debug messages.
 *     -ph proxy-host
 *         Use HTTP proxy at proxy-host. Default port is 80.
 *     -ps proxy-host
 *         Use SOCKS proxy at proxy-host. Default port is 1080.
 *     -pp proxy-port
 *         Override default port for proxy.
 *     -pu proxy-user
 *         Use proxy-user as user name for proxy authentication.
 *     -pw proxy-password
 *         Use proxy-password as password for proxy authentication.
 *
 * Description:
 *
 *     The program defines a device, configures the primary Enterprise server
 *     (using url argument) and installs command callbacks. After that it loops
 *     and waits for callback invocation.
 ******************************************************************************/
int main(int argc, char *argv[])
{
    AeDemoConfig config;
    AeInt32 iDeviceId, iServerId;
    AeTimeValue pingRate, timeLimit;
    AeError rc;

    /* process options */
    if (!AeDemoProcessOptions(&config, &argc, argv, 1))
    {
        AeDemoUsage(argv[0], "url");
        return 1;
    }

    /* initialize Axeda Agent Embedded */
    AeInitialize();

    /* apply options */
    AeDemoApplyConfig(&config);

    /* configure master device */
    rc = AeDRMAddDevice(AeDRMDeviceMaster, MODEL_NUMBER, SERIAL_NUMBER, &iDeviceId);
    if (rc != AeEOK)
    {
        fprintf(stderr, "Failed to add device (%s)\n", AeGetErrorString(rc));
        return 1;
    }

    /* configure primary DRM server */
    pingRate.iSec = PING_RATE;
    pingRate.iMicroSec = 0;
    rc = AeDRMAddServer(AeDRMServerConfigPrimary, argv[argc], config.pOwner,
        &pingRate, &iServerId);
    if (rc != AeEOK)
    {
        fprintf(stderr, "Failed to add server (%s)\n", AeGetErrorString(rc));
        return 1;
    }

    /* install command callbacks */
    AeDRMSetOnFileUploadBegin(OnFileUploadBegin);
    AeDRMSetOnFileUploadData(OnFileUploadData);
    AeDRMSetOnFileUploadEnd(OnFileUploadEnd);

#ifdef INITIATE_UPLOAD
    {
        AeFileUploadSpec uploadSpec;
        AeFileUploadSpec *ppUploads[2];
        AeFileUploadRequestParam param;

        /* prepare upload specification */
        uploadSpec.pName = UPLOAD_FILE_NAME;
        uploadSpec.bDelete = AeFalse;


        /* upload specification list must be NULL-terminated */
        ppUploads[0] = &uploadSpec;
        ppUploads[1] = NULL;

        /* no compression for the file */
        param.iMask = AE_FILE_UPLOAD_REQUEST_PARAM_COMPRESSION;
        param.iCompression = AeFileCompressionNone;

        /* submit upload request */
        AeDRMPostFileUploadRequestEx(iDeviceId, iServerId, AeDRMQueuePriorityNormal, ppUploads, &param);
    }
#endif /* INITIATE_UPLOAD */

    /* execute demo until keystroke */
#ifdef WIN32
    while (!_kbhit())
#else
    while (1)
#endif
    {
        /* set time limit for the DRM execution. this is also the data
         * poll rate. */
        timeLimit.iSec = 1;
        timeLimit.iMicroSec = 0;

		/* this is the execution cycle */
		AeDRMExecute(&timeLimit);
    }

    /* shutdown Axeda Agent Embedded */
    AeShutdown();

#if defined(WIN32) && defined(_DEBUG)&& (_MSC_VER < 1300)
    _CrtDumpMemoryLeaks();
#endif

    return 0;
}

/******************************************************************************
 * Callbacks
 ******************************************************************************/

/******************************************************************************/
static AeBool OnFileUploadBegin(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData)
{
    AeDemoUpload *pUpload;

    /* allocate upload descriptor */
    pUpload = (AeDemoUpload *) malloc(sizeof(AeDemoUpload));
    if (!pUpload)
        return AeFalse;

    /* initialize */
    memset(pUpload, 0, sizeof(AeDemoUpload));
    pUpload->ppUploads = ppUploads;
    pUpload->iUploadIdx = 0;
    pUpload->iCurFileHandle = AeFileInvalidHandle;

    *ppUserData = pUpload;

    return AeTrue;
}

/******************************************************************************/
static AeBool OnFileUploadData(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData)
{
    AeDemoUpload *pUpload;

    *ppFile = NULL;
    *ppData = NULL;
    *piSize = 0;

    pUpload = (AeDemoUpload *) pUserData;
    if (!pUpload)
        return AeFalse;

    /* no more files to upload: indicate that */
    if (!pUpload->ppUploads[pUpload->iUploadIdx])
        return AeTrue;

    /* initialize next file */
    if (pUpload->iCurFileHandle == AeFileInvalidHandle)
    {
        /* open file */
        pUpload->iCurFileHandle = AeFileOpen(pUpload->ppUploads[pUpload->iUploadIdx]->pName,
                                             AE_OPEN_READ_ONLY);
        if (pUpload->iCurFileHandle == AeFileInvalidHandle)
            return AeFalse;

        pUpload->curFileStat.pName = pUpload->ppUploads[pUpload->iUploadIdx]->pName;
        pUpload->curFileStat.iType = AeFileTypeRegular;
        pUpload->curFileStat.iSize =
#ifndef ENABLE_LARGEFILE64
            AeFileGetSize
#else
            AeFileGetSize64
#endif
            (pUpload->ppUploads[pUpload->iUploadIdx]->pName);
        pUpload->curFileStat.iMTime = 0;
    }

    *ppFile = &pUpload->curFileStat;

    /* try to read another portion of the file */
    *piSize = AeFileRead(pUpload->iCurFileHandle, pUpload->pBuffer, sizeof(pUpload->pBuffer));
    if (*piSize < 0)
        return AeFalse;
    else if (*piSize == 0)
    {
        AeFileClose(pUpload->iCurFileHandle);
        pUpload->iCurFileHandle = AeFileInvalidHandle;

        if (pUpload->ppUploads[pUpload->iUploadIdx]->bDelete)
            AeFileDelete(pUpload->ppUploads[pUpload->iUploadIdx]->pName);

        pUpload->iUploadIdx += 1;
    }
    else if (*piSize > 0)
        *ppData = pUpload->pBuffer;

    return AeTrue;
}

/******************************************************************************/
static void OnFileUploadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData)
{
    AeDemoUpload *pUpload;

    pUpload = (AeDemoUpload *) pUserData;
    if (!pUpload)
        return;

    if (pUpload->iCurFileHandle != AeFileInvalidHandle)
        AeFileClose(pUpload->iCurFileHandle);

    AeFree(pUpload);
}
