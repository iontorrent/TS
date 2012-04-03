/* $Id: AeDemo4.c,v 1.10 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation.  All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDemo4.c
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  Demo application: upgrade flash on NetComm-3CE
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

#include "fw_access.h"
#include "fwupdate.h"

/* Predefine the model/serial number. */
#define MODEL_NUMBER    "DemoModel1"
#define SERIAL_NUMBER   "DemoDevice1"

/* Define rate in seconds. */
#define PING_RATE       5

#define IMAGE_FILE_NAME  "image_3CE.rom"
#define FLASH_START_ADDR 0x11010000
#define EXEC_START_ADDR  0x11010400

typedef struct _AeFwImage AeFwImage;

struct _AeFwImage
{
    AeBool bValid;
    struct fwBlock *pLastBlock;
    struct _fwupdateParm *pFwParm;
};

/* function prototypes  */
static AeBool OnFileDownloadBegin(AeInt32 iDeviceId, AePointer *ppUserData);
static AeBool OnFileDownloadData(AeInt32 iDeviceId, AeFileStat *pFile, AeChar *pData, AeInt32 iSize, AePointer pUserData);
static void OnFileDownloadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData);
static void FwImageProgram(AeFwImage *pImage);
static void FwImageCleanup(AeFwImage *pImage);

#ifdef m68k
_bsc1(int,program,struct _fwupdateParm *, a1)
#else
static int program(struct _fwupdateParm *pFwParm)
{
    return 0;
}
#endif

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
    AeDRMSetOnFileDownloadBegin(OnFileDownloadBegin);
    AeDRMSetOnFileDownloadData(OnFileDownloadData);
    AeDRMSetOnFileDownloadEnd(OnFileDownloadEnd);


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

    return 0;
}

/******************************************************************************
 * Callbacks
 ******************************************************************************/

/******************************************************************************/
static AeBool OnFileDownloadBegin(AeInt32 iDeviceId, AePointer *ppUserData)
{
    AeFwImage *pImage;

    /* allocate image descriptor */
    pImage = (AeFwImage *) malloc(sizeof(AeFwImage));
    if (!pImage)
        return AeFalse;

    /* initialize */
    memset(pImage, 0, sizeof(AeFwImage));
    pImage->bValid = AeTrue;

    *ppUserData = pImage;

    return AeTrue;
}

/******************************************************************************/
static AeBool OnFileDownloadData(AeInt32 iDeviceId, AeFileStat *pFile, AeChar *pData, AeInt32 iSize, AePointer pUserData)
{
    AeFwImage *pImage;
    struct fwBlock *pBlock;

    /* zero size indicates end-of-file: just ignore */
    if (iSize == 0)
        return AeTrue;

    /* check file name */
    if (strcmp(pFile->pName, IMAGE_FILE_NAME))
        return AeTrue;

    /* check image status */
    pImage = (AeFwImage *) pUserData;
    if (!pImage || !pImage->bValid)
        return AeFalse;

    /* allocate _fwupdateParm structure first time */
    if (!pImage->pFwParm)
    {
        pImage->pFwParm = (struct _fwupdateParm *)
            malloc(sizeof(struct _fwupdateParm) + iSize);
        if (!pImage->pFwParm)
        {
            pImage->bValid = AeFalse;
            return AeFalse;
        }

        memset(pImage->pFwParm, 0, sizeof(struct _fwupdateParm) + iSize);
    }

    /* allocate/assign new block */
    if (!pImage->pLastBlock)
        pBlock = &pImage->pFwParm->blockHead;
    else
        pBlock = (struct fwBlock *) malloc(sizeof(struct fwBlock) + iSize);

    if (!pBlock)
    {
        pImage->bValid = AeFalse;
        return AeFalse;
    }

    /* fill block */
    pBlock->size = iSize;
    memcpy(pBlock->data, pData, iSize);
    pBlock->next = NULL;

    /* link blocks */
    if (pImage->pLastBlock)
        pImage->pLastBlock->next = pBlock;
    pImage->pLastBlock = pBlock;

    pImage->pFwParm->length += iSize;

    return AeTrue;
}

/******************************************************************************/
static void OnFileDownloadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData)
{
    AeFwImage *pImage;

    pImage = (AeFwImage *) pUserData;
    if (!pImage)
        return;

    /* discard invalid image */
    if (!bOK || !pImage->bValid || pImage->pFwParm->length == 0)
    {
        FwImageCleanup(pImage);
        return;
    }

    /* program the image into flash */
    FwImageProgram(pImage);
}

/******************************************************************************/
static void FwImageProgram(AeFwImage *pImage)
{
    unsigned int iWritten = 0;

    AeLog(AeLogInfo, "Programming image: size=%d", pImage->pFwParm->length);

    pImage->pFwParm->address = FLASH_START_ADDR;
    pImage->pFwParm->exec = EXEC_START_ADDR;

    iWritten = program(pImage->pFwParm);
    if (iWritten != pImage->pFwParm->length)
    {
        AeLog(AeLogError, "Programming failed");
        FwImageCleanup(pImage);
    }
}

/******************************************************************************/
static void FwImageCleanup(AeFwImage *pImage)
{
    struct fwBlock *pBlock, *pTmp;

    pBlock = pImage->pFwParm->blockHead.next;
    while (pBlock)
    {
        pTmp = pBlock->next;
        free(pBlock);
        pBlock = pTmp;
    }

    free(pImage->pFwParm);
    free(pImage);
}
