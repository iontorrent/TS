/* $Id: AeDemo2.c,v 1.11 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation.  All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDemo2.c
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  Demo application: command execution example
 *
 **************************************************************************/
#if (WIN32) && (_MSC_VER > 1300)
	#define _CRT_SECURE_NO_DEPRECATE  // re:  Secure Template Overloads see MSDN
    #define _USE_32BIT_TIME_T       // otherwise time_t is 64 bits in .2005
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef WIN32
#include <conio.h>
#else
#include <unistd.h>
#endif

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

/* function prototypes  */
static void AeDemoOnSetTag(AeInt32 iDeviceId, AeDRMDataItem *pDataItem, AeDRMSOAPCommandStatus *pStatus);
static void AeDemoOnSetTime(AeInt32 iDeviceId, AeTimeValue *pTime, AeInt32 *piTZOffset, AeDRMSOAPCommandStatus *pStatus);
static void AeDemoOnRestart(AeInt32 iDeviceId, AeBool bHard, AeDRMSOAPCommandStatus *pStatus);

/******************************************************************************
 * Synopsis:
 *
 *     AeDemo2 [-o owner] [-s] [-g] [{-ph|-ps} proxy-host [-pp proxy-port]
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

    AeWebSetPersistent(iServerId, AeTrue);

    /* install command callbacks */
    AeDRMSetOnCommandSetTag(AeDemoOnSetTag);
    AeDRMSetOnCommandSetTime(AeDemoOnSetTime);
    AeDRMSetOnCommandRestart(AeDemoOnRestart);

    timeLimit.iSec = 1;
    timeLimit.iMicroSec = 0;

    /* execute demo until keystroke */
#ifdef WIN32
    while (!_kbhit())
#else
    while (1)
#endif
    {
        /* do DRM processing */
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

/******************************************************************************
 * OnSetTag callback
 ******************************************************************************/
void AeDemoOnSetTag(AeInt32 iDeviceId, AeDRMDataItem *pDataItem, AeDRMSOAPCommandStatus *pStatus)
{
    fprintf(stdout, "Command: set tag value: device id=%ld, tag=%s, value=",
        iDeviceId, pDataItem->pName);

    switch (pDataItem->value.iQuality)
    {
        case AeDRMDataGood:
            switch (pDataItem->value.iType)
            {
                case AeDRMDataAnalog:
                    fprintf(stdout, "%f", pDataItem->value.data.dAnalog);
                    break;
                case AeDRMDataDigital:
                    fprintf(stdout, "%d", pDataItem->value.data.bDigital);
                    break;
                case AeDRMDataString:
                    fprintf(stdout, "%s", pDataItem->value.data.pString);
                    break;
            }
            break;
        case AeDRMDataBad:
            fprintf(stdout, "(bad quality)");
            break;
        case AeDRMDataUncertain:
            fprintf(stdout, "(uncertain quality)");
            break;
    }

    fprintf(stdout, "\n");

    pStatus->iStatusCode = 0;
    pStatus->pStatusReason = "OK";
}

/******************************************************************************
 * OnSetTime callback
 ******************************************************************************/
void AeDemoOnSetTime(AeInt32 iDeviceId, AeTimeValue *pTime, AeInt32 *piTZOffset, AeDRMSOAPCommandStatus *pStatus)
{
    char pTimeStr[64];

    strncpy(pTimeStr, ctime((time_t *) &pTime->iSec), sizeof(pTimeStr));
    pTimeStr[sizeof(pTimeStr) - 1] = 0;
    pTimeStr[strlen(pTimeStr) - 1] = 0;

    fprintf(stdout, "Command: time synchronization (%s) for device: id=%ld\n",
        pTimeStr, iDeviceId);

    pStatus->iStatusCode = 0;
    pStatus->pStatusReason = "OK";
}

/******************************************************************************
 * OnRestart callback
 ******************************************************************************/
void AeDemoOnRestart(AeInt32 iDeviceId, AeBool bHard, AeDRMSOAPCommandStatus *pStatus)
{
    fprintf(stdout, "Command: %s restart for device: id=%ld\n",
        bHard ? "hard" : "soft", iDeviceId);

    pStatus->iStatusCode = 0;
    pStatus->pStatusReason = "OK";
}
