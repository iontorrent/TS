/* $Id: AeDemo1.c,v 1.20 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation.  All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDemo1.c
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  Demo application: device data export example
 *
 **************************************************************************/


#ifdef WIN32
#if(_MSC_VER >= 1300)
	#define _CRT_SECURE_NO_DEPRECATE  // re:  Secure Template Overloads see MSDN
    #define _USE_32BIT_TIME_T       // otherwise time_t is 64 bits in .2005
#endif
#endif


#include <stdio.h>
#include <math.h>
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
#define MODEL_NUMBER    "PGM"
#define SERIAL_NUMBER   "sn23056874"

/* Define rate in seconds. */
#define PING_RATE       5

/* Demo tag structure takes a name, a function pointer to get the data value
   and an optional function pointer to check the data for an alarm condition. */
typedef struct
{
    AeChar  *pName;
    void    (*pAcquireFunc)(AeDRMDataValue *pValue);
    AeBool  (*pAlarmFunc)(AeDRMAlarm *pAlarm, AeDRMDataItem *pDataItem);
} AeDemoTag;

/* Function prototypes. */
static void AeDemoSine(AeDRMDataValue *pValue);
static void AeDemoRamp(AeDRMDataValue *pValue);
static void AeDemoString(AeDRMDataValue *pValue);
static AeBool AeDemoAlarm(AeDRMAlarm *pAlarm, AeDRMDataItem *pDataItem);

/* Demo tags. The sine and ramp tags also generate alarm data. */
static AeDemoTag g_demoTags[] =
{
    { "sine1",   AeDemoSine, AeDemoAlarm },
    { "ramp1",   AeDemoRamp, AeDemoAlarm },
    { "string1", AeDemoString, NULL }
};

static int g_demoTagsCount = sizeof(g_demoTags) / sizeof(AeDemoTag);

/******************************************************************************
 * Synopsis:
 *
 *     AeDemo1 [-o owner] [-s] [-g] [{-ph|-ps} proxy-host [-pp proxy-port]
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
 *     (using url argument). After that it loops and posts simulated data and
 *     alarms.
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

    /* use DRM server system time */
/*
    AeDRMSetTimeStampMode(AeTrue);
*/

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

	/* Add remote session if configured */
    if (config.pRemoteSessionName != NULL && config.pRemoteSessionType != NULL &&
        config.pRemoteSessionHost != NULL && config.iRemoteSessionPort != 0)
    {
	    rc = AeDRMAddRemoteSession(iDeviceId, config.pRemoteSessionName, "",
            config.pRemoteSessionType, config.pRemoteSessionHost, config.iRemoteSessionPort);
        if (rc != AeEOK)
        {
            fprintf(stderr, "Failed to add Remote session (%s)\n", AeGetErrorString(rc));
            return 1;
        }
    }

    /* execute demo until keystroke */
#ifdef WIN32
    while (!_kbhit())
#else
    while (1)
#endif
    {
        AeDRMDataItem dataItem;
        AeDRMAlarm alarmData;
        AeInt i;

        /* scan tags */
        for (i = 0; i < g_demoTagsCount; i++)
        {
            dataItem.pName = g_demoTags[i].pName;
            AeGetCurrentTime(&dataItem.value.timeStamp);

            /* acquire tag value */
            if (g_demoTags[i].pAcquireFunc)
            {
                (*g_demoTags[i].pAcquireFunc)(&dataItem.value);

                /* submit tag */
                AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
            }

            /* check alarm */
            if (g_demoTags[i].pAlarmFunc)
            {
                if (g_demoTags[i].pAlarmFunc(&alarmData, &dataItem))
                {
                    /* submit alarm */
                    AeDRMPostAlarm(iDeviceId, iServerId, AeDRMQueuePriorityUrgent, &alarmData);
                }
            }
        }

        /* set time limit for the DRM execution. this is also the data
         * poll rate. */
        timeLimit.iSec = 30;
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
 * Support Functions
 ******************************************************************************/

/******************************************************************************
 * Function to assign a sine value to a tag.
 ******************************************************************************/
void AeDemoSine(AeDRMDataValue *pValue)
{
    pValue->iType = AeDRMDataAnalog;
    pValue->iQuality = AeDRMDataGood;
    pValue->data.dAnalog = sin(pValue->timeStamp.iSec * 0.1) * 100 / 2 + 50;
}

/******************************************************************************
 * Function to assign a ramp value to a tag.
 ******************************************************************************/
void AeDemoRamp(AeDRMDataValue *pValue)
{
    pValue->iType = AeDRMDataAnalog;
    pValue->iQuality = AeDRMDataGood;
    pValue->data.dAnalog = ((AeUInt32) pValue->timeStamp.iSec) % 100;
}

/******************************************************************************
 * Function to assign a string value to a tag.
 ******************************************************************************/
void AeDemoString(AeDRMDataValue *pValue)
{
    pValue->iType = AeDRMDataString;
    pValue->iQuality = AeDRMDataGood;
    pValue->data.pString = (pValue->timeStamp.iSec % 20 < 10 ? "str1" : "STR2");
}

/******************************************************************************
 * Function to set an alarm based on a data item value
 ******************************************************************************/
AeBool AeDemoAlarm(AeDRMAlarm *pAlarm, AeDRMDataItem *pDataItem)
{
    if (pDataItem->value.iType != AeDRMDataAnalog)
        return AeFalse;

    pAlarm->pName = pDataItem->pName;
    pAlarm->timeStamp = pDataItem->value.timeStamp;
    pAlarm->bAck = AeFalse;
    pAlarm->bActive = AeTrue;
    pAlarm->pDataItem = pDataItem;

    if ((int) pDataItem->value.data.dAnalog == 90)
    {
        pAlarm->iSeverity = 30;
        pAlarm->pCondition = "HiHi";
        pAlarm->pDescription = "Demo HiHi alarm";
    }
    else if ((int) pDataItem->value.data.dAnalog == 10)
    {
        pAlarm->iSeverity = 40;
        pAlarm->pCondition = "LoLo";
        pAlarm->pDescription = "Demo LoLo alarm";
    }
    else
        return AeFalse;

    return AeTrue;
}
