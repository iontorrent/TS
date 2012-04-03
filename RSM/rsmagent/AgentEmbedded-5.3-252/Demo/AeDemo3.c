/* $Id: AeDemo3.c,v 1.7 2007/03/20 22:00:34 jklink Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation.  All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeDemo3.c
 *
 *  Subsystem  :  Axeda Agent Embedded
 *
 *  Description:  Demo application: raw HTTP request example
 *
 **************************************************************************/

#if (WIN32) && (_MSC_VER > 1300)
	#define _CRT_SECURE_NO_DEPRECATE  // re:  Secure Template Overloads see MSDN
    #define _USE_32BIT_TIME_T       // otherwise time_t is 64 bits in .2005
#endif

#include <stdio.h>
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

#ifdef WIN32
#include <io.h>
#define write   _write
#endif

/* function prototypes  */
void    AeDemoOnError(AeWebRequest *pRequest, AeError iError);
AeBool  AeDemoOnResponse(AeWebRequest *pRequest, AeInt iStatusCode);
AeBool  AeDemoOnEntity(AeWebRequest *pRequest, AeInt32 iDataOffset, AeChar *pData, AeInt32 iSize);
void    AeDemoOnCompleted(AeWebRequest *pRequest);

/******************************************************************************
 * Synopsis:
 *
 *     AeDemo3 [-s] [-g] [{-ph|-ps} proxy-host [-pp proxy-port]
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
 *     The program performs HTTP GET method on a specified URL and displays the
 *     content.
 ******************************************************************************/
int main(int argc, char *argv[])
{
    AeDemoConfig config;
    AeWebRequest *pRequest;

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

    /* create request object */
    pRequest = AeWebRequestNew();
    if (pRequest)
    {
        /* setup request */
        AeWebRequestSetURL(pRequest, argv[argc]);
        AeWebRequestSetOnError(pRequest, AeDemoOnError);
        AeWebRequestSetOnResponse(pRequest, AeDemoOnResponse);
        AeWebRequestSetOnEntity(pRequest, AeDemoOnEntity);
        AeWebRequestSetOnCompleted(pRequest, AeDemoOnCompleted);

        /* perform request */
        AeWebSyncExecute(pRequest);

        /* destroy request */
        AeWebRequestDestroy(pRequest);
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
 * OnError callback
 ******************************************************************************/
void AeDemoOnError(AeWebRequest *pRequest, AeError iError)
{
    fprintf(stderr, "\nError occured while performing %s on %s: %s\n",
        pRequest->pMethod, AeWebRequestGetURL(pRequest), AeGetErrorString(iError));
}

/******************************************************************************
 * OnResponse callback
 ******************************************************************************/
AeBool AeDemoOnResponse(AeWebRequest *pRequest, AeInt iStatusCode)
{
    AePointer header;
    AeChar *pName, *pValue;

    fprintf(stderr, "HTTP status=%d\n", iStatusCode);

    if (iStatusCode != HTTP_STATUS_OK)
        return AeFalse;

    fprintf(stderr, "\nHeaders:\n\n");
    header = AeWebRequestGetFirstResponseHeader(pRequest, &pName, &pValue);
    while (header)
    {
        fprintf(stderr, "%s: %s\n", pName, pValue);
        header = AeWebRequestGetNextResponseHeader(pRequest, header, &pName, &pValue);
    }

    fprintf(stderr, "\nContent follows\n\n");

    return AeTrue;
}

/******************************************************************************
 * OnEntity callback
 ******************************************************************************/
AeBool AeDemoOnEntity(AeWebRequest *pRequest, AeInt32 iDataOffset, AeChar *pData, AeInt32 iSize)
{
    write(1, pData, iSize);

    return AeTrue;
}

/******************************************************************************
 * OnCompleted callback
 ******************************************************************************/
void AeDemoOnCompleted(AeWebRequest *pRequest)
{
    fprintf(stderr, "\n\nCompleted\n");
}
