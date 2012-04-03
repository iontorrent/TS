/* $Id: AeError.h,v 1.7 2008/05/21 18:24:36 dkhodos Exp $ */

/**************************************************************************
 *
 *  Copyright (c) 1999-2007 Axeda Corporation. All rights reserved.
 *
 **************************************************************************
 *
 *  Filename   :  AeError.h
 *  
 *  Subsystem  :  Axeda Agent Embedded
 *  
 *  Description:  Error declarations
 *
 **************************************************************************/
#ifndef _AE_ERROR_H_
#define _AE_ERROR_H_

typedef enum _AeError AeError;

enum _AeError
{
    AeEOK,                          /* 0 */
    AeEInternal,                    /* 1 */
    AeEInvalidArg,                  /* 2 */
    AeEMemory,                      /* 3 */
    AeEExist,                       /* 4 */
    AeENetGeneral,                  /* 5 */
    AeENetTimeout,                  /* 6 */
    AeENetWouldBlock,               /* 7 */
    AeENetUnknownHost,              /* 8 */
    AeENetConnLost,                 /* 9 */
    AeENetConnRefused,              /* 10 */
    AeENetConnReset,                /* 11 */
    AeENetConnAborted,              /* 12 */
    AeENetNotConn,                  /* 13 */
    AeEWebBadResponse,              /* 14 */
    AeEWebAuthFailed,               /* 15 */
    AeEWebAuthUnsupported,          /* 16 */
    AeESSLGeneral,                  /* 17 */
    AeESSLWeakerCipher,             /* 18 */
    AeESSLCertIssuerUnknown,        /* 19 */
    AeESSLCertInvalid,              /* 20 */
    AeESSLCertVerifyFailed,         /* 21 */
    AeESSLHandshakeFailed,          /* 22 */
    AeESOCKSWrongVersion,           /* 23 */
    AeESOCKSAuthFailed,             /* 24 */
    AeESOCKSGeneral,                /* 25: general SOCKS server failure */
    AeESOCKSPerm,                   /* 26: connection not allowed by ruleset */
    AeESOCKSNetUnreach,             /* 27: network unreachable */
    AeESOCKSHostUnreach,            /* 28: host unreachable */
    AeESOCKSConnRefused,            /* 29: connection refused */
    AeESOCKSTTL,                    /* 30: TTL expired */
    AeESOCKSBadCommand,             /* 31: command not supported */
    AeESOCKSBadAddress,             /* 32: address type not supported */
    AeENetNetUnreach,               /* 33 */
    AeENetHostUnreach,              /* 34 */
    AeBadURL                        /* 35 */
};

#endif
