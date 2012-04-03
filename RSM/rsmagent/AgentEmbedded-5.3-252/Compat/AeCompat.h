/* $Id: AeCompat.h,v 1.1.4.2 2009/05/20 12:26:39 hfeng Exp $ */
#ifndef _AE_COMPAT_H_
#define _AE_COMPAT_H_


/* AXIS openssl does not have md4 */
#if defined(AXIS_LINUX) || !defined(HAVE_OPENSSL)
#include "md_global.h"
#include "md4.h"
#endif

#ifndef HAVE_OPENSSL
#include "md5.h"
#include "des.h"
#define MD5_DIGEST_LENGTH 16
#endif

/* AXIS openssl does not have md4 */
#if defined(AXIS_LINUX) || !defined(HAVE_OPENSSL)
#define MD4_Init   MD4Init
#define MD4_Update MD4Update
#define MD4_Final  MD4Final
#endif

#ifndef HAVE_OPENSSL
#define MD5_Init   MD5Init
#define MD5_Update MD5Update
#define MD5_Final  MD5Final
#endif

#endif
