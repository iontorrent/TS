/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

#include "compile_time_macro.h"

#ifdef _MSC_VER
typedef __int64 longlong;
typedef unsigned __int64 ulonglong;
#else
typedef long long signed int    longlong;
typedef long long unsigned int ulonglong;
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// HACK
#if !defined (__x86__)
#define __x86__
#endif

#if defined (__x86__) || defined (__powerpc__)
#define SCIDM_LITTLE_ENDIAN
//#warning LITTLE ENDIAN
#elif defined (__mips__) || defined (__alpha__) || defined (__rx000__)
#define SCIDM_BIG_ENDIAN
#warning BIG ENDIAN
#else
#error Byte order unknown
#endif

// the following works for MSVC on 32-bit intel and gcc on 32 and 64 bit (LP64 speciication) intel.
// Could fail on other platforms / compilers
typedef unsigned char BYTE;
typedef signed char SBYTE;
typedef unsigned short WORD;
typedef signed short SWORD;
typedef unsigned int DWORD;
typedef signed int SDWORD;
typedef ulonglong QWORD;
typedef longlong SQWORD;

ASSERT_EXACT_BITSIZE (BYTE, 8)
ASSERT_EXACT_BITSIZE (SBYTE, 8)
ASSERT_EXACT_BITSIZE (WORD, 16)
ASSERT_EXACT_BITSIZE (SWORD, 16)
ASSERT_EXACT_BITSIZE (DWORD, 32)
ASSERT_EXACT_BITSIZE (SDWORD, 32)
ASSERT_EXACT_BITSIZE (QWORD, 64)
ASSERT_EXACT_BITSIZE (SQWORD, 64)

#ifdef SCIDM_LITTLE_ENDIAN

#define GET32_U(ptr) (*(const DWORD*)(ptr))
#define GET32_UR(ptr) ( (((DWORD) ((const BYTE*) ptr) [0]) << 24) | \
                        (((DWORD) ((const BYTE*) ptr) [1]) << 16) | \
                        (((DWORD) ((const BYTE*) ptr) [2]) << 8 ) | \
                        (((DWORD) ((const BYTE*) ptr) [3])      ) )

#define GET64_U(ptr) (*(const QWORD*)(ptr))
#define GET64_UR(ptr) ( (((QWORD) ((const BYTE*) ptr) [0]) << 56) | \
                        (((QWORD) ((const BYTE*) ptr) [1]) << 48) | \
                        (((QWORD) ((const BYTE*) ptr) [2]) << 40) | \
                        (((QWORD) ((const BYTE*) ptr) [3]) << 32) | \
                        (((QWORD) ((const BYTE*) ptr) [4]) << 24) | \
                        (((QWORD) ((const BYTE*) ptr) [5]) << 16) | \
                        (((QWORD) ((const BYTE*) ptr) [6]) << 8 ) | \
                        (((QWORD) ((const BYTE*) ptr) [7])      ) )

#else

#define GET32_U(ptr)  ( (((DWORD) ((const BYTE*) ptr) [0]) << 24) | \
                        (((DWORD) ((const BYTE*) ptr) [1]) << 16) | \
                        (((DWORD) ((const BYTE*) ptr) [2]) << 8 ) | \
                        (((DWORD) ((const BYTE*) ptr) [3])      ) )
#define GET32_UR(ptr) (*(const DWORD*)(ptr))


#define GET64_U(ptr)  ( (((QWORD) ((const BYTE*) ptr) [0]) << 56) | \
                        (((QWORD) ((const BYTE*) ptr) [1]) << 48) | \
                        (((QWORD) ((const BYTE*) ptr) [2]) << 40) | \
                        (((QWORD) ((const BYTE*) ptr) [3]) << 32) | \
                        (((QWORD) ((const BYTE*) ptr) [4]) << 24) | \
                        (((QWORD) ((const BYTE*) ptr) [5]) << 16) | \
                        (((QWORD) ((const BYTE*) ptr) [6]) << 8 ) | \
                        (((QWORD) ((const BYTE*) ptr) [7])      ) )
#define GET64_UR(ptr) (*(const QWORD*)(ptr))

#endif

bool cpu_simd ();

#if defined (_MSC_VER)
longlong atoll (const char* strval);
#endif

char* lltoa (longlong val, char* buf, int base);
char* ulltoa (ulonglong val, char* buf, int base);

#endif //__PLATFORM_H__

