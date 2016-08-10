/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __common_types_h__
#define __common_types_h__

#include "compile_time_macro.h"

#include <vector>
#include <string>
#include <set>
#include <map>

typedef unsigned char           uchar;
typedef unsigned short int      ushort;
typedef unsigned int            uint;
typedef long unsigned int       ulong;
typedef long long signed int    longlong;
typedef long long unsigned int  ulonglong;

typedef unsigned char BYTE;
ASSERT_EXACT_BITSIZE (BYTE, 8)
typedef signed char SBYTE;
ASSERT_EXACT_BITSIZE (SBYTE, 8)
typedef unsigned short WORD;
ASSERT_EXACT_BITSIZE (WORD, 16)
typedef signed short SWORD;
ASSERT_EXACT_BITSIZE (SWORD, 16)
typedef unsigned int DWORD;
ASSERT_EXACT_BITSIZE (DWORD, 32)
typedef signed int SDWORD;
ASSERT_EXACT_BITSIZE (SDWORD, 32)
typedef ulonglong QWORD;
ASSERT_EXACT_BITSIZE (QWORD, 64)
typedef longlong SQWORD;
ASSERT_EXACT_BITSIZE (SQWORD, 64)


typedef std::vector<std::string> StrVec;
typedef std::map<std::string, std::string> StrStrMap;
typedef std::set<std::string> StrSet;
typedef std::set<int> IntSet;
typedef std::vector<int> IntVec;


char* lltoa (longlong val, char* buf, int base);
char* ulltoa (ulonglong val, char* buf, int base);
ulonglong atoull (const char* strval);

#endif // __common_types_h__
