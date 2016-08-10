/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __compile_time_macro_h__
#define __compile_time_macro_h__

#include <climits> // needed for CHAR_BIT definition

#define __CONCAT_2STR(x,y) x##y

#define __COMPILE_TIME_ASSERT_LINE(pred, uniq_tag) typedef char __CONCAT_2STR(assertion_failed_, __COUNTER__) [2 * !!(pred) - 1];

#define COMPILE_TIME_ASSERT(pred)             __COMPILE_TIME_ASSERT_LINE(pred, UNIQ_NAME)

#define ASSERT_MIN_BITSIZE(type, size)        COMPILE_TIME_ASSERT(sizeof(type) * CHAR_BIT >= (size))

#define ASSERT_EXACT_BITSIZE(type, size)      COMPILE_TIME_ASSERT(sizeof(type) * CHAR_BIT == (size))

#endif
