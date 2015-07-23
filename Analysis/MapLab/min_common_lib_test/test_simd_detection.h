/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_simd_detection_h__
#define __test_simd_detection_h__
#include <extesting.h>
#include <platform.h>

class TestSimd : public Test
{
public:
    TestSimd () : Test ("cpu_simd: SIMD capabilities detection") {}
    bool process ()
    {
        if (cpu_simd ()) o_ << "CPU supports SIMD instructions" << std::endl;
        else o_ << "CPU does not support SIMD instructions" << std::endl;
        return true;
    }
};

#endif // __test_simd_detection_h__
