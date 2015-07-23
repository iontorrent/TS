/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_bitiops_h__
#define __test_bitiops_h__

#include <extesting.h>

class TestBitops : public Test
{
public:
    TestBitops ()
    :
    Test ("Bitops: compile-time and run-time bit operations on scalar values")
    {
    }
    bool process ();
};


#endif