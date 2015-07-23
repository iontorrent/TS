/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_exbitw_h__
#define __test_exbitw_h__

#include <extesting.h>


class TestExbitw : public Test
{

public:
    TestExbitw ()
    :
    Test ("BitWidth: Bit width computing primitives")
    {
    }
    bool process ();
};


#endif // __test_exbitw_h__

