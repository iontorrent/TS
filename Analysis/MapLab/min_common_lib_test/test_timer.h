/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_timer_h__
#define __test_timer_h__

#include <extesting.h>

class TestTimer : public Test
{
public:
    TestTimer ()
    :
    Test ("Timer facility")
    {
    }
    bool process ();
};


#endif // __test_timer_h__
