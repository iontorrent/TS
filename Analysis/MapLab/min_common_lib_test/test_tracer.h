/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_tracer_h__
#define __test_tracer_h__

#include <extesting.h>

class TestTracer : public Test
{
public:
    TestTracer ()
    :
    Test ("Tracing: controllable logging streams facility")
    {
    }
    bool process ();
};


#endif // __test_tracer_h__
