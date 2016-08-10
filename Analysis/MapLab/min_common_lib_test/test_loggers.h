/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_loggers_h__
#define __test_loggers_h__
#include <extesting.h>
#include <tracer.h>

class TestLoggers : public Test
{
public:
    TestLoggers () : Test ("Logging") {}
    bool process ();
};
#endif // __test_loggers_h__
