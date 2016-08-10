/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_error_handling_h__
#define __test_error_handling_h__
#include <extesting.h>
#include <runtime_error.h>

class TestErrorHandling : public Test
{
public:
    TestErrorHandling () : Test ("Exceptions - Run-time Exception handling") {}
    bool process ();
};
#endif // __test_error_handling_h__
