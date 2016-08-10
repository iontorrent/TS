/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_temp_file_h__
#define __test_temp_file_h__

#include <extesting.h>

class TestTempFile : public Test
{
public:
    TestTempFile ()
    :
    Test ("TempFiles Temporary directory and temporary file facilities")
    {
    }
    bool process ();
};

#endif
