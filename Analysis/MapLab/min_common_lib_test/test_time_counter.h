/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_time_counter_h__
#define __test_time_counter_h__

#include <extesting.h>

class TestTimeCounter : public Test
{
public:
    TestTimeCounter ()
    :
    Test ("TimeCounter facility")
    {
    }
    bool process ();
};


#endif // __test_time_counter_h__
