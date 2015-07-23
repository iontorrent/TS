/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_byte_order_h__
#define __test_byte_order_h__

#include <extesting.h>

class TestByteOrder : public Test
{
public:
    TestByteOrder ()
    :
    Test ("ByteOrder: Bits and bytes order")
    {
    }
    bool process ();
};

#endif // __test_byte_order__