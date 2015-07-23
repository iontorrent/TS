/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_resource.h"


int TestResource::Tst::kk = 0;

bool TestResource::process ()
{
    MemWrapper<Tst> bb;
    MemWrapper<Tst> a (1);
    {
        o_ << "Entered scope" << std::endl;
        MemWrapper<Tst> b (2);
        MemWrapper<Tst> aa;
        o_ << "Allocated" << std::endl;
        bb = b;
        aa = a;
        o_ << "Assigned" << std::endl;
    }
    o_ << "Left scope" << std::endl;
    return true;
}
