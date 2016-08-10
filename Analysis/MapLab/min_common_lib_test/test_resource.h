/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __test_resource_h__
#define __test_resource_h__
#include <extesting.h>
#include <resource.h>

class TestResource : public Test
{
    struct Tst
    {
        int k;
        static int kk;
        Tst ()
        {
            k = kk ++;
            std::cout << "Tst allocated at " << this << ", ordinal " << k << std::endl;
        }
        ~Tst ()
        {
            std::cout << "Tst destroyed at " << this << ", ordinal " << k << std::endl;
        }
    };
public:
    TestResource ()
    :
    Test ("Wrappers: Automatic resource management")
    {
    }

    bool process ();
};
#endif // __test_resource_h__
