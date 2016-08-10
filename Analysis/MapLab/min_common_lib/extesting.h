/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __extesting_h__
#define __extesting_h__

#include "runtime_error.h"
#include <exception>
#include "test_case.h"

class Test : public TestCase
{
public:
    Test (const char* name, std::ostream& ostr = nullstream) 
    : 
    TestCase (name) 
    {
    }
    Test (const char* name, TestCase* base)
    : 
    TestCase (name, base) 
    {
    }
    bool wrap_process ()
    {
        try
        {
            return process ();
        }
        catch (RunTimeError& e)
        {
            o_ << "\nRun-time error exception caught: " << e << std::endl;
        }
        catch (std::exception& e)
        {
            o_ << "\nSTL exception (caught: " << e.what () << std::endl;
        }
        return false;
    }
};


#endif // __extesting_h__
