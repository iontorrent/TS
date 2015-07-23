/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __extesting_h__
#define __extesting_h__

#include "rerror.h"
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
        catch (Rerror& e)
        {
            o_ << "\nRun-time error (Rerror) exception caught: " << e << std::endl;
        }
        catch (std::exception& e)
        {
            o_ << "\nSTL exception (std::exception) caught: " << e.what () << std::endl;
        }
        return false;
    }
};


#endif // __extesting_h__
