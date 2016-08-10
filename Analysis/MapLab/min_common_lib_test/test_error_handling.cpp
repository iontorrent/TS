/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_error_handling.h"
bool TestErrorHandling::process ()
{
    try
    {
        ers << "Testing Throw mechanism";
        ers << Throw;
        TEST_FAILURE ("Throw mechanism testing failed");
    }
    catch (RunTimeError& r)
    {
        o_ << "Throw mechanism testing succeeded, msg = " << (const char*) r << ", exception is " << r << std::endl;
    }
    try
    {
        ERROR ("Testing ERROR macro");
        TEST_FAILURE ("ERROR macro testing failed");
    }
    catch (RunTimeError& r)
    {
        o_ << "ERROR macro testing succeeded, exception = " << r << std::endl;
    }
    try
    {
        ers << "Testing ThrowEx mechanism" << ThrowEx (InternalError);
        TEST_FAILURE ("ThrowEx mechanism testing failed");
    }
    catch (InternalError& ir)
    {
        o_ << "ThrowEx mechanism testing succeeded, exception = " << ir << std::endl;
    }
    catch (RunTimeError& r)
    {
        o_ << "ThrowEx mechanism testing FAILED, RunTimeError exception caught instead of InternalError : " << r << std::endl;
        TEST_FAILURE ("ThrowEx mechanism testing FAILED, RunTimeError exception caught instead of InternalError");
    }
    try
    {
        ers << "Testing OSRerror" << ThrowEx (OSError);
        TEST_FAILURE ("OSRerror testing failed");
    }
    catch (OSError& ose)
    {
        o_ << "OSError testing succeeded, exc = " << ose << std::endl;
    }
    catch (RunTimeError& r)
    {
        o_ << "OSError testing FAILED, RunTimeError caught instead : " << r << std::endl;
        TEST_FAILURE ("OSError testing FAILED, RunTimeError caught instead");
    }
    return true;
}
