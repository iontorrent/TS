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
    catch (Rerror& r)
    {
        o_ << "Throw mechanism testing succeeded, msg = " << (const char*) r << ", exception is " << r << std::endl;
    }
    try
    {
        ERR ("Testing ERR macro");
        TEST_FAILURE ("ERR macro testing failed");
    }
    catch (Rerror& r)
    {
        o_ << "ERR macro testing succeeded, exception = " << r << std::endl;
    }
    try
    {
        ers << "Testing ThrowEx mechanism" << ThrowEx (InternalRerror);
        TEST_FAILURE ("ThrowEx mechanism testing failed");
    }
    catch (InternalRerror& ir)
    {
        o_ << "ThrowEx mechanism testing succeeded, exception = " << ir << std::endl;
    }
    catch (Rerror& r)
    {
        o_ << "ThrowEx mechanism testing FAILED, Rerror exception caught instead of InternalRerror : " << r << std::endl;
        TEST_FAILURE ("ThrowEx mechanism testing FAILED, Rerror exception caught instead of InternalRerror");
    }
    try
    {
        ers << "Testing OSRerror" << ThrowEx (OSRerror);
        TEST_FAILURE ("OSRerror testing failed");
    }
    catch (OSRerror& osr)
    {
        o_ << "OSRerror testing succeeded, exc = " << osr << std::endl;
    }
    catch (Rerror& r)
    {
        o_ << "OSRerror testing FAILED, Rerror caught instead : " << r << std::endl;
        TEST_FAILURE ("OSRerror testing FAILED, Rerror caught instead");
    }
    return true;
}
