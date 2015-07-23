/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_suite_h__
#define __test_suite_h__

#include "test_case.h"

///
/// Facility for running a collection of test cases
class TestSuite
{
    typedef std::vector <TestCase*> TestVect;
    TestVect test_cases_;
    StreamWrap o_;
    bool verbose_;
    bool silent_;
    bool stop_on_error_;
    std::string name_;
protected:
    /// initialize the suite
    /// 
    /// by default, does nothing
    /// The user may chose to overwrite this method in the derived class to register the needed TestCases
    /// The registered instances of TestCases should remain alive upon completion of the testing
    virtual void init () {}
    /// placehorder for test suite cleanup tasks.
    /// 
    /// optionally overwrite this method to clean up what is instantiated in init () and not cleaned up automatically
    virtual void cleanup () {};
public:
    /// constructor. Takes execution environment as parameters
    ///
    /// The "environment" passed to test suite upon creation
    /// is supposed to be the test executable environment, namely main's argc, argv and envp
    /// The command line is parsed so that every string before first dashed arguments 
    /// is treated as (the case - insensitive prefix of ) the test name to run
    /// If no arguments before dashed found, all tests are performed
    TestSuite (int argc, char* argv [], char* envp [] = NULL);
    /// sets the name for the test suite (default is derived from executable name)
    void name (const char* new_name);
    /// retruns name of the test suite
    const char* name () const; 
    /// redirects output to a given stream (sent to stdout by default)
    void set_output (std::ostream&  ostr);
    /// sets 'error cruelty' flag: if set, raises exception on first error
    bool stop_on_error (bool stop_on_error);
    /// gets 'error cruelty' flag (if set, raises exception on first error)
    bool stop_on_error () const;
    /// registers the test case in a suite; registration is the only way to make the suite aware of a particular test case
    void reg (TestCase& test_case);
    /// runs the registered tests (optionally filtered through arguments passed to the constructor)
    virtual bool run ();
};

#endif //  __test_suite_h__
