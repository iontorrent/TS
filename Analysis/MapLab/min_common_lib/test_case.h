/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_case_h__
#define __test_case_h__

/// \file
/// Test case is a part of Testing subsystem\n
/// Declares TestCase class. 
/// The user should derive test cases from the TestCase class and then register them 
/// with the instance of of TestSuite (or derived) class.

#if defined (_MSC_VER)
#pragma warning (disable: 4786)
#endif
#include <iostream>
#include <cstring>
#include <sstream>
#include <vector>
#include <cassert>
#include "nullstream.h"
#include "streamwrap.h"
#include "platform.h"

#include "test_facet.h"

// helper class, representing the treee of selected test names
class TestSelection;

/// base class for the test case.
///
/// Provides facilites for functionality tests and for benchmarking
/// The simplest use if through overwriting loaded_iter method. The overwritten method should call the tested procedure once. 
/// User should call the public 'run' method for functionality test, or 'benchmark' to tetermine calls per second
/// by default, these methods invole protected loaded_iter once or many times
/// 
class TestCase
{
public:
    struct Log
    {
        const TestCase* test_;
        bool success_;
        Log (const TestCase* test = NULL, bool success = false)
        :
        test_ (test),
        success_ (success)
        {
        }
    };
    typedef std::vector <Log> TestLog;
private:
#if defined (TEST_BENCH_COMBINED)
    static const longlong DEF_TEST_DURATION = 100000; // approx. time in micrseconds (usec) over which the statistics is to be collected == 0.1 sec
#endif
    struct Err
    {
        int line_;
        const char* file_;
        std::string message_;
    };
    /// name of the test, identifies it within the test suite 
    const char* name_;
    bool name_indivisible_;
    TestCase* base_;
    typedef std::vector <TestCase*> TestCaseVec;
    TestCaseVec subords_;
    TestFacetSet facets_;
    FaSet external_;
    typedef std::vector <Err> Evec;
    Evec errors_; // list of errors reported during 'run' call
    static bool stop_on_error_; // flag specifying error reaction
    static unsigned test_count_;
    static unsigned failure_count_;
    static TestLog test_log_;
    void report_errors ();
protected:
    /// stream where test output is sent; nullstream by default
    static StreamWrap o_; 
    /// error reporting helper, stores message to internal error storeage
    void err (const char* filename, int lineno, const char* message = NULL); // helper for error reporting
    /// stores error with message if condition fails
    #define TEST_ASSERTX(cond,msg) if (!(cond)) {err (__FILE__, __LINE__, msg " : " #cond);}
    /// stores error if condition fails
    #define TEST_ASSERT(cond)     if (!(cond)) {err (__FILE__, __LINE__, "Condition failed : " #cond);}
    /// stores error message
    #define TEST_FAILURE(msg)     err (__FILE__, __LINE__, (msg))
    /// acquires / creates resources needed for testing.
    /// 
    /// Called by 'run'. 
    /// Should install facets needed for testing and add them to the facets_ set.
    /// \return boolean. On failure, may either return false or throw any exception
    virtual bool init    () { return true; }
    /// actually performs the test.
    /// Called by 'run'.
    /// should be overloaded for more specific testing
    virtual bool process () = 0;
    /// releases / destroys any resources captured or created for testing.
    /// Called by 'run' 
    /// should release any resources captured by 'init'
    /// this method is called regardless of the success of 'init' or 'process'
    virtual bool cleanup ();
    /// optional post-test user-defined report.
    /// called after 'process' is run (regardless of the success of the test); this is a place where 
    /// additional reporting can be implemented
    virtual void report  () {}
    /// a wrapper that calls the 'process' method. 
    /// By default, does nothing else but calls 'process'.
    /// can be overwritten for more rigorous exception handling (the run method by default recognizes
    /// only std::exception, all others are caught by ellipsis clause)
    /// should return false if detects failure..
    virtual bool wrap_process ();
    /// adds facet to a list of associated facets
    /// \param facet      The one to add 
    /// \param external   If true, the facet will not be destroyed upon TestCase destruction
    void add_facet (TestFacet* facet, bool external = false);
    /// access method for test facets
    /// \return root facet in the facet tree associated with the test
    TestFacetSet& facets () { return facets_; }
    /// searches for a facet of a given name in this and base test cases up to the root
    TestFacet* find_facet (const char* name);
    /// controls information output at test beginning
    virtual void print_header ();
    /// controls information output upon test completion
    virtual void print_footer (bool success, bool subords_success, time_t init_time, time_t beg_time, time_t done_time, time_t cleanup_time);
public:
    /// constructor.
    /// lightweight, just stores name and (optionally) output stream name
    /// \param  name      test name, forst word serves as unique test identifier and used by TestSuite's arguments-driven test selection mechanism
    /// \param  ostr      std::stream to direct the test output to; by default, no output is produced
    TestCase (const char* name, bool name_is_indivisible = false);
    /// destructor (virtual)
    /// ensures proper destruction of derived objects
    virtual ~TestCase ();
    /// returns the name of the test, set by the constructor
    /// \return           test name as pointer to a zero-terminated char string
    const char* name () const {return name_;}
    /// redirects output to a given stream
    /// \param  o         std::ostream to send the output to
    static void set_stream (std::ostream& o);
    /// gets stop-on-error flag
    /// \return flag status
    static bool stop_on_error () { return stop_on_error_; }
    /// sets stop-on-error flag
    /// \return original flag status
    static bool stop_on_error (bool flag);
    /// runs main testing method.
    /// should be called by the client directly.
    /// calls init, then process, then cleanup; sends reports to the o_ stream
    /// reports all remembered error messages stored by 'err' or TEST_* macros
    bool run (const TestSelection* selection = NULL);
    /// adds subordinate test to this one
    void add_subord (TestCase* testCase);
    /// accessor for the total test count
    static unsigned total_count () { return test_count_; }
    /// returns nestlevel
    unsigned nest_level () const;
    /// return fully qualified name, including names for all base cases, slash-separated
    std::string full_name () const;
    /// clears the log
    static void clear_log ();
    /// access to test log
    static const TestCase::TestLog& log ();
    /// print
    void print (std::ostream& oostr, unsigned nest = 0) const;
};
///
/// Simple class derived from TestCase - for testint the coherence of the test facility
class TestTestCase : public TestCase
{
public:
    TestTestCase () : TestCase ("TestTestCase") {}
    bool process ()
    {
        TEST_ASSERTX (1 == 1,"condition failed");
        TEST_ASSERT (2 == 2);
        return true;
    }
};

#endif // __test_case_h__
