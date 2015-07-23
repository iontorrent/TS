/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_suite.h"
#include "test_args.h"
#include "test_selection.h"
#include <cstdlib>


static const char* VERBOSE_OPT = "--verbose";
static const char* SILENT_OPT = "--silent";
static const char* CRUEL_OPT = "--cruel";
static const char* HELP_OPT = "--help";


TestSuite::TestSuite (int argc, char* argv [], char* envp [])
:
o_ (std::cout),
verbose_ (false),
silent_ (false),
stop_on_error_ (false)
{
    testEnv.init (argc, argv, envp);
    // derive name from command line
    if (argc)
    {
        std::string t = argv [0];
        size_t pos = t.find_last_of ("/");
        if (pos == std::string::npos)
            name_ = t;
        else
            name_.assign (t, pos+1, t.length () - pos - 1);
    }
    else
        name_ = "TEST";
}
void TestSuite::set_output (std::ostream&  ostr)
{
    o_ = ostr;
}
bool TestSuite::stop_on_error (bool val)
{
    bool rv = stop_on_error_;
    stop_on_error_ = val;
    return rv;
}
bool TestSuite::stop_on_error () const
{
    return stop_on_error_;
}
void TestSuite::reg (TestCase& test_case)
{
    test_cases_.push_back (&test_case);
}

bool TestSuite::run ()
{
    // check if selective test requested
    time_t begt = time (NULL);
    // typedef std::vector<const char*> CPV;
    bool tests_gathered = false;
    bool help_requested = false;
    TestSelection selected_tests;
    for (int ano = 1; ano != testEnv.argc_; ++ ano)
    {
        if (testEnv.argv_ [ano][0] == '-')
        {
            tests_gathered = true;
            if (!strcmp (testEnv.argv_ [ano], VERBOSE_OPT))
                verbose_ = true;
            else if (!strcmp (testEnv.argv_ [ano], SILENT_OPT))
                silent_ = true;
            else if (!strcmp (testEnv.argv_ [ano], CRUEL_OPT))
                stop_on_error_ = true;
            else if (!strcmp (testEnv.argv_ [ano], HELP_OPT))
                help_requested = true;
        }
        if (!tests_gathered)
            selected_tests.increment (testEnv.argv_ [ano]); // implicit conversion
    }
    TestCase::set_stream (verbose_ ? (std::ostream&) o_ : nullstream);
    TestCase::stop_on_error (stop_on_error_);
    TestCase::clear_log ();
    init ();

    if (help_requested)
    {
        o_ << "This is a test suite " << name_ << "\n";
        o_ << "The suite executes test cases; if all succeed, executable exits with status zero\n";
        o_ << "To control test suite from command  line, use following format:\n";
        o_ << testEnv.argv_ [0] << " [test1 [test2 ...]] [options]\n";
        o_ << "  Supported options are:\n";
        o_ << "    " << VERBOSE_OPT << " : output all logging messages from test cases and benchmarks\n";
        o_ << "    " << SILENT_OPT <<  " : suppress printing out testing results summary\n";
        o_ << "    " << CRUEL_OPT <<   " : stop processing after first failed test\n";
        o_ << "  By default, all tests are performed\n";
        o_ << "  For any composite test, if no subordinates are specified, all of them are performed\n";
        o_ << "  Available tests (" << test_cases_.size () << " top level ones):\n";
        for (TestVect::iterator titr = test_cases_.begin (), sent = test_cases_.end (); titr != sent; ++titr)
            (*titr)->print (o_, 2);
        o_ << std::flush;
        exit (0);
    }
    if (!silent_ && verbose_)
        o_ << "\nSelected tests list:\n" << selected_tests << "End of Selected tests list" << std::endl;

    unsigned toplevel_tests_count = 0;
    unsigned toplevel_success_count = 0;

    for (TestVect::iterator titr = test_cases_.begin (), sent = test_cases_.end (); titr != sent; ++titr)
    {
        const TestSelection* match = NULL;
        if (selected_tests.includes ((*titr)->name (), match))
        {
            ++toplevel_tests_count;
            if ((*titr)->run (match))
                ++toplevel_success_count;
            else if (stop_on_error_)
                break;
        }
    }
    cleanup ();
    time_t endt = time (NULL);
    // count successes and failures
    unsigned tot_failures = 0;
    typedef TestCase::TestLog::const_iterator LogItr;;
    for (LogItr citr = TestCase::log ().begin (), sent = TestCase::log ().end (); citr != sent; ++citr)
        if (!(*citr).success_) 
            ++tot_failures;
    unsigned tot_successes = TestCase::log ().size () - tot_failures;
    if (!silent_)
    {
        o_ << "\n" << toplevel_tests_count << " top-level";
        if (toplevel_tests_count != TestCase::total_count ())
            o_ << " and " << TestCase::total_count () - toplevel_tests_count << " subordinate tests performed (" << TestCase::total_count () << " total)";
        else
            o_ << " tests performed";
        o_ << "\n" << toplevel_success_count << " top-level tests succeeded, " << toplevel_tests_count - toplevel_success_count << " failed";
        o_ << "\n" << tot_successes << " tests succeeded, " << tot_failures << " failed\n";
        if (tot_successes)
        {
            o_ << "\nSucceeded tests:";
            for (LogItr citr = TestCase::log ().begin (), sent = TestCase::log ().end (); citr != sent; ++citr)
            {
                if ((*citr).success_)
                {
                    const TestCase& test = *((*citr).test_);
                    o_ << "\n    " << test.full_name ();
                }
            }
        }
        if (tot_failures)
        {
            o_ << "\nFailed tests:";
            for (LogItr citr = TestCase::log ().begin (), sent = TestCase::log ().end (); citr != sent; ++citr)
            {
                if (!(*citr).success_)
                {
                    const TestCase& test = *((*citr).test_);
                    o_ << "\n    " << test.full_name ();
                }
            }
        }

        o_ << "\nExecution time " << endt - begt << " sec" << std::endl;
    }
    return (tot_failures == 0);
}

void TestSuite :: name (const char* new_name)
{
    name_ = new_name;
}

const char* TestSuite :: name () const
{
    return name_.c_str ();
}
