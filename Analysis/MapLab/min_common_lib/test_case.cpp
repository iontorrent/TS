/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#define __test_case_cpp__
#include <iomanip>
#include <cassert>
#include <ctime>
#include <stdexcept>
#include "test_case.h"
#include "test_selection.h"

static const char* def_hdr = "Unnamed";
static const char* tststr = "test";
static const char* reppr  =  "|   ";

StreamWrap TestCase::o_ = nullstream;
bool TestCase::stop_on_error_ = false;
unsigned TestCase::test_count_ = 0;
unsigned TestCase::failure_count_ = 0;
TestCase::TestLog TestCase::test_log_;

void TestCase::report_errors ()
{
    if (!errors_.size ())
        return;
    o_ << reppr << errors_.size () << " errors recorded:" << std::endl;
    for (Evec::iterator i = errors_.begin (), sent = errors_.end (); i != sent; ++i)
    {
        o_ << reppr << i->file_ << ":" << i->line_;
        if (i->message_.size ())
            o_ << " " << i->message_;
        o_ << std::endl;
    }
}

static const char RUNTESTSTR [] = "Running test: ";
void TestCase::print_header ()
{
    std::ostringstream s;
    s << RUNTESTSTR << name_ << " [" << full_name () << "]";
    unsigned ll = s.str ().length () + strlen (reppr);
    o_ << "\n" << std::setfill ('=') << std::setw (ll) << "" << std::setfill (' ') << "\n";
    o_ << reppr << s.str ();
    o_ << "\n" << std::setfill ('-') << std::setw (ll) << "" << std::setfill (' ') << std::endl;
}

void TestCase::print_footer (bool success, bool subords_success, time_t init_time, time_t beg_time, time_t done_time, time_t cleanup_time)
{
    std::ostringstream s;
    s << (full_name ()) << tststr << " finished in " << done_time - beg_time << " sec. (initialization took " << beg_time - init_time << " sec, cleanup took " << cleanup_time - done_time << " sec)";
    unsigned ll = s.str ().length () + strlen (reppr);
    o_ << "\n" << std::setfill ('-') << std::setw (ll) << "" << std::setfill (' ') << "\n";
    o_ << reppr << s.str ();
    report_errors ();
    if (!success)
        o_ << reppr << "Test failed" << "\n";
    if (!subords_success)
        o_ << reppr << "Some of the subordinate tests failed" << "\n";
    o_ << "\n" << std::setfill ('=') << std::setw (ll) << "" << std::setfill (' ') << std::endl;
}

void TestCase::err (const char* filename, int lineno, const char* message)
{
    errors_.resize (errors_.size () + 1);
    errors_.rbegin ()->file_ = filename;
    errors_.rbegin ()->line_ = lineno;
    errors_.rbegin ()->message_ = message?message:"";
    if (stop_on_error_)
    {
        std::ostringstream stro;
        stro << "Stop on error enabled. Error occured: in file " << filename << ", line " << lineno << ": " << (message?message:"") << std::endl;
        throw std::runtime_error (stro.str ().c_str ());
    }
}

TestCase::TestCase (const char* name, bool name_indivisible)
:
name_ (name?name:def_hdr),
name_indivisible_ (name_indivisible),
base_ (NULL),
facets_ (name?name:def_hdr)
{
}
TestCase::~TestCase ()
{
    for (TestFacetSet::FVIter itr = facets_.begin (), sent = facets_.end (); itr != sent; ++ itr)
    {
        if (external_.count (*itr))
            *itr = NULL;
    }
}
bool TestCase::wrap_process () // overload in derived class to provide more then default exception processing
{
    return process ();
}
bool TestCase::run (const TestSelection* selected_tests)
{
    bool success = true, subords_success = true;
    time_t init_time = time (NULL);

    test_log_.resize (test_log_.size () + 1);
    unsigned log_pos = test_log_.size () - 1;

    print_header ();
    errors_.clear ();
    ++ test_count_;
    // perform fixtures initialization
    try
    {
        success = init ();
    }
    catch (std::exception& e)
    {
        o_ << reppr << "Error: " << e.what () << " while initializing test " <<  name () << std::endl;
        success = false;
    }
    catch (...)
    {
        o_ << reppr << "Error: unhandled exception while initializing test " << name () << std::endl;
        success = false;
    }
    // run the test
    time_t beg_time = time (NULL);
    if (success)
    {
        try
        {
            success = wrap_process ();
            if (success)
                success = (errors_.size () == 0);
        }
        catch (std::exception& e)
        {
            o_ << reppr << "Error: " << e.what () << " while running test  " <<  name () << std::endl;
            success = false;
        }
        catch (...)
        {
            o_ << reppr << "Error: unhandled exception while running test " << name () << std::endl;
            success = false;
        }
        report ();
        if (success || !stop_on_error_)
            for (TestCaseVec::iterator itr = subords_.begin (), sent = subords_.end (); itr != sent; ++itr)
            {
                const TestSelection* match = NULL;
                if (!selected_tests || selected_tests->includes ((*itr)->name (), match))
                {
                    if (!(*itr)->run (match))
                    {
                        subords_success = false;
                        if (stop_on_error_)
                            break;
                    }
                }
            }
    }
    time_t done_time = time (NULL);
    // perform cleanup
    try
    {
        success &= cleanup ();
    }
    catch (std::exception& e)
    {
        o_ << reppr << "Error: " << e.what () << " while cleaning up after test  " <<  name () << std::endl;
        success = false;
    }
    catch (...)
    {
        o_ << reppr << "Error: unhandled exception while cleaning up after test " << name () << std::endl;
        success = false;
    }
    time_t cleanup_time = time (NULL);
    print_footer (success, subords_success, init_time, beg_time, done_time, cleanup_time);
    test_log_ [log_pos].test_ = this;
    test_log_ [log_pos].success_ = success & subords_success;
    return success & subords_success;
}

bool TestCase::cleanup ()
{ 
    for (TestFacetSet::FVIter itr = facets_.begin (), sent = facets_.end (); itr != sent; ++ itr)
    {
        if (!external_.count (*itr))
            delete *itr;
        *itr = NULL;
    }
    return true; 
}


void TestCase::add_facet (TestFacet* facet, bool external)
{
    facets_.add (facet);
    if (external)
        external_.insert (facet);
}

TestFacet* TestCase::find_facet (const char* nm)
{
    TestFacet* r = facets_.find (nm);
    if (!r && base_)
        r = base_->find_facet (nm);
    return r;
}

void TestCase::add_subord (TestCase* testCase)
{
    testCase->base_ = this;
    subords_.push_back (testCase);
}

bool TestCase::stop_on_error (bool flag)
{
    bool rv = stop_on_error_; 
    stop_on_error_ = flag; 
    return rv;
}

void TestCase::set_stream (std::ostream& ostr)
{
    o_ = ostr;
}

unsigned TestCase::nest_level () const
{
    return base_ ? (1 + base_->nest_level ()) : 0;
}

// inefficient but  this is Ok - it is just a reporting convenience method
std::string TestCase :: full_name () const
{
    std::string result;
    if (base_)
    {
        result.append (base_->full_name ());
        result.append (TestFacet::separator);
    }
    const char* pname = name_ ? name () : def_hdr;
    if (name_indivisible_)
        result.append (pname);
    else
    {
        const char* nend = pname;
        while (*nend && !isspace (*nend))
            ++nend;
        result.append (pname, nend - pname);
    }
    return result;
}

void TestCase :: clear_log ()
{
    test_count_ = failure_count_ = 0;
    test_log_.clear ();
}

const TestCase::TestLog& TestCase::log ()
{
    return test_log_;
}

void TestCase::print (std::ostream& ostr, unsigned nest) const
{
    ostr << std::setw (nest * 2) << "" << std::setw (0) << full_name () << "\n";
    nest += 1;
    for (TestCaseVec::const_iterator itr = subords_.begin (), sent = subords_.end (); itr != sent; ++itr)
        if ((*itr))
            (*itr)->print (ostr, nest);
}
