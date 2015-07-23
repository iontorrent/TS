/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __benchmark_h__
#define __benchmark_h__

/// \file
/// Benchmarking facility;  measures timing performance of the given procedure
/// Declares abstract Benchmark class, and the Bench class, which is a panel 
/// of benchmarks, parametrized by the user. 
/// The user should derive concrete benchmarks from the Benchmark class 
/// The natural use for benchmarks are from test suite


#include <set>
#include "platform.h"
#include "test_facet.h"

extern const unsigned USECS_IN_SEC;
extern const unsigned NSECS_IN_USEC;
extern const unsigned NSEQS_IN_SEC;

/// encapsulates benchmarking functionality
///
/// measuring speed (in operations per second) for the desired operation.
/// benchmarks may optionally share the fixtures with test cases or other benchmarks. 
class Benchmark
{
private:
    static const longlong DEF_TEST_DURATION = 100000; // approx. time in micrseconds (usec) over which the statistics is to be collected == 0.1 sec
    const char* name_;
    TestFacetSet facets_;
    FaSet external_facets_;
protected:
    /// single invocation of the tested procedure.
    /// should be overloaded and perform one iteration of tested procedure
    /// overloading just this method (out of entire TestCase) is enough for minimal default testing to work
    /// called once by default 'process', and called repeatedly from 'benchmark'
    virtual bool loaded_iter () = 0;
    /// single invocation of all collateral actions of 'loaded_iter' without actual invocation of the tested procedure.
    /// used to compute the run-time of the collateral actions of loaded_iter
    /// that are not part of the tested procedure. 
    /// The time spent in the dummy_iter is subtracted from the time spent in the loaded_iter
    /// by 'benchmark' method
    virtual bool dummy_iter ();
    /// measures time in micro-seconds spent on running the 'proc' method for 'repetitions' number of times
    /// This method should be called from 'process' if timing is desired.
    //  It provides a fine grained control over the benchmarking then 'benchmark' method
    /// \param repetitions Number of time to run the 'loaded' method
    /// \param proc        Method to be timed, default is TestCase::loaded_iter
    /// \return            time in microseconds; -1 if proc returns false
    longlong timing (unsigned repetitions, bool (Benchmark::*proc)() = &Benchmark::loaded_iter); 
    /// determines approximate number of repetitions of 'proc' that would take at least 'duration' microseconds.
    /// This method should be called from 'process' if calibration is desired.
    //  It provides a fine grained control over the benchmarking then 'benchmark' method
    /// \param proc        Method to be calibrated, default is TestCase::loaded_iter
    /// \param duration    Minimal time, in microseconds, for which the needed number of iterations is determined
    /// \return            number of repetitions or 0 if proc returns false
    unsigned calibrate (bool (Benchmark::*proc) () = &Benchmark::loaded_iter, ulonglong duration = DEF_TEST_DURATION); // returns 0 on error, number of repetitions running in 'duration' ms on success
    /// accessor for facets; the facets container can not be modified directly
    /// \return const reference to facets container
    TestFacetSet& facets () { return facets_; } 
    /// adds facet to a list of associated facets
    /// \param facet      The one to add 
    void add_facet (TestFacet* facet, bool external = false);
public:
    /// default constructor. Makes standalone benchmark object
    Benchmark (const char* name); 
    /// Constructor. Creates benchmark with one given facet in facets_.
    /// This facet is not  destroyed upon destruction of benchmark object
    Benchmark (const char* name, TestFacet& facet);
    /// Destructor (virtual): ensures proper destructor chaining;
    /// also makes sure only owned facets are destroyed
    ~Benchmark ();
    /// measures speed (runs per second) of tested code.
    /// Computes speed as loaded_iter calls per second minus dummy_iter calls per second.
    /// This method should be called from 'process' if benchmarking is desired.
    /// \param duration    Approximate duration of the test run, in microseconds (default 100000 ms == 0.1 sec)
    /// \return            Speed in iterations per second as double
    double run (ulonglong duration = DEF_TEST_DURATION); 
    /// access to name
    const char* name () const { return name_;}
};

#endif // __benchmark_h__
