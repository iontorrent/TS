/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "benchmark.h"
#include <stdexcept>
#include <ctime>

// #define CALIBR_DBG

#ifdef CALIBR_DBG
#include <iostream>
#endif

const unsigned USECS_IN_SEC  = 1000000;
const unsigned NSECS_IN_USEC = 1000;
const unsigned NSEQS_IN_SEC = NSECS_IN_USEC * USECS_IN_SEC;

static const char* clockerr = "System clock access error";

// returns time difference in micro-seconds
static longlong clock_diff (timespec& t1, timespec& t2)
{
    longlong nsdiff = ((longlong) (t1.tv_sec - t2.tv_sec)) * NSEQS_IN_SEC;
    nsdiff += t1.tv_nsec;
    nsdiff -= t2.tv_nsec;
    return nsdiff / NSECS_IN_USEC;
}


Benchmark::Benchmark (const char* name)
:
name_ (name),
facets_ (name)
{
}
Benchmark::Benchmark (const char* name, TestFacet& facet)
:
name_ (name),
facets_ (name)
{
    add_facet (&facet, true);
}
Benchmark::~Benchmark ()
{
    for (TestFacetSet::FVIter itr = facets_.begin (), sent = facets_.end (); itr != sent; ++ itr)
    {
        if (external_facets_.count (*itr))
            *itr = NULL;
    }
}
bool Benchmark::dummy_iter () 
{
    return true; 
}
longlong Benchmark::timing (unsigned repetitions, bool (Benchmark::*proc)())
{
    timespec clock_before, clock_after;
    longlong elapsed;
    // measure the loaded time
    if (clock_gettime (CLOCK_MONOTONIC, &clock_before) != 0)
        throw std::runtime_error (clockerr);
    for (unsigned itr = 0; itr != repetitions; ++itr)
        if (!(this->*proc) ())
            return -1;
    if (clock_gettime (CLOCK_MONOTONIC, &clock_after) != 0)
        throw std::runtime_error (clockerr);
    elapsed = clock_diff (clock_after, clock_before);
    return std::max (elapsed, (longlong) 0);
}
unsigned Benchmark::calibrate (bool (Benchmark::*proc) (), ulonglong duration) // returns 0 on error, number of repetitions running in 'duration' ms on success
{
    ulonglong repetitions = 1, iter;
    longlong elapsed;
    timespec clock0, clock1;
    for (;;)
    {
        if (clock_gettime (CLOCK_MONOTONIC, &clock0) != 0)
            throw std::runtime_error (clockerr);
        for (iter = 0; iter != repetitions; ++iter)
            if (!(this->*proc) ()) 
                return 0;
        if (clock_gettime (CLOCK_MONOTONIC, &clock1) != 0)
            throw std::runtime_error (clockerr);
        elapsed = clock_diff (clock1, clock0);
#ifdef CALIBR_DBG
        std::cerr << "\r  Calibrated for " << repetitions << " in " <<  elapsed << " usec" << std::endl;
#endif
        if (elapsed <= 0)
            repetitions *= duration;
        else if ((unsigned) elapsed * 2 <= duration) // duration/elapsed >= 2, so that multiplication makes sense
        {
            repetitions *= duration;
            repetitions /= elapsed;
        }
        else if ((unsigned) elapsed < duration)
        {
            repetitions += repetitions - elapsed * repetitions / duration;
        }
        else if (repetitions > 2 && (unsigned) elapsed >= duration * 2)
        {
            repetitions *= duration;
            repetitions /= elapsed;
        }
        else
            break;
    }
#ifdef CALIBR_DBG
        o_ << "Calibration result is " << repetitions << " in " << elapsed  << " usec (requested duration " << duration << " usec)" << std::endl;
#endif
    return (unsigned) repetitions;
}
double Benchmark::run (ulonglong duration)
{
    // determine how many times to run a test
    unsigned repetitions = calibrate (&Benchmark::loaded_iter, duration);
    if (repetitions == 0)
        return 0;

    // measure the loaded time
    longlong elapsed_loaded = timing (repetitions, &Benchmark::loaded_iter);
    if (elapsed_loaded < 0)
        return 0;

    // measure the dummy time
    longlong elapsed_dummy = timing (repetitions, &Benchmark::dummy_iter);
    if (elapsed_dummy < 0)
        return 0;

    // subtract dummy from loaded time
    longlong elapsed;
    if (elapsed_dummy != 0 && elapsed_dummy < elapsed_loaded)
        elapsed = elapsed_loaded - elapsed_dummy;
    else
        elapsed = elapsed_loaded;

    // elapsed is in microsecs, scale the speed
    return ((double) repetitions) * USECS_IN_SEC / elapsed;
}

void Benchmark::add_facet (TestFacet* facet, bool external)
{
    facets_.add (facet);
    if (external)
        external_facets_.insert (facet);
}

