/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_benchmark_h__
#define __test_benchmark_h__
#include <extesting.h>
#include <benchmark.h>
#include <iomanip>
#include <unistd.h>

class TestBenchmark : public Test
{
    class TestBm : public Benchmark
    {
    public:
        static const unsigned sleep_mksec = 5500; // 5.5ms
        TestBm ()
        :
        Benchmark ("SimulatedBenchmark : mock-up benchmark for operability testing purposes")
        {
        }
        bool loaded_iter ()
        {
            usleep (sleep_mksec); 
            return true;
        }
    };
public:
    TestBenchmark ()
    :
    Test ("TestBenchmark : benchmarking capabilities")
    {
    }
    bool process ()
    {
        TestBm benchmark;
        double speed = benchmark.run ();
        double expected = double (USECS_IN_SEC) / TestBm::sleep_mksec;
        o_ << "Rated at " << std::fixed << std::setprecision (2) << speed << " turnovers per second\n";
        o_ << "Expected " << std::fixed << std::setprecision (2) << expected << " turnovers per second\n";
        return true;
    }
};
#endif // __test_benchmark_h__
