/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/* -*- c++ -*- */
#ifndef __benchmark_h__
#define __benchmark_h__ 1

#include <ctime>
#include <cassert>
#include <fstream>
#include <string>

#include <sys/resource.h>
#include <sys/time.h>

static const long int ONE_SECOND = 1000000;

class Benchmark {

public:
	Benchmark()
		:state(uninitialized)
	{
		// RAII
		// start();
	}

	void start()		// aka restart.
	{
		state = running;
		// Start resets internal counters
		//_mark();
		//t1 = _getTime();
		c0 = clock();
		gettimeofday(&w0,0);
		split(); // Initialize split counters, too.
	}

	void stop()
	{
		assert(state == running);
		split();
		state = stopped;
	}

	void split()
	{
		if (state == running) {
			//_mark();
			//t2 = _getTime();
			c1 = clock();
			gettimeofday(&w1,0);
		}
	}

	long double currentMemory()
	{
		assert(state != uninitialized);
		getrusage(RUSAGE_SELF, &ru);
		//_mark(); // Refresh rusage == always current
		return _getMemory();
	}

	// Since last split
	// Since last split
	long double elapsed_cputime()
	{
		assert(state != uninitialized);
		return static_cast<long double>(c1 - c0) / CLOCKS_PER_SEC;
		//return t2 - t1; // CPUTIME from RUSAGE
	}

	long double elapsed_walltime()
	{
		assert(state != uninitialized);
		//return t2 - t1;
		//return static_cast<long double>(c1 - c0) / CLOCKS_PER_SEC;
		//return static_cast<long double>( difftime(w1, w0);
		long double sec = w1.tv_sec - w0.tv_sec;
		long double usec = w1.tv_usec - w0.tv_usec;
		// Handle overflow / underflow
		if (usec < 0) {
		    usec += ONE_SECOND;
		    sec--;
		} else if (usec >= ONE_SECOND) {
		    usec -= ONE_SECOND;
		    sec++;
		}
		return sec + usec/ONE_SECOND;
	}

	// compat
	long double elapsed() { return elapsed_cputime(); }


	// Refactor repeated print statements for throughput reporting
	// Should be outside class, all public methods
	void reportThroughput(std::ostream & out, const char *prefix, const uint32_t count)
	{
		out << std::fixed;
		out.precision(2);
		out << prefix << " Benchmark: " << count << " in "
			<< " Time: [CPU] " << elapsed_cputime() << " [WALL] " << elapsed_walltime()
			<< " Throughput: " << (double) count  / elapsed_cputime() << "/s"
			<< " Memory: " << currentMemory() << " MB" << std::endl;
	}

private:
	// Prevent copying
	Benchmark(Benchmark&) { }
	void operator=(Benchmark&) { }
	
// 	void _mark()
// 	{
// 		getrusage(RUSAGE_SELF, &ru);
// 	}

	// Time in seconds, with decimal ms
	long double _getTime()
	{
		// Combining user time with system time
		double sec = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec;
		double ms  = ru.ru_utime.tv_usec + ru.ru_stime.tv_usec;		

		// System time only!
		//double sec = ru.ru_stime.tv_sec;
		//double ms  = ru.ru_stime.tv_usec;

		return sec + ms/ONE_SECOND;
	}

	// Memory in MB
	long double _getMemory()
	{
		double vm_usage		= 0.0;

#ifdef __linux__
		// LINUX only
		using std::ios_base;
		using std::ifstream;
		using std::string;


		// 'file' stat seems to give the most reliable results
		ifstream stat_stream("/proc/self/stat",ios_base::in);

		// dummy vars for leading entries in stat that we don't care about
		string pid, comm, state, ppid, pgrp, session, tty_nr;
		string tpgid, flags, minflt, cminflt, majflt, cmajflt;
		string utime, stime, cutime, cstime, priority, nice;
		string O, itrealvalue, starttime;

		// the two fields we want
		unsigned long vsize;
		long rss;

		stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
					>> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
					>> utime >> stime >> cutime >> cstime >> priority >> nice
					>> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

		vm_usage	 = vsize / 1024.0;

		// FIXME - compute actual resident memory size
		//long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
		//double resident_set = rss * page_size_kb;

#else // __linux__ -- Non linux POSIX environments (aka BSD/OSX)
		// Yes, the above is this easy on BSD systems.
		vm_usage = ru.ru_maxrss / 1024.0;
#endif // __linux__

		//return size in MB
		return (vm_usage / 1024.0);

	}

	enum { uninitialized, running, stopped } state;

	//long double t0, t1;
	struct rusage ru;
	// System Time
	clock_t c0, c1;
	// Wall Time
	struct timeval w0, w1;
};
#endif
