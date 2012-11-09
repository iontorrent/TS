/*
 *  Created on: 08-31-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: $
 *  Last changed by:  $Author: $
 *  Last change date: $Date: $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <log4cxx/logger.h>
#include <log4cxx/log4cxx.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <lifetech/logging/util.hpp>

using namespace std;
using namespace lifetechnologies;

void print_usage()
{
    cout << "usage: logging_example [signal, exception]" << endl;
}

void signal_example()
{
    // in this example we are using the inititialize_terminate_logging(true) which prints the backtrace to stderr
    //    we could have used the inititialize_terminate_logging(false) to log instead

    logging::initialize_logging();
    logging::inititialize_terminate_logging(true);

    int *foo = NULL;
    *foo = 42;  // force a seg fault

    // This will produce output similar to the following:
    //    ./logging_example(__gxx_personality_v0+0x3cb) [0x401363]
    //    /lib64/tls/libc.so.6 [0x359c22e300]
    //    ./logging_example(__gxx_personality_v0+0x457) [0x4013ef]
    //    ./logging_example [0x4014aa]
    //    /lib64/tls/libc.so.6(__libc_start_main+0xdb) [0x359c21c40b]
    //    ./logging_example(__gxx_personality_v0+0x72) [0x40100a]
    //
    // The addresses in the brackets will vary of course.
    // If you have a debug build of logging_example then you can use 'addr2line' to
    // convert address to function, file, & line number.
    //
    //    $>  addr2line -C -f -e logging_example 0x4013ef
    //
    // produces something like:
    //    signal_example()
    //    /home/bifx/moultoka/repos/corona/bioscope/trunk/common/c++/lifetech/examples/logging/logging_example.cpp:39
}

void exception_example()
{
    // In this example we will be logging the backtrace
    logging::initialize_logging();
    logging::inititialize_terminate_logging();
    throw 0; // throw an exception
}


int main (int argc, char *argv[])
{
    if (argc != 2)
    {
        print_usage();
        exit(0);
    }

    string mode = argv[1];

    if (mode == "signal")
        signal_example();
    else if (mode == "exception")
        exception_example();

    return 0;
}

