/*
 *  Created on: 8-31-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49961 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:31:15 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef LOGGING_UTIL_HPP
#define LOGGING_UTIL_HPP

#include <iostream>

/*!
    lifetechnologies namespace
*/
namespace lifetechnologies
{
/*!
    lifetechnologies::logging namespace
*/
namespace logging
{

/*!
Print a backtrace.

    This will produce output similar to the following:

    \verbatim
    ./my_app(__gxx_personality_v0+0x3cb) [0x401363]
    /lib64/tls/libc.so.6 [0x359c22e300]
    ./my_app(__gxx_personality_v0+0x457) [0x4013ef]
    ./my_app [0x4014aa]
    /lib64/tls/libc.so.6(__libc_start_main+0xdb) [0x359c21c40b]
    ./my_app(__gxx_personality_v0+0x72) [0x40100a]
    \endverbatim

     The addresses in the brackets will vary of course.
     If you have a debug build of my_app then you can use 'addr2line' to
     convert address to function, file, & line number.

    \verbatim
    $>  addr2line -C -f -e my_app 0x4013ef
    \endverbatim

     produces something like:

    \verbatim
    my_app()
    /home/bifx/me/bioscope/trunk/my_app/src/my_app.cpp:39
    \endverbatim
*/
void print_backtrace(std::ostream &out = std::cerr, unsigned int max_frames = 63);

/*!
Initialize the log4cxx logging.

Uses the specified configuration file name.  If the config file is not specified then will attempt to initialize
according to the following steps:

-# First check for a configuration file set via the environment variable LOG4CXX_CONFIGURATION
-# Then check in the current working directory for configuration files named; "log4cxx.xml", "log4cxx.properties", "log4j.xml" and "log4j.properties".
-# Use stderr

\exception std::invalid_argument
*/
void initialize_logging(const char*config_filename = NULL);

/*!
Initialize the logging of unhandled exceptions and segmentation faults.

Output goes to a log unlsee use_stderr is set to true.
*/
void inititialize_terminate_logging(bool use_stderr = false);

} // namespace logging
} //namespace lifetechnologies

#endif // LOGGING_UTIL_HPP
