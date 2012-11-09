/*
 *  Created on: 9-1-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49961 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:31:15 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdlib>
#include <stdexcept>
#include <execinfo.h>
#include <signal.h>
#include <string>
#include <sstream>
#include <fstream>
#include <execinfo.h>
#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/log4cxx.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <lifetech/logging/util.hpp>

namespace lifetechnologies
{
namespace logging
{

void initialize_logging(const char*config_filename)
{
    // set up logger
    if (config_filename)
    {
        std::ifstream file(config_filename, std::ios::in | std::ios::binary);
        if (file.fail())
        {
            std::stringstream sstrm;
            sstrm << "Unable to open file " << config_filename;
            throw std::invalid_argument(sstrm.str());
        }
        // found the config file so use it
        file.close();
        log4cxx::PropertyConfigurator::configure(config_filename);
    }
    else
    {
        log4cxx::LoggerPtr root = log4cxx::Logger::getRootLogger();
        root->isDebugEnabled();  // NOTE: this will kick off the default log4cxx initialization procedure
                                 //       (see http://logging.apache.org/log4cxx/index.html)

        log4cxx::AppenderList const& appenders = root->getAllAppenders();
        if (appenders.size() == 0)
        {
            // if we have no appenders then no config was found by the default log4cxx initialization procedure
            //     so, we'd prefer to use stderr as a last resort.  So, set that up here
            log4cxx::ConsoleAppenderPtr appender(new log4cxx::ConsoleAppender());
            appender->setTarget(LOG4CXX_STR("System.err"));
            appender->setName("stderr");
            log4cxx::LayoutPtr layout(new log4cxx::PatternLayout("%m%n"));
            appender->setLayout(layout);
            log4cxx::helpers::Pool pool;
            appender->activateOptions(pool);
            root->addAppender(appender);
        }
    }
}

void print_backtrace(std::ostream &out, unsigned int max_frames)
{
    // storage array for stack trace address data
    void* addrlist[max_frames+1];

    // retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if (addrlen == 0)
    {
        out << "backtrace empty" << std::endl;;
        return;
    }

    // resolve addresses into strings containing "filename(function+address)",
    char** symbol_list = backtrace_symbols(addrlist, addrlen);

    size_t symbol_size = 256;
    char* symbol = (char*)malloc(symbol_size);

    for (int i = 1; i < addrlen; i++) // NOTE: skipping the first symbol, it is the address of this function
    {
        out <<  symbol_list[i] << std::endl;
    }
    free(symbol);
    free(symbol_list);
}

extern "C"
{
    void terminate_err_signal_handler(int sig)
    {
        print_backtrace();
        abort();
    }
    void terminate_log_signal_handler(int sig)
    {
        std::stringstream sstrm;
        print_backtrace(sstrm);

        log4cxx::LoggerPtr log = log4cxx::Logger::getLogger("lifetechnologies.lifetech.TerminateLogger");
        LOG4CXX_ERROR(log, sstrm.str().c_str());
        abort();
    }
}


void terminate_err_exception_handler()
{
    print_backtrace();
}

void terminate_log_exception_handler()
{
    std::stringstream sstrm;
    print_backtrace(sstrm);

    log4cxx::LoggerPtr log = log4cxx::Logger::getLogger("lifetechnologies.lifetech.TerminateLogger");
    //std::cout << log->getName() << std::endl;
    LOG4CXX_ERROR(log, sstrm.str().c_str());
}

void inititialize_terminate_logging(bool use_stderr)
{
    if (use_stderr)
    {
        signal(SIGSEGV, terminate_err_signal_handler);
        std::set_terminate(terminate_err_exception_handler);
    }
    else
    {
        signal(SIGSEGV, terminate_log_signal_handler);
        std::set_terminate(terminate_log_exception_handler);
    }
}


} // namespace logging
} //namespace lifetechnologies


