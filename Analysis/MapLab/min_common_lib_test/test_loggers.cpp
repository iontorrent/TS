/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_loggers.h"
bool TestLoggers::process ()
{
    o_ << "Enable info" << std::endl;
    set_logging_level (Logger::INFO);
    errl << "Indirect error logger test" << std::endl;
    errlog << "Direct error logger test" << std::endl;
    warn << "Indirect warning logger test" << std::endl;
    warnlog << "Direct warning logger test" << std::endl;
    info << "Info logger test" << std::endl;
    dbglog << "If this is seen, level selector misfunctions" << std::endl;

    o_ << "Enable trace" << std::endl;
    set_logging_level (Logger::TRACE);
    dbg << "This is message to dbg" << std::endl;
    dbglog << "This is message to dbglog" << std::endl;
    trc << "This is trace message" << std::endl;
    trclog << "This is message to trclog" << std::endl;

    o_ << "Testing tty-aware code: clreol should print only on TTYs" << std::endl;

    for (unsigned i = 100000; i != 1; i /= 10)
        info << "\rIter " << i << clreol;
    info << std::endl;
    return true;
}
