/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#define __tracer_cpp__

#include "tracer.h"
// #include <ctime>
#include <iostream>
#include <cstdio>
#include "fileno.hpp"

static const int TMBUF_SZ = 64;
std::ostream& operator << (std::ostream& e, Trace__ t)
{
    struct tm* timeinfo;
    char tmbuffer [TMBUF_SZ];

    timeinfo = localtime (&(t.time_));
    strftime (tmbuffer, TMBUF_SZ, "[%x %X %Z]", timeinfo);

    if (t.func_)  e << t.func_ << ": ";
    if (t.fname_) e << t.fname_ << ":" << t.lno_ << ": ";
    if (t.func_) e << "thr " << t.thread_ << ":";
    e << tmbuffer << " ";
    return e;
}

Logger::Logger (bool enabled)
:
logger_on_ (enabled),
o_ (std::cerr)
{
    check_tty ();
}

Logger::Logger (std::ostream& o, bool enabled)
:
logger_on_ (enabled),
o_ (o)
{
    check_tty ();
}

void Logger::check_tty ()
{
    int fh = fileno (o_);
    tty_ = isatty (fh);
}

Logger trclog (false);
Logger dbglog (false);
Logger info (false);
Logger warnlog (true);
Logger errlog (true);

void set_logging_level (Logger::LEVEL level)
{
    trclog.enable   (level >= Logger::TRACE);
    dbglog.enable   (level >= Logger::DEBUG);
    info.enable     (level >= Logger::INFO);
    warnlog.enable  (level >= Logger::WARNING);
    errlog.enable   (level >= Logger::ERROR);
}

