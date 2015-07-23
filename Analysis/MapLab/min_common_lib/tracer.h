/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __tracer_h__
#define __tracer_h__

#include <ostream>
#include <iomanip>
#include <pthread.h>
#include "common_str.h"
#include "clreol.h"


class Trace__
{
public:
    const char* fname_;
    int lno_;
    const char* func_;
    time_t time_;
    pthread_t thread_;

    Trace__ (const char* fname = NULL, int lno = 0, const char* funcname = NULL)
    :
    fname_ (fname),
    lno_ (lno),
    func_ (funcname),
    time_ (time (NULL)),
    thread_ (pthread_self ())
    {
    }
};


std::ostream& operator << (std::ostream& e, Trace__ t);

#define Trace Trace__(__FILE__, __LINE__, __FUNCTION__)

class Logger
{
    bool logger_on_;
    bool tty_;
    void check_tty ();
public:

    enum LEVEL
    {
        CRITICAL = 0,
        ERROR   = 10,
        WARNING = 20,
        INFO    = 30,
        DEBUG   = 40,
        TRACE   = 50,
    };

    std::ostream& o_;

    Logger (bool enabled = false);
    Logger (std::ostream& o, bool enabled = true);

    void enable  (bool op = true) { logger_on_ = op;  }
    void disable () { logger_on_ = false; }
    bool enabled () const { return logger_on_; }
    bool tty () const { return tty_; }
    void flush () { if (logger_on_) o_ << std::flush; }
};

const Logger::LEVEL loglevels [] = {Logger::CRITICAL, Logger::ERROR, Logger::WARNING, Logger::INFO, Logger::DEBUG, Logger::TRACE};


// output operator support
template <class TT>
Logger& operator << (Logger& logger, const TT& operand)
{
    if (logger.enabled ())
        logger.o_ << operand;
    return logger;
}
// ostream manipulators support
inline Logger& operator << (Logger& logger, std::ostream& (*op) (std::ostream&))
{
    if (logger.enabled ())
        logger.o_ << op;
    return logger;
}

inline Logger& operator << (Logger& logger, const clreol_class&)
{
    if (logger.enabled () && logger.tty ())
        logger.o_ << clreol_esc;
    return logger;
}

#ifndef __tracer_cpp__
extern Logger trclog;   // common tracing, off by default
extern Logger dbglog;
extern Logger info;
extern Logger warnlog;
extern Logger errlog;
#endif

#define trc (trclog << Trace)
#define dbg (dbglog << Trace << "Debug: ")
#define errl (errlog << Trace << "Error: ")
#define warn (warnlog << "Warning: ")

// convinience function setting logging level
void set_logging_level (Logger::LEVEL level);

#define IND(x) std::setw (x) << EMPTY_STR << std::setw (0)


#endif
