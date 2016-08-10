/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_tracer.h"
#include <tracer.h>

bool TestTracer :: process ()
{
    trclog.enable ();
    trc << "This is a tracer message sent to stderr (default stream), this is number:" << std::setw (8) << 42 << ", this is longlong: " << 0x123456789ULL << ". Finita!" << std::endl;

    const char* level_names [] = {"ERROR", "WARNING", "INFO", "DEBUG", "TRACE"};
    Logger::LEVEL levels [] = {Logger::ERROR, Logger::WARNING, Logger::INFO, Logger::DEBUG, Logger::TRACE};

    for (unsigned lvlidx = 0; lvlidx < sizeof (levels) / sizeof (*levels); lvlidx ++)
    {
        o_ << "Setting logging level to " << level_names [lvlidx] << std::endl;
        set_logging_level (levels [lvlidx]);

        o_ << "Sending message to tracer stream" << std::endl;
        trc << "This message is sent to trc" << std::endl;
        o_ << "Sending message to debug stream" << std::endl;
        dbg << "This message is sent to dbg" << std::endl;
        o_ << "Sending message to info stream" << std::endl;
        info << "This message is sent to info" << std::endl;
        o_ << "Sending message to warn stream" << std::endl;
        warn << "This message is sent to warn" << std::endl;
        o_ << "Sending message to errlog stream" << std::endl;
        errl << "This message is sent to errlog" << std::endl;
    }
    return true;
}
