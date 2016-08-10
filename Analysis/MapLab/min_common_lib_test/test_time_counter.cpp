/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_time_counter.h"
#include <time_counter.h>

bool TestTimeCounter::process ()
{
    TimeCounter timer (2, 1000);
    const unsigned MAXTEST = 1000*1000*1000; //100M
    for (unsigned kk = 0; kk < MAXTEST; kk ++)
    {
        if (timer ())
            o_ << "\r" << std::dec << timer.curcnt () << " iters, " << timer.since_mark () << " sec elapsed, " << timer.average_speed () << " iter/sec, " << timer.percent (MAXTEST) << "% task" << std::flush;
    }
    return true;
}
