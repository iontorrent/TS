/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __timer_h__
#define __timer_h__

#include <ctime>
#include "platform.h"

typedef time_t time_type;  // could be clock_t

const time_type DEFAULT_REPORT_IVAL = 2; // 2 sec
const unsigned DEFAULT_CHECK_PERIOD = 10; // check every 10 iterations

inline time_type get_time ()
{
    return time (NULL); // could be clock ();
}

class Timer 
{
    time_type report_ival_;
    unsigned check_period_;
    time_type bt_;
    time_type pt_;
    time_type lt_;
    ulonglong cnt_;
    ulonglong ppcnt_;
    ulonglong pcnt_;
    time_type elapsed_;
public:
    // Constructor. 
    //    report_ival - time in seconds between last and current invocation sufficient for test to return true
    //    check_period - number of test calls before time is checked
    Timer (time_type report_ival = DEFAULT_REPORT_IVAL, unsigned check_period = DEFAULT_CHECK_PERIOD)
    {
        reset (report_ival, check_period);
    }
    // resets time tracking parameters to new values and restarts tracking
    void reset (time_type report_ival = DEFAULT_REPORT_IVAL, unsigned check_period = DEFAULT_CHECK_PERIOD)
    {
        report_ival_ = report_ival;
        check_period_ = check_period;
        mark ();
    }
    // restarts tracking with current parameters (sets start time to now and count to zero)
    void mark ()
    {
        cnt_ = pcnt_ = ppcnt_ = 0;
        bt_ = pt_ = lt_ = get_time ();
        elapsed_ = 0;

    }
    // get start time (floating point)
    time_type beg () const
    {
        return bt_;
    }
    // get prev last (before last) checkmark time (floating point)
    time_type prev () const
    {
        return pt_;
    }
    // get last checkmark time 
    time_type last () const
    {
        return lt_;
    }
    // gets current time (float)
    time_type cur () const
    {
        return get_time ();
    }
    // seconds (float) between last and prevous calls to 'operator ()' returning true
    time_type elapsed () const
    {
        return elapsed_;
    }
    // seconds (float) elapsed since start time till now
    time_type tot_elapsed () const
    {
        return get_time () - bt_;
    }
    // number of invocations since start time 
    ulonglong curcnt () const
    {
        return cnt_;
    }
    // number of invocations since start time and and prev call returning true
    ulonglong prevcnt () const
    {
        return pcnt_;
    }
    // number of invocations between prev-last and last call returning true
    ulonglong cnt_incr () const
    {
        return pcnt_ - ppcnt_;
    }
    // speed in calls/sec between last and previous calls returned true (float, scaled to a given scale)
    double speed (double scale = 1.0) const
    {
        return elapsed_ ? (cnt_incr () / (elapsed_ * scale)) : 0;
    }
    // speed in calls/sec since start time (float, scaled to a given scale)
    double ave_speed (double scale = 1.0) const
    {
        time_type el = tot_elapsed ();
        return el ? (cnt_ / (el * scale)) : 0;
    }
    // returns cnt as percent of passed total (float)
    double percent (ulonglong tot) const
    {
        return tot ? (cnt_ * 100. / tot) : 100.;
    }
    // overloaded call operator
    bool operator () (unsigned nitems = 1)
    {
        bool rv = false;
        if (nitems >= check_period_ - cnt_ % check_period_)
        {
            lt_ = cur ();
            time_type elapsed = lt_ - pt_;
            if (elapsed > report_ival_)
            {
                ppcnt_ = pcnt_;
                pcnt_ = cnt_;
                pt_ = lt_;
                elapsed_ = elapsed;
                rv = true;
            }
        }
        cnt_ += nitems;
        return rv;
    }
};

#endif // __timer_h__
