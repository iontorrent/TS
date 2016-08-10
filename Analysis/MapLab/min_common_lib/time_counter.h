/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __time_counter_h__
#define __time_counter_h__

#include <ctime>

typedef unsigned long long counter_t;
const time_t DEFAULT_REPORT_INTERVAL = 2; // 2 sec
const unsigned DEFAULT_CHECK_PERIOD = 10; // check every 10 iterations

class TimeCounter 
{
    time_t report_interval_;
    unsigned check_period_;
    time_t begin_time_;
    time_t prev_time_;
    time_t last_time_;
    counter_t counter_;
    counter_t ppcounter_;
    counter_t pcounter_;
    time_t elapsed_time_;
public:
    // Constructor. 
    //    report_interval - time in seconds between last and current invocation sufficient for test () to return true
    //    check_period - number of test () calls before time is checked
    TimeCounter (time_t report_interval = DEFAULT_REPORT_INTERVAL, unsigned check_period = DEFAULT_CHECK_PERIOD)
    {
        reset (report_interval, check_period);
    }
    // restarts time tracking
    void mark ()
    {
        counter_ = pcounter_ = ppcounter_ = 0;
        begin_time_ = prev_time_ = last_time_ = time (NULL);
        elapsed_time_ = 0;

    }
    // resets time tracking parameters to new values and restarts time tracking
    //    report_interval - time in seconds between last and current invocation sufficient for test () to return true
    //    check_period - number of test () calls before time is checked
    void reset (time_t report_interval = DEFAULT_REPORT_INTERVAL, unsigned check_period = DEFAULT_CHECK_PERIOD)
    {
        report_interval_ = report_interval;
        check_period_ = check_period;
        mark ();
    }
    // gets current time
    time_t cur () const
    {
        return time (NULL);
    }
    // get start time
    time_t beg () const
    {
        return begin_time_;
    }
    // get prev last (before last) marked time (floating point)
    time_t prev () const
    {
        return prev_time_;
    }
    // get last marked time 
    time_t last () const
    {
        return last_time_;
    }
    // seconds since last time mark time
    time_t since_mark () const
    {
        return time (NULL) - begin_time_;
    }
    // seconds since prevous call to 'operator ()' that returned true
    time_t sinse_last () const
    {
        return elapsed_time_;
    }
    // counter increment since time mark
    counter_t curcnt () const
    {
        return counter_;
    }
    // counter value at the time last call to 'operator ()' returned true
    counter_t prevcnt () const
    {
        return pcounter_;
    }
    // counter increment since last call to 'operator ()' returned true
    counter_t counter_incr () const
    {
        return pcounter_ - ppcounter_;
    }
    // speed in calls/sec between last and previous time 'operator ()' returned true (scaled)
    double speed (double scale = 1.0) const
    {
        return elapsed_time_ ? (counter_incr () / (elapsed_time_ * scale)) : 0;
    }
    // speed in calls/sec since start time (scaled)
    double average_speed (double scale = 1.0) const
    {
        time_t el = since_mark ();
        return el ? (counter_ / (el * scale)) : 0;
    }
    // returns cnt as percent of passed total 
    double percent (counter_t tot) const
    {
        return tot ? (counter_ * 100. / tot) : 100.;
    }
    // call operator - returns true when it is time to issue a processing status report
    bool operator () (unsigned incr_by = 1)
    {
        bool rv = false;
        if (incr_by >= check_period_ - counter_ % check_period_)
        {
            last_time_ = cur ();
            time_t elapsed = last_time_ - prev_time_;
            if (elapsed > report_interval_)
            {
                ppcounter_ = pcounter_;
                pcounter_ = counter_;
                prev_time_ = last_time_;
                elapsed_time_ = elapsed;
                rv = true;
            }
        }
        counter_ += incr_by;
        return rv;
    }
};

#endif // __time_counter_h__
