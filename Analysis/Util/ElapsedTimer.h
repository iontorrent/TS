/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ELAPSEDTIMER_H
#define ELAPSEDTIMER_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

namespace ION
{
/**
 * Calculates the elapsed time.
 */
class ElapsedTimer
{
public:
    
    ElapsedTimer() : _dDuration(0.0) {}
    ~ElapsedTimer() {}

    void Start()
    {
        gettimeofday( &_startTime, 0 );
    }

    void Stop()
    {
        timeval endTime;
        gettimeofday( &endTime, 0 );

        const long seconds = endTime.tv_sec - _startTime.tv_sec;
        const long useconds = endTime.tv_usec - _startTime.tv_usec;
        _dDuration = seconds + useconds / 1000000.0;
    }

    const std::string GetActualElapsedSeconds()
    {
        char strBuf[128];
        sprintf( strBuf, "%5.6f", _dDuration );
        return std::string( strBuf );
    }

private:
    
    timeval _startTime;
    double _dDuration;
    
};
// END ElapsedTimer

}
// END namespace ION

#endif // ELAPSEDTIMER_H
