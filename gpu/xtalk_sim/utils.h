/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>
#include "xtalk_sim.h"

typedef struct {
	DATA_TYPE x;
	DATA_TYPE y;
	DATA_TYPE z;
} Coordinate;

typedef struct {
	Coordinate vert[7];
} HexagonDescriptor;

Coordinate GetWellLocation(int row,int col,DATA_TYPE pitch);
HexagonDescriptor GetHexagon(Coordinate loc,DATA_TYPE pitch,DATA_TYPE well_fraction);
DATA_TYPE FindNextHexagonCrossing(HexagonDescriptor descr,DATA_TYPE xloc,DATA_TYPE prevy,DATA_TYPE maxy);
int MarkInHexagon(HexagonDescriptor descr,bool *mask,int ni,int nj,DATA_TYPE dx,DATA_TYPE dy,DATA_TYPE margin);


//
// Utility timer class
//
class Timer
{
  public:
    Timer()
    {
      restart();
    }
    void restart()
    {
      gettimeofday (&start_time, NULL);
    }
    double elapsed()
    {
      gettimeofday (&end_time, NULL);
      return (end_time.tv_sec - start_time.tv_sec + static_cast<double> (end_time.tv_usec - start_time.tv_usec) / (1000000.0));
    }
  private:
    timeval start_time;
    timeval end_time;
};

#endif // UTILS_H

