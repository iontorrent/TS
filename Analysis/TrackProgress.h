/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACKPROGRESS_H
#define TRACKPROGRESS_H

#include <stdio.h>
#include <time.h>

// this handles the processParameters log for Analysis
class TrackProgress{
  public:
  FILE *fpLog;
  time_t analysis_current_time;
  time_t analysis_start_time;
  
  TrackProgress();
  void ReportState(char *my_state);
  ~TrackProgress();
};


#endif // TRACKPROGRESS_H