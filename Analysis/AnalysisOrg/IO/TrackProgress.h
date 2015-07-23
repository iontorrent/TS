/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACKPROGRESS_H
#define TRACKPROGRESS_H

#include <stdio.h>
#include <time.h>
#include "CommandLineOpts.h"

// this handles the processParameters log for Analysis
class TrackProgress{
  public:  
  time_t analysis_current_time;
  time_t analysis_start_time;
  
  TrackProgress();
  void ReportState(const char *my_state);
  void InitFPLog (CommandLineOpts &inception_state);
  void WriteProcessParameters(CommandLineOpts &inception_state);
  
  ~TrackProgress();
  
  private:
    FILE *fpLog;
};


#endif // TRACKPROGRESS_H
