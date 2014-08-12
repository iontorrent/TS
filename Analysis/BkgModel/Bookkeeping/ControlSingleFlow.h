/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CONTROLSINGLEFLOW_H
#define CONTROLSINGLEFLOW_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "Serialization.h"


struct ControlSingleFlow{
  float krate_adj_limit;
  float dampen_kmult;
  float kmult_low_limit;
  float kmult_hi_limit;


  ControlSingleFlow();
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
      & dampen_kmult
      & krate_adj_limit
      & kmult_low_limit
      & kmult_hi_limit;
  }
};


#endif //CONTROLSINGLEFLOW_H
