/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SPATIALCORRELATOR_H
#define SPATIALCORRELATOR_H

#include "RegionalizedData.h"
#include "SlicedChipExtras.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "WellXtalk.h"


class HplusMap{
public:
  float *ampl_map;
  int NucId;
  float region_mean_sig;
  float bead_mean_sig;

  void Allocate(Region *region);
  void DeAllocate();
  HplusMap();
};

class SpatialCorrelator
{
  public:

    RegionalizedData *region_data;
    SlicedChipExtras *region_data_extras;
    Region *region; // extract for convenience

    WellXtalk my_xtalk;
    HplusMap my_hplus;

    SpatialCorrelator();

    void Defaults();
    void SetRegionData(RegionalizedData *_region_data, SlicedChipExtras *_region_data_extras);
    void MakeSignalMap(HplusMap &signal_map, int fnum);
    void NNAmplCorrect(int fnum, int flow_block_start);
    void SimpleXtalk(int fnum, int flow_block_start);
    void AmplitudeCorrectAllFlows( int flow_block_size, int flow_block_start );
    void BkgDistortion(int fnum, int flow_block_start);

};

#endif // SPATIALCORRELATOR_H
