/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SPATIALCONTEXT_H
#define SPATIALCONTEXT_H

#include <vector>
#include <string>
#include <map>
#include <set>
#include "Region.h"
#include "IonVersion.h"
#include "Utils.h"
#include "OptBase.h"

// cropping, displacing, locating items on the chip
// some of this is probably part of bkgmodel controls (analysis regions)
// some of this is also probably part of Image tracking
class SpatialContext{
  public:
    int numRegions;
    int cols;
    int rows;
    int regionXSize;
    int regionYSize;
    int regionsX;
    int regionsY;
    Region *cropRegions;
    int numCropRegions;
    bool isCropped;
     int cropped_region_x_offset;
    int cropped_region_y_offset;
    int chip_offset_x;
    int chip_offset_y;
    int chip_len_x;
    int chip_len_y;
    string chipType;
    
    struct Region chipRegion;
    
    double percentEmptiesToKeep;
     // do we have regions that are known to be excluded
   bool exclusionMaskSet;
   SpatialContext() { DefaultSpatialContext(); }

    struct Region GetChipRegion() {
        return chipRegion;
    }
    void FindDimensionsByType(char *explog_path);
    void DefaultSpatialContext();

    inline void SetRegionXYSize(int xsize, int ysize) {
      regionXSize = xsize;
      regionYSize = ysize;
    }
    inline bool IsSetRegionXYSize(){  return ( regionXSize != 0 );}
	void PrintHelp();
	void SetOpts(OptArgs &opts, Json::Value& json_params);

    ~SpatialContext();
};


//@TODO: actually methods, but why are they passed _rows and _cols?

void SetUpRegionDivisions(SpatialContext &loc_context, int _rows, int _cols);


void FixCroppedRegions(SpatialContext &loc_context, int _rows, int _cols);


void SetUpRegionsForAnalysis (int _rows, int _cols, SpatialContext &loc_context);

#endif // SPATIALCONTEXT_H
