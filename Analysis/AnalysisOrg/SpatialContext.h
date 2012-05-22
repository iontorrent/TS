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

// cropping, displacing, locating items on the chip
// some of this is probably part of bkgmodel controls (analysis regions)
// some of this is also probably part of Image tracking
class SpatialContext{
  public:
    int numRegions;
      int cols;
    int rows;
    int regionXOrigin;
    int regionYOrigin;
    int regionXSize;
    int regionYSize;
    int regionsX;
    int regionsY;
    Region *cropRegions;
    int numCropRegions;
     int cropped_region_x_offset;
    int cropped_region_y_offset;
    int chip_offset_x;
    int chip_offset_y;
    int chip_len_x;
    int chip_len_y;
    struct Region chipRegion;
    double percentEmptiesToKeep;
     // do we have regions that are known to be excluded
   bool exclusionMaskSet;
    struct Region GetChipRegion() {
        return chipRegion;
    }
   void FindDimensionsByType(char *dat_source_directory);
     void DefaultSpatialContext();
     ~SpatialContext();
};

//@TODO: actually methods, but why are they passed _rows and _cols?

void SetUpRegionDivisions(SpatialContext &loc_context, int _rows, int _cols);


void FixCroppedRegions(SpatialContext &loc_context, int _rows, int _cols);

void ExportSubRegionSpecsToImage(SpatialContext &loc_context);
void SetUpRegionsForAnalysis (int _rows, int _cols, SpatialContext &loc_context, Region &wholeChip);

#endif // SPATIALCONTEXT_H