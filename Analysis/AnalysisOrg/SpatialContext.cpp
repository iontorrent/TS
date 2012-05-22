/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SpatialContext.h"


void SpatialContext::DefaultSpatialContext()
{
  numRegions = 0;
  /* enables sub chip analysis */
  chipRegion.row=0;
  chipRegion.col=0;
  chipRegion.w=0;
  chipRegion.h=0;
  cols = 0;
  rows = 0;
  regionXOrigin = 624;
  regionYOrigin = 125;
  regionXSize = 50;
  regionYSize = 50;
  cropRegions = NULL;
  numCropRegions = 0;
  // some raw image processing (like cross talk correction in the Image class) needs the absolute coordinates of the
  // pixels in the image.  This is easy for a standard data set, but for a cropped data set the origin of the data is
  // unknown.  These allow the user to specify the location of the cropped region so that these parts of analysis
  // will work as designed.
  cropped_region_x_offset = 0;
  cropped_region_y_offset = 0;

  // datasets that are divided into blocks; each block has an offset from the chip's origin:
  chip_offset_x = -1;
  chip_offset_y = -1;
  chip_len_x = 0;
  chip_len_y = 0;
  percentEmptiesToKeep = 100;
}

SpatialContext::~SpatialContext()
{
  if (cropRegions)
    free (cropRegions);
}

void SpatialContext::FindDimensionsByType (char *dat_source_directory)
{
  char *chipType = GetChipId (dat_source_directory);
  ChipIdDecoder::SetGlobalChipId (chipType); // @TODO: bad coding style, function side effect setting global variable
  int dims[2];
  GetChipDim (chipType, dims, dat_source_directory); // @what if we're doing from wells and there are no dats?
  chip_len_x = dims[0];
  chip_len_y = dims[1];
}

//@TODO: these are really methods for location context, but...
//@TODO there are some weird effects in them that I don't like


//@TODO:  bad, bad to use side effects and global variables like this
void ExportSubRegionSpecsToImage (SpatialContext &loc_context)
{
  // Default analysis mode sets values to 0 and whole-chip processing proceeds.
  // otherwise, command line override (--analysis-region) can define a subchip region.
  Image::chipSubRegion.row = (loc_context.GetChipRegion()).row;
  Image::chipSubRegion.col = (loc_context.GetChipRegion()).col;
  Image::chipSubRegion.h = (loc_context.GetChipRegion()).h;
  Image::chipSubRegion.w = (loc_context.GetChipRegion()).w;

}


//@TODO: for example, doesn't loc_context contain rows & cols?
void FixCroppedRegions (SpatialContext &loc_context, int _rows, int _cols)
{
  //If no cropped regions defined on command line, set cropRegions to whole chip
  if (!loc_context.cropRegions)
  {
    loc_context.numCropRegions = 1;
    loc_context.cropRegions = (Region *) malloc (sizeof (Region));
    SetUpWholeChip (loc_context.cropRegions[0],_rows,_cols);
  }
}

//@TODO: doesn't loc_context >contain< rows & cols?
void SetUpRegionDivisions (SpatialContext &loc_context, int _rows, int _cols)
{
  int xinc, yinc;

  loc_context.regionsX = 1;
  loc_context.regionsY = 1;

  // fixed region size
  xinc = loc_context.regionXSize;
  yinc = loc_context.regionYSize;
  loc_context.regionsX = _cols / xinc;
  loc_context.regionsY = _rows / yinc;
  // make sure we cover the edges in case rows/yinc or cols/xinc not exactly divisible
  if ( ( (double) _cols / (double) xinc) != loc_context.regionsX)
    loc_context.regionsX++;
  if ( ( (double) _rows / (double) yinc) != loc_context.regionsY)
    loc_context.regionsY++;
  loc_context.numRegions = loc_context.regionsX * loc_context.regionsY;
}


void SetUpRegionsForAnalysis (int _rows, int _cols, SpatialContext &loc_context, Region &wholeChip)
{

  FixCroppedRegions (loc_context, _rows, _cols);

  SetUpRegionDivisions (loc_context,_rows,_cols);
  SetUpWholeChip (wholeChip,_rows,_cols);

}

