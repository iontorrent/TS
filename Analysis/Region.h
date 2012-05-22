/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGION_H
#define REGION_H

struct Region
{
// int x, y; // lower left corner X & Y
  int row, col; //upper left corner Row and Column
  int w, h; // width & height of region
  int index;            // index of this region
};
void SetUpWholeChip (Region &wholeChip,int rows, int cols);
void SetUpRegions (Region *regions, int rows, int cols, int xinc, int yinc);

// Add helper struct here to simplify indirection
struct RegionTiming
{
  float t_mid_nuc;
  float t_sigma;
};

#endif // REGION_H

