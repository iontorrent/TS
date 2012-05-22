/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Region.h"

void SetUpWholeChip (Region &wholeChip,int rows, int cols)
{
  //Used later to generate mask statistics for the whole chip
  wholeChip.row = 0;
  wholeChip.col = 0;
  wholeChip.w = cols;
  wholeChip.h = rows;
}

void SetUpRegions (Region *regions, int rows, int cols, int xinc, int yinc)
{
  int i,x,y;

  for (i = 0, x = 0; x < cols; x += xinc)
  {
    for (y = 0; y < rows; y += yinc)
    {
      regions[i].index = i;
      regions[i].col = x;
      regions[i].row = y;
      regions[i].w = xinc;
      regions[i].h = yinc;
      if (regions[i].col + regions[i].w > cols)   // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
        regions[i].w = cols - regions[i].col; // but better to be safe!
      if (regions[i].row + regions[i].h > rows)
        regions[i].h = rows - regions[i].row;
      i++;
    }
  }
}
