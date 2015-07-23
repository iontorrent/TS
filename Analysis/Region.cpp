/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Region.h"

/* Why is this col major when everything else is row major? */
void RegionHelper::SetUpRegions (std::vector<Region>& regions, int rows, int cols, int xinc, int yinc)
{
  int i,x,y;


  for (i = 0, x = 0; x < cols; x += xinc)
  {
    for (y = 0; y < rows; y += yinc)
    {
  //for (i = 0,y = 0; y < rows; y += yinc)
  //{
    //for ( x = 0; x < cols; x += xinc)
   // {
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
