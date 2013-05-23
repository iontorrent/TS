/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGION_H
#define REGION_H

#include "Serialization.h"
#include <vector>

struct Region
{
  int row, col; //upper left corner Row and Column
  int w, h;     // width & height of region
  int index;    // index of this region

  Region()                             : row(0), col(0), w(0),  h(0),  index(0) {}
  Region(int r, int c, int w_, int h_) : row(r), col(c), w(w_), h(h_), index(0) {}

private:
  friend class boost::serialization::access;
  template<typename Archive>
  void serialize (Archive& ar, const unsigned version) {
    // fprintf(stdout, "Serialize: Region ... ");
    ar & 
      row & col &
      w & h &
      index;
    // fprintf(stdout, "done\n");
  }
};

namespace RegionHelper {
  void SetUpRegions (std::vector<Region>& regions, int rows, int cols, int xinc, int yinc);
}

// Add helper struct here to simplify indirection
struct RegionTiming
{
  float t_mid_nuc;
  float t_sigma;
  float t0_frame;
private:
  friend class boost::serialization::access;
  template<typename Archive>
  void serialize (Archive& ar, const unsigned version) {
    // fprintf(stdout, "Serialize: RegionTiming ... ");
    ar & 
      t_mid_nuc &
      t_sigma &
      t0_frame;
    // fprintf(stdout, "done\n");
  }

};

#endif // REGION_H
