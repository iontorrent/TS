/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DARKHALO_H
#define DARKHALO_H

#include "BkgMagicDefines.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

class Halo{
  public:
      // this is a regional parameter, even if not obviously so
    float   *dark_matter_compensator;  // compensate for systematic errors in background hydrogen modeling, "dark matter"
    int nuc_flow_t;  // a useful number
    float *dark_nuc_comp[NUMNUC];
    int npts;
    float weight[NUMNUC];

    Halo();
    void Alloc(int npts);
    void Delete();
    void ResetDarkMatter();
    void NormalizeDarkMatter();
    void AccumulateDarkMatter(float *residual, int inuc);
    void DumpDarkMatter(FILE *my_fp, int x, int y, float darkness);
    void DumpDarkMatterTitle(FILE *my_fp);
};

#endif // DARKHALO_H
