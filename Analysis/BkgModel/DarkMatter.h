/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DARKMATTER_H
#define DARKMATTER_H

#include "BkgModel.h"

class Axion
{
  public:
    BkgModel &bkg; // reference to source class for now
    Axion(BkgModel &);
    void CalculateDarkMatter(int max_fnum, bool *well_region_fit, float *residual, float res_threshold);
    void AccumulateResiduals(reg_params *reg_p, int max_fnum, bool *well_region_fit, float *residual, float res_threshold);
    void AccumulateOneBead(bead_params *p, reg_params *reg_p, int max_fnum, float my_residual, float res_threshold);
};

#endif // DARKMATTER_H