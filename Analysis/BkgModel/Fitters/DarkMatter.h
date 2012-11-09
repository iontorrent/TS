/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DARKMATTER_H
#define DARKMATTER_H

#include "SignalProcessingMasterFitter.h"

using namespace std;

class Axion
{
  public:
    SignalProcessingMasterFitter &bkg; // reference to source class for now
    Axion(SignalProcessingMasterFitter &);
    void CalculateDarkMatter(int max_fnum, float *residual, float res_threshold);
    void AccumulateResiduals(reg_params *reg_p, int max_fnum, float *residual, float res_threshold);
    void AccumulateOneBead(bead_params *p, reg_params *reg_p, int max_fnum, float my_residual, float res_threshold);
};

#endif // DARKMATTER_H
