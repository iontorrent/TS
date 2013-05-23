/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DARKMATTER_H
#define DARKMATTER_H
#include <armadillo>

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
    void smooth_kern(float *out, float *in, float *kern, int dist, int npts);
    int  Average0MerOneBead(int ibd,float *avg0p);
    void PCACalc(arma::fmat &avg_0mers,bool region_sampled);
    void CalculatePCADarkMatter (bool region_sampled);
};

#endif // DARKMATTER_H
