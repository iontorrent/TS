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
    void CalculateDarkMatter(int max_fnum, float *residual, float res_threshold, int flow_block_size,
        int flow_block_start );
    void AccumulateResiduals(reg_params *reg_p, int max_fnum, float *residual, float res_threshold, int flow_block_size, int flow_block_start);
    void AccumulateOneBead(BeadParams *p, reg_params *reg_p, int max_fnum, float my_residual, float res_threshold, int flow_block_size, int flow_block_start );
    void smooth_kern(float *out, float *in, float *kern, int dist, int npts);
    int  Average0MerOneBead(int ibd,float *avg0p, int flow_block_size, int flow_block_start );
    void PCACalc(arma::fmat &avg_0mers,bool region_sampled);
    void CalculatePCADarkMatter (bool region_sampled, int flow_block_size, int flow_block_start );
};

#endif // DARKMATTER_H
