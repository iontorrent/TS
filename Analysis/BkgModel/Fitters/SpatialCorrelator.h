/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SPATIALCORRELATOR_H
#define SPATIALCORRELATOR_H

#include "SignalProcessingMasterFitter.h"


class SpatialCorrelator
{
  public:
    SignalProcessingMasterFitter &bkg; // reference to source class for now

    // hacky cross-talk info
    float *nn_odd_col_map;
    float *nn_even_col_map;
    float avg_corr;

    Region *region; // extract from bkg model for convenience

    SpatialCorrelator (SignalProcessingMasterFitter &);
    ~SpatialCorrelator();
    void Defaults();
    float MakeSignalMap(float *ampl_map, int fnum);
    float UnweaveMap(float *ampl_map, int row, int col, float default_signal);
    void MeasureConvolution(int *prev_same_nuc_tbl,int *next_same_nuc_tbl);
    void NNAmplCorrect(int fnum);
    void AmplitudeCorrectAllFlows();
};

#endif // SPATIALCORRELATOR_H
