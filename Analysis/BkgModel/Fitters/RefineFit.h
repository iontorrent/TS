/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REFINEFIT_H
#define REFINEFIT_H

#include "SignalProcessingMasterFitter.h"
#include "SingleFlowFit.h"

// make this code look >just< like the GPU option
class RefineFit
{
  public:
    SignalProcessingMasterFitter &bkg; // reference to source class for now

    single_flow_optimizer my_single_fit;
    EmphasisClass *local_emphasis; // computation is currently cheap, save per flow

    RefineFit (SignalProcessingMasterFitter &);
    void InitSingleFlowFit();
    void    FitAmplitudePerFlow ();
    void    FitAmplitudePerBeadPerFlow (int ibd, NucStep &cache_step);
    void SupplyMultiFlowSignal(float *block_signal_corrected, int ibd);
    void SpecializedEmphasisFunctions();
    void SetupLocalEmphasis();
    void TestXtalk();
    ~RefineFit();
};

#endif // REFINEFIT_H
