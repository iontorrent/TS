/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACECORRECTOR_H
#define TRACECORRECTOR_H

#include "SignalProcessingMasterFitter.h"


class TraceCorrector
{
  public:
    SignalProcessingMasterFitter &bkg; // reference to source class for now

      TraceCorrector (SignalProcessingMasterFitter &);
    ~TraceCorrector();

    void ReturnBackgroundCorrectedSignal(float *block_signal_corrected, float *block_signal_original, float *block_signal_sbg, int ibd, int flow_block_size,
        int flow_block_start );
};


#endif // TRACECORRECTOR_H
