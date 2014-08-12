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
        // abuse the trace buffers
    void BackgroundCorrectAllBeadsInPlace (int flow_block_size, int flow_block_start);
    void BackgroundCorrectBeadInPlace (int ibd, int flow_block_size, int flow_block_start);
    void ReturnBackgroundCorrectedSignal(float *block_signal_corrected, int ibd, int flow_block_size,
        int flow_block_start );
};


#endif // TRACECORRECTOR_H
