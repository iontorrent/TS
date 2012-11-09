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
    void BackgroundCorrectAllBeadsInPlace (void);
    void BackgroundCorrectBeadInPlace (int ibd);
    void ReturnBackgroundCorrectedSignal(float *block_signal_corrected, int ibd);
};


#endif // TRACECORRECTOR_H