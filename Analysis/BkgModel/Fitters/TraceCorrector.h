/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACECORRECTOR_H
#define TRACECORRECTOR_H

#include "BkgModel.h"


class TraceCorrector
{
  public:
    BkgModel &bkg; // reference to source class for now

      TraceCorrector (BkgModel &);
    ~TraceCorrector();
        // abuse the trace buffers
    void BackgroundCorrectAllBeadsInPlace (void);
    void BackgroundCorrectBeadInPlace (int ibd);
    void ReturnBackgroundCorrectedSignal(float *block_signal_corrected, int ibd);
};


#endif // TRACECORRECTOR_H