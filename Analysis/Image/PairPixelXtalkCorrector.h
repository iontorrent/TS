/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PAIRPIXELXTALKCORRECTOR_H
#define PAIRPIXELXTALKCORRECTOR_H

#include "RawImage.h"

class PairPixelXtalkCorrector
{
public:
    PairPixelXtalkCorrector();
    void Correct(RawImage *raw , float xtalk_fraction);
};

#endif // PAIRPIXELXTALKCORRECTOR_H
