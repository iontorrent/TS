/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PAIRPIXELXTALKCORRECTOR_H
#define PAIRPIXELXTALKCORRECTOR_H

#include "RawImage.h"
#include "string"
#include "fstream"
#include <iostream>

class PairPixelXtalkCorrector
{
public:
    PairPixelXtalkCorrector();
    void Correct(RawImage *raw , float xtalk_fraction);
    void CorrectThumbnailFromFile(RawImage *raw , const char * xtalkFileName);
};

#endif // PAIRPIXELXTALKCORRECTOR_H
