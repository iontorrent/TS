/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __GENCODE_H__
#define __GENCODE_H__

#include "translator_templ.h"

namespace genstr
{

#define CODONE_SIZE 3
#define GCSIZE 64

// GeneticCode is a nucleotide->protein sequence translator

class GeneticCode : public Translator <char, char>
{
    char gctable_ [GCSIZE];
public:
    GeneticCode (const char* gcode = NULL);
    char translate (const char* seq, unsigned off);
    unsigned xlat (const char* src, unsigned len, char* dest);
    unsigned unwind (const char* src, unsigned len, char* dest);
    unsigned unit () const
    {
        return CODONE_SIZE;
    }
    void configure (const char* gcode);
    const char* code () const
    {
        return gctable_;
    }
};

// the actual genetic codes are so well stabilized they can be hard-coded

extern GeneticCode standardGeneticCode;

}

#endif
