/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __ALPHABET_H__
#define __ALPHABET_H__

#include "alphabet_templ.h"

namespace genstr
{

// global aminoacid and nucleotide alphabets

extern const unsigned NUMBER_OF_BASES;
extern const unsigned BITS_PER_BASE;
extern const unsigned NUMBER_OF_RESIDUES;
extern const unsigned BITS_PER_RESIDUE;
extern Alphabet <char> nucleotides;
extern Alphabet <char> aminoacids;


}
#endif // __ALPHABET_H__

