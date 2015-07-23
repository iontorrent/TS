/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __align_common_h__
#define __align_common_h__

#include "align_batch.h"

#define ALIGN_DIAG 0
#define ALIGN_DOWN 1
#define ALIGN_LEFT 2
#define ALIGN_STOP 3

#define ALIGN_HSKIP 4
#define ALIGN_VSKIP 8
#define ALIGN_ZERO  0x10

struct ALIGN_VECT
{
    int w; // weight for the best path coming by diagonal
    int h; // horizontal == weight for the best path coming from right (horizontally)
    int r; // residue (==base :)
} __attribute__ ((packed));

struct ALIGN_FVECT
{
    double w; // weight for the best path coming by diagonal
    double h; // horizontal == weight for the best path coming from right (horizontally)
    int r;    // residue (==base :)
    int div;  // length of homooligotract for current residue
} __attribute__ ((packed));

// in-place array element order reversal
template <class T>
inline void reverse_inplace (T* pb, int cnt)
{
    T tmp;
    T* pe = pb + cnt - 1;
    cnt >>= 1;
    while (cnt-- > 0)
        tmp = *pb, *pb++ = *pe, *pe-- = tmp;
}


#endif // __align_common_h__
