/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_exbitw.h"
#include <bitops.h>

struct WEIRD1
{
    BYTE ananas;
    char kokos [27];
    WORD yabloko;
    QWORD banan;
};

#pragma pack (push, 1)
struct WEIRD2
{
    BYTE ananas;
    char kokos [27];
    WORD yabloko;
    QWORD banan;
};
#pragma pack (pop)

struct THREEB
{
    char aaa [3];
};

bool TestExbitw::process ()
{

    o_ << "WEIRD1 (sizeof is " << sizeof (WEIRD1) << ") consumes " << (int) ln_bit_width <WEIRD1>::width << " bits in address" << std::endl;
    o_ << "WEIRD2 (sizeof is " << sizeof (WEIRD2) << ") consumes " << (int) ln_bit_width <WEIRD2>::width << " bits in address" << std::endl;
    o_ << "QWORD  (sizeof is " << sizeof (QWORD)  << ") consumes " << (int) ln_bit_width <QWORD>::width  << " bits in address" << std::endl;
    o_ << "DWORD  (sizeof is " << sizeof (DWORD)  << ") consumes " << (int) ln_bit_width <DWORD>::width  << " bits in address" << std::endl;
    o_ << "WORD   (sizeof is " << sizeof (WORD)   << ") consumes " << (int) ln_bit_width <WORD>::width   << " bits in address" << std::endl;
    o_ << "BYTE   (sizeof is " << sizeof (BYTE)   << ") consumes " << (int) ln_bit_width <BYTE>::width   << " bits in address" << std::endl;
    o_ << "THREEB (sizeof is " << sizeof (THREEB) << ") consumes " << (int) ln_bit_width <THREEB>::width << " bits in address" << std::endl;

    const QWORD aa = 16ULL*1024*1024*1024;
    unsigned aac = significant_bits <aa>::number;
    o_ << aa << " has " << aac << " significant bits" << std::endl;
    const QWORD bb = 16;
    unsigned bbc = significant_bits <bb>::number;
    o_ << bb << " has " << bbc << " significant bits" << std::endl;
    const QWORD cc = 0xf;
    unsigned ccc = significant_bits <cc>::number;
    o_ << cc << " has " << ccc << " significant bits" << std::endl;

    return true;
}
