/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_byte_order.h"
#include "bitops.h"

bool TestByteOrder :: process ()
{
    DWORD aa = 1;
    unsigned pos;
    o_ << "\nDWORD of value " << aa << ": ";
    o_ << std::hex << aa << std::dec;
    o_ << "\nCast to *(char*), 4 positions beginning from original address:\n";
    print_bytes (aa, o_);
    o_ << "\nbits of first byte from most significant:\n";
    print_bits (aa, o_);
    o_ << "\nbits of entire DWORD from most significant:\n";
    for (pos =  8 * sizeof (aa); pos != 0; pos --)
        o_ << ((aa >> (pos-1)) & 1);
    o_ << std::endl;
    return true;
}
