/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define __GENCODE_CPP__

#include "alphabet.h"
#include "translator.h"

#include "gencode.h"

namespace genstr
{
GeneticCode::GeneticCode (const char* gcode)
{
    if (!gcode) std::fill (gctable_, gctable_ + GCSIZE, (char) 0);
    else configure (gcode);

}
char GeneticCode::translate (const char* seq, unsigned offset)
{
    unsigned codone_number = 0;
    unsigned base_idx;
    seq += offset;
    for (base_idx = 0; base_idx < CODONE_SIZE; base_idx ++, seq ++)
    {
        codone_number <<= BITS_PER_BASE;
        codone_number |= *seq;
    }
    return gctable_ [codone_number];
}
unsigned GeneticCode::xlat (const char* src, unsigned len, char* dest)
{
    if (len < CODONE_SIZE)
        return 0;
    if (dest != NULL)
    {
        unsigned codone_number = 0;
        unsigned base_idx;
        for (base_idx = 0; base_idx < len; base_idx ++, src ++)
        {
            if (base_idx && !(base_idx % CODONE_SIZE))
            {
                *dest++ = gctable_ [codone_number];
                codone_number = 0;
            }
            codone_number <<= BITS_PER_BASE;
            codone_number |= *src;
        }
    }
    return len / CODONE_SIZE;
}
unsigned GeneticCode::unwind (const char* src, unsigned len, char* dest)
{
    if (len < CODONE_SIZE)
        return 0;
    if (dest != NULL)
    {
        unsigned base_idx, phase_idx;
        unsigned phase = 0;
        unsigned codone_numbers [CODONE_SIZE];
        for (base_idx = 0; base_idx < len; base_idx ++, src ++)
        {
            if (base_idx >= CODONE_SIZE)
                *dest++ = gctable_ [codone_numbers [phase]];
            codone_numbers [phase] = 0;
            for (phase_idx = 0; phase_idx < CODONE_SIZE; phase_idx ++)
            {
                codone_numbers [phase_idx] <<= BITS_PER_BASE;
                codone_numbers [phase_idx] |= *src;
            }
            if (++phase == CODONE_SIZE)
                phase = 0;
        }
    }
    return len + 1 - CODONE_SIZE;
}
void GeneticCode::configure (const char* gcode)
{
    aa2num.xlat (gcode, GCSIZE, gctable_);
}


}
