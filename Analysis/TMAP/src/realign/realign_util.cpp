/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_util.h"
extern "C" {
#include "../samtools/bam.h"
#include "../util/tmap_error.h"
}
//#include <exception>
#include <string>

#define CONSUME_QRY 1
#define CONSUME_REF 2

#define CONSUME_BOTH 3

// this benefits from the string being pre-allocated to a needed length
void pretty_al_from_bin_cigar (const uint32_t* cigar_bin, unsigned cigar_bin_sz, const char* qry, const char* ref, std::string& dest)
{
    unsigned oplen, constype;
    if (!dest.empty ()) dest.clear (); // TODO: issue performance warning or change to assert (empty)
    for (const uint32_t* elem = cigar_bin, *sent = cigar_bin + cigar_bin_sz; elem != sent; ++elem)
    {
        oplen = bam_cigar_oplen (*elem);
        constype = bam_cigar_type (bam_cigar_op (*elem)); // fix for PowerPC: the shortcut 'bam_cigar_type (*elem)' works only for intel (bitwise shift op).
        switch (constype)
        {
            case CONSUME_QRY:
                dest.append (oplen, '+');
                qry += oplen;
                break;
            case CONSUME_REF:
                dest.append (oplen, '-');
                ref += oplen;
                break;
            case CONSUME_BOTH:
                for (;oplen; --oplen, ++qry, ++ref)
                    dest.append (1, (*qry == *ref) ? '|' : ' ');
                break;
            default:
                tmap_bug ();
        }
    }
}

void cigar_vector_to_bin (const std::vector<CigarOp>& cigar_vector, uint32_t* cigar_dest)
{
    for (std::vector<CigarOp>::const_iterator itr = cigar_vector.begin (), cent = cigar_vector.end (); itr != cent; ++itr)
    {
        switch (itr->Type)
        {
            case 'M':
                *cigar_dest++ = (itr->Length << BAM_CIGAR_SHIFT) | BAM_CMATCH;
                break;
            case 'I':
                *cigar_dest++ = (itr->Length << BAM_CIGAR_SHIFT) | BAM_CINS;
                break;
            case 'D':
                *cigar_dest++ = (itr->Length << BAM_CIGAR_SHIFT) | BAM_CDEL;
                break;
            case 'S':
                *cigar_dest++ = (itr->Length << BAM_CIGAR_SHIFT) | BAM_CSOFT_CLIP;
                break;
            default:
                tmap_bug ();
        }
    }
}

void cigar_vector_from_bin (const uint32_t* cigar, unsigned cigar_sz, std::vector<CigarOp>& cigar_vector)
{
    cigar_vector.resize (cigar_sz);
    std::vector<CigarOp>::iterator itr = cigar_vector.begin ();
    while (cigar_sz--)
    {
        (*itr).Type = bam_cigar_opchr (*cigar);
        (*itr++).Length = bam_cigar_oplen (*cigar++);
    }
}
