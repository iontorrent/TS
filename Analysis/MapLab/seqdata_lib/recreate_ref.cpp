/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "recreate_ref.h"

#include <tracer.h>
#include <cstddef>
#include "MDtag.h"
#include <CigarRoller.h>

// facility for re-creating reference sequence from BAM record using read sequence, CIGAR string and MD tag

size_t recreate_ref (const char* query, size_t query_len, const Cigar* cigar, const char* mdtag, char* dest, size_t destlen, bool include_softclip)
{
    // walk over the cigar
    // advance over the mdtag to same position
    MD md (mdtag);
    MDIterator mditer (md);
    size_t qpos = 0; // query positions
    size_t cigar_size = cigar->size (), cigar_idx = 0; // cigar size and current record index
    size_t cigar_pos = 0; // query position at the current CIGAR index
    size_t insert_sz;
    while (cigar_idx != cigar_size)
    {
        const Cigar::CigarOperator& cur = (*cigar)[cigar_idx ++];
        switch (cur.operation)
        {
            case Cigar::match:
            case Cigar::mismatch:
                cigar_pos += cur.count;
                // read MD while md_pos is below cigar_pos
                while (!mditer.done () && qpos != cigar_pos)
                {
                    if (mditer.pos () == destlen)
                        ers << "Destination buffer for reference sequence is too short." << ThrowEx (RecreateRefError) ;
                    if (qpos == query_len)
                        ers << "Query too short for CIGAR string" << ThrowEx (RecreateRefError);
                    size_t dpos = mditer.pos ();
                    dest [dpos] = (*mditer).chr (query [qpos]);
                    ++qpos;
                    ++mditer;
                }
                if (mditer.done () && qpos != cigar_pos)
                    ers << "MD tag too short for for CIGAR string" << ThrowEx (RecreateRefError);
                break;
            case Cigar::insert:
                // insertion with respect to reference sequence (query has insert, reference does not) - no ref update
                if (qpos + cur.count > query_len)
                    ers << "CIGAR denotes query longer then passed " << ThrowEx (RecreateRefError);
                cigar_pos += cur.count;
                qpos += cur.count;
                break;
            case Cigar::del:
            case Cigar::skip:
            case Cigar::pad:
                // deletion from reference sequence: assuming the MD has corresponding ^ sequence. This is not enforced, though - 
                // otherwise we should rely on the presence of '0' at the end of deleted sequence, which is not required by MD spec.
                // Instead, validation is done based on total sequence positions match
                // validate that here we had the '^' in MD
                insert_sz = cur.count;
                while (!mditer.done () && insert_sz)
                {
                    if ((*mditer).type_ == MD::match)
                        ers << "Match in MD tag over CIGAR's insert at position " << mditer.pos () <<  ThrowEx (RecreateRefError);
                    if ( !(*mditer).pasted ())
                        ers << "MD insert shorter then CIGAR's insert at position " << mditer.pos () << ThrowEx (RecreateRefError);
                    dest [mditer.pos ()] = (*mditer).chr ();
                    ++ mditer;
                    -- insert_sz;
                }
                if (mditer.done () && insert_sz)
                    ers << "MD tag too short for for CIGAR string" << ThrowEx (RecreateRefError);
                break;
            case Cigar::softClip: 
                // the soft clip seem to be in MD according to SAM format spec, 
                // but it is not included in MD by the Realigner code. The latter seem to be more trusted within Torrent suite. 
                // making this behavior parametrized.
                if (include_softclip)
                {
                    cigar_pos += cur.count;
                    // read MD while md_pos is below cigar_pos
                    // for the MD tags that do not have Zeros as deletion end markers, 
                    // this code may misbehave. 
                    while (!mditer.done () && qpos != cigar_pos)
                    {
                        if (mditer.pos () == destlen)
                            ers << "Destination buffer for reference sequence is too short" << ThrowEx (RecreateRefError);
                        if (qpos == query_len)
                            ers << "Query too short for CIGAR string" << ThrowEx (RecreateRefError);
                        dest [mditer.pos ()] = (*mditer).chr (query [qpos]);
                        if (!(*mditer).pasted ())
                            ++ qpos;
                        ++ mditer;
                    }
                    if (mditer.done () && qpos != cigar_pos)
                        ers << "MD tag too short for for CIGAR string" << ThrowEx (RecreateRefError);
                }
                else
                {
                    qpos += cur.count;
                    cigar_pos += cur.count;
                }
                break;
            case Cigar::hardClip: // the hard clip zone is not included in query or reference
                break;
            default:
                break;
        }
    }
    return mditer.pos ();
}

size_t recreate_ref (const char* query, size_t query_len, const char* cigar_str, const char* mdtag, char* dest, size_t destlen, bool include_softclip)
{
    CigarRoller cigar (cigar_str);
    return recreate_ref (query, query_len, &cigar, mdtag, dest, destlen, include_softclip);
}
