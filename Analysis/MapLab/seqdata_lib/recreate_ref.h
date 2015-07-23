/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __recreate_ref_h__
#define __recreate_ref_h__

#include <rerror.h>
// #include <Cigar.h>

// facility for re-creating reference sequence from BAM record using read sequence, CIGAR string and MD tag

class RecreateRefError : public Rerror {};

class Cigar;
size_t recreate_ref (const char* query, size_t query_len, const Cigar* cigar, const char* mdtag, char* dest, size_t destlen, bool include_softclip = false);
size_t recreate_ref (const char* query, size_t query_len, const char* cigar, const char* mdtag, char* dest, size_t destlen, bool include_softclip = false);

#endif // __recreate_ref_h__

