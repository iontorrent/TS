/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REALIGN_UTIL_H
#define REALIGN_UTIL_H

#include <stdint.h>
#include <string>
#include <vector>
#include "cigar_op.h"

// fills the dest_al with 'pretty' alignment. Benefits from pre-allocates dest_al space
void pretty_al_from_bin_cigar (const uint32_t* cigar_bin, unsigned cigar_bin_sz, const char* qry, const char* ref, std::string& dest_al);

// converts 'bamtools' CigarOp vector to TMAP internal cigar repr
// the cigar_dest has to be pre-allocated and contain at least cigar_vector.size () elements
void cigar_vector_to_bin (const std::vector<CigarOp>& cigar_vector, uint32_t* cigar_dest);

// converts TMAP internal cigar repr to 'bamtools' CigarOp vector
void cigar_vector_from_bin (const uint32_t* cigar, unsigned cigar_sz, std::vector<CigarOp>& cigar_op_vec);

#endif // REALIGN_UTIL_H
