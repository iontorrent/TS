/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     LocalContext.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef LOCALCONTEXT_H
#define LOCALCONTEXT_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>

#include <Variant.h>

using namespace std;

class LocalReferenceContext{
  public:
	// VCF stores positions in 1-based index; local_contig_sequence has a zero based index
	// all positions in this object are zero based so that they correspond to reference in memory.
    long     position0;                 //!< Zero based allele start position.
    string   contigName;                //!< Contig Name from VCF
    bool     context_detected;          //!< Check to see if the reference allele matches the given genome position.

    // Information about the bases in the reference allele
    string       reference_allele;      //!< The reference allele for this variant
    vector<int>  my_hp_length;          //!< Member HP length for each base in the reference allele.
    vector<int>  my_hp_start_pos;       //!< Start position of the HP, recorded for each base in the ref. allele

    // Information about the HP to the left of variant start position
    char     ref_left_hp_base;          //!< Base comprising the HP to the left of the one containing the vcf position
    int      left_hp_length;            //!< Length of the HP to the left of of the HP encompassing position
    int      left_hp_start;
    // Information about the HP to the right of variant start position
    char     ref_right_hp_base;         //!< Base comprising the HP to the right of the one containing the vcf position
    int      right_hp_length;           //!< Length of the HP to the left of of the HP encompassing position
    int      right_hp_start;

  LocalReferenceContext(){
    my_hp_length.clear();
    my_hp_start_pos.clear();
    context_detected = false;
    position0                  = -1;
	ref_left_hp_base           = 'X'; // Something that can't occur in the reference
	ref_right_hp_base          = 'X';
	left_hp_start              = 0;
    left_hp_length             = 0;
    right_hp_start             = 0;
    right_hp_length            = 0;
  }

  //! @brief  Fills in the member variables
  void DetectContext(const string & local_contig_sequence, vcf::Variant ** candidate_variant, int DEBUG);

  //! @brief  Basic sanity checks on the provided vcf positions
  //! Sets member context_detected to true if sanity checks are passed.
  //! Returns false if VCF position is not valid.
  bool ContextSanityChecks(const string & local_contig_sequence, vcf::Variant ** candidate_variant);
};

#endif //LOCALCONTEXT_H
