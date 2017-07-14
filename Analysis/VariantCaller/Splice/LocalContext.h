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
#include "ReferenceReader.h"
#include "MiscUtil.h"

using namespace std;

// example variant:
//  genome       42 42 44 45 46 47 48 49 50 51 52   (0 based)
//  ref is        C  C  A  A  A  A  T  G  T  A  A  A
//  alt is                       A  d  G  G
//  variant is                     |d  G  G|              
//  position0                   47 (includes anchor_right)
//  my_hp_start_pos            {47 48 49 49}
//  my_hp_length               { 1  1  2  2}
//  reference_allele             A  T  G  T
//  ref_left_hp_base C
//  left_hp_length   2
//  left_hp_start   42
//  ref_right_hp_base                        A
//  right_hp_length                          3
//  right_hp_start                          51

class LocalReferenceContext{
  public:
	// VCF stores positions in 1-based index; local_contig_sequence has a zero based index
	// all positions in this object are zero based so that they correspond to reference in memory.
    long     position0;                 //!< Zero based allele start position.
    string   contigName;                //!< Contig Name from VCF
    int      chr_idx;                   //!< index associated with contigName
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
    // MNR
    int min_mnr_rep_period;
    int max_mnr_rep_period;

  LocalReferenceContext(){
    my_hp_length.clear();
    my_hp_start_pos.clear();
    context_detected = false;
    chr_idx                    = -1;
    position0                  = -1;
	ref_left_hp_base           = 'X'; // Something that can't occur in the reference
	ref_right_hp_base          = 'X';
	left_hp_start              = 0;
    left_hp_length             = 0;
    right_hp_start             = 0;
    right_hp_length            = 0;
    min_mnr_rep_period         = 2;
    max_mnr_rep_period         = 5;
  }

  //! @brief  Fills in the member variables
  void DetectContext(const vcf::Variant &candidate_variant, int DEBUG,
      const ReferenceReader &ref_reader);
  //! @brief  Detect the context for the single base reference at position0 (zero-based) only.
  void DetectContextAtPosition(const ReferenceReader &ref_reader, int my_chr_idx, int position0, int ref_len);
  //! @brief  Basic sanity checks on the provided vcf positions
  //! Sets member context_detected to true if sanity checks are passed.
  //! Returns false if VCF position is not valid.
  bool ContextSanityChecks(const vcf::Variant &candidate_variant,
      const ReferenceReader &ref_reader);
  //! @brief  Calculate the left bound of the splicing window start in this context
  int SplicingLeftBound(const ReferenceReader &ref_reader) const;
  //! @brief  Calculate the splicing window start for MNR (if any) in this context
  int FindSplicingStartForMNR(const ReferenceReader &ref_reader, int variant_pos, int rep_period, CircluarBuffer<char>& window) const;
  //! @brief  Calculate the splicing window start if I don't want to expand the splicing window to the left (primarily for SNP, MNP)
  int StartSplicingNoExpansion() const;
  //! @brief  Calculate the splicing window start if I want to expand the splicing window to the left from the start position (primarily for INDEL)
  int StartSplicingExpandFromMyHpStart0() const;
  //! @brief  Calculate the splicing window start if I want to expand the splicing window to the left from the anchor-removed start position (primarily for some INS)
  int StartSplicingExpandFromMyHpStartLeftAnchor(int left_anchor) const;
};

#endif //LOCALCONTEXT_H
