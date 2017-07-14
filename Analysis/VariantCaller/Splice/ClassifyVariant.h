/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ClassifyVariant.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef CLASSIFYVARIANT_H
#define CLASSIFYVARIANT_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>


#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <Variant.h>


#include "MiscUtil.h"
#include "LocalContext.h"
#include "InputStructures.h"
#include "ExtendParameters.h"
//@TODO: remove when moving SSE detector
#include "VariantAssist.h"

using namespace std;


class VarButton {
  public:
  // Describes the basic identity and sub-classification of our allele
  bool isSNP;              // Single base substitution
  bool isMNV;              // Multiple base substitution
  bool isPaddedSNP;        // MNV that is in fact an anchored or padded SNP.

  bool isIndel;            // Anchor base + one or more copies of the same base in the longer allele
  bool isInsertion;        // Alternative allele longer than reference allele
  bool isDeletion;         // Alternative allele shorter than reference allele
  bool isHPIndel;          // InDel occurs in a reference HP of length > 1
  bool isDyslexic;

  //bool isComplex;          // A complex allele is anything but snp, mnv and Indel
  //bool isComplexHP;        // This complex allele involves a ref. HP of length > 1
  bool isHotSpotAllele;     // Signifies the hotspot allele
  bool isFakeHsAllele;      // Signifies a "fake" hotspot variant, i.e., the candidate generator sees no read supports me.
  bool isHotSpot;           // Signifies a hotspot variant (set per variant for all alleles, regardless of their specific origin)
  bool isProblematicAllele; // There is something wrong with this allele, we should filter it.
  bool isNoVariant;         // The alternative allele is not a variant, i.e., altAllele = reference_context.reference_allele
  bool doRealignment;       // Switch to turn realignment on or off 

  VarButton() {
    isHPIndel      = false;
    isSNP          = false;
    isInsertion    = false;
    isDeletion     = false;
    isDyslexic     = false;
    isMNV          = false;
    isPaddedSNP    = false;
    isIndel        = false;
    isHotSpot      = false;
    isHotSpotAllele= false;
    isFakeHsAllele = false;
    isProblematicAllele = false;
    isNoVariant    = false;
    doRealignment  = false;
  }
};

class AlleleIdentity {
  public:
    VarButton     status;     //!< A bunch of flags saying what's going on with this allele
    string        altAllele;  //!< May contain left and/or right anchor bases, cannot be empty
    int           DEBUG;
    int           chr_idx;    //!< Chromosome index>
    int           position0;  //!<The 0-based reference start position>
    int           ref_length; //!<The length of the reference allele>
    // useful context
    int left_anchor;        //!< Number of left bases that are common between the ref. and alt. allele
    int right_anchor;         //!< Number of right bases that are common between the ref. and alt. allele
                              //   left_anchor + right_anchor <= shorter allele length
    int inDelLength;          //!< Difference in length between longer and shorter allele
    int ref_hp_length;        //!< First base change is occurring in an HP of length ref_hp_length
    int start_splicing_window;         //!< Start of splicing window of interest for this alt allele.
    int end_splicing_window;           //!< End of splicing window of interest for this alt allele => splicing window = [start_splicing_window, end_splicing_window)
    int multiallele_window_start;         //!< Start of splicing window of interest for the multi-allele variant
    int multiallele_window_end;           //!< End of splicing window of interest for multi-allele variant
    int start_variant_window;         //!< Start of the anchor-removed window of interest for this alt allele
    int end_variant_window;           //!< End of the anchor-removed window of interest for this alt allele => variant window = [start_variant_window, end_variant_window)

    // need to know when I do filtering
    float  sse_prob_positive_strand;
    float  sse_prob_negative_strand;
    vector<string> filterReasons;

    bool indelActAsHPIndel;   // Switch to make all indels match HPIndel behavior

    // Level of flow-disruption vs ref: (-1, 0, 1, 2) = (indefinite, HP-INDEL, otherwise , FD)
    int fd_level_vs_ref;

    // Extra (prefix, suffix) padding bases added into the alt allele if the alt allele representation is obtained by grouping with other alt alleles.
    // (Note): For an standard indel representation e.g., T->TT, num_padding_added = (0, 0).
    pair<int, int> num_padding_added;

    AlleleIdentity() {
      position0 = -1;
      chr_idx = -1;
      ref_length = -1;
      inDelLength = 0;
      ref_hp_length = 0;
      //modified_start_pos = 0;
      left_anchor = 0;
      right_anchor = 0;
      start_splicing_window = 0;
      end_splicing_window = 0;
      multiallele_window_start = 0;
      multiallele_window_end = 0;
      start_variant_window = 0;
      end_variant_window = 0;
      DEBUG = 0;
      
      // filterable statuses
      sse_prob_positive_strand = 0;
      sse_prob_negative_strand = 0;

      indelActAsHPIndel = false;
      fd_level_vs_ref = -1;
    };

    bool Ordinary() const {
      return (status.isIndel and (not status.isHPIndel));
    };
    
    bool ActAsSNP() const {
      if (indelActAsHPIndel)
    	  return (status.isSNP or status.isPaddedSNP);
      else
    	  return (status.isSNP or status.isPaddedSNP or (status.isIndel and (not status.isHPIndel)));
    }
    bool ActAsMNP() const {
      return status.isMNV;
    }
    bool ActAsHPIndel() const {
      if (indelActAsHPIndel)
    	  return status.isIndel;
      else
    	  return (status.isIndel and status.isHPIndel);
    }
    //void DetectPotentialCorrelation(const LocalReferenceContext &reference_context);
    bool SubCategorizeInDel(const LocalReferenceContext &reference_context,
                            const ReferenceReader &ref_reader);
    void IdentifyHPdeletion(const LocalReferenceContext& reference_context);
    void IdentifyHPinsertion(const LocalReferenceContext& reference_context,
        const ReferenceReader &ref_reader);
    bool IdentifyDyslexicMotive(char base, int position,
        const ReferenceReader &ref_reader, int chr_idx);

    void SubCategorizeSNP(const LocalReferenceContext &reference_contextl);
    void SubCategorizeMNP(const LocalReferenceContext &reference_contextl);
    bool getVariantType(const string _altAllele, const LocalReferenceContext &reference_context,
                        const TIonMotifSet & ErrorMotifs,
                        const ClassifyFilters &filter_variant,
                        const ReferenceReader &ref_reader,
						const pair<int, int> &alt_orig_padding);
    bool CharacterizeVariantStatus(const LocalReferenceContext &reference_context,
                                   const ReferenceReader &ref_reader);
    bool CheckValidAltAllele(const LocalReferenceContext &reference_context);
    //void ModifyStartPosForAllele(int variantPos);

    bool IdentifyMultiNucRepeatSection(const LocalReferenceContext &seq_context, unsigned int rep_period,
        const ReferenceReader &ref_reader);

    void CalculateWindowForVariant(const LocalReferenceContext &seq_context,
        const ReferenceReader &ref_reader);

    void DetectCasesToForceNoCall(const LocalReferenceContext &seq_context, const ControlCallAndFilters& my_controls,
        const VariantSpecificParams& variant_specific_params);
    void DetectLongHPThresholdCases(const LocalReferenceContext &seq_context, int maxHPLength);
    void DetectHpIndelCases(const vector<int> &hp_indel_hrun, const vector<int> &hp_ins_len, const vector<int> &hp_del_len);
    void DetectNotAVariant(const LocalReferenceContext &seq_context);
    void PredictSequenceMotifSSE(const LocalReferenceContext &reference_context, const TIonMotifSet & ErrorMotifs,
                                 const ReferenceReader &ref_reader);
    bool DetectSplicingHazard(const AlleleIdentity& alt_x) const;
};

bool IsAllelePairConnected(const AlleleIdentity& alt1, const AlleleIdentity& alt2);

template <typename MyIndexType>
bool IsOverlappingWindows(MyIndexType win1_start, MyIndexType win1_end, MyIndexType win2_start, MyIndexType win2_end);

#endif //CLASSIFYVARIANT_H
