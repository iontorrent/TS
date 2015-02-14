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

  bool isHotSpot;           // Signifies a hotspot variant (set per variant for all alleles, regardless of their specific origin)
  bool isProblematicAllele; // There is something wrong with this allele, we should filter it.
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
    isProblematicAllele = false;
    doRealignment  = false;
  }
};

// ------------------------------------------------------------------------

// example variants:
// VarButton isSNP true
//  genome       44 45 46 47 48  49 (0 based)
//  ref is        A  A  A  A  T  T
//  alt is                    A
//  altAllele is              A
//  left_anchor               0 (always)
//  right_anchor              0 (always)
//  inDelLength               0 (always)
//  ref_hp_length             2 (>=1) -- T T satrting at 48
//  start_window  <=48 -- calculated as min over all alt Alleles
//  end_window    >=49 -- calculated as max over all alt Alleles

// VarButton isMNV true
//  genome       44 45 46 47 48  49 (0 based)
//  ref is        A  A  A  A  T  T
//  alt is              A  G  C
//  altAllele is        A  G  C
//  left_anchor         1
//  right_anchor                 0
//  inDelLength         0 (always)
//  ref_hp_length       2 (>=1 always) -- T T starting at 48
//  start_window   
//  end_window     

//  VarButton isIndel true, isDeletion false
//  genome       42 42 44 45 46 47 48 49 50 51 52   (0 based)
//  ref is        C  A  A  A  A  T  G  T  A  A  A
//  alt is                       d  C  G
//  altAllele is                 T  C  G 
//  left_anchor                  1
//  right_anchor                 0
//  inDelLength                  2
//  ref_hp_length                1 (G at 49)
//  start_window
//  end_window


// VarButton isIndel false, isInsertion true
//  genome       42 42 44 45 46 47  48 49 50 51 52   (0 based)
//  ref is        C  C  A  A  A  A   T  G  T  A  A  A
//  alt is                       G  GC
//  altAllele is                 G  G  C
//  left_anchor                  0
//  right_anchor                 0
//  inDelLength                  3
//  ref_hp_length                4  (A at 47)
//  start_window
//  end_window


class AlleleIdentity {
  public:
    VarButton     status;     //!< A bunch of flags saying what's going on with this allele
    string        altAllele;  //!< May contain left and/or right anchor bases, cannot be empty
    int           DEBUG;

    // useful context
    int left_anchor;        //!< Number of left bases that are common between the ref. and alt. allele
    int right_anchor;         //!< Number of right bases that are common between the ref. and alt. allele
                              //   left_anchor + right_anchor <= shorter allele length
    int inDelLength;          //!< Difference in length between longer and shorter allele
    int ref_hp_length;        //!< First base change is occurring in an HP of length ref_hp_length
    int start_window;         //!< Start of window of interest for this variant
    int end_window;           //!< End of window of interest for this variant

    // need to know when I do filtering
    float  sse_prob_positive_strand;
    float  sse_prob_negative_strand;
    vector<string> filterReasons;

    bool indelActAsHPIndel;   // Switch to make all indels match HPIndel behavior

    AlleleIdentity() {

      inDelLength = 0;
      ref_hp_length = 0;
      //modified_start_pos = 0;
      left_anchor = 0;
      right_anchor = 0;
      start_window = 0;
      end_window = 0;
      DEBUG = 0;
      
      // filterable statuses
      sse_prob_positive_strand = 0;
      sse_prob_negative_strand = 0;

      indelActAsHPIndel = false;
    };

    bool Ordinary() {
      return(status.isIndel && !(status.isHPIndel));
    };
    
    bool ActAsSNP(){
      // return(status.isSNP || status.isMNV || (status.isIndel && !status.isHPIndel));
      if (indelActAsHPIndel)
	return(status.isSNP || status.isPaddedSNP);
      else
	return(status.isSNP || status.isPaddedSNP || (status.isIndel && !status.isHPIndel));
    }
    bool ActAsMNP(){
      return(status.isMNV);
    }
    bool ActAsHPIndel(){
      if (indelActAsHPIndel)
	return(status.isIndel);
      else
	return(status.isIndel && status.isHPIndel);
    }
    //void DetectPotentialCorrelation(const LocalReferenceContext &reference_context);
    bool SubCategorizeInDel(const LocalReferenceContext &reference_context,
                            const ReferenceReader &ref_reader, int chr_idx);
    void IdentifyHPdeletion(const LocalReferenceContext& reference_context);
    void IdentifyHPinsertion(const LocalReferenceContext& reference_context,
        const ReferenceReader &ref_reader, int chr_idx);
    bool IdentifyDyslexicMotive(char base, int position,
        const ReferenceReader &ref_reader, int chr_idx);

    void SubCategorizeSNP(const LocalReferenceContext &reference_contextl);
    void SubCategorizeMNP(const LocalReferenceContext &reference_contextl);
    bool getVariantType(const string _altAllele, const LocalReferenceContext &reference_context,
                        const TIonMotifSet & ErrorMotifs,
                        const ClassifyFilters &filter_variant,
                        const ReferenceReader &ref_reader,
                        int chr_idx);
    bool CharacterizeVariantStatus(const LocalReferenceContext &reference_context,
                                   const ReferenceReader &ref_reader, int chr_idx);
    bool CheckValidAltAllele(const LocalReferenceContext &reference_context);
    //void ModifyStartPosForAllele(int variantPos);

    bool IdentifyMultiNucRepeatSection(const LocalReferenceContext &seq_context, unsigned int rep_period,
        const ReferenceReader &ref_reader, int chr_idx);
    void CalculateWindowForVariant(const LocalReferenceContext &seq_context, int DEBUG,
        const ReferenceReader &ref_reader, int chr_idx);

    void DetectCasesToForceNoCall(const LocalReferenceContext &seq_context, const ClassifyFilters &filter_variant,
        const VariantSpecificParams& variant_specific_params);
    void DetectLongHPThresholdCases(const LocalReferenceContext &seq_context, int maxHPLength);
    void DetectNotAVariant(const LocalReferenceContext &seq_context);
    void PredictSequenceMotifSSE(const LocalReferenceContext &reference_context, const TIonMotifSet & ErrorMotifs,
                                 const ReferenceReader &ref_reader, int chr_idx);
};




#endif //CLASSIFYVARIANT_H
