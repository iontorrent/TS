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
      isIndel        = false;
      isHotSpot      = false;
      isProblematicAllele = false;
      doRealignment  = false;
    }
};


// ------------------------------------------------------------------------

class AlleleIdentity {
  public:
    VarButton     status;     //!< A bunch of flags saying what's going on with this allele
    string        altAllele;
    int           DEBUG;

    // useful context
    int anchor_length;        //!< Number of left bases that are common between the ref. and alt. allele
    int right_anchor;         //!< Number of right bases that are common between the ref. and alt. allele
                              //   anchor_length + right_anchor <= shorter allele length
    int inDelLength;          //!< Differnence in length between longer and shorter allele
    int ref_hp_length;        //!< First base change is occurring in an HP of length ref_hp_length
    int start_window;         //!< Start of window of interest for this variant
    int end_window;           //!< End of window of interest for this variant

    // need to know when I do filtering
    float  sse_prob_positive_strand;
    float  sse_prob_negative_strand;
    string filterReason;

    AlleleIdentity() {

      inDelLength = 0;
      ref_hp_length = 0;
      //modified_start_pos = 0;
      anchor_length = 0;
      right_anchor = 0;
      start_window = 0;
      end_window = 0;
      DEBUG = 0;
      
      // filterable statuses
      sse_prob_positive_strand = 0;
      sse_prob_negative_strand = 0;
    };

    bool Ordinary() {
      return(status.isIndel && !(status.isHPIndel));
    };
    
    bool ActAsSNP(){
      return(status.isSNP ||status.isMNV || (status.isIndel && !status.isHPIndel));
    }
    bool ActAsHPIndel(){
      return(status.isIndel && status.isHPIndel);
    }
    //void DetectPotentialCorrelation(const LocalReferenceContext &reference_context);
    bool SubCategorizeInDel(const string & local_contig_sequence, const LocalReferenceContext &reference_context);
    void IdentifyHPdeletion(const LocalReferenceContext& reference_context);
    void IdentifyHPinsertion(const LocalReferenceContext& reference_context, const string & local_contig_sequence);
    bool IdentifyDyslexicMotive(const string & local_contig_sequence, char base, int position);

    void SubCategorizeSNP(const LocalReferenceContext &reference_contextl);
    bool getVariantType(const string _altAllele, const LocalReferenceContext &reference_context,
                        const string &local_contig_sequence, TIonMotifSet & ErrorMotifs,
                        const ClassifyFilters &filter_variant);
    bool CharacterizeVariantStatus(const string & local_contig_sequence, const LocalReferenceContext &reference_context);
    bool CheckValidAltAllele(const LocalReferenceContext &reference_context);
    //void ModifyStartPosForAllele(int variantPos);

    bool IdentifyMultiNucRepeatSection(const string &local_contig_sequence, const LocalReferenceContext &seq_context, unsigned int rep_period);
    void CalculateWindowForVariant(const LocalReferenceContext &seq_context, const string &local_contig_sequence, int DEBUG);

    void DetectCasesToForceNoCall(const LocalReferenceContext &seq_context, const ClassifyFilters &filter_variant);
    void DetectLongHPThresholdCases(const LocalReferenceContext &seq_context, int maxHPLength);
    void DetectNotAVariant(const LocalReferenceContext &seq_context);
    void PredictSequenceMotifSSE(const LocalReferenceContext &reference_context, const  string &local_contig_sequence, TIonMotifSet & ErrorMotifs);
};

// ------------------------------------------------------------------------
// Trying to make more clear what is variant (entry in vcf) and what is an allele property.

// This class stores general information about the variant as well as information about each allele
class MultiAlleleVariantIdentity{

  public:
	vcf::Variant **        variant;                 //!< VCF record of this variant position
	LocalReferenceContext  seq_context;             //!< Reference context of this variant position
    vector<AlleleIdentity> allele_identity_vector;  //!< Detailed information for each candidate allele
    int window_start;
    int window_end;
    bool doRealignment;

    //! @brief  Default constructor
    MultiAlleleVariantIdentity() {
      variant=NULL;
      SetInitialValues();
    }

    MultiAlleleVariantIdentity(vcf::Variant ** candidate_variant) {
      variant=candidate_variant;
      SetInitialValues();
    }


    void SetInitialValues() {
        window_start = -1;
        window_end = -1;
        doRealignment = false;
    };

    //! @brief  Create a detailed picture about this variant and all its alleles
    void SetupAllAlleles(vcf::Variant ** candidate_variant, const string & local_contig_sequence, ExtendParameters *parameters, InputStructures &global_context);

    void FilterAllAlleles(const ClassifyFilters &filter_variant);

    void GetMultiAlleleVariantWindow(const string & local_contig_sequence, int DEBUG);
};

//void calculate_window(string *local_contig_sequence, uint32_t startPos, WhatVariantAmI &variant_identity, int *start_window, int *end_window, int DEBUG);

#endif //CLASSIFYVARIANT_H
