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
    bool isHPIndel;
    bool isSNP ;
    bool isOverCallUnderCallSNP;
    bool isInsertion ;
    bool isDeletion ;
    bool isPotentiallyCorrelated;
    //bool isHP;
    bool isMNV;
    bool isIndel;
    bool isHotSpot;
    bool isNoCallVariant;
    bool isReferenceCall;
    bool isBadAllele;
    bool doRealignment;

    VarButton() {
      isHPIndel = false;
      isSNP = false;
      isOverCallUnderCallSNP = false;
      isInsertion = false;
      isDeletion = false;
      isPotentiallyCorrelated = false;
      //isHP = false;
      isMNV = false;
      isIndel = false;
      isHotSpot=false;
      isNoCallVariant = false;
      isReferenceCall = false;
      isBadAllele = false;
      doRealignment = false;
    }
};


// ------------------------------------------------------------------------

class AlleleIdentity {
  public:
    VarButton     status;     //!< A bunch of flags saying what's going on with this allele
    string        altAllele;
    int           DEBUG;

    // useful context
    int anchor_length;
    int inDelLength; // Used in Splice
    int ref_hp_length;
    int start_window;
    int end_window;
    int modified_start_pos; // This is just confusing

    // need to know when I do filtering
    float  sse_prob_positive_strand;
    float  sse_prob_negative_strand;
    string filterReason;

    // None of these variables is used in ensemble eval -> refactor away
    int underCallPosition;
    int overCallPosition;
    int underCallLength;
    int overCallLength; // */

    // Deleted member variables -> Do not create multiple copies of the same variables, pass them instead.
    //string refAllele;  <- in LocalReferenceContext & vcf variant
    //int refHpLen; <- in LocalReferenceContext
    //long int hp_start_position; <- in LocalReferenceContext
    //long int hp_end_position; <- in LocalReferenceContext
    //int left_hp_length; <- in LocalReferenceContext
    //int right_hp_length; <- in LocalReferenceContext

    AlleleIdentity() {

      inDelLength = 0;
      ref_hp_length = 0;
      modified_start_pos = 0;
      anchor_length = 0;
      start_window = 0;
      end_window = 0;
      underCallPosition = 0;
      underCallLength = 0;
      overCallPosition = 0;
      overCallLength = 0;
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
    void DetectPotentialCorrelation(LocalReferenceContext &reference_context);
    bool SubCategorizeInDel(LocalReferenceContext &reference_context);
    void SubCategorizeSNP(LocalReferenceContext &reference_context, int min_hp_for_overcall);
    bool getVariantType(string _altAllele, LocalReferenceContext &reference_context,
                        const string &local_contig_sequence, TIonMotifSet & ErrorMotifs,
                        ClassifyFilters &filter_variant);
    bool CharacterizeVariantStatus(LocalReferenceContext &reference_context, int min_hp_for_overcall);
    bool CheckValidAltAllele(LocalReferenceContext &reference_context);
    void ModifyStartPosForAllele(int variantPos);

    bool IdentifyMultiNucRepeatSection(const string &local_contig_sequence, const LocalReferenceContext &seq_context, unsigned int rep_period);
    void CalculateWindowForVariant(LocalReferenceContext seq_context, const string &local_contig_sequence, int DEBUG);

    void DetectCasesToForceNoCall(LocalReferenceContext seq_context, ClassifyFilters &filter_variant,
                                  map<string, vector<string> > & info, unsigned _altAlleIndex);
    void DetectSSEForNoCall(float sseProbThreshold, float minRatioReadsOnNonErrorStrand, float relative_safety_level, map<string, vector<string> > & info, unsigned _altAlleIndex);
    void DetectLongHPThresholdCases(LocalReferenceContext seq_context, int maxHPLength, int adjacent_max_length);
    void DetectNotAVariant(LocalReferenceContext seq_context);
    void PredictSequenceMotifSSE(LocalReferenceContext &reference_context, const  string &local_contig_sequence, TIonMotifSet & ErrorMotifs);
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
    };

    //! @brief  Create a detailed picture about this variant and all its alleles
    void SetupAllAlleles(vcf::Variant ** candidate_variant, const string & local_contig_sequence, ExtendParameters *parameters, InputStructures &global_context);

    void FilterAllAlleles(vcf::Variant ** candidate_variant,ClassifyFilters &filter_variant);

    void GetMultiAlleleVariantWindow(const string & local_contig_sequence, int DEBUG);
};

//void calculate_window(string *local_contig_sequence, uint32_t startPos, WhatVariantAmI &variant_identity, int *start_window, int *end_window, int DEBUG);

#endif //CLASSIFYVARIANT_H
