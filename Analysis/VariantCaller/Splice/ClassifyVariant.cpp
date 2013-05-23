/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ClassifyVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "ClassifyVariant.h"
#include "ErrorMotifs.h"

// This function only works for the 1Base -> 1 Base snp representation
void AlleleIdentity::SubCategorizeSNP(LocalReferenceContext &reference_context, int min_hp_for_overcall) {

// This classification only works if allele lengths == 1
  char altBase = altAllele.at(0);
  ref_hp_length = reference_context.my_hp_length.at(0);
  // Construct legacy variables from new structure of LocalReferenceContext
  char refBaseLeft = (reference_context.position0 == reference_context.my_hp_start_pos.at(0)) ? reference_context.ref_left_hp_base : reference_context.reference_allele.at(0);
  char refBaseRight = (reference_context.position0 == reference_context.my_hp_start_pos.at(0) + reference_context.my_hp_length.at(0) - 1) ? reference_context.ref_right_hp_base : reference_context.reference_allele.at(0);

  //in case of SNP test case for possible undercall/overcall leading to FP SNP evidence
  if (reference_context.my_hp_length.at(0) > 1 && (altBase == refBaseLeft || altBase == refBaseRight)) {
    if (altBase == refBaseLeft && reference_context.left_hp_length > min_hp_for_overcall) {
      status.isOverCallUnderCallSNP = true;
      // None of these variables is used in ensemble eval
      underCallLength = reference_context.my_hp_length.at(0) - 1;
      underCallPosition = reference_context.position0; //going to 0-based anchor position
      overCallLength = reference_context.left_hp_length + 1;
      overCallPosition = (reference_context.position0) - (reference_context.left_hp_length);
      // */
    }
    else
      if (altBase == refBaseRight && reference_context.right_hp_length > min_hp_for_overcall) {
        status.isOverCallUnderCallSNP = true;
        // None of these variables is used in ensemble eval
        underCallLength = reference_context.my_hp_length.at(0) - 1;
        underCallPosition = (reference_context.position0) - (reference_context.left_hp_length); //going to 0-based anchor position
        overCallLength = reference_context.left_hp_length + 1;
        overCallPosition = reference_context.position0;
        // */
      }
  }
  if (DEBUG > 0)
    cout << " is a snp. OverUndercall? " << status.isOverCallUnderCallSNP << endl;
}

// CK: Newly written function.
// Old function was problematic; only worked accidentally because of a bug in local context object.
bool AlleleIdentity::SubCategorizeInDel(LocalReferenceContext& reference_context) {

  status.isDeletion  = (reference_context.reference_allele.length() > altAllele.length());
  status.isInsertion = (reference_context.reference_allele.length() < altAllele.length());
  string shorterAllele, longerAllele;
  ref_hp_length = 1;

  // Sanity checks
  if (anchor_length == 0) {
    cerr << "Non-fatal ERROR in InDel classification: InDel needs at least one anchor base. VCF position: "
    	 << reference_context.contigName << ":" << reference_context.position0+1
         << " Ref: " << reference_context.reference_allele <<  " Alt: " << altAllele << endl;
    cout << "Non-fatal ERROR in InDel classification: InDel needs at least one anchor base. VCF position: "
    	 << reference_context.contigName << ":" << reference_context.position0+1
         << " Ref: " << reference_context.reference_allele <<  " Alt: " << altAllele << endl;
    // Function level above turns this into a ",NOCALLxBADCANDIDATE"
    return (false);
  }

  if (status.isDeletion) {
    longerAllele  = reference_context.reference_allele;
    shorterAllele = altAllele;
    status.isHPIndel = reference_context.my_hp_length.at(anchor_length) > 1;
    ref_hp_length = reference_context.my_hp_length.at(anchor_length);
  }
  else { // Insertion

    shorterAllele = reference_context.reference_allele;
    longerAllele  = altAllele;
    char ref_base_right_of_insertion;
    if (anchor_length == (int)reference_context.reference_allele.length())
      ref_base_right_of_insertion = reference_context.ref_right_hp_base;
    else
      ref_base_right_of_insertion = reference_context.reference_allele.at(anchor_length);

    // Investigate HPIndel -> if length change results in an HP > 1.
    if (longerAllele.at(anchor_length) == longerAllele.at(anchor_length - 1)) {
      status.isHPIndel = true;
      ref_hp_length = reference_context.my_hp_length.at(anchor_length - 1);
    }
    if (longerAllele.at(anchor_length) == ref_base_right_of_insertion) {
      status.isHPIndel = true;
      ref_hp_length = reference_context.right_hp_length;
    }
    if (!status.isHPIndel)
      ref_hp_length = 0;
  }
  inDelLength  = longerAllele.length() - shorterAllele.length();

  // only isHPIndel if all inserted/deleted bases past anchor bases are equal.
  for (int b_idx = anchor_length + 1; b_idx < anchor_length + inDelLength; b_idx++) {
    if (longerAllele[b_idx] != longerAllele[anchor_length])
      status.isHPIndel = false;
  }

  if (DEBUG > 0)
    cout << " is an InDel. Insertion?: " << status.isInsertion << " InDelLength: " << inDelLength << " isHPIndel?: " << status.isHPIndel << " ref. HP length: " << ref_hp_length << endl;
  return (true);
}


bool AlleleIdentity::CharacterizeVariantStatus(LocalReferenceContext &reference_context, int min_hp_for_overcall) {
  //cout << "Hello from CharacterizeVariantStatus; " << altAllele << endl;
  bool is_ok = true;
  status.isIndel   = false;
  status.isHPIndel = false;
  status.isSNP     = false;
  status.isMNV     = false;

  // Get Anchor length
  anchor_length = 0;
  unsigned int a_idx = 0;
  while (a_idx < altAllele.length() and a_idx < reference_context.reference_allele.length()
         and altAllele[a_idx] == reference_context.reference_allele[a_idx]) {
    a_idx++;
    anchor_length++;
  }
  if (DEBUG > 0)
    cout << "- Alternative Allele " << altAllele << " (anchor length " << anchor_length << ")";

  // Change classification to better reflect what we can get with haplotyping
  if (altAllele.length() != reference_context.reference_allele.length()) {
    status.isIndel = true;
    is_ok = SubCategorizeInDel(reference_context);
  }
  else
    if ((int)altAllele.length() == 1) { // Categorize function only works with this setting
      status.isSNP = true;
      SubCategorizeSNP(reference_context, min_hp_for_overcall);
    }
    else {
      status.isMNV = true;
      if (anchor_length < (int)reference_context.reference_allele.length())
        ref_hp_length = reference_context.my_hp_length.at(anchor_length);
      if (DEBUG > 0)
        cout << " is an MNV." << endl;
    }
  return (is_ok);
}

bool AlleleIdentity::CheckValidAltAllele(LocalReferenceContext &reference_context) {

  for (unsigned int idx=0; idx<altAllele.length(); idx++) {
    switch (altAllele.at(idx)) {
      case ('A'): break;
      case ('C'): break;
      case ('G'): break;
      case ('T'): break;
      default:
        cerr << "Non-fatal ERROR: Alt Allele contains characters other than ACGT at VCF Position "
             << reference_context.contigName << ":" << reference_context.position0+1
             << " Alt Allele: " << altAllele << endl;
        cout << "Non-fatal ERROR: Alt Allele contains characters other than ACGT at VCF Position "
             << reference_context.contigName << ":" << reference_context.position0+1
             << " Alt Allele: " << altAllele << endl;
        return (false);
    }
  }
  return true;
}


// Entry point for variant classification
bool AlleleIdentity::getVariantType(
  string _altAllele,
  LocalReferenceContext &reference_context,
  const string & local_contig_sequence,
  TIonMotifSet & ErrorMotifs,
  ClassifyFilters &filter_variant) {

  altAllele = _altAllele;
  bool is_ok = reference_context.context_detected;

  // We should now be guaranteed a valid variant position in here
  if (is_ok) {
    is_ok = CharacterizeVariantStatus(reference_context, filter_variant.min_hp_for_overcall);

    PredictSequenceMotifSSE(reference_context, local_contig_sequence, ErrorMotifs);

    // Just confusing -> refactor away
    ModifyStartPosForAllele(reference_context.position0 + 1);
  }
  is_ok = is_ok and CheckValidAltAllele(reference_context);

  if (!is_ok) {
    status.isNoCallVariant = true;
    filterReason += ",NOCALLxBADCANDIDATE";
  }

  return(is_ok);
}


// Should almost not be called anywhere anymore...
void AlleleIdentity::ModifyStartPosForAllele(int variantPos) {
  if (status.isSNP || status.isMNV)
    modified_start_pos = variantPos - 1; //0 based position for SNP location
  else
    modified_start_pos = variantPos;
}


// Checks the reference area around variantPos for a multi-nucleotide repeat and it's span
// Logic: When shifting a window of the same period as the MNR, the base entering the window has to be equal to the base leaving the window.
// example with period 2: XYZACACA|CA|CACAIJK
bool AlleleIdentity::IdentifyMultiNucRepeatSection(const string &local_contig_sequence, const LocalReferenceContext &seq_context, unsigned int rep_period) {

  //cout << "Hello from IdentifyMultiNucRepeatSection with period " << rep_period << "!"<< endl;
  unsigned int variantPos = seq_context.position0 + anchor_length;
  if (variantPos + rep_period >= local_contig_sequence.length())
    return (false);

  CircluarBuffer<char> window(rep_period);
  for (unsigned int idx = 0; idx < rep_period; idx++)
    window.assign(idx, local_contig_sequence[variantPos+idx]);

  // Investigate (inclusive) start position of MNR region
  start_window = variantPos - 1; // 1 anchor base
  window.shiftLeft(1);
  while (start_window > 0 and window.first() == local_contig_sequence.at(start_window)) {
    start_window--;
    window.shiftLeft(1);
  }

  // Investigate (exclusive) end position of MNR region
  end_window = variantPos + rep_period;
  for (unsigned int idx = 0; idx < rep_period; idx++)
    window.assign(idx, local_contig_sequence[variantPos+idx]);
  window.shiftRight(1);
  while (end_window < (int)local_contig_sequence.length() and window.last() == local_contig_sequence[end_window]) {
    end_window++;
    window.shiftRight(1);
  }

  //cout << "Found repeat stretch of length: " << (end_window - start_window) << endl;
  // Require that a stretch of at least 3*rep_period has to be found to count as a MNR
  if ((end_window - start_window) >= (3*(int)rep_period)) {
    // Correct start and end of the window if they are not fully outside variant allele
    if (start_window >= seq_context.position0)
        start_window = seq_context.my_hp_start_pos.at(0) - 1;
    if (end_window <= seq_context.right_hp_start) {
      if (status.isInsertion)
        end_window = seq_context.right_hp_start + seq_context.right_hp_length + 1;
      else
        end_window = seq_context.right_hp_start + 1;
    }
    return (true);
  }
  else
    return (false);
}


// -------------------------------------------------------------
// xxx All the Nucleotide repeat functions start here
// Functions below not used any more <- replaced with AlleleIdentity::IdentifyMultiNucRepeatSection

bool DiNucRepeat(const string &local_contig_sequence, int variant_start_pos, AlleleIdentity &variant_identity, int &mnr_start, int &mnr_end) {
  string mnrAllele = "";
  char currentBase;
  char nextBase;

  stringstream mnrss;
  int homPolyLength = 0;
  int j = 0;
  bool isMNR = false;

  string refSubSeq;
  refSubSeq = local_contig_sequence.substr(variant_start_pos - 10, 100);
  size_t seqlength =  refSubSeq.length();


  //check diNuc repeat
  for (size_t c = 0; c < seqlength - 2;) {
    currentBase = refSubSeq[c];
    nextBase = refSubSeq[c+1];

    homPolyLength = 1;

    j = 2;
    while (((c + j + 2) < seqlength) && /*(currentBase != nextBase)
           && */ (refSubSeq[c+j] == currentBase
                             && refSubSeq[c+j+1] == nextBase)) {
      homPolyLength++;
      j = j + 2;
    }
    if (homPolyLength > 3) {
      isMNR = true;
      mnrss << currentBase << nextBase ;
      mnr_start = c + variant_start_pos - 10;
      mnr_end = c + (homPolyLength * 2) + variant_start_pos - 10;

      if (mnr_start <= variant_start_pos && mnr_end >= variant_start_pos) {
        mnrAllele = mnrss.str();
        break;
      }

    }

    if (homPolyLength > 1)
      c += homPolyLength * 2;
    else
      c++;
  }
  //check if the repeat sequence spans the variant position
  if (isMNR && (mnr_start > variant_start_pos || mnr_end <= variant_start_pos))
    isMNR = false;

  return(isMNR);
}

bool TriNucRepeat(const string &local_contig_sequence, int variant_start_pos, AlleleIdentity &variant_identity, int &mnr_start, int &mnr_end) {
  string mnrAllele = "";
  char currentBase;
  char nextBase;
  char nextnextBase;

  stringstream mnrss;
  int homPolyLength = 0;
  int j = 0;

  bool isMNR = false;

  string refSubSeq;
  refSubSeq = local_contig_sequence.substr(variant_start_pos - 10, 100);
  size_t seqlength =  refSubSeq.length();

  //check for tri Nuc repeat ATCATCATC....
  for (size_t c = 0; c < seqlength - 2;) {
    currentBase = refSubSeq[c];
    nextBase = refSubSeq[c+1];
    nextnextBase = refSubSeq[c+2];
    homPolyLength = 1;

    j = 3;
    while (((c + j + 3) < seqlength) /*&& (currentBase != nextBase)
           && (nextBase != nextnextBase) */ && (refSubSeq[c+j] == currentBase
               && refSubSeq[c+j+1] == nextBase && refSubSeq[c+j+2] == nextnextBase)) {
      homPolyLength++;
      j = j + 3;
    }
    if (homPolyLength > 3) {
      isMNR = true;
      mnrss << currentBase << nextBase << nextnextBase;
      mnr_start = c + variant_start_pos - 10;
      mnr_end = c + (homPolyLength * 3) + variant_start_pos - 10;

      if (mnr_start <= variant_start_pos && mnr_end >= variant_start_pos) {
        mnrAllele = mnrss.str();
        break;
      }
      //cout << "MNR 3 " << mnr_start << " end " << mnr_end << " c = " << c << " mnr allele " << mnrss.str() << endl;
    }

    if (homPolyLength > 1)
      c += homPolyLength * 3;
    else
      c++;
  }

  //check if the repeat sequence spans the variant position
  if (isMNR && (mnr_start > variant_start_pos || mnr_end <= variant_start_pos))
    isMNR = false;

  return(isMNR);
}

bool TetraNucRepeat(const string &local_contig_sequence, int variant_start_pos, AlleleIdentity &variant_identity, int &mnr_start, int &mnr_end) {
  string mnrAllele = "";
  char currentBase;
  char nextBase;
  char nextnextBase;
  char nextnextnextBase;
  stringstream mnrss;
  int homPolyLength = 0;
  int j = 0;
  bool isMNR = false;

  string refSubSeq;
  refSubSeq = local_contig_sequence.substr(variant_start_pos - 10, 100);
  size_t seqlength =  refSubSeq.length();

//check for four Nuc Repeat CTCGCTCGCTCG.....
  for (size_t c = 0; c < seqlength - 3;) {
    currentBase = refSubSeq[c];
    nextBase = refSubSeq[c+1];
    nextnextBase = refSubSeq[c+2];
    nextnextnextBase = refSubSeq[c+3];
    homPolyLength = 1;

    j = 4;
    while (((c + j + 4) < seqlength) && /* (currentBase != nextBase)
           && (nextBase != nextnextBase)  && nextnextBase != nextnextnextBase
           && */ (refSubSeq[c+j] == currentBase && refSubSeq[c+j+1] == nextBase
                             && refSubSeq[c+j+2] == nextnextBase && refSubSeq[c+j+3] == nextnextnextBase)) {
      homPolyLength++;
      j = j + 4;
    }
    if (homPolyLength > 3) {
      isMNR = true;
      mnrss << currentBase << nextBase << nextnextBase << nextnextnextBase;
      mnr_start = c + variant_start_pos - 10;
      mnr_end = c + (homPolyLength * 4) + variant_start_pos - 10;

      if (mnr_start <= variant_start_pos && mnr_end >= variant_start_pos) {
        mnrAllele = mnrss.str();
        break;
      }

    }

    if (homPolyLength > 1)
      c += homPolyLength * 4;
    else
      c++;
  }

  //check if the repeat sequence spans the variant position
  if (isMNR && (mnr_start > variant_start_pos || mnr_end <= variant_start_pos))
    isMNR = false;

  return(isMNR);
}

//@TODO: fix this copy/paste code to do the right thing
bool CheckMNR(const string &local_contig_sequence, int variant_start_pos, AlleleIdentity &variant_identity, int &mnr_start, int &mnr_end) {

  bool isMNR = false;
  bool isDiNuc = false;
  bool isTriNuc = false;
  bool isTetraNuc = false;
  int dimnr_start = 0;
  int dimnr_end = 0;
  int trimnr_start = 0;
  int trimnr_end = 0;
  int tetramnr_start = 0;
  int tetramnr_end = 0;

  isDiNuc = DiNucRepeat(local_contig_sequence, variant_start_pos, variant_identity, dimnr_start, dimnr_end);

  isTriNuc = TriNucRepeat(local_contig_sequence, variant_start_pos, variant_identity, trimnr_start, trimnr_end);

  isTetraNuc = TetraNucRepeat(local_contig_sequence, variant_start_pos, variant_identity, tetramnr_start, tetramnr_end);

  if (isDiNuc || isTriNuc || isTetraNuc) {
    isMNR = true;
    //find the most optimal start and end positions
    if (isDiNuc) {
      mnr_start = dimnr_start;
      mnr_end = dimnr_end;

    }
    if (isTriNuc) {

      mnr_start = trimnr_start;
      mnr_end = trimnr_end;

    }
    if (isTetraNuc) {
      mnr_start = tetramnr_start;
      mnr_end = tetramnr_end;
    }

  }

  return(isMNR);
}

// Functions above not used any more <- replaced with AlleleIdentity::IdentifyMultiNucRepeatSection
// -------------------------------------------------------

// Function not used any more
void WindowSizedForLongAllele(const string &local_contig_sequence, int variant_start_pos, string &allele, AlleleIdentity &variant_identity, int &start_window, int &end_window, int DEBUG) {
  string refSubSeq;
  size_t endInc = 20;
  refSubSeq = local_contig_sequence.substr(variant_start_pos - 10, endInc); // not used

  if (variant_start_pos > (int)local_contig_sequence.length()) {
    start_window = -1;
    end_window = -1;
  }
  else {

    start_window = max(variant_start_pos - 2, 0);
    end_window = variant_start_pos + 2;
    if (variant_identity.status.isDeletion)
      end_window += allele.length();
  }
}
// -----------------------------------------------------------------


void AlleleIdentity::CalculateWindowForVariant(LocalReferenceContext seq_context, const string &local_contig_sequence, int DEBUG) {

  // If we have an invalid vcf candidate, set a length zero window and exit
  if (!seq_context.context_detected) {
    start_window = seq_context.position0;
    end_window = seq_context.position0;
    return;
  }

  // Check for MNRs first, for InDelLengths 2,3,4
  if (status.isIndel and !status.isHPIndel and inDelLength < 5)
    for (int rep_period = 2; rep_period < 5; rep_period++)
      if (IdentifyMultiNucRepeatSection(local_contig_sequence, seq_context, rep_period)) {
        if (DEBUG > 0) {
          cout << "MNR found in allele " << seq_context.reference_allele << " -> " << altAllele << endl;
          cout << "Window for allele " << altAllele << ": (" << start_window << ") ";
          for (int p_idx = start_window; p_idx < end_window; p_idx++)
            cout << local_contig_sequence.at(p_idx);
          cout << " (" << end_window << ") " << endl;
        }
        return; // Found a matching period and computed window
      }

  // OK, not an MNR. Moving on along to InDels.
  // need at least one anchor base left and right of InDel allele
  if (status.isIndel) {
    if (status.isDeletion) {
      start_window = seq_context.my_hp_start_pos.at(anchor_length) - 1;
      end_window = seq_context.right_hp_start;
    }
    else { // Insertions require a bit more thought
      if (altAllele.at(anchor_length) == altAllele.at(anchor_length - 1))
        start_window = seq_context.my_hp_start_pos.at(anchor_length - 1) - 1;
      else
        start_window = seq_context.position0 + anchor_length - 1; // 1 anchor base before we insert a new HP
      if (start_window < 0)
        start_window = 0; // Safety for something happening in the first HP of the ref. and not left-aligned...
      end_window = seq_context.right_hp_start;
      if (altAllele.at(altAllele.length() - 1) == seq_context.ref_right_hp_base)
        end_window += seq_context.right_hp_length;
    }
    if ((unsigned int)end_window < local_contig_sequence.length())
      end_window++; // anchor base to the right
  }
  else {
    // SNPs and MNVs are 1->1 base replacements
    // make window as short as possible, only around bases to be replaced
    // Think: include full HPs affected like in the InDel case? <- no for now, like in old code
    start_window = seq_context.position0;
    end_window = seq_context.position0 + seq_context.reference_allele.length();
  } // */

  if (DEBUG > 0) {
    cout << "Window for allele " << altAllele << ": (" << start_window << ") ";
    for (int p_idx = start_window; p_idx < end_window; p_idx++)
      cout << local_contig_sequence.at(p_idx);
    cout << " (" << end_window << ") " << endl;
  }
}


// ------------------------------------------------------------------------------
// Filtering functions

void AlleleIdentity::PredictSequenceMotifSSE(LocalReferenceContext &reference_context, const  string &local_contig_sequence, TIonMotifSet & ErrorMotifs) {

  //cout << "Hello from PredictSequenceMotifSSE" << endl;
  sse_prob_positive_strand = 0;
  sse_prob_negative_strand = 0;
  //long vcf_position = reference_context.position0+1;
  long var_position = reference_context.position0 + anchor_length; // This points to the first deleted base

  string seqContext;
  // status.isHPIndel && status.isDeletion implies reference_context.my_hp_length.at(anchor_length) > 1
  if (status.isHPIndel && status.isDeletion) {

    // cout << start_pos << "\t" << variant_context.refBaseAtCandidatePosition << variant_context.ref_hp_length << "\t" << variant_context.refBaseLeft << variant_context.left_hp_length << "\t" << variant_context.refBaseRight  << variant_context.right_hp_length << "\t";

    unsigned context_left = var_position >= 10 ? 10 : var_position;
    if (var_position + reference_context.my_hp_length.at(anchor_length) + 10 < (int) local_contig_sequence.length())
      seqContext = local_contig_sequence.substr(var_position - context_left, context_left + (unsigned int)reference_context.my_hp_length.at(anchor_length) + 10);
    else
      seqContext = local_contig_sequence.substr(var_position - context_left);

    if (seqContext.length() > 0 && context_left < seqContext.length()) {
      sse_prob_positive_strand = ErrorMotifs.get_sse_probability(seqContext, context_left);

      // cout << seqContext << "\t" << context_left << "\t" << sse_prob_positive_strand << "\t";

      context_left = seqContext.length() - context_left - 1;
      string reverse_seqContext;
      ReverseComplement(seqContext, reverse_seqContext);

      sse_prob_negative_strand = ErrorMotifs.get_sse_probability(reverse_seqContext, context_left);

      // cout << reverse << "\t" << context_left << "\t" << sse_prob_negative_strand << "\t";

    }
  }
}


void AlleleIdentity::DetectSSEForNoCall(float sseProbThreshold, float minRatioReadsOnNonErrorStrand, map<string, vector<string> > & allele_info, unsigned _altAlleleIndex) {

  if (sse_prob_positive_strand >= sseProbThreshold && sse_prob_negative_strand >= sseProbThreshold) {
    status.isNoCallVariant = true;
    filterReason += ",NOCALLxPredictedSSE";
  }
  else {
    unsigned alt_counts_positive = atoi(allele_info.at("SAF")[_altAlleleIndex].c_str());
    unsigned alt_counts_negative = atoi(allele_info.at("SAR")[_altAlleleIndex].c_str());

    // remember to trap zero-count div by zero here with safety value
    if (sse_prob_positive_strand >= sseProbThreshold && alt_counts_negative / (alt_counts_positive + alt_counts_negative + 0.1f) < minRatioReadsOnNonErrorStrand) {
      status.isNoCallVariant = true;
      filterReason += ",NOCALLxPositiveSSE";
    }

    if (sse_prob_negative_strand >= sseProbThreshold && alt_counts_positive / (alt_counts_positive + alt_counts_negative + 0.1f) < minRatioReadsOnNonErrorStrand) {
      filterReason += ",NOCALLxNegativeSSE";
      status.isNoCallVariant = true;
    }
  }
  // cout << alt_counts_positive << "\t" << alt_counts_negative << "\t" << ref_counts_positive << "\t" << ref_counts_negative << endl;
}


void AlleleIdentity::DetectLongHPThresholdCases(LocalReferenceContext seq_context, int maxHPLength, int adjacent_max_length) {
  if (status.isIndel && ref_hp_length > maxHPLength) {
    filterReason += ",NOCALLxHPLEN";
    status.isNoCallVariant = true;
  }
  // Turning off the over- undercall filter, should trigger quality filters if it is too messy to call
  //if (status.isOverCallUnderCallSNP && (seq_context.left_hp_length > adjacent_max_length || seq_context.right_hp_length > adjacent_max_length))  {
  //  status.isNoCallVariant = true;
  //  filterReason+=",NOCALLxADJACENTHPLEN";
  //}
}

// XXX Shouldn't this be a reference call and not a no call?
void AlleleIdentity::DetectNotAVariant(LocalReferenceContext seq_context) {
  if (altAllele.compare(seq_context.reference_allele) == 0) {
    //incorrect allele status is passed thru make it a no call
    status.isNoCallVariant = true;
    filterReason += ",NOCALLxNOTAVARIANT";
  }
}


void AlleleIdentity::DetectCasesToForceNoCall(LocalReferenceContext seq_context, ClassifyFilters &filter_variant, map<string, vector<string> >& info, unsigned _altAlleIndex) {

  //filterReason = ""; moved up, Classifier might already throw a NoCall for a bad candidate
  DetectNotAVariant(seq_context);
  DetectLongHPThresholdCases(seq_context, filter_variant.hp_max_length, filter_variant.adjacent_max_length);
  DetectSSEForNoCall(filter_variant.sseProbThreshold, filter_variant.minRatioReadsOnNonErrorStrand, info, _altAlleIndex);
}

// ====================================================================

void MultiAlleleVariantIdentity::GetMultiAlleleVariantWindow() {

  window_start = -1;
  window_end   = -1;
  // TODO: Should we exclude already filtered alleles?
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    //if (allele_identity_vector[i_allele].status.isNoCallVariant) {
    if (allele_identity_vector[i_allele].start_window < window_start or window_start == -1)
      window_start = allele_identity_vector[i_allele].start_window;
    if (allele_identity_vector[i_allele].end_window > window_end or window_end == -1)
      window_end = allele_identity_vector[i_allele].end_window;
  }
}

// ------------------------------------------------------------

void MultiAlleleVariantIdentity::SetupAllAlleles(vcf::Variant ** candidate_variant, const string & local_contig_sequence, ExtendParameters *parameters, InputStructures &global_context) {

  seq_context.DetectContext(local_contig_sequence, candidate_variant, global_context.DEBUG);
  allele_identity_vector.resize((*candidate_variant)->alt.size());
  variant = candidate_variant;

  //now calculate the allele type (SNP/Indel/MNV/HPIndel etc.) and window for hypothesis calculation for each alt allele.
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {

	// TODO: Hotspot should be an allele property but we only set all or none to Hotspots, depending on the vcf record
    allele_identity_vector[i_allele].status.isHotSpot = (*candidate_variant)->isHotSpot;
    allele_identity_vector[i_allele].filterReason.clear();
    allele_identity_vector[i_allele].DEBUG = global_context.DEBUG;

    allele_identity_vector[i_allele].getVariantType((*candidate_variant)->alt[i_allele], seq_context,
        local_contig_sequence, global_context.ErrorMotifs,  parameters->my_controls.filter_variant);
    allele_identity_vector[i_allele].CalculateWindowForVariant(seq_context, local_contig_sequence, global_context.DEBUG);
  }
}

// ------------------------------------------------------------

void MultiAlleleVariantIdentity::FilterAllAlleles(vcf::Variant ** candidate_variant, ClassifyFilters &filter_variant) {
  if (seq_context.context_detected) {
    for (uint8_t i_allele = 0; i_allele < (*candidate_variant)->alt.size(); i_allele++) {
      allele_identity_vector[i_allele].DetectCasesToForceNoCall(seq_context,
    		                            filter_variant, (*candidate_variant)->info, i_allele);
    }
  }
}
