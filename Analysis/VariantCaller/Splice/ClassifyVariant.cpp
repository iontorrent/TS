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
  char refBaseLeft = (reference_context.position0 == reference_context.my_hp_start_pos.at(0)) ?
		  reference_context.ref_left_hp_base : reference_context.reference_allele.at(0);
  char refBaseRight = (reference_context.position0 == reference_context.my_hp_start_pos.at(0) + reference_context.my_hp_length.at(0) - 1) ?
		  reference_context.ref_right_hp_base : reference_context.reference_allele.at(0);

  //in case of SNP test case for possible undercall/overcall leading to FP SNP evidence
  if (altBase == refBaseLeft || altBase == refBaseRight) {
    // Flag possible misalignment for further investigation --- I am an awful hack!
    status.doRealignment = true;

    if (reference_context.my_hp_length.at(0) > 1) {
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
  }
  if (DEBUG > 0) {
    cout << " is a snp. OverUndercall? " << status.isOverCallUnderCallSNP << endl;
    if (status.doRealignment)
      cout << "Possible alignment error detected." << endl;
  }
}

void AlleleIdentity::DetectPotentialCorrelation(LocalReferenceContext& reference_context){
  // in the case in which we are deleting/inserting multiple different bases
  // there may be extra correlation in the measurements because of over/under normalization in homopolymers
  // we head off this case
  
  // count transitions in reference
  // for now the only probable way I can see this happening is  XXXXXXYYYYY case, yielding XY as the variant
  // i.e. NXY -> N (deletion)
  // N -> NXY (insertion)
  // in theory, if the data is bad enough, could happen to 1-mers, but unlikely.
  
  // note that SNPs are anticorrelated, so don't really have this problem
  if (reference_context.reference_allele.length()==3 && altAllele.length()==1 && status.isDeletion)
    if (reference_context.reference_allele[1]!=reference_context.reference_allele[2])
      status.isPotentiallyCorrelated = true;
    
  if (altAllele.length()==3 && reference_context.reference_allele.length()==1 && status.isInsertion)
    if (altAllele[1]!=altAllele[2])
      status.isPotentiallyCorrelated = true;
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
    cout << endl << "Non-fatal ERROR in InDel classification: InDel needs at least one anchor base. VCF position: "
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
    int ref_right_hp_length = 0;
    if (anchor_length == (int)reference_context.reference_allele.length()) {
      ref_base_right_of_insertion = reference_context.ref_right_hp_base;
      ref_right_hp_length = reference_context.right_hp_length;
    }
    else {
      ref_base_right_of_insertion = reference_context.reference_allele.at(anchor_length);
      ref_right_hp_length = reference_context.my_hp_length.at(anchor_length);
    }

    // Investigate HPIndel -> if length change results in an HP > 1.
    if (longerAllele.at(anchor_length) == longerAllele.at(anchor_length - 1)) {
      status.isHPIndel = true;
      ref_hp_length = reference_context.my_hp_length.at(anchor_length - 1);
    }
    if (longerAllele.at(anchor_length) == ref_base_right_of_insertion) {
      status.isHPIndel = true;
      ref_hp_length = ref_right_hp_length;
    }
    if (!status.isHPIndel) {
      ref_hp_length = 0; // A new base is inserted that matches neither the right nor the left side
    }
  }
  inDelLength  = longerAllele.length() - shorterAllele.length();

  // only isHPIndel if all inserted/deleted bases past anchor bases are equal.
  for (int b_idx = anchor_length + 1; b_idx < anchor_length + inDelLength; b_idx++) {
    if (longerAllele[b_idx] != longerAllele[anchor_length])
      status.isHPIndel = false;
  }
  
  DetectPotentialCorrelation(reference_context); // am I a very special problem for the likelihood function?

  if (DEBUG > 0)
    cout << " is an InDel. Insertion?: " << status.isInsertion << " InDelLength: " << inDelLength << " isHPIndel?: " << status.isHPIndel << " ref. HP length: " << ref_hp_length << endl;
  return (true);
}


bool AlleleIdentity::CharacterizeVariantStatus(LocalReferenceContext &reference_context, int min_hp_for_overcall) {
  //cout << "Hello from CharacterizeVariantStatus; " << altAllele << endl;
  bool is_ok = true;
  status.isIndel       = false;
  status.isHPIndel     = false;
  status.isSNP         = false;
  status.isMNV         = false;
  status.doRealignment = false;

  // Get Anchor length
  ref_hp_length = reference_context.my_hp_length.at(0);
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

  if ((reference_context.position0 + altAllele.length()) > local_contig_sequence.length()) {
    is_ok = false;
  }

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
    status.isBadAllele = true;
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
  if (end_window >= (int)local_contig_sequence.length())
    return false;
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
    if (start_window < 0)
      start_window = 0;
    if (end_window > (int)local_contig_sequence.length())
      end_window = (int)local_contig_sequence.length();
    return (true);
  }
  else
    return (false);
}


// -----------------------------------------------------------------


void AlleleIdentity::CalculateWindowForVariant(LocalReferenceContext seq_context, const string &local_contig_sequence, int DEBUG) {

  // If we have an invalid vcf candidate, set a length zero window and exit
  if (!seq_context.context_detected or status.isBadAllele) {
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

  // not an MNR. Moving on along to InDels.
  if (status.isIndel) {
	// Default variant window
    end_window = seq_context.right_hp_start +1; // Anchor base to the right of allele
    start_window = seq_context.position0;

    // Adjustments if necessary
    if (status.isDeletion)
      if (seq_context.my_hp_start_pos.at(anchor_length) == seq_context.my_hp_start_pos.at(0))
        start_window = seq_context.my_hp_start_pos.at(0) - 1;

    if (status.isInsertion) {
      if (altAllele.at(anchor_length) == altAllele.at(anchor_length - 1) and
          seq_context.position0 > (seq_context.my_hp_start_pos.at(anchor_length - 1) - 1))
        start_window = seq_context.my_hp_start_pos.at(anchor_length - 1) - 1;
      if (altAllele.at(altAllele.length() - 1) == seq_context.ref_right_hp_base)
        end_window += seq_context.right_hp_length;
    }

    // Safety
    if (start_window < 0)
      start_window = 0;
    if (end_window > (int)local_contig_sequence.length())
      end_window = (int)local_contig_sequence.length();
  }
  else {
    // SNPs and MNVs are 1->1 base replacements
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

//@TODO Move this to decisiontree!!!!
void AlleleIdentity::DetectSSEForNoCall(float sseProbThreshold, float minRatioReadsOnNonErrorStrand, float relative_safety_level, map<string, vector<string> > & allele_info, unsigned _altAlleleIndex) {

  if (sse_prob_positive_strand >= sseProbThreshold && sse_prob_negative_strand >= sseProbThreshold) {
    status.isNoCallVariant = true;
    filterReason += ",NOCALLxPredictedSSE";
  }
  else {
    // use the >original< counts to determine whether we were affected by this problem 
    unsigned alt_counts_positive = atoi(allele_info.at("SAF")[_altAlleleIndex].c_str());
    unsigned alt_counts_negative = atoi(allele_info.at("SAR")[_altAlleleIndex].c_str());
    // always only one ref count
   unsigned ref_counts_positive = atoi(allele_info.at("SRF")[0].c_str());
   unsigned ref_counts_negative = atoi(allele_info.at("SRR")[0].c_str());

    // remember to trap zero-count div by zero here with safety value
    float safety_val = 0.5f;  // the usual "half-count" to avoid zero
    unsigned total_depth = alt_counts_positive + alt_counts_negative + ref_counts_positive + ref_counts_negative;
    float relative_safety_val = safety_val + relative_safety_level * total_depth;
    
    float strand_ratio = ComputeTransformStrandBias(alt_counts_positive, alt_counts_positive+ref_counts_positive, alt_counts_negative, alt_counts_negative+ref_counts_negative, relative_safety_val);
    
    float transform_threshold = (1-minRatioReadsOnNonErrorStrand)/(1+minRatioReadsOnNonErrorStrand);
    bool pos_strand_bias_reflects_SSE = (strand_ratio > transform_threshold); // more extreme than we like
    bool neg_strand_bias_reflects_SSE = (strand_ratio < -transform_threshold); // more extreme
//    // note: this breaks down at low allele counts
//    float positive_ratio = (alt_counts_positive+safety_val) / (alt_counts_positive + alt_counts_negative + safety_val);
//    float negative_ratio = (alt_counts_negative+safety_val) / (alt_counts_positive + alt_counts_negative + safety_val);
//    bool pos_strand_bias_reflects_SSE = (negative_ratio < minRatioReadsOnNonErrorStrand);
//    bool neg_strand_bias_reflects_SSE = (positive_ratio < minRatioReadsOnNonErrorStrand);
    if (sse_prob_positive_strand >= sseProbThreshold &&  pos_strand_bias_reflects_SSE) {
      status.isNoCallVariant = true;
      filterReason += ",NOCALLxPositiveSSE";
    }

    if (sse_prob_negative_strand >= sseProbThreshold && neg_strand_bias_reflects_SSE) {
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
  DetectSSEForNoCall(filter_variant.sseProbThreshold, filter_variant.minRatioReadsOnNonErrorStrand, filter_variant.sse_relative_safety_level, info, _altAlleIndex);
}

// ====================================================================

void MultiAlleleVariantIdentity::GetMultiAlleleVariantWindow(const string & local_contig_sequence, int DEBUG) {

  window_start = -1;
  window_end   = -1;
  // TODO: Should we exclude already filtered alleles?
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    //if (!allele_identity_vector[i_allele].status.isNoCallVariant) {
    if (allele_identity_vector[i_allele].start_window < window_start or window_start == -1)
      window_start = allele_identity_vector[i_allele].start_window;
    if (allele_identity_vector[i_allele].end_window > window_end or window_end == -1)
      window_end = allele_identity_vector[i_allele].end_window;
  }
  // Hack: pass allele windows back down the object
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    allele_identity_vector[i_allele].start_window = window_start;
    allele_identity_vector[i_allele].end_window = window_end;
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
  GetMultiAlleleVariantWindow(local_contig_sequence, global_context.DEBUG);
  if (global_context.DEBUG > 0) {
    cout << "Final window for multi-allele: " << ": (" << window_start << ") ";
	for (int p_idx = window_start; p_idx < window_end; p_idx++)
	  cout << local_contig_sequence.at(p_idx);
	cout << " (" << window_end << ") " << endl;
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
