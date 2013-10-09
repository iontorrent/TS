/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ClassifyVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "ClassifyVariant.h"
#include "ErrorMotifs.h"

// This function only works for the 1Base -> 1 Base snp representation
void AlleleIdentity::SubCategorizeSNP(const LocalReferenceContext &reference_context) {

// This classification only works if allele lengths == 1
  char altBase = altAllele.at(0);
  ref_hp_length = reference_context.my_hp_length.at(0);
  // Construct legacy variables from new structure of LocalReferenceContext
  char refBaseLeft = (reference_context.position0 == reference_context.my_hp_start_pos.at(0)) ?
		  reference_context.ref_left_hp_base : reference_context.reference_allele.at(0);
  char refBaseRight = (reference_context.position0 == reference_context.my_hp_start_pos.at(0) + reference_context.my_hp_length.at(0) - 1) ?
		  reference_context.ref_right_hp_base : reference_context.reference_allele.at(0);

  if (altBase == refBaseLeft || altBase == refBaseRight) {
    // Flag possible misalignment for further investigation --- I am an awful hack!
    status.doRealignment = true;
  }
  if (DEBUG > 0) {
    //cout << " is a snp. OverUndercall? " << status.isOverCallUnderCallSNP << endl;
    if (status.doRealignment)
      cout << "Possible alignment error detected." << endl;
  }
}

/*void AlleleIdentity::DetectPotentialCorrelation(const LocalReferenceContext& reference_context){
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
}*/

// Test whether this is an HP-InDel
void AlleleIdentity::IdentifyHPdeletion(const LocalReferenceContext& reference_context) {

  // Get right anchor for better HP-InDel classification
  right_anchor = 0;
  // It's a deleltion, so reference allele must be longer than alternative allele
  int shorter_test_pos = altAllele.length() - 1;
  int longer_test_pos  = reference_context.reference_allele.length() - 1;
  while (shorter_test_pos >= anchor_length and
         altAllele.at(shorter_test_pos) == reference_context.reference_allele.at(longer_test_pos)) {
    right_anchor++;
    shorter_test_pos--;
    longer_test_pos--;
  }

  if (anchor_length+right_anchor < (int)altAllele.length()){
    // If the anchors do not add up to the length of the shorter allele,
    // a more complex substitution happened and we don't classify as HP-InDel
    status.isHPIndel = false;
  }
  else {
    status.isHPIndel = reference_context.my_hp_length.at(anchor_length) > 1;
    for (int i_base=anchor_length+1; (status.isHPIndel and i_base<(int)reference_context.reference_allele.length()-right_anchor); i_base++){
	    status.isHPIndel = status.isHPIndel and (reference_context.my_hp_length.at(anchor_length) > 1);
    }
  }
  inDelLength = reference_context.reference_allele.length() - altAllele.length();
}

// Test whether this is an HP-InDel
void AlleleIdentity::IdentifyHPinsertion(const LocalReferenceContext& reference_context, const string & local_contig_sequence) {

  char ref_base_right_of_anchor;
  int ref_right_hp_length = 0;
  ref_hp_length = 0;
  status.isHPIndel = false;

  if (anchor_length == (int)reference_context.reference_allele.length()) {
    ref_base_right_of_anchor = reference_context.ref_right_hp_base;
    ref_right_hp_length = reference_context.right_hp_length;
  }
  else {
    ref_base_right_of_anchor = reference_context.reference_allele.at(anchor_length);
    ref_right_hp_length = reference_context.my_hp_length.at(anchor_length);
  }

  if (anchor_length > 0  and altAllele.at(anchor_length) == altAllele.at(anchor_length - 1)) {
    status.isHPIndel = true;
    ref_hp_length = reference_context.my_hp_length.at(anchor_length - 1);
  }
  if (altAllele.at(anchor_length) == ref_base_right_of_anchor) {
    status.isHPIndel = true;
    ref_hp_length = ref_right_hp_length;
  }
  inDelLength  = altAllele.length() - reference_context.reference_allele.length();

  if (status.isHPIndel) {
    for (int b_idx = anchor_length + 1; b_idx < anchor_length + inDelLength; b_idx++) {
      if (altAllele.at(b_idx) != altAllele.at(anchor_length))
        status.isHPIndel = false;
    }
  } else if (inDelLength == 1) {
    status.isHPIndel = IdentifyDyslexicMotive(local_contig_sequence, altAllele.at(anchor_length), reference_context.position0+anchor_length);
  }
}

// Identify some special motives
bool AlleleIdentity::IdentifyDyslexicMotive(const string & local_contig_sequence, char base, int position) {

  status.isDyslexic = false;
  int  test_position = position-2;

  unsigned int max_hp_distance = 4;
  unsigned int hp_distance = 0;
  unsigned int my_hp_length = 0;

  // Test left vicinity of insertion
  while (!status.isDyslexic and test_position>0 and hp_distance < max_hp_distance) {
    if (local_contig_sequence.at(test_position) != local_contig_sequence.at(test_position-1)) {
      hp_distance++;
      my_hp_length = 0;
    }
    else if (local_contig_sequence.at(test_position) == base) {
      my_hp_length++;
      if(my_hp_length >= 2) {  // trigger when a 3mer or more is found
    	  status.isDyslexic = true;
      }
    }
    test_position--;
  }
  if (status.isDyslexic) return (true);

  // test right vicinity of insertion
  hp_distance = 0;
  my_hp_length = 0;
  test_position = position+1;

  while (!status.isDyslexic and test_position<(int)local_contig_sequence.length() and hp_distance < max_hp_distance) {
    if (local_contig_sequence.at(test_position) != local_contig_sequence.at(test_position-1)) {
      hp_distance++;
      my_hp_length = 0;
    }
    else if (local_contig_sequence.at(test_position) == base) {
      my_hp_length++;
      if(my_hp_length >= 2) {  // trigger when a 3mer or more is found
    	  status.isDyslexic = true;
      }
    }
    test_position++;
  }
  return status.isDyslexic;
}


// We categorize InDels
bool AlleleIdentity::SubCategorizeInDel(const string &local_contig_sequence, const LocalReferenceContext& reference_context) {

  // These fields are set no matter what
  status.isDeletion  = (reference_context.reference_allele.length() > altAllele.length());
  status.isInsertion = (reference_context.reference_allele.length() < altAllele.length());

  if (status.isDeletion) {
	IdentifyHPdeletion(reference_context);
	ref_hp_length = reference_context.my_hp_length.at(anchor_length);
  }
  else { // Insertion
    IdentifyHPinsertion(reference_context, local_contig_sequence);
  }

  if (DEBUG > 0){
    cout << " is an InDel";
    if (status.isInsertion) cout << ", an Insertion of length " << inDelLength;
    if (status.isDeletion)  cout << ", a Deletion of length " << inDelLength;
    if (status.isHPIndel)   cout << ", and an HP-Indel";
    if (status.isDyslexic)  cout << ", and dyslexic";
    cout << "." << endl;
  }
  return (true);
}


bool AlleleIdentity::CharacterizeVariantStatus(const string & local_contig_sequence, const LocalReferenceContext &reference_context) {
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
    is_ok = SubCategorizeInDel(local_contig_sequence, reference_context);
  }
  else
    if ((int)altAllele.length() == 1) { // Categorize function only works with this setting
      status.isSNP = true;
      SubCategorizeSNP(reference_context);
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

bool AlleleIdentity::CheckValidAltAllele(const LocalReferenceContext &reference_context) {

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
  const string _altAllele,
  const LocalReferenceContext &reference_context,
  const string & local_contig_sequence,
  TIonMotifSet & ErrorMotifs,
  const ClassifyFilters &filter_variant) {

  altAllele = _altAllele;
  bool is_ok = reference_context.context_detected;

  if ((reference_context.position0 + altAllele.length()) > local_contig_sequence.length()) {
    is_ok = false;
  }

  // We should now be guaranteed a valid variant position in here
  if (is_ok) {
    is_ok = CharacterizeVariantStatus(local_contig_sequence, reference_context);
    PredictSequenceMotifSSE(reference_context, local_contig_sequence, ErrorMotifs);
  }
  is_ok = is_ok and CheckValidAltAllele(reference_context);

  if (!is_ok) {
    status.isProblematicAllele = true;
    filterReason += ",BADCANDIDATE";
  }

  return(is_ok);
}


/*/ Should almost not be called anywhere anymore...
void AlleleIdentity::ModifyStartPosForAllele(int variantPos) {
  if (status.isSNP || status.isMNV)
    modified_start_pos = variantPos - 1; //0 based position for SNP location
  else
    modified_start_pos = variantPos;
} //*/


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


void AlleleIdentity::CalculateWindowForVariant(const LocalReferenceContext &seq_context, const string &local_contig_sequence, int DEBUG) {

  // If we have an invalid vcf candidate, set a length zero window and exit
  if (!seq_context.context_detected or status.isProblematicAllele) {
    start_window = seq_context.position0;
    end_window = seq_context.position0;
    return;
  }

  // Check for MNRs first, for InDelLengths 2,3,4,5
  if (status.isIndel and !status.isHPIndel and inDelLength < 5)
    for (int rep_period = 2; rep_period < 6; rep_period++)
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
      if (anchor_length == 0) {
        start_window = seq_context.my_hp_start_pos.at(0) - 1;
      }
      else if (altAllele.at(anchor_length) == altAllele.at(anchor_length - 1) and
          seq_context.position0 > (seq_context.my_hp_start_pos.at(anchor_length - 1) - 1)) {
        start_window = seq_context.my_hp_start_pos.at(anchor_length - 1) - 1;
      }
      if (altAllele.at(altAllele.length() - 1) == seq_context.ref_right_hp_base) {
        end_window += seq_context.right_hp_length;
      }
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

void AlleleIdentity::PredictSequenceMotifSSE(const LocalReferenceContext &reference_context,
                             const  string &local_contig_sequence, TIonMotifSet & ErrorMotifs) {

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

       //cout << seqContext << "\t" << context_left << "\t" << sse_prob_positive_strand << "\t";

      context_left = seqContext.length() - context_left - 1;
      string reverse_seqContext;
      ReverseComplement(seqContext, reverse_seqContext);

      sse_prob_negative_strand = ErrorMotifs.get_sse_probability(reverse_seqContext, context_left);

     // cout << reverse_seqContext << "\t" << context_left << "\t" << sse_prob_negative_strand << "\t";

    }
  }
}


void AlleleIdentity::DetectLongHPThresholdCases(const LocalReferenceContext &seq_context, int maxHPLength) {
  if (status.isIndel && ref_hp_length > maxHPLength) {
    filterReason += "HPLEN";
    status.isProblematicAllele = true;
  }
}


void AlleleIdentity::DetectNotAVariant(const LocalReferenceContext &seq_context) {
  if (altAllele.compare(seq_context.reference_allele) == 0) {
    //incorrect allele status is passed thru make it a no call
    status.isProblematicAllele = true;
    filterReason += "NOTAVARIANT";
  }
}


void AlleleIdentity::DetectCasesToForceNoCall(const LocalReferenceContext &seq_context, const ClassifyFilters &filter_variant) {

  //filterReason = ""; moved up, Classifier might already throw a NoCall for a bad candidate
  DetectNotAVariant(seq_context);
  DetectLongHPThresholdCases(seq_context, filter_variant.hp_max_length);
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
    doRealignment = doRealignment or allele_identity_vector[i_allele].status.doRealignment;
  }
  // Hack: pass allele windows back down the object
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    allele_identity_vector[i_allele].start_window = window_start;
    allele_identity_vector[i_allele].end_window = window_end;
  }
}

// ------------------------------------------------------------

void MultiAlleleVariantIdentity::SetupAllAlleles(vcf::Variant ** candidate_variant, const string & local_contig_sequence,
                                                 ExtendParameters *parameters, InputStructures &global_context) {

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

void MultiAlleleVariantIdentity::FilterAllAlleles(const ClassifyFilters &filter_variant) {
  if (seq_context.context_detected) {
    for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
      allele_identity_vector[i_allele].DetectCasesToForceNoCall(seq_context, filter_variant);
    }
  }
}
