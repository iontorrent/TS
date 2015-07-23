/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ClassifyVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "ClassifyVariant.h"
#include "ErrorMotifs.h"
#include "StackEngine.h"

// This function only works for the 1Base -> 1 Base snp representation
void AlleleIdentity::SubCategorizeSNP(const LocalReferenceContext &reference_context) {

  // This classification only works if allele lengths == 1
  // Flag for realignment if alt Base matches either of the flanking Ref bases
  // Variant may actually be part of an undercalled or overcalled HP

  char altBase = altAllele[0];
  ref_hp_length = reference_context.my_hp_length[0];
  // Construct legacy variables from new structure of LocalReferenceContext
  
  // SNP with bases different on both sides
  //  genome       44 45 46 47 48  49 (0 based)
  //  ref is        A  A  A  A  T  G
  //  alt is                    A (position 48, left_anchor == 0 for SNPs)
  //  position0                48
  //  my_hp_start_pos         {48}
  //         matches position0 so refBaseLeft = ref_left_hp_base='A'
  //  my_hp_length            { 1}
  //         my_hp_start_pos[0] + my_hp_length[0] - 1 = 48 + 1 -1 = 48
  //         matches position0 so refBaseRight = ref_right_hp_base='G'
  //  left_hp_length    = 4
  //  ref_left_hp_base   =A
  //  left_hp_start     =44
  //   altBase A matches refBaseLeft A so realign

  // SNP with bases the same on the left
  //  ref is        C  C  A  A  A  G  G
  //  alt is                    T
  //  position0                48
  //  my_hp_start_pos         {46}
  //         doesn't match position0 so refBaseLeft = ref base = 'A'
  //  my_hp_length            { 3}
  //         my_hp_start_pos[0] + my_hp_length[0] - 1 = 46 + 3 -1 = 48
  //         matches position0 so refBaseRight = ref_right_hp_base='G'
  //   altBase T does not match refBaseLeft A or refBaseRight G so do not realign

  // SNP with bases the same on the left
  //  ref is        C  C  C  C  A  A  G
  //  alt is                    T
  //  position0                48
  //  my_hp_start_pos         {48}
  //         matches position0 so refBaseLeft = ref_left_hp_base='C'
  //  my_hp_length            { 2}
  //         my_hp_start_pos[0] + my_hp_length[0] - 1 = 48 + 2 -1 = 49
  //         doesn't match position0 so refBaseRight = ref base = 'A'
  //   altBase T does not match refBaseLeft C or refBaseRight A so do not realign
  char refBaseLeft = (reference_context.position0 == reference_context.my_hp_start_pos[0]) ?
		  reference_context.ref_left_hp_base : reference_context.reference_allele[0];
  char refBaseRight = (reference_context.position0 == reference_context.my_hp_start_pos[0] + reference_context.my_hp_length[0] - 1) ?
		  reference_context.ref_right_hp_base : reference_context.reference_allele[0];

  if (altBase == refBaseLeft || altBase == refBaseRight) {
    // Flag possible misalignment for further investigation
    status.doRealignment = true;
    if (DEBUG > 0)
      cout << "-Possible SNP alignment error detected-";
  }
}

#define invalidBase 'X'
inline bool isValidBase(char base) { return (base != invalidBase); }

inline char getNextHPBase(string const& allele, int *ix, char currentBase, int direction){
  // side effect is to modify pos to be the index of the next HP base
  int newIx = *ix + direction;
 
  while( newIx >= 0 &&  newIx < (int)allele.size() &&
	 allele.at(newIx) == currentBase )
    newIx = newIx + direction;

  if (newIx >= 0 && newIx < (int)allele.size() ){
    *ix = newIx;
    return ( allele.at(newIx) );
  }
  return (invalidBase);
}
  
void AlleleIdentity::SubCategorizeMNP(const LocalReferenceContext &reference_context) {
  // This function only works for the n Base -> n Base mnp representation
  // Possible misalign, align by right shift, example:
  //    Ref:   AAGGGTTTTCCAT
  //    Alt:    GGTTTCCCAT
  // Algorithm:
  //    for HP base in Alt (eg G, T, C)
  //       if (0th base)
  //          find HP base in ref to right
  //       else
  //          find next right adjacent HP base
  //       test if same base
  //   If all match, flag for realignment (possible on right)
  //   repeat for left side
  // 
  // Variant may actually be part of an undercalled or overcalled HP

  if ( ! reference_context.context_detected )
    return;

  assert(altAllele.size()==reference_context.reference_allele.size());
  
  //   possible misalign, align by left shift)
  //   Ref:   GGTTAC   (refBaseLeft = G = reference_context.ref_left_hp_base)
  //   Var:     GTAA
  //   Ref:   GTTTAC   (refBaseLeft = T = reference_context.reference_allele.at(start_index))
  //   Var:     GTAC
  int start_index = left_anchor;                    // start index of real variant
  char altBaseLeft = altAllele.at(start_index);
  int altPos = start_index;

  long ref_start_pos = reference_context.position0 + start_index;  
  char refBaseLeft = (reference_context.my_hp_start_pos.at(start_index) == ref_start_pos)
    ? reference_context.ref_left_hp_base : reference_context.reference_allele.at(start_index);
  int refPos =  (reference_context.my_hp_start_pos.at(start_index) == ref_start_pos)
    ? start_index -1 : start_index;

  bool leftAlign = true;

  char altBase =  altBaseLeft;
  char refBase =  refBaseLeft;
  while ( isValidBase(altBase) && isValidBase(refBase) ) {
    if ( altBase != refBase ){
      leftAlign = false;
      break;
    }
    refBase = getNextHPBase(reference_context.reference_allele, &refPos, refBase, 1);
    altBase = getNextHPBase(altAllele, &altPos, altBase, 1);  
  }
  
  if (leftAlign) {
    status.doRealignment = true;
  }
   
  if (!leftAlign) {
    //   Ref:   AGTT
    //   Var:   GT   (possible misalign, align by right shift)
    int end_index = altAllele.size()-1-right_anchor;  // end index of real variant
    char altBaseRight = altAllele.at(end_index);
    altPos = end_index;

    ref_hp_length = reference_context.my_hp_length.at(start_index);
    long ref_end_pos = reference_context.position0 + end_index;
    long ref_right_hp_end = reference_context.my_hp_start_pos.at(end_index) + reference_context.my_hp_length.at(end_index);

    char refBaseRight = (ref_end_pos == ref_right_hp_end) ?
      reference_context.ref_right_hp_base : reference_context.reference_allele.at(end_index);
    refPos =  (ref_end_pos == ref_right_hp_end) ? end_index+1 : end_index;

    bool rightAlign = true;

    altBase =  altBaseRight;
    refBase =  refBaseRight;
    while ( isValidBase(altBase) && isValidBase(refBase) ) {
      if ( altBase != refBase ){
	rightAlign = false;
	break;
      }
      refBase = getNextHPBase(reference_context.reference_allele, &refPos, refBase, -1);
      altBase = getNextHPBase(altAllele, &altPos, altBase, -1);  
    }

    if(rightAlign) {
      status.doRealignment = true;
    }
  }

  if (DEBUG > 0 and status.doRealignment) {
    cout << "-Possible MNP alignment error detected-";
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
  // It's a deletion, so reference allele must be longer than alternative allele
  int shorter_test_pos = altAllele.length() - 1;
  int longer_test_pos  = reference_context.reference_allele.length() - 1;
  while (shorter_test_pos >= left_anchor and
         altAllele[shorter_test_pos] == reference_context.reference_allele[longer_test_pos]) {
    right_anchor++;
    shorter_test_pos--;
    longer_test_pos--;
  }

  if (left_anchor+right_anchor < (int)altAllele.length()){
    // If the anchors do not add up to the length of the shorter allele,
    // a more complex substitution happened and we don't classify as HP-InDel
    status.isHPIndel = false;
  }
  else {
    status.isHPIndel = reference_context.my_hp_length[left_anchor] > 1;
    for (int i_base=left_anchor+1; (status.isHPIndel and i_base<(int)reference_context.reference_allele.length()-right_anchor); i_base++){
	    status.isHPIndel = status.isHPIndel and (reference_context.my_hp_length[left_anchor] > 1);
    }
  }
  inDelLength = reference_context.reference_allele.length() - altAllele.length();
}

// Test whether this is an HP-InDel
void AlleleIdentity::IdentifyHPinsertion(const LocalReferenceContext& reference_context,
    const ReferenceReader &ref_reader, int chr_idx) {

  char ref_base_right_of_anchor;
  int ref_right_hp_length = 0;
  ref_hp_length = 0;
  status.isHPIndel = false;

  if (left_anchor == (int)reference_context.reference_allele.length()) {
    ref_base_right_of_anchor = reference_context.ref_right_hp_base;
    ref_right_hp_length = reference_context.right_hp_length;
  }
  else {
    ref_base_right_of_anchor = reference_context.reference_allele[left_anchor];
    ref_right_hp_length = reference_context.my_hp_length[left_anchor];
  }

  if (left_anchor > 0  and altAllele[left_anchor] == altAllele[left_anchor - 1]) {
    status.isHPIndel = true;
    ref_hp_length = reference_context.my_hp_length[left_anchor - 1];
  }
  if (altAllele[left_anchor] == ref_base_right_of_anchor) {
    status.isHPIndel = true;
    ref_hp_length = ref_right_hp_length;
  }
  inDelLength  = altAllele.length() - reference_context.reference_allele.length();

  if (status.isHPIndel) {
    for (int b_idx = left_anchor + 1; b_idx < left_anchor + inDelLength; b_idx++) {
      if (altAllele[b_idx] != altAllele[left_anchor])
        status.isHPIndel = false;
    }
  } else if (inDelLength == 1) {
    status.isHPIndel = IdentifyDyslexicMotive(altAllele[left_anchor], reference_context.position0+left_anchor,
        ref_reader, chr_idx);
  }
}

// Identify some special motives
bool AlleleIdentity::IdentifyDyslexicMotive(char base, int position,
    const ReferenceReader &ref_reader, int chr_idx) {

  status.isDyslexic = false;
  long  test_position = position-2;

  unsigned int max_hp_distance = 4;
  unsigned int hp_distance = 0;
  unsigned int my_hp_length = 0;

  // Test left vicinity of insertion
  while (!status.isDyslexic and test_position>0 and hp_distance < max_hp_distance) {
    if (ref_reader.base(chr_idx,test_position) != ref_reader.base(chr_idx,test_position-1)) {
      hp_distance++;
      my_hp_length = 0;
    }
    else if (ref_reader.base(chr_idx,test_position) == base) {
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

  while (!status.isDyslexic and test_position<ref_reader.chr_size(chr_idx) and hp_distance < max_hp_distance) {
    if (ref_reader.base(chr_idx,test_position) != ref_reader.base(chr_idx,test_position-1)) {
      hp_distance++;
      my_hp_length = 0;
    }
    else if (ref_reader.base(chr_idx,test_position) == base) {
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
bool AlleleIdentity::SubCategorizeInDel(const LocalReferenceContext& reference_context,
    const ReferenceReader &ref_reader, int chr_idx) {

  // These fields are set no matter what
  status.isDeletion  = (reference_context.reference_allele.length() > altAllele.length());
  status.isInsertion = (reference_context.reference_allele.length() < altAllele.length());

  if (status.isDeletion) {
	IdentifyHPdeletion(reference_context);
	ref_hp_length = reference_context.my_hp_length[left_anchor];
  }
  else { // Insertion
    IdentifyHPinsertion(reference_context, ref_reader, chr_idx);
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


bool AlleleIdentity::CharacterizeVariantStatus(const LocalReferenceContext &reference_context,
    const ReferenceReader &ref_reader, int chr_idx)
{
  //cout << "Hello from CharacterizeVariantStatus; " << altAllele << endl;
  bool is_ok = true;
  status.isIndel       = false;
  status.isHPIndel     = false;
  status.isSNP         = false;
  status.isMNV         = false;
  status.isPaddedSNP   = false;
  status.doRealignment = false;

  // Get Anchor length
  ref_hp_length = reference_context.my_hp_length[0];
  left_anchor = 0;
  unsigned int a_idx = 0;
  while (a_idx < altAllele.length() and a_idx < reference_context.reference_allele.length()
         and altAllele[a_idx] == reference_context.reference_allele[a_idx]) {
    a_idx++;
    left_anchor++;
  }
  if (DEBUG > 0)
    cout << "- Alternative Allele " << altAllele << " (anchor length " << left_anchor << ") ";


  const string& ref_allele = reference_context.reference_allele;
  const string& alt_allele = altAllele;
  int ref_length = ref_allele.length();
  int alt_length = alt_allele.length();
  while (alt_length > 1 and ref_length > 1 and alt_allele[alt_length-1] == ref_allele[ref_length-1]) {
    --alt_length;
    --ref_length;
  }
  int prefix = 0;
  while (prefix < alt_length and prefix < ref_length and alt_allele[prefix] == ref_allele[prefix])
    ++prefix;
  ref_length -= prefix;
  alt_length -= prefix;

  // Change classification to better reflect what we can get with haplotyping
  if (altAllele.length() != reference_context.reference_allele.length()) {
    status.isIndel = true;
    is_ok = SubCategorizeInDel(reference_context, ref_reader, chr_idx);

  } else if ((int)altAllele.length() == 1) { // Categorize function only works with this setting
    status.isSNP = true;
    SubCategorizeSNP(reference_context);
    if (DEBUG > 0) cout << " is a SNP." << endl;

  } else {
    status.isMNV = true;
    ref_hp_length = reference_context.my_hp_length[left_anchor];
    if (ref_length == 1 and alt_length == 1)
      status.isPaddedSNP = true;
    SubCategorizeMNP(reference_context);
    if (DEBUG > 0)
      cout << " is an MNP." << endl;
  }
  return (is_ok);
}

bool AlleleIdentity::CheckValidAltAllele(const LocalReferenceContext &reference_context) {

  for (unsigned int idx=0; idx<altAllele.length(); idx++) {
    switch (altAllele[idx]) {
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
  const TIonMotifSet & ErrorMotifs,
  const ClassifyFilters &filter_variant,
  const ReferenceReader &ref_reader,
  int chr_idx) {

  altAllele = _altAllele;
  bool is_ok = reference_context.context_detected;

  if ((reference_context.position0 + (long)altAllele.length()) > ref_reader.chr_size(chr_idx)) {
    is_ok = false;
  }

  // We should now be guaranteed a valid variant position in here
  if (is_ok) {
    is_ok = CharacterizeVariantStatus(reference_context, ref_reader, chr_idx);
    PredictSequenceMotifSSE(reference_context, ErrorMotifs, ref_reader, chr_idx);
  }
  is_ok = is_ok and CheckValidAltAllele(reference_context);

  if (!is_ok) {
    status.isProblematicAllele = true;
    filterReasons.push_back("BADCANDIDATE");
  }

  return(is_ok);
}


/*
// Should almost not be called anywhere anymore...
void AlleleIdentity::ModifyStartPosForAllele(int variantPos) {
  if (status.isSNP || status.isMNV)
    modified_start_pos = variantPos - 1; //0 based position for SNP location
  else
    modified_start_pos = variantPos;
}
*/


// Checks the reference area around variantPos for a multi-nucleotide repeat and it's span
// Logic: When shifting a window of the same period as the MNR, the base entering the window has to be equal to the base leaving the window.
// example with period 2: XYZACACA|CA|CACAIJK
bool AlleleIdentity::IdentifyMultiNucRepeatSection(const LocalReferenceContext &seq_context, unsigned int rep_period,
    const ReferenceReader &ref_reader, int chr_idx) {

  //cout << "Hello from IdentifyMultiNucRepeatSection with period " << rep_period << "!"<< endl;
  unsigned int variantPos = seq_context.position0 + left_anchor;
  if (variantPos + rep_period >= (unsigned long)ref_reader.chr_size(chr_idx))
    return (false);

  CircluarBuffer<char> window(rep_period);
  for (unsigned int idx = 0; idx < rep_period; idx++)
    window.assign(idx, ref_reader.base(chr_idx,variantPos+idx));

  // Investigate (inclusive) start position of MNR region
  start_window = variantPos - 1; // 1 anchor base
  window.shiftLeft(1);
  while (start_window > 0 and window.first() == ref_reader.base(chr_idx,start_window)) {
    start_window--;
    window.shiftLeft(1);
  }

  // Investigate (exclusive) end position of MNR region
  end_window = variantPos + rep_period;
  if (end_window >= ref_reader.chr_size(chr_idx))
    return false;
  for (unsigned int idx = 0; idx < rep_period; idx++)
    window.assign(idx, ref_reader.base(chr_idx,variantPos+idx));
  window.shiftRight(1);
  while (end_window < ref_reader.chr_size(chr_idx) and window.last() == ref_reader.base(chr_idx,end_window)) {
    end_window++;
    window.shiftRight(1);
  }

  //cout << "Found repeat stretch of length: " << (end_window - start_window) << endl;
  // Require that a stretch of at least 3*rep_period has to be found to count as a MNR
  if ((end_window - start_window) >= (3*(int)rep_period)) {

    // Correct start and end of the window if they are not fully outside variant allele
    if (start_window >= seq_context.position0)
        start_window = seq_context.my_hp_start_pos[0] - 1;
    if (end_window <= seq_context.right_hp_start) {
      if (status.isInsertion)
        end_window = seq_context.right_hp_start + seq_context.right_hp_length + 1;
      else
        end_window = seq_context.right_hp_start + 1;
    }
    if (start_window < 0)
      start_window = 0;
    if (end_window > ref_reader.chr_size(chr_idx))
      end_window = ref_reader.chr_size(chr_idx);
    return (true);
  }
  else
    return (false);
}


// -----------------------------------------------------------------


void AlleleIdentity::CalculateWindowForVariant(const LocalReferenceContext &seq_context, int DEBUG,
    const ReferenceReader &ref_reader, int chr_idx) {

  // If we have an invalid vcf candidate, set a length zero window and exit
  if (!seq_context.context_detected or status.isProblematicAllele) {
    start_window = seq_context.position0;
    end_window = seq_context.position0;
    return;
  }

  // Check for MNRs first, for InDelLengths 2,3,4,5
  if (status.isIndel and !status.isHPIndel and inDelLength < 5)
    for (int rep_period = 2; rep_period < 6; rep_period++)
      if (IdentifyMultiNucRepeatSection(seq_context, rep_period, ref_reader, chr_idx)) {
        if (DEBUG > 0) {
          cout << "MNR found in allele " << seq_context.reference_allele << " -> " << altAllele << endl;
          cout << "Window for allele " << altAllele << ": (" << start_window << ") ";
          for (int p_idx = start_window; p_idx < end_window; p_idx++)
            cout << ref_reader.base(chr_idx,p_idx);
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
      if (seq_context.my_hp_start_pos[left_anchor] == seq_context.my_hp_start_pos[0])
        start_window = seq_context.my_hp_start_pos[0] - 1;

    if (status.isInsertion) {
      if (left_anchor == 0) {
        start_window = seq_context.my_hp_start_pos[0] - 1;
      }
      else if (altAllele[left_anchor] == altAllele[left_anchor - 1] and
          seq_context.position0 > (seq_context.my_hp_start_pos[left_anchor - 1] - 1)) {
        start_window = seq_context.my_hp_start_pos[left_anchor - 1] - 1;
      }
      if (altAllele[altAllele.length() - 1] == seq_context.ref_right_hp_base) {
        end_window += seq_context.right_hp_length;
      }
    }

    // Safety
    if (start_window < 0)
      start_window = 0;
    if (end_window > ref_reader.chr_size(chr_idx))
      end_window = ref_reader.chr_size(chr_idx);
  }
  else {
    // SNPs and MNVs are 1->1 base replacements
    start_window = seq_context.position0;
    end_window = seq_context.position0 + seq_context.reference_allele.length();
  } // */

  if (DEBUG > 0) {
    cout << "Window for allele " << altAllele << ": (" << start_window << ") ";
    for (int p_idx = start_window; p_idx < end_window; p_idx++)
      cout << ref_reader.base(chr_idx,p_idx);
    cout << " (" << end_window << ") " << endl;
  }
}


// ------------------------------------------------------------------------------
// Filtering functions

void AlleleIdentity::PredictSequenceMotifSSE(const LocalReferenceContext &reference_context,
                             const TIonMotifSet & ErrorMotifs,
                             const ReferenceReader &ref_reader, int chr_idx) {

  //cout << "Hello from PredictSequenceMotifSSE" << endl;
  sse_prob_positive_strand = 0;
  sse_prob_negative_strand = 0;
  //long vcf_position = reference_context.position0+1;
  long var_position = reference_context.position0 + left_anchor; // This points to the first deleted base

  string seqContext;
  // status.isHPIndel && status.isDeletion implies reference_context.my_hp_length.at(left_anchor) > 1
  if (status.isHPIndel && status.isDeletion) {

    // cout << start_pos << "\t" << variant_context.refBaseAtCandidatePosition << variant_context.ref_hp_length << "\t" << variant_context.refBaseLeft << variant_context.left_hp_length << "\t" << variant_context.refBaseRight  << variant_context.right_hp_length << "\t";

    unsigned context_left = var_position >= 10 ? 10 : var_position;
    //if (var_position + reference_context.my_hp_length.at(left_anchor) + 10 < ref_reader.chr_size(chr_idx))
      seqContext = ref_reader.substr(chr_idx, var_position - context_left, context_left + (unsigned int)reference_context.my_hp_length[left_anchor] + 10);
    //  else
    //  seqContext = ref_reader.substr(chr_idx, var_position - context_left);

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
    filterReasons.push_back("HPLEN");
    status.isProblematicAllele = true;
  }
}

void AlleleIdentity::DetectNotAVariant(const LocalReferenceContext &seq_context) {
  if (altAllele.compare(seq_context.reference_allele) == 0) {
    //incorrect allele status is passed thru make it a no call
    status.isProblematicAllele = true;
    filterReasons.push_back("NOTAVARIANT");
  }
}


void AlleleIdentity::DetectCasesToForceNoCall(const LocalReferenceContext &seq_context, const ClassifyFilters &filter_variant,
    const VariantSpecificParams& variant_specific_params)
{
  DetectNotAVariant(seq_context);
  DetectLongHPThresholdCases(seq_context, variant_specific_params.hp_max_length_override ?
      variant_specific_params.hp_max_length : filter_variant.hp_max_length);
}

// ====================================================================


void EnsembleEval::SetupAllAlleles(const ExtendParameters &parameters,
                                                 const InputStructures  &global_context,
                                                 const ReferenceReader &ref_reader,
                                                 int chr_idx)
{
  seq_context.DetectContext(*variant, global_context.DEBUG, ref_reader, chr_idx);
  allele_identity_vector.resize(variant->alt.size());

  if (global_context.DEBUG > 0 and variant->alt.size()>0) {
    cout << "Investigating variant candidate " << seq_context.reference_allele
         << " -> " << variant->alt[0];
    for (uint8_t i_allele = 1; i_allele < allele_identity_vector.size(); i_allele++)
      cout << ',' << variant->alt[i_allele];
    cout << endl;
  }

  //now calculate the allele type (SNP/Indel/MNV/HPIndel etc.) and window for hypothesis calculation for each alt allele.
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {

    // TODO: Hotspot should be an allele property but we only set all or none to Hotspots, depending on the vcf record
    allele_identity_vector[i_allele].status.isHotSpot = variant->isHotSpot;
    allele_identity_vector[i_allele].filterReasons.clear();
    allele_identity_vector[i_allele].DEBUG = global_context.DEBUG;

    allele_identity_vector[i_allele].indelActAsHPIndel = parameters.my_controls.filter_variant.indel_as_hpindel;

    allele_identity_vector[i_allele].getVariantType(variant->alt[i_allele], seq_context,
        global_context.ErrorMotifs,  parameters.my_controls.filter_variant, ref_reader, chr_idx);
    allele_identity_vector[i_allele].CalculateWindowForVariant(seq_context, global_context.DEBUG, ref_reader, chr_idx);
  }

  //GetMultiAlleleVariantWindow();
  multiallele_window_start = -1;
  multiallele_window_end   = -1;


  // Mark Ensemble for realignment if any of the possible variants should be realigned
  // TODO: Should we exclude already filtered alleles?
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    //if (!allele_identity_vector[i_allele].status.isNoCallVariant) {
    if (allele_identity_vector[i_allele].start_window < multiallele_window_start or multiallele_window_start == -1)
      multiallele_window_start = allele_identity_vector[i_allele].start_window;
    if (allele_identity_vector[i_allele].end_window > multiallele_window_end or multiallele_window_end == -1)
      multiallele_window_end = allele_identity_vector[i_allele].end_window;

    if (allele_identity_vector[i_allele].ActAsSNP() && parameters.my_controls.filter_variant.do_snp_realignment) {
      doRealignment = doRealignment or allele_identity_vector[i_allele].status.doRealignment;
    }
    if (allele_identity_vector[i_allele].ActAsMNP() && parameters.my_controls.filter_variant.do_mnp_realignment) {
      doRealignment = doRealignment or allele_identity_vector[i_allele].status.doRealignment;
    }
  }
  // Hack: pass allele windows back down the object
  for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
    allele_identity_vector[i_allele].start_window = multiallele_window_start;
    allele_identity_vector[i_allele].end_window = multiallele_window_end;
  }


  if (global_context.DEBUG > 0) {
	cout << "Realignment for this candidate is turned " << (doRealignment ? "on" : "off") << endl;
    cout << "Final window for multi-allele: " << ": (" << multiallele_window_start << ") ";
    for (int p_idx = multiallele_window_start; p_idx < multiallele_window_end; p_idx++)
      cout << ref_reader.base(chr_idx,p_idx);
    cout << " (" << multiallele_window_end << ") " << endl;
  }
}

// ------------------------------------------------------------

void EnsembleEval::FilterAllAlleles(const ClassifyFilters &filter_variant, const vector<VariantSpecificParams>& variant_specific_params) {
  if (seq_context.context_detected) {
    for (uint8_t i_allele = 0; i_allele < allele_identity_vector.size(); i_allele++) {
      allele_identity_vector[i_allele].DetectCasesToForceNoCall(seq_context, filter_variant, variant_specific_params[i_allele]);
    }
  }
}




