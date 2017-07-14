/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ClassifyVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "ClassifyVariant.h"
#include "ErrorMotifs.h"
#include "CrossHypotheses.h"

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

  if (left_anchor+right_anchor != (int) altAllele.length()){
    // If the anchors do not add up to the length of the shorter allele,
    // a more complex substitution happened and we don't classify as HP-InDel
    status.isHPIndel = false;
  }
  else {
    //status.isHPIndel = (left_anchor+right_anchor) == (int) altAllele.length();
    //for (int i_base=left_anchor+1; (status.isHPIndel and i_base<(int)reference_context.reference_allele.length()-right_anchor); i_base++){
	//    status.isHPIndel = status.isHPIndel and (reference_context.my_hp_length[left_anchor] > 1);
    //}
	  string padding_ref = string(1, reference_context.ref_left_hp_base) + reference_context.reference_allele + string(1, reference_context.ref_right_hp_base);
	  string padding_alt = string(1, reference_context.ref_left_hp_base) + altAllele + string(1, reference_context.ref_right_hp_base);
	  status.isHPIndel = IsHpIndel(padding_ref, padding_alt);
  }
  inDelLength = abs((int) reference_context.reference_allele.length() - (int) altAllele.length());
}

// Test whether this is an HP-InDel
void AlleleIdentity::IdentifyHPinsertion(const LocalReferenceContext& reference_context,
    const ReferenceReader &ref_reader) {

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
  inDelLength  = abs((int) altAllele.length() - (int) reference_context.reference_allele.length());

  if (status.isHPIndel) {
    for (int b_idx = left_anchor + 1; b_idx < left_anchor + inDelLength; b_idx++) {
      if (altAllele[b_idx] != altAllele[left_anchor])
        status.isHPIndel = false;
    }
  } else if (inDelLength == 1) {
    status.isHPIndel = IdentifyDyslexicMotive(altAllele[left_anchor], reference_context.position0+left_anchor,
        ref_reader, reference_context.chr_idx);
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
    const ReferenceReader &ref_reader) {

  // These fields are set no matter what
  status.isDeletion  = (reference_context.reference_allele.length() > altAllele.length());
  status.isInsertion = (reference_context.reference_allele.length() < altAllele.length());

  if (status.isDeletion) {
	IdentifyHPdeletion(reference_context);
	ref_hp_length = reference_context.my_hp_length[left_anchor];
  }
  else { // Insertion
    IdentifyHPinsertion(reference_context, ref_reader);
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
    const ReferenceReader &ref_reader)
{
  //cout << "Hello from CharacterizeVariantStatus; " << altAllele << endl;
  bool is_ok = true;
  status.isIndel       = false;
  status.isHPIndel     = false;
  status.isSNP         = false;
  status.isMNV         = false;
  status.isPaddedSNP   = false;
  status.doRealignment = false;

  if (altAllele == reference_context.reference_allele){
	  status.isNoVariant = true;
	  status.isProblematicAllele = true;
	  status.isSNP = (altAllele.size() == 1);
	  status.isMNV = not (status.isSNP);
	  is_ok = false;
	  // I don't want to continue since it is not a variant.
	  return is_ok;
  }

  // Get left anchor length
  ref_hp_length = reference_context.my_hp_length[0];
  left_anchor = 0;
  while (left_anchor < (int) altAllele.length() and left_anchor < ref_length
         and altAllele[left_anchor] == reference_context.reference_allele[left_anchor]) {
    ++left_anchor;
  }
  // Get right anchor length
  // right anchor is obtained after I remove left anchor, which implies I prefer left alignment.
  right_anchor = 0;
  // It's a deletion, so reference allele must be longer than alternative allele
  int alt_test_pos = (int) altAllele.length() - 1;
  int ref_test_pos  = ref_length - 1;
  while (alt_test_pos >= left_anchor
		  and ref_test_pos >= left_anchor
		  and altAllele[alt_test_pos] == reference_context.reference_allele[ref_test_pos]) {
    right_anchor++;
    alt_test_pos--;
    ref_test_pos--;
  }
  // Calculate the variant window (in reference coordinate)
  start_variant_window = position0 + num_padding_added.first;
  end_variant_window = position0 + ref_length - num_padding_added.second;
  if (num_padding_added.first > 0 or num_padding_added.second > 0){
	  assert(reference_context.reference_allele.substr(0, num_padding_added.first) == altAllele.substr(0,  num_padding_added.first));
	  assert(reference_context.reference_allele.substr((int) reference_context.reference_allele.size() -  num_padding_added.second) == altAllele.substr((int) altAllele.size() - num_padding_added.second));
	  assert(start_variant_window < end_variant_window);
  }

  if (DEBUG > 0)
    cout << "- Alternative Allele " << altAllele << " (left anchor length = " << left_anchor << " , right anchor length = " << right_anchor << ")";

  // ref_length is the length of the anchor-removed reference allele
  int ref_length = (int) reference_context.reference_allele.length() - (left_anchor + right_anchor);
  // alt_length is the length of the anchor-removed alt allele
  int alt_length = (int) altAllele.length() - (left_anchor + right_anchor);
  assert(ref_length >= 0 and alt_length >= 0);

  // Change classification to better reflect what we can get with haplotyping
  if (altAllele.length() != reference_context.reference_allele.length()) {
    status.isIndel = true;
    is_ok = SubCategorizeInDel(reference_context, ref_reader);

  } else if ((int)altAllele.length() == 1) { // Categorize function only works with this setting
    status.isSNP = true;
    SubCategorizeSNP(reference_context);
    if (DEBUG > 0) cout << " is a SNP." << endl;

  } else {
    status.isMNV = true;
    ref_hp_length = reference_context.my_hp_length[left_anchor];
    status.isPaddedSNP = (ref_length == 1 and alt_length == 1);
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
  const pair<int, int> &alt_orig_padding) {

  altAllele = _altAllele;
  position0 = (int) reference_context.position0;
  ref_length = (int) reference_context.reference_allele.length();
  chr_idx = reference_context.chr_idx;
  num_padding_added = alt_orig_padding;

  bool is_ok = reference_context.context_detected;
  // check position does not beyond the chromosome
  is_ok *=  not ((reference_context.position0 + (long) altAllele.length()) > ref_reader.chr_size(reference_context.chr_idx));

  // check alternative allele contains TACG only
  is_ok *= CheckValidAltAllele(reference_context);

  // We should now be guaranteed a valid variant position in here
  if (is_ok) {
    is_ok = CharacterizeVariantStatus(reference_context, ref_reader);
  }

  if (is_ok) {
    PredictSequenceMotifSSE(reference_context, ErrorMotifs, ref_reader);
  }else{
    status.isProblematicAllele = true;
    filterReasons.push_back("BADCANDIDATE");
  }

  return is_ok;
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
    const ReferenceReader &ref_reader) {

  //cout << "Hello from IdentifyMultiNucRepeatSection with period " << rep_period << "!"<< endl;
  unsigned int variantPos = seq_context.position0 + left_anchor;
  if (variantPos + rep_period >= (unsigned long)ref_reader.chr_size(seq_context.chr_idx))
    return false;

  CircluarBuffer<char> window(0);
  start_splicing_window = seq_context.FindSplicingStartForMNR(ref_reader, variantPos, rep_period, window);

  // Investigate (exclusive) end position of MNR region
  end_splicing_window = variantPos + rep_period;
  if (end_splicing_window >= ref_reader.chr_size(seq_context.chr_idx))
    return false;
  for (unsigned int idx = 0; idx < rep_period; idx++)
    window.assign(idx, ref_reader.base(seq_context.chr_idx,variantPos+idx));
  window.shiftRight(1);
  while (end_splicing_window < ref_reader.chr_size(seq_context.chr_idx) and window.last() == ref_reader.base(seq_context.chr_idx,end_splicing_window)) {
    end_splicing_window++;
    window.shiftRight(1);
  }

  //cout << "Found repeat stretch of length: " << (end_window - start_window) << endl;
  // Require that a stretch of at least 3*rep_period has to be found to count as a MNR
  if ((end_splicing_window - start_splicing_window) >= (3*(int)rep_period)) {

    // Correct start and end of the window if they are not fully outside variant allele
    if (start_splicing_window >= seq_context.position0)
        start_splicing_window = seq_context.StartSplicingExpandFromMyHpStart0();
    if (end_splicing_window <= seq_context.right_hp_start) {
      if (status.isInsertion)
        end_splicing_window = seq_context.right_hp_start + seq_context.right_hp_length + 1;
      else
        end_splicing_window = seq_context.right_hp_start + 1;
    }
    if (start_splicing_window < 0)
      start_splicing_window = 0;
    if (end_splicing_window > ref_reader.chr_size(seq_context.chr_idx))
      end_splicing_window = ref_reader.chr_size(seq_context.chr_idx);
    return (true);
  }
  else
    return (false);
}


// -----------------------------------------------------------------


void AlleleIdentity::CalculateWindowForVariant(const LocalReferenceContext &seq_context,
    const ReferenceReader &ref_reader) {

  // If we have an invalid vcf candidate, set a length zero window and exit
  if (!seq_context.context_detected or status.isProblematicAllele) {
	  cout <<" I am probematic!" <<endl;
    start_splicing_window = seq_context.StartSplicingNoExpansion();
    end_splicing_window = seq_context.position0;
    return;
  }

  // Check for MNRs first, for InDelLengths 2,3,4,5
  if (status.isIndel and !status.isHPIndel and inDelLength < 5)
    for (int rep_period = seq_context.min_mnr_rep_period; rep_period <= seq_context.max_mnr_rep_period; rep_period++)
      if (IdentifyMultiNucRepeatSection(seq_context, rep_period, ref_reader)) {
        if (DEBUG > 0) {
          cout << "MNR found in allele " << seq_context.reference_allele << " -> " << altAllele << endl;
          cout << "Window for allele " << altAllele << ": (" << start_splicing_window << ") ";
          for (int p_idx = start_splicing_window; p_idx < end_splicing_window; p_idx++)
            cout << ref_reader.base(seq_context.chr_idx,p_idx);
          cout << " (" << end_splicing_window << ") " << endl;
        }
        return; // Found a matching period and computed window
      }

  // not an MNR. Moving on along to InDels.
  if (status.isIndel) {
	// Default variant window
    end_splicing_window = seq_context.right_hp_start +1; // Anchor base to the right of allele
    start_splicing_window = seq_context.StartSplicingNoExpansion();

    // Adjustments if necessary
    if (status.isDeletion)
      if (seq_context.my_hp_start_pos[left_anchor] == seq_context.my_hp_start_pos[0])
        start_splicing_window = seq_context.StartSplicingExpandFromMyHpStart0();

    if (status.isInsertion) {
      if (left_anchor == 0) {
        start_splicing_window = seq_context.StartSplicingExpandFromMyHpStart0();
      }
      else if (altAllele[left_anchor] == altAllele[left_anchor - 1] and
          seq_context.position0 > (seq_context.my_hp_start_pos[left_anchor - 1] - 1)) {
        start_splicing_window = seq_context.StartSplicingExpandFromMyHpStartLeftAnchor(left_anchor);
      }
      if (altAllele[altAllele.length() - 1] == seq_context.ref_right_hp_base) {
        end_splicing_window += seq_context.right_hp_length;
      }
    }

    // Safety
    if (start_splicing_window < 0)
      start_splicing_window = 0;
    if (end_splicing_window > ref_reader.chr_size(seq_context.chr_idx))
      end_splicing_window = ref_reader.chr_size(seq_context.chr_idx);
  }
  else {
    // SNPs and MNVs are 1->1 base replacements
    start_splicing_window = seq_context.StartSplicingNoExpansion();
    end_splicing_window = seq_context.position0 + seq_context.reference_allele.length();
  }

  // Final safety: splicing window is a super set of the interval spanned by the reference allele.
  assert(start_splicing_window <= (int) seq_context.position0 and end_splicing_window >= (int) seq_context.position0 + (int) seq_context.reference_allele.length());

  if (DEBUG > 0) {
    cout << "Window for allele " << altAllele << ": (" << start_splicing_window << ") ";
    for (int p_idx = start_splicing_window; p_idx < end_splicing_window; p_idx++)
      cout << ref_reader.base(seq_context.chr_idx,p_idx);
    cout << " (" << end_splicing_window << ") " << endl;
  }
}


// ------------------------------------------------------------------------------
// Filtering functions

void AlleleIdentity::PredictSequenceMotifSSE(const LocalReferenceContext &reference_context,
                             const TIonMotifSet & ErrorMotifs,
                             const ReferenceReader &ref_reader) {

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
      seqContext = ref_reader.substr(reference_context.chr_idx, var_position - context_left, context_left + (unsigned int)reference_context.my_hp_length[left_anchor] + 10);
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

void AlleleIdentity::DetectHpIndelCases(const vector<int> &hp_indel_hrun, const vector<int> &hp_ins_len, const vector<int> &hp_del_len) {
	if (status.isIndel && status.isHPIndel and inDelLength > 0) {
		assert((hp_indel_hrun.size() == hp_ins_len.size()) and (hp_indel_hrun.size() == hp_del_len.size()));
		for (unsigned int i_hp = 0; i_hp < hp_indel_hrun.size(); ++i_hp) {
			if (ref_hp_length == hp_indel_hrun[i_hp]){
				if (status.isInsertion and inDelLength <= hp_ins_len[i_hp]){
				    filterReasons.push_back("HPINSLEN");
				    status.isProblematicAllele = true;
				}
				else if (status.isDeletion and inDelLength <= hp_del_len[i_hp]){
				    filterReasons.push_back("HPDELLEN");
				    status.isProblematicAllele = true;
				}
			}
		}
	}
}


void AlleleIdentity::DetectNotAVariant(const LocalReferenceContext &seq_context) {
  if (status.isNoVariant) {
    //incorrect allele status is passed thru make it a no call
    status.isProblematicAllele = true;
    filterReasons.push_back("NOTAVARIANT");
  }
}

void AlleleIdentity::DetectCasesToForceNoCall(const LocalReferenceContext &seq_context, const ControlCallAndFilters& my_controls,
    const VariantSpecificParams& variant_specific_params)
{
  DetectNotAVariant(seq_context);
  DetectLongHPThresholdCases(seq_context, variant_specific_params.hp_max_length_override ?
			  variant_specific_params.hp_max_length : my_controls.filter_variant.hp_max_length);
  DetectHpIndelCases(my_controls.filter_variant.filter_hp_indel_hrun, my_controls.filter_variant.filter_hp_ins_len, my_controls.filter_variant.filter_hp_del_len);
}


// ====================================================================




// win_1 = [win1_start, win1_end), win_2 = [win2_start, win2_end),
// return (win_1 \cap win_2 != \emptyset) where \cap is the set intersection operator.
template <typename MyIndexType>
bool IsOverlappingWindows(MyIndexType win1_start, MyIndexType win1_end, MyIndexType win2_start, MyIndexType win2_end)
{
	if (win1_start >= win1_end or win2_start >= win2_end){
		// win_1 or win_2 is an empty set => empty set intersects any set is always empty
		return false;
	}
	return (win1_start < win2_end) and (win1_end > win2_start);
}

// Is there splicing hazard of this allele interfered by alt_x?
// Splicing hazard happened if the (my splicing window) \cap (my variant window)^c overlaps (alt_x's variant window)
// where \cap means intersection, ^c means complement.
bool AlleleIdentity::DetectSplicingHazard(const AlleleIdentity& alt_x) const{
	bool is_splicing_hazard = IsOverlappingWindows(start_splicing_window, start_variant_window, alt_x.start_variant_window, alt_x.end_variant_window)
				or  IsOverlappingWindows(end_variant_window, end_splicing_window, alt_x.start_variant_window, alt_x.end_variant_window);
	return is_splicing_hazard;
}

// The two alleles are connected (i.e., need to be evaluated together) if any of the following conditions is satisfied
// a) (variant window of alt1) intersects (variant window of alt2)
// b) There is splicing hazard of alt1 interfered by alt2.
// c) There is splicing hazard of alt2 interfered by alt1.
// Note: Fake HS allele means the HS allele has no read support. It won't interfere other alleles. But I need to make sure other alleles don't interfere it.
bool IsAllelePairConnected(const AlleleIdentity& alt1, const AlleleIdentity& alt2)
{
	bool debug = alt1.DEBUG or alt2.DEBUG;
    // Alleles start at the same position should be evaluated together.
	bool is_connect = alt1.start_variant_window == alt2.start_variant_window;

	// Print the allele information for debug
	if (debug){
		cout << "+ Detecting connectivity of the allele pair (altX, altY) = ("
			 << alt1.altAllele << "@[" << alt1.position0 << ", " <<  alt1.position0 + alt1.ref_length << "), "
			 << alt2.altAllele << "@[" << alt2.position0 << ", " <<  alt2.position0 + alt2.ref_length << "))" << endl;
		cout << "  - (altX, altY) is Fake HS Allele? (" << alt1.status.isFakeHsAllele << ", " << alt2.status.isFakeHsAllele <<")" << endl;
	}

	// Rule number one: start at the same position must be connected.
	if (is_connect){
		if (debug){
			cout << "  - Connected: altX and altY start at the same position." << endl;
		}
		return is_connect;
	}

	// Exception for problematic alleles
	if (alt1.status.isProblematicAllele or alt2.status.isProblematicAllele){
		if (debug){
			cout << (is_connect? "  - Connected: altX or altY is problematic at the same position." : "  - Not connected: altX or altY is problematic.") << endl;
		}
		return is_connect;
	}

	// Exceptions for Fake HS alleles
	if (alt1.status.isFakeHsAllele and alt2.status.isFakeHsAllele){
		if (debug){
			cout << (is_connect? "  - Connected: both fake HS at the same position." : "  - Not connected: both fake HS." ) << endl;
		}
		return is_connect;
	}else if (alt1.status.isFakeHsAllele and alt1.ref_length >= 10 and (not alt1.status.doRealignment)
				and alt2.status.isHPIndel and alt2.inDelLength == 1){
		if (debug){
			cout << (is_connect? "  - Connected: altX and altY start at the same position." : "  - Not connected: long Fake HS altX meets 1-mer HP-INDEL altY." ) << endl;
		}
		return is_connect;
	}else if (alt2.status.isFakeHsAllele and alt2.ref_length >= 10 and (not alt2.status.doRealignment)
				and alt1.status.isHPIndel and alt1.inDelLength == 1){
		if (debug){
			cout << (is_connect? "  - Connected: altX and altY start at the same position." : "  - Not connected: 1-mer HP-INDEL altX meets long Fake HS altY." ) << endl;
		}
		return is_connect;
	}

	// Condition a)
	bool is_variant_window_overlap = IsOverlappingWindows(alt1.start_variant_window, alt1.end_variant_window, alt2.start_variant_window, alt2.end_variant_window);
	// Condition b)
	bool is_alt1_interfered_by_alt2 = false;
	if (not alt2.status.isFakeHsAllele){
		is_alt1_interfered_by_alt2 = alt1.DetectSplicingHazard(alt2);
	}
	// Condition c)
	bool is_alt2_interfered_by_alt1 = false;
	if (not alt1.status.isFakeHsAllele){
		is_alt2_interfered_by_alt1 = alt2.DetectSplicingHazard(alt1);
	}
	is_connect = is_variant_window_overlap or is_alt1_interfered_by_alt2 or is_alt2_interfered_by_alt1;

	// Print debug message
	if (debug){
		if (not is_connect){
			cout << "  - Not connected." << endl;
		}
		else{
			if (is_variant_window_overlap){
				cout << "  - Connected: Overlapping variant windows: "
					 <<	"var_win_X = [" << alt1.start_variant_window << ", "<< alt1.end_variant_window << "), "
					 << "var_win_Y = [" << alt2.start_variant_window << ", "<< alt2.end_variant_window << ")" << endl;
			}
			if (is_alt1_interfered_by_alt2){
				cout << "  - Connected: Splicing hazard of altX interfered by altY: "
					 << "splice_win_X = [" << alt1.start_splicing_window << ", "<< alt1.end_splicing_window << "), "
					 << "var_win_X = [" << alt1.start_variant_window << ", "<< alt1.end_variant_window << "), "
					 << "var_win_Y = [" << alt2.start_variant_window << ", "<< alt2.end_variant_window << ")" << endl;
			}
			if (is_alt2_interfered_by_alt1){
				cout << "  - Connected: Splicing hazard of altY interfered by altX: "
					 <<	"splice_win_Y = [" << alt2.start_splicing_window << ", "<< alt2.end_splicing_window << "), "
					 << "var_win_Y = [" << alt2.start_variant_window << ", "<< alt2.end_variant_window << "), "
					 << "var_win_X = [" << alt1.start_variant_window << ", "<< alt1.end_variant_window << ")" << endl;
			}
		}
	}
	return is_connect;
}

