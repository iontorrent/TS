/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     LocalContext.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "LocalContext.h"

void LocalReferenceContext::DetectContextAtPosition(const ReferenceReader &ref_reader, int my_chr_idx, int position0, int ref_len){
	assert(ref_len > 0);
	chr_idx = my_chr_idx;
	vcf::Variant dummy_variant;
	dummy_variant.sequenceName = ref_reader.chr_str(chr_idx);
	dummy_variant.position = (long) (position0 + 1);
	dummy_variant.ref = ref_reader.substr(chr_idx, (long) position0, (long) ref_len);
	DetectContext(dummy_variant, 0, ref_reader);
}

void LocalReferenceContext::DetectContext(const vcf::Variant &candidate_variant, int DEBUG,
    const ReferenceReader &ref_reader) {

  // VCF stores positions in 1-based index; local_contig_sequence has a zero based index
  // all positions in this object are zero based so that they correspond to reference in memory.
  position0 = candidate_variant.position-1;
  contigName = candidate_variant.sequenceName;
  if (chr_idx < 0)
	  chr_idx = ref_reader.chr_idx(contigName.c_str());
  else if (contigName != ref_reader.chr_str(chr_idx))
	  chr_idx = ref_reader.chr_idx(contigName.c_str());

  // Sanity checks if position is valid and reference allele matches reference
  if (!ContextSanityChecks(candidate_variant, ref_reader))
    return;

  my_hp_length.resize(reference_allele.length(), 0);
  my_hp_start_pos.resize(reference_allele.length(), 0);

  // Process first HP (inclusive, zero based start position)
  my_hp_start_pos[0] = position0;
  my_hp_length[0] = 1;
  while (my_hp_start_pos[0] > 0
         and ref_reader.base(chr_idx,my_hp_start_pos[0]-1) == ref_reader.base(chr_idx,position0)) {
    my_hp_start_pos[0]--;
    my_hp_length[0]++;
  }
  // Now get base and length of the HP to the left of the one containing variant start
  long temp_position = my_hp_start_pos[0] -1;
  ref_left_hp_base = 'X'; //
  left_hp_length = 0;
  left_hp_start = temp_position;
  if (temp_position >= 0) {
    ref_left_hp_base = ref_reader.base(chr_idx,temp_position);
    left_hp_length++;
  }
  while (temp_position > 0 and ref_reader.base(chr_idx,temp_position-1) == ref_left_hp_base) {
    temp_position--;
    left_hp_start--;
    left_hp_length++;
  }

  // Get HP context of the remaining bases in the reference allele and record for each base
  for (unsigned int b_idx = 1; b_idx < reference_allele.length(); b_idx++) {
    // See if next base in reference allele starts a new HP and adjust length
	if (ref_reader.base(chr_idx,position0 + b_idx-1) == ref_reader.base(chr_idx,position0 + b_idx)) {
      my_hp_start_pos[b_idx] = my_hp_start_pos[b_idx-1];
      for (unsigned int l_idx = 0; l_idx < b_idx; l_idx++)
        if (my_hp_start_pos[l_idx] == my_hp_start_pos[b_idx])
          my_hp_length[l_idx]++;
      my_hp_length[b_idx] = my_hp_length[b_idx-1];
    }
    else {
      my_hp_start_pos[b_idx] = position0 + b_idx;
      my_hp_length[b_idx] = 1;
    }
  }

  // Complete the HP length of the last base in the reference allele
  temp_position = position0 + reference_allele.length() -1;
  while (temp_position < ref_reader.chr_size(chr_idx)-1 and
      ref_reader.base(chr_idx,temp_position+1) ==
          ref_reader.base(chr_idx,position0 + reference_allele.length() -1)) {
    temp_position++;
    for (unsigned int b_idx = 0; b_idx < reference_allele.length(); b_idx++) {
      if (my_hp_start_pos[b_idx] == my_hp_start_pos[reference_allele.length()-1])
        my_hp_length[b_idx]++;
    }
  }

  // Get HP to the right of the one containing last base of the reference allele
  ref_right_hp_base = 'X';
  right_hp_length = 0;
  temp_position = my_hp_start_pos[reference_allele.length()-1] + my_hp_length[reference_allele.length()-1];
  right_hp_start = temp_position;
  if (temp_position < ref_reader.chr_size(chr_idx)) {
    ref_right_hp_base = ref_reader.base(chr_idx,temp_position);
    right_hp_length++;
  }
  while (temp_position < ref_reader.chr_size(chr_idx)-1 and ref_reader.base(chr_idx,temp_position+1) == ref_right_hp_base) {
    temp_position++;
    right_hp_length++;
  }

  if (DEBUG>0) {
    cout << "Local Reference context at (zero-based) " << position0 << ", " << reference_allele << " :";
    cout << left_hp_length << ref_left_hp_base << " ";
    for (unsigned int idx = 0; idx < reference_allele.length(); idx++)
      cout << my_hp_length[idx] << reference_allele[idx] << "(" << my_hp_start_pos[idx]  << ") ";
    cout << right_hp_length << ref_right_hp_base << " ";
    // Some additional printout to see if I get index right:
    for (int idx=max(position0-7, (long)0); idx<position0; idx++)
      cout << ref_reader.base(chr_idx,idx);
    cout << "|" << ref_reader.base(chr_idx,position0) << "|";
    for (int idx=position0+1; idx < min(position0+8,ref_reader.chr_size(chr_idx)); idx++)
      cout << ref_reader.base(chr_idx,idx);
    cout << endl;
  }
}

bool LocalReferenceContext::ContextSanityChecks(const vcf::Variant &candidate_variant,
    const ReferenceReader &ref_reader) {

  // Sanity checks that abort context detection
  reference_allele = candidate_variant.ref;
  context_detected = true;

  if (candidate_variant.position < 1 or candidate_variant.position > (long)ref_reader.chr_size(chr_idx)) {
    cerr << "Non-fatal ERROR: Candidate Variant Position is not within the Contig Bounds at VCF Position "
         << candidate_variant.sequenceName << ":" << candidate_variant.position
         << " Contig length = " << ref_reader.chr_size(chr_idx) <<  endl;
    cout << "Non-fatal ERROR: Candidate Variant Position is not within the Contig Bounds at VCF Position "
         << candidate_variant.sequenceName << ":" << candidate_variant.position
         << " Contig length = " << ref_reader.chr_size(chr_idx) << endl;
    // Choose safe parameter
    position0 = 0;
    context_detected = false;
  }

  if (reference_allele.length() == 0) {
    cerr << "Non-fatal ERROR: Reference allele has zero length at vcf position "
         << candidate_variant.sequenceName << ":" << candidate_variant.position << endl;
    cout << "Non-fatal ERROR: Reference allele has zero length at vcf position "
         << candidate_variant.sequenceName << ":" << candidate_variant.position << endl;
    // Choose safe parameter
    reference_allele = ref_reader.base(chr_idx,position0); //local_contig_sequence.at(position0);
    context_detected = false;
  }

  if ((candidate_variant.position + (long)reference_allele.length() -1) > ref_reader.chr_size(chr_idx)) {
    cerr << "Non-fatal ERROR: Reference Allele stretches beyond Contig Bounds at VCF Position "
	     << candidate_variant.sequenceName << ":" << candidate_variant.position
	     << " Contig length = " << ref_reader.chr_size(chr_idx)
	     << " Reference Allele: " << reference_allele << endl;
    cout << "Non-fatal ERROR: Reference Allele stretches beyond Contig Bounds at VCF Position "
	     << candidate_variant.sequenceName << ":" << candidate_variant.position
	     << " Contig length = " << ref_reader.chr_size(chr_idx)
	     << " Reference Allele: " << reference_allele << endl;
    // Choose safe parameter
    reference_allele = reference_allele[0];
    context_detected = false;
  }

  //string contig_str(local_contig_sequence, position0, reference_allele.length());
  string contig_str = ref_reader.substr(chr_idx, position0, reference_allele.length());
  if (reference_allele.compare(contig_str) != 0) {
    cerr << "Non-fatal ERROR: Reference allele does not match reference at VCF position "
         << candidate_variant.sequenceName << ":" << candidate_variant.position
         << " Reference Allele: " << reference_allele << " Reference: " << contig_str << endl;
    cout << "Non-fatal ERROR: Reference allele does not match reference at VCF position "
         << candidate_variant.sequenceName << ":" << candidate_variant.position
         << " Reference Allele: " << reference_allele << " Reference: " << contig_str << endl;
    context_detected = false;
  }

  return (context_detected);
}

// return the minimum possible splicing window start in this context
int LocalReferenceContext::SplicingLeftBound(const ReferenceReader &ref_reader) const{
	if (not context_detected){
		return StartSplicingNoExpansion();
	}
	int splicing_left_bound = StartSplicingNoExpansion(); // splicing window start for SNP/MNP
	splicing_left_bound = min(splicing_left_bound, StartSplicingExpandFromMyHpStart0()); // splicing window start for INDEL
    for (int rep_period = min_mnr_rep_period; rep_period <= max_mnr_rep_period; ++rep_period){
    	CircluarBuffer<char> mnr_window(0);
    	int mnr_start = FindSplicingStartForMNR(ref_reader, position0, rep_period, mnr_window);
        splicing_left_bound = min(splicing_left_bound, mnr_start); // splicing window start for MNR
    }
    splicing_left_bound = max(0, splicing_left_bound); // safety
    return splicing_left_bound;
}

// Splicing start for MNR only depends on context, so I do it here.
int LocalReferenceContext::FindSplicingStartForMNR(const ReferenceReader &ref_reader, int variant_pos, int rep_period, CircluarBuffer<char>& window) const{
	int mnr_start = variant_pos;
	if (variant_pos + rep_period >= (int) ref_reader.chr_size(chr_idx)){
	    return mnr_start;
	}

	--mnr_start; // 1 anchor base
	window = CircluarBuffer<char>(rep_period);
	for (int idx = 0; idx < rep_period; idx++){
	   window.assign(idx, ref_reader.base(chr_idx, variant_pos + idx));
	}

    // Investigate (inclusive) start position of MNR region
	window.shiftLeft(1);
	while (mnr_start > 0 and window.first() == ref_reader.base(chr_idx, mnr_start)) {
		mnr_start--;
	    window.shiftLeft(1);
	}
	return mnr_start;
}

// Originally, the start splicing window was calculated in AlleleIdentity::CalculateWindowForVariant.
// Now LocalReferenceContext::SplicingLeftBound wants to calculate the left bound of the splicing window,
// so I calculate the splicing window start here to make sure that LocalReferenceContext can handle all possible start splicing windows.
int LocalReferenceContext::StartSplicingNoExpansion() const{
	return (int) position0;
}
int LocalReferenceContext::StartSplicingExpandFromMyHpStart0() const{
	return my_hp_start_pos[0] - 1;
}
int LocalReferenceContext::StartSplicingExpandFromMyHpStartLeftAnchor(int left_anchor) const{
	return my_hp_start_pos[left_anchor - 1] - 1;
}
