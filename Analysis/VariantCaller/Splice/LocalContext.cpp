/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     LocalContext.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "LocalContext.h"



void LocalReferenceContext::DetectContext(const string & local_contig_sequence, vcf::Variant ** candidate_variant, int DEBUG) {

  // VCF stores positions in 1-based index; local_contig_sequence has a zero based index
  // all positions in this object are zero based so that they correspond to reference in memory.
  position0 = (*candidate_variant)->position-1;
  contigName = (*candidate_variant)->sequenceName;

  // Sanity checks if position is valid and reference allele matches reference
  if (!ContextSanityChecks(local_contig_sequence, candidate_variant))
    return;

  my_hp_length.resize(reference_allele.length(), 0);
  my_hp_start_pos.resize(reference_allele.length(), 0);

  // Process first HP (inclusive, zero based start position)
  my_hp_start_pos.at(0) = position0;
  my_hp_length.at(0) = 1;
  while (my_hp_start_pos.at(0) > 0
         and local_contig_sequence.at(my_hp_start_pos.at(0)-1) == local_contig_sequence.at(position0)) {
    my_hp_start_pos.at(0)--;
    my_hp_length.at(0)++;
  }
  // Now get base and length of the HP to the left of the one containing variant start
  long temp_position = my_hp_start_pos.at(0) -1;
  ref_left_hp_base = 'X'; //
  left_hp_length = 0;
  left_hp_start = temp_position;
  if (temp_position >= 0) {
    ref_left_hp_base = local_contig_sequence.at(temp_position);
    left_hp_length++;
  }
  while (temp_position > 0 and local_contig_sequence.at(temp_position-1) == ref_left_hp_base) {
    temp_position--;
    left_hp_start--;
    left_hp_length++;
  }

  // Get HP context of the remaining bases in the reference allele and record for each base
  for (unsigned int b_idx = 1; b_idx < reference_allele.length(); b_idx++) {
    // See if next base in reference allele starts a new HP and adjust length
	if (local_contig_sequence.at(position0 + b_idx-1) == local_contig_sequence.at(position0 + b_idx)) {
      my_hp_start_pos.at(b_idx) = my_hp_start_pos.at(b_idx-1);
      for (unsigned int l_idx = 0; l_idx < b_idx; l_idx++)
        if (my_hp_start_pos.at(l_idx) == my_hp_start_pos.at(b_idx))
          my_hp_length.at(l_idx)++;
      my_hp_length.at(b_idx) = my_hp_length.at(b_idx-1);
    }
    else {
      my_hp_start_pos.at(b_idx) = position0 + b_idx;
      my_hp_length.at(b_idx) = 1;
    }
  }

  // Complete the HP length of the last base in the reference allele
  temp_position = position0 + reference_allele.length() -1;
  while (temp_position < (int)local_contig_sequence.length()-1 and
	     local_contig_sequence.at(temp_position+1) ==
	     local_contig_sequence.at(position0 + reference_allele.length() -1)) {
    temp_position++;
    for (unsigned int b_idx = 0; b_idx < reference_allele.length(); b_idx++) {
      if (my_hp_start_pos.at(b_idx) == my_hp_start_pos.at(reference_allele.length()-1))
        my_hp_length.at(b_idx)++;
    }
  }

  // Get HP to the right of the one containing last base of the reference allele
  ref_right_hp_base = 'X';
  right_hp_length = 0;
  temp_position = my_hp_start_pos.at(reference_allele.length()-1) + my_hp_length.at(reference_allele.length()-1);
  right_hp_start = temp_position;
  if (temp_position < (int)local_contig_sequence.length()) {
    ref_right_hp_base = local_contig_sequence.at(temp_position);
    right_hp_length++;
  }
  while (temp_position < (int)local_contig_sequence.length()-1 and local_contig_sequence.at(temp_position+1) == ref_right_hp_base) {
    temp_position++;
    right_hp_length++;
  }

  if (DEBUG>0) {
    cout << "Local Reference context at (zero-based) " << position0 << ", " << reference_allele << " :";
    cout << left_hp_length << ref_left_hp_base << " ";
    for (unsigned int idx = 0; idx < reference_allele.length(); idx++)
      cout << my_hp_length.at(idx) << reference_allele.at(idx) << "(" << my_hp_start_pos.at(idx)  << ") ";
    cout << right_hp_length << ref_right_hp_base << " ";
    // Some additional printout to see if I get index right:
    for (int idx=max(position0-7, (long)0); idx<position0; idx++)
      cout << local_contig_sequence[idx];
    cout << "|" << local_contig_sequence[position0] << "|";
    for (int idx=position0+1; idx < min(position0+8,(long)local_contig_sequence.length()); idx++)
      cout << local_contig_sequence[idx];
    cout << endl;
  }
}

bool LocalReferenceContext::ContextSanityChecks(const string & local_contig_sequence, vcf::Variant ** candidate_variant) {

  // Sanity checks that abort context detection
  reference_allele = (*candidate_variant)->ref;
  context_detected = true;

  if ((*candidate_variant)->position < 1 or (*candidate_variant)->position > (long)local_contig_sequence.length()) {
    cerr << "Non-fatal ERROR: Candidate Variant Position is not within the Contig Bounds at VCF Position "
         << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position
         << " Contig length = " << local_contig_sequence.length() <<  endl;
    cout << "Non-fatal ERROR: Candidate Variant Position is not within the Contig Bounds at VCF Position "
         << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position
         << " Contig length = " << local_contig_sequence.length() << endl;
    // Choose safe parameter
    position0 = 0;
    context_detected = false;
  }

  if (reference_allele.length() == 0) {
    cerr << "Non-fatal ERROR: Reference allele has zero length at vcf position "
         << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position << endl;
    cout << "Non-fatal ERROR: Reference allele has zero length at vcf position "
         << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position << endl;
    // Choose safe parameter
    reference_allele = local_contig_sequence.at(position0);
    context_detected = false;
  }

  if (((unsigned int)(*candidate_variant)->position + reference_allele.length() -1) > local_contig_sequence.length()) {
    cerr << "Non-fatal ERROR: Reference Allele stretches beyond Contig Bounds at VCF Position "
	     << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position
	     << " Contig length = " << local_contig_sequence.length()
	     << " Reference Allele: " << reference_allele << endl;
    cout << "Non-fatal ERROR: Reference Allele stretches beyond Contig Bounds at VCF Position "
	     << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position
	     << " Contig length = " << local_contig_sequence.length()
	     << " Reference Allele: " << reference_allele << endl;
    // Choose safe parameter
    reference_allele = reference_allele.at(0);
    context_detected = false;
  }

  string contig_str(local_contig_sequence, position0, reference_allele.length());
  if (reference_allele.compare(contig_str) != 0) {
    cerr << "Non-fatal ERROR: Reference allele does not match reference at VCF position "
         << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position
         << " Reference Allele: " << reference_allele << " Reference: " << contig_str << endl;
    cout << "Non-fatal ERROR: Reference allele does not match reference at VCF position "
         << (*candidate_variant)->sequenceName << ":" << (*candidate_variant)->position
         << " Reference Allele: " << reference_allele << " Reference: " << contig_str << endl;
    context_detected = false;
  }

  return (context_detected);
}
