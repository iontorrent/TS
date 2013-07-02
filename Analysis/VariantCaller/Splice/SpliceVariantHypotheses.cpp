/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#include "SpliceVariantHypotheses.h"


bool SpliceVariantHypotheses(const ExtendedReadInfo &current_read, const AlleleIdentity &variant_identity,
                        const LocalReferenceContext &local_context, PersistingThreadObjects &thread_objects,
                        vector<string> &my_hypotheses, const InputStructures &global_context) {

  // Three hypotheses: 1) Null; read as called 2) Reference Hypothesis 3) Variant Hypothesis
  my_hypotheses.resize(3);

  // 1) Aligned portion of read as called
  unsigned int null_hyp_length = current_read.alignment.QueryBases.length() - current_read.startSC - current_read.endSC;
  my_hypotheses[0] = current_read.alignment.QueryBases.substr(current_read.startSC, null_hyp_length);

  // Initialize variables
  my_hypotheses[1].reserve(current_read.pretty_aln.length() + local_context.reference_allele.length());
  my_hypotheses[1].clear();
  my_hypotheses[2].reserve(current_read.pretty_aln.length() + local_context.reference_allele.length());
  my_hypotheses[2].clear();

  int read_idx = current_read.startSC;
  int ref_idx  = current_read.alignment.Position;
  int read_idx_max = null_hyp_length + current_read.startSC;
  bool did_splicing = false;
  string pretty_alignment;

  // do realignment of a small region around snp variant if desired
  if (global_context.do_snp_realignment and variant_identity.status.doRealignment) {
    pretty_alignment = SpliceDoRealignement(thread_objects, current_read,
    		                                local_context.position0, global_context.DEBUG);
    if (pretty_alignment.empty() and global_context.DEBUG > 1)
      cout << "Realignment returned an empty string!" << endl;
  }

  if (pretty_alignment.empty())
    pretty_alignment = current_read.pretty_aln;

  // Now fill in 2) and 3)

  for (unsigned int pretty_idx = 0; pretty_idx < pretty_alignment.length(); pretty_idx++) {

    bool outside_of_window = ref_idx < variant_identity.start_window or ref_idx >= variant_identity.end_window;
    bool outside_ref_allele = (long)ref_idx < local_context.position0 or ref_idx >= (int)(local_context.position0 + local_context.reference_allele.length());

    // Sanity checks
    if (read_idx >= read_idx_max or (unsigned int)ref_idx >= thread_objects.local_contig_sequence.length()) {
      did_splicing = false;
      break;
    }

    // --- Splice ---
    if (ref_idx == local_context.position0 and !did_splicing and !outside_of_window) {
      // New school way: treat insertions before SNPs & MNVs as if they were outside of window
      if (variant_identity.status.isSNP or variant_identity.status.isMNV) {
        while (pretty_idx < pretty_alignment.length() and pretty_alignment.at(pretty_idx) == '+') {
          my_hypotheses[1].push_back(current_read.alignment.QueryBases.at(read_idx));
          my_hypotheses[2].push_back(current_read.alignment.QueryBases.at(read_idx));
          read_idx++;
          pretty_idx++;
        }
      }

      did_splicing = SpliceAddVariantAlleles(current_read, pretty_alignment, variant_identity,
    		                    local_context, my_hypotheses, pretty_idx, global_context.DEBUG);
      /* // Old school way
      my_hypotheses[1] += local_context.reference_allele;
      my_hypotheses[2] += variant_identity.altAllele;
      did_splicing = true;
      // */
    }

    // Have reference bases inside of window but outside of span of reference allele
    if (outside_ref_allele and !outside_of_window and pretty_alignment.at(pretty_idx) != '+') {
      my_hypotheses[1].push_back(thread_objects.local_contig_sequence.at(ref_idx));
      my_hypotheses[2].push_back(thread_objects.local_contig_sequence.at(ref_idx));
    }

    // Have read bases as called outside of variant window
    if (outside_of_window and pretty_alignment.at(pretty_idx) != '-') {
      my_hypotheses[1].push_back(current_read.alignment.QueryBases.at(read_idx));
      my_hypotheses[2].push_back(current_read.alignment.QueryBases.at(read_idx));
    }

    IncrementAlignmentIndices(pretty_alignment.at(pretty_idx), ref_idx, read_idx);

  } // end of for loop over extended pretty alignment

  if (!current_read.is_forward_strand)
    for (int idx = 0; idx<3; idx++)
      RevComplementInPlace(my_hypotheses[idx]);

  // Check whether the whole reference allele fit
  if (ref_idx < (int)(local_context.position0 + local_context.reference_allele.length()))
    did_splicing = false;

  // Fail safe for hypotheses and verbose
  if (!did_splicing) {
    my_hypotheses[1] = my_hypotheses[0];
    my_hypotheses[2] = my_hypotheses[0];
    if (global_context.DEBUG > 1)
      cout << "Failed to splice " << local_context.reference_allele << "->" << variant_identity.altAllele
           << " into read " << current_read.alignment.Name << endl;
  }
  else if (global_context.DEBUG > 1) {
	  cout << "Spliced " << local_context.reference_allele << "->" << variant_identity.altAllele
	             << " into read " << current_read.alignment.Name << endl;
	  cout << "Read as called: " << my_hypotheses[0] << endl;
	  cout << "Reference Hyp.: " << my_hypotheses[1] << endl;
	  cout << "Variant Hyp.  : " << my_hypotheses[2] << endl;
  }

  return did_splicing;
};

// -------------------------------------------------------------------

void IncrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx) {

  switch (aln_symbol) {
    case ('-'):
      ref_idx++;
    break;
    case ('+'):
    case (' '):
    case ('|'):
      read_idx++;
      if (aln_symbol != '+')
        ref_idx++;
      break;
  }
}

void DecrementAlignmentIndices(const char aln_symbol, int &ref_idx, int &read_idx) {

  switch (aln_symbol) {
    case ('-'):
      ref_idx--;
    break;
    case ('+'):
    case (' '):
    case ('|'):
      read_idx--;
      if (aln_symbol != '+')
        ref_idx--;
      break;
  }
}

// -------------------------------------------------------------------

// This function is useful in the case that insertion count towards reference index before them.
bool SpliceAddVariantAlleles(const ExtendedReadInfo &current_read, const string pretty_alignment,
                             const AlleleIdentity &variant_identity, const LocalReferenceContext &local_context,
                             vector<string> &my_hypotheses, unsigned int pretty_idx, int DEBUG) {

  int shifted_position = 0;
  my_hypotheses[1] += local_context.reference_allele;

  // Special SNP splicing to not accidentally split HPs in the presence of insertions at start of HP
  if (variant_identity.status.isSNP) {

	unsigned int splice_idx = my_hypotheses[2].length();
    my_hypotheses[2] += local_context.reference_allele;

    // move left if there are insertions of the same base as the reference hypothesis base
    while (pretty_idx > 0 and pretty_alignment.at(pretty_idx-1)=='+' and splice_idx > 0
            and current_read.alignment.QueryBases.at(splice_idx-1)==local_context.reference_allele.at(0)) {
      pretty_idx--;
      splice_idx--;
      shifted_position++;
    }
    if (DEBUG > 1 and shifted_position > 0) {
      // printouts
      cout << "Shifted splice position by " << shifted_position << " in " << current_read.alignment.Name
           << " " << local_context.position0 << local_context.reference_allele
           << "->" << variant_identity.altAllele << endl;
      cout << my_hypotheses[2] << endl;
    }

    my_hypotheses[2].at(splice_idx) = variant_identity.altAllele.at(0);
  }
  else { // Default splicing
    my_hypotheses[2] += variant_identity.altAllele;
  }
  return true;
}

// -------------------------------------------------------------------


string SpliceDoRealignement (PersistingThreadObjects &thread_objects, const ExtendedReadInfo &current_read,
		                     long variant_position, int DEBUG) {

  //Realigner realigner(30, 1);
  thread_objects.realigner.SetClipping(0);
  thread_objects.realigner.SetStrand(current_read.is_forward_strand);
  string new_alignment;


  // --- Get index positions at snp variant position
  int read_idx = current_read.startSC;
  int ref_idx  = current_read.alignment.Position;
  unsigned int pretty_idx = 0;

  while (pretty_idx < current_read.pretty_aln.length() and ref_idx < variant_position) {
    IncrementAlignmentIndices(current_read.pretty_aln.at(pretty_idx), ref_idx, read_idx);
    pretty_idx++;
  }
  if (DEBUG > 1)
    cout << "Computed variant position as (red, ref, pretty) " << read_idx << " " << ref_idx << " " << pretty_idx << endl;

  if (pretty_idx >= current_read.pretty_aln.length()
       or ref_idx  >= (int)thread_objects.local_contig_sequence.length()
       or read_idx >= (int)current_read.alignment.QueryBases.length() - current_read.endSC)
    return new_alignment;

  // --- Get small sequence context for very local realignment ------------------------
  int min_bases = 5;

  // Looking at alignment to the left of variant position to find right place to cut sequence
  int read_left = read_idx;
  int ref_left  = ref_idx;
  unsigned int pretty_left = pretty_idx;
  bool continue_looking = pretty_idx > 0;

  while (continue_looking) {
    pretty_left--;
	DecrementAlignmentIndices(current_read.pretty_aln.at(pretty_left), ref_left, read_left);

	// Stopping criterion
	if (pretty_left < 1) {
      continue_looking = false;
      break;
	}
	if (ref_idx - ref_left < min_bases)
      continue_looking = true;
	else {
	  // make sure to start with a matching base and don't split large HPs
	  if (current_read.pretty_aln.at(pretty_left) != '|'
          or (thread_objects.local_contig_sequence.at(ref_left+1) == thread_objects.local_contig_sequence.at(ref_left)))
	    continue_looking = true;
	  else
	    continue_looking = false;
	}
  }
  if (DEBUG > 1)
    cout << "Computed left realignment window as (red, ref, pretty) " << read_left << " " << ref_left << " " << pretty_left << endl;


  // Looking at alignment to the right to find right place to cut sequence
  int read_right = read_idx;
  int ref_right  = ref_idx;
  unsigned int pretty_right = pretty_idx;
  continue_looking = pretty_idx < current_read.pretty_aln.length()-1;

  while (continue_looking) {
  	IncrementAlignmentIndices(current_read.pretty_aln.at(pretty_right), ref_right, read_right);
    pretty_right++;
  	// Stopping criterion (half open interval)
  	if (pretty_right >= current_read.pretty_aln.length()
        or ref_right >= (int)thread_objects.local_contig_sequence.length()) {
      continue_looking = false;
      break;
  	}
  	if (ref_right - ref_idx < min_bases)
        continue_looking = true;
  	else {
  	  // make sure to stop with a matching base and don't split large HPs
  	  if (current_read.pretty_aln.at(pretty_right-1) != '|'
          or (thread_objects.local_contig_sequence.at(ref_right-1) == thread_objects.local_contig_sequence.at(ref_right)))
  	    continue_looking = true;
  	  else
  	    continue_looking = false;
  	}
  }
  if (DEBUG > 1)
    cout << "Computed right realignment window as (red, ref, pretty) " << read_right << " " << ref_right << " " << pretty_right << endl;
  // Put in some sanity checks for alignment boundaries found...


  // --- Realign -------------------------
  unsigned int start_position_shift;
  vector<CigarOp>    new_cigar_data;
  vector<MDelement>  new_md_data;

  // printouts
  if (DEBUG > 1) {
    thread_objects.realigner.verbose_ = true;
    cout << "Realigned " << current_read.alignment.Name << " from " << endl;
  }

  thread_objects.realigner.SetSequences(current_read.alignment.QueryBases.substr(read_left, read_right-read_left),
                         thread_objects.local_contig_sequence.substr(ref_left, ref_right-ref_left),
                         current_read.pretty_aln.substr(pretty_left, pretty_right-pretty_left),
                         current_read.is_forward_strand);


  if (!thread_objects.realigner.computeSWalignment(new_cigar_data, new_md_data, start_position_shift)) {
	if (DEBUG > 1)
      cout << "ERROR: realignment failed! " << endl;
    return new_alignment;
  }

  // --- Fuse realigned partial sequence back into pretty_aln string
  new_alignment = current_read.pretty_aln;
  new_alignment.replace(pretty_left, (pretty_right-pretty_left), thread_objects.realigner.pretty_aln());
  return new_alignment;
}

