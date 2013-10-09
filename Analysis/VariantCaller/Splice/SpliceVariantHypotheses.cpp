/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#include "SpliceVariantHypotheses.h"


bool SpliceVariantHypotheses(ExtendedReadInfo &current_read, const MultiAlleleVariantIdentity &variant_identity,
                        const LocalReferenceContext &local_context, PersistingThreadObjects &thread_objects,
                        int &splice_start_flow, int &splice_end_flow, vector<string> &my_hypotheses,
                        const InputStructures &global_context) {

  // Hypotheses: 1) Null; read as called 2) Reference Hypothesis 3-?) Variant Hypotheses
  my_hypotheses.resize(variant_identity.allele_identity_vector.size()+2);

  // Set up variables to log the flows we splice into
  splice_start_flow = -1;
  splice_end_flow = -1;
  int splice_start_idx = -1;
  vector<int> splice_end_idx;
  splice_end_idx.assign(my_hypotheses.size(), -1);

  // 1) Null hypothesis is read as called
  if (global_context.resolve_clipped_bases) {
    unsigned int null_hyp_length = current_read.alignment.QueryBases.length() - current_read.leftSC - current_read.rightSC;
    unsigned int startSC = (current_read.is_forward_strand) ? current_read.leftSC : current_read.rightSC;
    my_hypotheses[0] = current_read.read_bases.substr(startSC, null_hyp_length);
    current_read.IncreaseStartFlow(); // Increment start flow to first aligned base
  }
  else
    my_hypotheses[0] = current_read.read_bases;

  // Initialize hypotheses variables for splicing
  for (unsigned int i_hyp = 1; i_hyp < my_hypotheses.size(); i_hyp++) {
    my_hypotheses[i_hyp].reserve(current_read.alignment.QueryBases.length() + 20 + local_context.reference_allele.length());
    my_hypotheses[i_hyp].clear();
    // Add soft clipped bases on the left side of alignment if desired
    if (!global_context.resolve_clipped_bases)
      my_hypotheses[i_hyp] += current_read.alignment.QueryBases.substr(0, current_read.leftSC);
  }

  int read_idx = current_read.leftSC;
  int ref_idx  = current_read.alignment.Position;
  int read_idx_max = current_read.alignment.QueryBases.length() - current_read.rightSC;
  bool did_splicing = false;
  bool just_did_splicing = false;
  string pretty_alignment;

  // do realignment of a small region around snp variant if desired
  if (global_context.do_snp_realignment and variant_identity.doRealignment) {
    pretty_alignment = SpliceDoRealignement(thread_objects, current_read,
    		                                local_context.position0, global_context.DEBUG);
    if (pretty_alignment.empty() and global_context.DEBUG > 0)
      cerr << "Realignment returned an empty string in read " << current_read.alignment.Name << endl;
  }

  if (pretty_alignment.empty())
    pretty_alignment = current_read.pretty_aln;

  // Now fill in 2) and 3)

  for (unsigned int pretty_idx = 0; pretty_idx < pretty_alignment.length(); pretty_idx++) {

    bool outside_of_window = ref_idx < variant_identity.window_start or ref_idx >= variant_identity.window_end;
    bool outside_ref_allele = (long)ref_idx < local_context.position0 or ref_idx >= (int)(local_context.position0 + local_context.reference_allele.length());

    // Basic sanity checks
    if (read_idx >= read_idx_max
        or  (unsigned int)ref_idx >  thread_objects.local_contig_sequence.length()
        or ((unsigned int)ref_idx == thread_objects.local_contig_sequence.length() and pretty_alignment.at(pretty_idx) != '+')) {
      did_splicing = false;
      break;
    }

    // --- Splice ---
    if (ref_idx == local_context.position0 and !did_splicing and !outside_of_window) {
      // Add insertions before variant window
      while (pretty_idx < pretty_alignment.length() and pretty_alignment.at(pretty_idx) == '+') {
    	for (unsigned int i_hyp = 1; i_hyp < my_hypotheses.size(); i_hyp++)
          my_hypotheses[i_hyp].push_back(current_read.alignment.QueryBases.at(read_idx));
        read_idx++;
        pretty_idx++;
      }
      did_splicing = SpliceAddVariantAlleles(current_read, pretty_alignment, variant_identity,
    		                    local_context, my_hypotheses, pretty_idx, global_context.DEBUG);
      just_did_splicing = did_splicing;
    } // --- ---

    // Have reference bases inside of window but outside of span of reference allele
    if (outside_ref_allele and !outside_of_window and pretty_alignment.at(pretty_idx) != '+') {
      for (unsigned int i_hyp = 1; i_hyp < my_hypotheses.size(); i_hyp++)
        my_hypotheses[i_hyp].push_back(thread_objects.local_contig_sequence.at(ref_idx));
    }

    // Have read bases as called outside of variant window
    if (outside_of_window and pretty_alignment.at(pretty_idx) != '-') {
      for (unsigned int i_hyp = 1; i_hyp < my_hypotheses.size(); i_hyp++)
        my_hypotheses[i_hyp].push_back(current_read.alignment.QueryBases.at(read_idx));

      // --- Information to log flows. Indices are w.r.t. aligned portion of the read
      if (!did_splicing) { // Log index of the last base left of window which is the same for all hypotheses.
        splice_start_idx = read_idx - current_read.leftSC;
      }
      else if (just_did_splicing) { // Log length of hypothesis after splicing
    	splice_end_idx[0] = read_idx  - current_read.leftSC;
    	int clipped_bases = 0;
    	if (!global_context.resolve_clipped_bases)
    	  clipped_bases = current_read.leftSC;
        for (unsigned int i_hyp=1; i_hyp<my_hypotheses.size(); i_hyp++)
          splice_end_idx[i_hyp] = my_hypotheses[i_hyp].length()-1 - clipped_bases; // Hyp length depends on whether there is resolving!
        just_did_splicing = false;
      }
      // --- ---
    }

    IncrementAlignmentIndices(pretty_alignment.at(pretty_idx), ref_idx, read_idx);

  } // end of for loop over extended pretty alignment

  // Check whether the whole reference allele fit
  if (ref_idx < (int)(local_context.position0 + local_context.reference_allele.length())) {
    did_splicing = false;
    cout << "Warning in Splicing: Reference allele "<< local_context.reference_allele << " did not fit into read " << current_read.alignment.Name << endl;
  }

  if (did_splicing) {
    // --- Add soft clipped bases to the right of the alignment and reverse complement ---
    for (unsigned int i_hyp = 1; i_hyp<my_hypotheses.size(); i_hyp++) {
      if (!global_context.resolve_clipped_bases)
        my_hypotheses[i_hyp] += current_read.alignment.QueryBases.substr(current_read.alignment.QueryBases.length()-current_read.rightSC, current_read.rightSC);

      if (!current_read.is_forward_strand)
        RevComplementInPlace(my_hypotheses[i_hyp]);
    }

    // Get the main flows before and after splicing
    splice_end_flow = GetSpliceFlows(current_read, global_context, my_hypotheses,
                                     splice_start_idx, splice_end_idx, splice_start_flow);
    if (splice_start_flow < 0 or splice_end_flow <= splice_start_flow) {
      did_splicing = false;
      cout << "Warning in Splicing: Splice flows are not valid in read " << current_read.alignment.Name
           << ". splice start flow: "<< splice_start_flow << " splice end flow " << splice_end_flow << endl;
    }
  }

  // --- Fail safe for hypotheses and verbose
  if (!did_splicing) {
	for (unsigned int i_hyp=1; i_hyp<my_hypotheses.size(); i_hyp++)
      my_hypotheses[i_hyp] = my_hypotheses[0];
    if (global_context.DEBUG > 1) {
      cout << "Failed to splice " << local_context.reference_allele << "->";
      for (unsigned int i_alt = 0; i_alt<variant_identity.allele_identity_vector.size(); i_alt++) {
    	cout << variant_identity.allele_identity_vector[i_alt].altAllele;
        if (i_alt < variant_identity.allele_identity_vector.size()-1)
          cout << ",";
      }
      cout << " into read " << current_read.alignment.Name << endl;
    }
  }
  else if (global_context.DEBUG > 1) {
	cout << "Spliced " << local_context.reference_allele << "->";
    for (unsigned int i_alt = 0; i_alt<variant_identity.allele_identity_vector.size(); i_alt++) {
      cout << variant_identity.allele_identity_vector[i_alt].altAllele;
      if (i_alt < variant_identity.allele_identity_vector.size()-1)
        cout << ",";
    }
    cout << " into ";
    if (current_read.is_forward_strand) cout << "forward ";
    else cout << "reverse ";
    cout <<	"strand read read " << current_read.alignment.Name << endl;
    cout << "- Read as called: " << my_hypotheses[0] << endl;
    cout << "- Reference Hyp.: " << my_hypotheses[1] << endl;
    for (unsigned int i_hyp = 2; i_hyp<my_hypotheses.size(); i_hyp++)
      cout << "- Variant Hyp. " << (i_hyp-1) << ": " << my_hypotheses[i_hyp] << endl;
    cout << "- Splice start flow: " << splice_start_flow << " Splice end flow: " << splice_end_flow << endl;
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

void IncrementFlow(const ion::FlowOrder &flow_order, const char &nuc, int &flow) {
  while (flow < flow_order.num_flows() and flow_order.nuc_at(flow) != nuc)
    flow++;
}

void IncrementFlows(const ion::FlowOrder &flow_order, const char &nuc, vector<int> &flows) {
  for (unsigned int idx = 1; idx < flows.size(); idx++)
    while (flows[idx] < flow_order.num_flows() and flow_order.nuc_at(flows[idx]) != nuc)
      flows[idx]++;
}

// -------------------------------------------------------------------

// This function is useful in the case that insertion count towards reference index before them.
bool SpliceAddVariantAlleles(const ExtendedReadInfo &current_read, const string pretty_alignment,
                             const MultiAlleleVariantIdentity &variant_identity,
                             const LocalReferenceContext &local_context, vector<string> &my_hypotheses,
                             unsigned int pretty_idx, int DEBUG) {

  int shifted_position = 0;
  // Splice reference Hypothesis
  my_hypotheses[1] += local_context.reference_allele;

  for (unsigned int i_hyp=2; i_hyp<my_hypotheses.size(); i_hyp++) {
    int my_allele_idx = i_hyp-2;

	// Special SNP splicing to not accidentally split HPs in the presence of insertions at start of HP
    if (variant_identity.allele_identity_vector[my_allele_idx].status.isSNP) {
      unsigned int splice_idx = my_hypotheses[i_hyp].length();
      my_hypotheses[i_hyp] += local_context.reference_allele;
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
             << "->" << variant_identity.allele_identity_vector[my_allele_idx].altAllele << endl;
        cout << my_hypotheses[i_hyp] << endl;
      }
      my_hypotheses[i_hyp].at(splice_idx) = variant_identity.allele_identity_vector[my_allele_idx].altAllele.at(0);
    }
    else { // Default splicing
      my_hypotheses[i_hyp] += variant_identity.allele_identity_vector[my_allele_idx].altAllele;
    }
  } // end looping over hypotheses
  return true;
}

// -------------------------------------------------------------------


int GetSpliceFlows(ExtendedReadInfo &current_read, const InputStructures &global_context,
                   vector<string> &my_hypotheses, int splice_start_idx,
                   vector<int> splice_end_idx, int &splice_start_flow) {

  int splice_end_flow = -1;
  splice_start_flow = -1;
  int my_start_idx = splice_start_idx;
  vector<int> splice_length(my_hypotheses.size());
  bool error_occurred = false;

  // Hypotheses have already been reverse complemented by the time we call this function
  // Set relevant indices
  int added_SC_bases = 0;
  if (!global_context.resolve_clipped_bases) {
	// We added the soft clipped bases to our hypotheses to be simulated
    added_SC_bases = current_read.leftSC + current_read.rightSC;
  }
  for (unsigned int i_hyp = 0; i_hyp < my_hypotheses.size(); i_hyp++) {
    if (splice_end_idx.at(i_hyp) == -1) { // We did not splice another base after the variant window
      // splice start & end indices are w.r.t the aligned portion of the read
      splice_end_idx.at(i_hyp) = my_hypotheses.at(i_hyp).length() - added_SC_bases;
    }
    splice_length.at(i_hyp) = splice_end_idx.at(i_hyp) - splice_start_idx -1;
  }
  if (!current_read.is_forward_strand) { // The same number of bases have been added beyond the window
    my_start_idx = my_hypotheses[0].length() - added_SC_bases -1 - splice_end_idx.at(0);
    if (global_context.DEBUG>2)
      cout << "--> reverse strand splicing:" << endl;
  }
  else if (global_context.DEBUG>2) {
    cout << "--> forward strand splicing:" << endl;;
  }

  // --- Get splice start flow and adjust index of first spliced base
  if (my_start_idx < 0) { // Our variant window started from the first base of the alignment
    if (current_read.GetStartSC() > 0) { // We have trimmed bases that can act as an anchor to get the flows right
      splice_start_flow = current_read.flowIndex.at(current_read.GetStartSC()-1);
      // splice_start_idx remains at -1;
	}
    else { // Test if all allele windows start with the same base
      bool have_anchor = true;
      for (unsigned int i_hyp=1; i_hyp<my_hypotheses.size(); i_hyp++)
        have_anchor = have_anchor and (my_hypotheses[i_hyp].at(0) == my_hypotheses[0].at(0));
      if (have_anchor) {
        my_start_idx = 0;
        splice_start_flow = current_read.start_flow;
	for (unsigned int i_hyp = 0; i_hyp < my_hypotheses.size(); i_hyp++) {
          // We shrank the size of the splicing window by one
          splice_length.at(i_hyp)--;
        }
      }
      else { // In this case, the splice_start_flow depends on the prefix (key+barcode) of the read, which we solve later
             // Prediction generation is doing the right thing, even though we botch things up a bit here.
             // And we are giving a bit of leeway in the test flow window to compensate for our botching.
        splice_start_flow = current_read.start_flow-1;
        // splice_start_idx remains at -1;
      }
    }
  } // the above block handles the my_start_idx == -1 exception
  else
    splice_start_flow = current_read.flowIndex.at(current_read.GetStartSC() + my_start_idx);
  // ---

  if (!global_context.resolve_clipped_bases)
    my_start_idx += current_read.GetStartSC(); // Add soft clipped start to index
  my_start_idx++; // my_start_idx is now pointing at the first spliced base

  // Computing splice_end_flow
  for (unsigned int i_hyp=0; i_hyp<my_hypotheses.size(); i_hyp++) {
    int my_flow = splice_start_flow;
    int my_end_idx = my_start_idx + splice_length.at(i_hyp);
    for (int i_base=my_start_idx; i_base<my_end_idx; i_base++) {
      if (i_base >= (int)my_hypotheses[i_hyp].length()) {
        error_occurred = true;
        break;
      }
      IncrementFlow(global_context.treePhaserFlowOrder,my_hypotheses[i_hyp].at(i_base), my_flow);
    }
    if (my_flow > splice_end_flow)
      splice_end_flow = my_flow;
    // reverse verbose
    if (global_context.DEBUG>2)
      cout << "Hypothesis " << i_hyp << " splice_end_idx " << splice_end_idx.at(i_hyp) << " splice_length " << splice_length.at(i_hyp)
           << " my_start_idx " << my_start_idx << " my_end_idx " << my_end_idx
           << " splice_start_flow " << splice_start_flow << " my_end_flow " << my_flow << endl;
  }

  if (error_occurred)
    splice_end_flow = -1;
  // verbose
  if (global_context.DEBUG>2)
    cout << "Splice_end_flow: " << splice_end_flow << endl;;

  return splice_end_flow;
}

// -------------------------------------------------------------------


string SpliceDoRealignement (PersistingThreadObjects &thread_objects, const ExtendedReadInfo &current_read,
		                     long variant_position, int DEBUG) {

  //Realigner realigner(30, 1);
  thread_objects.realigner.SetClipping(0);
  thread_objects.realigner.SetStrand(current_read.is_forward_strand);
  string new_alignment;


  // --- Get index positions at snp variant position
  int read_idx = current_read.leftSC;
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
       or read_idx >= (int)current_read.alignment.QueryBases.length() - current_read.rightSC)
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

