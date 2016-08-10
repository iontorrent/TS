/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "FlowAlignment.h"
#include <stdio.h>
#include "MiscUtil.h"

using namespace std;
using namespace BamTools;


// Helper functions first
bool IsInDelAlignSymbol(char symbol)
{
  return (symbol==ALN_DEL or symbol==ALN_INS);
}

/*char ReverseComplement (char nuc)
{
  switch(nuc) {
    case 'A': return 'T';
    case 'C': return 'G';
    case 'G': return 'C';
    case 'T': return 'A';
    case 'a': return 't';
    case 'c': return 'g';
    case 'g': return 'c';
    case 't': return 'a';
    default:  return nuc;
  }
}

void ReverseComplementInPlace (string& sequence)
{
  if (sequence.length()==0)
    return;

  char *forward = (char *)sequence.c_str();
  char *reverse = forward + sequence.length() - 1;
  while (forward < reverse) {
    char f = *forward;
    char r = *reverse;
    *forward++ = ReverseComplement(r);
    *reverse-- = ReverseComplement(f);
  }
  if (forward == reverse)
    *forward = ReverseComplement(*forward);
}*/




// -----------------------------------------------------------------------

// Generates Reference sequence for aligned portion of read from cigar and md tag
// Output:  tseq_bases : refernce (target) bases for aligned portion of the read
//          qseq_bases : read (query) bases for aligned portion of the read
//         pretty_tseq : padded (incl. '-' gaps) target sequence
//         pretty_qseq : padded (incl. '-' gaps) target sequence
//          pretty_aln : Alignment operations for pretty strings
//             left_sc : amount of soft clipped bases on the left side
//            right_sc : amount of soft clipped bases on the right side

void RetrieveBaseAlignment(
    // Inputs:
    const string&           alignment_query_bases,
    const vector<CigarOp>&  alignment_cigar_data,
    const string&           md_tag,
    // Outputs:
    string&                 tseq_bases,
    string&                 qseq_bases,
    string&                 pretty_tseq,
    string&                 pretty_qseq,
    string&                 pretty_aln,
    unsigned int&           left_sc,
    unsigned int&           right_sc)
{

  //
  // Step 1. Generate reference sequence based on QueryBases and Cigar alone
  //

  tseq_bases.reserve(2 * alignment_query_bases.size());
  qseq_bases.reserve(2 * alignment_query_bases.size());
  pretty_tseq.reserve(2 * alignment_query_bases.size());
  pretty_qseq.reserve(2 * alignment_query_bases.size());
  pretty_aln.reserve(2 * alignment_query_bases.size());

  const char *read_ptr = alignment_query_bases.c_str();
  bool match_found = false;
  left_sc = right_sc = 0;

  for (vector<CigarOp>::const_iterator cigar = alignment_cigar_data.begin(); cigar != alignment_cigar_data.end(); ++cigar) {
    switch (cigar->Type) {
      case (Constants::BAM_CIGAR_MATCH_CHAR)    :
      case (Constants::BAM_CIGAR_SEQMATCH_CHAR) :
      case (Constants::BAM_CIGAR_MISMATCH_CHAR) :
        tseq_bases.append(read_ptr, cigar->Length);
        qseq_bases.append(read_ptr, cigar->Length);
        pretty_tseq.append(read_ptr, cigar->Length);
        pretty_qseq.append(read_ptr, cigar->Length);
        pretty_aln.append(cigar->Length, '|');
        match_found = true;
        read_ptr += cigar->Length; break;

      case (Constants::BAM_CIGAR_INS_CHAR)      :
        qseq_bases.append(read_ptr, cigar->Length);
        pretty_tseq.append(cigar->Length,'-');
        pretty_qseq.append(read_ptr, cigar->Length);
        pretty_aln.append(cigar->Length, '+');
        read_ptr += cigar->Length; break;

      case (Constants::BAM_CIGAR_SOFTCLIP_CHAR) :
        read_ptr += cigar->Length;
        if (match_found)
          right_sc = cigar->Length;
        else
          left_sc = cigar->Length;
        break;

      case (Constants::BAM_CIGAR_DEL_CHAR)      :
      case (Constants::BAM_CIGAR_PAD_CHAR)      :
      case (Constants::BAM_CIGAR_REFSKIP_CHAR)  :
        tseq_bases.append(cigar->Length, '-');
        pretty_tseq.append(cigar->Length,'-');
        pretty_qseq.append(cigar->Length,'-');
        pretty_aln.append(cigar->Length, '-');
        break;
    }
  }

  //
  // Step 2: Further patch the sequence based on MD tag
  //

  char *ref_ptr = (char *)tseq_bases.c_str();
  int pretty_idx = 0;
  const char *MD_ptr = md_tag.c_str();

  while (*MD_ptr and *ref_ptr) {
    if (*MD_ptr >= '0' and *MD_ptr <= '9') {    // Its a match
      int item_length = 0;
      for (; *MD_ptr >= '0' and *MD_ptr <= '9'; ++MD_ptr)
        item_length = 10*item_length + *MD_ptr - '0';
      ref_ptr += item_length;
      while (item_length > 0 or pretty_aln[pretty_idx] == '+') {
        if (pretty_aln[pretty_idx] != '+')
          item_length--;
        pretty_idx++;
      }
    } else {
      if (*MD_ptr == '^')                       // Its a deletion or substitution
        MD_ptr++;
      while (*ref_ptr and *MD_ptr >= 'A' and *MD_ptr <= 'Z') {
        if (pretty_aln[pretty_idx] == '|')
          pretty_aln[pretty_idx] = ' ';
        pretty_tseq[pretty_idx++] = *MD_ptr + 'a' - 'A';
        *ref_ptr++ = *MD_ptr++;
      }
    }
  }
}

// =================================================================================

// Function does a flow alignment and tries to match homopolymer lengths
// Inputs:         tseq_bases : Target bases
//                 qseq_bases : Query bases
//            main_flow_order : Flow order string of run
//          first_useful_flow : start flow for flow alignment ()
// removed-not needed  fz_tag : scaled distortion from ideal signal        

bool PerformFlowAlignment(
    // Inputs:
    const string&             tseq_bases,
    const string&             qseq_bases,
    const string&             main_flow_order,
    int                       first_useful_flow,
    //const vector<uint16_t>&   fz_tag,
    // Outputs:
    vector<char>&             flowOrder,
    vector<int>&              qseq,
    vector<int>&              tseq,
    vector<int>&              aln_flow_index,
    vector<char>&             aln,
    bool                      debug_output)
{

  bool startLocal = false; // CK: I not sure these options work correctly
  bool endLocal = false;
  int phasePenalty = PHASE_PENALTY;


  // **** Generate homopolymer representation for the qseq_bases.


  vector<char>  qseq_hp_nuc;
  vector<int>   qseq_hp;
  vector<int>   qseq_flow_idx;

  qseq_hp_nuc.reserve(main_flow_order.size());
  qseq_hp.reserve(main_flow_order.size());
  qseq_flow_idx.reserve(main_flow_order.size());

  const char *base_ptr = qseq_bases.c_str();
  for (int flow = first_useful_flow; flow < (int)main_flow_order.size() and *base_ptr; ++flow) {
    qseq_hp_nuc.push_back(main_flow_order[flow]);
    //qseq_hp_perturbation.push_back(fz_tag.at(flow));
    qseq_flow_idx.push_back(flow);
    qseq_hp.push_back(0);
    while (*base_ptr!='\0' and *base_ptr == main_flow_order[flow]) {
      base_ptr++;
      qseq_hp.back()++;
    }
  }

  vector<int>   qseq_hp_previous_nuc(qseq_hp.size(),0);

  int last_nuc_pos[8] = {-100,-100,-100,-100,-100,-100,-100,-100};
  for (int q_idx = 0; q_idx < (int)qseq_hp.size(); ++q_idx) {
    if (last_nuc_pos[qseq_hp_nuc[q_idx]&7] >= 0)
      qseq_hp_previous_nuc[q_idx] = last_nuc_pos[qseq_hp_nuc[q_idx]&7] + 1;
    last_nuc_pos[qseq_hp_nuc[q_idx]&7] = q_idx;
  }

  // **** Generate homopolymer representation of tseq_bases.

  vector<char>  tseq_hp_nuc;
  vector<int>   tseq_hp;

  tseq_hp_nuc.reserve(tseq_bases.size());
  tseq_hp.reserve(tseq_bases.size());
  char prev_base = 0;

  // TODO: Better handling of 'N's
  for (unsigned int tseq_bases_idx = 0; tseq_bases_idx < tseq_bases.size(); ++tseq_bases_idx) {
    char current_base = tseq_bases[tseq_bases_idx];
    switch (current_base) {
      case 'N':   current_base = 'A';   // weird rule from FlowSeq
      case 'A':
      case 'C':
      case 'G':
      case 'T':   break;
      default:    continue;
    }

    if (current_base != prev_base) {
      tseq_hp_nuc.push_back(current_base);
      tseq_hp.push_back(0);
      prev_base = current_base;
    }
    tseq_hp.back()++;
  }
  // **** Done


  // Initialize gaps sums
  vector<int> gapSumsI(qseq_hp.size(), phasePenalty);
  for(int q_idx = 0; q_idx < (int)qseq_hp.size(); ++q_idx)
    for(int idx = qseq_hp_previous_nuc[q_idx]; idx <= q_idx; ++idx)
      gapSumsI[q_idx] += qseq_hp[idx];


  // Stores the score for extending with a match.
  vector<vector<int> > dp_matchScore(1+qseq_hp.size(), vector<int>(1+tseq_hp.size(), MINOR_INF));
  // Stores the score for extending with a insertion.
  vector<vector<int> > dp_insScore(1+qseq_hp.size(), vector<int>(1+tseq_hp.size(), MINOR_INF));
  // Stores the score for extending with a deletion.
  vector<vector<int> > dp_delScore(1+qseq_hp.size(), vector<int>(1+tseq_hp.size(), MINOR_INF));

  // Stores the previous cell in the path to a match.
  vector<vector<int> > dp_matchFrom(1+qseq_hp.size(), vector<int>(1+tseq_hp.size(), FROM_S));
  // Stores the previous cell in the path to a insertion.
  vector<vector<int> > dp_insFrom(1+qseq_hp.size(), vector<int>(1+tseq_hp.size(), FROM_S));
  // Stores the previous cell in the path to a deletion.
  vector<vector<int> > dp_delFrom(1+qseq_hp.size(), vector<int>(1+tseq_hp.size(), FROM_S));

  // Vertical: Insertion score for first column of dp matrix
  // only allow phasing from an insertion
  for(int q_idx = 1; q_idx <= (int)qseq_hp.size(); ++q_idx) {
    int q_jump_idx = qseq_hp_previous_nuc[q_idx-1];
    if(0 == q_jump_idx) {
      dp_insScore[q_idx][0] = 0 - gapSumsI[q_idx-1];
      dp_insFrom[q_idx][0] = FROM_IP;
    } else {
      dp_insScore[q_idx][0] = dp_insScore[q_jump_idx][0] - gapSumsI[q_idx-1];
      dp_insFrom[q_idx][0] = FROM_IP;
    }
  }

  // Horizontal: Deletion score for first row of dp matrix
  for (unsigned int t_idx = 1; t_idx <= tseq_hp.size(); ++t_idx) {
    int previous = 0;
    if (t_idx > 1)
      previous = dp_delScore[0][t_idx-1];
    dp_delScore[0][t_idx] = previous - 2*tseq_hp[t_idx-1];
    dp_delFrom[0][t_idx]  = FROM_D;
  }

  // init start cells
  dp_matchScore[0][0] = 0;

  // align

  for(int q_idx = 1; q_idx <= (int)qseq_hp.size(); ++q_idx) { // query
    int q_jump_idx = qseq_hp_previous_nuc[q_idx-1];

    for(unsigned int t_idx = 1; t_idx <= tseq_hp.size(); ++t_idx) { // target

      // horizontal. Preference: D, M, I

      dp_delScore[q_idx][t_idx]     = dp_delScore[q_idx][t_idx-1] - tseq_hp[t_idx-1];
      dp_delFrom[q_idx][t_idx]      = FROM_D;

      if (dp_delScore[q_idx][t_idx] < dp_matchScore[q_idx][t_idx-1] - tseq_hp[t_idx-1]) {
        dp_delScore[q_idx][t_idx]   = dp_matchScore[q_idx][t_idx-1] - tseq_hp[t_idx-1];
        dp_delFrom[q_idx][t_idx]    = FROM_M;
      }
      if (dp_delScore[q_idx][t_idx] < dp_insScore[q_idx][t_idx-1] - tseq_hp[t_idx-1]) {
        dp_delScore[q_idx][t_idx]   = dp_insScore[q_idx][t_idx-1] - tseq_hp[t_idx-1];
        dp_delFrom[q_idx][t_idx]    = FROM_I;
      }

      // vertical
      // four moves:
      // 1. phased from match
      // 2. phased from ins
      // 3. empty from match
      // 4. empty from ins
      // Note: use the NEXT reference base for flow order matching

      dp_insScore[q_idx][t_idx]       = MINOR_INF;
      dp_insFrom[q_idx][t_idx]        = FROM_ME;

      if(t_idx < tseq_hp.size() and q_idx >= 1 and qseq_hp_nuc[q_idx-1] != tseq_hp_nuc[t_idx]) {
        if (dp_insScore[q_idx][t_idx] < dp_delScore[q_idx-1][t_idx] - qseq_hp[q_idx-1]) {
          dp_insScore[q_idx][t_idx]   = dp_delScore[q_idx-1][t_idx] - qseq_hp[q_idx-1];
          dp_insFrom[q_idx][t_idx]    = FROM_ME;
        }
        if (dp_insScore[q_idx][t_idx] < dp_matchScore[q_idx-1][t_idx] - qseq_hp[q_idx-1]) {
          dp_insScore[q_idx][t_idx]   = dp_matchScore[q_idx-1][t_idx] - qseq_hp[q_idx-1];
          dp_insFrom[q_idx][t_idx]    = FROM_ME;
        }
        if (dp_insScore[q_idx][t_idx] < dp_insScore[q_idx-1][t_idx] - qseq_hp[q_idx-1]) {
          dp_insScore[q_idx][t_idx]   = dp_insScore[q_idx-1][t_idx] - qseq_hp[q_idx-1];
          dp_insFrom[q_idx][t_idx]    = FROM_IE;
        }
      }

      if (dp_insScore[q_idx][t_idx]   < dp_matchScore[q_jump_idx][t_idx] - gapSumsI[q_idx-1]) {
        dp_insScore[q_idx][t_idx]     = dp_matchScore[q_jump_idx][t_idx] - gapSumsI[q_idx-1];
        dp_insFrom[q_idx][t_idx]      = FROM_MP;
      }

      if (dp_insScore[q_idx][t_idx]   < dp_insScore[q_jump_idx][t_idx] - gapSumsI[q_idx-1]) {
        dp_insScore[q_idx][t_idx]     = dp_insScore[q_jump_idx][t_idx] - gapSumsI[q_idx-1];
        dp_insFrom[q_idx][t_idx]      = FROM_IP;
      }


      // diagonal

      dp_matchScore[q_idx][t_idx]       = MINOR_INF;
      dp_matchFrom[q_idx][t_idx]        = FROM_S;

      if(qseq_hp_nuc[q_idx-1] == tseq_hp_nuc[t_idx-1]) {
        int delta_hp = (q_idx == 1 or q_idx == (int)qseq_hp.size()) ? 0 : abs(tseq_hp[t_idx-1] - qseq_hp[q_idx-1]);

        // Preference: D, M, I
        dp_matchScore[q_idx][t_idx]     = dp_delScore[q_idx-1][t_idx-1] - delta_hp;
        dp_matchFrom[q_idx][t_idx]      = FROM_D;

        if (dp_matchScore[q_idx][t_idx] < dp_matchScore[q_idx-1][t_idx-1] - delta_hp) {
          dp_matchScore[q_idx][t_idx]   = dp_matchScore[q_idx-1][t_idx-1] - delta_hp;
          dp_matchFrom[q_idx][t_idx]    = FROM_M;
        }

        if (dp_matchScore[q_idx][t_idx] < dp_insScore[q_idx-1][t_idx-1] - delta_hp) {
          dp_matchScore[q_idx][t_idx]   = dp_insScore[q_idx-1][t_idx-1] - delta_hp;
          dp_matchFrom[q_idx][t_idx]    = FROM_I;
        }
        // Start anywhere in tseq
        if(dp_matchScore[q_idx][t_idx]  < -delta_hp and startLocal and q_idx == 1) {
          dp_matchScore[q_idx][t_idx]   = -delta_hp;
          dp_matchFrom[q_idx][t_idx]    = FROM_S;  // From arbitrarily located start point
        }
      }
    }
  }

  // Get best scoring cell
  int best_score = MINOR_INF-1;    // The best alignment score found so far.
  int from_traceback = FROM_S;
  int q_traceback = -1;
  int t_traceback = -1;

  // TODO: want to map the query into a sub-sequence of the target
  // We can end anywhere in the target, but we haven't done the beginning.
  // We also need to return where the start end in the target to update start/end position(s).
  for(unsigned int t_idx = endLocal ? 1 : tseq_hp.size(); t_idx <= tseq_hp.size(); ++t_idx) { // target
    if(best_score <= dp_delScore[qseq_hp.size()][t_idx]) {
      q_traceback = qseq_hp.size();
      t_traceback = t_idx;
      best_score = dp_delScore[qseq_hp.size()][t_idx];
      from_traceback = FROM_D;
    }
    if(best_score <= dp_insScore[qseq_hp.size()][t_idx]) {
      q_traceback = qseq_hp.size();
      t_traceback = t_idx;
      best_score = dp_insScore[qseq_hp.size()][t_idx];
      from_traceback = FROM_I;
    }
    if(best_score <= dp_matchScore[qseq_hp.size()][t_idx]) {
      q_traceback = qseq_hp.size();
      t_traceback = t_idx;
      best_score = dp_matchScore[qseq_hp.size()][t_idx];
      from_traceback = FROM_M;
    }
  }

  // ***** Back tracking

  flowOrder.clear();
  qseq.clear();
  tseq.clear();
  aln_flow_index.clear();
  aln.clear();

  flowOrder.reserve(qseq_hp.size() + tseq_hp.size());
  qseq.reserve(qseq_hp.size() + tseq_hp.size());
  tseq.reserve(qseq_hp.size() + tseq_hp.size());
  aln_flow_index.reserve(qseq_hp.size() + tseq_hp.size());
  aln.reserve(qseq_hp.size() + tseq_hp.size());

  // trace path back
  while(q_traceback > 0) { // qseq flows left

    switch(from_traceback) {
      case FROM_M:
      case FROM_ME:
      case FROM_MP:
        from_traceback = dp_matchFrom.at(q_traceback).at(t_traceback);
        q_traceback--;
        t_traceback--;

        flowOrder.push_back(qseq_hp_nuc.at(q_traceback));
        qseq.push_back(qseq_hp.at(q_traceback));
        tseq.push_back(tseq_hp.at(t_traceback));
        aln_flow_index.push_back(qseq_flow_idx.at(q_traceback));
        aln.push_back((qseq_hp.at(q_traceback) == tseq_hp.at(t_traceback)) ? ALN_MATCH : ALN_MISMATCH);
        break;

      case FROM_I:
      case FROM_IE:
      case FROM_IP:
        from_traceback = dp_insFrom.at(q_traceback).at(t_traceback);

        if(from_traceback == FROM_ME or from_traceback == FROM_IE) {
          q_traceback--;

          flowOrder.push_back(qseq_hp_nuc.at(q_traceback));
          qseq.push_back(qseq_hp.at(q_traceback));
          tseq.push_back(0);
          aln_flow_index.push_back(qseq_flow_idx.at(q_traceback));
          aln.push_back((qseq_hp.at(q_traceback) == 0) ? ALN_MATCH : ALN_MISMATCH);

        } else if(from_traceback == FROM_MP or from_traceback == FROM_IP or from_traceback == FROM_S) {
          int q_jump_idx = qseq_hp_previous_nuc.at(q_traceback-1);
          if(from_traceback == FROM_S)
            q_jump_idx = 0;
          while(q_traceback > q_jump_idx) {
            q_traceback--;

            flowOrder.push_back(qseq_hp_nuc.at(q_traceback));
            qseq.push_back(qseq_hp.at(q_traceback));
            tseq.push_back(0);
            aln_flow_index.push_back(qseq_flow_idx.at(q_traceback));
            aln.push_back(ALN_INS);
          }

        } else {
          //printf("ERROR: Failed check A; from_traceback=%d, q_traceback=%d, t_traceback=%d\n", from_traceback, q_traceback, t_traceback);
          return false;
        }
        break;

      case FROM_D:
        from_traceback = dp_delFrom.at(q_traceback).at(t_traceback);
        t_traceback--;

        flowOrder.push_back(tseq_hp_nuc.at(t_traceback) - 'A' + 'a');
        qseq.push_back(0);
        tseq.push_back(tseq_hp.at(t_traceback));
        aln_flow_index.push_back(-1);
        aln.push_back(ALN_DEL);
        break;

      case FROM_S:
      default:
        //printf("ERROR: Failed check B; from_traceback=%d, q_traceback=%d, t_traceback=%d\n", from_traceback, q_traceback, t_traceback);
        return false;
    }
  }

  int tseqStart = 0;   // The zero-based index in the input tseq where the alignment starts.
  for(int t_idx = 0; t_idx < t_traceback; ++t_idx)
    tseqStart += tseq_hp[t_idx];

  // reverse the arrays tseq, qseq, aln, flowOrder <- because backtracking filled it in reverse!
  for(int q_idx = 0; q_idx < (int)qseq.size()/2;q_idx++) {
    int p = aln_flow_index[q_idx];
    aln_flow_index[q_idx] = aln_flow_index[qseq.size()-q_idx-1];
    aln_flow_index[qseq.size()-q_idx-1] = p;

    int b = qseq[q_idx];
    qseq[q_idx] = qseq[qseq.size()-q_idx-1];
    qseq[qseq.size()-q_idx-1] = b;

    char c = aln[q_idx];
    aln[q_idx] = aln[qseq.size()-q_idx-1];
    aln[qseq.size()-q_idx-1] = c;

    b = tseq[q_idx];
    tseq[q_idx] = tseq[qseq.size()-q_idx-1];
    tseq[qseq.size()-q_idx-1] = b;

    c = flowOrder[q_idx];
    flowOrder[q_idx] = flowOrder[qseq.size()-q_idx-1];
    flowOrder[qseq.size()-q_idx-1] = c;
  }
  return true;
}



// =================================================================================

// Function cheats and generates a flow alignment for a very special case
// Inputs:         tseq_bases : Target bases
//                 qseq_bases : Query bases
//            main_flow_order : Flow order string of run
//          first_useful_flow : start flow for flow alignment ()

bool NullFlowAlignment(
    // Inputs:
    const string&             tseq_bases,
    const string&             qseq_bases,
    const string&             main_flow_order,
    int                       first_useful_flow,
    // Outputs:
    vector<char>&             flowOrder,
    vector<int>&              qseq,
    vector<int>&              tseq,
    vector<int>&              aln_flow_index,
    vector<char>&             aln,
    bool                      debug_output)
{
  // know my output size will be length of flow order
  flowOrder.resize(main_flow_order.size(),'A');
  qseq.resize(main_flow_order.size(),0);
  tseq.resize(main_flow_order.size(),0);
  aln_flow_index.resize(main_flow_order.size(),-1);
  aln.resize(main_flow_order.size(),ALN_MATCH);

  unsigned int q_ndx=0; // first base in query
  for(unsigned int m_flow=0; m_flow<main_flow_order.size();m_flow++){
    aln_flow_index[m_flow]=m_flow;
    aln[m_flow]=ALN_MATCH; // trivially true
    flowOrder[m_flow]=main_flow_order[m_flow];
    // if we still have sequence remaining to consume
    if (q_ndx<qseq_bases.size()){
      bool done = false;
      // consume all sequence matching
      while (!done){
        if (qseq_bases[q_ndx]==main_flow_order[m_flow]){
          qseq[m_flow]++;
          tseq[m_flow]++;  // know that qseq = tseq
          //aln_flow_index[q_ndx]=m_flow;
          q_ndx++;
        } else {
          // bases no longer match so stop counting
          done = true;
        }
        if (q_ndx>=qseq_bases.size())
          done = true;
      }
    } else {
      aln[m_flow]=ALN_INS; // ran out of sequence, skip all these flows when calibrating
    }

  }
  // done! everyone is length main_flow_order.size()
  return(true);
}



// =================================================================================

// Function cheats and generates a flow alignment assuming the base alignment can mostly be relied upon
// Inputs:         tseq_bases : Target bases
//                 qseq_bases : Query bases
//            main_flow_order : Flow order string of run
//          first_useful_flow : start flow for flow alignment ()

bool LightFlowAlignment(
    // Inputs:
    const string&             t_char,
    const string&             q_char,
    const string&             main_flow_order,
    bool match_nonzero, // enforce phase identity
    float strange_gap,  // close gaps containing too many errors
    // Outputs: all same length
    vector<char>&             flowOrder,
    vector<int>&              qseq,
    vector<int>&              tseq,
    vector<int>&              aln_flow_index,
    vector<char>&             aln,
    // synthetic reference for use in simulation
    string &                  synthetic_reference)
{
  int n=main_flow_order.size();
  int nq=q_char.size();
  bool fill_strange=false;
  if (strange_gap>0.0f) fill_strange = true;

  // know my output size will be length of flow order
  flowOrder.resize(n,'A');
  qseq.resize(n,0);
  tseq.resize(n,0);
  aln_flow_index.resize(n,-1);
  aln.resize(n,ALN_MATCH);

  vector<int> q_flow_index;
  q_flow_index.resize(nq,-1); // indexing of >bases<
  vector<bool> qmark;
  vector<bool> tmark;
  qmark.resize(nq,false);
  tmark.resize(nq,false);
  int STARTBASE=0;
  int STARTFLOW=0;
  int LASTFLOW=n-1;
  int LASTBASE=nq-1;
  char OTHER = ALN_INS; // not really, but no appropriate state

  // consume all my query bases and deletion marks
  // consume >matching< target bases, assign to same flows as aligned query bases
  int q_ndx=STARTBASE;
  for (int m_flow=STARTFLOW; m_flow<n; m_flow++){
    aln_flow_index[m_flow]=m_flow;
    aln[m_flow]=ALN_MATCH;
    flowOrder[m_flow]=main_flow_order[m_flow];
    if (q_ndx<nq){
      bool done=false;
      while (!done){
        if (q_char[q_ndx]==main_flow_order[m_flow]){
          qseq[m_flow]++;
          qmark[q_ndx]=true;
          q_flow_index[q_ndx]=m_flow;
          if (q_char[q_ndx]==t_char[q_ndx]){
            tseq[m_flow]++;
            tmark[q_ndx]=true;
          }
          if (t_char[q_ndx]=='-') tmark[q_ndx]=true;  //consumed t.char deletion as an 'add zero' operation
          q_ndx++;
        } else {
          if (q_char[q_ndx]=='-') {
            qmark[q_ndx]=true;  //consume q.char deletion, >does not consume t.char<
            q_ndx++;
          }else done=true; // #q.char not deletion, not equal to flow, so must advance flow to continue
        }
        if (q_ndx>=nq) done=true;
      }
    } else{
      aln[m_flow]=OTHER;
    }
  }



  // now we have processed query bases and matching target bases
  // and forced >matching< target bases into the same flows as the query bases

  // make a helper interval set flanking all the bases
  vector<int> fwd_aln_index;
  vector<int> rev_aln_index;
  fwd_aln_index.resize(nq,-1);
  rev_aln_index.resize(nq,-1);


  int lflow=STARTFLOW;
  for (int t_ndx=STARTBASE; t_ndx<nq; t_ndx++){
    fwd_aln_index[t_ndx]=lflow;
    if (q_flow_index[t_ndx]>-1) lflow=q_flow_index[t_ndx];
  }
  lflow=LASTFLOW;  // last valid flow
  for (int t_ndx=LASTBASE; t_ndx>-1; t_ndx--){
    rev_aln_index[t_ndx]=lflow;
    if (q_flow_index[t_ndx]>-1)
      lflow=q_flow_index[t_ndx];
  }


  // fwd_aln_index is 'last registered flow'
  // rev_aln_index is 'next registered flow'
  OTHER=ALN_DEL; // swap to mark
  lflow=STARTFLOW;
  for (int t_ndx=STARTBASE; t_ndx<nq; t_ndx++){
    if (fwd_aln_index[t_ndx]>lflow) lflow=fwd_aln_index[t_ndx];
    if (!tmark[t_ndx]){ //unconsumed non-deletion t.char
      bool happy=false;
      tmark[t_ndx]=true; // consumed whether happy or unhappy

      // inclusive interval: flows which may match this character
      for (int i_flow=lflow; i_flow<=rev_aln_index[t_ndx]; i_flow++){
        //cout << i_flow << "\t" <<t_ndx <<"\t"<< main_flow_order[i_flow]<<"\t" <<t_char[t_ndx] << "\t" << fwd_aln_index[t_ndx] << "\t" << rev_aln_index[t_ndx] << endl;
        if (main_flow_order[i_flow]==toupper(t_char[t_ndx])){
          tseq[i_flow]++;
          lflow=i_flow;
          happy=true; // found a matching flow to put the character in
          break;
        }
      }

      if (!happy){
        // mark interval as 'incommparable' including >endpoints<
        // either biology has happened or the alignment is erroneous
        // in either case, for calibration purposes, cannot trust values in that interval
        for (int i_flow=lflow; i_flow<=rev_aln_index[t_ndx]; i_flow++){
          aln[i_flow]=OTHER;
        }
        lflow=rev_aln_index[t_ndx];
      }
    }
  }

  OTHER=ALN_INS;
  //every base now counted in both t.char and q.char
  //every flow has been assigned, or marked impossible

  int uflow=LASTFLOW;
  lflow = uflow;
  char tval=ALN_MATCH;// #good until we find the first impossible flow event
  char lchar='-';//  #impossible to match until we have a positive flow
  for (int i_flow=LASTFLOW; i_flow>-1; i_flow--){
    if (aln[i_flow]==ALN_MATCH) aln[i_flow]=tval;  // while 'impossible' keep overwriting matches
    if (tseq[i_flow]>0){
      if (main_flow_order[i_flow]==lchar){ // if matched last incorporating flow, also impossible
        //#impossible sequence
        for (int x_flow=i_flow; x_flow<=lflow; x_flow++) {aln[x_flow]=OTHER;}
        tval=OTHER;
      }
      lchar=main_flow_order[i_flow];
      uflow=i_flow;
      if (qseq[i_flow]>0){
        tval=ALN_MATCH;
        lflow=i_flow; //  #not necessarily correct,but close enough
      }
    } else {
      if (main_flow_order[i_flow]==lchar){
        //#impossible sequence
        for (int x_flow=i_flow; x_flow<=lflow; x_flow++) {aln[x_flow]=OTHER;}
        tval=OTHER;
      }
    }
  }

  // in case we really cannot trust our alignments
  // case: badly aligned segments sometimes contain segments that appear to align, but with many errors XXXBADXXX
  // filling them in produces improved results by omitting high probability errors
  if (fill_strange){
  float ssq=0.0f;
  float gapsize=0.5f; // avoid divide by zero
  float badthresh=strange_gap; // 0.1 seems pretty good for a round number
  float startgap=STARTFLOW;
  //#look for gaps likely due to bad alignments
  for (int iflow=0; iflow<=LASTFLOW; iflow++){
    if (aln[iflow]!=ALN_MATCH){
      //# check to see if in bad gap
     if ((ssq/gapsize)>badthresh){
         //#too many peculiar circumstances between gaps indicating a bad alignment of some sort
         for (int xflow=startgap; xflow<=iflow; xflow++){
           aln[xflow]=OTHER;
         }
      }
      ssq=0.0f; //#reset
      gapsize=0.5f;
      startgap=iflow;
    } else {
      // scale variation by average to crudely account for multiplicative variation
      float delta=(tseq[iflow]-qseq[iflow])/(0.5*(tseq[iflow]+qseq[iflow]+1));
      ssq=ssq+delta*delta;
      gapsize+=1.0f;
    }

  }
  }


  //#all impossible sequence intervals have been marked
  //#generate pseudo-sequence to fill in

  for (int i_flow=STARTFLOW; i_flow<n; i_flow++){
    if (aln[i_flow]!=ALN_MATCH)
      tseq[i_flow]=qseq[i_flow]; //#known sequenceable sequence
  }

  //#fix 0-nonzero changes to make sure phase effects are identical
  //#mark not useful
  if (match_nonzero){
    for (int i_flow=STARTFLOW; i_flow<n; i_flow++){
      if ((tseq[i_flow]>0) & (qseq[i_flow]==0)){
        tseq[i_flow]=0;
        aln[i_flow]=OTHER;
      }
      if ((tseq[i_flow]==0) & (qseq[i_flow]>0)){
        tseq[i_flow]=qseq[i_flow];
        aln[i_flow]=OTHER;
      }
    }
  }

  // last step: retrieve new synthetic 'target' sequence
  synthetic_reference.clear();
  synthetic_reference.reserve(nq);
  for (int i_flow=STARTFLOW; i_flow<n; i_flow++){
    int i_count=tseq[i_flow];
    while (i_count>0){
      synthetic_reference.push_back(main_flow_order[i_flow]);
      i_count--;
    }
  }


  // done! everyone is length main_flow_order.size()
  // flowOrder contains flowOrder (same as main here)
  // qseq contains consistent 'read-as-called' runs per flow
  // tseq contains consistent 'read-as-reference' runs per flow, with minimum modifications for impossibility
  // aln_flow_index contains qseq entry to flow order index mapping
  // aln contains MATCH or OTHER depending on if the flow is incomparable with reference for some reason
  // synthetic reference generates tseq

  return(true);
}
