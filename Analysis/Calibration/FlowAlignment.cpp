/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "FlowAlignment.h"

using namespace std;
using namespace BamTools;


// Helper functions first
bool IsInDelAlignSymbol(char symbol)
{
  return (symbol==ALN_DEL or symbol==ALN_INS);
}

char ReverseComplement (char nuc)
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
}




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
    vector<char>&             aln)
{

  bool startLocal = true;
  bool endLocal = true;
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
    while (*base_ptr == main_flow_order[flow]) {
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

  for(int q_idx = 1; q_idx <= (int)qseq_hp.size(); ++q_idx) {
    int q_jump_idx = qseq_hp_previous_nuc[q_idx-1];
    // vertical
    // only allow phasing from an insertion
    if(0 == q_jump_idx) {
      dp_insScore[q_idx][0] = 0 - gapSumsI[q_idx-1];
      dp_insFrom[q_idx][0] = FROM_IP;
    } else {
      dp_insScore[q_idx][0] = dp_insScore[q_jump_idx][0] - gapSumsI[q_idx-1];
      dp_insFrom[q_idx][0] = FROM_IP;
    }
  }


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
      // 4. empth from ins
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



  int tseqEnd = 0;  // The zero-based index in the input tseq where the alignment ends.
  for(int t_idx = 0; t_idx < t_traceback; ++t_idx)
    tseqEnd += tseq_hp[t_idx];
  tseqEnd--;


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
        from_traceback = dp_matchFrom[q_traceback][t_traceback];
        q_traceback--;
        t_traceback--;

        flowOrder.push_back(qseq_hp_nuc[q_traceback]);
        qseq.push_back(qseq_hp[q_traceback]);
        tseq.push_back(tseq_hp[t_traceback]);
        aln_flow_index.push_back(qseq_flow_idx[q_traceback]);
        aln.push_back((qseq_hp[q_traceback] == tseq_hp[t_traceback]) ? ALN_MATCH : ALN_MISMATCH);
        break;

      case FROM_I:
      case FROM_IE:
      case FROM_IP:
        from_traceback = dp_insFrom[q_traceback][t_traceback];

        if(from_traceback == FROM_ME or from_traceback == FROM_IE) {
          q_traceback--;

          flowOrder.push_back(qseq_hp_nuc[q_traceback]);
          qseq.push_back(qseq_hp[q_traceback]);
          tseq.push_back(0);
          aln_flow_index.push_back(qseq_flow_idx[q_traceback]);
          aln.push_back((qseq_hp[q_traceback] == 0) ? ALN_MATCH : ALN_MISMATCH);

        } else if(from_traceback == FROM_MP or from_traceback == FROM_IP or from_traceback == FROM_S) {
          int q_jump_idx = qseq_hp_previous_nuc[q_traceback-1];
          if(from_traceback == FROM_S)
            q_jump_idx = 0;
          while(q_traceback > q_jump_idx) {
            q_traceback--;

            flowOrder.push_back(qseq_hp_nuc[q_traceback]);
            qseq.push_back(qseq_hp[q_traceback]);
            tseq.push_back(0);
            aln_flow_index.push_back(qseq_flow_idx[q_traceback]);
            aln.push_back(ALN_INS);
          }

        } else {
          //fprintf(stderr, "ERROR: Failed check A\n");
          return false;
        }
        break;

      case FROM_D:
        from_traceback = dp_delFrom[q_traceback][t_traceback];
        t_traceback--;

        flowOrder.push_back(tseq_hp_nuc[t_traceback] - 'A' + 'a');
        qseq.push_back(0);
        tseq.push_back(tseq_hp[t_traceback]);
        aln_flow_index.push_back(-1);
        aln.push_back(ALN_DEL);
        break;

      case FROM_S:
      default:
        //fprintf(stderr, "ERROR: Failed check B\n");
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

