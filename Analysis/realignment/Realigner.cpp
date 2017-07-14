/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     Realigner.h
//! @brief    Perform local realignment of read.

#include "Realigner.h"

#include <ctype.h>
#include <stdio.h>
#include <iostream>


void AlignmentCell::initialize(int init_score) {
  int FROM_NOWHERE = 4;
  int kNotApplicable = -1000000;
  is_match = false;
  best_score = init_score;
  best_path_direction = FROM_NOWHERE;
  scores.assign(FROM_NOWHERE+1, kNotApplicable);
  scores[FROM_NOWHERE] = init_score;
  in_directions.assign(FROM_NOWHERE, FROM_NOWHERE);
}

// -------------------------------------------------------------------

void reverseString(string& S){
  char c;
  for (unsigned int i=0; i<S.size()/2; i++) {
      c = S[i];
      S[i] = S[S.size()-i-1];
      S[S.size()-i-1] = c;
  }
}

// -------------------------------------------------------------------

template <class T>
void reverseVector (vector<T>& myvector) {
  T temp;
  for (unsigned int i=0; i<myvector.size()/2; i++) {
    temp = myvector[i];
    myvector[i] = myvector[myvector.size()-i-1];
    myvector[myvector.size()-i-1] = temp;
  }
}


// ===================================================================


Realigner::Realigner() {
  InitializeRealigner(1000, 50);
}

Realigner::Realigner(unsigned int reserve_size,  unsigned int clipping_size) {
  InitializeRealigner(reserve_size, clipping_size);
}


void Realigner::InitializeRealigner(unsigned int reserve_size, unsigned int clipping_size)
{
  // Set default values
  kMatchScore    =  4;
  kMismatchScore = -6;
  kGapOpen       = -5;
  kGapExtend     = -2;
  debug_ = false;

  start_anywhere_in_ref_ = true;
  stop_anywhere_in_ref_  = true;
  soft_clip_left_        = false;
  soft_clip_right_       = true;
  isForwardStrandRead_   = true;
  verbose_               = false;
  invalid_cigar_in_input = false;
  
  alignment_bandwidth_   = 20;
  q_limit_minus_.reserve(reserve_size);
  q_limit_plus_.reserve(reserve_size);
  clipped_anchors_.cigar_left.reserve(50);
  clipped_anchors_.cigar_right.reserve(50);
  clipped_anchors_.md_left.reserve(50);
  clipped_anchors_.md_right.reserve(50);

  pretty_tseq_.reserve(reserve_size);
  pretty_qseq_.reserve(reserve_size);
  pretty_aln_.reserve(reserve_size);
  q_seq_.reserve(reserve_size);
  t_seq_.reserve(reserve_size);
  DP_matrix.resize(reserve_size);
  for (unsigned int i=0; i<DP_matrix.size(); i++)
    DP_matrix[i].resize(reserve_size);
  
  cr_error = CR_SUCCESS;
}

// -------------------------------------------------------------------

bool Realigner::SetScores(const vector<int> score_vec) {
  if (score_vec.size() != 4)
    return false;
  kMatchScore    =  score_vec[0];
  kMismatchScore =  score_vec[1];
  kGapOpen       =  score_vec[2];
  kGapExtend     =  score_vec[3];
  if (verbose_)
    cout << "Set aligner scores: match " << kMatchScore << ", mismatch " << kMismatchScore
         << ", gap open " << kGapOpen << " gap extend " <<kGapExtend << endl;
  return true;
}

// -------------------------------------------------------------------

void Realigner::SetClipping(int clipping, bool is_forward_strand)
{
  // These settings are set for a forward strand read
  switch (clipping) {
    case 0: // align full strings, no clipping in read or ref
      start_anywhere_in_ref_ = false;
      stop_anywhere_in_ref_  = false;
      soft_clip_left_        = false;
      soft_clip_right_       = false; break;
    case 1: // start or end anywhere in ref
      start_anywhere_in_ref_ = true;
      stop_anywhere_in_ref_  = true;
      soft_clip_left_        = false;
      soft_clip_right_       = false; break;
    case 2: // semi-global + soft-clip bead end of the read
      start_anywhere_in_ref_ = true;
      stop_anywhere_in_ref_  = true;
      soft_clip_left_        = false;
      soft_clip_right_       = true;  break;
    case 3: // semi-global + soft-clip key end of the read
      start_anywhere_in_ref_ = true;
      stop_anywhere_in_ref_  = true;
      soft_clip_left_        = true;
      soft_clip_right_       = false; break;
    case 4: // semi-global + soft-clip both ends of read
      start_anywhere_in_ref_ = true;
      stop_anywhere_in_ref_  = true;
      soft_clip_left_        = true;
      soft_clip_right_       = true;  break;
  }
  // Adjust for read strand
  SetStrand(is_forward_strand);
  if (!is_forward_strand)
    ReverseClipping();

  if (verbose_ and debug_) {
    cout << "Clipping settings for read from the ";
    if (is_forward_strand) cout << "forward ";
    else cout << "reverse ";
    cout << "strand:" << endl
         << "start_anywhere_in_ref_" << start_anywhere_in_ref_ << endl
         << "stop_anywhere_in_ref_" << stop_anywhere_in_ref_ << endl
         << "soft_clip_left_" << soft_clip_left_ << endl
         << "soft_clip_right_" << soft_clip_right_ << endl << endl;
  }
}

// -------------------------------------------------------------------

void Realigner::ReverseClipping() {
  bool temp_start_anywhere_in_ref_ = start_anywhere_in_ref_;
  start_anywhere_in_ref_ = stop_anywhere_in_ref_;
  stop_anywhere_in_ref_  = temp_start_anywhere_in_ref_;

  bool temp_soft_clip_left = soft_clip_left_;
  soft_clip_left_  = soft_clip_right_;
  soft_clip_right_ = temp_soft_clip_left;
}

// -------------------------------------------------------------------

bool Realigner::isMatch(char nuc1, char nuc2)
{
  bool isM = false;
  vector<bool> nuc_ensemble1(4, false);
  vector<bool> nuc_ensemble2(4, false);
  nuc_ensemble1 = getNucMatches(nuc1);
  nuc_ensemble2 = getNucMatches(nuc2);
  for (int i=0; i<4; i++)
    isM = isM || (nuc_ensemble1[i] && nuc_ensemble2[i]);
  return isM;
}

// -------------------------------------------------------------------

vector<bool> Realigner::getNucMatches(char nuc)
{
  vector<bool> nuc_ensemble(4, false);
  nuc = toupper(nuc);

  switch(nuc) {
      case 'A': nuc_ensemble[0] = true; break;
      case 'C': nuc_ensemble[1] = true; break;
      case 'G': nuc_ensemble[2] = true; break;
      case 'T': nuc_ensemble[3] = true; break;
      case 'U': nuc_ensemble[3] = true; break;
      case 'W': nuc_ensemble[0] = true; nuc_ensemble[3] = true; break;
      case 'S': nuc_ensemble[1] = true; nuc_ensemble[2] = true; break;
      case 'M': nuc_ensemble[0] = true; nuc_ensemble[1] = true; break;
      case 'K': nuc_ensemble[2] = true; nuc_ensemble[3] = true; break;
      case 'R': nuc_ensemble[0] = true; nuc_ensemble[2] = true; break;
      case 'Y': nuc_ensemble[1] = true; nuc_ensemble[3] = true; break;
      case 'B': nuc_ensemble[1] = true; nuc_ensemble[2] = true; nuc_ensemble[3] = true; break;
      case 'D': nuc_ensemble[0] = true; nuc_ensemble[2] = true; nuc_ensemble[3] = true; break;
      case 'H': nuc_ensemble[0] = true; nuc_ensemble[1] = true; nuc_ensemble[3] = true; break;
      case 'I': nuc_ensemble[0] = true; nuc_ensemble[1] = true; nuc_ensemble[3] = true; break;
      case 'V': nuc_ensemble[0] = true; nuc_ensemble[1] = true; nuc_ensemble[2] = true; break;
      case 'N': nuc_ensemble.assign(4, true); break;
  }
  return nuc_ensemble;
}

// -------------------------------------------------------------------

void Realigner::SetSequences(const string& q_seq, const string& t_seq, const string& aln_path, const bool isForward)
{
  if (debug_ and verbose_)
    cout << "Hello from SetSequences." << endl;

  // If we have an empty aln_path, and alignment_bandwidth_>0 we align along the diagonal
  // Empty aln_path and alignment_bandwidth_==0 evaluates the whole DP matrix
  if (aln_path.length()==0 and alignment_bandwidth_>0){
    create_dummy_pretty(t_seq.length(), q_seq.length());
  }
  else
    pretty_aln_ = aln_path;

  // We align all sequences in forward direction
  // but have to worry about clipping the correct end of the alignment
  q_seq_ = q_seq;
  t_seq_ = t_seq;
  if (isForwardStrandRead_ != isForward) {
    ReverseClipping();
    isForwardStrandRead_ = isForward;
  }

  // Resize DP_matrix if necessary
  if (t_seq_.size() > DP_matrix.size()-1)
    DP_matrix.resize(t_seq_.size()+1);
  // initialize first row and column of DP matrix
  for (unsigned int t_idx=0; t_idx<t_seq_.size()+1; t_idx++) {
    if (DP_matrix[t_idx].size() < q_seq_.size() +1)
      DP_matrix[t_idx].resize(q_seq_.size()+1);
    DP_matrix[t_idx][0].initialize(0);
  }
  for (unsigned int q_idx=0; q_idx<q_seq_.size()+1; q_idx++)
    DP_matrix[0][q_idx].initialize(0);
  
  if (debug_ and verbose_)
    cout << "Successfully set sequences." << endl;
}

// -------------------------------------------------------------------
// For tubed alignment to work without a prior alignment input
// Create a pseudo diagonal pretty alignment string for initialization

void Realigner::create_dummy_pretty(unsigned int target_length, unsigned int query_length)
{
  pretty_aln_.assign(max(target_length, query_length), '|');
  unsigned int plen = pretty_aln_.length();
  char indel = '-';
  int n_indels = target_length - query_length;

  if (n_indels==0)
    return;
  else if (n_indels<0) {
    indel = '+';
    n_indels = -n_indels;
  }
  int interval = plen / n_indels;
  unsigned int pos = (plen - ((n_indels-1)*interval))/2;
  int count = 0;

  while (count<n_indels and pos < plen){
    pretty_aln_.at(pos) = indel;
    ++count;
    pos += interval;
  }
}

// -------------------------------------------------------------------


bool Realigner::ComputeTubedAlignmentBoundaries()
{
    // Compute boundaries for tubed alignment around previously found alignment
  q_limit_minus_.assign(t_seq_.size()+1, 0);
  q_limit_plus_.assign(t_seq_.size()+1, q_seq_.size()+1);
  
  int center_point_q = 0;
  int center_point_t = 0;
  
  for (unsigned int idx=0; idx < pretty_aln_.size(); idx++) {
    
    int q_idx, t_idx;
    switch (pretty_aln_[idx]) {
      case ('|') :
      case (' ') :
        center_point_q++;
        center_point_t++;  break;
      case ('-') :
        center_point_t++;  break;
      case ('+') :
        center_point_q++;  break;
    }
    
    // Check upper diagonal point
    q_idx = center_point_q + (int)alignment_bandwidth_;
    t_idx = center_point_t - (int)alignment_bandwidth_ -1;
    if (t_idx >= 0 and t_idx <= (int)t_seq_.size()) {
      if (q_idx < (int)q_limit_plus_[t_idx])
        q_limit_plus_[t_idx] = (unsigned int)q_idx;  // exclusive limit
      if (q_idx <= (int)q_seq_.size())
        DP_matrix[t_idx][q_idx].initialize(kNotApplicable);
    }
    // Check lower diagonal point
    q_idx = center_point_q - (int)alignment_bandwidth_ - 1;
    t_idx = center_point_t + (int)alignment_bandwidth_;
    if (q_idx >= 0 and q_idx <= (int)q_seq_.size()) {
      if (t_idx <= (int)t_seq_.size()) {
        q_limit_minus_[t_idx] = (unsigned int)(q_idx+1); // inclusive limit
        DP_matrix[t_idx][q_idx].initialize(kNotApplicable);
      }
    }
  }
  // Sanity check whether a correct path was followed to create tube
  if (center_point_t != (int)t_seq_.size() or center_point_q != (int)q_seq_.size()) {
    if (verbose_)
      cout << "Error: An invalid alignment path was used to create the tube." << endl
           << "        Ending coordinates: (" << center_point_t << "," << center_point_q << ")" << endl
           << "        Sequence sizes:     (" << t_seq_.size() << "," << q_seq_.size() << ")" << endl;
    //verbose_ = true;
    return false;
  } else {
    if (debug_ and verbose_) {
      cout << "Tube lower limits for each row: " << endl;
      for (unsigned int i=0; i<q_limit_minus_.size(); i++)
        cout << q_limit_minus_[i] << " ";
      cout << endl << "Tube upper limits for each row: " << endl;
      for (unsigned int i=0; i<q_limit_plus_.size(); i++)
        cout << q_limit_plus_[i] << " ";
      cout << endl << endl;
    }
    return true;
  }
}

// -------------------------------------------------------------------


bool Realigner::computeSWalignment(vector<CigarOp>& CigarData, vector<MDelement>& MD_data,
                       unsigned int& start_pos_update) {

  // Compute boundaries for tubed alignment around previously found alignment
  if (!ComputeTubedAlignmentBoundaries())
    return false;

  // Path ordering creates left aligned InDels
  vector<int> insertion_path_ordering(3);
  vector<int> deletion_path_ordering(3);
  insertion_path_ordering[0] = FROM_I;
  insertion_path_ordering[1] = FROM_MATCH;
  insertion_path_ordering[2] = FROM_MISM;
  deletion_path_ordering[0] = FROM_D;
  deletion_path_ordering[1] = FROM_MATCH;
  deletion_path_ordering[2] = FROM_MISM;

  // --- Compute first row  and column of the matrix
  // First row: moving horizontally for insertions
  if (!soft_clip_left_) {
    DP_matrix[0][1].best_path_direction = FROM_I;
    DP_matrix[0][1].best_score = kGapOpen;
    DP_matrix[0][1].scores[FROM_I] = kGapOpen;
    DP_matrix[0][1].scores[FROM_NOWHERE] = kNotApplicable;
    for (unsigned int q_idx=2; q_idx<q_limit_plus_[0]; q_idx++) {
      DP_matrix[0][q_idx].in_directions[FROM_I] = FROM_I;
      DP_matrix[0][q_idx].scores[FROM_NOWHERE] = kNotApplicable;
      DP_matrix[0][q_idx].scores[FROM_I] = DP_matrix[0][q_idx-1].best_score + kGapExtend;
      DP_matrix[0][q_idx].best_path_direction = FROM_I;
      DP_matrix[0][q_idx].best_score = DP_matrix[0][q_idx-1].best_score + kGapExtend;
    }
  }

  if (!start_anywhere_in_ref_) {
    // First column: moving vertically for deletions
    DP_matrix[1][0].best_path_direction = FROM_D;
    DP_matrix[1][0].best_score = kGapOpen;
    DP_matrix[1][0].scores[FROM_D] = kGapOpen;
    DP_matrix[1][0].scores[FROM_NOWHERE] = kNotApplicable;
    unsigned int t = 2;
    while (t < q_limit_minus_.size() and q_limit_minus_[t] == 0) {
      DP_matrix[t][0].in_directions[FROM_D] = FROM_D;
      DP_matrix[t][0].scores[FROM_NOWHERE] = kNotApplicable;
      DP_matrix[t][0].scores[FROM_D] = DP_matrix[t-1][0].best_score + kGapExtend;
      DP_matrix[t][0].best_path_direction = FROM_D;
      DP_matrix[t][0].best_score = DP_matrix[t-1][0].best_score + kGapExtend;
      t++;
    }
  }

  // ------ Main alignment loop ------
  vector<int>   temp_scores(FROM_NOWHERE);
  vector<int>   highest_score_cell(2, 0);

  for (unsigned int t_idx=1; t_idx<t_seq_.size()+1; t_idx++) {

    for (unsigned int q_idx=q_limit_minus_[t_idx]; q_idx<q_limit_plus_[t_idx]; q_idx++) {

      if (q_idx == 0)
        continue;

      // Scoring for Match; Mismatch / Insertion / Deletion;
      // work around a c++11 issue
      int kNotApplicable_tmp = kNotApplicable;
      int FROM_NOWHERE_tmp = FROM_NOWHERE;
      DP_matrix[t_idx][q_idx].scores.assign(FROM_NOWHERE_tmp+1, kNotApplicable_tmp); //C++11 issue
      DP_matrix[t_idx][q_idx].in_directions.assign(FROM_NOWHERE_tmp, FROM_NOWHERE_tmp); //C++11 issue
      if (soft_clip_left_)
        DP_matrix[t_idx][q_idx].scores[FROM_NOWHERE] = 0;

      // 1) - Match / Mismatch Score
      DP_matrix[t_idx][q_idx].is_match = isMatch(q_seq_[q_idx-1], t_seq_[t_idx-1]);
      if (DP_matrix[t_idx][q_idx].is_match) {
        DP_matrix[t_idx][q_idx].in_directions[FROM_MATCH] = DP_matrix[t_idx-1][q_idx-1].best_path_direction;
        DP_matrix[t_idx][q_idx].scores[FROM_MATCH] = DP_matrix[t_idx-1][q_idx-1].best_score + kMatchScore;
      } else {
        DP_matrix[t_idx][q_idx].in_directions[FROM_MISM] = DP_matrix[t_idx-1][q_idx-1].best_path_direction;
        DP_matrix[t_idx][q_idx].scores[FROM_MISM] = DP_matrix[t_idx-1][q_idx-1].best_score + kMismatchScore;
      }

      // 2) - Insertion Score
      temp_scores.assign(FROM_NOWHERE, kNotApplicable_tmp);
      temp_scores[FROM_MATCH] = DP_matrix[t_idx][q_idx-1].scores[FROM_MATCH] + kGapOpen;
      temp_scores[FROM_I] = DP_matrix[t_idx][q_idx-1].scores[FROM_I] + kGapExtend;
      temp_scores[FROM_MISM] = DP_matrix[t_idx][q_idx-1].scores[FROM_MISM] + kGapOpen;
      DP_matrix[t_idx][q_idx].scores[FROM_I] = kNotApplicable;
      DP_matrix[t_idx][q_idx].in_directions[FROM_I] = FROM_NOWHERE;
      for (int i=0; i<(int)insertion_path_ordering.size(); i++) {
        if (temp_scores[insertion_path_ordering[i]] > DP_matrix[t_idx][q_idx].scores[FROM_I]) {
          DP_matrix[t_idx][q_idx].scores[FROM_I] = temp_scores[insertion_path_ordering[i]];
          DP_matrix[t_idx][q_idx].in_directions[FROM_I] = insertion_path_ordering[i];
        }
      }

      // 3) - Deletion Score
      temp_scores.assign(FROM_NOWHERE, kNotApplicable_tmp);
      temp_scores[FROM_MATCH] = DP_matrix[t_idx-1][q_idx].scores[FROM_MATCH] + kGapOpen;
      temp_scores[FROM_D] = DP_matrix[t_idx-1][q_idx].scores[FROM_D] + kGapExtend;
      temp_scores[FROM_MISM] = DP_matrix[t_idx-1][q_idx].scores[FROM_MISM] + kGapOpen;
      DP_matrix[t_idx][q_idx].scores[FROM_D] = kNotApplicable;
      DP_matrix[t_idx][q_idx].in_directions[FROM_D] = FROM_NOWHERE;
      for (int i=0; i<(int)deletion_path_ordering.size(); i++) {
        if (temp_scores[deletion_path_ordering[i]] > DP_matrix[t_idx][q_idx].scores[FROM_D]) {
          DP_matrix[t_idx][q_idx].scores[FROM_D] = temp_scores[deletion_path_ordering[i]];
          DP_matrix[t_idx][q_idx].in_directions[FROM_D] = deletion_path_ordering[i];
        }
      }

      // Choose best move for this cell
      DP_matrix[t_idx][q_idx].best_score = kNotApplicable-1;
      DP_matrix[t_idx][q_idx].best_path_direction = FROM_NOWHERE;
      for (unsigned int iMove=0; iMove<DP_matrix[t_idx][q_idx].scores.size(); iMove++) {
        if (DP_matrix[t_idx][q_idx].scores[iMove] > DP_matrix[t_idx][q_idx].best_score) {
          DP_matrix[t_idx][q_idx].best_score = DP_matrix[t_idx][q_idx].scores[iMove];
          DP_matrix[t_idx][q_idx].best_path_direction = iMove;
        }
      }

      // Clipping settings determine where we search for the best scoring cell to stop aligning
      bool valid_t_idx = stop_anywhere_in_ref_ or (t_idx == t_seq_.size());
      bool valid_q_idx = soft_clip_right_ or (q_idx == q_seq_.size());
      bool investigate_highscore = valid_t_idx and valid_q_idx;

      if (investigate_highscore and DP_matrix[t_idx][q_idx].best_score
               > DP_matrix[highest_score_cell[0]][highest_score_cell[1]].best_score) {
        highest_score_cell[0] = t_idx;
        highest_score_cell[1] = q_idx;
      }

    }
  }
  // ------- end alignment matrix loop ------

  // Force full string alignment if desired, no matter what the score is.
  if (!stop_anywhere_in_ref_ and !soft_clip_right_) {
    highest_score_cell[0] = t_seq_.size();
    highest_score_cell[1] = q_seq_.size();
  }

  // Backtrack alignment in dynamic programming matrix, generate cigar string / MD tag
  backtrackAlignment(highest_score_cell[0], highest_score_cell[1], CigarData, MD_data, start_pos_update);
  return true;
}

// -------------------------------------------------------------------

void Realigner::backtrackAlignment(unsigned int t_idx, unsigned int q_idx,
            vector<CigarOp>& CigarData, vector<MDelement>& MD_data, unsigned int& start_pos_update) {

  pretty_tseq_.clear();
  pretty_qseq_.clear();
  pretty_aln_.clear();
  start_pos_update = t_seq_.size()+2;

  CigarOp current_cigar_element;
  CigarData.clear();
  // Determine soft clipped end of alignment
  if (q_idx < q_seq_.size()) {
    current_cigar_element.Type = 'S';
    current_cigar_element.Length = q_seq_.size() - q_idx;
    CigarData.push_back(current_cigar_element);
  }
  MD_data.clear();
  MDelement current_MD_element;
  current_MD_element.Type = '=';
  current_MD_element.Length = 0;

  int current_move = DP_matrix[t_idx][q_idx].best_path_direction;
  int next_move = FROM_NOWHERE;
  int last_move = -1;

  while (current_move != FROM_NOWHERE) {
    switch (current_move) {

      case FROM_MATCH: // Match
        pretty_tseq_.push_back(t_seq_[t_idx-1]);
        pretty_qseq_.push_back(q_seq_[q_idx-1]);
        addMDelement(FROM_MATCH, DP_matrix[t_idx][q_idx].is_match, last_move, t_idx, current_MD_element, MD_data);
        addCigarElement(FROM_MATCH, last_move, current_cigar_element, CigarData);
        pretty_aln_.push_back(ALN_MATCH);
        if (t_idx <= start_pos_update)
          start_pos_update = t_idx-1;
        next_move = DP_matrix[t_idx][q_idx].in_directions[FROM_MATCH];
        t_idx--;
        q_idx--;
        break;

      case FROM_MISM: // Mismatch
        pretty_tseq_.push_back(t_seq_[t_idx-1]);
        pretty_qseq_.push_back(q_seq_[q_idx-1]);
        addMDelement(FROM_MATCH, DP_matrix[t_idx][q_idx].is_match, last_move, t_idx, current_MD_element, MD_data);
        addCigarElement(FROM_MATCH, last_move, current_cigar_element, CigarData);
        pretty_aln_.push_back(ALN_MISMATCH);
        if (t_idx <= start_pos_update)
          start_pos_update = t_idx-1;
        next_move = DP_matrix[t_idx][q_idx].in_directions[FROM_MISM];
        t_idx--;
        q_idx--;
        break;

      case FROM_I: // Insertion
        pretty_aln_.push_back(ALN_INS);
        pretty_tseq_.push_back(ALN_INS);
        pretty_qseq_.push_back(q_seq_[q_idx-1]);
        addMDelement(FROM_I, false, last_move, t_idx, current_MD_element, MD_data);
        addCigarElement(FROM_I, last_move, current_cigar_element, CigarData);
        next_move = DP_matrix[t_idx][q_idx].in_directions[FROM_I];
        q_idx--;
        break;

      case FROM_D: // Deletion
        pretty_aln_.push_back(ALN_DEL);
        pretty_qseq_.push_back(ALN_DEL);
        pretty_tseq_.push_back(t_seq_[t_idx-1]);
        addMDelement(FROM_D, false, last_move, t_idx, current_MD_element, MD_data);
        addCigarElement(FROM_D, last_move, current_cigar_element, CigarData);
        next_move = DP_matrix[t_idx][q_idx].in_directions[FROM_D];
        t_idx--;
        break;
    }
    last_move = current_move;
    current_move = next_move;
    if (verbose_ and debug_) {
      cout << "Added: " << PrintAlignType(last_move) << " Next: " << PrintAlignType(current_move)
           << " at (" << t_idx << ", " << q_idx
           << ") Score: " << DP_matrix[t_idx][q_idx].scores[current_move] << endl;
    }
  }
  MD_data.push_back(current_MD_element);
  // Add a match of length zero to MD vector if it ends with a deletion or a snp
  if (MD_data[MD_data.size()-1].Length < 0) {
    current_MD_element.Type = '=';
    current_MD_element.Length  = 0;
    MD_data.push_back(current_MD_element);
  }
  
  if (last_move >= 0)
    CigarData.push_back(current_cigar_element);
  // Add soft clipped beginning to cigar string
  if (q_idx > 0) {
    current_cigar_element.Type = 'S';
    current_cigar_element.Length = q_idx;
    CigarData.push_back(current_cigar_element);
  }

  // reverse alignment strings because backtrack filled them out in reverse direction
  reverseString(pretty_qseq_);
  reverseString(pretty_tseq_);
  reverseString(pretty_aln_);
  reverseVector<CigarOp>(CigarData);
  reverseVector<MDelement>(MD_data);
  for (unsigned int i=0; i<MD_data.size(); i++) {
    if (MD_data[i].Type.size() > 1)
      reverseString(MD_data[i].Type);
  }

  if (verbose_) {
    cout << "The newly computed alignments are (query Seq., alignment, target Seq.):" << endl
         << pretty_qseq_ << endl << pretty_aln_ << endl << pretty_tseq_ << endl;
  }
}

// -------------------------------------------------------------------

void Realigner::PrintScores() const {
  printf("Cell scores:\n");
  for (unsigned int t_idx=0; t_idx<t_seq_.size()+1; t_idx++) {
    for (unsigned int q_idx=0; q_idx<q_seq_.size()+1; q_idx++)
      printf("%d\t", DP_matrix[t_idx][q_idx].best_score);
    printf("\n");
  }
  printf("Path directions:\n");
    for (unsigned int t_idx=0; t_idx<t_seq_.size()+1; t_idx++) {
      for (unsigned int q_idx=0; q_idx<q_seq_.size()+1; q_idx++)
        printf("%d\t", DP_matrix[t_idx][q_idx].best_path_direction);
      printf("\n");
    }
}

string Realigner::PrintAlignType(int align_type) {
  string type;
  switch (align_type) {
    case FROM_MATCH:   type="    match"; break;
    case FROM_MISM:    type=" mismatch"; break;
    case FROM_I:       type="insertion"; break;
    case FROM_D:       type=" deletion"; break;
    case FROM_NOWHERE: type="  nothing"; break;
    default: type="ERROR!";
  }
  return type;
}

// -------------------------------------------------------------------


char Realigner::NucComplement (char nuc) const
{
  switch(nuc) {
    case ('A') : return 'T';
    case ('C') : return 'G';
    case ('G') : return 'C';
    case ('T') : return 'A';
    case ('a') : return 't';
    case ('c') : return 'g';
    case ('g') : return 'c';
    case ('t') : return 'a';

    default:  return nuc; // e.g. 'N' and '-' handled by default
  }
}

// -------------------------------------------------------------------

string Realigner::ComplementSequence(string seq, bool reverse) const
{

  if (reverse) {
    char c;
    int forward_idx = 0;
    int backward_idx = seq.size()-1;
    while (forward_idx < backward_idx) {
      c = seq[forward_idx];
      seq[forward_idx]  = NucComplement(seq[backward_idx]);
      seq[backward_idx] = NucComplement(c);
      forward_idx++;
      backward_idx--;
    }
    if (forward_idx == backward_idx)
      seq[forward_idx] = NucComplement(seq[forward_idx]);
  } else {
    for (unsigned int i=0; i<seq.size(); i++)
      seq[i] = NucComplement(seq[i]);
  }
  return seq;
}


// -------------------------------------------------------------------

void Realigner::addCigarElement(int align_type, int last_move,
		CigarOp& current_cigar_element, vector<CigarOp>& CigarData) {

  if (last_move == FROM_MISM)
    last_move = FROM_MATCH;
  if (last_move == align_type)
    current_cigar_element.Length++;
  else {
    if (last_move >= 0) {
      CigarData.push_back(current_cigar_element);
      if (verbose_ and debug_)
        cout << "Added cigar element " << current_cigar_element.Length << current_cigar_element.Type << endl;
    }
    current_cigar_element.Length = 1;
    switch (align_type) {
      case FROM_MATCH: current_cigar_element.Type = 'M'; break;
      case FROM_I:     current_cigar_element.Type = 'I'; break;
      case FROM_D:     current_cigar_element.Type = 'D'; break;
      default :
    	  cout << "Error in addCigarElement; align_type = " << align_type << " Last move = " << last_move << endl;
        break;
    }
  }
}

// -------------------------------------------------------------------

void Realigner::addMDelement(int align_type, bool is_match, int last_move, unsigned int t_idx,
		MDelement& current_MD_element, vector<MDelement>& MD_data) {

  if (last_move == FROM_D) {
    if (align_type == FROM_D)
      current_MD_element.Type += t_seq_[t_idx-1];
    else {
      if (MD_data.size() == 0) {
        MDelement zero_element;
        zero_element.Type = '=';
      }
      MD_data.push_back(current_MD_element);
      current_MD_element.Type = '=';
      current_MD_element.Length = 0;
    }
  }
  if (current_MD_element.Type[0] == '=') {
    if ((align_type == FROM_MATCH and !is_match) or align_type == FROM_D) {
      MD_data.push_back(current_MD_element);
      current_MD_element.Type = t_seq_[t_idx-1];
      if (align_type == FROM_MATCH and !is_match) {
        current_MD_element.Length = -FROM_I;
        MD_data.push_back(current_MD_element);
        current_MD_element.Type = '=';
        current_MD_element.Length = 0;
      }
      else
        current_MD_element.Length = -FROM_D;
    } else if (align_type == FROM_MATCH and is_match)
      current_MD_element.Length++;
  }
}

// -------------------------------------------------------------------

string Realigner::GetMDstring(vector<MDelement>& MD_data) const
{
  ostringstream ss;
  for (unsigned int i=0; i<MD_data.size(); i++) {
    if (MD_data[i].Type[0] == '=')
      ss << MD_data[i].Length;
    else if (MD_data[i].Length == -FROM_D)
      ss << '^' << MD_data[i].Type;
    else
      ss << MD_data[i].Type;
  }
  return ss.str();
}

// -------------------------------------------------------------------

bool Realigner::addClippedBasesToTags(vector<CigarOp>& cigar_data, vector<MDelement>& MD_data, unsigned int nr_read_bases) const
{

  // Restore left end of cigar string incl. original soft clips
  vector<CigarOp>::const_iterator cigar_it = (clipped_anchors_.cigar_left.end()-1);
  if (cigar_it->Type == cigar_data.begin()->Type) {
    cigar_data.begin()->Length += cigar_it->Length;
    --cigar_it;
  } else if (cigar_data.begin()->Type == 'S') {
	  if (verbose_)
	    cout << "Error, invalid cigar: Soft clipping occurred after left anchor!" << endl;
	  return false;
  }
  while (cigar_it > clipped_anchors_.cigar_left.begin()) {
	if (cigar_it->Length > 0)
      cigar_data.insert(cigar_data.begin(), *cigar_it);
    --cigar_it;
  }
  if (cigar_it == clipped_anchors_.cigar_left.begin() and cigar_it->Length > 0)
    cigar_data.insert(cigar_data.begin(), *cigar_it);

  // Restore right end of cigar string incl. original soft clips
  cigar_it = clipped_anchors_.cigar_right.end()-1;
  if (cigar_it->Type == (cigar_data.end()-1)->Type) {
    (cigar_data.end()-1)->Length += cigar_it->Length;
    --cigar_it;
  } else if ((cigar_data.end()-1)->Type == 'S') {
	  if (verbose_)
	    cout << "Error, invalid cigar: Soft clipping occurred before right anchor!" << endl;
	  return false;
  }
  while (cigar_it > clipped_anchors_.cigar_right.begin()) {
	if (cigar_it->Length > 0)
      cigar_data.push_back(*cigar_it);
    --cigar_it;
  }
  if (cigar_it == clipped_anchors_.cigar_right.begin() and cigar_it->Length > 0)
    cigar_data.push_back(*cigar_it);

  // Restore left end of MD tag
  vector<MDelement>::const_iterator md_it = clipped_anchors_.md_left.end()-1;
  if (clipped_anchors_.md_left.size() > 0) {
    if (md_it->Type[0] != '=') {
      if (verbose_)
        cout << "Error, invalid md tag: Left clipping MD does not end with a match field!" << endl;
      return false;
    }
    MD_data.begin()->Length += md_it->Length;
    --md_it;
    while (md_it != (clipped_anchors_.md_left.begin()-1)) {
      MD_data.insert(MD_data.begin(), *md_it);
      --md_it;
    }
  }

  // Restore right end of MD tag
  if (clipped_anchors_.md_right.size() > 0) {
    md_it = clipped_anchors_.md_right.end()-1;
    if (md_it->Type[0] != '=') {
      if (verbose_)
        cout << "Error, invalid md tag: Right clipping MD does not end with a match field!" << endl;
      return false;
    }
    (MD_data.end()-1)->Length += md_it->Length;
    --md_it;
    while (md_it != (clipped_anchors_.md_right.begin()-1)) {
      MD_data.push_back(*md_it);
      --md_it;
    }
  }

  // Sanity check for newly created tag:
  unsigned int nr_bases = 0;
  for (cigar_it = cigar_data.begin(); cigar_it != cigar_data.end(); ++cigar_it) {
    if (cigar_it->Type == 'S' or cigar_it->Type == 'M' or cigar_it->Type == 'I' or cigar_it->Type == '=' or cigar_it->Type == 'X')
      nr_bases += cigar_it->Length;
  }
  if (nr_bases != nr_read_bases) {
	if (verbose_)
      cout << "Warning: generated an erroneous cigar string. Not updating alignment." << endl;
    return false;
  }
  if (verbose_){
    cout << "New cigar tag:";
    for (vector<CigarOp>::const_iterator cigar = cigar_data.begin(); cigar != cigar_data.end(); ++cigar)
      cout << cigar->Length << cigar->Type;
    cout << endl << "New MD tag   : " << GetMDstring(MD_data) << endl;
  }

  return true;
}

// -------------------------------------------------------------------
// This function takes the weird behavior of tmap into account that
// reads may start with a deletion

int Realigner::updateReadPosition(const vector<CigarOp>& original_cigar,
		const int& start_position_shift, const int& org_position) const
{
  // Count the number deletions before the first match
  int new_position, nr_deletions = 0;

  vector<CigarOp>::const_iterator cigar_it = original_cigar.begin();
  bool is_match = cigar_it->Type == 'M' or cigar_it->Type == '=' or cigar_it->Type == 'X';
  while (!is_match) {
    if (cigar_it->Type == 'D')
      nr_deletions += (int)cigar_it->Length;
    ++cigar_it;
    if (cigar_it != original_cigar.end())
      is_match = cigar_it->Type == 'M' or cigar_it->Type == '=' or cigar_it->Type == 'X';
    else
      break;
  }

  new_position = org_position - nr_deletions + start_position_shift;
  return new_position;
}

// -------------------------------------------------------------------


bool Realigner::CreateRefFromQueryBases(
    // Inputs:
    const string&                algn_query_bases,
    const vector<CigarOp>&       algn_cigar_data,
    const string&                md_tag,
    const bool                   clip_anchors)
{

  // Step 1. Generate reference sequence based on QueryBases and Cigar alone
  if (verbose_) {
    cout << "Original Cigar : ";
    for (vector<CigarOp>::const_iterator cigar = algn_cigar_data.begin(); cigar != algn_cigar_data.end(); ++cigar)
      cout << cigar->Length << cigar->Type;
    cout << endl << "Original MD tag: " << md_tag << endl;
  }

  cr_error = CR_SUCCESS;
  // Initialize variables
  pretty_tseq_.clear();
  pretty_qseq_.clear();
  pretty_aln_.clear();
  q_seq_.clear();
  t_seq_.clear();

  clipped_anchors_.cigar_left.resize(1);
  clipped_anchors_.cigar_left[0].Type = 'S';
  clipped_anchors_.cigar_left[0].Length = 0;
  clipped_anchors_.cigar_right.resize(1);
  clipped_anchors_.cigar_right[0].Type = 'S';
  clipped_anchors_.cigar_right[0].Length = 0;
  clipped_anchors_.md_left.clear();
  clipped_anchors_.md_right.clear();

  const char *read_ptr = algn_query_bases.c_str();


  for (vector<CigarOp>::const_iterator cigar = algn_cigar_data.begin(); cigar != algn_cigar_data.end(); ++cigar) {
    switch (cigar->Type) {
      case ('M') :
      case ('=') :
      case ('X') :
        t_seq_.append(read_ptr, cigar->Length);
        q_seq_.append(read_ptr, cigar->Length);
        pretty_tseq_.append(read_ptr, cigar->Length);
        pretty_qseq_.append(read_ptr, cigar->Length);
        pretty_aln_.append(cigar->Length, '|');
        read_ptr += cigar->Length; break;

      case ('I') :
        q_seq_.append(read_ptr, cigar->Length);
        pretty_tseq_.append(cigar->Length,'-');
        pretty_qseq_.append(read_ptr, cigar->Length);
        pretty_aln_.append(cigar->Length, '+');
        read_ptr += cigar->Length; break;

      case ('S') :
        read_ptr += cigar->Length;
        if (cigar == algn_cigar_data.begin() or (cigar == algn_cigar_data.begin()+1 and (algn_cigar_data.begin())->Type == 'H'))
          clipped_anchors_.cigar_left[0].Length = cigar->Length;
        else if (cigar == algn_cigar_data.end()-1 or (cigar ==algn_cigar_data.end()-2 and (algn_cigar_data.end()-1)->Type == 'H'))
          clipped_anchors_.cigar_right[0].Length = cigar->Length;
        else {
          if (verbose_)
            cout << "Error: invalid cigar string with soft clipped bases in the middle in the input." << endl;
	  cr_error = CR_ERR_RECREATE_REF;
          return false;
        }
        break;

      case ('D') :
      case ('P') :
      case ('N') :
        t_seq_.append(cigar->Length, '-');
        pretty_tseq_.append(cigar->Length,'-');
        pretty_qseq_.append(cigar->Length,'-');
        pretty_aln_.append(cigar->Length, '-');
        break;
    }
  }

  // Step 2: Further patch the sequence based on MD tag

  unsigned int t_idx = 0;
  unsigned int md_idx = 0;
  unsigned int pretty_idx = 0;
  int item_length = 0;

  while (md_idx < md_tag.length()) {

    if (md_tag.at(md_idx) >= '0' and md_tag.at(md_idx) <='9') {  // it's a match
      item_length = 0;
      for (; md_idx < md_tag.length() and md_tag.at(md_idx) >= '0' and md_tag.at(md_idx) <= '9'; ++md_idx)
        item_length = 10*item_length + md_tag.at(md_idx) - '0';
      t_idx += item_length;
      while (pretty_idx < pretty_aln_.length() and (item_length > 0 or pretty_aln_.at(pretty_idx) == '+')) {
        if (pretty_aln_[pretty_idx] != '+')
          --item_length;
        ++pretty_idx;
      }
    }
    else {
      bool is_deletion = false;
      if (md_tag.at(md_idx) == '^') {                     // Its a deletion or substitution
        md_idx++;
        is_deletion = true;
      }
      while (t_idx < t_seq_.length() and md_idx < md_tag.length() and pretty_idx < pretty_tseq_.length() and
              md_tag.at(md_idx) >= 'A' and md_tag.at(md_idx) <= 'Z') {
        if (pretty_aln_[pretty_idx] == '|') {
          if (is_deletion) {
            invalid_cigar_in_input = true;
	    cr_error = CR_ERR_RECREATE_REF;
            return false;
          }
          pretty_aln_[pretty_idx] = ' ';
        }
        pretty_tseq_[pretty_idx++] = md_tag.at(md_idx) + 'a' - 'A';
        t_seq_.at(t_idx) = md_tag.at(md_idx);
        ++t_idx;
        ++md_idx;
      }
    }

    // Checks if Cigar and MD tag are consistent with each other
    bool invalid_pair = item_length > 0;
    invalid_pair = invalid_pair or (t_idx > t_seq_.length());
    invalid_pair = invalid_pair or (pretty_idx == pretty_aln_.length() and md_idx < md_tag.length()-1);
    invalid_pair = invalid_pair or (md_idx == md_tag.length() and t_idx != t_seq_.length());
    invalid_pair = invalid_pair or (md_idx == md_tag.length() and pretty_idx != pretty_aln_.length());

    if (invalid_pair) {
      invalid_cigar_in_input = true;
      cr_error = CR_ERR_RECREATE_REF;
      return false;
    }
  }

  // ------------------------------------------------ */
  
  if (verbose_) {
    cout << "Original Alignment from BAM: " << endl << pretty_qseq_ << endl;
    cout << pretty_aln_ << endl << pretty_tseq_ << endl << endl;
  }

  // Step 3: Decide whether to realign this read and whether to clip ends

  bool realign_read = ClipAnchors(clip_anchors);
  if (realign_read)
    SetSequences(q_seq_, t_seq_, pretty_aln_, isForwardStrandRead_);
  return realign_read;
}

// ---------------------------------------------------------
// Prototype for more advanced anchor clipping


bool Realigner::ClipAnchors(bool perform_clipping) {

  if (!perform_clipping)
    return true;

  CigarOp current_cigar_element;
  MDelement current_md_element;
  unsigned int pretty_idx = 0;
  unsigned int start_idx = 0;
  unsigned int align_type, offset, nr_matches;
  unsigned int t_start = 0;
  unsigned int q_start  = 0;
  bool get_next_point = true;


  // *** Investigate left end of read for clipping potential

  while (get_next_point) {

    while (pretty_idx < pretty_aln_.size() and pretty_aln_[pretty_idx] == '|')
      pretty_idx++;    // pretty_idx is now pointing to first non-match or end

    // Do not realign perfect matches
    if (start_idx == 0 and pretty_idx == pretty_aln_.size()) {
      if (verbose_)
        cout << "Nothing to realign!" << endl << endl;
      return false;
    } else if (pretty_idx > pretty_aln_.size() - 2* alignment_bandwidth_)
      get_next_point = false;

    bool update_start = (pretty_idx - start_idx) > 3* alignment_bandwidth_;
    update_start = update_start or (start_idx == 0 and (pretty_idx - start_idx) > 2* alignment_bandwidth_);

    if (update_start) { // start_idx is the index of first match in segment

      nr_matches = pretty_idx - start_idx;
      if (start_idx == 0) {
        // First matching Anchor
        current_cigar_element.Type = 'M';
        current_cigar_element.Length = nr_matches; // One bases index
        current_md_element.Type = '=';
        current_md_element.Length = nr_matches;  // One based index
        t_start = nr_matches; // Start pointing at index after last match
        q_start = nr_matches;
      } else {
        // Add previous single non-match to Anchor and first match of latest Anchor
    	switch (pretty_aln_[start_idx -1]) {
          case (' ') :
            align_type = FROM_MATCH;
            t_start++;
            q_start++;
            break;
          case ('+') :
            align_type = FROM_I;
            q_start++;
            break;
          case ('-') :
            align_type = FROM_D;
            t_start++;
            break;
          default :
            if (verbose_)
        	  cout << "Error in ClipAnchors: position before start point is '"
                   << pretty_aln_[start_idx -1] << "'" << endl;
            RestoreAnchors();
	    cr_error = CR_ERR_CLIP_ANCHOR;
            return true;
            break;
    	}
        addCigarElement(align_type, FROM_MATCH, current_cigar_element, clipped_anchors_.cigar_left);
        addCigarElement(FROM_MATCH, align_type, current_cigar_element, clipped_anchors_.cigar_left);
        current_cigar_element.Length += nr_matches -1;
    	addMDelement(align_type, false, FROM_MATCH, t_start, current_md_element, clipped_anchors_.md_left);
    	addMDelement(FROM_MATCH, true, align_type, t_start+1, current_md_element, clipped_anchors_.md_left);
    	current_md_element.Length += nr_matches -1;
    	t_start += nr_matches;
    	q_start += nr_matches;
      }

      pretty_idx++;
      start_idx = pretty_idx; // start_idx is now pointing at base after non-match
    } else
    	get_next_point = false;
  }

  if (start_idx > pretty_aln_.size()) {
    if (verbose_)
      cout << "Nothing to realign!" << endl << endl;
    return false;
  } else {
    // get start points and do not split HPs
    if (start_idx > 0) {
      offset = 2*alignment_bandwidth_;
      while (t_start > offset+5 and t_seq_[t_start-offset-1] == t_seq_[t_start-offset] and offset < 3*alignment_bandwidth_)
        offset++;
      // check whether we found the end of the HP and if not, search in the other direction
      if (t_start <= offset+5 or offset == 3*alignment_bandwidth_){
    	if (t_start <= offset+5)
          offset = 2*alignment_bandwidth_-5;
    	else
          offset = 2*alignment_bandwidth_;
        while (t_seq_[t_start-offset+1] == t_seq_[t_start-offset] and offset > alignment_bandwidth_)
          --offset;
      }
      if (offset == alignment_bandwidth_){
        if (verbose_)
          cout << "Warning: Failed to find start or end of HP on left anchor. Aligning whole read." << endl;
        RestoreAnchors();
	cr_error = CR_ERR_CLIP_ANCHOR;
        return true;
      }
      start_idx = start_idx - offset -1;
      t_start -= offset;
      q_start -= offset;
      current_cigar_element.Length -= offset;
      clipped_anchors_.cigar_left.push_back(current_cigar_element);
      current_md_element.Length -= offset;
      clipped_anchors_.md_left.push_back(current_md_element);
    }

	if (verbose_) {
      cout << "Clipped anchor bases on the left side; Cigar: ";
      for (vector<CigarOp>::const_iterator cigar = clipped_anchors_.cigar_left.begin(); cigar != clipped_anchors_.cigar_left.end(); ++cigar)
        cout << (cigar->Length) << (cigar->Type);
      cout << " MD: " << GetMDstring(clipped_anchors_.md_left) << endl;
	}
  }
  // */

  // *** Investigate right end of read for clipping potential

  pretty_idx = pretty_aln_.size()-1;
  unsigned int stop_idx = pretty_aln_.size();
  unsigned int temp_stop, t_stop = t_seq_.size();
  unsigned int q_stop = q_seq_.size();
  get_next_point = true;


  while (get_next_point) {

    while (pretty_idx > 0 and pretty_aln_[pretty_idx] == '|')
      --pretty_idx;    // pretty_idx is now pointing to latest non-match or start

    if (pretty_idx <= start_idx) {
      if (verbose_)
        cout << "Nothing to realign!" << endl << endl;
      return false;
    } else if (pretty_idx < 2* alignment_bandwidth_)
      get_next_point = false;

    bool update_stop = (stop_idx - pretty_idx-1) > 3* alignment_bandwidth_;
    update_stop = update_stop or (stop_idx == pretty_aln_.size() and (stop_idx - pretty_idx-1) > 2* alignment_bandwidth_);

    if (update_stop) {

      nr_matches = stop_idx-pretty_idx-1;
      if (stop_idx == pretty_aln_.size()) {
        // First matching Anchor
        current_cigar_element.Type = 'M';
        current_cigar_element.Length = nr_matches; // One bases index
        current_md_element.Type = '=';
        current_md_element.Length = nr_matches;  // One based index
        t_stop -= nr_matches; // Stop pointing at non-match
        q_stop -= nr_matches;
      } else {
        // Add previous single non-match to Anchor and first match of latest Anchor
        temp_stop = t_stop;
        switch (pretty_aln_[stop_idx]) {
          case (' ') :
            align_type = FROM_MATCH;
            t_stop--;
            q_stop--;
            break;
          case ('+') :
            align_type = FROM_I;
            q_stop--;
            break;
          case ('-') :
            align_type = FROM_D;
            t_stop--;
            break;
          default :
            if (verbose_)
              cout << "Error in ClipAnchors: stop point is '"
                   << pretty_aln_[stop_idx] << "'" << endl;
            RestoreAnchors();
	    cr_error = CR_ERR_CLIP_ANCHOR;
            return true;
            break;
        }
        addCigarElement(align_type, FROM_MATCH, current_cigar_element, clipped_anchors_.cigar_right);
        addCigarElement(FROM_MATCH, align_type, current_cigar_element, clipped_anchors_.cigar_right);
        current_cigar_element.Length += nr_matches -1;
        addMDelement(align_type, false, FROM_MATCH, temp_stop, current_md_element, clipped_anchors_.md_right);
        addMDelement(FROM_MATCH, true, align_type, temp_stop-1, current_md_element, clipped_anchors_.md_right);
        current_md_element.Length += nr_matches -1;
        t_stop -= nr_matches;
        q_stop -= nr_matches;
      }
      stop_idx = pretty_idx;
      pretty_idx--;

    } else
      get_next_point = false;
  }

  // Get stop point
  if (stop_idx < pretty_aln_.size()) {
    offset = 2*alignment_bandwidth_;
    while (offset < t_seq_.size()-t_stop-5 and t_seq_[t_stop+offset] == t_seq_[t_stop+offset-1] and offset < 3*alignment_bandwidth_)
      offset++;
    // check whether we found the end of the HP and if not, search in the other direction
    if (offset >= (t_seq_.size()-t_stop-5) or offset == 3*alignment_bandwidth_){
      if (offset >= (t_seq_.size()-t_stop-5))
        offset = 2*alignment_bandwidth_-5;
      else
        offset = 2*alignment_bandwidth_;
      while (t_seq_[t_stop+offset] == t_seq_[t_stop+offset-1] and offset > alignment_bandwidth_)
        offset--;
    }
    if (offset == alignment_bandwidth_){
      if (verbose_)
        cout << "Warning: Failed to find start or end of HP on right anchor. Aligning whole read." << endl;
      RestoreAnchors();
      cr_error = CR_ERR_CLIP_ANCHOR;
      return true;
    }

    stop_idx += offset+1;
    t_stop += offset;
    q_stop += offset;
    current_cigar_element.Length -= offset;
    clipped_anchors_.cigar_right.push_back(current_cigar_element);
    current_md_element.Length -= offset;
    clipped_anchors_.md_right.push_back(current_md_element);
  }

  if (verbose_) {
    cout << "Clipped anchor bases on the right side; Cigar: ";
    for (vector<CigarOp>::const_iterator cigar = clipped_anchors_.cigar_right.begin(); cigar != clipped_anchors_.cigar_right.end(); ++cigar)
      cout << (cigar->Length) << (cigar->Type);
    cout << " MD: " << GetMDstring(clipped_anchors_.md_right) << endl;
  }
  // */

  // *** create substrings to be aligned and adjust clipping modes

  if (q_start >= q_seq_.size() or t_start >= t_seq_.size() or start_idx >= pretty_aln_.size()) {
	  if (verbose_)
	    cout << "Error in anchor clipping: start indices are no good. q: "
			  << q_start << " qseq: " << q_seq_.size() << " t: "
			  << t_start << " tseq: " << t_seq_.size() << " p: "
			  << start_idx << " pseq: " << pretty_aln_.size() << endl;
	  cr_error = CR_ERR_CLIP_ANCHOR;
	  return false;
  } else if (q_stop > q_start and (q_stop <= q_seq_.size()) and
      t_stop > t_start and (t_stop <= t_seq_.size()) and
      stop_idx > start_idx and (stop_idx <= pretty_aln_.size())) {

	q_seq_ = q_seq_.substr(q_start, (q_stop-q_start));
	t_seq_ = t_seq_.substr(t_start, (t_stop-t_start));
	pretty_aln_ = pretty_aln_.substr(start_idx, (stop_idx-start_idx));

  } else {
    if (verbose_)
	  cout << "Nothing to realign!" << endl << endl;
    return false;
  }
  if (verbose_)
    cout << "Sequences after anchor clipping: " << endl << q_seq_ << endl << pretty_aln_ << endl << t_seq_ << endl;

  // Make sure clipping settings are consistent with chosen substring i.e. no soft clipping if anchor has been reduced
  bool changed_clipping = false;
  if (clipped_anchors_.md_left.size() > 0) {
    start_anywhere_in_ref_ = false;
    soft_clip_left_        = false;
    changed_clipping       = true;
  }
  if (clipped_anchors_.md_right.size() > 0) {
    stop_anywhere_in_ref_  = false;
    soft_clip_right_       = false;
    changed_clipping       = true;
  }
  if (verbose_ and debug_) {
    if (changed_clipping) {
      cout << "New clipping after anchor trimming:" << endl
           << " - soft_clip_left_" << soft_clip_left_<< endl
           << " - soft_clip_right_" << soft_clip_right_ << endl
           << " - start_anywhere_in_ref_" << start_anywhere_in_ref_ << endl
           << " - stop_anywhere_in_ref_" << stop_anywhere_in_ref_ << endl;
    }
    else {
      cout << "No change to clipping by anchor trimming." << endl;
    }
  }

  return true;
}

// ------------------------------------------------------------------
// Resetting Anchors to soft clipped bases only

void Realigner::RestoreAnchors()
{
  clipped_anchors_.md_left.clear();
  clipped_anchors_.md_right.clear();
  while (clipped_anchors_.cigar_right.size() > 1)
	clipped_anchors_.cigar_right.pop_back();
  while (clipped_anchors_.cigar_left.size() > 1)
  	clipped_anchors_.cigar_left.pop_back();
}


