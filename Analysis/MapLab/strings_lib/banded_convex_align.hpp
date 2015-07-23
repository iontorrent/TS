/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <rerror.h>

namespace genstr
{

//////////////////////////////////////////////////////////
// BandedConvexScore

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void BandedConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::init_dynamic ()
{
    // allocate dynamic columns
    cols_storage_.resize (this->translator1_->unit () + 1);
    for (unsigned history_pos = 0; history_pos <= this->translator1_->unit (); history_pos ++)
    {
        cols_storage_ [history_pos].resize (this->wid_);
        cols_.push_back (&(cols_storage_ [history_pos]));
    }
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void BandedConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::init_dynamic_col (unsigned pos1)
{
    // rotate column history, so that last (cur) becomes last - 1, 0 becomes last:
    // cur is to be used at as prev at this iteration; the one that was prev is re-used for currently computed scores
    cols_.push_back (cols_.front ());
    cols_.pop_front ();
    // zero the new cur column
    ScoreVec& cur = *(cols_.back ());
    for (unsigned idx = 0; idx < this->wid_; idx ++)
        cur [idx].init ();

#ifdef DEBUG_TRACE
    std::cout << std::endl;
#endif

}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType BandedConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::score_diag (unsigned pos1, unsigned pos2)
{
    Score& cur = (*(cols_.back ())) [col_pos (pos1, pos2)];
    Score& prev = (pos1 > this->translator1_->unit () && pos2 > this->translator2_->unit ()) ? (*(cols_.front ())) [col_pos (pos1 - this->translator1_->unit (), pos2 - this->translator2_->unit ())] : sentinel_;

    // calculate diagonal score
    ValueType score = prev.best;
    if (pos1 >= this->translator1_->unit () - 1 && pos2 >= this->translator2_->unit () - 1)
        score += this->weight_matrix_->weight (this->translator1_->translate (this->seq1_, pos1 + 1 - this->translator1_->unit ()), this->translator2_->translate (this->seq2_, pos2 + 1 - this->translator2_->unit ()));
    score = std::max (score, (ValueType) 0);
    cur.diag = score;
    if (score > cur.best) cur.best = score;
    return score;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType BandedConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::score_1 (unsigned pos1, unsigned pos2)
{
    unsigned cur_pos = col_pos (pos1, pos2);
    Score& cur = (*(cols_.back ())) [cur_pos];
    unsigned prev_pos;
    Score& prev = (pos1 && ((prev_pos = col_pos (pos1 - 1, pos2)) != UINT_MAX)) ? (**(cols_.rbegin () + 1)) [prev_pos] : sentinel_;

    ValueType gap_init = prev.best + (ValueType) normGapWeight (this->weight_matrix_, (*this->gap_cost_eval_1_) (1));
    ValueType gap_ext  = prev.before_gap1 + (ValueType) normGapWeight (this->weight_matrix_, (*this->gap_cost_eval_1_) (prev.gap_len1 + 1));
    ValueType score;
    if (gap_init > gap_ext)
    {
        gap_init = std::max ((ValueType) 0, gap_init);
        cur.gap_len1 = 1;
        cur.before_gap1 = prev.best;
        score = gap_init;
    }
    else
    {
        gap_ext = std::max ((ValueType) 0, gap_ext);
        cur.gap_len1 = prev.gap_len1 + 1;
        cur.before_gap1 = prev.before_gap1;
        score = gap_ext;
    }
    if (score > cur.best) cur.best = score;
    return score;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType BandedConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::score_2 (unsigned pos1, unsigned pos2)
{
    Score& cur = (*(cols_.back ())) [col_pos (pos1, pos2)];
    unsigned prev_pos;
    Score& prev = (((prev_pos = col_pos (pos1, pos2 - 1))!= UINT_MAX) && pos2) ? (*(cols_.back ())) [prev_pos] : sentinel_;

    ValueType gap_init = prev.best + (ValueType) normGapWeight (this->weight_matrix_, (*this->gap_cost_eval_2_) (1));
    ValueType gap_ext  = prev.before_gap2 + (ValueType) normGapWeight (this->weight_matrix_, (*this->gap_cost_eval_2_) (prev.gap_len2 + 1));
    ValueType score;
    if (gap_init > gap_ext)
    {
        gap_init = std::max ((ValueType) 0, gap_init);
        cur.gap_len2 = 1;
        cur.before_gap2 = prev.best;
        score = gap_init;
    }
    else
    {
        gap_ext = std::max ((ValueType) 0, gap_ext);
        cur.gap_len2 = prev.gap_len2 + 1;
        cur.before_gap2 = prev.before_gap2;
        score = gap_ext;
    }
    if (score > cur.best) cur.best = score;
    return score;
}

}; // namespace genstr
