/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <rerror.h>
#include <limits.h>


namespace genstr
{
//////////////////////////////////////////////////////////
// FullHistoryTracer

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void FullHistoryTracer <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::init_dynamic ()
{
    // allocate score matrix
    ulonglong capacity = ulonglong (this->len1_) * this->len2_;
    if (capacity > UINT_MAX) ERR ("Search zones are too long");
    this->scores_.resize (unsigned (capacity));

    // initialize first row
    for (unsigned pos2 = 0; pos2 < this->len2_; pos2 ++)
        this->scores_ [pos2].init ();
}


template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void FullHistoryTracer <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::init_dynamic_col (unsigned pos1)
{
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType FullHistoryTracer <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::score_diag (unsigned pos1, unsigned pos2) // compute best score for paths coming to cell at pos1, pos2 by diagonal; updates cell at pos1, pos2 as needed
{
    Score& prev = (pos1 >= this->translator1_->unit () && pos2 >= this->translator2_->unit ()) ? scores_ [index (pos1 - this->translator1_->unit (), pos2 - this->translator2_->unit ())] : sentinel_;
    ValueType score = std::max (std::max (prev.diag, prev.gap1), prev.gap2);
    if (pos1 + 1 >= this->translator1_->unit () && pos2 + 1 >= this->translator2_->unit ())
        score += this->weight_matrix_->weight (this->translator1_->translate (this->seq1_, pos1 + 1 - this->translator1_->unit ()), this->translator2_->translate (this->seq2_, pos2 + 1 - this->translator2_->unit ()));
    score = std::max (score, (ValueType) 0);
    Score& cur = this->scores_ [index (pos1, pos2)];
    cur.diag = score;
    return score;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType FullHistoryTracer <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::score_1 (unsigned pos1, unsigned pos2) // compute best score for paths coming to cell at pos1, pos2 from lower pos2s (same pos1); updates cell at pos1, pos2 as needed
{
    // calculate best gap weight along axis 1
    ValueType best1 = (ValueType) 0;
    unsigned bestPos1 = 0;
    for (unsigned pos = 0; pos < pos1; pos ++)
    {
        // we allow the gap to be a combination of several independent deletion events - if it gives smaller cost
        Score& prev = pos ? scores_ [index (pos, pos2)] : sentinel_;
        ValueType score = std::max (std::max (prev.diag, prev.gap1), prev.gap2);
        score += (ValueType) normGapWeight (this->weight_matrix_, (*this->gap_cost_eval_1_) (pos1 - pos));
        score  = std::max (score , (ValueType) 0);
        if (score  > best1)
        {
            best1 = score;
            bestPos1 = pos;
        }
    }
    Score& cur = this->scores_ [index (pos1, pos2)];
    cur.gap1 = best1;
    cur.gapBeg1 = bestPos1;
    return best1;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType FullHistoryTracer <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::score_2 (unsigned pos1, unsigned pos2) // compute best score for paths coming to cell at pos1, pos2 from lower pos1s (same pos2); updates cell at pos1, pos2 as needed
{
    // calculate best gap weight along axis 2
    ValueType best2 = (ValueType) 0;
    unsigned bestPos2 = 0;
    for (unsigned pos = 0; pos < pos2; pos ++)
    {
        // we allow the gap to be a combination of several independent deletion events - if it gives smaller cost
        Score& prev = pos ? scores_ [index (pos1, pos)] : sentinel_;
        ValueType score = std::max (std::max (prev.diag, prev.gap1), prev.gap2);
        score += (ValueType) normGapWeight (this->weight_matrix_, (*this->gap_cost_eval_2_) (pos2 - pos));
        score = std::max (score, (ValueType) 0);
        if (score > best2)
        {
            best2 = score;
            bestPos2 = pos;
        }
    }
    Score& cur = this->scores_ [index (pos1, pos2)];
    cur.gap2 = best2;
    cur.gapBeg2 = bestPos2;
    return best2;
}

}; // namespace genstr
