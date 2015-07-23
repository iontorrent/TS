/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __BANDED_CONVEX_ALIGN_H__
#define __BANDED_CONVEX_ALIGN_H__

#include <vector>
#include <limits.h>
#include "banded_align.h"
#include "trace_matrix.h"


namespace genstr
{
// algorithms for convex gap cost function (needleman-wunch derivative)

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class BandedConvexScore :
    virtual public BandedScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
{
protected:
    struct Score
    {
        ValueType best;
        ValueType before_gap1;
        ValueType before_gap2;
        int gap_len1; // integer - so no prev gap is -1, exended gap after nogap is 0.
        int gap_len2;
        ValueType diag;
        void init ()
        {
            best = before_gap1 = before_gap2 = diag = (ValueType) 0;
            gap_len1 = gap_len2 = -1;
        }
        Score ()
        {
            init ();
        }
    };
    Score sentinel_; // not static to avoid template-related complications.
    typedef std::vector <Score> ScoreVec;
    typedef std::vector <ScoreVec> ScoreVecVec;
    ScoreVecVec cols_storage_; // the storage to hold to (translator1->unit () previous score columns and the currnent score column while calculating scores
    typedef std::deque <ScoreVec*> ScoreVecPtrs; // rotating storage for column pointers
    ScoreVecPtrs cols_;

    unsigned col_pos (unsigned pos1, unsigned pos2)
    {
        unsigned dbox_beg2 = (pos1 / this->translator1_->unit ()) * this->translator2_->unit ();
        if (pos2 + this->dev_ < dbox_beg2 || pos2 >= dbox_beg2 + this->translator2_->unit () + this->dev_)
            return UINT_MAX;
        return pos2 + this->dev_ - dbox_beg2;
    }

    void init_dynamic (); // called before dynamic algorithm invocation; initializes storage for intermediate data if needed
    void init_dynamic_col (unsigned pos1); // called before processing next column
    ValueType score_1 (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 from lower pos2s (same pos1); updates cell at pos1, pos2 as needed
    ValueType score_2 (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 from lower pos1s (same pos2); updates cell at pos1, pos2 as needed
    ValueType score_diag (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 by diagonal; updates cell at pos1, pos2 as needed
};


template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class BandedConvexAlign :
    virtual public BandedConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>,
    virtual public BandedAlign <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
{
};

}; // namespace genstr

#include "banded_convex_align.hpp"

#endif // __BANDED_CONVEX_ALIGN_H__
