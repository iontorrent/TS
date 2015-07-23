/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __CONVEX_ALIGN_H__
#define __CONVEX_ALIGN_H__

#include <vector>
#include <limits.h>
#include "align_templ.h"
#include "trace_matrix.h"


namespace genstr
{
// algorithms for convex gap cost function (needleman-wunch derivative)

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class ConvexScore :
    virtual public Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
{
protected:
    struct Score
    {
        ValueType best;
        ValueType before_gap1;
        ValueType before_gap2;
        int gap_len1;
        int gap_len2; // integer so that next position gap is always cur_gap+1 (0 if -1)
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

    void init_dynamic (); // called before dynamic algorithm invocation; initializes storage for intermediate data if needed
    void init_dynamic_col (unsigned pos1); // called before processing next column
    ValueType score_1 (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 from lower pos2s (same pos1); updates cell at pos1, pos2 as needed
    ValueType score_2 (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 from lower pos1s (same pos2); updates cell at pos1, pos2 as needed
    ValueType score_diag (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 by diagonal; updates cell at pos1, pos2 as needed
};


template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class ConvexAlign :
    virtual public ConvexScore <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>,
    virtual public Align <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
{
};

}; // namespace genstr

#include "convex_align.hpp"

#endif // __CONVEX_ALIGN_H__
