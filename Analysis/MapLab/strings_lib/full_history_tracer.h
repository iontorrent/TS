/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __FULL_HISTORY_TRACER__
#define __FULL_HISTORY_TRACER__

#include <vector>
#include <limits.h>
#include "align_templ.h"

namespace genstr
{

// very slow complete algorithm-sutable for arbitrary gap cost function
template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class FullHistoryTracer : public Align <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
{
protected:
    struct Score
    {
        ValueType gap1;
        unsigned gapBeg1;
        ValueType gap2;
        unsigned gapBeg2;
        ValueType diag;
        void init ()
        {
            gap1 = (ValueType) 0;
            gap2 = (ValueType) 0;
            diag = (ValueType) 0;
            gapBeg1 = UINT_MAX;
            gapBeg2 = UINT_MAX;
        }
        Score ()
        {
            init ();
        }
    };
    Score sentinel_; // not static to avoid template-related complications (object initialization etc)
    typedef std::vector< Score > ScoreVec;
    ScoreVec scores_;

    unsigned index (unsigned off1, unsigned off2)
    {
        return this->len2_ * off1 + off2;
    }

    void init_dynamic (); // called before dynamic algorithm invocation; initializes storage for intermediate data if needed
    void init_dynamic_col (unsigned pos1); // called before processing next column
    ValueType score_1 (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 from lower pos2s (same pos1); updates cell at pos1, pos2 as needed
    ValueType score_2 (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 from lower pos1s (same pos2); updates cell at pos1, pos2 as needed
    ValueType score_diag (unsigned pos1, unsigned pos2); // compute best score for paths coming to cell at pos1, pos2 by diagonal; updates cell at pos1, pos2 as needed
};

}; // namespace genstr

#include "full_history_tracer.hpp"

#endif // __FULL_HISTORY_TRACER__
