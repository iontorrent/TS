/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __ALIGN_TEMPL_H__
#define __ALIGN_TEMPL_H__

#ifdef _MSC_VER
#pragma warning (disable:4250)
#endif


#include <algorithm>
#include <limits.h>
#include "wmatrix_templ.h"
#include "gap_cost_templ.h"
#include "translator_templ.h"
#include "trace_matrix.h"
#include "batch.h"

namespace genstr
{

// basic tracer (abstract)
// The device to find the best local alignment score and the best local alignment box
// in the pair of sequence fragments

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class Score
{
protected:

    WeightMatrix <DestSymbolType, ValueType, AveType> * weight_matrix_;
    GapCost <GapType> * gap_cost_eval_1_;
    GapCost <GapType> * gap_cost_eval_2_;

    // search area
    const Src1SymbolType* seq1_; // sequence 1 pointer
    const Src2SymbolType* seq2_; // sequence 2 pointer
    unsigned len1_;  // end of the search zone on sequence 1
    unsigned len2_;  // end of the search zone on sequence 2

    // translator
    typedef Translator<Src1SymbolType, DestSymbolType> Translator1;
    Translator1* translator1_;
    typedef Translator<Src2SymbolType, DestSymbolType> Translator2;
    Translator2* translator2_;

    typedef NullTranslator<Src1SymbolType, DestSymbolType> DefTranslator1;
    typedef NullTranslator<Src2SymbolType, DestSymbolType> DefTranslator2;
    DefTranslator1 def_translator1_;
    DefTranslator2 def_translator2_;

    // result holder
    unsigned maxPos1_;
    unsigned maxPos2_;
    ValueType bestWeight_;

    // find the location of maximum score
    virtual void dynamic ();
    virtual void dynamic_seq2 (unsigned pos1);
    virtual ValueType dynamic_position (unsigned pos1, unsigned pos2);
    // obligtory functions for overloading
    virtual void init_dynamic () = 0; // called before dynamic algorithm invocation; initializes storage for intermediate data if needed
    virtual void init_dynamic_col (unsigned pos1) = 0; // called before processing next column
    virtual ValueType score_1 (unsigned pos1, unsigned pos2) = 0; // compute best score for paths coming to cell at pos1, pos2 from lower pos1s (same pos2); updates cell at pos1, pos2 as needed
    virtual ValueType score_2 (unsigned pos1, unsigned pos2) = 0; // compute best score for paths coming to cell at pos1, pos2 from lower pos2s (same pos1); updates cell at pos1, pos2 as needed
    virtual ValueType score_diag (unsigned pos1, unsigned pos2) = 0; // compute best score for paths coming to cell at pos1, pos2 by diagonal; updates cell at pos1, pos2 as needed
    // if direction 1 is horizontal, direction 2 is vertical, then:
    // score_1 is for paths coming from left
    // score_2 is for paths coming from below

public:
    virtual ~Score ()
    {
    }
    void configure (WeightMatrix <DestSymbolType, ValueType, AveType>* weight_matrix,
                    GapCost<GapType>* gap_cost1,
                    GapCost<GapType>* gap_cost2,
                    Translator<Src1SymbolType, DestSymbolType>* tr1 = NULL,
                    Translator<Src2SymbolType, DestSymbolType>* tr2 = NULL);
    ValueType eval (const Src1SymbolType* seq1,
                    unsigned len1,
                    const Src2SymbolType* seq2,
                    unsigned len2);
    ValueType weight () const
    {
        return bestWeight_;
    }
    unsigned bestPos1 () const
    {
        return maxPos1_;
    }
    unsigned bestPos2 () const
    {
        return maxPos2_;
    }
};

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
class Align : virtual public Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
{
protected:
    TraceMatrix <false> trace_;
    virtual void backtrace (); // actually computes the Alignment (by backtracing matrix or by any other mean)
    Alignment* result_;
    void dynamic ()
    {
        result_ = NULL;
        trace_.init (this->len1_, this->len2_);
        Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType> :: dynamic ();
    }
    ValueType dynamic_position (unsigned pos1, unsigned pos2); // overloaded to memorize trace
public:
    Align ()
    :
    result_ (NULL)
    {
    }
    Alignment* trace ()
    {
        if (result_ == NULL) backtrace ();
        return result_;
    }
    void print_trace_matrix (unsigned mp1 = UINT_MAX, unsigned mp2 = UINT_MAX)
    {
        trace_.print (mp1, mp2);
    }
};


}; // namespace genstr

#include "align_templ.hpp"

#endif // __ALIGN_TEMPL_H__

