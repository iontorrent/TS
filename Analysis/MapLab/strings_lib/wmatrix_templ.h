/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __WMATRIX_TEMPL_H__
#define __WMATRIX_TEMPL_H__

#include <algorithm>
#include <rerror.h>
#include <resource.h>
#include "alphabet_templ.h"

namespace genstr
{

// WeightMatrix is a mapping from pair of symols in an Alphabet to a value

// SymType is a type of symbols that Alphabet consists of
// ValueType is a destination type
// AveType is numeric type that holds value statistics


template <typename SymType = char, typename ValueType = int, typename AveType = double>
class WeightMatrix
{
    Alphabet<SymType>        alphabet_;
    MemWrapper <ValueType>  values_;
    MemWrapper <ValueType*> rows_;
    AveType                 average_;
    AveType                 aveMism_;
    AveType                 aveMatch_;
    ValueType               minScore_;
    ValueType               maxScore_;

public:
    WeightMatrix ();
    void  configure (const SymType* symbols, unsigned size, const ValueType* data);
    ValueType weight (SymType idx1, SymType idx2)
    {
#ifdef DEBUG
        if (idx1 >= alphabet_.size ()) ers << "Symbol index '" << idx1 << "' out of range" << ThrowEx (SymbolNotInAlphabet);
        if (idx2 >= alphabet_.size ()) ers << "Symbol index '" << idx2 << "' out of range" << ThrowEx (SymbolNotInAlphabet);
#endif
        return ((ValueType*) (((ValueType**) rows_) [(unsigned) idx1]))[(unsigned) idx2];
    }
    ValueType symWeight (SymType sym1, SymType sym2)
    {
        return weight (alphabet_.index (sym1), alphabet_.index (sym2));
    }
    ValueType** rows () const
    {
        return rows_;
    }
    ValueType* row (SymType sym_index) const
    {
#ifdef DEBUG
        if (sym_index >= alphabet_.size ()) ers << "Symbol index '" << idx1 << "' out of range" << ThrowEx (SymbolNotInAlphabet);
#endif
        return rows_[sym_index];
    }
    ValueType* symRow (SymType sym1) const
    {
        return rows_ [alphabet_.index (sym1)];
    }
    AveType getAverage ()
    {
        return average_;
    }
    AveType getAverageMismatch ()
    {
        return aveMism_;
    }
    AveType getAverageMatch ()
    {
        return aveMatch_;
    }
    ValueType getMinScore ()
    {
        return minScore_;
    }
    ValueType getMaxScore ()
    {
        return maxScore_;
    }
    const Alphabet<SymType>& getAlphabet () const
    {
        return alphabet_;
    }
};

template <typename SymType, typename ValueType, typename AveType>
WeightMatrix<SymType, ValueType, AveType>
::WeightMatrix ()
:
average_ ((AveType) 0),
aveMism_ ((AveType) 0),
aveMatch_ ((AveType) 0),
minScore_ ((ValueType) 0),
maxScore_ ((ValueType) 0)
{
}

template <typename SymType, typename ValueType, typename AveType>
void WeightMatrix <SymType, ValueType, AveType>
::configure (const SymType* symbols, unsigned size, const ValueType* data)
{
    // fill in alphabet
    alphabet_.configure (symbols, size);
    // copy data
    values_ = new ValueType [size * size];
    std::copy (data, data + size * size, (ValueType*) values_);
    // fill in rows
    rows_ = new ValueType* [size];
    for (unsigned rowidx_ = 0; rowidx_ < size; rowidx_ ++)
        rows_ [rowidx_] = (ValueType*) values_ + size * rowidx_;
    // calculate averages
    average_ = aveMatch_ = aveMism_ = (AveType) 0;
    ValueType* cur = values_;
    for (unsigned idx = 0; idx < size; idx ++)
        for (unsigned idx1 = 0; idx1 < size; idx1 ++)
        {
            average_ += *cur;
            if (idx == idx1) aveMatch_ += *cur;
            else aveMism_ += *cur;
            if (minScore_ > *cur || !(idx || idx1)) minScore_ = *cur;
            if (maxScore_ < *cur || !(idx || idx1)) maxScore_ = *cur;
            cur ++;
        }
    average_  /= (ValueType) (size * size);
    aveMatch_ /= (ValueType) size;
    aveMism_  /= (ValueType) (size * (size - 1));
}

template <typename ValueType, unsigned dim>
class UnitaryMatrix
{
    ValueType values_ [dim*dim];
public:
    UnitaryMatrix ()
    {
        for (unsigned i = 0; i < dim; i ++)
            for (unsigned j = 0; j < dim; j ++)
                if (i == j) values_ [i*dim + j] = 1;
                else values_ [i*dim + j] = 0;
    }
    ValueType* values ()
    {
        return values_;
    }
};

template <typename ValueType, unsigned dim>
class NegUnitaryMatrix
{
    ValueType values_ [dim*dim];
public:
    NegUnitaryMatrix ()
    {
        for (unsigned i = 0; i < dim; i ++)
            for (unsigned j = 0; j < dim; j ++)
                if (i == j) values_ [i*dim + j] = 1;
                else values_ [i*dim + j] = -1;
    }
    ValueType* values ()
    {
        return values_;
    }
};

template <typename MatrixType, typename GapWeightType>
GapWeightType normGapWeight (MatrixType* matrix, GapWeightType gapWeight)
{
    return (GapWeightType) (matrix->getMinScore () * gapWeight);
}

} // namespace genstr

#endif // __WMATRIX_TEMPL_H__
