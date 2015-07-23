/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __ALPHABET_TEMPL_H__
#define __ALPHABET_TEMPL_H__

#if defined (_MSC_VER)
#pragma warning (disable:4290)
#endif

#include <algorithm>
#include <resource.h>
#include <rerror.h>

namespace genstr
{

#ifndef __ALPHABET_TEMPL_CPP__
extern const char* ERR_SymbolNotInAlphabet;
#endif

MAKE_ERROR_TYPE (SymbolNotInAlphabet, ERR_SymbolNotInAlphabet);

//////////////////////////////////////////////////////////////////////////////
// class Alphabet
//    encapsulates a set of enumeratable symbols.
//    provides facility for translating symbols into indices and back.
//////////////////////////////////////////////////////////////////////////////

template <typename SymType>
class Alphabet // Note: Indices have same type as symbols
{
    SymType     sym2num_ [0x1 << (sizeof (SymType) << 3)];
    MemWrapper <SymType> symbols_;
    unsigned    size_;

    void clear ();

public:

    Alphabet ();
    void configure (const SymType* symbols, unsigned size);
    SymType index (SymType symbol) const
    {
        SymType toRet = sym2num_ [(unsigned) symbol];
        if (((unsigned) toRet) == size_) ers << "'" << symbol << "'" << ThrowEx(SymbolNotInAlphabet);
        return toRet;
    }
    SymType symbol (SymType index) const
    {
        if (index < 0 || ((unsigned) index) >= size_) ers << "'" << index << "'" << ThrowEx(SymbolNotInAlphabet);
        return ((const SymType*) symbols_) [(unsigned) index];
    }
    unsigned size () const
    {
        return size_;
    }
    void to_symbols (SymType* indices, unsigned size) const
    {
        while (size-- > 0)
        {
            *indices = symbol (*indices);
            indices ++;
        }
    }
    void to_numbers (SymType* symbols, int size) const
    {
        while (size-- > 0)
        {
            *symbols = index (*symbols);
            symbols ++;
        }
    }
    const SymType* symbols () const
    {
        return symbols_;
    }
};

template <typename SymType> Alphabet<SymType>::Alphabet ()
:
size_ (0)
{
}

template <typename SymType> void Alphabet<SymType>::clear ()
{
    symbols_.free ();
    size_ = 0;
}

template <typename SymType> void Alphabet<SymType>::configure (const SymType* symbols, unsigned size)
{
    clear ();
    size_ = size;
    symbols_ = new SymType [size_];
    std::copy ((SymType*) symbols, (SymType*) symbols + size_, (SymType*) symbols_);
    std::fill (sym2num_, sym2num_ + sizeof (sym2num_) / sizeof (*sym2num_), size_);

    unsigned idx;
    SymType* ptr;
    for (idx = 0, ptr = symbols_; idx < size_; idx ++, ptr ++)
    {
        sym2num_ [(unsigned) *ptr] = idx;
    }
}

} // namespace genstr

#endif // __ALPHABET_TEMPL_H__
