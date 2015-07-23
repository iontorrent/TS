/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __TRANSLATOR_TEMPL_H__
#define __TRANSLATOR_TEMPL_H__

#include "alphabet_templ.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace genstr
{

// Translator provides facility for converting between sequences of different alphabets (like Polynucleotide->Polypeptide)

template <typename SourceType, typename DestType>
class Translator
{
public:
    virtual DestType translate (const SourceType* seq, unsigned offset) = 0; // translates symbol at position 'offset' in seq
    virtual unsigned xlat (const SourceType* src, unsigned len, DestType* dest = NULL) = 0; // translates len symbols starting from src; places them in dest or, if dest is Null,
                                                                                            // returns number of symbols in a source that is translated in a symbol of dest
    virtual unsigned unwind (const SourceType* src, unsigned len, DestType* dest = NULL) = 0; // unwinds each unit in each phase from src into a symbol in dest.
                                                                                              // dest must have len + 1 - unit () slots available
    virtual unsigned unit () const = 0;
};

template <typename SourceType, typename DestType>
class NullTranslator : public Translator <SourceType, DestType> // Null translator does nothing, dest = src
{
public:
    DestType translate (const SourceType* seq, unsigned offset)
    {
        return (DestType) seq [offset];
    }
    unsigned xlat (const SourceType* src, unsigned len, DestType* dest = NULL)
    {
        if (dest && dest != src) std::copy (src, src + len, dest);
        return len;
    }
    unsigned unwind (const SourceType* src, unsigned len, DestType* dest = NULL)
    {
        return xlat (src, len, dest);
    }
    virtual unsigned unit () const
    {
        return 1;
    }
};

// Composite translator allows to chain two translator if DestType of first equals SrcType of second one.

template <typename SourceType, typename MedType, typename DestType>
class CompositeTranslator : public Translator <SourceType, DestType>
{
    Translator <SourceType, MedType> &tr1_;
    Translator <MedType, DestType> &tr2_;
public:
    CompositeTranslator (Translator <SourceType, MedType> &tr1, Translator <MedType, DestType> &tr2)
    :
    tr1_ (tr1),
    tr2_ (tr2)
    {
    }
    DestType translate (const SourceType* src, unsigned offset)
    {
        MedType* tmp = (MedType*) alloca (tr2_.unit () * sizeof (MedType));
        tr1_.xlat (src + offset, tr1_.unit () * tr2_.unit (), tmp);
        return tr2_.translate (tmp, 0);
    }
    unsigned xlat (const SourceType* src, unsigned len, DestType* dest)
    {

        unsigned medSize = (len + 1 - tr1_.unit ()) / tr1_.unit ();

        if (dest == NULL)
            return (medSize + 1 - tr2_.unit ()) / tr2_.unit ();

        MedType* mbuf = (MedType*) alloca (sizeof (MedType) * medSize);
        tr1_.xlat (src, len + 1 - tr1_.unit (), mbuf);

        return tr2_.xlat (mbuf, medSize - tr2_.unit () + 1, dest);
    }
    unsigned unwind (const SourceType* src, unsigned len, DestType* dest)
    {

        unsigned medSize = (len + 1 - tr1_.unit ());

        if (dest == NULL)
            return (medSize + 1 - tr2_.unit ());

        MedType* mbuf = (MedType*) alloca (sizeof (MedType) * medSize);
        tr1_.unwind (src, len + 1 - tr1_.unit (), mbuf);

        return tr2_.unwind (mbuf, medSize + 1 - tr2_.unit (), dest);
    }
    unsigned unit () const
    {
        return tr1_.unit () * tr2_.unit ();
    }
};

// Wrapper for Alphabet's symbol-to-index translator

template <typename SymType>
class AlphaFwd : public Translator <SymType, SymType>
{
    Alphabet <SymType>& alpha_;
public:
    AlphaFwd (Alphabet <SymType>& alpha)
    :
    alpha_ (alpha)
    {
    }
    SymType translate (const SymType* seq, unsigned offset)
    {
        return alpha_.index (seq [offset]);
    }
    unsigned xlat (const SymType* src, unsigned len, SymType* dest = NULL)
    {
        if (dest)
        {
            if (dest != src) std::copy (src, src+len, dest);
            alpha_.to_numbers (dest, len);
        }
        return len;
    }
    unsigned unwind  (const SymType* src, unsigned len, SymType* dest = NULL)
    {
        return xlat (src, len, dest);
    }
    unsigned unit () const
    {
        return 1;
    }
};

// Wrapper for Alphabet's index-to-symbol translator
template <typename SymType>
class AlphaRev : public Translator <SymType, SymType>
{
    Alphabet <SymType>& alpha_;
public:
    AlphaRev (Alphabet <SymType>& alpha)
    :
    alpha_ (alpha)
    {
    }
    SymType translate (const SymType* seq, unsigned offset)
    {
        return alpha_.symbol (seq [offset]);
    }
    unsigned xlat (const SymType* src, unsigned len, SymType* dest = NULL)
    {
        if (dest)
        {
            if (dest != src) std::copy (src, src+len, dest);
            alpha_.to_symbols (dest, len);
        }
        return len;
    }
    unsigned unwind  (const SymType* src, unsigned len, SymType* dest = NULL)
    {
        return xlat (src, len, dest);
    }
    unsigned unit () const
    {
        return 1;
    }
};

} // namespace genstr

#endif // __TRANSLATOR_TEMPL_H__
