/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __circbuf_h__
#define __circbuf_h__

#include "resource.h"
#include "rerror.h"

template <typename Elemtype, unsigned Sz> class CircBuffer
{
private:
    Elemtype buf_ [Sz];
    unsigned cpos_;
    unsigned fill_;
public:
    CircBuffer () { reset (); }
    void reset () { cpos_ = 0, fill_ = 0; }
    void add (Elemtype toAdd)
    {
        buf_ [cpos_] = toAdd;
        reserve ();
    }
    Elemtype& reserve ()
    {
        Elemtype& toR = buf_ [cpos_];
        cpos_ ++;
        cpos_ %= Sz;
        if (fill_ < Sz) fill_ ++;
        return toR;
    }
    unsigned size () const { return fill_; }
    const Elemtype* data () const { return buf_; }
    Elemtype* data () { return buf_; }
    unsigned zeropos () const { return fill_ < Sz ? 0 : cpos_; }
    const Elemtype& operator [] (unsigned idx) const
    {
        if (fill_ < Sz)
        {
            if (idx >= fill_) ers << idx << ThrowEx (OutOfBoundsRerror);
            return buf_ [fill_ - 1 - idx];
        }
        else
        {
            if (idx >= Sz) ers << idx << ThrowEx (OutOfBoundsRerror);
            return buf_ [(cpos_ + (Sz - 1 - idx)) % Sz];
        }
    }
    Elemtype& operator [] (unsigned idx)
    {
        if (fill_ < Sz)
        {
            if (idx >= fill_) ers << idx << ThrowEx (OutOfBoundsRerror);
            return buf_ [fill_ - 1 - idx];
        }
        else
        {
            if (idx >= Sz) ers << idx << ThrowEx (OutOfBoundsRerror);
            return buf_ [(cpos_ + (Sz - 1 - idx)) % Sz];
        }
    }
};

// Warning: the following class is NOT similar to the one above.
// They have different structure AND SEMANTICS!
// (the one above keeps track of number of elements added and does not alow access to positions over the max added one.)

template <typename Elemtype> class CircBufferVlen
{
private:
    MemWrapper <Elemtype> buf_;
    unsigned size_;
    unsigned lpos_;
    unsigned upos_;
public:
    CircBufferVlen () : size_ (0) { reset ();}
    CircBufferVlen (unsigned size) { init (size); }
    void init (unsigned size) { reset (); size_ = size; buf_ = new Elemtype [size_]; }
    void reset () { lpos_ = 0, upos_ = 0; }
    void rotate ()
    {
        if (++lpos_ == size_) lpos_ = 0;
        if (++upos_ == size_) upos_ = 0;
    }
    void push_back (Elemtype toAdd)
    {
        if (!size_) ers << "Access to circular buffer of zero size" << ThrowEx (OutOfBoundsRerror);
        buf_ [upos_] = toAdd;
        if (++upos_ == size_) upos_ = 0;
    }
    Elemtype pop_back ()
    {
        if (upos_ == lpos_) ers << "Circular buffer underflow" << ThrowEx (OutOfBoundsRerror);
        Elemtype toR = buf_ [upos_];
        if (lpos_) lpos_ --;
        else lpos_ = size_ - 1;
        return toR;
    }
    void push_front (Elemtype toAdd)
    {
        if (!size_) ers << "Access to circular buffer of zero size" << ThrowEx (OutOfBoundsRerror);
        if (lpos_) lpos_ --;
        else lpos_ = size_ - 1;
        buf_ [lpos_] = toAdd;
    }
    Elemtype pop_front ()
    {
        if (upos_ == lpos_) ers << "Circular buffer underflow" << ThrowEx (OutOfBoundsRerror);
        Elemtype toR = buf_ [lpos_];
        if (++lpos_ == size_) lpos_ = 0;
        return toR;
    }
    unsigned size () { return size_; }
    Elemtype& operator [] (unsigned idx)
    {
        idx += lpos_;
        idx %= size_;
        return buf_ [idx];
    }
};

#endif
