/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __TRACE_HOLDER_H__
#define __TRACE_HOLDER_H__

#include "tracebox_templ.h"

namespace genstr
{

template <typename ValueType> class TraceHolder
{
public:
    virtual void add (unsigned end1, unsigned end2, ValueType score) = 0;
    virtual unsigned size () const = 0;
    virtual TraceBox& operator [] (unsigned idx) = 0;
    virtual void sort () = 0;
};


template <typename ValueType> class SimpleTraceHolder : public TraceHolder <ValueType>
{
    std::vector <TraceBox<ValueType> > boxes_;
public:
    void add (unsigned end1, unsigned end2, ValueType score)
    {
        boxes_.push_back (TraceBox<ValueType> (end1, end2, score));
    }
    unsigned size () const
    {
        return boxes_.size ();
    }
    TraceBox& operator [] (unsigned idx)
    {
        return boxes_ [idx];
    }
    void sort ()
    {
        std::sort (boxes_.begin (), boxes_.end (), compareTraceBox<ValueType>);
    }
};


template <typename ValueType>
class PrioritizedTraceHolder : public TraceHolder <ValueType>
{
    std::vector<TraceBox<ValueType> > boxes_;
    unsigned capacity_;
    ValueType std::minscore_;
public:
    PrioritizedTraceHolder ()
    :
    capacity_ (0),
    std::minscore_ ((ValueType) 0)
    {
    }
    void configure (unsigned capacity, ValueType std::minscore)
    {
        capacity_ = capacity;
        while (_capacity && boxes_.size () > capacity_)
            std::pop_heap (boxes_.begin (), boxes_.end (), compareTraceBox<ValueType>);
        std::minscore_ = std::minscore;
    }
    ValueType minScore () const
    {
        return std::minscore_;
    }
    ValueType capacity () const
    {
        return capacity_;
    }
    void add (unsigned end1, unsigned end2, ValueType score)
    {
        if (std::minscore_ > score) return;
        boxes_.push_back (TraceBox<ValueType> (end1, end2, score));
        std::push_heap (boxes_.begin (), boxes_.end (), compareTraceBox<ValueType>);
        if (capacity_ && boxes_.size () > capacity_)
            std::pop_heap (boxes_.begin (), boxes_.end (), compareTraceBox<ValueType>);
    }
    unsigned size () const
    {
        return boxes_.size ();
    }
    TraceBox& operator [] (unsigned idx)
    {
        return boxes_ [idx];
    }
    void sort ()
    {
        std::sort_heap (boxes_.begin (), boxes_.end (), compareTraceBox<ValueType>);
    }
};

} // namespace genstr

#endif // __TRACE_HOLDER_H__
