/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __TRACEBOX_TEMPL_H__
#define __TRACEBOX_TEMPL_H__

namespace genstr
{

template <typename ValueType> class TraceBox
{
public:
    unsigned beg1;
    unsigned beg2;
    unsigned end1;
    unsigned end2;
    ValueType score;
    TraceBox ()
    {
        TraceBox::beg1 = -1;
        TraceBox::beg2 = -1;
        TraceBox::end1 = -1;
        TraceBox::end2 = -1;
        TraceBox::score = (ValueType) 0;
    }
    TraceBox (unsigned end1, unsigned end2, ValueType score)
    {
        TraceBox::beg1 = -1;
        TraceBox::beg2 = -1;
        TraceBox::end1 = end1;
        TraceBox::end2 = end2;
        TraceBox::score = score;
    }
    TraceBox (const TraceBox& toCopy)
    {
        operator = (toCopy);
    }
    TraceBox& operator = (const TraceBox& toCopy)
    {
        beg1 = toCopy.beg1;
        beg2 = toCopy.beg2;
        end1 = toCopy.end1;
        end2 = toCopy.end2;
        score = toCopy.score;
        return *this;
    }
    bool operator < (const TraceBox& other) const
    {
        return score < other.score;
    }
};

template <typename ValueType> compareTraceBox (const TraceBox<ValueType>& b1, const TraceBox<ValueType>& b2)
{
    return ! (b1.operator < (b2));
}


} // namespace genstr

#endif // __TRACEBOX_TEMPL_H__
