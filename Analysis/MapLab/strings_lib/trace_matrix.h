/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __TRACE_MATRIX__
#define __TRACE_MATRIX__

#include <vector>

#include <iostream>

#include <limits.h>
#include <platform.h>
#include <rerror.h>

namespace genstr
{

#define TRACE_STOP 0
#define ALONG_FIRST 1
#define ALONG_SECOND 2
#define ALONG_DIAG 3
typedef int PATH_DIR;

// if 1 is horizontal, 2 is vertical, then
// ALONG_FIRST is from left
// ALONG_SECOND is from below

class OutOfBand : public Rerror {};

template <bool banded = false>
class TraceMatrix
{
public:
    void init (unsigned len1, unsigned len2);
    void put (unsigned idx1, unsigned idx2, PATH_DIR dir);
    PATH_DIR get (unsigned idx1, unsigned idx2);
    void print (unsigned p1 = UINT_MAX, unsigned p2 = UINT_MAX);
};
template <>
class TraceMatrix<false>
{
private:
    unsigned dim1_;
    unsigned dim2_;
    std::vector <unsigned char> data_;
public:
    TraceMatrix () : dim1_ (0), dim2_ (0) {}
    void init (unsigned len1, unsigned len2)
    {
        dim1_ = len1;
        dim2_ = len2;
        unsigned blen = (unsigned) ((((ulonglong) dim1_) * dim2_ + 3) >> 2);
        data_.resize (blen);
        std::fill (data_.begin (), data_.end (), 0);
    }
    void put (unsigned idx1, unsigned idx2, PATH_DIR dir)
    {
        ulonglong off = ((ulonglong) dim2_) * idx1 + idx2;
        unsigned shift = (unsigned) ((off & 0x3) << 1);
        unsigned char& val = data_ [(unsigned) (off >> 2)];
        val &= ~(0x3 << shift);
        val |= (dir << shift);
    }
    PATH_DIR get (unsigned idx1, unsigned idx2)
    {
        ulonglong off = ((ulonglong) dim2_) * idx1 + idx2;
        unsigned shift = (unsigned) ((off & 0x3) << 1);
        unsigned char& val = data_ [(unsigned) (off >> 2)];
        return (val >> shift) & 0x3;
    }
    void print (unsigned p1 = UINT_MAX, unsigned p2 = UINT_MAX, std::ostream& o = std::cout);
};

template <>
class TraceMatrix<true>
{
private:
    unsigned len_;
    unsigned dev_;
    unsigned wid_;
    unsigned unit1_;
    unsigned unit2_;
    std::vector <unsigned char> data_;
    unsigned wid ()
    {
        return (dev_ << 1) + unit2_;
    }
    unsigned offset (unsigned idx1, unsigned idx2)
    {
        unsigned dbox_beg2 = (idx1 / unit1_) * unit2_;
        if (idx2 + dev_ < dbox_beg2 || idx2 >= dbox_beg2 + unit2_ + dev_)
            throw OutOfBand ();
        return idx1 * wid_ + idx2 - dbox_beg2 + dev_;
    }
public:
    TraceMatrix () : len_ (0), dev_ (0), wid_ (0), unit1_ (0), unit2_ (0) {}
    void init (unsigned len, unsigned dev, unsigned unit1, unsigned unit2)
    {
        // len is a length of rectangle-shaped match region,
        // dev is the max distance from diagonal accounted,
        // uint1 is number of elements per unit on axis 1
        // uint2 is number of elements per unit on axis 2
        // the band width thus is dev_*2 + unit2_
        // (yes the lower left and upper right triangles are waisted)
        len_ = len;
        dev_ = dev;
        unit1_ = unit1;
        unit2_ = unit2;
        wid_ = wid ();
        unsigned blen = (unsigned) ((((ulonglong) len_) * wid_ + 3) >> 2);
        data_.resize (blen);
        std::fill (data_.begin (), data_.end (), 0);
    }
    void put (unsigned idx1, unsigned idx2, PATH_DIR dir)
    {
        ulonglong off = offset (idx1, idx2);
        unsigned shift = (unsigned) ((off & 0x3) << 1);
        unsigned char& val = data_ [(unsigned) (off >> 2)];
        val &= ~(0x3 << shift);
        val |= (dir << shift);
    }
    PATH_DIR get (unsigned idx1, unsigned idx2)
    {
        ulonglong off = offset (idx1, idx2);
        unsigned shift = (unsigned) ((off & 0x3) << 1);
        unsigned char& val = data_ [(unsigned) (off >> 2)];
        return (val >> shift) & 0x3;
    }
    void print (unsigned p1 = UINT_MAX, unsigned p2 = UINT_MAX, std::ostream& o = std::cout);
};

}; // namespace genstr

#endif
