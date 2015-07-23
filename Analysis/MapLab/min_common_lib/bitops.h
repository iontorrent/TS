/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __bitops_h__
#define __bitops_h__

/// \file Bit operations on scalar values, both run-time and compile-time
///

#include <ostream>
#include <climits>
#include <limits>
#include <cstdlib>
#include "platform.h"
#include "tracer.h"
#include "common_str.h"


/// bits per byte
#define BITS_PER_BYTE CHAR_BIT
// log2 of bits per byte 
#define BITS_PER_BYTE_SHIFT 3
/// bits per QWORD (8-byte) value
#define BITS_PER_QWORD (BITS_PER_BYTE * sizeof (QWORD))

/// bit width for a value of the type
template <typename T>
class bit_width
{
public:
    enum { width = sizeof (T) * BITS_PER_BYTE };
    static size_t v () { return width; }
};

template <int SZ>
class EXBITW_
{
public:
    enum { exbitw = EXBITW_ < (SZ+1)/2 >::exbitw + 1 };
};

template <>
class EXBITW_ < 1 >
{
public:
    enum { exbitw = BITS_PER_BYTE_SHIFT };
};

/// logarithmic bit width - how many address bits are 'eaten up' by an instance of given type
template <typename T>
class ln_bit_width
{
public:
    enum { width = EXBITW_ < sizeof (T) >::exbitw };
    static size_t v () { return width; }
};

/// compile-time countig of set bits
template <QWORD value>
class set_bits
{
public:
    /// number of set bits in the value used as template argument
    enum
    {
        number = (value & 1) + (unsigned) set_bits < value / 2 >::number
    };
    static size_t v () { return number; }
};
template <>
class set_bits <0>
{
public:
    enum 
    {
        number = 0
    };
};

/// compile-time countig of significant bits
template <QWORD value>
class significant_bits
{
public:
    /// number of significant bits in the value used as template argument
    /// same as the index of highest set bit + 1
    enum
    { 
        number = 1 + (unsigned) significant_bits < value / 2 >::number
    };
    static size_t v () { return number; }
};
template <>
class significant_bits <0>
{
public:
    enum 
    {
        number = 0
    };
};

/// run-time counting of significant bits
/// \param value to examine
/// \return number of set bits
template <typename T>
inline BYTE count_significant_bits (T value)
{
    BYTE cnt = 0;
    while (value) value >>= 1, ++cnt;
    return cnt;
}
/// run-time counting of set bits
/// \param value to examine
/// \return number of significant bits (same as highest set bit index + 1)
template <typename T>
inline BYTE count_set_bits (T value)
{
    BYTE cnt = 0;
    while (value) { if (value & 1) ++cnt; value >>= 1; }
    return cnt;
}
/// run-time counting of unset (zero) bits
/// \param value to examine
/// \return number of bits having the value of zero 
template <typename T>
inline BYTE count_unset_bits (T value)
{
    return sizeof (T) * BITS_PER_BYTE - count_set_bits <T> (value);
}

/// writes the bit-wise representation of passed value to the given stream
///
/// uses C-style byte layout: writes "most significant" component first.
/// This is opposite to the memory layout on little-endian (intel-based) CPUs
/// this template is layout-agnostic, it does not change order depending on 'endiness' of architecture
/// To get byte order that matches little-endian memory layout, pass rev as 'false'
/// \param value to print
/// \param ostr  stream to send the output to
/// \param rev   flag indicating (reverse) direction, when set to False, bits are printed lowest first (leftmost)
template <typename TYPE>
inline void print_bits (TYPE& value, std::ostream& ostr, bool rev = true)
{
    BYTE* base = (BYTE*) &value;
    unsigned begb = rev ?  sizeof (TYPE) : 1;
    unsigned endb = rev ?  0 : sizeof (TYPE) + 1;
    unsigned step = rev ? -1 : 1 ;
    for (unsigned bpos = begb; bpos != endb; bpos += step)
    {
        if (bpos != begb) ostr << SPACE_STR;
        print_bits (base [bpos-1], ostr, rev);
    }
}
// specialization of print_bits for 1-byte values
template <>
inline void print_bits <BYTE> (BYTE& val, std::ostream& o, bool)
{
    // assume all architectures are "bitwise little-endian" (Most Significant Bit is on the left)
    for (unsigned pos = BITS_PER_BYTE; pos != 0; --pos)
        o << ((val >> (pos-1)) & 1);
}

/// writes the byte-wise (hexadecimal) representation of passed value to the given stream
///
/// uses C-style byte layout: writes "most significant" component first.
/// This is opposite to the memory layout on little-endian (intel-based) CPUs
/// this template is layout-agnostic, it does not change order depending on 'endiness' of architecture
/// To get byte order that matches little-endian memory layout, pass rv as 'false'
/// \param value to print
/// \param ostr  stream to send the output to
/// \param rev   flag indicating (reverse) direction, when set to False, bytes are printed lowest first (leftmost)
template <typename TYPE>
inline void print_bytes (TYPE& value, std::ostream& ostr, bool rev = true)
{
    BYTE* base = (BYTE*) &value;
    unsigned begb = rev ?  sizeof (TYPE) : 1;
    unsigned endb = rev ?  0 : sizeof (TYPE) + 1;
    unsigned step = rev ? -1 : 1 ;
    char fillchar = ostr.fill ();
    for (unsigned bpos = begb; bpos != endb; bpos += step)
    {
        if (bpos) ostr << SPACE_STR;
        ostr << std::hex << std::setfill ('0') << std::setw (2) << (unsigned) base [bpos-1];
    }
    ostr.fill (fillchar);
}

/// helper class for printing out 32-bit masks of given length in bits
struct MASK
{
    public:
        /// value of the mask
        DWORD mask_;
        /// length of the mask in bits
        BYTE l_;
        /// constructor
        /// \param m value of the mask
        /// \param l length of the mask in bits
        MASK (DWORD m, BYTE l)
        :
        mask_ (m),
        l_ (l)
        {
        }
        /// default constructor, creates zero-length mask
        MASK ()
        :
        l_ (0)
        {
        }
};
/// outputs bit-structure of a mask to a stream
/// \param ostr stream to send the output to
/// \param mask mask value to print, instance of MASK struct
/// \return the passed in stream, allowing to chain stream insertion operators
inline std::ostream& operator << (std::ostream& ostr, const MASK& mask)
{
    DWORD mm = mask.mask_ & ((~ (DWORD) 0) >> (sizeof (DWORD) * 8 - mask.l_));
    print_bits (mm, ostr);
    return ostr;
}
/// outputs bit-structure of a mask to a logger
/// \param logger logger to send the output to
/// \param mask mask value to print, instance of MASK struct
/// \return the passed in logger, allowing to chain logger insertion operators
inline Logger& operator << (Logger& logger, const MASK& mask)
{
    if (logger.enabled ())
        logger.o_ << mask;
    return logger;
}

/// computes the index of the lowest set bit (starting from the most significant one)
/// if no bits set, returns index that is larger then any bit position in the TYPE's value
/// \param val value for which hihest set bit index is computed
template <typename TYPE>
inline BYTE last_set (TYPE val)
{
    if (val == 0)
        return ~0;
    BYTE rv = bit_width <TYPE>::width;
    while (--rv && !(val & 1))
        val >>= 1;
    return rv;
}

template <typename TYPE>
TYPE randval ()
{
    const size_t randsz = sizeof rand ();
    const size_t typesz = sizeof (TYPE);
    const size_t randbitsz = randsz * BITS_PER_BYTE;
    const size_t compno = (typesz + randsz - 1) / randsz;
    TYPE randval = (TYPE) 0;
    size_t shift = 0;
    for (size_t comp = 0; comp != compno; ++comp)
    {
        randval |= ((TYPE) rand ()) << shift;
        shift += randbitsz;
    }
    return randval;
}

#endif // __bitops_h__
