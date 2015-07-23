/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __fundamental_preprocessing_h__
#define __fundamental_preprocessing_h__




template <typename SymbolType, typename SizeType, typename SymbolPtrType = SymbolType*, typename SizePtrType = SizeType* >
class PrefixTable
{
public:
    static void make (SymbolPtrType string, SizeType string_size, SizePtrType table);
};


// The following code assumes use of direct pointers to the SymbolType arrays. 
// To use on compressed data (binary-compressed nucleotide or packed 5 bit/residue aminoacids, or more complex cases,
// the indirect types should be used

// compute the Z-index using Dan Gusfield's method
template < typename SymbolType, typename SizeType, typename SymbolPtrType, typename SizePtrType >
void PrefixTable <SymbolType, SizeType, SymbolPtrType, SizePtrType >::make (SymbolPtrType string, SizeType string_size, SizePtrType table)
{
    // Z-box at K is a longest substrng starting at position K in a string S that matches the prefix of S
    // all indexing is zero-based
    // all intervals are [inclusive, exclusive)
    // Zk is a length of the Z-box at K
    // rk is the right-most boundary of any Z-box for positions k and below 
    // lk is the left boundary of (any) Z-box ending at rk
    // while processing string left-to-right, consecutively computing Zks:
    // r is the rightmost rk so far
    // l is the left boundary of (some) already observed Z-box ending at r

    SizeType l = (SizeType) 0, r = (SizeType) 0;
    if (!string_size) 
        return;
    *table = (SizeType) 0;
    for (SizeType k = (SizeType) 1; k != string_size; ++ k)
    {
        if (k >= r)
        {
            SizeType zk = 0;
            while (string [zk] == string [k + zk] && k + zk != string_size)
                ++zk;
            table [k] = zk;
            if (zk)
            {
                r = k + zk;
                l = k;
            }
        }
        else
        {
            SizeType kprime = k - l;
            SizeType zkprime = table [kprime];
            SizeType zk;
            if (zkprime < r - k)
                zk = zkprime;
            else
            {
                zk = r - k;
                while (string [zk] == string [r] && r != string_size)
                    ++zk, ++r;
                l = k;
            }
            table [k] = zk;
        }
    }
}



#include "fundamental_preprocessing.hpp"

#endif // __fundamental_preprocessing_h__
