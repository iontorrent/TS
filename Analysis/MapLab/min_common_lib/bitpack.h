/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __bitpack_h__
#define __bitpack_h__

#include "portability.h"
#include "bitops.h"

/// determine size in bytes needed to hold compact array with of given size and element size
/// \param  bits_per_element number of significant bits in the element
/// \param  size array size
/// \return number of bytes the compacted array occupies
template <typename CHUNK> 
inline size_t compact_size (size_t bits_per_element, size_t size)
{
    return (bits_per_element * size + bit_width<CHUNK>::width  - 1) >> ln_bit_width<CHUNK>::width;
}

/// packs the array of values each represented by certain number of bits into the continous bit array
/// \param  TYPE (template parameter) type of the array element, defining the size of individual array element
/// \param  array the array of values to pack
/// \param  elsize number of significant bits in the array element
/// \param  size the number of values to compact
/// \param  dest the memory area where the resulting bit array will be written. Caller have to ensure sufficient space
/// \param  offset the offset in the destination (in elements!) where the packed array should be placed (replacing original content)
/// \return size of the resulting compacted array in CHUNKs
template <typename TYPE, typename CHUNK>
inline size_t bitwise_compact (const TYPE* array, size_t elsize, size_t size, CHUNK* dest, size_t offset = 0)
{
    // compute address in destination
    size_t bit_offset = offset * elsize;
    size_t dest_bitoff = bit_offset % bit_width<CHUNK>::width;
    size_t dest_offset = bit_offset / bit_width<CHUNK>::width;
    CHUNK* chunk = dest + dest_offset;

    size_t unused_bits_in_chunk = bit_width<CHUNK>::width - dest_bitoff;
    size_t used_bits_in_chunk = dest_bitoff;

    size_t unpacked_bits_in_elem = elsize;
    size_t packed_bits_in_elem = elsize - unpacked_bits_in_elem;

    size_t transfer;
    CHUNK zero_mask, transfer_mask;
    
    const TYPE* sent = array + size;
    
    while (array != sent)
    {
        if (!unused_bits_in_chunk)
        {
            ++ chunk;
            unused_bits_in_chunk = bit_width<CHUNK>::width;
            used_bits_in_chunk = 0;
        }
        else if (!unpacked_bits_in_elem)
        {
            ++ array;
            unpacked_bits_in_elem = elsize;
            packed_bits_in_elem = 0;
        }
        else
        {
            transfer = std::min (unpacked_bits_in_elem, unused_bits_in_chunk);
            // transfer_mask = (((CHUNK) 1) << transfer) - 1;
            transfer_mask = ((CHUNK) ~((CHUNK) 0)) >> (bit_width<CHUNK>::width - transfer);
            zero_mask = ~(transfer_mask << used_bits_in_chunk);
            // zero bits in chunk 
            (*chunk) &= zero_mask;
            // copy bits into chunk
            (*chunk) |= ((*array >> packed_bits_in_elem) & transfer_mask) << used_bits_in_chunk;
            unpacked_bits_in_elem -= transfer;
            unused_bits_in_chunk -= transfer;
            used_bits_in_chunk += transfer;
            packed_bits_in_elem += transfer;
        }
    }
    return chunk + (used_bits_in_chunk?1:0) - dest;
}

/// unpacks the continous bit array into array of values each represented by certain number of bits
/// \param  TYPE (template parameter) type of the array element, defining the size of individual array element
/// \param  compact_array the array of bits to unpack
/// \param  elsize number of significant bits in the array element
/// \param  size the number of values to expand
/// \param  dest the memory area where the resulting array will be written. Caller have to ensure sufficient space
/// \param  offset the offset (in elements!) where the unpacking should start
template <typename TYPE, typename CHUNK>
inline void bitwise_expand (CHUNK* compact_array, size_t elsize, size_t size, TYPE* dest, size_t offset = 0)
{
    // compute address in source
    size_t bit_offset = offset * elsize;
    size_t src_offset = bit_offset / bit_width<CHUNK>::width;
    size_t src_bitoff = bit_offset % bit_width<CHUNK>::width;
    
    CHUNK* chunk = compact_array + src_offset;
    
    size_t unseen_bits_in_chunk = bit_width<CHUNK>::width - src_bitoff;
    size_t seen_bits_in_chunk = src_bitoff;
    
    size_t unfilled_bits_in_elem = elsize;
    size_t filled_bits_in_elem = 0;
    
    size_t transfer;
    TYPE zero_mask, transfer_mask;
    
    if (size) 
    {
        TYPE* elem = dest; // not using dest directly just to make debugging easier
        const TYPE* sent = dest + size;
        *elem = 0;

        while (1)
        {
            if (!unseen_bits_in_chunk)
            {
                ++ chunk;
                unseen_bits_in_chunk = bit_width<CHUNK>::width;
                seen_bits_in_chunk = 0;
            }
            else if (!unfilled_bits_in_elem)
            {
                ++ elem;
                if (elem == sent)
                    break;
                *elem = 0;
                unfilled_bits_in_elem = elsize;
                filled_bits_in_elem = 0;
            }
            else
            {
                transfer = std::min (unseen_bits_in_chunk, unfilled_bits_in_elem);
                transfer_mask = ((TYPE) ~((TYPE) 0)) >> (bit_width <TYPE>::width - transfer);
                //zero_mask = ~(transfer_mask << filled_bits_in_elem);
                //(*elem) &= zero_mask ;
                (*elem) |= ((*chunk >> seen_bits_in_chunk) & transfer_mask) << filled_bits_in_elem;
                filled_bits_in_elem += transfer;
                unfilled_bits_in_elem -= transfer;
                seen_bits_in_chunk += transfer;
                unseen_bits_in_chunk -= transfer;
            }
        }
    }
}
#endif // __bitpack_h__