/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BYTESWAPUTILS_H
#define BYTESWAPUTILS_H

#ifdef GNUC
#include <inttypes.h>
#endif

#define BYTE_SWAP_2(n) (uint16_t)((((n)>>8) | ((n)<<8)))
#define BYTE_SWAP_4(n) (uint32_t)(((((n)&0xff)<<24) | (((n)&0xff00)<<8) | (((n)&0xff0000)>>8) | (((n)&0xff000000)>>24)))
#define BYTE_SWAP_8(n) (((uint64_t)(BYTE_SWAP_4((uint32_t)((n)&0xffffffff)))<<32) | BYTE_SWAP_4((uint32_t)((n)>>32)))

#define ByteSwap2(n) (n = BYTE_SWAP_2(n))
#define ByteSwap4(n) (n = BYTE_SWAP_4(n))
#define ByteSwap8(n) (n = BYTE_SWAP_8(n))

#endif // BYTESWAPUTILS_H
