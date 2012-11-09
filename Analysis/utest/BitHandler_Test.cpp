/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <fstream>
#include <gtest/gtest.h>
#include "BitHandler.h"

using namespace std;

// Typical Test Case
TEST(BitHandlerTest, BasicTest)
{
  BitPacker bp;
  for (unsigned int i = 1; i < 32; i++) {
    bp.put_bits(i,i);
  }
  bp.put_u8(8);
  bp.put_u16(16);
  bp.put_u32(32);
  bp.put_u32(0);
  bp.flush();
  BitUnpacker bu(bp.get_data());
  for (unsigned int i = 1; i < 32; i++) {
    unsigned int x = bu.get_bits(i);
    ASSERT_EQ(x,i);
  }
  unsigned int x = bu.get_u8();
  ASSERT_EQ(x,8);
  x = bu.get_u16();
  ASSERT_EQ(x,16);
  x = bu.get_u32();
  ASSERT_EQ(x,32);
}

TEST(BitHanderTest,  BasicHuffmanTest) 
{
  int size = 2049;
  uint8_t buff[size];
  for (int i = 0; i < size; i++) {
    if (i % 2 == 0) {
      if (i % 4 == 0) {
        buff[i] = 4;
      }
      else {
        buff[i] = 2;
      }
    }
    else {
      buff[i] = 3;
    }
  }
  BitPacker bp;
  bp.put_compressed(buff, size);
  bp.put_u32(0);
  bp.flush();
  uint8_t out[size];
  BitUnpacker bu(bp.get_data());
  bu.get_compressed(out, size);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(out[i], buff[i]);
  }
}


