/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include "Mask.h"
#include "Utils.h"

using namespace std;

TEST(Mask_Test, HexNeighborsTest) {
  
  Mask m(3794,3792);
  vector<int> wells;
  m.GetNeighbors(0,0,wells);
  EXPECT_EQ(wells.size(), 6);
  EXPECT_EQ(wells[0], -1);
  EXPECT_EQ(wells[2], -1);
  EXPECT_EQ(wells[3], 3794);
  EXPECT_EQ(wells[4], 1);
  EXPECT_EQ(wells[5], -1);
  m.GetNeighbors(1,1,wells);
  EXPECT_EQ(wells[0], 1);
  EXPECT_EQ(wells[3], 7590);
  m.GetNeighbors(2,2,wells);
  EXPECT_EQ(wells[0], 1 * 3794 + 1);
  EXPECT_EQ(wells[3], 3 * 3794 + 2);
  m.GetNeighbors(3791,3793, wells);
  EXPECT_EQ(wells[0], 3790 * 3794 + 3793);
  EXPECT_EQ(wells[4], -1);
}


TEST(Mask_Test, SquareNeighborsTest) {
  
  Mask m(200,200);
  vector<int> wells;
  m.GetNeighbors(0,0,wells);
  EXPECT_EQ(wells.size(), 8);
  EXPECT_EQ(wells[0], -1);
  EXPECT_EQ(wells[1], -1);
  EXPECT_EQ(wells[2], -1);
  EXPECT_EQ(wells[3], 1 * 200);
  EXPECT_EQ(wells[4], 1 * 200 + 1);
  EXPECT_EQ(wells[5],  1);
  EXPECT_EQ(wells[6],  -1);
  EXPECT_EQ(wells[7],  -1);
  // m.GetNeighbors(1,1,wells);
  // m.GetNeighbors(2,2,wells);
  // m.GetNeighbors(199,198, wells);
}

