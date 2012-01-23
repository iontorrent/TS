/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <stdexcept>
#include <stdio.h>
#include "MergeAcq.h"
#include "Acq.h"

using namespace std;

TEST(MergeAcq_Test, TopMergeTest) {
  Image top;
  Image bottom;
  Image combo;
  MergeAcq merger;
  // Load up our test files
  const char *file = "test.dat";
  ION_ASSERT(top.LoadRaw(file), "Couldn't load file.");
  ION_ASSERT(bottom.LoadRaw(file), "Couldn't load file.");
  merger.SetFirstImage(&bottom);
  merger.SetSecondImage(&top, bottom.GetRows(), 0); // starting vertically raised but columns the same.
  // Merge the two images into a single image
  merger.Merge(combo);
  cout << "Done." << endl;
  // Write out the merged image for some testing
  Acq acq;
  acq.SetData(&combo);
  acq.WriteVFC("combo.dat", 0, 0, combo.GetCols(), combo.GetRows());
  Image test;
  ION_ASSERT(test.LoadRaw("combo.dat"), "Couldn't load file.");
  // Check to make sure that the combined file has same values as original
  for (int row = 0; row < top.GetRows(); row++) {
    for (int col = 0; col < top.GetCols(); col++) {
      for (int frame = 0; frame < top.GetFrames(); frame++) {
        EXPECT_EQ(top.At(row, col, frame), bottom.At(row,col,frame));
        short orig = top.At(row, col, frame);
        short combined = test.At(row+top.GetRows(),col,frame);
        EXPECT_EQ(orig, combined);
        combined = test.At(row,col,frame);
        EXPECT_EQ(orig, combined);
      }
    }
  }
}
