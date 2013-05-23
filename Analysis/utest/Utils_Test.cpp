/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include "Mask.h"
#include "Utils.h"

using namespace std;

TEST(Utils_Test, FastMedianEvenTest) {
  float vals[10] = {-1,-2, 0.0,0.1,0.2,0.3, 4.0, 10.0,100.0,9.0};
  float f = fast_median(vals, ArraySize(vals));
  EXPECT_NEAR(f, .25f, 0.000001);
}

TEST(Utils_Test, FastMedianOddTest) {
  float vals[11] = {-1,-2, 0.0,0.1,0.2,0.3, 4.0, 10.0,100.0,9.0,50.0};
  float f = fast_median(vals, ArraySize(vals));
  EXPECT_NEAR(f, .3f, 0.000001);
}
