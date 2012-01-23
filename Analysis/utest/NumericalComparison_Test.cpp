/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "NumericalComparison.h"
#include <gtest/gtest.h>

TEST(NumericalComarison_Test, FloatCorrelationTest) {
  float x[] = {1,2,3,4,5,6,7};
  float y[] = {-0.06208414,2.85911208,2.94912247,4.97006061,6.30555794,6.09010956,6.36966924};
  int length = sizeof(x) / sizeof(float);
  NumericalComparison<float> compare(.001);
  for (int i = 0; i < length; i++) {
    compare.AddPair(x[i], y[i]);
  }
  ASSERT_EQ(compare.GetNumSame(), 0);
  ASSERT_EQ(compare.GetNumDiff(), length);
  ASSERT_EQ(compare.GetCount(), length);
  double correlation = compare.GetCorrelation();
  ASSERT_NEAR(correlation, 0.932223, .0001);
}

TEST(NumericalComarison_Test, DoubleSimilarTest) {
  double x[] = {1,2,3,4,5,6,7};
  double y[] = {1,2,3.001,4.0011,4.9991,6,7};
  int length = sizeof(x) / sizeof(double);
  NumericalComparison<double> compare(.001);
  for (int i = 0; i < length; i++) {
    compare.AddPair(x[i], y[i]);
  }
  ASSERT_EQ(compare.GetNumSame(), length-1);
  ASSERT_EQ(compare.GetNumDiff(), 1);
  ASSERT_EQ(compare.GetCount(), length);
  double correlation = compare.GetCorrelation();
  ASSERT_NEAR(correlation, 1.0, .0001);
}
