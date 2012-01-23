/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include "DualGaussMixModel.h"
 
#define  PI 3.1415926535897932384626433832795


/* Quick test - Just run a fuction and see if value is expected */
TEST(DualGaussMixModel_Test, DNorm_Test) {
  double z = DualGaussMixModel::DNorm(.35, 0, 1, 1.0/sqrt(2*PI));
  ASSERT_NEAR(z, 0.3752403, .00001);
}
 
/* Class with a SetUp() function for test fixtures, could also have
   a TearDown() */
class DualGaussMixModelTest : public ::testing::Test {
protected :
 
  virtual void SetUp() {
    // Silly example - copy data from array to vector
    float d[] = {-0.00199, 0.42323, 0.69296,-0.05181,-0.75783,
                 0.96412,-2.71104, 1.01185, 0.81247,-1.35673,
                 3.26636, 2.51157, 3.07927, 1.74444, 2.08672,
                 3.53636, 3.30963, 2.56633, 5.52059, 3.64384};
    int length = sizeof(d)/sizeof(float);
    mData.resize(length);
    std::copy(&d[0], &d[0] + length, mData.begin());
  }
 
  std::vector<float> mData;
 
};
 
/* Test using test fixture DualGaussMixModelTest with SetUp() function
   called by framework. Note TEST_F() rather than TEST() macro */
TEST_F(DualGaussMixModelTest, MixFit) {
  DualGaussMixModel model(mData.size());
  MixModel m = model.FitDualGaussMixModel(&mData[0], mData.size());
  ASSERT_NEAR(m.mix, .346, .001);
  ASSERT_NEAR(m.mu1, .6645, .001);
  ASSERT_NEAR(m.mu2, 3.15, .01);
  ASSERT_NEAR(m.var1, 1.34, .01);
  ASSERT_NEAR(m.var2, .2117, .001);
}
