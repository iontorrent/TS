/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <vector>
#include "FindSlopeChange.h"
#include "Utils.h"
#include "T0Model.h"

TEST(T0Model_Test, SegmentFitterTest) {
  float yvals[] = {0,1,2,3,4};
  float xvals[] = {0,1,2,3,4};
  LineModel sf;
  sf.FitModel(&yvals[0], &yvals[0] + 5, &xvals[0], &xvals[0] + 5);
  Col<float> prediction;
  sf.GetPrediction(prediction);
  for (size_t i = 0; i < prediction.n_rows; i++) {
    ASSERT_NEAR(prediction.at(i), i, .001);
  }
  ASSERT_NEAR(sf.SumSqErr(), 0, .0001);
}

TEST(T0Model_Test, T0FinderTest) {
  float yvals[] = {0,-0.325926,-0.937037,-1.54815,-2.17407,-2.8,-3.12593,-3.37037,-3.76296,-4.00741,-4.31852,-4.57778,-4.44074,-4.32222,-4.89259,-5.64444,-7.58889,-10.6704,-10.137,-6.66296,-3.7037,-2.07778,-0.618519,1.4,4.17778,7.67037,11.4333,15.163,14.7185,8.12963,-7.56296,-37.0259,-89.0296,-177.522,-323.696,-543.819,-834.726,-1178.87,-1551.51,-1933.76,-2307.1,-2660.14,-2978.72,-3253.29,-3482.33,-3669.13,-3822.49,-3952.26,-4064.52,-4163.15,-4250.39,-4327,-4394.18,-4454.1,-4509.66,-4558.39,-4601.8,-4641.9,-4677.85,-4710.32,-4739.09,-4765.88,-4789.59,-4811.23,-4830.96,-4847.48,-4862.96,-4875.26,-4875.26,-4875.26,-4875.26,-4875.26,-4875.26};
  size_t size = sizeof(yvals)/sizeof(float);
  float xvals[size];
  for (size_t i = 0; i < size; i++) {
    xvals[i] = i;
  }
  T0Finder t0;
  int windowSize = 7;
  t0.SetFirstSlopeRange(-1,10);
  t0.SetSecondSlopeRange(-10000,-10);
  t0.SetWindowSize(windowSize);
  t0.SetSearchRange(0+windowSize, size-windowSize);
  t0.FindT0(yvals, size);
  float t0Est = t0.GetT0Est();
  cout << "T0 is: " << t0Est << endl;
}
