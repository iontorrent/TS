/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <iostream>
#include "ComparatorClean.h"


using namespace std;
using namespace arma;

TEST(ComparatorClean_Test, GetNEigenScatterTest) {
  Mat<float> Y;
  Y.load("comp_clean_test_data.txt");
  Mat<float> E;
  ComparatorClean cc;
  cc.GetNEigenScatter(Y, E, 3);
  //  E.print("Eigenvectors");
  Mat<float> G;
  G.load("comp_clean_test_ev_3.txt");
  float maxD = max(max(abs(G - E)));
  //  cout << "Max difference is: " << maxD << endl;
  EXPECT_NEAR(maxD, 5.03063e-05,.0001);
}

TEST(ComparatorClean_Test, GetEigenProjectionTest) {
  Mat<float> Y;
  Y.load("comp_clean_test_data.txt");
  Mat<float> S; // smoothed
  ComparatorClean cc;
  Col<unsigned int> good(Y.n_rows/2);
  for (size_t i = 0; i < Y.n_rows/2; i++) {
    good(i) = 2*i;
  }
  cc.GetEigenProjection(Y, good, 3, S);
  float meanD = mean(mean(abs(Y - S), 1));
  EXPECT_NEAR(meanD, 6.0434947, .0001);
}

TEST(ComparatorClean_Test, GetComparatorCorrection) {
  Mat<float> Y;
  Y.load("comp_clean_test_data.txt");
  Mat<float> S; // smoothed
  Mat<float> C; // corrections (nmod * cols);
  ComparatorClean cc;
  Col<unsigned int> good(Y.n_rows);
  for (size_t i = 0; i < Y.n_rows; i++) {
    good(i) = i;
  }
  cc.GetEigenProjection(Y, good, 3, S);
  //  S.print("Smoohted.");
  cc.GetComparatorCorrection(Y, good, S, 20, 20, C, 4, 0);
  Col<float> c = C.row(3*20 + 3).t();
  Row<float> g;
  g << -11.9071 << 1.9328 << 1.9328 << 1.9328 << 1.9328 << 1.9328 << 1.9328 << 1.9328 << 1.9328 << -10.1708 << -9.1762 << -7.1357 << -14.2075 << -3.3285 << 31.1593 << 24.4231 << 29.0290 << 23.2491 << 22.0458 << 11.6261 << 7.1955 << -5.6924 << 4.0252 << -2.5189 << -4.1276 << -5.1912 << -0.5605 << -1.0819 << 1.1052 << -13.0457 << -5.3109 << -8.3747 << -7.5303 << -11.1656 << -9.9431 << 0.1885 << -7.0269 << 25.9144 << 25.1729 << 23.4235 << -7.7795 << -6.0722 << -6.0722 << -6.6909 << -6.6902 << -5.1344 << -5.1344 << 1.1332 << 1.1332 << 2.6005 << 2.6005 << -0.0014 << -0.0014 << 0.1638 << 0.1638 << -5.5478 << -5.5478 << 4.1425 << 4.1424 << 7.5510;
  c = abs(c - g.t());
  float d = arma::max(c);
  EXPECT_NEAR(d, 0.0f, .002);
  // c.print("Correction.");
}
