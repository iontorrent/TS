/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <armadillo>
#include <stdexcept>
#include <time.h>
#include <sys/time.h>
#include "ZeromerDiff.h"
#include "Utils.h"

using namespace std;

TEST(ZeromerDiff_Test, FitZeromer_Test) {
  ZeromerDiff<double> zero;
  vector<double> trace = char2Vec<double>("1.88889 6.41112 16.8633 29.22 47.29 70.7422 95.6467 121.647 149.456 177.647 203.647 229.647 252.933 279.717 310.194 337.003 362.838 384.838 406.386 429.647 456.099 481.29 503.838 525.838 547.386 566.577 589.456 614.029 633.838 655.386 677.742 698.577 717.838 737.125 751.768 766.125 778.959 794.742 817.838 837.125 856.29 870.698 881.029 897.672 913.125 929.125 942.863 959.742 974.246 981.316",' ');
  vector<float> reference = char2Vec<float>("-0.213532 7.64527 25.3417 51.4831 82.3904 115.723 150.318 184.759 218.504 251.411 283.431 314.875 345.461 375.013 403.53 431.286 458.229 484.794 509.376 533.733 557.616 581.489 603.816 626.419 648.572 669.373 689.368 709.753 728.819 747.436 765.859 783.427 800.966 818.181 834.941 850.855 866.776 881.975 897.294 912.312 926.718 940.283 953.925 968.019 981.709 994.399 1007.09 1020.33 1033.46 1044.8", ' ');
  Mat<double> wells(reference.size(), 1);
  copy(trace.begin(), trace.end(), wells.begin_col(0));
  Mat<double> refs(reference.size(), 1);
  copy(reference.begin(), reference.end(), refs.begin_col(0));
  uvec zeroFlows(1);
  zeroFlows(0) = 0;
  Col<double> time(reference.size());
  for (size_t i = 0; i < reference.size(); i++) {
    time(i) = i;
  }
  Col<double> param;
  // int N = 10000000; // 10,000,000
  int N = 1;
  struct timeval st;
  gettimeofday(&st, NULL);
  for (int i = 0; i < N; i++) {
    zero.FitZeromer(wells, refs, zeroFlows, time, param);
  }
  struct timeval et;
  gettimeofday(&et, NULL);
  //double mil = 1000000.0;
	//  cout << "For: " << N << " interations: " << ((et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec))/mil << " seconds." << endl;
  //  cout << "Param are: " << param(0) << " " << param(1) << endl;
  Col<double> z;
  zero.PredictZeromer(refs, time, param, z);
  Col<double> diff = z - wells;
	//  diff.raw_print();
  double sum = 0;
  for (size_t i = 0; i < diff.n_rows; i++) {
    sum += diff(i) * diff(i);
  }
  EXPECT_NEAR(param.at(0), 8.89934, .01);
  EXPECT_NEAR(param.at(1), 4.73301, .01);
  EXPECT_NEAR(sum/diff.n_rows, 8.77863, .01);

}

TEST(ZeromerDiff_Test, FitZeromer2_Test) {
  ZeromerDiff<double> zero;
  vector<double> trace = char2Vec<double>("-4 -4 1 1 5 5 11 17 22 20 27 24 29 43 53 57 68 67 70 72 95 88 111 111 125 125 131 152 152 152 170 184 195 193 204 215 223 230 236 252 265 277 268 284 290 277 302 315 318 323",' ');
  vector<float> reference = char2Vec<float>("-2.7185 -2.24417 -0.955165 1.38792 4.4125 8.11103 12.4606 17.5764 22.815 28.9851 35.6587 42.765 49.9863 58.0229 65.7982 74.4877 82.7154 91.0914 99.9222 108.62 117.3 126.273 135.148 144.027 153.086 162.028 170.928 179.823 188.656 197.746 206.507 214.895 223.638 232.346 240.722 249.26 257.695 265.959 274.465 282.914 290.855 298.583 306.587 314.701 322.283 330.698 338.589 346.491 354.082 361.79", ' ');
  Mat<double> wells(reference.size(), 1);
  copy(trace.begin(), trace.end(), wells.begin_col(0));
  Mat<double> refs(reference.size(), 1);
  copy(reference.begin(), reference.end(), refs.begin_col(0));
  uvec zeroFlows(1);
  zeroFlows(0) = 0;
  Col<double> time(reference.size());
  for (size_t i = 0; i < reference.size(); i++) {
    time(i) = i;
  }
  Col<double> param;
  // int N = 10000000; // 10,000,000
  int N = 1;
  struct timeval st;
  gettimeofday(&st, NULL);
  for (int i = 0; i < N; i++) {
    zero.FitZeromer(wells, refs, zeroFlows, time, param);
  }
  struct timeval et;
  gettimeofday(&et, NULL);
	//  double mil = 1000000.0;
	//  cout << "For: " << N << " interations: " << ((et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec))/mil << " seconds." << endl;
	//  cout << "Param are: " << param(0) << " " << param(1) << endl;
  Col<double> z;
  zero.PredictZeromer(refs, time, param, z);
  Col<double> diff = z - wells;
	//  diff.raw_print();
  double sum = 0;
  for (size_t i = 0; i < diff.n_rows; i++) {
    sum += diff(i);
  }
  EXPECT_NEAR(param.at(0), 6.18843, .01);
  EXPECT_NEAR(param.at(1), 2.2967, .01);
  EXPECT_NEAR(sum/diff.n_rows, .491809, .01);

}


TEST(ZeromerDiff_Test, FitZeromer3_Test) {
  ZeromerDiff<double> zero;
  vector<double> trace = char2Vec<double>("-4 -4 1 1 5 5 11 17 22 20 27 24 29 43 53 57 68 67 70 72 95 88 111 111 125 125 131 152 152 152 170 184 195 193 204 215 223 230 236 252 265 277 268 284 290 277 302 315 318 323",' ');
  vector<float> reference = char2Vec<float>("-2.7185 -2.24417 -0.955165 1.38792 4.4125 8.11103 12.4606 17.5764 22.815 28.9851 35.6587 42.765 49.9863 58.0229 65.7982 74.4877 82.7154 91.0914 99.9222 108.62 117.3 126.273 135.148 144.027 153.086 162.028 170.928 179.823 188.656 197.746 206.507 214.895 223.638 232.346 240.722 249.26 257.695 265.959 274.465 282.914 290.855 298.583 306.587 314.701 322.283 330.698 338.589 346.491 354.082 361.79", ' ');
  Mat<double> wells(reference.size(), 3);
  copy(trace.begin(), trace.end(), wells.begin_col(0));
  copy(trace.begin(), trace.end(), wells.begin_col(1));
  copy(trace.begin(), trace.end(), wells.begin_col(2));
  Mat<double> refs(reference.size(), 3);
  copy(reference.begin(), reference.end(), refs.begin_col(0));
  copy(reference.begin(), reference.end(), refs.begin_col(1));
  copy(reference.begin(), reference.end(), refs.begin_col(2));
  uvec zeroFlows(3);
  zeroFlows(0) = 0;
  zeroFlows(1) = 1;
  zeroFlows(2) = 2;
  Col<double> time(reference.size());
  for (size_t i = 0; i < reference.size(); i++) {
    time(i) = i;
  }
  Col<double> param;
  // int N = 10000000; // 10,000,000
  int N = 1;
  struct timeval st;
  gettimeofday(&st, NULL);
  for (int i = 0; i < N; i++) {
    zero.FitZeromer(wells, refs, zeroFlows, time, param);
  }
  struct timeval et;
  gettimeofday(&et, NULL);
	//  double mil = 1000000.0;
	//  cout << "For: " << N << " interations: " << ((et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec))/mil << " seconds." << endl;
	//  cout << "Param are: " << param(0) << " " << param(1) << endl;
  Col<double> z;
  zero.PredictZeromer(refs.unsafe_col(0), time, param, z);
  Col<double> diff = z - wells.col(0);
	//  diff.raw_print();
  double sum = 0;
  for (size_t i = 0; i < diff.n_rows; i++) {
    sum += diff(i);
  }
  EXPECT_NEAR(param.at(0), 6.19398, .01);
  EXPECT_NEAR(param.at(1), 2.30151, .01);
  EXPECT_NEAR(sum/diff.n_rows, 0.49069, .01);

}
