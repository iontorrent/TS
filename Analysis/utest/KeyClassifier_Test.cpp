/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <armadillo>
#include <stdexcept>
#include <time.h>
#include <sys/time.h>
#include "ZeromerDiff.h"
#include "KeyClassifier.h"
#include "Utils.h"

using namespace std;

TEST(KeyClassifier_Test, ProjectionTest) {
  Col<double> well;
  well << 1 << 1;
  Col<double> incorporation;
  incorporation << 1 << 0;
  double p = KeyClassifier::GetProjection(incorporation, well);
  ASSERT_NEAR(p, 1, .0001);// 1.110721
  well <<  0.2193781 <<  1.9257150 <<  -0.0256586 <<  0.3911019;
  incorporation <<   0.7616795 <<  -0.8993396 << -0.1143299 << -1.3601566;
  p = KeyClassifier::GetProjection(well, incorporation);
  ASSERT_NEAR(-2.0938, p , .0001);

}

TEST(KeyClassifier_Test, Classify_Test) {
  //                                    0 1 2 3 4 5 6 7
  //                                    T A C G T A C G
    vector<int> libKey = char2Vec<int>("1 0 1 0 0 1 0", ' ');
    vector<int> tfKey = char2Vec<int> ("0 1 0 0 1 0 1", ' ');
    vector<KeySeq> keys;
    KeySeq lKey, tKey;
    lKey.name = "lib";
    lKey.flows = libKey;
    lKey.zeroFlows.set_size(4);
    lKey.zeroFlows << 1 << 3 << 4 << 6 ;
    keys.push_back(lKey);
    tKey.name = "tf";
    tKey.flows = tfKey;
    tKey.zeroFlows.set_size(4);
    tKey.zeroFlows << 0 << 2 << 3 << 5;
    keys.push_back(tKey);

    ZeromerDiff<double> bg;
    Mat<double> wellFlows;
    wellFlows.load("traces.txt",raw_ascii);
    wellFlows = trans(wellFlows);
    Mat<double> refFlows;
    refFlows.load("reference.txt", raw_ascii);
    refFlows = trans(refFlows);
    Col<double> time(refFlows.n_rows);
    for (size_t i = 0; i < time.n_rows; i++) {
      time[i] = i;
    }
    KeyFit fit;
    Mat<double> predicted(wellFlows.n_rows, wellFlows.n_cols);
    vector<float> reference;
    vector<float> signal;
		//    KeyClassifier::PrintVec(wellFlows);
		//    KeyClassifier::PrintVec(refFlows);
    //KeyClassifier::Classify(keys, bg, wellFlows, refFlows, reference, signal, time, 4, fit, predicted);
    KeyClassifier kClass;
    kClass.Classify(keys, bg, wellFlows, refFlows, time, 4, fit, predicted);
    Mat<double> diff = wellFlows - predicted;
    // cout << "Diff is: " << endl;
    // diff.raw_print();
    uvec zeroFlows(2);
    zeroFlows(0) = 0;
    zeroFlows(1) = 1;
    //    cout << "Index is: " <<  fit.keyIndex << " is: " << fit.snr << endl;
    Col<double> param;
    Mat<double> subW = wellFlows.submat(0,3,49,4);
    Mat<double> subR = refFlows.submat(0,3,49,4);
    bg.FitZeromer(subW, subR,zeroFlows, time, param);
    Col<double> z;
    bg.PredictZeromer(refFlows.col(1), time, param, z);
    Col<double> d = wellFlows.col(1) - z;
		//    cout << "diff is: " << endl;
    //    d.raw_print();
    bg.PredictZeromer(refFlows.col(3), time, param, z);
    d = wellFlows.col(3) - z;
		//    cout << "diff is: " << endl;
    //    d.raw_print();
}
