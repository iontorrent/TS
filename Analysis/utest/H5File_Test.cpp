/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <stdexcept>
#include <stdio.h>
#include <armadillo>
#include "H5File.h"

using namespace std;
using namespace arma;
TEST(H5Handle_Test, QuickMatrixTest) {
  Mat<float> mymat = ones<Mat<float> >(5,3);
  Mat<float>::iterator i;
  int count = 0;
  for (i = mymat.begin(); i != mymat.end(); i++) {
    *i = count++;
  }
  //  mymat.raw_print(cout);
  H5File::WriteMatrix("tmp.h5:/path/to/mat", mymat);
  Mat<float> m;
  H5File::ReadMatrix("tmp.h5:/path/to/mat", m);
  Mat<float>::iterator gold, test;
  for (gold = mymat.begin(), test = m.begin(); gold != mymat.end(); gold++, test++) {
    float g = *gold;
    float t = *test;
    EXPECT_EQ(t, g);
  }    
}


TEST(H5Handle_Test, MatrixTest) {
  Mat<float> mymat = ones<Mat<float> >(5,3);
  Mat<float>::iterator i;
  int count = 0;
  for (i = mymat.begin(); i != mymat.end(); i++) {
    *i = count++;
  }
  //  mymat.raw_print(cout);
  {
    H5File file;
    file.SetFile("tmp.h5");
    file.Open(true);
    H5DataSet *ds = file.CreateDataSet("/tmp/fun/mat", mymat, 0);
    //H5DataSet *ds = file.CreateDataSet("mat", mymat, 0);
    ds->WriteMatrix(mymat);
    ds->Close();
  }
  {
    H5File file;
    file.SetFile("tmp.h5");
    file.Open();
    H5DataSet *ds = file.OpenDataSet("/tmp/fun/mat");
    //    H5DataSet *ds = file.OpenDataSet("mat");
    Mat<float> m;
    ds->ReadMatrix(m);
    //    m.raw_print(cout);
    Mat<float>::iterator gold, test;
    for (gold = mymat.begin(), test = m.begin(); gold != mymat.end(); gold++, test++) {
      float g = *gold;
      float t = *test;
      EXPECT_EQ(t, g);
    }    
    ds->Close();
  }

 }

TEST(H5Handle_Test, WriteReadTest) {
  float data[12];
  for (size_t i = 0; i < ArraySize(data); i++) {
    data[i] = i;
  }
  {
    H5File file;
    file.SetFile("tmp.h5");
    file.Open(true);
    hsize_t dims[] = {3,4};
    hsize_t chunking[] = {3,4};
    H5DataSet *ds =  file.CreateDataSet("/tmp/here/testdata", 2, dims, chunking, 2, H5T_NATIVE_FLOAT);

    size_t starts[] = {0,0};
    size_t ends[] = {3,4};
    //ds->WriteRangeData(starts, ends, ArraySize(data), data);
	ds->WriteRangeData(starts, ends, data);
    ds->Close();
  }
  { 
    H5File file;
    file.SetFile("tmp.h5");
    file.Open();
    H5DataSet *ds = file.OpenDataSet("/tmp/here/testdata");
    float query[12];
    size_t starts[] = {0,0};
    size_t ends[] = {3,4};
    ds->ReadRangeData(starts, ends, ArraySize(query), query);
    for (size_t i = 0; i < ArraySize(data); i++) {
      if (data[i] != query[i]) {
        EXPECT_EQ(query[i], data[i]);
      }
    }
    
    {
      float update[4];
      size_t starts[] = {1,2};
      size_t ends[] = {3,4};
      size_t count = 0;
      for (size_t i = 11; i >= 8; i--) {
        update[count++] = i;
      }
      //ds->WriteRangeData(starts, ends, ArraySize(update), update);
	  ds->WriteRangeData(starts, ends, update);
      float query[12];
      size_t rstarts[] = {0,0};
      size_t rends[] = {3,4};
      float gold[12] = {0,1,2,3,4,5,11,10,8,9,9,8};
      ds->ReadRangeData(rstarts, rends, ArraySize(query), query);
      for (size_t i = 0; i < 1; i++) {
        EXPECT_EQ(gold[i], query[i]);
      }

    }
    //    ds->Close();
    ds = file.OpenDataSet("shouldbenull");
    EXPECT_EQ(NULL,ds);

  }
}


