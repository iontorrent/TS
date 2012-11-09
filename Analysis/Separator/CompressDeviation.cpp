/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <armadillo>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "IonErr.h"
#include "Image.h"
#include "Mask.h"
#include "OptArgs.h"
#include "SampleQuantiles.h"

using namespace arma;
using namespace std;

#define D '\t'
#define NUM_SAMPLE 10000
#define MAX_MILL_SEC 80
class CompressReporter {
public: 
  virtual void Init() = 0;
  virtual void Report(int rowStart, int rowEnd, int colStart, int colEnd, 
                      int flow, const std::vector<int> &wellMapping,
                      Mask &mask,
                      const Mat<float> &raw, const Mat<float> &compressed) = 0;
  virtual void Finish() = 0;
};

class WellReporter : public CompressReporter {
public:
  WellReporter(const std::string &fileOut) {
    mFileName = fileOut;
    mMad.Init(10000);
    mAutoCor.Init(10000);
  }
  void Init() { 
    mOut.open(mFileName.c_str()); 
    mOut << "well\trow\tcol\tmad\tabs25\tabs75\trms\tacor" << endl;
  }

  void Report(int rowStart, int rowEnd, int colStart, int colEnd, 
              int flow, const std::vector<int> &wellMapping,
              Mask &mask,
              const Mat<float> &raw, const Mat<float> &compressed) {
    Mat<float> diff = raw - compressed;
    for (size_t row = 0; row < raw.n_rows; row++) {
      SampleQuantiles<float> q(diff.n_cols);
      SampleStats<float>  s;
      SampleStats<float>  m;
      for (size_t col = 0; col < diff.n_cols; col++) {
        float val = diff(row, col);
        m.AddValue(val);
        q.AddValue(fabs(val));
        s.AddValue(val*val);
      }
      SampleStats<float> a;
      double mean = m.GetMean();
      for (size_t col = 1; col < diff.n_cols; col++) {
        float val1 = diff(row, col);
        float val2 = diff(row, col-1);
        a.AddValue((val1 - mean) * (val2 - mean));
      }
      double acor = a.GetMean() / m.GetVar();
      mAutoCor.AddValue(acor);
      mMad.AddValue(q.GetQuantile(.5));
      mOut << wellMapping[row] << D << wellMapping[row] / mask.W() << D << wellMapping[row] % mask.W() << D
           << q.GetQuantile(.5) << D << q.GetQuantile(.25) << D << q.GetQuantile(.75) << D << sqrt(s.GetMean()) << D << acor << endl;
    }
  }

  virtual void Finish() { 
    cout << "Quantiles\tmad\tautocor" << endl;
    for (int i = 0; i <= 10; i++) {
      cout << i * 10 << D << mMad.GetQuantile(i / 10.0) << D << mAutoCor.GetQuantile(i / 10.0) << endl;
    }
    mOut.close(); 
  }
  SampleQuantiles<float> mMad;
  SampleQuantiles<float> mAutoCor;
  std::string mFileName;
  std::ofstream mOut;
};


class RegionIndividualReporter : public CompressReporter {
public:
  RegionIndividualReporter(const std::string &fileOut, int rowStart, int rowEnd, int colStart, int colEnd) {
    mFileName = fileOut;
    mRowStart = rowStart;
    mRowEnd = rowEnd;
    mColStart = colStart;
    mColEnd = colEnd;
    mFirst = true;
  }

  void Init() { 
    mOut.open(mFileName.c_str()); 
    mOut << "well\ttype";
  }

  void Report(int rowStart, int rowEnd, int colStart, int colEnd, 
              int flow, const std::vector<int> &wellMapping,
              Mask &mask,
              const Mat<float> &raw, const Mat<float> &compressed) {
    if (rowStart == mRowStart && rowEnd == mRowEnd && colStart == mColStart && colEnd == mColEnd) {
      if (mFirst) {
        for (size_t col = 0; col < raw.n_cols; col++) {
          mOut << D << "col." << col;
        }
        mOut << endl;
        mFirst = false;
      }
      for (size_t row = 0; row < raw.n_rows; row++) {
        mOut << wellMapping[row] << D << "raw";
        for (size_t col = 0; col < raw.n_cols; col++) {
          mOut << D << raw(row,col);
        }
        mOut << endl;
      }
      for (size_t row = 0; row < raw.n_rows; row++) {
        mOut << wellMapping[row] << D << "compressed";
        for (size_t col = 0; col < raw.n_cols; col++) {
          mOut << D << compressed(row,col);
        }
        mOut << endl;
      }
    }
  }

  virtual void Finish() { mOut.close(); }
  bool mFirst;
  int mRowStart, mRowEnd, mColStart, mColEnd;
  std::string mFileName;
  std::ofstream mOut;
};


class DeviationSummaryReporter : public CompressReporter {
public:
  DeviationSummaryReporter(const std::string &fileOut, const std::string &runName) {
    mFileName = fileOut;
    mRunName = runName;
  }

  void Init() { 
    mQuantiles.Init(10000);
    mSample.Clear();
    mSqSample.Clear();
  }

  void Report(int rowStart, int rowEnd, int colStart, int colEnd, 
              int flow, const std::vector<int> &wellMapping,
              Mask &mask,
              const Mat<float> &raw, const Mat<float> &compressed) {
    Mat<float> diff = raw - compressed;
    for (size_t row = 0; row < raw.n_rows; row++) {
      for (size_t col = 0; col < diff.n_cols; col++) {
        float val = fabs(diff(row, col));
        mQuantiles.AddValue(val);
        mSqSample.AddValue(val*val);
        mSample.AddValue(val);
      }
    }
  }

  virtual void Finish() { 
    mOut.open(mFileName.c_str()); 
    mOut << "{" << endl;;
    mOut << "  run_name : " << mRunName << "," << endl;
    mOut << "  fabs_mean : " << mSample.GetMean() << ", "  << endl;
    mOut << "  fabs_sd : " << mSample.GetSD() << ", " << endl;
    mOut << "  rms : " << sqrt(mSqSample.GetMean()) << ", "  << endl;
    mOut << "  fabs_median : " << mQuantiles.GetQuantile(.5) << ", " << endl;
    mOut << "  fabs_iqr : " << mQuantiles.GetIQR() << ", "  << endl;
    mOut << "  fabs_quantiles: [";
    for (size_t i = 0; i < 10; i++) {
      float q = i / 10.0f;
      mOut << mQuantiles.GetQuantile(q) << ",";
    }
    mOut << mQuantiles.GetQuantile(1.0) <<  "], "  << endl;
    mOut << "  fabs_quantiles_100: [";
    for (size_t i = 0; i < 100; i++) {
      float q = i / 100.0f;
      mOut << mQuantiles.GetQuantile(q) << ",";
    }
    mOut << mQuantiles.GetQuantile(1.0) <<  "]"  << endl;
    mOut << "}" << endl;
    mOut.close(); 
  }

  std::string mFileName;
  SampleQuantiles<float> mQuantiles;
  SampleStats<float> mSqSample;
  SampleStats<float> mSample;
  std::string mRunName;
  std::ofstream mOut;
};


class FrameReporter : public CompressReporter {
public:
  FrameReporter(const std::string &fileOut) {
    mFileName = fileOut;
    mStarted = false;
  }
  void Init() { 
    mOut.open(mFileName.c_str()); 
    mOut << "rowStart\trowEnd\tcolStart\tcolEnd\tnumWells\tflow\tstat";
    mStarted = false;
  }

  void Report(int rowStart, int rowEnd, int colStart, int colEnd, 
              int flow, const std::vector<int> &wellMapping,
              Mask &mask,
              const Mat<float> &raw, const Mat<float> &compressed) {
    if (!mStarted) {
      for (size_t i = 0; i < raw.n_cols; i++) {
        mOut << D << "frame." << i;
      }
      mOut << endl;
      mQuantiles.resize(raw.n_cols);
      mStats.resize(raw.n_cols);
      mStarted = true;
    }
    for (size_t i = 0; i < raw.n_cols; i++) {
      mQuantiles[i].Clear();
      mQuantiles[i].Init(raw.n_rows);
      mStats[i].Clear();
    }
    Mat<float> diff = raw - compressed;
    for (size_t row = 0; row < diff.n_rows; row++) {
      for (size_t col = 0; col < diff.n_cols; col++) {
        float val = fabs(diff(row, col));
        mQuantiles[col].AddValue(val);
        mStats[col].AddValue(val);
      }
    }
    mOut << rowStart << D << rowEnd << D << colStart << D << raw.n_rows << D << flow << D << "Fabs.Q25";
    for (size_t i = 0; i < mQuantiles.size(); i++) { mOut << D << mQuantiles[i].GetQuantile(.25); } 
    mOut << std::endl;
    mOut << rowStart << D << rowEnd << D << colStart << D << raw.n_rows << D << flow << D << "Fabs.Q50";
    for (size_t i = 0; i < mQuantiles.size(); i++) { mOut << D << mQuantiles[i].GetQuantile(.5); } 
    mOut << std::endl;
    mOut << rowStart << D << rowEnd << D << colStart << D << raw.n_rows << D << flow << D << "Fabs.Q75";
    for (size_t i = 0; i < mQuantiles.size(); i++) { mOut << D << mQuantiles[i].GetQuantile(.75); } 
    mOut << std::endl;
    mOut << rowStart << D << rowEnd << D << colStart << D << raw.n_rows << D << flow << D << "Fabs.IQR";
    for (size_t i = 0; i < mQuantiles.size(); i++) { mOut << D << mQuantiles[i].GetQuantile(.75) - mQuantiles[i].GetQuantile(.25); } 
    mOut << std::endl;
    mOut << rowStart << D << rowEnd << D << colStart << D << raw.n_rows << D << flow << D << "Fabs.Mean";
    for (size_t i = 0; i < mStats.size(); i++) { mOut << D << mStats[i].GetMean(); } 
    mOut << std::endl;
    mOut << rowStart << D << rowEnd << D << colStart << D << raw.n_rows << D << flow << D << "Fabs.SD";
    for (size_t i = 0; i < mStats.size(); i++) { mOut << D << mStats[i].GetSD(); } 
    mOut << std::endl;
  }

  virtual void Finish() { mOut.close(); }
  
  std::vector<SampleQuantiles<float> > mQuantiles;
  std::vector<SampleStats<float>  > mStats;
  bool mStarted;
  std::string mFileName;
  std::ofstream mOut;
};

class CompressOpts {
public:
  int rowStart, rowEnd, colStart, colEnd;
  int rowStep, colStep;
  double minPercent;
  std::string datIn;
  std::string maskFile;
  int flowNum;
  std::string prefixOut;
  std::string wellOut;
  std::string frameOut;
  std::string runName;
  int numBasis;
};

void AddMatVector(Mat<float> &M, const Col<float> &vec) {
  for (size_t i = 0; i < M.n_rows; i++) {
    for (size_t j = 0; j < M.n_cols; j++) {
      M(i,j) = M(i,j) + vec(i);
    }
  }
}

void Compress(const Mat<float> &raw, int numBasis, Mat<float> &compressed) {
  Col<float> dc = mean(raw, 1);
  Mat<float> X = raw;
  dc = dc * -1;
  AddMatVector(X, dc);
  Mat<float> Cov = X.t() * X;    // L x L
  Mat<float> EVec;
  Col<float> EVal;
  eig_sym(EVal, EVec, Cov);
  Mat<float> V(EVec.n_rows, numBasis );  // L x numBasis
  int count = 0;
  for(size_t v = V.n_rows - 1; v >= V.n_rows - numBasis; v--) {
    copy(EVec.begin_col(v), EVec.end_col(v), V.begin_col(count++));
  }
  Mat<float> A = X * V;  // N x numBasis
  Mat<float> P = A * V.t();  // Prediction N x L 
  dc = dc * -1;
  AddMatVector(P, dc);
  compressed = P;
}

int LoadData(int rowStart, int rowEnd, int colStart, int colEnd,
              Mask &mask, Image &img, vector<int> &mapping, Mat<float> &raw) {
  int numOk = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      if (!mask.Match(col, row, MaskPinned) && !mask.Match(col, row, MaskExclude)) {
        numOk++;
        mapping.push_back(mask.ToIndex(row, col));
      }
    }
  }
  const RawImage *rawImg = img.GetImage(); 
  cout << "Timestamps:" << endl;
  vector<bool> colOk(img.GetFrames(), false);
  int count = 0;
  if (rawImg->timestamps[0] < MAX_MILL_SEC) {
    count++; 
    colOk[0] = true;
  }
  for (int i = 1; i < img.GetFrames(); i++) {
    cout << i << ":\t" << rawImg->timestamps[i] << "\t" << rawImg->timestamps[i] - rawImg->timestamps[i-1] << endl;
    if (rawImg->timestamps[i] - rawImg->timestamps[i-1] < MAX_MILL_SEC) {
      colOk[i] = true;
      count++;
    }
  }
  cout << "Got: " << count << " frames with no vfr" << endl;
  raw.set_size(numOk, count);
  for (size_t wIx = 0; wIx < mapping.size(); wIx++) {
    int idx = mapping[wIx];
    int cIx = 0;
    for (int fIx = 0; fIx < img.GetFrames(); fIx++) {
      if (colOk[fIx]) {
        raw(wIx,cIx++) = img.At(idx, fIx);
      }
    }
  }
  return numOk;
}

int main(int argc, const char * argv[]) {

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  CompressOpts o;
  opts.GetOption(o.maskFile, "", '-', "mask");
  Mask mask(o.maskFile.c_str());
  opts.GetOption(o.rowStart, "0", '-', "row-start");
  opts.GetOption(o.rowEnd, "-1", '-', "row-end");
  if(o.rowEnd < 0) { o.rowEnd = mask.H(); }

  opts.GetOption(o.colStart, "0", '-', "col-start");
  opts.GetOption(o.colEnd, "-1", '-', "col-end");
  if(o.colEnd < 0) { o.colEnd = mask.W(); }

  opts.GetOption(o.datIn, "", '-', "dat-file");
  opts.GetOption(o.prefixOut, "noise", '-', "out-prefix");
  opts.GetOption(o.numBasis, "6", '-', "num-basis");
  opts.GetOption(o.rowStep, "100", '-', "row-step");
  opts.GetOption(o.colStep, "100", '-', "col-step");
  opts.GetOption(o.minPercent, ".5", '-', "min-percent");
  opts.GetOption(o.runName, "", '-', "run-name");
  // Load file
  Image img;
  img.SetImgLoadImmediate (false);
  img.SetIgnoreChecksumErrors (false);
  ION_ASSERT(img.LoadRaw (o.datIn.c_str()), "Couldn't load: " + o.datIn);

  // Create reporters
  vector<CompressReporter *> reporters;
  WellReporter wRep(o.prefixOut + ".wells.txt");
  reporters.push_back(&wRep);
  FrameReporter frameRep(o.prefixOut + ".frames.txt");
  reporters.push_back(&frameRep);
  DeviationSummaryReporter devRep(o.prefixOut + ".summary.json", o.runName);
  reporters.push_back(&devRep);
  int regRowStart = 0;
  if (o.rowStep * 5 < mask.H()) {
    regRowStart = o.rowStep * 4;
  }
  int regColStart = 0;
  if (o.colStep * 5 < mask.H()) {
    regColStart = o.colStep * 4;
  }
  RegionIndividualReporter regionRep(o.prefixOut + ".region.txt", regRowStart, regRowStart + o.rowStep, regColStart, regColStart + o.colStep);
  reporters.push_back(&regionRep);
  for (size_t rIx = 0; rIx < reporters.size(); rIx++) {
    reporters[rIx]->Init();
  }
  // For each region do compression and run report.
  for (int row = o.rowStart; row < o.rowEnd; row += o.rowStep) {
    for (int col = o.colStart; col < o.colEnd; col += o.colStep) {
      int rowStop = min(row + o.rowStep, o.rowEnd);
      int colStop = min(col + o.colStep, o.colEnd);
      Mat<float> raw;
      vector<int> mapping;
      int numOk = LoadData(row, rowStop, col, colStop, mask, img,  mapping, raw);
      if (numOk > o.minPercent * (rowStop - row) * (colStop - col)) {
        Mat<float> compressed(raw.n_rows, raw.n_cols);
        Compress(raw, o.numBasis, compressed);
        for (size_t rIx = 0; rIx < reporters.size(); rIx++) {
          reporters[rIx]->Report(row, rowStop, col, colStop, 0,
                                 mapping, mask, raw, compressed);
        }
      }
    }
  }
  for (size_t rIx = 0; rIx < reporters.size(); rIx++) {
    reporters[rIx]->Finish();
  }

  // Spin things down.
  
}


