/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include "Image.h"
#include "Mask.h"
#include "GridMesh.h"
#include "T0Calc.h"
#include "T0Model.h"
#include "TraceChunk.h"
#include "Utils.h"

using namespace std;

class TraceConfig {

public:
  TraceConfig() {
    start_detailed_time = -5;
    stop_detailed_time = 16;
    left_avg = 5;
    rate_sigma_intercept = -0.2577416f;
    rate_sigma_slope = 1.0195878f;
    t0_tmid_intercept = 0.6623247;
    t0_tmid_slope = 0.9382263;
    time_start_slop = 0;
    t0_prior_pre_millis = 333; // about 5 frames at 15fps
    t0_prior_post_millis = 333; // about 5 frames at 15fps
    t0_prior_weight = 0.0f;
    row_step = 50;
    col_step = 50;
  }

  int start_detailed_time; ///< From TimeCompression::SetUpTime()
  int stop_detailed_time; ///< From TimeCompression::SetUpTime()
  int left_avg; ///< From TimeCompression::SetUpTime()
  int row_step;
  int col_step;
  float rate_sigma_intercept; ///< intercept for estimating sigma from 1/slope
  float rate_sigma_slope; ///< coefficient for estimating sigma from 1/slope
  float t0_tmid_intercept; ///< intercept for estimating t_mid_nuc from t0
  float t0_tmid_slope; ///< coefficient for estimating t_mid_nuc from t0
  float time_start_slop; ///< minimum frames back from t0
  int t0_prior_pre_millis;
  int t0_prior_post_millis;
  float t0_prior_weight;
};


void GenerateBfT0Prior(TraceConfig &config,
                       short *img, 
                       int baseFrameRate,
                       size_t numRow,
                       size_t numCol,
                       size_t numFrame,
                       int *timeStamps,
                       size_t colStep,
                       size_t rowStep,
                       Mask *mask,
                       T0Calc &t0,
                       GridMesh<T0Prior> &t0Prior) {
  t0.SetWindowSize(4);
  t0.SetMinFirstHingeSlope(-1/(float) baseFrameRate);
  t0.SetMaxFirstHingeSlope(30/(float) baseFrameRate);
  t0.SetMinSecondHingeSlope(-1000/(float) baseFrameRate);
  t0.SetMaxSecondHingeSlope(-10/(float) baseFrameRate);
  t0.Init(numRow, numCol, numFrame, rowStep, colStep, 1);
  t0.SetTimeStamps(timeStamps, numFrame);
  t0.SetMask(mask);
  t0.SetStepSize(3);
  ClockTimer timer;
  cout << "Calculating sum" << endl;
  timer.StartTimer();
  t0.CalcAllSumTrace(img);
  timer.PrintMilliSeconds(cout,"Sum took:");
  cout << "Calculating t0" << endl;
  timer.StartTimer();
  t0.CalcT0FromSum();
  timer.PrintMilliSeconds(cout,"T0 took");
  t0.FillInT0Prior(t0Prior, config.t0_prior_pre_millis, config.t0_prior_post_millis, config.t0_prior_weight);
  string refFile = "t0_bf_reference.txt";
  ofstream out(refFile.c_str());
  t0.WriteResults(out);
  out.close();
}

void GenerateAcqT0Prior(TraceConfig &config,
                        short *img, 
                        int baseFrameRate,
                        size_t numRow,
                        size_t numCol,
                        size_t numFrame,
                        int *timeStamps,
                        size_t colStep,
                        size_t rowStep,
                        Mask *mask,
                        T0Calc &t0,
                        GridMesh<T0Prior> &t0Prior) {
  /* How wide of a window on either side to use for linear model. */
  t0.SetWindowSize(5);
  /* Set constraints on allowable slope. */
  t0.SetMinFirstHingeSlope(-10/(float) baseFrameRate);
  t0.SetMaxFirstHingeSlope(1/(float) baseFrameRate);
  t0.SetMinSecondHingeSlope(2/(float) baseFrameRate);
  t0.SetMaxSecondHingeSlope(100/(float) baseFrameRate);
  /* Configure the size of the image and the grid size we want to look at. */
  t0.Init(numRow, numCol, numFrame, rowStep, colStep, 1);
  /* What time is each frame. */
  t0.SetTimeStamps(timeStamps, numFrame);
  t0.SetStepSize(3);
  t0.SetMask(mask);
  t0.SetT0Prior(t0Prior);
  ClockTimer timer;
  cout << "Calculating sum" << endl;
  /* Calculate the sum for all regions to get avg trace. */
  timer.StartTimer();
  t0.CalcAllSumTrace(img);
  timer.PrintMilliSeconds(cout,"Sum took:");
  cout << "Calculating t0" << endl;
  /* Look for point with best reduction in ssq error using hinge over line. */
  timer.StartTimer();
  t0.CalcT0FromSum();
  timer.PrintMilliSeconds(cout,"T0 took");
  /* Output some statistics for debugging. */
  string refFile = "t0_dat_reference.txt";
  ofstream out(refFile.c_str());
  t0.WriteResults(out);
  out.close();
}

void GenerateDataChunks(TraceConfig &config, T0Calc &t0, const struct RawImage *raw, 
                        int rowStep, int colStep, GridMesh<TraceChunk> &mTraceChunks) {
  mTraceChunks.Init(raw->rows, raw->cols, rowStep, colStep);
  for (size_t bIx = 0; bIx < t0.GetNumRegions(); bIx++) {
    int rowStart,rowEnd,colStart, colEnd;
    t0.GetRegionCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    TraceChunk &chunk = mTraceChunks.GetItem(bIx);
    chunk.SetDimensions(rowStart, rowEnd-rowStart, colStart, colEnd-colStart, 0, raw->frames);
    float t0Time = t0.GetT0(bIx);
    t0Time -= config.time_start_slop;
    chunk.mTime.SetUpTime(raw->uncompFrames, t0Time/raw->baseFrameRate, config.start_detailed_time, config.stop_detailed_time, config.left_avg);
    chunk.mTime.SetupConvertVfcTimeSegments(raw->frames, raw->timestamps, raw->baseFrameRate);
    //    chunk.mTime.ReportVfcConversion(raw->frames, raw->timestamps, raw->baseFrameRate, std::cout);
    //    cout << "Done with: " << bIx << endl;
  }

}

int main(int argc, const char *argv[]) {
  string bfFile = argv[1];
  Image img;
  assert(argc == 3);
  img.LoadRaw(bfFile.c_str());
  TraceConfig config;
  const RawImage *raw = img.GetImage(); 
  Mask mask(raw->cols, raw->rows);
  GridMesh<T0Prior> t0Prior;
  T0Calc t0;
  /* Calc t0 and get prior. */
  cout << "Doing beadfind t0" << endl;
  GenerateBfT0Prior(config, raw->image, raw->baseFrameRate, raw->rows, raw->cols,
                    raw->frames, raw->timestamps,
                    config.row_step, config.col_step, &mask, t0, t0Prior);
  /* Use prior to calculate t0 and slope. */
  string datFile = argv[2];
  Image datImg;
  datImg.LoadRaw(datFile.c_str());
  const RawImage *datRaw = datImg.GetImage(); 
  cout << "Doing acquisition t0" << endl;
  GenerateAcqT0Prior(config, datRaw->image, datRaw->baseFrameRate, datRaw->rows, datRaw->cols,
                     datRaw->frames, datRaw->timestamps,
                     config.row_step, config.col_step, &mask, t0, t0Prior);
  datImg.Close();
  /* Use t0 and slope to get the time compression. */
  GridMesh<TraceChunk> mTraceChunks;
  GenerateDataChunks(config, t0, raw, config.row_step, config.col_step, mTraceChunks);

  /* For each region do shifting, t0 and compression. */

  /* Serialize onto disk. */

  /* Read back in */
  
  /* Intiailizae background model empahsis vectors without usual process. */

  /* Serialize into backbround models */
}
