/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include "Utils.h"
#include "Image.h"
#include "Mask.h"
#include "PcaSpline.h"
#include "SampleQuantiles.h"
#include "OptArgs.h"
#include "ComparatorNoiseCorrector.h"
//#define EIGEN_USE_MKL_ALL 1
#include <malloc.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
using namespace Eigen;
using namespace std;

namespace pca_spline_driver {

  void CalcDcOffset(RawImage *raw, Mask &mask, int * __restrict good, int &good_wells, short *dc_offset) {
    //Calculate dc offset for each well. Currently hardcoded at 4
    int n_wells = raw->cols * raw->rows;
    short * __restrict s_zeros_start_1 = raw->image;
    short * __restrict s_zeros_start_2 = raw->image + n_wells;
    short * __restrict s_zeros_start_3 = s_zeros_start_2 + n_wells;
    short * __restrict s_zeros_start_4 = s_zeros_start_3 + n_wells;
    short * __restrict p_dc_start = dc_offset;
    short * __restrict p_dc_end = dc_offset + n_wells;
    int mask_skip = MaskPinned | MaskIgnore;
    const unsigned short *__restrict masked = mask.GetMask();
    good_wells = 0;
    while((p_dc_start != p_dc_end)) {
      if (likely((*masked++ & mask_skip) == 0)) {
	good_wells++;
	*good++ = 1;
      }
      else {
	*good++ = 0;
      }
      *p_dc_start++ = (*s_zeros_start_1++ + *s_zeros_start_2++ + *s_zeros_start_3++ + *s_zeros_start_4++) >> 2;
    }
  }

  void CalcColumnAverage(int n_wells, short * __restrict s_start, short * __restrict s_end, 
			 int * __restrict good, short * __restrict dc_offset, short & avg) {
    int64_t sum = 0;
    while ((s_start != s_end)) {
      short x = *s_start - *dc_offset++;
      *s_start++ = x;
      sum += x;
    }
    avg = (short)((float)sum / n_wells + .5);
  }

  void CalcFrameAvg(RawImage *raw, Mask &mask, int * __restrict good, 
		    short *__restrict dc_offset, short *__restrict frame_avg) {
    int64_t frame_sum_int[raw->frames];
    int n_wells = raw->cols * raw->rows;
    memset(frame_sum_int, 0, sizeof(frame_sum_int));
    memset(frame_avg, 0, sizeof(short) * raw->frames);
    for (int i = 0; i < raw->frames; i++) {
      short *__restrict s_start = raw->image + i * n_wells;
      short *__restrict s_end = s_start + n_wells;
      CalcColumnAverage(n_wells, s_start, s_end, good, dc_offset, frame_avg[i]);
    }
  }

  void LoadMatrixColumn(const short *__restrict s_start, const short *__restrict s_end,
			float *__restrict mem, float *__restrict mem_sub,
			int sample_rate, short favg, int n_sub_rows,
			const int * __restrict good) {
    int good_count = sample_rate - 1;
    while (likely(s_start != s_end)) {
      float x = float(*s_start++ - favg);
      *mem++ = x;
      if (likely(*good++) && unlikely(++good_count == sample_rate)) {
	  *mem_sub++ = x;
	  n_sub_rows--;
	  good_count = 0;
      }
    }
    assert(n_sub_rows == 0);
  }

  void LoadMatrices(RawImage *raw, const short *frame_avg, Mask &mask, int sample_rate, int * __restrict good, MatrixXf &Y, MatrixXf &Ysub) {
    float * __restrict mem = Y.data();
    float * __restrict mem_sub = Ysub.data();
    int n_wells = raw->rows * raw->cols;
    for (int i = 0; i < raw->frames; i++) {
      short * __restrict s_start = raw->image + i * n_wells;
      short * __restrict s_end = s_start + n_wells;
      short favg = frame_avg[i];
      LoadMatrixColumn(s_start, s_end, mem, mem_sub, sample_rate, favg, Ysub.rows(), good);
      mem += n_wells;
      mem_sub += Ysub.rows();
    }
  }
}

using namespace pca_spline_driver;

void help_abort() {
  cout << "PcaSplineDriver - Try compressing data using our modules." << endl;
  cout << "options: " << endl;
  cout << "  -d,--dat-file   dat file to try and compress." << endl;
  cout << "  -k,--knots      knot strategy description." << endl;
  cout << "     --num-pca    number of pca basis vectors to use." << endl;
  cout << "     --do-stats   should we calculate summary stats." << endl;
  cout << "     --mask-file  mask file. " << endl;
  exit(1);
}

int main(int argc, const char *argv[]) {
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string dat_file;
  //  bool help = false;
  string strategy = "no-knots";
  int num_pca_vectors = 5;
  int do_stats = 0;
  string mask_file;
  double tik_k;
  opts.GetOption(dat_file, "", 'd', "dat-file");
  opts.GetOption(strategy, "no-knots", 'k', "knots");
  opts.GetOption(num_pca_vectors, "5", '-', "num-pca");
  opts.GetOption(do_stats, "0", '-', "do-stats");
  opts.GetOption(mask_file, "", '-', "mask-file");
  opts.GetOption(tik_k, "0", '-', "tik_k");
  Eigen::initParallel();
  if (mask_file.empty() || dat_file.empty()) {
    help_abort();
  }
  //  Eigen::setNbThreads(1);
  Mask mask(mask_file.c_str());
  Image img;
  img.SetImgLoadImmediate (false);
  bool loaded = img.LoadRaw (dat_file.c_str());
  assert(loaded);
  RawImage *raw = img.raw;
  ComparatorNoiseCorrector cnc;
  cnc.CorrectComparatorNoise(raw, &mask, false, true);

  size_t n_wells = raw->cols * raw->rows;  
  short dc_offset[n_wells];
  short frame_avg[raw->frames];
  memset(frame_avg, 0, raw->frames * sizeof(short));
  int good_wells = 0;

  ClockTimer timer;
  int *good = (int *)memalign(32, sizeof(int) *n_wells);
  memset(good, 0, n_wells * sizeof(int));
  CalcDcOffset(raw, mask, good, good_wells, dc_offset);
  timer.PrintMicroSeconds(cout, "To DC offset took: ");
  CalcFrameAvg(raw, mask, good, dc_offset, frame_avg);
  //  memset(frame_avg, 0, raw->frames * sizeof(short));
  timer.PrintMicroSeconds(cout, "To calc frame avg took: ");
  Matrix<float, Dynamic, Dynamic, AutoAlign | ColMajor> Y(n_wells, raw->frames);
  Matrix<float, Dynamic, Dynamic, AutoAlign | ColMajor> Yh(n_wells, raw->frames);
  int target_sample = 500;
  int sample_rate = max(1,good_wells/target_sample);
  int n_sample_rows = (int)(ceil((float)good_wells/sample_rate));
  Matrix<float, Dynamic, Dynamic, AutoAlign | ColMajor> Ysub(n_sample_rows, raw->frames);
  Y.fill(0);
  Ysub.fill(0);
  LoadMatrices(raw, frame_avg, mask, sample_rate, good, Y, Ysub);
  timer.PrintMicroSeconds(cout, "To Load Matrices took: ");

  PcaSpline compressor(num_pca_vectors, strategy);
  compressor.SetTikhonov(tik_k);
  PcaSplineRegion compressed;
  compressed.n_wells = raw->rows * raw->cols;
  compressed.n_frames = raw->frames;
  compressed.n_basis = compressor.NumBasisVectors(num_pca_vectors, compressed.n_frames, 
						  compressor.GetOrder(), strategy);
  cout << "Num basis: " << compressed.n_basis << endl;
  compressed.basis_vectors = (float *)memalign(32, sizeof(float) * compressed.n_frames * compressed.n_basis);
  //  memset(compressed.basis_vectors, 0, sizeof(float) * compressed.n_frames * compressed.n_basis);
  compressed.coefficients = (float *)memalign(32, sizeof(float) * compressed.n_wells * compressed.n_basis);
  //  memset(compressed.coefficients, 0, sizeof(float) * compressed.n_wells * compressed.n_basis);
  float *ysub2_mem;
  int num_sample2;
  compressor.SampleImageMatrix(n_wells, raw->frames, Y.data(), good, target_sample, num_sample2, &ysub2_mem);
  timer.PrintMicroSeconds(cout, "To compressor took: ");
  ClockTimer compress_timer;
   compressor.LossyCompress(raw->rows * raw->cols, raw->frames, Y.data(), num_sample2, ysub2_mem, compressed);
  compress_timer.PrintMicroSeconds(cout, "Compressing traces took: ");
   //  compressor.LossyCompress(raw->rows * raw->cols, raw->frames, Y.data(), Ysub.rows(), Ysub.data(), compressed);
  timer.PrintMicroSeconds(cout, "Full load and compression: ");
  compressor.LossyUncompress(raw->rows * raw->cols, raw->frames, Yh.data(), compressed);
  if(do_stats > 0) {
    ofstream o("error.txt");
    vector<SampleQuantiles<float> > frame_quantiles(raw->frames);
    for (size_t i = 0; i < frame_quantiles.size(); i++) {
      frame_quantiles[i].Init(10000);
    }
    SampleQuantiles<float> total_mad(10000);
    for (int i = 0; i < Y.cols(); i++) {
      SampleQuantiles<float> &frame = frame_quantiles[i];
      float *start = Yh.data() + n_wells * i;
      float *end = start + n_wells;
      short *i_start = raw->image + n_wells * i;
      short favg = frame_avg[i];
      while (start != end) {
        float value = *start + favg - *i_start;
        frame.AddValue(fabs(value));
        total_mad.AddValue(fabs(value));
        start++;
        if (start != end) {
          o << value << "\t";
        }
        else {
          o << value;
        }
        i_start++;
      }
      o << endl;
    }
    cout << "Total error: " << total_mad.GetMedian() << " +/- " << total_mad.GetIqrSd() << endl;
    SampleStats<double> frame_mean_err;
    for (size_t i = 0; i < frame_quantiles.size(); i++) {
      frame_mean_err.AddValue(frame_quantiles[i].GetMedian());
    }
    cout << "Mean of median frame errors: " << frame_mean_err.GetMean()  << " +/- " << frame_mean_err.GetSD() << endl;
    //   cout << "Frame mean error: " << total_mad.GetMedian() << " +/- " << total_mad.GetIqrSd() << endl;
    if (do_stats > 1) {
      cout << "Frames: " <<endl;
      for (size_t i = 0; i < frame_quantiles.size(); i++) {
        cout << i << "\t" << frame_quantiles[i].GetMedian() << endl;
      }
    }
}
return 0;
}
