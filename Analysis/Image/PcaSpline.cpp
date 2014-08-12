/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <algorithm>
#include "PcaSpline.h"
#include "Utils.h"
#include <malloc.h>
//#define EIGEN_USE_MKL_ALL 1
#include <Eigen/Dense>
#include <Eigen/LU>

using namespace std;
using namespace Eigen;

/**
 * Create the basis splines for order requested at a particular value of x.
 * From "A Practical Guide To Splines - Revised Edition" by Carl de Boor p 111
 * the algorithm for BSPLVB. The value of the spline coefficients can be calculated
 * based on a cool recursion. This is a simplified and numerically stable algorithm
 * that seems to be the basis of most bspline implementations including gsl, matlab,etc.
 * @param all_knots - Original knots vector augmented with order number of endpoints
 * @param n_knots - Total number of knots.
 * @param order - Order or order of polynomials, 4 for cubic splines.
 * @param i - index of value x in knots such that all_knots[i] <= x && all_knots[i+1] > x
 * @param x - value that we want to evaluate spline at.
 * @param b - memory to write our coefficients into, must be at least order long.
 */
void basis_spline_xi(float *all_knots, int n_knots, int order, int i, float x, float *b) {
  b[0] = 1;
  double kr[order];
  double kl[order];
  double term;
  double saved;
  for (int j = 0; j < order - 1; j++) {
    kr[j] = all_knots[i + j + 1] - x; // k right values
    kl[j] = x - all_knots[i - j];     // k left values
    saved = 0;
    for (int r = 0; r <= j; r++) {
      term = b[r] / (kr[r] + kl[j-r]);
      b[r] = saved + kr[r] * term;
      saved = kl[j-r] * term;
    }
    b[j+1] = saved;
  }
}

/**
 * Create the augmented knots vector and fill in matrix Xt with the spline basis vectors
 */
void basis_splines_endreps_local(float *knots, int n_knots, int order, int *boundaries, int n_boundaries,  MatrixXf &Xt) {
  assert(n_boundaries == 2 && boundaries[0] < boundaries[1]);
  int n_rows = boundaries[1] - boundaries[0]; // this is frames so evaluate at each frame we'll need
  int n_cols = n_knots + order;  // number of basis vectors we'll have at end 
  Xt.resize(n_cols, n_rows); // swapped to transpose later. This order lets us use it as a scratch space
  Xt.fill(0);
  int n_std_knots = n_knots + 2 * order;
  float std_knots[n_std_knots];

  // Repeat the boundary knots on there to ensure linear out side of knots
  for (int i = 0; i < order; i++) {
    std_knots[i] = boundaries[0];
  }
  // Our original boundary knots here
  for (int i = 0; i < n_knots; i++) {
    std_knots[i+order] = knots[i];
  }
  // Repeat the boundary knots on there to ensure linear out side of knots
  for (int i = 0; i < order; i++) {
    std_knots[i+order+n_knots] = boundaries[1];
  }
  // Evaluate our basis splines at each frame.
  for (int i = boundaries[0]; i < boundaries[1]; i++) {
    int idx = -1;
    // find index such that i >= knots[idx] && i < knots[idx+1]
    float *val = std::upper_bound(std_knots, std_knots + n_std_knots - 1, 1.0f * i);
    idx = val - std_knots - 1;
    assert(idx >= 0);
    float *f = Xt.data() + i * n_cols + idx - (order - 1); //column offset
    basis_spline_xi(std_knots, n_std_knots, order, idx, i, f);
  }
  // Put in our conventional format where each column is a basis vector
  Xt.transposeInPlace();
}


void PcaSpline::FillInKnots(const std::string &strategy, int n_frame, std::vector<float> &knots) {
  if (strategy != "no-knots") {
    if (strategy == "even4") {
      float stride = n_frame / 3.0f;
      knots.push_back(stride);
      knots.push_back(stride * 2.0f);
    }
    else if (strategy == "middle") {
      float stride = n_frame / 2.0f;
      knots.push_back(stride);
    }
    else if (strategy.find("explicit:") == 0) {
      string spec = strategy.substr(9, strategy.length() - 9);
      knots = char2Vec<float>(spec.c_str(), ',');
    }
    else {
      assert(false);
    }
  }
}

int PcaSpline::NumBasisVectors(int n_pca, int n_frames, int n_order, const std::string &knot_strategy) {
  std::vector<float> knots;
  FillInKnots(knot_strategy, n_frames, knots);
  int spline_basis = 0;
  if (knots.size() > 0) {
    spline_basis = knots.size() + n_order;
  }
  return spline_basis + n_pca;
}

void PcaSpline::LossyCompress(int n_wells, int n_frame, float *image, 
			      int n_sample_wells, float *image_sample,
			      PcaSplineRegion &compressed) {
  Map<MatrixXf, Aligned> Ysub(image_sample, n_sample_wells, n_frame);
  Map<MatrixXf, Aligned> Y(image, n_wells, n_frame);
  MatrixXf C = Ysub.transpose() * Ysub;
  SelfAdjointEigenSolver<MatrixXf> es;
  es.compute(C);
  MatrixXf Pca_Basis = es.eigenvectors();
  MatrixXf Spline_Basis;
  vector<float> knots;
  FillInKnots(m_knot_strategy, n_frame, knots);
  if (!knots.empty()) {
    int boundaries[2];
    boundaries[0] = 0;
    boundaries[1] = n_frame;
    basis_splines_endreps_local(&knots[0], knots.size(), m_order, boundaries, sizeof(boundaries)/sizeof(boundaries[0]), Spline_Basis);
  }
  assert(compressed.n_basis == m_num_pca + Spline_Basis.cols());
  Map<MatrixXf, Aligned> Basis(compressed.basis_vectors, compressed.n_frames, compressed.n_basis);
  for(int i = 0; i < m_num_pca; i++) {
    Basis.col(i) = Pca_Basis.col(Pca_Basis.cols() - i - 1);
  }
  for (int i = 0; i < Spline_Basis.cols(); i++) {
    Basis.col(i+m_num_pca) = Spline_Basis.col(i);
  }
  MatrixXf SX;
  if (m_tikhonov_reg > 0) {
    MatrixXf diag(Basis.cols(), Basis.cols());
    diag.fill(0);
    for (int i = 0; i < diag.rows(); i++) {
      diag(i,i) = 1.0f * m_tikhonov_reg;
    }
    SX = ((Basis.transpose() * Basis) + (diag.transpose() * diag)).inverse() * Basis.transpose();
  }
  else { 
    SX = (Basis.transpose() * Basis).inverse() * Basis.transpose();
  }
  Map<MatrixXf, Aligned> B(compressed.coefficients, n_wells, Basis.cols());
  B = Y * SX.transpose();
}

void PcaSpline::SampleImageMatrix(int n_wells, int n_frames, float * __restrict orig, int *__restrict trcs_state, int target_sample,
				  int &n_sample, float ** __restrict sample) {
  int good_wells = 0;
  int * __restrict state_begin = trcs_state;
  int * __restrict state_end  = trcs_state + n_wells;
  while (state_begin != state_end) {
    if (*state_begin++ >= 0) {
      good_wells++;
    }
  }
  int sample_rate = max(1,good_wells/target_sample);
  int n_sample_rows = (int)(ceil((float)good_wells/sample_rate));
  n_sample = n_sample_rows;
  *sample = (float *) memalign(32, sizeof(float) * n_frames * n_sample_rows);
  for (int i = 0; i < n_frames; i++) {
    float *__restrict col_start = orig + i * n_wells;
    float *__restrict col_end = col_start + n_wells;
    state_begin = trcs_state;
    float *__restrict sample_col_begin = (*sample) + n_sample_rows * i;
    int check_count = 0;
    int good_count = sample_rate - 1;
    while (col_start != col_end) {
      if (*state_begin >= 0 && ++good_count == sample_rate) {
	*sample_col_begin++ = *col_start;
	good_count = 0;
	check_count++;
      }
      col_start++;
    }
    assert(check_count == n_sample_rows); 
  }
}

void PcaSpline::LossyUncompress(int n_wells, int n_frame, float *image, 
				const PcaSplineRegion &compressed) {
  Map<MatrixXf, Aligned> Yh(image, n_wells, compressed.n_frames);
  Map<MatrixXf, Aligned> Basis(compressed.basis_vectors, compressed.n_frames, compressed.n_basis);
  Map<MatrixXf, Aligned> B(compressed.coefficients, n_wells, compressed.n_basis);
  Yh = B * Basis.transpose();
}
