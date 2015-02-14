/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <malloc.h>
#include "Utils.h"
#include "TraceStoreCol.h"
#include "H5Eigen.h"
//#define EIGEN_USE_MKL_ALL 1
#include <Eigen/Dense>
#include <Eigen/LU>

#define MIN_SAMPLE_WELL 100
#define SMOOTH_REDUCE_STEP 10
#define SMOOTH_REDUCE_REGION 100
#define INTEGRATION_START 6
#define INTEGRATION_END 12
void TraceStoreCol::WellProj(TraceStoreCol &store,
                             std::vector<KeySeq> & key_vectors,
                             vector<char> &filter,
                             vector<float> &mad) {
  int useable_flows = 0;
  for (size_t i = 0; i < key_vectors.size(); i++) {
    useable_flows = std::max((int)key_vectors[i].usableKeyFlows, useable_flows);
    //    useable_flows = std::max(store.GetNumFlows(), (size_t)useable_flows);
  }
  Eigen::VectorXf norm(store.mFrameStride * useable_flows);
  Eigen::VectorXf sum(store.mFrameStride * useable_flows);

  norm.setZero();
  sum.setZero();
  int start_frame = store.GetNumFrames() - 2;
  int end_frame = store.GetNumFrames();
  int count = 0;
  for (int frame_ix = start_frame; frame_ix < end_frame; frame_ix++) {
    count++;
    int16_t *__restrict data_start = store.GetMemPtr() + frame_ix * store.mFlowFrameStride;
    int16_t *__restrict data_end = data_start + store.mFrameStride * useable_flows;
    float *__restrict norm_start = &norm[0];
    while(data_start != data_end) {
      *norm_start++ += *data_start++;
    }
  }

  // std::vector<ChipReduction> smoothed_avg(useable_flows);
  // int x_clip = mCols;
  // int y_clip = mRows;
  // if (mUseMeshNeighbors == 0) {
  //   x_clip = THUMBNAIL_SIZE;
  //   y_clip = THUMBNAIL_SIZE;
  // }
  // int last_frame = store.GetNumFrames() - 1;
  // for (int flow_ix = 0; flow_ix < useable_flows; flow_ix++) {
  //   smoothed_avg[flow_ix].Init(mRows, mCols, 1,
  //                              SMOOTH_REDUCE_STEP, SMOOTH_REDUCE_STEP,
  //                              y_clip, x_clip,
  //                              SMOOTH_REDUCE_STEP * SMOOTH_REDUCE_STEP * .4);
  //   smoothed_avg[flow_ix].ReduceFrame(&mData[0] + flow_ix * mFrameStride + last_frame  * mFlowFrameStride, &filter[0], 0);
  //   smoothed_avg[flow_ix].SmoothBlocks(SMOOTH_REDUCE_REGION, SMOOTH_REDUCE_REGION);
  // }
  // int x_count = 0;
  // for (int flow_ix = 0; flow_ix < useable_flows; flow_ix++) {
  //   for (size_t row_ix = 0; row_ix < mRows; row_ix++) {
  //     for (size_t col_ix = 0; col_ix < mCols; col_ix++) {
  //       float avg = smoothed_avg[flow_ix].GetSmoothEst(row_ix, col_ix, 0);
  //       norm[x_count] = avg;
  //       x_count++;
  //     }
  //   }
  // }

  norm = norm / count;
  //  start_frame = INTEGRATION_START;
  //  end_frame = INTEGRATION_END;
  start_frame = 5;
  end_frame = store.GetNumFrames() - 6;
  
  for (int frame_ix = start_frame; frame_ix < end_frame; frame_ix++) {
    int16_t *__restrict data_start = store.GetMemPtr() + frame_ix * store.mFlowFrameStride;
    int16_t *__restrict data_end = data_start + store.mFrameStride * useable_flows;
    float *__restrict norm_start = &norm[0];
    float *__restrict sum_start = &sum[0];
    int well_offset = 0;
    while(data_start != data_end) {
      if (*norm_start == 0.0f) {
        *norm_start = 1.0f;
        // avoid divide by zero but mark that something is wrong with this well..
        if (filter[well_offset] == 0) {
          filter[well_offset] = 6;
        }
      }
      *sum_start += *data_start / *norm_start;
      well_offset++;
      // reset each flow
      if (well_offset == (int) store.mFrameStride) {
        well_offset = 0;
      }
      sum_start++;
      data_start++;
      norm_start++;
    }
  }
  
  Eigen::MatrixXf flow_mat(store.mFrameStride, useable_flows);
  for (int flow_ix = 0; flow_ix < flow_mat.cols(); flow_ix++) {
    for (size_t well_ix = 0; well_ix < store.mFrameStride; well_ix++) {
      flow_mat(well_ix, flow_ix) = sum(flow_ix * store.mFrameStride + well_ix);
    }
  }

  Eigen::VectorXf n2;
  n2 = flow_mat.rowwise().squaredNorm();
  n2 = n2.array().sqrt();
  float *n2_start = n2.data();
  float *n2_end = n2_start + n2.rows();
  while (n2_start != n2_end) {
    if (*n2_start == 0.0f) { *n2_start = 1.0f; }
    n2_start++;
  }
  for (int flow_ix = 0; flow_ix < flow_mat.cols(); flow_ix++) {
    flow_mat.col(flow_ix).array() = flow_mat.col(flow_ix).array() / n2.array();
  }


  Eigen::VectorXf proj;
  Eigen::VectorXf max_abs_proj(flow_mat.rows());
  max_abs_proj.setZero();
  for (size_t key_ix = 0; key_ix < key_vectors.size(); key_ix++) {
    Eigen::VectorXf key(useable_flows);
    key.setZero();
    for (int f_ix = 0; f_ix < useable_flows; f_ix++) {
      key[f_ix] = key_vectors[key_ix].flows[f_ix];
    }
    proj = flow_mat * key;
    proj = proj.array().abs();
    for (int i = 0; i < proj.rows(); i++) {
      max_abs_proj(i) = max(max_abs_proj(i), proj(i));
    }
  }
  
  for (int i = 0; i < max_abs_proj.rows(); i++) {
    mad[i] = max_abs_proj(i);
  }
  
}

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
void basis_spline_xi_v2(float *all_knots, int n_knots, int order, int i, float x, float *b) {
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
void basis_splines_endreps_local_v2(float *knots, int n_knots, int order, int *boundaries, int n_boundaries,  Eigen::MatrixXf &Xt) {
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
    basis_spline_xi_v2(std_knots, n_std_knots, order, idx, i, f);
  }
  // Put in our conventional format where each column is a basis vector
  Xt.transposeInPlace();
}

void FillInKnots(const std::string &strategy, int n_frame, std::vector<float> &knots) {
  if (strategy != "no-knots") {
    if (strategy == "even4") {
      float stride = n_frame / 3.0f;
      knots.push_back(stride);
      knots.push_back(stride * 2.0f);
    }
    else if (strategy == "every4") {
      int current = 4;
      while (current < (n_frame -1)) {
        knots.push_back(current);
        current+=4;
      }
    }
    else if (strategy == "every3") {
      int current = 3;
      while (current < (n_frame -1)) {
        knots.push_back(current);
        current+=3;
      }
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

void TraceStoreCol::SplineLossyCompress(const std::string &strategy, int order, char *bad_wells, float *mad) {
  Eigen::MatrixXf Basis;
  vector<float> knots;
  FillInKnots(strategy, mFrames, knots);
  //  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> Basis(, compressed.n_frames, compressed.n_basis);
  if (!knots.empty()) {
    int boundaries[2];
    boundaries[0] = 0;
    boundaries[1] = mFrames;
    basis_splines_endreps_local_v2(&knots[0], knots.size(), order, boundaries, sizeof(boundaries)/sizeof(boundaries[0]), Basis);
  }
  Eigen::MatrixXf SX = (Basis.transpose() * Basis).inverse() * Basis.transpose();
  Eigen::MatrixXf Y(mFlowFrameStride, mFrames);
  //  Eigen::MatrixXf FlowMeans(mFlows, mFrames);

  //  FlowMeans.setZero();
  int good_wells = 0;
  char *bad_start = bad_wells;
  char *bad_end = bad_start + mFrameStride;
  while(bad_start != bad_end) {
    if (*bad_start++ == 0) {
      good_wells++;
    }
  }
  
  // if nothing good then skip it
  if (good_wells < MIN_SAMPLE_WELL) {
    return;
  }

  std::vector<ChipReduction> smoothed_avg(mFlows);
  int x_clip = mCols;
  int y_clip = mRows;
  if (mUseMeshNeighbors == 0) {
    x_clip = THUMBNAIL_SIZE;
    y_clip = THUMBNAIL_SIZE;
  }
  for (size_t flow_ix = 0; flow_ix < mFlows; flow_ix++) {
    smoothed_avg[flow_ix].Init(mRows, mCols, mFrames,
                               SMOOTH_REDUCE_STEP, SMOOTH_REDUCE_STEP,
                               y_clip, x_clip,
                               SMOOTH_REDUCE_STEP * SMOOTH_REDUCE_STEP * .4);
    for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
      smoothed_avg[flow_ix].ReduceFrame(&mData[0] + flow_ix * mFrameStride + frame_ix * mFlowFrameStride, bad_wells, frame_ix);
    }
    smoothed_avg[flow_ix].SmoothBlocks(SMOOTH_REDUCE_REGION, SMOOTH_REDUCE_REGION);
  }

  float *y_start = Y.data();
  float *y_end = Y.data() + mData.size();
  int16_t *trace_start = &mData[0];
  while(y_start != y_end) {
    *y_start++ = *trace_start++;
  }

  // // get the flow means per frame.
  // for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
  //   for(size_t flow_ix = 0; flow_ix < mFlows; flow_ix++) {
  //     float *start = Y.data() + mFlowFrameStride * frame_ix + flow_ix * mFrameStride;
  //     float *end = start + mFrameStride;
  //     float *sum = &FlowMeans(flow_ix, frame_ix);
  //     char *bad = bad_wells;
  //     while (start != end) {
  //       if (*bad == 0) {
  //         *sum += *start;
  //       }
  //       start++;
  //       bad++;
  //     }
  //     *sum /= good_wells;
  //   }
  // }

  // subtract off flow,frame avg
  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    for(size_t flow_ix = 0; flow_ix < mFlows; flow_ix++) {
      float *start = Y.data() + mFlowFrameStride * frame_ix + flow_ix * mFrameStride;
      float *end = start + mFrameStride;
      for (size_t row = 0; row < mRows; row++) {
        for (size_t col = 0; col < mCols; col++) {
          float avg = smoothed_avg[flow_ix].GetSmoothEst(row, col, frame_ix);
          *start++ -= avg;
        }
      }
    }
  }


  // // subtract them off
  // Eigen::VectorXf col_mean = Y.colwise().sum();
  // col_mean /= Y.rows();

  // for (int i = 0; i < Y.cols(); i++) {
  //   Y.col(i).array() -= col_mean.coeff(i);
  // }

  // Get coefficients to solve
  Eigen::MatrixXf B = Y * SX.transpose();
  // Uncompress data into yhat matrix
  Eigen::MatrixXf Yhat = B * Basis.transpose();


  // add the flow/frame averages back
  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    for(size_t flow_ix = 0; flow_ix < mFlows; flow_ix++) {
      float *start = Y.data() + mFlowFrameStride * frame_ix + flow_ix * mFrameStride;
      float *end = start + mFrameStride;
      float *hstart = Yhat.data() + mFlowFrameStride * frame_ix + flow_ix * mFrameStride;
      for (size_t row = 0; row < mRows; row++) {
        for (size_t col = 0; col < mCols; col++) {
          float avg = smoothed_avg[flow_ix].GetSmoothEst(row, col, frame_ix);
          *start++ += avg;
          *hstart++ += avg;
        }
      }
    }
  }

  // for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
  //   for(size_t flow_ix = 0; flow_ix < mFlows; flow_ix++) {
  //     float *start = Y.data() + mFlowFrameStride * frame_ix + flow_ix * mFrameStride;
  //     float *hstart = Yhat.data() + mFlowFrameStride * frame_ix + flow_ix * mFrameStride;
  //     float *end = start + mFrameStride;
  //     float avg = FlowMeans(flow_ix, frame_ix);
  //     while (start != end) {
  //       *start++ += avg;
  //       *hstart++ += avg;
  //     }
  //   }
  // }

  // for (int i = 0; i < Yhat.cols(); i++) {
  //   Yhat.col(i).array() += col_mean.coeff(i);
  //   Y.col(i).array() += col_mean.coeff(i);
  // }

  float *yhat_start = Yhat.data();
  float *yhat_end = Yhat.data() + mData.size();
  trace_start = &mData[0];
  while(yhat_start != yhat_end) {
    *trace_start++ = (int)(*yhat_start + .5);
    yhat_start++;
  }

  Y = Y - Yhat;
  Eigen::VectorXf M = Y.rowwise().squaredNorm();

  for (size_t flow_ix = 0; flow_ix < mFlows; flow_ix++) {
    float *mad_start = mad;
    float *mad_end = mad_start + mFrameStride;
    float *m_start = M.data() + flow_ix * mFrameStride;
    while (mad_start != mad_end) {
      *mad_start += *m_start;
      mad_start++;
      m_start++;
    }
  }

  float *mad_start = mad;
  float *mad_end = mad + mFrameStride;
  int norm_factor = mFlows * mFrames;
  while (mad_start != mad_end) {
    *mad_start /= norm_factor;
    mad_start++;
  }
}


void TraceStoreCol::SplineLossyCompress(const std::string &strategy, int order, int flow_ix, char *bad_wells, float *mad) {
  Eigen::MatrixXf Basis;
  vector<float> knots;
  FillInKnots(strategy, mFrames, knots);
  //  Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> Basis(, compressed.n_frames, compressed.n_basis);
  if (!knots.empty()) {
    int boundaries[2];
    boundaries[0] = 0;
    boundaries[1] = mFrames;
    basis_splines_endreps_local_v2(&knots[0], knots.size(), order, boundaries, sizeof(boundaries)/sizeof(boundaries[0]), Basis);
  }
  Eigen::MatrixXf SX = (Basis.transpose() * Basis).inverse() * Basis.transpose();
  Eigen::MatrixXf Y(mFrameStride, mFrames);
  //  Eigen::MatrixXf FlowMeans(mFlows, mFrames);

  //  FlowMeans.setZero();
  int good_wells = 0;
  char *bad_start = bad_wells;
  char *bad_end = bad_start + mFrameStride;
  while(bad_start != bad_end) {
    if (*bad_start++ == 0) {
      good_wells++;
    }
  }
  
  // if nothing good then skip it
  if (good_wells < MIN_SAMPLE_WELL) {
    return;
  }

  ChipReduction smoothed_avg;
  int x_clip = mCols;
  int y_clip = mRows;
  if (mUseMeshNeighbors == 0) {
    x_clip = THUMBNAIL_SIZE;
    y_clip = THUMBNAIL_SIZE;
  }

  smoothed_avg.Init(mRows, mCols, mFrames,
                    SMOOTH_REDUCE_STEP, SMOOTH_REDUCE_STEP,
                    y_clip, x_clip,
                    SMOOTH_REDUCE_STEP * SMOOTH_REDUCE_STEP * .4);
  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    smoothed_avg.ReduceFrame(&mData[0] + flow_ix * mFrameStride + frame_ix * mFlowFrameStride, bad_wells, frame_ix);
  }
  smoothed_avg.SmoothBlocks(SMOOTH_REDUCE_REGION, SMOOTH_REDUCE_REGION);


  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    float *y_start = Y.data() + frame_ix * mFrameStride;
    float *y_end = y_start + mFrameStride;
    int16_t *trace_start = &mData[0] + flow_ix * mFrameStride + frame_ix * mFlowFrameStride;
    while(y_start != y_end) {
      *y_start++ = *trace_start++;
    }
  }
    
  // subtract off flow,frame avg
  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    float *start = Y.data() + mFrameStride * frame_ix;
    float *end = start + mFrameStride;
    for (size_t row = 0; row < mRows; row++) {
      for (size_t col = 0; col < mCols; col++) {
        float avg = smoothed_avg.GetSmoothEst(row, col, frame_ix);
        *start++ -= avg;
      }
    }
  }

  // Get coefficients to solve
  Eigen::MatrixXf B = Y * SX.transpose();
  // Uncompress data into yhat matrix
  Eigen::MatrixXf Yhat = B * Basis.transpose();


  // add the flow/frame averages back
  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    float *start = Y.data() + mFrameStride * frame_ix;
    float *end = start + mFrameStride;
    float *hstart = Yhat.data() + mFrameStride * frame_ix;
    for (size_t row = 0; row < mRows; row++) {
      for (size_t col = 0; col < mCols; col++) {
        float avg = smoothed_avg.GetSmoothEst(row, col, frame_ix);
        *start++ += avg;
        *hstart++ += avg;
      }
    }
  }

  for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
    float *yhat_start = Yhat.data() + frame_ix * mFrameStride;
    float *yhat_end = yhat_start + mFrameStride;
    int16_t *trace_start = &mData[0] + flow_ix * mFrameStride + frame_ix * mFlowFrameStride;
    while(yhat_start != yhat_end) {
      *trace_start++ = (int)(*yhat_start + .5f);
      yhat_start++;
    }
  }

  Y = Y - Yhat;
  Eigen::VectorXf M = Y.rowwise().squaredNorm();

  float *mad_start = mad;
  float *mad_end = mad_start + mFrameStride;
  float *m_start = M.data();
  while (mad_start != mad_end) {
    *mad_start += *m_start;
    mad_start++;
    m_start++;
  }

  mad_start = mad;
  mad_end = mad + mFrameStride;
  int norm_factor = mFrames;// mFlows * mFrames;
  while (mad_start != mad_end) {
    *mad_start /= norm_factor;
    mad_start++;
  }
}



// Compress a block of a data using pca
void TraceStoreCol::PcaLossyCompress(int row_start, int row_end,
                                     int col_start, int col_end,
                                     int flow_ix,
                                     float *ssq, char *filters,
                                     int row_step, int col_step,
                                     int num_pca) {

  Eigen::MatrixXf Ysub, Y, S, Basis;

  int loc_num_wells = (col_end - col_start) * (row_end - row_start);
  int loc_num_cols = col_end - col_start;

  // Take a sample of the data at rate step, avoiding flagged wells
  // Count the good rows
  int sample_wells = 0;
  for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step) {
    char *filt_start = filters + row_ix * mCols + col_start;
    char *filt_end = filt_start + loc_num_cols;
    while (filt_start < filt_end) {
      if (*filt_start == 0) {
        sample_wells++;
      }
      filt_start += col_step;
    }
  }
  // try backing off to every well rather than just sampled if we didn't get enough
  if (sample_wells < MIN_SAMPLE_WELL) {
    row_step = 1;
    col_step = 1;
    int sample_wells = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step) {
      char *filt_start = filters + row_ix * mCols + col_start;
      char *filt_end = filt_start + loc_num_cols;
      while (filt_start < filt_end) {
        if (*filt_start == 0) {
          sample_wells++;
        }
        filt_start += col_step;
      }
    }
  }

  if (sample_wells < MIN_SAMPLE_WELL) {
    return; // just give up
  }

  // Copy the sampled data in Matrix, frame major
  Ysub.resize(sample_wells, mFrames);
  for (int frame_ix = 0; frame_ix < (int)mFrames; frame_ix++) {
    int sample_offset = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix+=row_step) {
      size_t store_offset = row_ix * mCols + col_start;
      char *filt_start = filters + store_offset;
      char *filt_end = filt_start + loc_num_cols;
      int16_t *trace_start = &mData[0] + (mFlowFrameStride * frame_ix) + (flow_ix * mFrameStride) + store_offset;
      float *ysub_start = Ysub.data() + sample_wells * frame_ix + sample_offset;
      while (filt_start < filt_end) {
        if (*filt_start == 0) {
          *ysub_start = *trace_start;
          ysub_start++;
          sample_offset++;
        }
        trace_start += col_step;
        filt_start += col_step;
      }
    }
  }

  // Copy in all the data into working matrix
  Y.resize(loc_num_wells, (int)mFrames);
  for (int frame_ix = 0; frame_ix < (int)mFrames; frame_ix++) {
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      size_t store_offset = row_ix * mCols + col_start;
      int16_t *trace_start = &mData[0] + (mFlowFrameStride * frame_ix) + (flow_ix * mFrameStride) + store_offset;
      int16_t *trace_end = trace_start + loc_num_cols;
      float * y_start = Y.data() + loc_num_wells * frame_ix + (row_ix - row_start) * loc_num_cols;
      while( trace_start != trace_end ) {
        *y_start++ = *trace_start++;
      }
    }
  }
  Eigen::VectorXf col_mean = Y.colwise().sum();
  col_mean /= Y.rows();

  for (int i = 0; i < Y.cols(); i++) {
    Y.col(i).array() -= col_mean.coeff(i);
  }
  // Create scatter matrix
  S = Ysub.transpose() * Ysub;
  // Compute the eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es;
  es.compute(S);
  Eigen::MatrixXf Pca_Basis = es.eigenvectors();
  Eigen::VectorXf Pca_Values = es.eigenvalues();
  // Copy top eigen vectors into basis for projection
  Basis.resize(mFrames, num_pca);
  for (int i = 0; i < Basis.cols(); i++) {
    //    Basis.col(i) = es.eigenvectors().col(es.eigenvectors().cols() - i -1);
    Basis.col(i) = Pca_Basis.col(Pca_Basis.cols() - i - 1);
  }
  // Create solver matrix, often not a good way of solving things but eigen vectors should be stable and fast
  Eigen::MatrixXf SX = (Basis.transpose() * Basis).inverse() * Basis.transpose();
  // Get coefficients to solve
  Eigen::MatrixXf B = Y * SX.transpose();
  // Uncompress data into yhat matrix
  Eigen::MatrixXf Yhat = B * Basis.transpose();

  for (int i = 0; i < Yhat.cols(); i++) {
    Yhat.col(i).array() += col_mean.coeff(i);
    Y.col(i).array() += col_mean.coeff(i);
  }
  
  // H5File h5("pca_lossy.h5");
  // h5.Open();
  // char buff[256];
  // snprintf(buff, sizeof(buff), "/Y_%d_%d_%d", flow_ix, row_start, col_start);
  // H5Eigen::WriteMatrix(h5, buff, Y);
  // snprintf(buff, sizeof(buff), "/Yhat_%d_%d_%d", flow_ix, row_start, col_start);
  // H5Eigen::WriteMatrix(h5, buff, Yhat);
  // snprintf(buff, sizeof(buff), "/Basis_%d_%d_%d", flow_ix, row_start, col_start);
  // H5Eigen::WriteMatrix(h5, buff, Basis);
  // h5.Close();
  // Copy data out of yhat matrix into original data structure, keeping track of residuals
  for (int frame_ix = 0; frame_ix < (int)mFrames; frame_ix++) {
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      size_t store_offset = row_ix * mCols + col_start;
      int16_t *trace_start = &mData[0] + mFlowFrameStride * frame_ix + flow_ix * mFrameStride + store_offset;
      int16_t *trace_end = trace_start + loc_num_cols;
      float * ssq_start = ssq + store_offset;
      size_t loc_offset = (row_ix - row_start) * loc_num_cols;
      float * y_start = Y.data() + loc_num_wells * frame_ix + loc_offset;
      float * yhat_start = Yhat.data() + loc_num_wells * frame_ix + loc_offset;
      while( trace_start != trace_end ) {
        *trace_start = (int16_t)(*yhat_start + .5);
        float val = *y_start - *yhat_start;
        *ssq_start += val * val;
        y_start++;
        yhat_start++;
        trace_start++;
        ssq_start++;
      }
    }
  }

  // divide ssq data out for per frame avg
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    size_t store_offset = row_ix * mCols + col_start;
    float * ssq_start = ssq + store_offset;
    float * ssq_end = ssq_start + loc_num_cols;
    while (ssq_start != ssq_end) {
      *ssq_start /= mFrames;
      ssq_start++;
    }
  }
}

bool TraceStoreCol::PcaLossyCompressChunk(int row_start, int row_end,
                                          int col_start, int col_end,
                                          int num_rows, int num_cols, int num_frames,
                                          int frame_stride,
                                          int flow_ix, int flow_frame_stride,
                                          short *data, bool replace,
                                          float *ssq, char *filters,
                                          int row_step, int col_step,
                                          int num_pca) {

  Eigen::MatrixXf Ysub, Y, S, Basis;

  int loc_num_wells = (col_end - col_start) * (row_end - row_start);
  int loc_num_cols = col_end - col_start;

  // Take a sample of the data at rate step, avoiding flagged wells
  // Count the good rows
  int sample_wells = 0;
  for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step) {
    char *filt_start = filters + row_ix * num_cols + col_start;
    char *filt_end = filt_start + loc_num_cols;
    while (filt_start < filt_end) {
      if (*filt_start == 0) {
        sample_wells++;
      }
      filt_start += col_step;
    }
  }
  // try backing off to every well rather than just sampled if we didn't get enough
  if (sample_wells < MIN_SAMPLE_WELL) {
    row_step = 1;
    col_step = 1;
    int sample_wells = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix+= row_step) {
      char *filt_start = filters + row_ix * num_cols + col_start;
      char *filt_end = filt_start + loc_num_cols;
      while (filt_start < filt_end) {
        if (*filt_start == 0) {
          sample_wells++;
        }
        filt_start += col_step;
      }
    }
  }

  if (sample_wells < MIN_SAMPLE_WELL) {
    return false; // just give up
  }

  // Got enough data to work with, zero out the ssq array for accumulation
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    float *ssq_start = ssq + row_ix * num_cols + col_start;
    float *ssq_end = ssq_start + loc_num_cols;
    while (ssq_start != ssq_end) {
      *ssq_start++ = 0;
    }
  }
  // Copy the sampled data in Matrix, frame major
  Ysub.resize(sample_wells, num_frames);
  for (int frame_ix = 0; frame_ix < (int)num_frames; frame_ix++) {
    int sample_offset = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix+=row_step) {
      size_t store_offset = row_ix * num_cols + col_start;
      char *filt_start = filters + store_offset;
      char *filt_end = filt_start + loc_num_cols;
      int16_t *trace_start = data + (flow_frame_stride * frame_ix) + (flow_ix * frame_stride) + store_offset;
      float *ysub_start = Ysub.data() + sample_wells * frame_ix + sample_offset;
      while (filt_start < filt_end) {
        if (*filt_start == 0) {
          *ysub_start = *trace_start;
          ysub_start++;
          sample_offset++;
        }
        trace_start += col_step;
        filt_start += col_step;
      }
    }
  }

  // Copy in all the data into working matrix
  Y.resize(loc_num_wells, (int)num_frames);
  for (int frame_ix = 0; frame_ix < (int)num_frames; frame_ix++) {
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      size_t store_offset = row_ix * num_cols + col_start;
      int16_t *trace_start = data + (flow_frame_stride * frame_ix) + (flow_ix * frame_stride) + store_offset;
      int16_t *trace_end = trace_start + loc_num_cols;
      float * y_start = Y.data() + loc_num_wells * frame_ix + (row_ix - row_start) * loc_num_cols;
      while( trace_start != trace_end ) {
        *y_start++ = *trace_start++;
      }
    }
  }
  Eigen::VectorXf col_mean = Y.colwise().sum();
  col_mean /= Y.rows();

  for (int i = 0; i < Y.cols(); i++) {
    Y.col(i).array() -= col_mean.coeff(i);
  }
  // Create scatter matrix
  S = Ysub.transpose() * Ysub;
  // Compute the eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es;
  es.compute(S);
  Eigen::MatrixXf Pca_Basis = es.eigenvectors();
  Eigen::VectorXf Pca_Values = es.eigenvalues();
  // Copy top eigen vectors into basis for projection
  Basis.resize(num_frames, num_pca);
  for (int i = 0; i < Basis.cols(); i++) {
    //    Basis.col(i) = es.eigenvectors().col(es.eigenvectors().cols() - i -1);
    Basis.col(i) = Pca_Basis.col(Pca_Basis.cols() - i - 1);
  }
  // Create solver matrix, often not a good way of solving things but eigen vectors should be stable and fast
  Eigen::MatrixXf SX = (Basis.transpose() * Basis).inverse() * Basis.transpose();
  // Get coefficients to solve
  Eigen::MatrixXf B = Y * SX.transpose();
  // Uncompress data into yhat matrix
  Eigen::MatrixXf Yhat = B * Basis.transpose();

  for (int i = 0; i < Yhat.cols(); i++) {
    Yhat.col(i).array() += col_mean.coeff(i);
    Y.col(i).array() += col_mean.coeff(i);
  }
  
  // H5File h5("pca_lossy.h5");
  // h5.Open();
  // char buff[256];
  // snprintf(buff, sizeof(buff), "/Y_%d_%d_%d", flow_ix, row_start, col_start);
  // H5Eigen::WriteMatrix(h5, buff, Y);
  // snprintf(buff, sizeof(buff), "/Yhat_%d_%d_%d", flow_ix, row_start, col_start);
  // H5Eigen::WriteMatrix(h5, buff, Yhat);
  // snprintf(buff, sizeof(buff), "/Basis_%d_%d_%d", flow_ix, row_start, col_start);
  // H5Eigen::WriteMatrix(h5, buff, Basis);
  // h5.Close();
  // Copy data out of yhat matrix into original data structure, keeping track of residuals
  for (int frame_ix = 0; frame_ix < (int)num_frames; frame_ix++) {
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      size_t store_offset = row_ix * num_cols + col_start;
      int16_t *trace_start = data + flow_frame_stride * frame_ix + flow_ix * frame_stride + store_offset;
      int16_t *trace_end = trace_start + loc_num_cols;
      float * ssq_start = ssq + store_offset;
      size_t loc_offset = (row_ix - row_start) * loc_num_cols;
      float * y_start = Y.data() + loc_num_wells * frame_ix + loc_offset;
      float * yhat_start = Yhat.data() + loc_num_wells * frame_ix + loc_offset;
      while( trace_start != trace_end ) {
        if (replace) {
          *trace_start = (int16_t)(*yhat_start + .5);
        }
        float val = *y_start - *yhat_start;
        *ssq_start += val * val;
        y_start++;
        yhat_start++;
        trace_start++;
        ssq_start++;
      }
    }
  }

  // divide ssq data out for per frame avg
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    size_t store_offset = row_ix * num_cols + col_start;
    float * ssq_start = ssq + store_offset;
    float * ssq_end = ssq_start + loc_num_cols;
    while (ssq_start != ssq_end) {
      *ssq_start /= num_frames;
      ssq_start++;
    }
  }
  return true;
}

void TraceStoreCol::SplineLossyCompress(const std::string &strategy, int order, int flow_ix, char *bad_wells, 
                                        float *mad, size_t num_rows, size_t num_cols, size_t num_frames, size_t num_flows,
                                        int use_mesh_neighbors, size_t frame_stride, size_t flow_frame_stride, int16_t *data) {
  Eigen::MatrixXf Basis;
  vector<float> knots;
  FillInKnots(strategy, num_frames, knots);
  if (!knots.empty()) {
    int boundaries[2];
    boundaries[0] = 0;
    boundaries[1] = num_frames;
    basis_splines_endreps_local_v2(&knots[0], knots.size(), order, boundaries, sizeof(boundaries)/sizeof(boundaries[0]), Basis);
  }
  Eigen::MatrixXf SX = (Basis.transpose() * Basis).inverse() * Basis.transpose();
  Eigen::MatrixXf Y(frame_stride, num_frames);
  //  Eigen::MatrixXf FlowMeans(num_flows, num_frames);

  //  FlowMeans.setZero();
  int good_wells = 0;
  char *bad_start = bad_wells;
  char *bad_end = bad_start + frame_stride;
  while(bad_start != bad_end) {
    if (*bad_start++ == 0) {
      good_wells++;
    }
  }
  
  // if nothing good then skip it
  if (good_wells < MIN_SAMPLE_WELL) {
    return;
  }

  ChipReduction smoothed_avg;
  int x_clip = num_cols;
  int y_clip = num_rows;
  if (use_mesh_neighbors == 0) {
    x_clip = THUMBNAIL_SIZE;
    y_clip = THUMBNAIL_SIZE;
  }

  smoothed_avg.Init(num_rows, num_cols, num_frames,
                    SMOOTH_REDUCE_STEP, SMOOTH_REDUCE_STEP,
                    y_clip, x_clip,
                    SMOOTH_REDUCE_STEP * SMOOTH_REDUCE_STEP * .4);
  for (size_t frame_ix = 0; frame_ix < num_frames; frame_ix++) {
    smoothed_avg.ReduceFrame(data + flow_ix * frame_stride + frame_ix * flow_frame_stride, bad_wells, frame_ix);
  }
  smoothed_avg.SmoothBlocks(SMOOTH_REDUCE_REGION, SMOOTH_REDUCE_REGION);


  for (size_t frame_ix = 0; frame_ix < num_frames; frame_ix++) {
    float *y_start = Y.data() + frame_ix * frame_stride;
    float *y_end = y_start + frame_stride;
    int16_t *trace_start = data + flow_ix * frame_stride + frame_ix * flow_frame_stride;
    while(y_start != y_end) {
      *y_start++ = *trace_start++;
    }
  }
    
  // subtract off flow,frame avg
  for (size_t frame_ix = 0; frame_ix < num_frames; frame_ix++) {
    float *start = Y.data() + frame_stride * frame_ix;
    float *end = start + frame_stride;
    for (size_t row = 0; row < num_rows; row++) {
      for (size_t col = 0; col < num_cols; col++) {
        float avg = smoothed_avg.GetSmoothEst(row, col, frame_ix);
        *start++ -= avg;
      }
    }
  }

  // Get coefficients to solve
  Eigen::MatrixXf B = Y * SX.transpose();
  // Uncompress data into yhat matrix
  Eigen::MatrixXf Yhat = B * Basis.transpose();


  // add the flow/frame averages back
  for (size_t frame_ix = 0; frame_ix < num_frames; frame_ix++) {
    float *start = Y.data() + frame_stride * frame_ix;
    float *end = start + frame_stride;
    float *hstart = Yhat.data() + frame_stride * frame_ix;
    for (size_t row = 0; row < num_rows; row++) {
      for (size_t col = 0; col < num_cols; col++) {
        float avg = smoothed_avg.GetSmoothEst(row, col, frame_ix);
        *start++ += avg;
        *hstart++ += avg;
      }
    }
  }

  for (size_t frame_ix = 0; frame_ix < num_frames; frame_ix++) {
    float *yhat_start = Yhat.data() + frame_ix * frame_stride;
    float *yhat_end = yhat_start + frame_stride;
    int16_t *trace_start = data + flow_ix * frame_stride + frame_ix * flow_frame_stride;
    while(yhat_start != yhat_end) {
      *trace_start++ = (int)(*yhat_start + .5f);
      yhat_start++;
    }
  }

  Y = Y - Yhat;
  Eigen::VectorXf M = Y.rowwise().squaredNorm();

  float *mad_start = mad;
  float *mad_end = mad_start + frame_stride;
  float *m_start = M.data();
  while (mad_start != mad_end) {
    *mad_start += *m_start;
    mad_start++;
    m_start++;
  }

  mad_start = mad;
  mad_end = mad + frame_stride;
  int norm_factor = num_frames;// num_flows * num_frames;
  while (mad_start != mad_end) {
    *mad_start /= norm_factor;
    mad_start++;
  }
}

