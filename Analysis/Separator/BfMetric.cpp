/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <algorithm>
#include <limits>
#include "BfMetric.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include "IonH5File.h"
#include "IonH5Eigen.h"
#include "Stats.h"
#include "ChipReduction.h"
#include "SampleStats.h"

#define BFMETRIC_REDUCE_STEP 5
#define BFMETRIC_MIN_GOOD_WELLS 15

void BfMetric::CalcBfSdMetric(const short *__restrict image, 
                              Mask *mask,
                              const char *__restrict bad_wells,
                              const float *__restrict t0,
                              const std::string &dbg_name,
                              int t0_window_start,
                              int t0_window_end,
                              int row_step,
                              int col_step,
                              int num_row_neighbors,
                              int num_col_neighbors) {
    
  // Calculate the average trace in region around each well
  m_nn_avg.Init(m_num_rows, m_num_cols, m_num_frames);
  m_nn_avg.SetGainMinMult(m_gain_min_mult);
  m_nn_avg.CalcCumulativeSum(image, mask, bad_wells);
  m_nn_avg.CalcNNAvgAndMinFrame(row_step, col_step, 
                                num_row_neighbors, num_col_neighbors,
                                m_trace_min, m_trace_min_frame,
                                image, true);

  short * __restrict well_nn_diff = (short *) memalign(VEC8F_SIZE_B, (sizeof(short) * m_num_rows) * (m_num_cols * m_num_frames));    
  int * __restrict well_nn_sq_diff = (int *) memalign(VEC8F_SIZE_B, (sizeof(int) * m_num_rows) * (m_num_cols * m_num_frames));    
  memset(well_nn_diff, 0, (sizeof(short) * m_num_rows) * (m_num_cols * m_num_frames));
  memset(well_nn_sq_diff, 0, (sizeof(int) * m_num_rows) * (m_num_cols * m_num_frames));
  // Subtract each well from the average around it, this is beadfind buffering estimate
  short *__restrict diff_start = well_nn_diff;
  short *__restrict diff_end = diff_start + (m_num_rows * m_num_cols * m_num_frames);
  const short *__restrict img = image;
  const float *__restrict nn_avg = m_nn_avg.GetNNAvgImagePtr();
  const float *__restrict nn_avg_start = m_nn_avg.GetNNAvgImagePtr();
  while (diff_start != diff_end) {
    *diff_start = (short) (*img - *nn_avg_start + .5);
    diff_start++;
    img++;
    nn_avg_start++;
  }

  // if (!dbg_name.empty()) {
  //   H5File h5(dbg_name);
  //   h5.Open(true);
  //   int num_wells = m_num_rows * m_num_cols;
  //   const short *__restrict diff = well_nn_diff;
  //   const float *__restrict avg = nn_avg;
  //   const short *__restrict img = image;
  //   Eigen::MatrixXf mat_diff(m_num_cols * m_num_rows, m_num_frames);
  //   Eigen::MatrixXf mat_avg(m_num_cols * m_num_rows, m_num_frames);
  //   Eigen::MatrixXf mat_data(m_num_cols * m_num_rows, m_num_frames);
  //   for (int fIx = 0; fIx < m_num_frames; fIx++) {
  //     for (int well_ix = 0; well_ix < num_wells; well_ix++) {
  //       mat_diff(well_ix,fIx) = *diff;
  //       mat_avg(well_ix, fIx) = *avg;
  //       mat_data(well_ix, fIx) = *img;
  //       diff++;
  //       avg++;
  //       img++;
  //     }
  //   }
  //   H5Eigen::WriteMatrix(h5, "/bf_diff", mat_diff);
  //   H5Eigen::WriteMatrix(h5, "/bf_avg", mat_avg);
  //   H5Eigen::WriteMatrix(h5, "/bf_img", mat_data);
  //   h5.Close();
  // }

  // Gotta cleanup memory asap as these things take a lot of mem for 318 series
  m_nn_avg.Cleanup();

  // Now calculate the sum of differences and squared differences to determine frame in region
  // that has the maximum variance over wells in region and use that as bf metric

  // Calculate the cumulative sum of difference in the region per frame
  m_diff_sum_avg.Init(m_num_rows, m_num_cols, m_num_frames);
  m_diff_sum_avg.FreeNNAvg();
  m_diff_sum_avg.CalcCumulativeSum(well_nn_diff, mask, bad_wells);

  // Calculate the squared cumulative sum of difference
  diff_start = well_nn_diff;
  diff_end = diff_start + (m_num_rows * m_num_cols * m_num_frames);
  int *__restrict diff_sq_start = well_nn_sq_diff;
  while (diff_start != diff_end) {
    *diff_sq_start = (*diff_start) * (*diff_start);
    diff_start++;
    diff_sq_start++;
  }
  m_diff_sum_sq_avg.Init(m_num_rows, m_num_cols, m_num_frames);
  m_diff_sum_sq_avg.FreeNNAvg();
  m_diff_sum_sq_avg.CalcCumulativeSum(well_nn_sq_diff, mask, bad_wells);
  m_diff_sum_sq_avg.FreeNumGoodWells();
  FREEZ(&well_nn_sq_diff);

  // Figure out the frame around which a well that has the largest deviation
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  size_t frame_stride = m_num_cols * m_num_rows;
  const int64_t *__restrict cum_sum = m_diff_sum_avg.GetNNCumSumPtr();
  const int64_t *__restrict cum_sum_sq = m_diff_sum_sq_avg.GetNNCumSumPtr();
  const int *__restrict num_good_wells = m_diff_sum_avg.GetNumGoodWellsPtr();

  // Allocate scratch data frames to keep track of the frame with maximum variance in region
  uint8_t *max_sd_frame = (uint8_t*) memalign(VEC8F_SIZE_B, sizeof(uint8_t) * m_num_cols * m_num_rows);
  memset(max_sd_frame, 0, sizeof(uint8_t) * m_num_cols * m_num_rows);
  float *max_sd_value = (float*) memalign(VEC8F_SIZE_B, sizeof(float) * m_num_cols * m_num_rows);
  memset(max_sd_value, 0, sizeof(float) * m_num_cols * m_num_rows);

  // Go through each frame and calculate the sd for each window for each well. 
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    const int64_t *__restrict cum_sum_frame = cum_sum + (fIx * cs_frame_stride);
    const int64_t *__restrict cum_sum_sq_frame = cum_sum_sq + (fIx * cs_frame_stride);
    int cs_col_size = m_num_cols + 1;
    int q1, q2,q3,q4;
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      for (int colIx = 0; colIx < m_num_cols; colIx++) {
        int well_ix = rowIx * m_num_cols + colIx;
        int row_below_clip = row_step * (rowIx / row_step) -1;
        int row_above_clip = std::min(row_below_clip + row_step, m_num_rows - 1);
        int col_below_clip = col_step * (colIx / col_step)  -1;
        int col_above_clip = std::min(col_below_clip + col_step, m_num_cols - 1);
          
        int r_start = std::max(row_below_clip,rowIx - num_row_neighbors - 1); // -1 ok for calc of edge conditions
        int r_end = std::min(rowIx+num_row_neighbors, row_above_clip);
        int c_start = std::max(col_below_clip, colIx - num_col_neighbors - 1); // -1 ok for calc of edge conditions
        int c_end = std::min(col_above_clip,colIx + num_col_neighbors);

        int64_t sum_sq = 0, sum = 0;
        int good_wells = 0;
        int bad_wells = 0;
        /* calculate sd stats for wells */
        q1 = (r_end + 1) * cs_col_size + c_end + 1;
        q2 = (r_start +1) * cs_col_size + c_end + 1;
        q3 = (r_end+1) * cs_col_size + c_start + 1;
        q4 = (r_start+1) * cs_col_size + c_start + 1;

        good_wells = num_good_wells[q1] - num_good_wells[q2] - num_good_wells[q3] + num_good_wells[q4];
        /* combine them if we got some good wells */

        if (good_wells > 0) {
          sum  = cum_sum_frame[q1] - cum_sum_frame[q2] - cum_sum_frame[q3] + cum_sum_frame[q4];
          sum_sq  = cum_sum_sq_frame[q1] - cum_sum_sq_frame[q2] - cum_sum_sq_frame[q3] + cum_sum_sq_frame[q4];
          float region_sd = (sum_sq - (float)(sum*sum/good_wells))/good_wells;
          if (region_sd > max_sd_value[well_ix] && fIx >= (t0[well_ix] + t0_window_start) && fIx <= (t0[well_ix] + t0_window_end)) {
            max_sd_value[well_ix] = region_sd;
            max_sd_frame[well_ix] = (uint8_t)fIx;
          }
        }
        else {
          // do nothing, leave at zero
        }

      }
    }
  }

  // Cleanup as fast as possible these things take up a lot of memory
  m_diff_sum_avg.Cleanup();
  m_diff_sum_sq_avg.Cleanup();
  FREEZ(&max_sd_value);
  // Loop through and grab the data from frame with the value at the maximum sd frame point
  int *__restrict bf_start = m_bf_metric;
  int *__restrict bf_end = bf_start + frame_stride;
  uint8_t *__restrict max_frame_start = max_sd_frame;
  int *__restrict trace_min_start = m_trace_min_frame;
  int well_index = 0;
  int frame_counts[m_num_frames];
  memset(frame_counts, 0, sizeof(int) * m_num_frames);
  while(bf_start != bf_end) {
    frame_counts[*max_frame_start]++;
    *trace_min_start = *max_frame_start;
    *bf_start += *(well_nn_diff + (*max_frame_start) * frame_stride + well_index);
    *bf_start += *(well_nn_diff + (std::max(*max_frame_start-1,0)) * frame_stride + well_index);
    *bf_start += *(well_nn_diff + (std::min(*max_frame_start+1,m_num_frames-1)) * frame_stride + well_index);
    trace_min_start++;
    well_index++;
    max_frame_start++;
    bf_start++;
  }

  FREEZ(&well_nn_diff);
  FREEZ(&max_sd_frame);
  FREEZ(&max_sd_value);
}


void BfMetric::CalcBfSdMetricReduce(const short *__restrict image, 
                                    Mask *mask,
                                    const char *__restrict bad_wells,
                                    const float *__restrict t0,
                                    const std::string &dbg_name,
                                    int t0_window_start,
                                    int t0_window_end,
                                    int row_step,
                                    int col_step,
                                    int num_row_neighbors,
                                    int num_col_neighbors,
                                    int frame_window) {
    
  // Calculate the average trace in region around each well
  ChipReduction reduce_avg(m_num_rows, m_num_cols, m_num_frames,
                       BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                       row_step, col_step, BFMETRIC_MIN_GOOD_WELLS);
  reduce_avg.Reduce(image, bad_wells);
  reduce_avg.SmoothBlocks(num_row_neighbors, num_col_neighbors);
                       
  short * __restrict well_nn_diff = (short *) memalign(VEC8F_SIZE_B, (sizeof(short) * m_num_rows) * (m_num_cols * m_num_frames));    
  int * __restrict well_nn_sq_diff = (int *) memalign(VEC8F_SIZE_B, (sizeof(int) * m_num_rows) * (m_num_cols * m_num_frames));    
  memset(well_nn_diff, 0, (sizeof(short) * m_num_rows) * (m_num_cols * m_num_frames));
  memset(well_nn_sq_diff, 0, (sizeof(int) * m_num_rows) * (m_num_cols * m_num_frames));

  // Subtract each well from the average around it, this is beadfind buffering estimate
  short *__restrict diff_start = well_nn_diff;
  int *__restrict diff_sq_start = well_nn_sq_diff;
  const short *__restrict img = image;
  
  for (int frame_ix = 0; frame_ix < m_num_frames; frame_ix++) {
    for (int row_ix = 0; row_ix < m_num_rows; row_ix++) {
      for (int col_ix = 0; col_ix < m_num_cols; col_ix++) {
        *diff_start = (short)(*img - reduce_avg.GetSmoothEst(row_ix, col_ix, frame_ix) + .5f);
        *diff_sq_start = (int)(*diff_start) * (*diff_start);
        diff_start++;
        diff_sq_start++;
        img++;
      }
    }
  }
  
  if (!dbg_name.empty()) {
    H5File h5(dbg_name);
    h5.Open(true);
    int num_wells = m_num_rows * m_num_cols;
    const short *__restrict diff = well_nn_diff;
    const short *__restrict img = image;
    Eigen::MatrixXf mat_diff(m_num_cols * m_num_rows, m_num_frames);
    Eigen::MatrixXf mat_avg(m_num_cols * m_num_rows, m_num_frames);
    Eigen::MatrixXf mat_data(m_num_cols * m_num_rows, m_num_frames);
    for (int fIx = 0; fIx < m_num_frames; fIx++) {
      for (int well_ix = 0; well_ix < num_wells; well_ix++) {
        mat_diff(well_ix,fIx) = *diff;
        mat_avg(well_ix, fIx) = reduce_avg.GetSmoothEst(well_ix / m_num_cols, well_ix % m_num_cols, fIx);
        mat_data(well_ix, fIx) = *img;
        diff++;
        img++;
      }
    }
    H5Eigen::WriteMatrix(h5, "/bf_diff", mat_diff);
    H5Eigen::WriteMatrix(h5, "/bf_avg", mat_avg);
    H5Eigen::WriteMatrix(h5, "/bf_img", mat_data);
    h5.Close();
  }

  ChipReduction diff_avg(m_num_rows, m_num_cols, m_num_frames,
                         BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                         row_step, col_step, BFMETRIC_MIN_GOOD_WELLS);
  diff_avg.SetAvgReduce(false);
  diff_avg.Reduce(well_nn_diff, bad_wells);
  diff_avg.SmoothBlocks(row_step, col_step);

  ChipReduction diff_sq_avg(m_num_rows, m_num_cols, m_num_frames,
                            BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                            row_step, col_step, BFMETRIC_MIN_GOOD_WELLS);
  diff_sq_avg.SetAvgReduce(false);
  diff_sq_avg.Reduce(well_nn_sq_diff, bad_wells);
  diff_sq_avg.SmoothBlocks(row_step, col_step);
  
  NNAvg good_wells(diff_avg.GetBlockHeight(), diff_avg.GetBlockWidth(), 1);
  good_wells.CalcCumulativeSum(diff_avg.GetGoodWells(), diff_avg.GetBadWells());
  
  // Now calculate the sum of differences and squared differences to determine frame in region
  // that has the maximum variance over wells in region and use that as bf metric

  // Allocate scratch data frames to keep track of the frame with maximum variance in region
  uint8_t *max_sd_frame = (uint8_t*) memalign(VEC8F_SIZE_B, sizeof(uint8_t) * m_num_cols * m_num_rows);
  memset(max_sd_frame, 0, sizeof(uint8_t) * m_num_cols * m_num_rows);
  float *max_sd_value = (float*) memalign(VEC8F_SIZE_B, sizeof(float) * m_num_cols * m_num_rows);
  memset(max_sd_value, 0, sizeof(float) * m_num_cols * m_num_rows);

  // Go through each frame and calculate the sd for each window for each well. 
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    int well_ix = 0;
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      for (int colIx = 0; colIx < m_num_cols; colIx++) {
        float sum_sq = 0;
        float sum = 0;
        float total_sum = 0;
        int good_count = 0;
        diff_avg.m_nn_avg.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, fIx,
                                       row_step/BFMETRIC_REDUCE_STEP, col_step/BFMETRIC_REDUCE_STEP,
                                       num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                       sum, good_count);
        diff_sq_avg.m_nn_avg.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, fIx,
                                          row_step/BFMETRIC_REDUCE_STEP, col_step/BFMETRIC_REDUCE_STEP,
                                          num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                          sum_sq, good_count);
        good_wells.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                                row_step/BFMETRIC_REDUCE_STEP, col_step/BFMETRIC_REDUCE_STEP,
                                num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                total_sum, good_count);
        // if (rowIx == 100 && colIx == 100) {
        //   fprintf(stdout, "Here we go\n");
        // }
        if (total_sum > 0) {
          float region_sd = (sum_sq - (float)(sum * sum/total_sum))/total_sum;
          if (region_sd > max_sd_value[well_ix] && fIx >= (t0[well_ix] + t0_window_start) && fIx <= (t0[well_ix] + t0_window_end)) {
            max_sd_value[well_ix] = region_sd;
            max_sd_frame[well_ix] = (uint8_t)fIx;
          }
        }
        else {
          // do nothing, leave at zero
        }
        well_ix++;
      }
    }
  }

  FREEZ(&max_sd_value);
  // Loop through and grab the data from frame with the value at the maximum sd frame point
  int *__restrict bf_start = m_bf_metric;
  int frame_stride = (m_num_rows * m_num_cols);
  int *__restrict bf_end = bf_start + frame_stride; 
  uint8_t *__restrict max_frame_start = max_sd_frame;
  int *__restrict trace_min_start = m_trace_min_frame;
  int well_index = 0;
  int frame_counts[m_num_frames];
  memset(frame_counts, 0, sizeof(int) * m_num_frames);
  while(bf_start != bf_end) {
    frame_counts[*max_frame_start]++;
    *trace_min_start = *max_frame_start;
    // *bf_start += *(well_nn_diff + (*max_frame_start) * frame_stride + well_index);
    // *bf_start += *(well_nn_diff + (std::max(*max_frame_start-1,0)) * frame_stride + well_index);
    // *bf_start += *(well_nn_diff + (std::min(*max_frame_start+1,m_num_frames-1)) * frame_stride + well_index);
    int f_start = max(0, *max_frame_start - frame_window);
    int f_end = min(m_num_frames - 1, *max_frame_start + frame_window);
    for (int f_ix = f_start; f_ix <= f_end; f_ix++) {
      *bf_start += *(well_nn_diff + f_ix * frame_stride + well_index);
    }
    trace_min_start++;
    well_index++;
    max_frame_start++;
    bf_start++;
  }

  FREEZ(&well_nn_diff);
  FREEZ(&well_nn_sq_diff);
  FREEZ(&max_sd_frame);
}

void BfMetric::NormalizeMetric(const char *bad_wells,
                               int row_clip, int col_clip,
                               int num_row_neighbors, 
                               int num_col_neighbors) {

  /* Get some filteres to avoid really bad outliers... */
  std::vector<char> norm_bad_wells(m_num_cols * m_num_cols);
  std::vector<float> bf_metric_sorted(norm_bad_wells.size());
  std::copy(m_bf_metric, m_bf_metric + norm_bad_wells.size(), bf_metric_sorted.begin());
  std::sort(bf_metric_sorted.begin(), bf_metric_sorted.end());
  float high_threshold = std::numeric_limits<float>::max();
  float low_threshold = -1.0f * high_threshold;
  if (bf_metric_sorted.size() > BFMETRIC_MIN_GOOD_WELLS) {
    float q25 = ionStats::quantile_sorted(bf_metric_sorted, .25);
    float q75 = ionStats::quantile_sorted(bf_metric_sorted, .75);
    float q02 = ionStats::quantile_sorted(bf_metric_sorted, .02);
    float q98 = ionStats::quantile_sorted(bf_metric_sorted, .98);
    float iqr = q75-q25;
    float hx3 = q75 + 3 * iqr;
    float lx3 = q25 - 3 * iqr;
    high_threshold = min(q98, hx3);
    low_threshold = max(lx3, q02);
  }
  /* If we fail our filters don't use it, otherwise default to regular mask. */
  for (size_t i = 0; i < norm_bad_wells.size(); i++) {
    if (m_bf_metric[i] <= low_threshold || m_bf_metric[i] >= high_threshold) {
      norm_bad_wells[i] = 1;
    }
    else {
      norm_bad_wells[i] = bad_wells[i];
    }
  }

  ChipReduction sum(m_num_rows, m_num_cols, 1,
                    BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                    row_clip, col_clip,
                    BFMETRIC_MIN_GOOD_WELLS);
  sum.SetAvgReduce(false);
  sum.Reduce(m_bf_metric, &norm_bad_wells[0]);
  sum.SmoothBlocks(num_row_neighbors, num_col_neighbors);
  std::vector<int> sq_bf_metric(m_num_cols * m_num_rows);
  int *__restrict bf_start = m_bf_metric;
  int *__restrict bf_end = m_bf_metric + m_num_rows * m_num_cols;
  int *__restrict sq_start = &sq_bf_metric[0];
  while(bf_start != bf_end) {
    *sq_start = *bf_start * *bf_start;
    sq_start++;
    bf_start++;
  }
  ChipReduction sum_sq(m_num_rows, m_num_cols, 1,
                       BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                       row_clip, col_clip,
                       BFMETRIC_MIN_GOOD_WELLS);
  sum_sq.SetAvgReduce(false);
  sum_sq.Reduce(&sq_bf_metric[0], &norm_bad_wells[0]);
  sum_sq.SmoothBlocks(num_row_neighbors, num_col_neighbors);
    
  std::vector<float> norm_ref(m_num_cols * m_num_rows);
  NNAvg good_wells(sum.GetBlockHeight(), sum.GetBlockWidth(), 1);
  good_wells.CalcCumulativeSum(sum.GetGoodWells(), sum.GetBadWells());
  SampleStats<float> bf_stats;
  SampleStats<float> ref_bf_stats;
  bf_start = m_bf_metric;
  bf_end = bf_start + norm_ref.size();
  const char *bad_start = &norm_bad_wells[0];
  while(bf_start != bf_end) {
    if (*bad_start == 0) {
      bf_stats.AddValue(*bf_start);
    }
    bad_start++;
    bf_start++;
  }

  int well_ix = 0;
  for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
    for (int colIx = 0; colIx < m_num_cols; colIx++) {
      float s = 0;
      float sq = 0;
      float total_sum = 0;
      int good_count = 0;
      sum.m_nn_avg.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                                row_clip/BFMETRIC_REDUCE_STEP, col_clip/BFMETRIC_REDUCE_STEP,
                                num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                s, good_count);
      sum_sq.m_nn_avg.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                                   row_clip/BFMETRIC_REDUCE_STEP, col_clip/BFMETRIC_REDUCE_STEP,
                                   num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                   sq, good_count);
      good_wells.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                              row_clip/BFMETRIC_REDUCE_STEP, col_clip/BFMETRIC_REDUCE_STEP,
                              num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                              total_sum, good_count);
      float ref_avg = sum.GetSmoothEst(rowIx, colIx, 0) / total_sum;
      float var = (sq - (float)(s * s)/total_sum)/total_sum;
      //        norm_ref[well_ix] = (m_bf_metric[well_ix] - ref_avg)/var
      if (good_count > 0 && var > 0) {
        m_norm_bf_metric[well_ix] = (m_bf_metric[well_ix] - ref_avg)/sqrt(var);
      }
      else {
        m_norm_bf_metric[well_ix] = (m_bf_metric[well_ix] - bf_stats.GetMean())/bf_stats.GetSD();
      }
      well_ix++;
    }
  }
}


void BfMetric::NormalizeMetricRef(const char *ref_wells, const char *bad_wells,
                                  int row_clip, int col_clip,
                                  int num_row_neighbors, 
                                  int num_col_neighbors) {

  /* Get some filteres to avoid really bad outliers... */
  std::vector<char> norm_bad_wells(m_num_rows * m_num_cols);
  std::vector<char> ref_flip(norm_bad_wells.size());
  std::vector<float> bf_metric_sorted(norm_bad_wells.size());
  std::copy(m_bf_metric, m_bf_metric + norm_bad_wells.size(), bf_metric_sorted.begin());
  std::sort(bf_metric_sorted.begin(), bf_metric_sorted.end());
  float high_threshold = std::numeric_limits<float>::max();
  float low_threshold = -1.0f * high_threshold;
  if (bf_metric_sorted.size() > BFMETRIC_MIN_GOOD_WELLS) {
    float q25 = ionStats::quantile_sorted(bf_metric_sorted, .25);
    float q75 = ionStats::quantile_sorted(bf_metric_sorted, .75);
    float q02 = ionStats::quantile_sorted(bf_metric_sorted, .02);
    float q98 = ionStats::quantile_sorted(bf_metric_sorted, .98);
    float iqr = q75-q25;
    float hx3 = q75 + 3 * iqr;
    float lx3 = q25 - 3 * iqr;
    high_threshold = min(q98, hx3);
    low_threshold = max(lx3, q02);
  }
  /* If we fail our filters don't use it, otherwise default to regular mask. */
  for (size_t i = 0; i < norm_bad_wells.size(); i++) {
    if (ref_wells[i] == 0) {
      ref_flip[i] = 1;
    }
    else {
      ref_flip[i] = 0;
    }
    if (m_bf_metric[i] <= low_threshold || m_bf_metric[i] >= high_threshold) {
      norm_bad_wells[i] = 1;
    }
    else {
      norm_bad_wells[i] = bad_wells[i];
    }
  }

  ChipReduction sum(m_num_rows, m_num_cols, 1,
                    BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                    row_clip, col_clip,
                    BFMETRIC_MIN_GOOD_WELLS);
  sum.SetAvgReduce(false);
  sum.Reduce(m_bf_metric, &norm_bad_wells[0]);
  sum.SmoothBlocks(num_row_neighbors, num_col_neighbors);
  std::vector<int> sq_bf_metric(m_num_cols * m_num_rows);
  int *__restrict bf_start = m_bf_metric;
  int *__restrict bf_end = m_bf_metric + m_num_rows * m_num_cols;
  int *__restrict sq_start = &sq_bf_metric[0];
  while(bf_start != bf_end) {
    *sq_start = *bf_start * *bf_start;
    sq_start++;
    bf_start++;
  }
  ChipReduction sum_sq(m_num_rows, m_num_cols, 1,
                       BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                       row_clip, col_clip,
                       BFMETRIC_MIN_GOOD_WELLS);
  sum_sq.SetAvgReduce(false);
  sum_sq.Reduce(&sq_bf_metric[0], &norm_bad_wells[0]);
  sum_sq.SmoothBlocks(num_row_neighbors, num_col_neighbors);
    
  std::vector<float> norm_ref(m_num_cols * m_num_rows);
  NNAvg good_wells(sum.GetBlockHeight(), sum.GetBlockWidth(), 1);
  good_wells.CalcCumulativeSum(sum.GetGoodWells(), sum.GetBadWells());
  SampleStats<float> bf_stats;
  SampleStats<float> ref_bf_stats;
  bf_start = m_bf_metric;
  bf_end = bf_start + norm_ref.size();
  const char *bad_start = &norm_bad_wells[0];
  const char *ref_start = ref_wells;
  while(bf_start != bf_end) {
    if (*bad_start == 0 && *ref_start == 1) {
      bf_stats.AddValue(*bf_start);
    }
    bad_start++;
    bf_start++;
    ref_start++;
  }

  ChipReduction ref_avg(m_num_rows, m_num_cols, 1,
                       BFMETRIC_REDUCE_STEP, BFMETRIC_REDUCE_STEP,
                       row_clip, col_clip,
                       BFMETRIC_MIN_GOOD_WELLS);
  ref_avg.Reduce(m_bf_metric, &ref_flip[0]);
  ref_avg.SmoothBlocks(num_row_neighbors, num_col_neighbors);

  int well_ix = 0;
  for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
    for (int colIx = 0; colIx < m_num_cols; colIx++) {
      float s = 0;
      float sq = 0;
      float total_sum = 0;
      int good_count = 0;
      sum.m_nn_avg.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                                row_clip/BFMETRIC_REDUCE_STEP, col_clip/BFMETRIC_REDUCE_STEP,
                                num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                s, good_count);
      sum_sq.m_nn_avg.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                                   row_clip/BFMETRIC_REDUCE_STEP, col_clip/BFMETRIC_REDUCE_STEP,
                                   num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                                   sq, good_count);
      good_wells.GetRegionSum(rowIx/BFMETRIC_REDUCE_STEP, colIx/BFMETRIC_REDUCE_STEP, 0,
                              row_clip/BFMETRIC_REDUCE_STEP, col_clip/BFMETRIC_REDUCE_STEP,
                              num_row_neighbors/BFMETRIC_REDUCE_STEP, num_col_neighbors/BFMETRIC_REDUCE_STEP,
                              total_sum, good_count);
      float all_mean = sum.GetSmoothEst(rowIx, colIx, 0) / total_sum;
      float ref_mean = ref_avg.GetSmoothEst(rowIx, colIx, 0);
      float var = (sq - (float)(s * s)/total_sum)/total_sum;
      if (good_count > 0 && var > 0) {
        float sd = sqrt(var);
        float ref_center = ref_mean / sd;
        m_norm_bf_metric[well_ix] = (m_bf_metric[well_ix] - all_mean)/sd - ref_center;
      }
      else {
        m_norm_bf_metric[well_ix] = (m_bf_metric[well_ix] - bf_stats.GetMean())/bf_stats.GetSD();
      }
      well_ix++;
    }
  }
}
