/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include "ImageNNAvg.h"
#include "Vecs.h"

ImageNNAvg::ImageNNAvg(int n_rows, int n_cols, int n_frames) {
  m_gain_min_mult = 1;
  InitializeClean();
  Alloc(n_rows, n_cols, n_frames);
}

void ImageNNAvg::Alloc(int n_rows, int n_cols, int n_frames) {
  Cleanup();
  m_num_rows = n_rows;
  m_num_cols = n_cols;
  m_num_frames = n_frames;
  int frame_stride = m_num_cols * m_num_rows;
    /* Data cube for cumulative sum for calculating averages fast. 
       Note the padding by 1 row and colum for easy code flow */
  m_cum_sum_size = (size_t)(m_num_cols +1) * (m_num_rows + 1) * m_num_frames;
  m_cum_sum = (int64_t *__restrict)memalign(VEC8F_SIZE_B, sizeof(int64_t) * m_cum_sum_size);
  assert(m_cum_sum);
  memset(m_cum_sum, 0, sizeof(int64_t) * m_cum_sum_size); // zero out
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  
  /* Mask of the cumulative number of good wells so we know denominator of average also padded. */
  m_num_good_wells = (int *__restrict) memalign(VEC8F_SIZE_B, sizeof(int) * cs_frame_stride);
  assert(m_num_good_wells);
  memset(m_num_good_wells, 0, sizeof(int) * cs_frame_stride);
  
  /* Data cube for our averages */
  m_nn_avg = (float *__restrict) memalign(VEC8F_SIZE_B, sizeof(float) * (size_t) frame_stride * m_num_frames);
  assert(m_nn_avg);
}

void  ImageNNAvg::Cleanup() {
  FREEZ(&m_cum_sum);
  FREEZ(&m_num_good_wells);
  FREEZ(&m_nn_avg);
  InitializeClean();
}

void ImageNNAvg::CalcCumulativeSum(const short *__restrict image, const Mask *mask, 
                                   const char *__restrict bad_wells) {
  /* Calculate the cumulative sum of images for each frame. */
  enum MaskType mask_ignore = (MaskType) (MaskPinned | MaskExclude | MaskIgnore);
  int64_t *__restrict cs_cur, *__restrict cs_prev; 
  int frame_stride = m_num_cols * m_num_rows;
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  const unsigned short *__restrict p_mask = mask->GetMask();
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      const short *__restrict col_ptr =  image + fIx * frame_stride + rowIx * m_num_cols;
      const short *__restrict col_ptr_end = col_ptr + m_num_cols;
      const unsigned short *__restrict c_mask = p_mask + rowIx * m_num_cols;
      const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
      cs_prev = m_cum_sum + (fIx * cs_frame_stride + rowIx * (m_num_cols+1)); // +1 due to padding
      cs_cur = cs_prev + m_num_cols + 1; // pointing at zero so needs to be incremented before assignment
      int64_t value;
      while(col_ptr != col_ptr_end) {
        value = *col_ptr++;
        if ((*c_mask & mask_ignore) != 0 || *c_bad_wells != 0) { value = 0.0f; }
        c_mask++;
        c_bad_wells++;
        value -= *cs_prev++;
        value += *cs_cur++ + *cs_prev;
        *cs_cur = value;
      }
    }
  }
    
  /* Calculate the cumulative sum of good wells similar to above. */
  for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
    const unsigned short *__restrict c_mask = p_mask + rowIx * m_num_cols;
    int *__restrict g_prev = m_num_good_wells + (rowIx * (m_num_cols+1));
    int *__restrict g_cur = g_prev + m_num_cols + 1;
    const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
    for (int colIx = 0; colIx < m_num_cols; colIx++) {
      int good = 1;
      if ((*c_mask & mask_ignore) != 0 || *c_bad_wells != 0) { good = 0; }
      c_mask++;
      c_bad_wells++;
      int x =  *g_cur++ + good - *g_prev++;
      x += *g_prev;
      *g_cur = x;
    }
  }
}

void ImageNNAvg::CalcCumulativeSum(const int *__restrict image, const Mask *mask, 
                                   const char *__restrict bad_wells) {
  /* Calculate the cumulative sum of images for each frame. */
  enum MaskType mask_ignore = (MaskType) (MaskPinned | MaskExclude | MaskIgnore);
  int64_t *__restrict cs_cur, *__restrict cs_prev; 
  int frame_stride = m_num_cols * m_num_rows;
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  const unsigned short *__restrict p_mask = mask->GetMask();
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      const int *__restrict col_ptr =  image + fIx * frame_stride + rowIx * m_num_cols;
      const int *__restrict col_ptr_end = col_ptr + m_num_cols;
      const unsigned short *__restrict c_mask = p_mask + rowIx * m_num_cols;
      const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
      cs_prev = m_cum_sum + (fIx * cs_frame_stride + rowIx * (m_num_cols+1)); // +1 due to padding
      cs_cur = cs_prev + m_num_cols + 1; // pointing at zero so needs to be incremented before assignment
      int64_t value;
      while(col_ptr != col_ptr_end) {
        value = *col_ptr++;
        if ((*c_mask & mask_ignore) != 0 || *c_bad_wells != 0) { value = 0.0f; }
        c_mask++;
        c_bad_wells++;
        value -= *cs_prev++;
        value += *cs_cur++ + *cs_prev;
        *cs_cur = value;
      }
    }
  }
    
  /* Calculate the cumulative sum of good wells similar to above. */
  for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
    const unsigned short *__restrict c_mask = p_mask + rowIx * m_num_cols;
    int *__restrict g_prev = m_num_good_wells + (rowIx * (m_num_cols+1));
    int *__restrict g_cur = g_prev + m_num_cols + 1;
    const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
    for (int colIx = 0; colIx < m_num_cols; colIx++) {
      int good = 1;
      if ((*c_mask & mask_ignore) != 0 || *c_bad_wells != 0) { good = 0; }
      c_mask++;
      c_bad_wells++;
      int x =  *g_cur++ + good - *g_prev++;
      x += *g_prev;
      *g_cur = x;
    }
  }
}

// Doesn't use the NN neighbors from same column due to column flicker
void ImageNNAvg::CalcNNAvgAndMinFrame(int row_step, int col_step, 
                                      int num_row_neighbors, int num_col_neighbors,
                                      float *__restrict trace_min, int *__restrict trace_min_frame,
                                      const short *__restrict image,
                                      bool replace_with_val) {
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  size_t frame_stride = m_num_cols * m_num_rows;
  /* Go through each frame and calculate the avg for each well. */
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    int64_t *__restrict cum_sum_frame = m_cum_sum + (fIx * cs_frame_stride);
    float *__restrict nn_avg_frame = m_nn_avg + (fIx * frame_stride);
    float *__restrict trace_min_cur = trace_min;
    const short *__restrict img_cur = image + fIx * frame_stride;
    int *__restrict trace_min_frame_cur = trace_min_frame;
    int cs_col_size = m_num_cols + 1;
    int q1, q2,q3,q4;
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      for (int colIx = 0; colIx < m_num_cols; colIx++) {
        // int r_start = std::max(-1,rowIx - 4); // -1 ok for calc of edge conditions
        // int r_end = std::min(rowIx+3, m_num_rows-1);
        // int c_start = std::max(-1, colIx - 7); // -1 ok for calc of edge conditions
        // int c_end = colIx-1;
        int row_below_clip = row_step * (rowIx / row_step) -1;
        int row_above_clip = std::min(row_below_clip + row_step, m_num_rows-1);
        int col_below_clip = col_step * (colIx / col_step)  -1;
        int col_above_clip = std::min(col_below_clip + col_step, m_num_cols-1);
          
        int r_start = std::max(row_below_clip,rowIx - num_row_neighbors - 1); // -1 ok for calc of edge conditions
        int r_end = std::min(rowIx+num_row_neighbors, row_above_clip);
        int c_start = std::max(col_below_clip, colIx - num_col_neighbors - 1); // -1 ok for calc of edge conditions
        int c_end = colIx-1;

        int64_t sum1 = 0.0f, sum2 = 0.0f;
        int count1 = 0, count2 = 0;
          
        /* average for wells in columns left of our well. */
        if (c_end >= 0) {
          q1 = (r_end + 1) * cs_col_size + c_end + 1;
          q2 = (r_start +1) * cs_col_size + c_end + 1;
          q3 = (r_end+1) * cs_col_size + c_start + 1;
          q4 = (r_start+1) * cs_col_size + c_start + 1;
          count1 = m_num_good_wells[q1] - m_num_good_wells[q2] - m_num_good_wells[q3] + m_num_good_wells[q4];
          sum1 = cum_sum_frame[q1] - cum_sum_frame[q2] - cum_sum_frame[q3] + cum_sum_frame[q4];
        }
          
        /* average for wells in column right of our well, rows still the same. */
        c_start = colIx;
        c_end = std::min(col_above_clip,colIx + num_col_neighbors);
        if (c_start < m_num_cols) {
          q1 = (r_end + 1) * cs_col_size + c_end+1;
          q2 = (r_start +1) * cs_col_size + c_end + 1;
          q3 = (r_end+1) * cs_col_size + c_start + 1;
          q4 = (r_start+1) * cs_col_size + c_start + 1;
          count2 = m_num_good_wells[q1] - m_num_good_wells[q2] - m_num_good_wells[q3] + m_num_good_wells[q4];
          sum2 = cum_sum_frame[q1] - cum_sum_frame[q2] - cum_sum_frame[q3] + cum_sum_frame[q4];
        }
        /* combine them if we got some good wells */
        if (count1 + count2 > 0) {
          *nn_avg_frame = ((float)sum1 + sum2)/(count1 + count2);
          if (m_gain_min_mult == 1 && *trace_min_cur > *img_cur) {
            *trace_min_cur = *img_cur;
            *trace_min_frame_cur = fIx;
          }
          else if (m_gain_min_mult == 0 && *trace_min_cur < *img_cur) {
            *trace_min_cur = *img_cur;
            *trace_min_frame_cur = fIx;
          }
        }
        else {
          if (replace_with_val) 
            *nn_avg_frame = *img_cur;
          else
            *nn_avg_frame = 0.0f;
        }
        img_cur++;
        nn_avg_frame++;
        trace_min_cur++;
        trace_min_frame_cur++;
      }
    }
  }
}

