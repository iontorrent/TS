/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include "NNAvg.h"
#include "Vecs.h"

NNAvg::NNAvg(int n_rows, int n_cols, int n_frames) {
  InitializeClean();
  Alloc(n_rows, n_cols, n_frames);
}

void NNAvg::Alloc(int n_rows, int n_cols, int n_frames) {
  Cleanup();
  m_num_rows = n_rows;
  m_num_cols = n_cols;
  m_num_frames = n_frames;
  m_cs_col_size = m_num_cols + 1;
  m_cs_frame_stride = m_cs_col_size * (m_num_rows + 1);
  int frame_stride = m_num_cols * m_num_rows;
  /* Data cube for cumulative sum for calculating averages fast. 
     Note the padding by 1 row and colum for easy code flow */
  m_cum_sum_size = (size_t)(m_num_cols +1) * (m_num_rows + 1) * m_num_frames;
  m_cum_sum = (float *__restrict)memalign(VEC8F_SIZE_B, sizeof(float) * m_cum_sum_size);
  assert(m_cum_sum);
  memset(m_cum_sum, 0, sizeof(float) * m_cum_sum_size); // zero out
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  
  /* Mask of the cumulative number of good wells so we know denominator of average also padded. */
  m_num_good_wells = (int *__restrict) memalign(VEC8F_SIZE_B, sizeof(int) * cs_frame_stride);
  assert(m_num_good_wells);
  memset(m_num_good_wells, 0, sizeof(int) * cs_frame_stride);
  
  /* Data cube for our averages */
  m_nn_avg = (float *__restrict) memalign(VEC8F_SIZE_B, sizeof(float) * (size_t) frame_stride * m_num_frames);
  assert(m_nn_avg);
}

void  NNAvg::Cleanup() {
  FREEZ(&m_cum_sum);
  FREEZ(&m_num_good_wells);
  FREEZ(&m_nn_avg);
  InitializeClean();
}

void NNAvg::CalcCumulativeSum(const float *__restrict data, const char *__restrict bad_wells) {
  /* Calculate the cumulative sum of images for each frame. */
  float *__restrict cs_cur, *__restrict cs_prev; 
  int frame_stride = m_num_cols * m_num_rows;
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      const float *__restrict col_ptr =  data + fIx * frame_stride + rowIx * m_num_cols;
      const float *__restrict col_ptr_end = col_ptr + m_num_cols;
      const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
      cs_prev = m_cum_sum + (fIx * cs_frame_stride + rowIx * (m_num_cols+1)); // +1 due to padding
      cs_cur = cs_prev + m_num_cols + 1; // pointing at zero so needs to be incremented before assignment
      float value;
      while(col_ptr != col_ptr_end) {
        value = *col_ptr++;
        if (*c_bad_wells != 0) { 
          value = 0.0f; 
        }
        c_bad_wells++;
        value -= *cs_prev++;
        value += *cs_cur++ + *cs_prev;
        *cs_cur = value;
      }
    }
  }
    
  /* Calculate the cumulative sum of good wells similar to above. */
  for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
    int *__restrict g_prev = m_num_good_wells + (rowIx * (m_num_cols+1));
    int *__restrict g_cur = g_prev + m_num_cols + 1;
    const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
    for (int colIx = 0; colIx < m_num_cols; colIx++) {
      int good = 1;
      if (*c_bad_wells != 0) { good = 0; }
      c_bad_wells++;
      int x =  *g_cur++ + good - *g_prev++;
      x += *g_prev;
      *g_cur = x;
    }
  }
}

void NNAvg::CalcCumulativeSum(const int *__restrict data, const char *__restrict bad_wells) {
  /* Calculate the cumulative sum of images for each frame. */
  float *__restrict cs_cur, *__restrict cs_prev; 
  int frame_stride = m_num_cols * m_num_rows;
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      const int *__restrict col_ptr =  data + fIx * frame_stride + rowIx * m_num_cols;
      const int *__restrict col_ptr_end = col_ptr + m_num_cols;
      const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
      cs_prev = m_cum_sum + (fIx * cs_frame_stride + rowIx * (m_num_cols+1)); // +1 due to padding
      cs_cur = cs_prev + m_num_cols + 1; // pointing at zero so needs to be incremented before assignment
      float value;
      while(col_ptr != col_ptr_end) {
        value = *col_ptr++;
        if (*c_bad_wells != 0) { 
          value = 0.0f; 
        }
        c_bad_wells++;
        value -= *cs_prev++;
        value += *cs_cur++ + *cs_prev;
        *cs_cur = value;
      }
    }
  }
    
  /* Calculate the cumulative sum of good wells similar to above. */
  for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
    int *__restrict g_prev = m_num_good_wells + (rowIx * (m_num_cols+1));
    int *__restrict g_cur = g_prev + m_num_cols + 1;
    const char *__restrict c_bad_wells = bad_wells + rowIx * m_num_cols;
    for (int colIx = 0; colIx < m_num_cols; colIx++) {
      int good = 1;
      if (*c_bad_wells != 0) { good = 0; }
      c_bad_wells++;
      int x =  *g_cur++ + good - *g_prev++;
      x += *g_prev;
      *g_cur = x;
    }
  }
}

// Doesn't use the NN neighbors from same column due to column flicker
void NNAvg::CalcNNAvg(int row_clip, int col_clip, 
                      int num_row_neighbors, int num_col_neighbors) {
  int cs_frame_stride = (m_num_cols + 1) * (m_num_rows + 1);
  size_t frame_stride = m_num_cols * m_num_rows;
  /* Go through each frame and calculate the avg for each well. */
  for (int fIx = 0; fIx < m_num_frames; fIx++) {
    float *__restrict cum_sum_frame = m_cum_sum + (fIx * cs_frame_stride);
    float *__restrict nn_avg_frame = m_nn_avg + (fIx * frame_stride);
    int cs_col_size = m_num_cols + 1;
    int q1, q2,q3,q4;
    for (int rowIx = 0; rowIx < m_num_rows; rowIx++) {
      for (int colIx = 0; colIx < m_num_cols; colIx++) {
        int row_below_clip = row_clip * (rowIx / row_clip) -1;
        int row_above_clip = std::min(row_below_clip + row_clip, m_num_rows-1);
        int col_below_clip = col_clip * (colIx / col_clip)  -1;
        int col_above_clip = std::min(col_below_clip + col_clip, m_num_cols-1);
          
        int r_start = std::max(row_below_clip,rowIx - num_row_neighbors - 1); // -1 ok for calc of edge conditions
        int r_end = std::min(rowIx+num_row_neighbors, row_above_clip);
        int c_start = std::max(col_below_clip, colIx - num_col_neighbors - 1); // -1 ok for calc of edge conditions
        int c_end = std::min(col_above_clip,colIx + num_col_neighbors);

        float sum = 0.0f;
        int good_wells = 0;

        /* calculate sd stats for wells */
        q1 = (r_end + 1) * cs_col_size + c_end + 1;
        q2 = (r_start +1) * cs_col_size + c_end + 1;
        q3 = (r_end+1) * cs_col_size + c_start + 1;
        q4 = (r_start+1) * cs_col_size + c_start + 1;
        
        good_wells = m_num_good_wells[q1] - m_num_good_wells[q2] - m_num_good_wells[q3] + m_num_good_wells[q4];
        if (good_wells > 0) {
          sum  = cum_sum_frame[q1] - cum_sum_frame[q2] - cum_sum_frame[q3] + cum_sum_frame[q4];
          *nn_avg_frame = sum/good_wells;
        }
        else {
          *nn_avg_frame = 0.0f;
        }
        nn_avg_frame++;
      }
    }
  }
}

