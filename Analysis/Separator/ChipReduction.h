/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CHIPREDUCTION_H
#define CHIPREDUCTION_H

#include <stdlib.h>
#include <string.h>
#include "NNAvg.h"

class ChipReduction {

public:

  ChipReduction() { Zero(); }
  ChipReduction(int chip_height, int chip_width, int num_frames,
                int y_step, int x_step,
                int y_clip, int x_clip, int min_good_wells);

  ~ChipReduction() { Cleanup(); }
  
  void Init(int chip_height, int chip_width, int num_frames,
            int y_step, int x_step,
            int y_clip, int x_clip, int min_good_wells);

  void Reset() {
    int total_size = TotalSize();
    memset(m_block_avg, 0, sizeof(float) * total_size);
    memset(m_block_smooth, 0, sizeof(float) * total_size);
    memset(m_well_cont_block_smooth, 0, sizeof(float) * total_size);
    memset(m_good_wells, 0, sizeof(int) * total_size);
    memset(m_bad_wells, 0, sizeof(char) * total_size);
  }

  void Cleanup();

  void SetAvgReduce(bool flag) { m_avg_reduce = flag; }

  template <typename T> 
  void ReduceFrame(const T *values, const char *bad_wells, int frame_ix) {
    // loop through and create averages for each region
    for (int row_ix = 0; row_ix < m_chip_height; row_ix++) {
      int row_bin = row_ix / m_y_step;
      const T * val_last = values + (row_ix+1) * m_chip_width;
      int col_bin = 0;
      for (int col_ix = 0; col_ix < m_chip_width; col_ix += m_x_step) {
        int chip_offset = row_ix * m_chip_width + col_ix;
        const T *__restrict val_start = values + chip_offset;
        const T *__restrict val_end = std::min((const T *)(val_start + m_x_step),  val_last);
        const char *__restrict bad_start = bad_wells + chip_offset;
        int reduce_offset = row_bin * m_block_width + col_bin;
        float *__restrict block_avg = m_block_avg + frame_ix * m_block_frame_stride + reduce_offset;
        int *__restrict good_wells = m_good_wells + reduce_offset;
        while(val_start != val_end) {
          if (*bad_start == 0 && isfinite(*val_start)) {
            *block_avg += (float)(*val_start);
            if (frame_ix == 0) {
              (*good_wells)++;
            }
          }
          val_start++;
          bad_start++;
        }
        col_bin++;
      }
    }
    int *__restrict good_start = m_good_wells;
    int *__restrict good_end = good_start + m_block_frame_stride;
    char *__restrict bad_start = m_bad_wells;
    while(good_start != good_end) {
      if (*good_start >= m_min_good_reduced_wells) {
        *bad_start = 0;
      }
      else {
        *good_start = 0;
        *bad_start = 1;
      }
      good_start++;
      bad_start++;
    }
    if (m_avg_reduce) {
      float *__restrict block_start = m_block_avg + frame_ix * m_block_frame_stride;
      float *__restrict block_end =  block_start + m_block_frame_stride;
      good_start = m_good_wells;
      while (block_start != block_end) {
        if (*good_start >= m_min_good_reduced_wells) {
          *block_start =  *block_start / *good_start;
        }
        else {
          *block_start = 0;
        }
        block_start++;
        good_start++;
      }
    }
  }
      
  template <typename T>
  void Reduce(const T *values, const char *bad_wells) {
    for (int frame_ix = 0; frame_ix < m_num_frames; frame_ix++) {
      ReduceFrame(values + frame_ix * (m_chip_width * m_chip_height), bad_wells, frame_ix);
    }
  }

  size_t TotalSize() { return (size_t) m_block_width * m_block_height * m_num_frames; }

  void SmoothBlocks(int well_height, int well_width);

  inline float *GetSmoothEst() {
    return m_block_smooth; 
  }

  template<typename T>
  inline void GetSmoothEstFrames(int row, int col, T* __restrict out) {
    int brow = row / m_y_step;
    int bcol = col / m_x_step;
    float *__restrict start = m_well_cont_block_smooth + (brow * m_block_width + bcol) * m_num_frames;
    float *__restrict end = start + m_num_frames;
    while(start != end) {
      *out++ = (T)(*start++);
    }
  }

  const int *GetGoodWells() { return m_good_wells; }
  const char *GetBadWells() { return m_bad_wells; }

  inline float GetSmoothEst(int row, int col, int frame) {
    return GetSmoothEst((row / m_y_step) * m_block_width + (col / m_x_step), frame);
  }

  inline float GetBlockAvg(int block_ix, int frame_ix) {
    return *(m_block_avg + m_block_frame_stride * frame_ix + block_ix);
  }

  inline float GetSmoothEst(int block_ix, int frame_ix) {
    return *(m_block_smooth + m_block_frame_stride * frame_ix + block_ix);
  }

  inline int GetNumSmoothBlocks() { return m_block_width * m_block_height; }

  inline void GetBlockDims(int block_ix, int &row_start, int &row_end, int &col_start, int &col_end) {
    int block_row = block_ix / m_block_width;
    row_start = block_row * m_y_step;
    row_end = std::min((block_row + 1) * m_y_step, m_chip_height);
    int block_col = block_ix % m_block_width;
    col_start = block_col * m_x_step;
    col_end = std::min((block_col + 1) * m_x_step, m_chip_width);
  }

  inline int GetBlockWidth() { return m_block_width; }
  inline int GetBlockHeight() { return m_block_height; }
  NNAvg m_nn_avg;

private:
  void Zero();
  int m_avg_reduce;
  int m_min_good_reduced_wells;
  int m_x_clip, m_y_clip;
  int m_block_frame_stride;
  int m_block_width, m_block_height;
  int m_chip_width, m_chip_height, m_num_frames;
  int m_x_step, m_y_step;
  float *m_block_avg;
  float *m_block_smooth;
  float *m_well_cont_block_smooth;
  int *m_good_wells;
  char *m_bad_wells;

};

#endif // CHIPREDUCTION_H
