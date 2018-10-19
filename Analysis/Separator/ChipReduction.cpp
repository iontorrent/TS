/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include "ChipReduction.h"
#include "Utils.h"
#include "Vecs.h"
#include "NNAvg.h"

ChipReduction::ChipReduction(int chip_height, int chip_width, int num_frames,
                             int y_step, int x_step,
                             int y_clip, int x_clip, int min_good_wells) {
  Zero();
  Init(chip_height, chip_width, num_frames,
       y_step, x_step,
       y_clip, x_clip,
       min_good_wells);
}

void ChipReduction::Init(int chip_height, int chip_width, int num_frames,
                         int y_step, int x_step,
                         int y_clip, int x_clip, int min_good_wells) {
  Cleanup();
  m_chip_height = chip_height;
  m_chip_width = chip_width;
  m_num_frames = num_frames;
  m_y_step = y_step;
  m_x_step = x_step;
  m_y_clip = ceil((float)y_clip / m_y_step);
  m_x_clip = ceil((float)x_clip / m_x_step);
  m_min_good_reduced_wells = min_good_wells;
  m_block_height = ceil((float)m_chip_height / m_y_step);
  m_block_width = ceil((float) m_chip_width / m_x_step);
  m_block_frame_stride = m_block_width * m_block_height;
  int total_size = TotalSize();
  m_block_avg = (float *)memalign(VEC8F_SIZE_B, sizeof(float) * total_size);
  m_block_smooth = (float *)memalign(VEC8F_SIZE_B, sizeof(float) * total_size);
  m_well_cont_block_smooth = (float *)memalign(VEC8F_SIZE_B, sizeof(float) * total_size);
  m_good_wells = (int *)memalign(VEC8F_SIZE_B, sizeof(int) * total_size);
  m_bad_wells = (char *)memalign(VEC8F_SIZE_B, sizeof(char) * total_size);
  Reset();
}

void ChipReduction::Zero() {
  m_avg_reduce = true;
  m_min_good_reduced_wells = 0;
  m_x_clip = m_y_clip = m_block_frame_stride = m_block_width = m_block_height = 0;
  m_chip_width = m_chip_height = m_num_frames = m_x_step = m_y_step = 0;
  m_block_avg = m_block_smooth = m_well_cont_block_smooth = NULL;
  m_good_wells = NULL;
  m_bad_wells = NULL;
}

void ChipReduction::Cleanup() {
  FREEZ(&m_block_avg);
  FREEZ(&m_block_smooth);
  FREEZ(&m_well_cont_block_smooth);
  FREEZ(&m_good_wells);
  FREEZ(&m_bad_wells);
  Zero();
}

void ChipReduction::SmoothBlocks(int well_height, int well_width) {
  m_nn_avg.Init(m_block_height, m_block_width, m_num_frames);
  m_nn_avg.CalcCumulativeSum(m_block_avg, m_bad_wells);
  m_nn_avg.CalcNNAvg(m_y_clip, m_x_clip, 
               well_height / m_y_step, well_width / m_x_step);
  const float *nn_avg = m_nn_avg.GetNNAvgPtr();
  memcpy(m_block_smooth, nn_avg, sizeof(float) * TotalSize());
  for (int row = 0; row < m_block_height; row++) {
    for (int col = 0; col < m_block_width; col++) {
      for (int frame = 0; frame < m_num_frames; frame++) {
        m_well_cont_block_smooth[(row * m_block_width + col) * m_num_frames + frame] = m_block_smooth[frame * m_block_frame_stride + row * m_block_width + col];
      }
    }
  }
}

