/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWCHUNK_H
#define FLOWCHUNK_H


struct FlowChunk {
  size_t CompressionType;
  size_t ChipRow, ChipCol, ChipFrame;
  size_t RowStart, ColStart, FrameStart, FrameStep;
  size_t Height, Width, Depth; // row, col, frame
  size_t OrigFrames;
  int StartDetailedTime, StopDetailedTime, LeftAvg;
  float T0;
  float Sigma;
  float TMidNuc;
  float BaseFrameRate;
  hvl_t Data;
  hvl_t DeltaFrame;
};

#endif // FLOWCHUNK_H
