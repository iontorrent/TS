/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PSYHOCOMPRESSION_H
#define PSYHOCOMPRESSION_H

#include "DatCompression.h"

class PsyhoDatCompression : public DatCompression {
public:
  int GetCompressionType();// { return TraceCompressor::Psyho; }
  std::vector<int> compress(const std::vector<int> &data);
  std::vector<int> decompress(const std::vector<int> &data);
};

#endif // PSYHOCOMPRESSION_H
