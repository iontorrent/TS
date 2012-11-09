/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DELICATODATCOMPRESSION_H
#define DELICATODATCOMPRESSION_H

#include "DatCompression.h"

class DelicatoDatCompression : public DatCompression {
public:
  int GetCompressionType();// { return TraceCompressor::Delicato; }
  std::vector<int> compress(const std::vector<int> &data);
  std::vector<int> decompress(const std::vector<int> &data);
};

#endif // DELICATODATCOMPRESSION_H
