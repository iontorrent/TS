/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DATCOMPRESSION_H
#define DATCOMPRESSION_H

#include <vector>

class DatCompression {
 public:
  virtual int GetCompressionType() = 0;
  virtual std::vector<int> compress(const std::vector<int> &input) = 0;
  virtual std::vector<int> decompress(const std::vector<int> &input) = 0;

};

#endif // DATCOMPRESSION_H
