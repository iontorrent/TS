/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * DelicatoCompression.h
 * Final interface
 *
 * @author Magnus Jedvert
 * @version 1.1 april 2012
 *
 * Uses armadillo:
 * Conrad Sanderson.
 * Armadillo: An Open Source C++ Linear Algebra Library for
 * Fast Prototyping and Computationally Intensive Experiments.
 * Technical Report, NICTA, 2010.
 */
#ifndef DELICATOCOMPRESSION_H
#define DELICATOCOMPRESSION_H

#include <vector>
#include <cstring>
#include <stdint.h>
using std::vector;
class DelicatoCompression {
 public:
  DelicatoCompression() { doBoth = false; }
  /**
   * compress - Takes as input a pointer to a packed array of uint16_t and 
   * interprets this as a Lx(w*h) matrix (column-major order). Returns a 
   * compressed version of the data in argument compressed. 
   * Returns 0 if success, 1 otherwise.
   */
  //static int compress(const vector<uint16_t> &input_data, size_t w, size_t h, size_t L, vector<uint8_t> &compressed, float errCon);
  int compress(const vector<uint16_t> &input_data, size_t w, size_t h, size_t L, vector<uint8_t> &compressed, float errCon=49, int rankGood = 8, float pivot = 0);

  /**
   * decompress - Reverse of compress. Takes the compressed data as input and
   * fills in output_data with uncompressed data, and also fills in the size of 
   * the region (w, h) and number of frames (L). 
   * Returns 0 if success, 1 otherwise.
   */
  //static int decompress(const vector<uint8_t> &compressed, vector<uint16_t> &output_data, size_t &w, size_t &h, size_t &L);
  int decompress(const vector<uint8_t> &compressed, vector<uint16_t> &output_data, size_t &w, size_t &h, size_t &L);
  void SetDoBoth(bool db) { doBoth = db; }
 private:
  bool doBoth;
};

#endif // DELICATOCOMPRESSION_H
