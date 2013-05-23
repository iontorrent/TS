/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BITHANDLER_H
#define BITHANDLER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <map>
#include <list>
#include <set>
#include <queue>
#include <bitset>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <complex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sstream>
#include <sys/time.h>
#include <x86intrin.h>
#include <string.h>
#include <stdint.h>
#include <armadillo>
#include "compression.h"
#include "IonErr.h"

using namespace std;
using namespace arma;

class BitUnpacker {
 public:

  BitUnpacker(const vector<uint8_t>& data);

  BitUnpacker(const int8_t *data, size_t _size) : cur_data(0), cur_data_bits(0), ptr((uint8_t *)data), start((uint8_t *)data) { size = _size;}

  BitUnpacker(const uint8_t *_data, size_t _size) : cur_data(0), cur_data_bits(0), ptr(_data), start(_data) { size = _size; }

  inline void set_size(int64_t n) { size = n; }

  inline int size_elements() { 
    return ptr - start;
  }

  inline void overflow() {
    cout << "overflow." << endl;
    assert(0);
  }

  inline void x_get() {
    if (size_elements() >= size) {
      overflow();
    }
    cur_data |= *ptr++ << cur_data_bits;
    cur_data_bits += 8;
  }

  inline void fill() { while ( cur_data_bits <= 24 ) x_get(); }

  int get_bit();
    
  inline uint32_t get_bits(unsigned bits) {
    if ( !bits ) return 0;
    fill();
    uint32_t ret;
    if ( bits >= cur_data_bits ) {
      ret = cur_data;
      unsigned got = cur_data_bits;
      cur_data = 0;
      cur_data_bits = 0;
      fill();
      ret |= (cur_data<<got)&((1u<<bits)-1);
      cur_data >>= bits-got;
      cur_data_bits -= bits-got;
      return ret;
    }
    ret = cur_data&((1u<<bits)-1);
    cur_data >>= bits;
    cur_data_bits -= bits;
    return ret;
  }

  inline uint32_t peek_bits(unsigned bits) {
    fill();
    return cur_data&((1u<<bits)-1);
  }

  unsigned peek_size() const {
    return cur_data_bits;
  }

  inline void skip_bits(unsigned bits) {
    cur_data >>= bits;
    cur_data_bits -= bits;
  }

  inline uint8_t get_u8() { return get_bits(8); }
  uint16_t get_u16() { return get_bits(16); }
  uint32_t get_u32() {
    return get_bits(32);  
  }

  void get_compressed(u8* vv, unsigned size);

  void get_compressed(vector<u8>& vv, unsigned size);

 private:
  unsigned cur_data, cur_data_bits;
  const uint8_t* ptr;
  const uint8_t* start;
  int64_t size;
};

class BitPacker {
 public:
  //  BitPacker(std::vector<uint8_t> &compressed);
  BitPacker();
  ~BitPacker();
  vector<uint8_t> &get_data() { return data; }
  unsigned size() const { return (data.size()*8+cur_data_bits); }

  inline void x_put() {
    while ( cur_data_bits >= 8 ) {
      data.push_back(cur_data&255);
      cur_data >>= 8;
      cur_data_bits -= 8;
    }
  }

  void put_bit(unsigned b);

  inline void put_bits(uint32_t v, unsigned bits) {
    uint32_t h = 0;
    asm("shldl %%cl, %[v], %[h]; shll %%cl,%[v]"
        : [v]"+r"(v), [h]"+r"(h) : [c]"c"(cur_data_bits));
    cur_data |= v;
    cur_data_bits += bits;
    if ( cur_data_bits >= 32 ) {
      unsigned more = cur_data_bits-32;
      cur_data_bits = 32;
      x_put();
      cur_data |= h << cur_data_bits;
      cur_data_bits += more;
    }
    x_put();
  }

  void clear() { data.resize(0); cur_data_bits = 0; cur_data = 0; flushed = false;}

  void flush();
  void put_u8(uint8_t v);
  void put_u16(uint16_t v);
  void put_u32(uint32_t v);

  void put_compressed(const uint8_t vv[], unsigned size);

  inline void put_compressed(const vector<uint8_t>& vv) {
    put_compressed(&vv[0], vv.size());
  }

  vector<uint8_t> data;
  unsigned cur_data, cur_data_bits;
  bool flushed;
};

struct Decompressor {
  Decompressor() { S = 0; L = 0; vv = NULL; };
  ~Decompressor() { };
	
  unsigned S, L;
  uint16_t* vv;
  void decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret);
  void decompress_nohuf(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret);
  void byteDecompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret);
};

struct Compressor {
  Compressor() { vv = NULL; S = 0; L = 0; };
  ~Compressor() { };
  BitPacker bc;
  unsigned S, L;
  const uint16_t* vv;
  std::vector<uint8_t> v0;
  std::vector<uint8_t> v1;
  std::vector<unsigned> v1x;
  void compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed); 
  void byteCompress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, std::vector<uint8_t> &compressed); 
};

class CompressorNH {
public:
  CompressorNH() { vv = NULL; S = 0; L = 0; }
  ~CompressorNH() { };
  BitPacker bc;
  unsigned S, L;
  const uint16_t* vv;
  std::vector<unsigned> v1x;
  void compress(const std::vector<uint16_t> &data, size_t nRows, size_t nCols, size_t nFrames, vector<uint8_t> &compressed);
};

struct DecompressorNH {
  DecompressorNH() { S = 0; L = 0; vv = NULL; };
  ~DecompressorNH() { };
	
  unsigned S, L;
  uint16_t* vv;
  void decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret);
};


#endif // BITHANDLER_H
