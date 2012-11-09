/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BitHandler.h"
#include "HuffmanEncode.h"

typedef vector<string> VS;
// typedef long long i64;
// typedef unsigned long long u64;
// typedef unsigned char uu8;
// typedef unsigned uu32;
// typedef unsigned short uu16;
// typedef signed char byte;
// typedef signed char byte;
typedef vector<uint16_t> V16;
typedef vector<int> VI;
typedef vector<VI> VVI;
typedef vector<double> VD;
//}
////////////////////////// macros and typedefs ///////////////////////////////

#define STRINGx(a) #a
#define STRING(a) STRINGx(a)
#define NAME2x(a,b) a##b
#define NAME2(a,b) NAME2x(a,b)
#define ALL(C) (C).begin(), (C).end()
#define forIter(I,C) for(typeof((C).end()) I=(C).begin(); I!=(C).end(); ++I)
#define forIterE(I,C) for(typeof((C).end()) I=(C).begin(), E=(C).end(); I!=E; ++I)
#define forNF(I,F,C) for(int I=(F); I<int(C); ++I)
#define forN(I,C) forNF(I,0,C)
#define forEach(I,C) for(int I=0; I<int((C).size()); ++I)
#define NOP() do{}while(0)
#define EATSEMICOLON extern int eat_semicolon_variable
#define BRA { EATSEMICOLON
#define KET } EATSEMICOLON
#define ALIGN16 __attribute__((aligned(16)))

template<class V>
__attribute__((always_inline)) static inline void push_back_hack(vector<V>& col, V v) {
  assert(col.size() < col.capacity());
  assert(&col[0] == ((V**)&col)[0]);
  assert(&*col.end() == ((V**)&col)[1]);
  *((V**)&col)[1]++ = v;
}
template<class V>
__attribute__((always_inline)) static inline void set_end_hack(vector<V>& dst, V* end) {
  ((V**)(&dst))[1] = end;
}
template<class V>
static V* reserve_add_realloc(vector<V>& dst, size_t len) {
  V* end = ((V**)(&dst))[1];
  V* end_alloc = ((V**)(&dst))[2];
  assert(end <= end_alloc);
  assert(end+len > end_alloc);
  V* begin = ((V**)(&dst))[0];
  assert(begin <= end);
  size_t need_size = end-begin+len;
  size_t new_size = max(size_t(end_alloc-end),(size_t)256u);
  while ( new_size < need_size ) new_size *= 16;
  dst.reserve(new_size);
  end = ((V**)(&dst))[1];
  return end;
}
template<class V>
__attribute__((always_inline)) static inline V* reserve_add(vector<V>& dst, size_t len) {
  V* end = ((V**)(&dst))[1];
  V* end_alloc = ((V**)(&dst))[2];
  assert(end <= end_alloc);
  if ( end+len > end_alloc ) {
    end = reserve_add_realloc(dst, len);
  }
  return end;
}
template<class V>
__attribute__((always_inline)) static inline void add_hack(vector<V>& dst, const V* src, size_t len) {
  V* end = reserve_add(dst, len);
  for ( ; len; --len ) *end++ = *src++;
  set_end_hack(dst, end);
}

template<class V>
__attribute__((always_inline)) static inline void add_hack(vector<V>& dst, V v) {
  V* end = reserve_add(dst, 1);
  *end++ = v;
  set_end_hack(dst, end);
}


//BitPacker::BitPacker(std::vector<uint8_t> &compressed) : data(compressed), cur_data(0), cur_data_bits(0) { }

BitPacker::BitPacker() : cur_data(0), cur_data_bits(0), flushed(false) { }

BitPacker::~BitPacker() {
}

void BitPacker::put_compressed(const uint8_t vv[], unsigned size)
{
  if ( !size ) return;
  Huffman h(vv, size);
  h.encode(*this);
  forN ( i, size ) h.put(vv[i], *this);
}

BitUnpacker::BitUnpacker(const vector<uint8_t>& data) : cur_data(0), cur_data_bits(0), ptr(&data[0]), start(&data[0]) { size = data.size();}

void BitUnpacker::get_compressed(uint8_t* vv, unsigned size) {
  if ( !size ) return;
  Huffman h;
  h.decode(*this);
  forN ( i, size ) vv[i] = h.get(*this);
}

void Decompressor::decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret) {
  ret.clear();
  BitUnpacker bc(compressed);
  unsigned X = bc.get_bits(31);
  unsigned Y = bc.get_bits(31);
  S = X*Y;
  L = bc.get_bits(31);
  nRows = X;
  nCols = Y;
  nFrames = L;
  ret.resize(S*L);
  set_end_hack(ret, &*ret.begin()+S*L);
  vv = &ret[0];
  vector<uint8_t> v0r(S);
  forN ( i, S ) v0r[i] = bc.get_bits(6);
  vector<uint8_t> v0;
  bc.get_compressed(v0, S);
  vector<uint8_t> v1;
  bc.get_compressed(v1, S*(L-1));
  uint16_t* p = vv;
  forN ( i, S ) {
    unsigned v = (v0[i]<<6)+v0r[i];
    *p++ = v;
    forNF ( j, 1, L ) {
      int d = v1[i*(L-1)+j-1];
      if ( d == 255 ) d = bc.get_bits(17);
      d = (d>>1)^-(d&1);
      v += d;
      *p++ = v;
    }
  }
}

/* byteDecompress */
void Decompressor::byteDecompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret) {
  ret.clear();
  //BitUnpacker bc(compressed);
  ByteUnpacker bp((char *)compressed.data());
  //unsigned X = bc.get_bits(31);
  //unsigned Y = bc.get_bits(31);
  vector<size_t> header = bp.pop<size_t>(6);
  unsigned X = header[0];
  unsigned Y = header[1];
  S = X*Y;
  //L = bc.get_bits(31);
  L = header[2];
  nRows = X;
  nCols = Y;
  nFrames = L;
  //  TR(X|Y|L);
  ret.resize(S*L);
  set_end_hack(ret, &*ret.begin()+S*L);
  vv = &ret[0];
  //vector<uint8_t> v0r(S);
  //forN ( i, S ) v0r[i] = bc.get_bits(6);
  vector<uint16_t> v0;
  vector<uint8_t> v1;
  vector<uint16_t> v1x;
  v0 = bp.pop<uint16_t>(header[3]);
  v1 = bp.pop<uint8_t>(header[4]);
  v1x = bp.pop<uint16_t>(header[5]);
  //vector<uint8_t> v0;
  //bc.get_compressed(v0, S);
  //vector<uint8_t> v1;
  //bc.get_compressed(v1, S*(L-1));
  size_t count = 0;
  uint16_t* p = vv;
  forN ( i, S ) {
    //unsigned v = (v0[i]<<6)+v0r[i];
    unsigned v = v0[i];
    *p++ = v;
    forNF ( j, 1, L ) {
      int d = v1[i*(L-1)+j-1];
      //if ( d == 255 ) d = bc.get_bits(17);
      if ( d == 255 ) d = v1x[count++];
      d = (d>>1)^-(d&1);
      v += d;
      *p++ = v;
    }
  }
}


void Compressor::compress(const V16& data, size_t nRows, size_t nCols, size_t nFrames, vector<uint8_t> &compressed) {
  compressed.clear();
  BitPacker bc;
  unsigned X = nRows, Y = nCols;
  S = X*Y;
  L = nFrames;
  bc.data.reserve(unsigned(S*L*.7));
  //  TR(X|Y|L);
  bc.put_bits(nRows, 31);
  bc.put_bits(nCols, 31);
  bc.put_bits(nFrames, 31);
  vv = &data[0];
  v0.clear();
  v1.clear();
  v1x.clear();
  v0.reserve(S);
  v1.reserve(S*(L-1));

  const uint16_t* p = vv;
  forN ( i, S ) {
    unsigned v = *p++;
    bc.put_bits(v&63, 6);
    push_back_hack(v0, uint8_t(v>>6));
    forNF ( j, 1, L ) {
      unsigned n = *p++;
      int d = n-v;
      d = (d*2)^(d>>31);
      assert(d >= 0);
      if ( d >= 255 ) {
	v1x.push_back(d);
	d = 255;
      }
      push_back_hack(v1, uint8_t(d));
      v = n;
    }
  }
  bc.put_compressed(v0);
  bc.put_compressed(v1);
  //forIter ( i, v1x ) bc.put_bits(*i, 17);
  for(vector<unsigned>::iterator i=v1x.begin(); i!=v1x.end(); ++i)
    bc.put_bits(*i, 17);
  bc.flush();
  compressed = bc.get_data();
}

void CompressorNH::compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, vector<uint8_t> &compressed) {
  compressed.clear();
  unsigned X = nRows, Y = nCols;
  S = X*Y;
  L = nFrames;
  bc.clear();
  bc.data.reserve(unsigned(X*L));
  compressed.reserve(X*Y*(L+1)*2+12); 
  compressed.resize(X*Y*(L+1)+12); 
  int *header = (int *)&compressed[0];
  // todo - network order bytes here...
  header[0] = X;
  header[1] = Y;
  header[2] = L;
  size_t current = 12;

  const uint16_t* p = &data[0];
  forN ( i, S ) {
    unsigned v = *p++;
    compressed[current++] = v;
    compressed[current++] = v >> 8;
    forNF ( j, 1, L ) {
      unsigned n = *p++;
      int d = n-v;
      d = (d*2)^(d>>31);
      if ( d >= 255 ) {
        bc.put_bits(d, 17);
        d = 255;
      }
      compressed[current++] = d;
      v = n;
    }
  }
  bc.flush();
  size_t currentSize = compressed.size();
  compressed.resize(currentSize + bc.data.size());
  copy(bc.data.begin(), bc.data.end(), compressed.begin() + currentSize);
}

void DecompressorNH::decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret) {
  ret.clear();
  // @todo - neorder these bits
  int *header = (int *)&compressed[0];
  unsigned X = header[0];
  unsigned Y = header[1];
  unsigned L = header[2];
  S = X*Y;

  size_t current = 12;
  size_t end = X*Y*(L+1)+12;
  nRows = X;
  nCols = Y;
  nFrames = L;
  ret.resize(S*L);
  uint16_t* p = &ret[0];
  const uint8_t *eptr = &compressed[0] + end;
  BitUnpacker bc(eptr, S*L*3);
  const uint8_t *sptr = &compressed[0] + current;
  size_t sIdx = 0;
  forN ( i, S ) {
    short v = (sptr[sIdx++]);
    short u = (sptr[sIdx] << 8);
    sIdx++;
    *p++ = v = v + u;
    forNF ( j, 1, L ) {
      int d = sptr[sIdx++];
      if ( d == 255 ) {
        d = bc.get_bits(17);
      }
      d = (d>>1)^-(d&1);
      v += d;
      *p++ = v;
    }
  }
}


// void CompressorNH::compress(const std::vector<uint16_t>& data, size_t nRows, size_t nCols, size_t nFrames, vector<uint8_t> &compressed) {
//   compressed.clear();
//   unsigned X = nRows, Y = nCols;
//   S = X*Y;
//   L = nFrames;
//   compressed.reserve(X*Y*(L+1)*2+12); 
//   compressed.resize(X*Y*(L+1)+12); 
//   int *header = (int *)&compressed[0];
//   // todo - network order bytes here...
//   header[0] = X;
//   header[1] = Y;
//   header[2] = L;
//   size_t current = 12;

//   const uint16_t* p = &data[0];
//   forN ( i, S ) {
//     unsigned v = *p++;
//     compressed[current++] = v;
//     compressed[current++] = v >> 8;
//     forNF ( j, 1, L ) {
//       short n = *p++;
//       short d = (short)n-v;
//       int dd = n-v;
//       if (current - 12 == 101911) {
//         cout << "Here we go";
//       }
//       if (dd != d) {
//           cout << "How?" << endl;
//       }
//       //      assert(d >= 0);
//       if ( d >= 127 || d <= -127 ) {
//         short x = n - v;
//         int xx = n - v;
//         if (xx >= 32768 || xx <= -32768) {
//           cout << "How?" << endl;
//         }
//         v1x.push_back(x);
// 	v1x.push_back(x >> 8);
// 	d = 127;
//       }
//       compressed[current++] = d;
//       v = n;
//     }
//   }
//   size_t currentSize = compressed.size();
//   compressed.resize(currentSize + v1x.size());
//   copy(v1x.begin(), v1x.end(), compressed.begin() + currentSize);
// }

// void DecompressorNH::decompress(const std::vector<uint8_t> &compressed, size_t &nRows, size_t &nCols,size_t &nFrames, std::vector<uint16_t>& ret) {
//   ret.clear();
//   @todo - neorder these bits
//   int *header = (int *)&compressed[0];
//   unsigned X = header[0];
//   unsigned Y = header[1];
//   unsigned L = header[2];
//   S = X*Y;

//   size_t current = 12;
//   size_t end = X*Y*(L+1)+12;
//   nRows = X;
//   nCols = Y;
//   nFrames = L;
//   ret.resize(S*L);
//   uint16_t* p = &ret[0];
//   const uint8_t *eptr = &compressed[0] + end;
//   const uint8_t *sptr = &compressed[0] + current;
//   size_t sIdx = 0;
//   size_t eIdx = 0;
//   forN ( i, S ) {
//     short v = (sptr[sIdx++]);
//     short u = (sptr[sIdx] << 8);
//     sIdx++;
//     *p++ = v = v + u;
//     forNF ( j, 1, L ) {
//       if (sIdx == 101911) {
//         cout << "Here we go";
//       }
//       short d = (char)sptr[sIdx++];
//       if ( d == 127 ) {
//         short x = eptr[eIdx++];
//         short y = eptr[eIdx++];
//         d = (short)(y << 8) | x;
//       }
//       else {
//         d = (d>>1)^-(d&1);
//       }
//       v += d;
//       *p++ = v;
//     }
//   }
// }



void Compressor::byteCompress(const V16& data, size_t nRows, size_t nCols, size_t nFrames, vector<uint8_t> &compressed) {
  compressed.clear();
  //BitPacker bc(compressed);
  BytePacker bp(compressed);
  unsigned X = nRows, Y = nCols;
  S = X*Y;
  L = nFrames;
  //TR(X|Y|L);
  vector<size_t> header(6);
  header[0] = nRows;
  header[1] = nCols;
  header[2] = nFrames;
  vv = &data[0];
  vector<uint16_t> v0;
  v0.reserve(S);
  vector<uint8_t> v1;
  v1.reserve(S*(L-1));
  vector<uint16_t> v1x;
  const uint16_t* p = vv;
  forN ( i, S ) {
    unsigned v = *p++;
    v0.push_back(v);
    forNF ( j, 1, L ) {
      unsigned n = *p++;
      int d = n-v;
      d = (d*2)^(d>>31);
      assert(d >= 0);
      if ( d >= 255 ) {
	v1x.push_back(d);
	d = 255;
      }
      //push_back_hack(v1, uint8_t(d));
      v1.push_back(uint8_t(d));
      v = n;
    }
  }
  //bc.put_compressed(v0);
  //bc.put_compressed(v1);
  //forIter ( i, v1x ) bc.put_bits(*i, 17);
  header[3] = v0.size();
  header[4] = v1.size();
  header[5] = v1x.size();
  bp.push<size_t>(header);
  bp.push<uint16_t>(v0);
  bp.push<uint8_t>(v1);
  bp.push<uint16_t>(v1x);
  bp.finalize();
  vector<uint8_t> out;
  BitPacker bc;
  bc.put_compressed(&compressed[0],compressed.size());
  bc.flush();
  out = bc.get_data();
  cout << "ByteCompress: HME ratio: " << compressed.size()/(float)out.size() << endl;
  
}


void BitPacker::put_bit(unsigned b) {
  cur_data += b << cur_data_bits;
  ++cur_data_bits;
  x_put();
}

// void BitPacker::put_bits(u32 v, unsigned bits) {
//   u32 h = 0;
//   assert(bits <= 32);
//   asm("shldl %%cl, %[v], %[h]; shll %%cl,%[v]"
//       : [v]"+r"(v), [h]"+r"(h) : [c]"c"(cur_data_bits));
//   cur_data |= v;
//   cur_data_bits += bits;
//   if ( cur_data_bits >= 32 ) {
//     unsigned more = cur_data_bits-32;
//     cur_data_bits = 32;
//     x_put();
//     cur_data |= h << cur_data_bits;
//     cur_data_bits += more;
//   }
//   x_put();
// }
 
void BitPacker::flush() {
  assert(!flushed);
  put_u32(0);  // as padding for peeks unpacking
  if ( cur_data_bits ) {
    assert(cur_data_bits < 8);
    cur_data_bits = 8;
    x_put();
  }
  flushed = true;
}

void BitPacker::put_u8(uint8_t v) {
  put_bits(v, 8);
}

void BitPacker::put_u16(uint16_t v) {
  //  unsigned b = v >= 0x100;
  //  put_bit(b);
  //  put_bits(v, (b+1)*8);
  put_bits(v, 16);
}

void BitPacker::put_u32(uint32_t v) {
  // unsigned b = v >= 0x10000? 2+(v >= 0x1000000): (v >= 0x100);
  // put_bits(b, 2);
  // put_bits(v, (b+1)*8);
  put_bits(v, 32);
}

int BitUnpacker::get_bit() {
  fill();
  int ret = cur_data&1;
  cur_data >>= 1;
  cur_data_bits -= 1;
  return ret;
}


// u32 BitUnpacker::peek_bits(unsigned bits) {
//   fill();
//   return cur_data&((1u<<bits)-1);
// }

void BitUnpacker::get_compressed(vector<uint8_t>& vv, unsigned size) {
  vv.reserve(size);
  set_end_hack(vv, &*vv.begin()+size);
  get_compressed(&vv[0], size);
}
