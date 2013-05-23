/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
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
#include "SampleStats.h"
#include "SvdDatCompress.h"
#include "BitHandler.h"
#include "HuffmanEncode.h"
#define TIME_LIMIT (60)

using namespace arma;
//#include <gmp.h>


using namespace std; using namespace __gnu_cxx; typedef vector<string> VS;
typedef long long i64;typedef unsigned long long u64;typedef unsigned char uu8;
typedef unsigned uu32; typedef unsigned short uu16; typedef signed char byte;
//typedef vector<int> VI; typedef vector<VI> VVI; typedef vector<double> VD;
typedef vector<uint16_t> V16; typedef vector<int> VI; typedef vector<VI> VVI; typedef vector<double> VD;

////////////////////////// macros and typedefs ///////////////////////////////
/*
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

#if !defined(HOME_RUN)
# define NDEBUG
# define SPEED 1
# define MAX_TIME TIME_LIMIT
# define CLOCK_TIME (1/2.000e9)
# ifdef TR
#  undef TR
# endif
# ifdef assert
#  undef assert
# endif
# ifdef verify
#  undef verify
# endif
# ifdef INLINE
#  undef INLINE
# endif
# define SHARED
# define A(p,s) ""
#else
# define SHARED extern "C"
#endif
#ifndef TR
# define TR(v) NOP()
# define xTR(v) NOP()
# define NOTRACE
#endif
#ifndef assert
# define assert(v) NOP()
#endif
#ifndef verify
# define verify(v) ((void)(v))
#endif
#ifndef INLINE
# define INLINE inline
#endif
#ifndef OUT
# ifdef DEBUG_OUTPUT
#  define OUT(v) xOUT(v)
# else
#  define OUT(v) NOP()
# endif
#endif
#ifdef CLOCK
# define TIMES ' '<<SNS::Time(2)<<' '<<SNS::Clock()<<' '
#else
# define TIMES ' '<<SNS::Time(2)<<' '
#endif
#undef TIMES
#define TIMES ""
#ifdef RESULT_STATS
#  define ROUT(v) do { xOUT(TIMES<<v); } while (0)
#else
#  define ROUT(v) NOP()
#endif

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

struct BitUnpacker
{
  unsigned cur_data, cur_data_bits;
  const uint8_t* ptr;
  //const int* ptr;
  BitUnpacker(const vector<uint8_t>& data) : cur_data(0), cur_data_bits(0), ptr(&data[0]) { }
  void x_get() {
    cur_data |= *ptr++ << cur_data_bits;
    cur_data_bits += 8;
  }
  void fill() { while ( cur_data_bits <= 24 ) x_get(); }
  int get_bit() {
    fill();
    int ret = cur_data&1;
    cur_data >>= 1;
    cur_data_bits -= 1;
    return ret;
  }
  uint32_t get_bits(unsigned bits) {
    assert(bits <= 32);
    if ( !bits ) return 0;
    fill();
    if ( bits > cur_data_bits ) {
      uint32_t ret = cur_data;
      unsigned got = cur_data_bits;
      cur_data = 0;
      cur_data_bits = 0;
      fill();
      ret |= (cur_data<<got)&((1u<<bits)-1);
      cur_data >>= bits-got;
      cur_data_bits -= bits-got;
      return ret;
    }
    uint32_t ret = cur_data&((1u<<bits)-1);
    cur_data >>= bits;
    cur_data_bits -= bits;
    return ret;
  }
  uint32_t peek_bits(unsigned bits) {
    fill();
    return cur_data&((1u<<bits)-1);
  }
  unsigned peek_size() const {
    return cur_data_bits;
  }
  void skip_bits(unsigned bits) {
    assert(bits <= cur_data_bits);
    cur_data >>= bits;
    cur_data_bits -= bits;
  }
  uint8_t get_u8() {
    return get_bits(8);
  }
  uint16_t get_u16() {
    return get_bits((get_bit()+1)*8);
  }
  uint32_t get_u32() {
    return get_bits((get_bits(2)+1)*8);
  }
  void get_compressed(uint8_t* vv, unsigned size);
  void get_compressed(vector<uint8_t>& vv, unsigned size) {
    vv.reserve(size);
    set_end_hack(vv, &*vv.begin()+size);
    get_compressed(&vv[0], size);
  }
};

struct BitPacker
{
    vector<uint8_t> &data;
    unsigned cur_data, cur_data_bits;
    BitPacker(std::vector<uint8_t> &compressed) : data(compressed), cur_data(0), cur_data_bits(0) { }
    unsigned size() const { return (data.size()*8+cur_data_bits); }
    void x_put() {
        while ( cur_data_bits >= 8 ) {
            data.push_back(cur_data&255);
            //add_hack(data, int(cur_data&255));
            cur_data >>= 8;
            cur_data_bits -= 8;
        }
    }
    void put_bit(unsigned b) {
        cur_data += b << cur_data_bits;
        ++cur_data_bits;
        x_put();
    }
    void put_bits(uint32_t v, unsigned bits) {
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
    void flush() {
        if ( cur_data_bits ) {
            assert(cur_data_bits < 8);
            cur_data_bits = 8;
            x_put();
        }
    }
    void put_uint8_t(uint8_t v) {
        put_bits(v, 8);
    }
    void put_u16(uint16_t v) {
        unsigned b = v >= 0x100;
        put_bit(b);
        put_bits(v, (b+1)*8);
    }
    void put_u32(uint32_t v) {
        unsigned b = v >= 0x10000? 2+(v >= 0x1000000): (v >= 0x100);
        put_bits(b, 2);
        put_bits(v, (b+1)*8);
    }
    void put_compressed(const uint8_t vv[], unsigned size);
    void put_compressed(const vector<uint8_t>& vv) {
        put_compressed(&vv[0], vv.size());
    }
};

struct Node {
  Node() { next0 = NULL; }
  Node* next0;
  union {
    Node* next1;
    unsigned value;
  };
};

struct Encoding {
  Encoding() { bit_count = 0; bits = 0; }
  unsigned bit_count;
  unsigned bits;
};

class Huffman
{
public:
  Huffman() : bit_size(0), node_curr(0), prefix(0) { 
    Init();
  }

  void Init() {
    Node n;
    for (int i = 0; i < (2<< 8); i++) {
      node_pool[i] = n;
    }
    Encoding encode;
    for (int i = 0; i < (1<< 8); i++) {
      encoding[i] = encode;
    }
  }

  Huffman(const uint8_t* vv, size_t size) : bit_size(8), node_curr(0), prefix(0) {
    fill_n(cnt, 1<<bit_size, 0);
    unsigned max_v0 = 0;
    for ( ; size >= 2; size -= 2, vv += 2 ) {
      unsigned v0 = vv[0];
      unsigned v1 = vv[1];
      ++cnt[v0];
      if ( v0 > max_v0 ) max_v0 = v0;
      ++cnt[v1];
      if ( v1 > max_v0 ) max_v0 = v1;
    }
    forN ( i, size ) {
      unsigned v0 = vv[i];
      ++cnt[v0];
      if ( v0 > max_v0 ) max_v0 = v0;
    }
    while ( bit_size > 1 && !(max_v0>>(bit_size-1)) )
      --bit_size;
    make_tree();
    make_encoding(root, 0, 0);
  }
  ~Huffman() { forIter ( i, prefixes ) delete[] *i; }
	
  unsigned bit_size;
	
  Node root;
	
  unsigned cnt[1<<8];
  Node node_pool[2<<8];
	
  Node* node_curr;
  Node* alloc_node() { return node_curr++; }

  Encoding encoding[1<<8];
  void make_tree();
  void make_encoding(const Node& node, unsigned bits, unsigned bit_count);
  void put_tree(const Node& node, BitPacker& bc) const;
  void encode(BitPacker& bc) {
    bc.put_bits(bit_size-1, 4);
    put_tree(root, bc);
  }
    
  void put(unsigned v, BitPacker& bc) const {
    assert(v < (1<<bit_size));
    const Encoding& e = encoding[v];
    bc.put_bits(e.bits, e.bit_count);
  }
    
  static const unsigned prefix_bits = 10;
  struct Prefix {
    unsigned bit_count;
    union {
      unsigned value;
      Prefix* prefix;
    };
  };
	
  Prefix* prefix;
  vector<Prefix*> prefixes;
	
  void make_prefix(const Node& node, unsigned bits, unsigned bit_count, Prefix* prefix);
	
  void get_tree(Node& node, BitUnpacker& bc, unsigned bit_count);
	
  void decode(BitUnpacker& bc) {
    bit_size = bc.get_bits(4)+1;
    node_curr = node_pool;
    get_tree(root, bc, 0);
    if ( root.next0 ) {
      prefix = new Prefix[1<<prefix_bits];
      prefixes.push_back(prefix);
      make_prefix(root, 0, 0, prefix);
    }
  }
    
  static unsigned get(const Node& node, BitUnpacker& bc) {
    if ( node.next0 ) {
      return get(*(bc.get_bit()? node.next1: node.next0), bc);
    }
    else {
      return node.value;
    }
  }
  unsigned get(BitUnpacker& bc) const {
    if ( !root.next0 ) return root.value;
    const Prefix* prefix = this->prefix;
    for ( ;; ) {
      unsigned peek = bc.peek_bits(prefix_bits);
      unsigned peek_size = bc.peek_size();
      assert(peek < (1<<prefix_bits));
      const Prefix* p = &prefix[peek];
      if ( p->bit_count > prefix_bits ) {
        if ( peek_size >= prefix_bits ) {
          bc.skip_bits(prefix_bits);
          prefix = p->prefix;
          continue;
        }
      }
      else {
        if ( peek_size >= p->bit_count ) {
          bc.skip_bits(p->bit_count);
          return p->value;
        }
      }
      //            bc.fill();
      unsigned more = prefix_bits - peek_size;
      peek |= bc.peek_bits(more)<<peek_size;
      p = &prefix[peek];
      if ( p->bit_count > prefix_bits ) {
        bc.skip_bits(more);
        prefix = p->prefix;
        continue;
      }
      else {
        bc.skip_bits(p->bit_count - peek_size);
        return p->value;
      }
    }
  }
};

void Huffman::make_tree()
{
    unsigned n = 1<<bit_size;
    pair<int, Node*> nn[1<<16];
    unsigned m = 0;
    node_curr = node_pool;
    forN ( v, n ) {
        int c = cnt[v];
        if ( c ) {
            nn[m].first = -c;
            nn[m].second = alloc_node();
            nn[m].second->next0 = 0;
            nn[m].second->value = v;
            ++m;
        }
        else {
            encoding[v].bit_count = ~0u;
        }
    }
    make_heap(nn, nn+m);
    while ( m > 1 ) {
        pop_heap(nn, nn + m--);
        pop_heap(nn, nn + m);
        nn[m-1].first += nn[m].first;
        Node* n = alloc_node();
        n->next0 = nn[m-1].second;
        n->next1 = nn[m].second;
        nn[m-1].second = n;
        push_heap(nn, nn + m);
    }
    root = *nn[0].second;
}
void Huffman::make_encoding(const Node& node, unsigned bits, unsigned bit_count)
{
    if ( node.next0 ) {
        make_encoding(*node.next0, bits  , bit_count+1);
        make_encoding(*node.next1, bits|(1<<bit_count), bit_count+1);
    }
    else {
        assert(bit_count <= 32);
        assert(node.value < (1<<bit_size));
        encoding[node.value].bits = bits;
        encoding[node.value].bit_count = bit_count;
    }
}
void Huffman::make_prefix(const Node& node, unsigned bits, unsigned bit_count,
                          Prefix* prefix)
{
    if ( node.next0 ) {
        if ( bit_count == prefix_bits ) {
            assert(bits < (1<<prefix_bits));
            prefix[bits].bit_count = prefix_bits+1;
            prefix[bits].prefix = new Prefix[1<<prefix_bits];
            prefixes.push_back(prefix[bits].prefix);
            make_prefix(node, 0, 0, prefix[bits].prefix);
        }
        else {
            make_prefix(*node.next0, bits  , bit_count+1, prefix);
            make_prefix(*node.next1, bits|(1<<bit_count), bit_count+1, prefix);
        }
    }
    else {
        for ( ; bits < (1<<prefix_bits); bits += 1<<bit_count ) {
            prefix[bits].bit_count = bit_count;
            prefix[bits].value = node.value;
        }
    }
}
void Huffman::put_tree(const Node& node, BitPacker& bc) const
{
    unsigned more = node.next0 != 0;
    bc.put_bit(more);
    if ( more ) {
        put_tree(*node.next0, bc);
        put_tree(*node.next1, bc);
    }
    else {
        bc.put_bits(node.value, bit_size);
    }
}
void Huffman::get_tree(Node& node, BitUnpacker& bc, unsigned bit_count)
{
    if ( bc.get_bit() ) {
        node.next0 = alloc_node();
        get_tree(*node.next0, bc, bit_count+1);
        node.next1 = alloc_node();
        get_tree(*node.next1, bc, bit_count+1);
    }
    else {
        node.next0 = 0;
        node.value = bc.get_bits(bit_size);
        assert(node.value < (1<<bit_size));
    }
}

inline void BitPacker::put_compressed(const uint8_t vv[], unsigned size)
{
    if ( !size ) return;
    Huffman h(vv, size);
    h.encode(*this);
    forN ( i, size ) h.put(vv[i], *this);
}
inline void BitUnpacker::get_compressed(uint8_t* vv, unsigned size) {
    if ( !size ) return;
    Huffman h;
    h.decode(*this);
    forN ( i, size ) vv[i] = h.get(*this);
}

*/
void SvdDatCompress::Compress(TraceChunk &chunk, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  mCompressed.clear(); 
  Y.set_size(chunk.mHeight * chunk.mWidth, chunk.mDepth);
  // copy data in, column order major and subtract off mean
  vector<int16_t> means(chunk.mHeight * chunk.mWidth);
  for(size_t i = 0; i < chunk.mData.size(); i++) { Y[i] = chunk.mData[i]; }
  for (size_t row = 0; row < Y.n_rows; row++) {
    double m = 0;
    for (size_t col = 0; col < Y.n_cols; col++) {
      m += Y(row,col);
    }
    m /= Y.n_cols;
    means[row] = m;
    for (size_t col = 0; col < Y.n_cols; col++) {
      Y(row,col) = round(Y(row,col) - m);
    }
  }
  // Calculate eigen vectors
  Cov = Y.t() * Y;
  eig_sym(EVal, EVec, Cov);
  X.set_size(chunk.mDepth, mNumEvec);
  // Copy best N eigen vectors as our basis vectors
  int count = 0;
  for(size_t v = Cov.n_rows - 1; v >= Cov.n_rows - mNumEvec; v--) {
    copy(EVec.begin_col(v), EVec.end_col(v), X.begin_col(count++));
  }
  // Calculate our best projection of data onto eigen vectors
  Y = Y.t();

  if(!solve(B,X,Y)) {
    ION_ABORT("Couldn't solve matrix.");
  }

  // Store our values for number of eigenvectors and precision
  BitPacker bc;
  bc.put_u16((short)mNumEvec);
  bc.put_u32((int)mPrecision);

  // Pull off the mean for coefficients in B
  vector< SampleStats<double> > stats(B.n_rows);
  vector<float> coeffMeans(B.n_rows);
  coeffMeans.resize(B.n_rows);
  for (size_t row = 0; row < B.n_rows; row++) {
    float m = 0;
    for (size_t col = 0; col < B.n_cols; col++) {
      //      m += B(row,col);
      stats[row].AddValue(B(row,col));
    }
    m = m / B.n_cols;
    m = stats[row].GetMean();
    coeffMeans[row] = m;
    for (size_t col = 0; col < B.n_cols; col++) {
      B(row,col) = B(row,col) - m; // @todo - keep this mean
    }
  }
  uint8_t *coeffmean = (uint8_t *) &coeffMeans[0];
  bc.put_compressed(coeffmean, coeffMeans.size() * sizeof(float));
  // Store our means
  uint8_t *mean = (uint8_t *) &means[0];
  bc.put_compressed(mean, means.size() * sizeof(int16_t));

  // Truncate values in B
  BB.set_size(B.n_rows, B.n_cols);
  for (size_t i = 0; i < B.n_rows * B.n_cols; i++) {
    BB[i] = round(B[i] * mPrecision);
  }

  uint8_t *mem = (uint8_t*)X.memptr();
  bc.put_compressed(mem, X.n_elem * sizeof(float));
  // Store our truncated coefficients
  uint8_t *vals = (uint8_t*)BB.memptr();
  unsigned size = BB.n_rows * BB.n_cols * sizeof(short);
  bc.put_compressed(vals, size);
  // pad out to a multiple of size 4 for valgrind errors reading back in
  // while(bc.size() % 4 != 0) {
  //   bc.put_bit(0);
  // }
  // bc.put_u32(0);
  bc.flush();
  //  (*compressed) = (int8_t *) malloc(sizeof(int8_t) * bc.get_data().size()); // use malloc for compatiblity with hdf5. // new int8_t [mCompressed.size()];
  if (bc.get_data().size() > *maxsize) { ReallocBuffer(bc.get_data().size(), compressed, maxsize); }
  *outsize = bc.get_data().size();
  memcpy(*compressed, &bc.data[0], *outsize);
}

void SvdDatCompress::Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
  mCompressed.resize(size);
  memcpy(&mCompressed[0], compressed, size);
  BitUnpacker bc(mCompressed);
  int numBasis = bc.get_u16();
  float precision = bc.get_u32();

  vector<float> coeffMeans(numBasis,0);
  uint8_t *coeffmean = (uint8_t *) &coeffMeans[0];
  bc.get_compressed(coeffmean, coeffMeans.size() * sizeof(float));

  vector<int16_t> means(chunk.mHeight * chunk.mWidth);
  uint8_t *mean = (uint8_t *) &means[0];
  bc.get_compressed(mean, means.size() * sizeof(int16_t));
  
  X.set_size(chunk.mDepth, numBasis);
  uint8_t *mem = (uint8_t*)X.memptr();
  bc.get_compressed(mem, X.n_elem * sizeof(float));


  BB.set_size(numBasis, chunk.mHeight * chunk.mWidth);
  B.set_size(BB.n_rows, BB.n_cols);
  uint8_t *vals = (uint8_t*)BB.memptr();
  bc.get_compressed(vals, BB.n_elem * sizeof(short));
  //  for (size_t i = 0; i < BB.n_elem; i++) {
  for (size_t row = 0; row < BB.n_rows; row++) {
    for (size_t col = 0; col < BB.n_cols; col++) {
      B(row,col) = (BB(row,col) / (float)precision) + coeffMeans[row];
    }
  }
  Y = X * B;

  size_t count = 0;
  chunk.mData.resize(chunk.mHeight * chunk.mWidth * chunk.mDepth);
  size_t rowEnd = chunk.mRowStart + chunk.mHeight;
  size_t colEnd = chunk.mColStart + chunk.mWidth;
  size_t frameEnd = chunk.mFrameStart + chunk.mDepth;
  int well = 0;
  for (size_t row = chunk.mRowStart; row < rowEnd; row++) {
    for (size_t col = chunk.mColStart; col < colEnd; col++) {
      for (size_t frame = chunk.mFrameStart; frame < frameEnd; frame++) {
        chunk.At(row, col, frame) = round(Y[count++] + means[well]);
      }
      well++;
    }
  }
}
  
