/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "HuffmanEncode.h"
typedef vector<string> VS;
// typedef long long i64;
// typedef unsigned long long u64;
// //typedef unsigned char u8;
// typedef unsigned char uu8;
// typedef unsigned uu32;
// typedef unsigned short uu16;
// typedef signed char byte;
// //typedef unsigned u32;
// //typedef unsigned short u16;
// typedef signed char byte;
// //typedef vector<int> VI; typedef vector<VI> VVI; typedef vector<double> VD;
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

void Huffman::Init() {
  Node n;
  for (int i = 0; i < (2<< 8); i++) {
    node_pool[i] = n;
  }
  Encoding encode;
  for (int i = 0; i < (1<< 8); i++) {
    encoding[i] = encode;
  }
}

Huffman::Huffman(const uint8_t* vv, size_t size) : bit_size(8), node_curr(0), prefix(0) {
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

Huffman::~Huffman() { 
  //forIter ( i, prefixes ) delete[] *i; 
  for(vector<Prefix*>::iterator i=prefixes.begin();i!=prefixes.end();++i)
      delete[] *i;
}
	
	
void Huffman::encode(BitPacker& bc) {
  if (bit_size > 16) {
    cout << "Error coming..." << endl;
  }
  bc.put_bits(bit_size-1, 4);
  put_tree(root, bc);
}
    
/*void Huffman::decode(BitUnpacker& bc) {
  bit_size = bc.get_bits(4)+1;
  node_curr = node_pool;
  get_tree(root, bc, 0);
  if ( root.next0 ) {
    prefixes.push_back(new Prefix[1<<prefix_bits]);
    make_prefix(root, 0, 0, prefixes.back());
  }
  }
*/

void Huffman::decode(BitUnpacker& bc) {
  bit_size = bc.get_bits(4)+1;
  node_curr = node_pool;
  get_tree(root, bc, 0);
  if ( root.next0 ) {
    prefixes.push_back(new Prefix[1<<prefix_bits]);
    prefix = prefixes.back();
    make_prefix(root, 0, 0, prefix);
  }
}

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
    assert(node.value < (1u<<bit_size));
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
      prefixes.push_back(new Prefix[1<<prefix_bits]);
      prefix[bits].prefix = prefixes.back();
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
    assert(node.value < (1u<<bit_size));
  }
}

// unsigned Huffman::get(BitUnpacker& bc) const {
//   if ( !root.next0 ) return root.value;
//   const Prefix* prefix = this->prefix;
//   for ( ;; ) {
//     unsigned peek = bc.peek_bits(prefix_bits);
//     unsigned peek_size = bc.peek_size();
//     const Prefix* p = &prefix[peek];
//     if ( p->bit_count > prefix_bits ) {
//       if ( peek_size >= prefix_bits ) {
// 	bc.skip_bits(prefix_bits);
// 	prefix = p->prefix;
// 	continue;
//       }
//     }
//     else {
//       if ( peek_size >= p->bit_count ) {
// 	bc.skip_bits(p->bit_count);
// 	return p->value;
//       }
//     }
//     //            bc.fill();
//     unsigned more = prefix_bits - peek_size;
//     peek |= bc.peek_bits(more)<<peek_size;
//     p = &prefix[peek];
//     if ( p->bit_count > prefix_bits ) {
//       bc.skip_bits(more);
//       prefix = p->prefix;
//       continue;
//     }
//     else {
//       bc.skip_bits(p->bit_count - peek_size);
//       return p->value;
//     }
//   }
// }
