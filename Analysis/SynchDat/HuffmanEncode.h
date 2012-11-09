/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef HUFFMANENCODE_H
#define HUFFMANENCODE_H

#include "BitHandler.h"

class Node {
 public:
  Node() { next0 = NULL; }
  Node* next0;
  union {
    Node* next1;
    unsigned value;
  };
};

class Encoding {
 public:
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

  void Init();

  Huffman(const uint8_t* vv, size_t size);

  ~Huffman();
	
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
  void encode(BitPacker& bc);
    
  void put(unsigned v, BitPacker& bc) const {
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
	
  void decode(BitUnpacker& bc);
    
  static unsigned get(const Node& node, BitUnpacker& bc) {
    if ( node.next0 ) {
      return get(*(bc.get_bit()? node.next1: node.next0), bc);
    }
    else {
      return node.value;
    }
  }

  //  unsigned get(BitUnpacker& bc) const;
  inline unsigned get(BitUnpacker& bc) const {
    if ( !root.next0 ) return root.value;
    const Prefix* prefix = this->prefix;
    for ( ;; ) {
      unsigned peek = bc.peek_bits(prefix_bits);
      unsigned peek_size = bc.peek_size();
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


#endif // HUFFMANENCODE_H
