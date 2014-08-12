/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     MiscUtil.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef MISCUTIL_H
#define MISCUTIL_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include <cstring>
using namespace std;


char NucComplement (char nuc);
void RevComplementInPlace(string& seq);
void reverseComplement(string s, char *x);
void ReverseComplement(string &seqContext, string &reverse_complement);

std::string intToString(int x, int width);
template<typename T>
std::string convertToString(const T& r) {
    std::ostringstream iss;
    iss << r;
    return iss.str();
}


template<typename T>
class CircluarBuffer {
public:

  void  shiftRight(unsigned int shift) {
    start_idx += shift;
    start_idx = start_idx % (int)buffer.size();
  };

  void  shiftLeft (unsigned int shift) {
    start_idx -= shift;
    start_idx = start_idx % (int)buffer.size();
    while (start_idx < 0)
      start_idx += buffer.size(); // Safety, modolo on negative values not precisely defined.
  };

  void  assign(unsigned int position, T value) { buffer.at(translate_position(position)) = value; };
  T     at(unsigned int position) { return buffer.at(translate_position(position)); };
  T     first() { return buffer.at(start_idx); };
  T     last() { return buffer.at(translate_position(buffer.size()-1)); };

  void print() {
   for (unsigned int idx=0; idx<buffer.size(); idx++)
     cout << buffer.at(translate_position(idx));
   cout << endl;
  };

  CircluarBuffer(unsigned int size) {
    buffer.resize(size, 0);
    start_idx = 0;
  };

private:
  int               start_idx;
  vector<T>  	    buffer;

  unsigned int  translate_position(unsigned int pos_in) {
    unsigned int pos_out = pos_in + start_idx;
    pos_out = pos_out % buffer.size();
    return (pos_out);
  };
};

// ---------------------------------------------------

#endif // MISCUTIL_H
