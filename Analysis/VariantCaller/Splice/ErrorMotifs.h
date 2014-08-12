/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ErrorMotifs.h
//! @ingroup  VariantCaller
//! @brief    Handles motifs with high systematic error


#ifndef ERRORMOTIFS_H
#define ERRORMOTIFS_H

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <cstdlib> 
#include <map>
#include <vector>

using namespace std;

//-----------------------------------------------------

struct TMetaData
{
  unsigned short covered_sse_;
  unsigned short covered_nonsse_;

  TMetaData(unsigned short cov_sse,unsigned short  cov_nonsse) {
    covered_sse_    = cov_sse;
    covered_nonsse_ = cov_nonsse;
  }

  inline bool operator=(const TMetaData & md) {
    covered_sse_ = md.covered_sse_;
    covered_nonsse_ = md.covered_nonsse_;
    return false;
  }

  inline double calculate_probability() const {
    // Divide by zero protection
    double denominator = (double)(covered_sse_ + covered_nonsse_);
    if (denominator > 0)
      return (double)covered_sse_ / denominator;
    else
      return -1.0;
  }
};

//-----------------------------------------------------
class TIonMotif
{

  map <unsigned, TMetaData> motif_list;

  public:
    short left_size;
    short right_size;

    TIonMotif(){ left_size = 0; right_size = 0; };

    inline void add(unsigned hashKey, const TMetaData &mdata) {
      map <unsigned, TMetaData>::iterator itr;
      itr = motif_list.find(hashKey);

      if (itr == motif_list.end())
        motif_list.insert(make_pair(hashKey, mdata));
      else if (itr->second.calculate_probability() > mdata.calculate_probability())
        motif_list.at(hashKey) = mdata;
     };

     inline bool has_hashkey(unsigned hashKey, TMetaData &md) const {
       map <unsigned, TMetaData>::const_iterator itr;
       itr = motif_list.find(hashKey);

       if (itr == motif_list.end())
         return false;
       else {
         md = itr->second;
         return true;
       }
     };

     inline bool isEmpty() const {
       return (motif_list.size()==0);
     };
};

//-----------------------------------------------------
class TIonMotifSet
{
  unsigned MAX_HP_SIZE;
  vector<TIonMotif> motif_table;
  bool isempty;
  void add_my_motif(short key, string motif, const TMetaData &mdata, unsigned pos);

  public:
    TIonMotifSet() {
      MAX_HP_SIZE = 20;
      motif_table.resize(4*(MAX_HP_SIZE+1));
      isempty=true;
    };

    void     load_from_file(const char * fname);
    void     add_motif(string motif, const TMetaData &mdata);
    short    get_key(char hpChar, int hpSize) const;
    unsigned make_hashkey(string motif) const;

    float    get_sse_probability(string context, unsigned err_base_pos) const;

    inline bool isEmpty() { return isempty; };
};
//-----------------------------------------------------

#endif //ERRORMOTIFS_H
