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

using namespace std;

//-----------------------------------------------------

struct TMetaData
{
 unsigned short covered_sse,
                  covered_nonsse;
 TMetaData(unsigned short cov_sse,unsigned short  cov_nonsse){
   covered_sse = cov_sse;
   covered_nonsse = cov_nonsse;   
 }
 inline bool operator=(const TMetaData & md) 
   {
     covered_sse = md.covered_sse; 
     covered_nonsse = md.covered_nonsse;
     return false; 
   }
 inline double calculate_probability() 
   { 
     return (double)covered_sse/(covered_sse+covered_nonsse);
   }
};

//-----------------------------------------------------
class TIonMotif
{

  map <unsigned, TMetaData> * motif_list;

 public:
 short left_size , right_size;
 TIonMotif(){ left_size = 0; right_size = 0; motif_list = new map <unsigned, TMetaData>();}
 ~TIonMotif(){ delete motif_list; }

 inline void add(unsigned hashKey, TMetaData * mdata){
   try{
       TMetaData * tmp_md = & motif_list->at(hashKey);
       if(tmp_md->calculate_probability()>mdata->calculate_probability()) *tmp_md = *mdata;
      }catch(...){motif_list->insert(make_pair(hashKey,*mdata));}
 };

 inline bool has_hashkey(unsigned hashKey, TMetaData & md){
  try{
       md = motif_list->at(hashKey);
       return true;
      }catch(...){}
  return false;
 };

 inline bool isEmpty(){ return (motif_list->size()==0); }
};

//-----------------------------------------------------
class TIonMotifSet
{
 unsigned MAX_HP_SIZE;
 TIonMotif * motif_table;
 bool isempty;
 void add_motif(short key, string motif, TMetaData * mdata, unsigned pos);
 public:
 TIonMotifSet(){ MAX_HP_SIZE = 20; motif_table = new TIonMotif[4*(MAX_HP_SIZE+1)];isempty=true; }
 ~TIonMotifSet(){ delete [] motif_table; }
 void load_from_file(const char * fname);
 void add_motif(string motif, TMetaData * mdata);
 short get_key(char hpChar, int hpSize);
 unsigned make_hashkey(string motif);
 float get_sse_probability(string context, unsigned err_base_pos);
 inline bool isEmpty(){return isempty;}
};
//-----------------------------------------------------

#endif //ERRORMOTIFS_H
