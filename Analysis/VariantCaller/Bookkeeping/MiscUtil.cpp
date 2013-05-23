/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     MiscUtil.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#include "MiscUtil.h"

void Tokenize(const string& str,
              vector<string>& tokens,
              const string& delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos     =  str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

// --------------------------------------------------------------
// Ensemble of functions to reverse complement a sequence

char NucComplement (char nuc)
{
  switch(nuc) {
    case ('A') : return 'T';
    case ('C') : return 'G';
    case ('G') : return 'C';
    case ('T') : return 'A';
    case ('a') : return 't';
    case ('c') : return 'g';
    case ('g') : return 'c';
    case ('t') : return 'a';

    default:  return nuc; // e.g. 'N' and '-' handled by default
  }
}

void RevComplementInPlace(string& seq) {

  char c;
  int forward_idx = 0;
  int backward_idx = seq.size()-1;
  while (forward_idx < backward_idx) {
    c = seq[forward_idx];
    seq[forward_idx]  = NucComplement(seq[backward_idx]);
    seq[backward_idx] = NucComplement(c);
    forward_idx++;
    backward_idx--;
  }
  if (forward_idx == backward_idx)
    seq[forward_idx] = NucComplement(seq[forward_idx]);
}

void reverseComplement(string s, char * x) {
  x[s.length()] = '\0';
  for (int i=s.length()-1;i>=0;i--)
    x[s.length()-1-i] = NucComplement(s[i]);
}

void ReverseComplement(string &seqContext, string &reverse_complement) {
  reverse_complement = seqContext;
  for (unsigned i=0;i<seqContext.length(); i++)
    reverse_complement[reverse_complement.length()-1-i] = NucComplement(seqContext[i]);
}

// --------------------------------------------------------------------

void deleteArray(float *** darray, int HT, int WD) {

  for (int i = 0; i < HT; i++) {
    delete [](*darray)[i];

  }
  delete [] *darray;

}

void allocateArray(float *** array, int HT, int WD) {
  *array = new float*[HT];
  for (int i = 0; i < HT; i++)
    (*array)[i] = new float[WD];

  //initialize the array once allocated
  for (int i = 0; i < HT; i++)
    for (int j = 0; j < WD; j++)
      (*array)[i][j] = 0;
}


std::string intToString(int x, int width) {
  std::string r;
  std::stringstream s;
  s << x;
  r = s.str();
  for (int i = r.length(); i <= width; i++) {
    r += " ";
  }

  return r;
}

void getBaseName(const char *path, char *basename) {
  std::string strPath((const char*)path);
  int pos = strPath.rfind(".");
  if (pos != (int)std::string::npos) {
    strcpy(basename, (strPath.substr(0, pos)).c_str());
  } else {
    strcpy(basename, path);

  }
}


