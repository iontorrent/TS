/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ErrorMotifs.h
//! @ingroup  VariantCaller
//! @brief    Handles motifs with high systematic error

#include "ErrorMotifs.h"

//-----------------------------------------------------
void TIonMotifSet::add_motif(string motif, const TMetaData &mdata) {

  unsigned hpPos = 0;
  int firstBase = -1;

  while (hpPos < motif.length() && !isdigit(motif.at(hpPos))) {
    if (motif.at(hpPos) != '.' && firstBase == -1)
      firstBase = hpPos;
    hpPos++;
  }
  if (hpPos == 0) {
    cout << "TIonMotifSet: bad motif " << motif << endl;
    return;
  }

  int key = get_key(motif.at(hpPos-1), atoi(&motif.at(hpPos)));
  if (key >= (int)(4*MAX_HP_SIZE))
    return;
  
  int pref_length = hpPos - 1 - firstBase;
  if (pref_length > motif_table.at(key).left_size)
    motif_table.at(key).left_size = pref_length;

  string submotif = (pref_length <= 0 ? "" : motif.substr(firstBase, pref_length));
  submotif += "|";

  int i = hpPos, suff_start = -1, lastBase = -1;
  while (i < (int)motif.length()) {
    if (!isdigit(motif.at(i)) && suff_start == -1) suff_start = i;
    if (motif.at(i) != '.' && suff_start != -1) lastBase = i + 1;
    i++;
  }

  int suff_length = lastBase - suff_start;
  if (suff_length > motif_table.at(key).right_size)
    motif_table.at(key).right_size = suff_length;
  submotif += (suff_length <= 0 ? "" : motif.substr(suff_start, suff_length));

  //cout << key << " " << submotif << " " << mdata->covered_sse << " " << mdata->covered_nonsse << endl;
  add_my_motif(key, submotif, mdata, 0);
  isempty = false;
}


//-----------------------------------------------------
unsigned TIonMotifSet::make_hashkey(string motif) const {
  unsigned hashKey = 1;
  for (unsigned i = 0;i < motif.length(); i++) {
    hashKey <<= 2;
    switch (motif[i]) {
      case 'C': hashKey|=1; break;
      case 'G': hashKey|=2; break;
      case 'T': hashKey|=3; break;
    }
  }
  return hashKey;
}


//-----------------------------------------------------
void TIonMotifSet::add_my_motif(short key, string motif, const TMetaData &mdata, unsigned pos) {
  for (unsigned i = pos; i < motif.length(); i++)
    if (motif.at(i) == '.') {
      motif[i] = 'A'; add_my_motif(key, motif, mdata, pos + 1);
      motif[i] = 'C'; add_my_motif(key, motif, mdata, pos + 1);
      motif[i] = 'G'; add_my_motif(key, motif, mdata, pos + 1);
      motif[i] = 'T'; add_my_motif(key, motif, mdata, pos + 1);
      return;
    }
  //cout << key << "-" << motif << " " << mdata->covered_sse << " " << mdata->covered_nonsse << endl;
  motif_table.at(key).add(make_hashkey(motif), mdata);
}


//-----------------------------------------------------
short TIonMotifSet::get_key(char hpChar, int hpSize) const {
  hpSize = 4 * (hpSize - 1);
  switch (hpChar) {
    case 'T': hpSize++;
    case 'G': hpSize++;
    case 'C': hpSize++;
  }
  return (short)hpSize;
}


//-----------------------------------------------------
void TIonMotifSet::load_from_file(const char * fname) {

  ifstream infile;
  infile.open(fname);
  string line;

  if (!infile) {
    cerr << "Unable to read " <<  fname << endl;
    exit(1);
  }

  while (getline(infile, line)) {
    if (line.length() > 0 && line[0] != '#') {
      string::size_type pos = line.find("\t");
      if (pos != string::npos) {
        string motif = line.substr(0, pos);
        int val[2];
        int i = 0;
        while (i < 2 && pos != string::npos) {
          val[i++] = atoi(line.substr(pos + 1).c_str());
          pos = line.find("\t", pos + 1);
        }
        if (i < 2) cerr << "skipping corrupted motif record " <<  line << endl;
        else {
          TMetaData mdata(val[0], val[1]);
          add_motif(motif, mdata);
        }
      }
    }
  }

  infile.close();

}
//-----------------------------------------------------
float TIonMotifSet::get_sse_probability(string context, unsigned err_base_pos) const {

  if (err_base_pos >= context.length()) {
    cout << "TIonMotifSet::get_sse_probability() position is out of context. context='" << context << "' pos=" << err_base_pos << endl;
    return 0;
  }

  int i = err_base_pos - 1, j = err_base_pos + 1;
  while (i >= 0 && context.at(i) == context.at(err_base_pos))
    i--;
  while (j < (int)context.length() && context.at(j) == context.at(err_base_pos))
    j++;

  int hpLen = (err_base_pos - 1) - i + 1 + j - (err_base_pos + 1);
  int key = get_key(context.at(err_base_pos), hpLen);
  float weight = 0;
  
  if(key>=(int)(4*MAX_HP_SIZE))
    return weight;
  
  if (!(motif_table.at(key).isEmpty())) {
    unsigned left_size = motif_table.at(key).left_size;
    unsigned right_size = motif_table.at(key).right_size;

    string prefix = i < 0 ? "" : context.substr(0, i + 1);
    string suffix = j >= (int)context.length() ? "" : context.substr(j);

    if (prefix.length() > left_size)
      prefix = prefix.substr(prefix.length() - left_size);
    if (suffix.length() > right_size)
      suffix = suffix.substr(0, right_size);
    //cout << prefix << " " << (char) context[err_base_pos] << hpLen << " " << suffix << endl;

    TMetaData md(0, 0);
    for (i = 0;i <= (int)prefix.length();i++)
      for (j = 0;j <= (int)suffix.length();j++)
        if (motif_table.at(key).has_hashkey(make_hashkey((i < (int)prefix.length() ? prefix.substr(i) : "") + "|" + (j > 0 ? suffix.substr(0, j) : "")), md)) {
          float tmp_weight = md.calculate_probability();
          if (tmp_weight > weight)
            weight = tmp_weight;
        }
  }
  return weight;
}
//-------------------------------------------------------

int test_main(const char * fname) {
  TIonMotifSet mf;



  string context = "GCTATGCCCGT"; //any size, but the best if prefix.lenght >= 7, and suffix.lenght >= 7
  int err_base_position = 6; // zero based position of the query base (in this case first C in CCC
  double pErr = mf.get_sse_probability(context, err_base_position);
  cout << "context=" << context << " pos=" << err_base_position << " pErr=" << pErr << endl;

  err_base_position = 7; // zero based position of the query base (in this case second C in CCC
  pErr = mf.get_sse_probability(context, err_base_position);
  cout << "context=" << context << " pos=" << err_base_position << " pErr=" << pErr << endl;

  context = "TGGCACGT";
  err_base_position = 1; // zero based position of the query base (in this case first G in GG
  pErr = mf.get_sse_probability(context, err_base_position);
  cout << "context=" << context << " pos=" << err_base_position << " pErr=" << pErr << endl;

// not in motif set
  context = "TGGCACGT";
  err_base_position = 3; // zero based position of the query base (in this case first C
  pErr = mf.get_sse_probability(context, err_base_position);
  cout << "context=" << context << " pos=" << err_base_position << " pErr=" << pErr << endl;


  return 0;
}
