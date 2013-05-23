/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <armadillo>
#include <iostream>
#include <vector>
#include "H5File.h"
#include "Mask.h"
#include "Utils.h"
#include "NumericalComparison.h"
 
using namespace std;
using namespace arma;
/**
 * Some stats about mask comparisons
 */
class MaskDiff {
public:

  /** Basic constructor. */
  MaskDiff() {
    Init(MaskNone, "MaskNone");
  }

  /** Constructor for type of mask entry. */
  MaskDiff(uint16_t type, const std::string &n) {
    Init(type,n);
  }

  /** Initialize with a type of mask entry. */
  void Init(uint16_t type, const std::string &n) {
    entry = type;
    diff = 0;
    added = 0;
    deleted = 0;
    seen = 0;
    same = 0;
    name = n;
  }

  /** Compare two entries of mask types. */
  void Compare(uint16_t r1, uint16_t r2) {
    seen++;
    if (r1 == r2) {
      same++;
    }
    else {
      diff++;
      if (r1 >= 1) {
	added++;
      }
      else {
	deleted++;
      }
    }
  }

  void Out(std::ostream &out, float tag_percent=3.0f) {
    char buff[256];
    size_t diff = seen - same;
    char tag = ' ';
    if ((added*100.0/seen + deleted*100.0/seen)/2.0f >= tag_percent) {
      tag = '*';
    }
    snprintf(buff, sizeof(buff), "%-12s\t%8u (%6.2f%%)\t%8u (%6.2f%%)\t%8u (%6.2f%%)%c", name.c_str(), 
	     (uint32_t)diff, (diff*100.0)/seen,
	     (uint32_t)added, (added*100.0)/seen, 
	     (uint32_t)deleted, (deleted*100.0)/seen, tag);
    out << buff << endl;
  }
    

  /** Are the two sets of entries the same? */
  bool IsSame() { return diff == 0; }

  std::string name;
  size_t diff;
  size_t same;
  uint16_t entry;
  size_t added;
  size_t deleted;
  size_t seen;

};

/** Everbody's favorite function. */
int main(int argc, const char *argv[]) {
  if (argc != 5) {
    cout << "Wrong number of arguments.\nusage:\n\tCompareBf mask1.bin mask2.bin sep1.h5 sep2.h5\n";
    exit(1);
  }
  uint16_t mask_types[] = {MaskBead, MaskEmpty, MaskLib, MaskTF, MaskPinned, MaskIgnore};
  const char *mask_names[] = {"MaskBead", "MaskEmpty", "MaskLib", "MaskTF", "MaskPinned", "MaskIgnore"};
  vector<MaskDiff> mask_compare(ArraySize(mask_types));
  for (size_t i = 0; i < mask_compare.size(); i++) {
    mask_compare[i].Init(mask_types[i], mask_names[i]);
  }
  Mask m1, m2;
  m1.SetMask(argv[1]);
  m2.SetMask(argv[2]);
  size_t num_wells = m1.H() * m1.W();
  for (size_t wIx = 0; wIx < num_wells; wIx++) {
    for (size_t cIx = 0; cIx < mask_compare.size(); cIx++) {
      mask_compare[cIx].Compare(m1[wIx] & mask_compare[cIx].entry, m2[wIx] & mask_compare[cIx].entry);
    }
  }

  Mat<float> s1, s2;
  string suffix = ":/separator/summary";
  string path1 = argv[3] + suffix;
  string path2 = argv[4] + suffix;
  const char *col_names[] = {"index","t0","snr","mad","sd","bfmetric","a","c","g","t","peak","flag","goodlive","isref","buffmetric"};
  H5File::ReadMatrix(path1, s1);
  H5File::ReadMatrix(path2, s2);
  vector<NumericalComparison<float> > column_compare(s1.n_cols);
  size_t num_names = ArraySize(col_names);
  for (size_t cIx = 0; cIx < num_names; cIx++) {
    column_compare[cIx].SetName(col_names[cIx]);
  }
  for (size_t cIx = num_names; cIx < column_compare.size(); cIx++) {
    string tmp = "unknown";
    column_compare[cIx].SetName(tmp + ToStr(cIx - num_names));
  }

  for (size_t rIx = 0; rIx < s1.n_rows; rIx++) {
    for (size_t cIx = 0; cIx < s1.n_cols; cIx++) {
      float v1 = s1.at(rIx,cIx);
      float v2 = s2.at(rIx,cIx);
      if (isfinite(v1*v2)) {
	column_compare[cIx].AddPair(v1,v2);
      }
    }
  }
  
  cout << "Separator Comparison" << endl
       << "--------- ----------" << endl;
  for (size_t cIx = 0; cIx < column_compare.size(); cIx++) {
    column_compare[cIx].Out(cout);
  }
  cout << endl;
  cout << "Mask Comparison" << endl
       << "---- ----------" << endl;
  bool same = true;
  for (size_t cIx = 0; cIx < mask_compare.size(); cIx++) {
    same &= mask_compare[cIx].IsSame();
    mask_compare[cIx].Out(cout);
  }
  if (!same) {
    cout << "Differences found." << endl;
    exit(1);
  }
  cout << "Same." << endl;
  return 0;
}
