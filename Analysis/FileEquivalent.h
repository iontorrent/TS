/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FILEEQUIVALENT_H
#define FILEEQUIVALENT_H
#include <vector>
#include <string>
#include "NumericalComparison.h"
#include "SFFWrapper.h"

class SffComparison {
public:
  string name;
  int same;
  int different;
  int total;
  int missing;

  double correlation;

  SffComparison(const std::string &_name,
                int _same, int _different, int _total,
                int _missing, double _correlation) {
      name = _name;
      same = _same;
      different = _different;
      total = _total;
      missing = _missing;
      correlation = _correlation;
  }

  SffComparison() {
      same = 0;
      different = 0;
      total = 0;
      missing = 0;
      correlation = -2;
  }

  int GetNumDiff() { return different; }
  double GetCorrelation() { return correlation; }

};

class SFFInfo {
  
 public:

  SFFInfo();

  SFFInfo(const sff_t *info);

  static bool Abort(const std::string &msg);

  static vector<SffComparison> CompareSFF(const std::string &queryFile, const std::string &goldFile, double epsilon,
                                            int &found, int &missing, int &goldOnly);

 private:

  uint16_t	read_header_length;
  uint16_t	name_length;
  uint32_t	number_of_bases;
  uint16_t	clip_qual_left;
  uint16_t	clip_qual_right;
  uint16_t	clip_adapter_right;
  uint16_t	clip_adapter_left;
  std::string name;
  std::vector<uint16_t> flowgram_values; // [NUMBER_OF_FLOWS_PER_READ];
  std::vector<uint8_t> flow_index_per_base; // * number_of_bases;
  std::string bases; // * number_of_bases;
  std::vector<uint8_t> quality_scores; // * number_of_bases;
};

NumericalComparison<double> CompareRawWells(const std::string &queryFile, const std::string &goldFile, 
					    float epsilon, double maxAbsVal);

#endif // FILEEQUIVALENT_H
