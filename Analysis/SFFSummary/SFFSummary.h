/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFFSUMMARY_H
#define SFFSUMMARY_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cmath>

#include "sff.h"
#include "sff_file.h"

using namespace std;

class SFFSummary {
  public:
    SFFSummary();
    void summarizeFromSffFile(string sffFile, vector <uint16_t> &_qual, vector <unsigned int> &_readLength, vector <unsigned int> &_minReadLength, bool _keepPerReadData);
    void writeTSV(std::ostream &out);
    void writePrettyText(std::ostream &out);
    void writePerReadData(std::ostream &out);
    void setReportPredictedQlen(bool b) { reportPredictedQlen = b; };

  private:
    void qualToErrLength(vector <uint16_t> &qScore, vector <double> &errThreshold, vector <unsigned int> &readLen);
    void qualToErrLength(vector <uint16_t> &qScore, double           errThreshold, unsigned int &         readLen);
    void phredToErr(uint16_t           qScore, double          &errorRate);
    void phredToErr(vector <uint16_t> &qScore, vector <double> &errorRate);
    void summaryStatInit();
    void summaryStatUpdate(vector <uint16_t> qScore, string rName);
    void AddElementSNR (uint16_t *corValues, const string& flowOrder);


    // variables describing what we aim to collect
    vector <uint16_t>                                  qual;               // Phred quality levels for which we collect stats
    vector <double>                                    errThreshold;       // Error probabilities corresponding to qual
    vector <unsigned int>                              readLength;         // Read lengths for which we collect stats

    // variables related to filtering out reads that fail to meet a minimum standard
    vector <unsigned int>                              minReadLength;      // Minimum length to use a read - one per quality level

    // variables holding the summary statistics that we collect
    unsigned int                                       nReads;              // Total number of reads
    map < uint16_t, unsigned int >                     nReadsByQual;        // Number of reads exceeding the minimum length
    map < uint16_t, map <unsigned int, unsigned int> > nReadsByQualLength;  // Number of reads exceeding a given length at a qual threhsold
    map < uint16_t, unsigned int >                     nReadsByLength;      // Number of reads exceeding length tresholds, regardless of quality
    map < uint16_t, unsigned int >                     nBasesByQual;        // Number of bases in reads exceeding the min length at a qual threshold
    map < uint16_t, unsigned int >                     maxLengthByQual;     // Max read length at a qual threhsold
    map < uint16_t, double >                           meanLengthByQual;    // Mean read length among all reads exceeding min lenght 
    bool                                               keepPerReadData;     // Should per-read data be retained when summarizing?
    bool                                               reportPredictedQlen; // Should per-read predictions of qlen be included when writing per-read data?
    vector < string >                                  readName;            // Read names
    vector < vector < unsigned int > >                 perReadQualLength;   // Read length at a given qual threshold
    vector < unsigned int >                            perReadLength;       // Untrimmed read length

    // variables used in key SNR calculation
    int count;
    double zeromerFirstMoment[8];
    double zeromerSecondMoment[8];
    double onemerFirstMoment[8];
    double onemerSecondMoment[8];

};

#endif // SFFSUMMARY_H
