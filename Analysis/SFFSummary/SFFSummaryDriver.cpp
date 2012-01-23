/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "OptArgs.h"
#include "SFFSummary.h"

using namespace std;

void usage() {
  cout << endl
       << "SFFSummary - Produce summary of read lengths and predicted qualities." << endl
       << endl
       << "usage: " << endl
       << "  SFFSummary [-h -l 50,100,150 -o out.txt] -m 0,21,21 -q 0,17,20 -s in.sff" << endl
       << endl
       << "options: " << endl
       << "  -h,--help              This message" << endl
       << "  -d,--qual-length-file  File to which to write quality lengths" << endl
       << "  -l,--read-length       Read lengths to summarize" << endl
       << "  -m,--min-length        Minimum lengths for a read to be considered" << endl
       << "  -o,--out-file          File to write summary to" << endl
       << "  -p,--report-pred-qlen  Reports predicted qlength to per-read output file" << endl
       << "  -q,--qual              Quality levels (Phred-scale) to summarize" << endl
       << "  -s,--sff-file          Input SFF file" << endl
       << endl;
    exit(1);
}

int main(int argc, const char *argv[]) {

  // Options handling
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  bool help;
  bool reportPredictedQlen;
  string qualLengthFile;
  string sffFile;
  string outFile;
  vector <int> tempMinLength;
  vector <int> tempReadLength;
  vector <int> tempQual;
  opts.GetOption(qualLengthFile,      "",      'd', "qual-length-file");
  opts.GetOption(help,                "false", 'h', "help");
  opts.GetOption(tempReadLength,      "",      'l', "read-length");
  opts.GetOption(tempMinLength,       "",      'm', "min-length");
  opts.GetOption(outFile,             "",      'o', "out-file");
  opts.GetOption(reportPredictedQlen, "false", 'p', "report-pred-qlen");
  opts.GetOption(tempQual,            "",      'q', "qual");
  opts.GetOption(sffFile,             "",      's', "sff-file");
  if (help || sffFile.empty() || tempQual.size()==0 || tempMinLength.size() == 0 ) {
    usage();
  }
  // Confirm lengths option is concordant with quality levels requested
  if ( (tempMinLength.size() > 1) && (tempQual.size() != tempMinLength.size()) ) {
    cerr << "ERROR: -m option should specify either one or same number of values as -q option." << endl << endl;
    usage();
  }
  // Confirm qualities were supplied in ascending order
  for(unsigned int iQual=1; iQual < tempQual.size(); iQual++) {
    if(tempQual[iQual] < tempQual[iQual-1]) {
      cerr << "ERROR: -q option should specify ascending list of quality thresholds." << endl << endl;
      usage();
    }
  }
  opts.CheckNoLeftovers();

  // Recast qualities
  vector <uint16_t> qual;
  for(unsigned int iQual=0; iQual<tempQual.size(); iQual++)
    qual.push_back((uint16_t) tempQual[iQual]);
  // Recast min lengths, and recycle value if only one specified
  vector <unsigned int> minLength(qual.size(),tempMinLength[0]);
  if(tempMinLength.size() > 1) {
    for(unsigned int iLen=0; iLen<tempMinLength.size(); iLen++)
      minLength[iLen] = (unsigned int) tempMinLength[iLen];
  }
  // Recast lengths
  vector <unsigned int> readLength;
  for(unsigned int iLen=0; iLen<tempReadLength.size(); iLen++)
    readLength.push_back((unsigned int) tempReadLength[iLen]);

  if (sffFile != "") {
    bool keepPerReadData = (qualLengthFile != "");

    // Iterate through sff and collect statistics
    SFFSummary sff;
    sff.setReportPredictedQlen(reportPredictedQlen);
    sff.summarizeFromSffFile(sffFile,qual,readLength, minLength, keepPerReadData);

    // Write pretty print results to cout
    sff.writePrettyText(cout);

    // Write TSV results to file
    if(outFile != "") {
      ofstream summary;
      summary.open(outFile.c_str());
      sff.writeTSV(summary);
      summary.close();
    }

    // Write file of pre-read qual lengths, if requested
    if(qualLengthFile != "") {
      ofstream summary;
      summary.open(qualLengthFile.c_str());
      sff.writePerReadData(summary);
      summary.close();
    }

    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }

}
