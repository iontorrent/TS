/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "OptArgs.h"
#include "SFFSummary/SFFSummary.h"

using namespace std;

int usage_summary()
{
  fprintf(stderr, "SFFSummary - Produce summary of read lengths and predicted qualities.\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s summary [-h -l 50,100,150 -o out.txt] -m 0,21,21 -q 0,17,20 -s in.sff\n", "SFFUtils");
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h,--help              This message\n");
  fprintf(stderr, "  -d,--qual-length-file  File to which to write quality lengths\n");
  fprintf(stderr, "  -l,--read-length       Read lengths to summarize\n");
  fprintf(stderr, "  -m,--min-length        Minimum lengths for a read to be considered\n");
  fprintf(stderr, "  -o,--out-file          File to write summary to\n");
  fprintf(stderr, "  -p,--report-pred-qlen  Reports predicted qlength to per-read output file\n");
  fprintf(stderr, "  -q,--qual              Quality levels (Phred-scale) to summarize\n");
  fprintf(stderr, "  -s,--sff-file          Input SFF file\n");
  fprintf(stderr, "\n");
  return 1;
}

int main_summary(int argc, const char *argv[])
{

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
  if (help || sffFile.empty() || tempQual.size()==0 || tempMinLength.size() == 0 )
    return usage_summary();

  // Confirm lengths option is concordant with quality levels requested
  if ( (tempMinLength.size() > 1) && (tempQual.size() != tempMinLength.size()) ) {
    cerr << "ERROR: -m option should specify either one or same number of values as -q option." << endl << endl;
    return usage_summary();
  }
  // Confirm qualities were supplied in ascending order
  for(unsigned int iQual=1; iQual < tempQual.size(); iQual++) {
    if(tempQual[iQual] < tempQual[iQual-1]) {
      cerr << "ERROR: -q option should specify ascending list of quality thresholds." << endl << endl;
      return usage_summary();
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

  if (sffFile == "")
    return EXIT_FAILURE;

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

}
