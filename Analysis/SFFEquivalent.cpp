/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <assert.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

#include "Utils.h"
#include "NumericalComparison.h"
#include "SFFWrapper.h"
#include "OptArgs.h"
#include "IonVersion.h"
#include "FileEquivalent.h"
#include "SampleStats.h"

/** Utility for comparisons... */

bool Abort(const string &msg) {
  cerr << msg << endl;
  exit(1);
  return false;
}

int main(int argc, const char *argv[]) {

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string queryFile, goldFile;
  double epsilon;
  bool help = false;
  bool version = false;
  int allowedWrong = 0;
  double minCorrelation = 1;
  opts.GetOption(queryFile, "", 'q', "query-sff");
  opts.GetOption(goldFile, "", 'g', "gold-sff");
  opts.GetOption(epsilon, "0.0", 'e', "epsilon");
  opts.GetOption(allowedWrong, "0", 'm', "max-mixmatch");
  opts.GetOption(minCorrelation, "1", 'c', "min-cor");
  opts.GetOption(help, "false", 'h', "help");
  opts.GetOption(version, "false", 'v', "version");
  opts.CheckNoLeftovers();
  
  if (version) {
  	fprintf (stdout, "%s", IonVersion::GetFullVersion("SFFEquivalent").c_str());
  	exit(0);
  }
  
  if (queryFile.empty() || goldFile.empty() || help) {
    cout << "SFFEquivalent - Check to see how similar two sff files are to each other" << endl 
	 << "options: " << endl
	 << "   -g,--gold-sff      trusted sff to compare against." << endl
	 << "   -q,--query-sff     new sff to check." << endl
	 << "   -e,--epsilon       maximum allowed difference to be considered equivalent." << endl 
	 << "   -m,--max-mixmatch  maximum number of non-equivalent entries to allow." << endl
	 << "   -c,--min-cor       minimum correlation allowed to be considered equivalent." << endl
	 << "   -h,--help          this message." << endl
	 << "" << endl 
         << "usage: " << endl
	 << "   SFFEquivalent -e 10 --query-sff query.sff --gold-sff gold.sff " << endl;
    exit(1);
  }

  int found = 0, missing = 0, goldOnly;
  vector<SffComparison> compare = SFFInfo::CompareSFF(queryFile, goldFile, epsilon, found, missing, goldOnly);
  bool passed = true;
  for (size_t i = 0; i < compare.size(); i++) {
    bool ok = true;
    if((compare[i].total - allowedWrong) > compare[i].same || 
       (isfinite(compare[i].correlation) && compare[i].correlation < minCorrelation)) {
      ok = false;
    }
    if (!ok) {
      cout << "*FAIL* ";
    }
    cout << "Field: " << compare[i].name << endl;
    cout << compare[i].same << ", " << compare[i].different << ", " << compare[i].total  << " matching, different, total reads" << endl;
    if (i == 0) {
      cout << compare[i].missing << " non-matching reads" << endl;
    }
    cout << compare[i].same << " (" << (100.0 * compare[i].same/compare[i].total) <<  "%) are equivalent. " 
	 << compare[i].different << " (" << (100.0 * compare[i].different)/compare[i].total <<  "%) are not. Correlation of: " << compare[i].correlation << endl;
    cout << endl;
    if((compare[i].total - allowedWrong) > compare[i].same || 
       (isfinite(compare[i].correlation) && compare[i].correlation < minCorrelation)) {
      passed = false;
    }
  }
  if (!passed) {
    cout << "FAIL - Sff files not equivalent for allowed mismatch: " << allowedWrong 
	 << " minimum correlation: " << minCorrelation << endl;
    
    return 1;
  }
  cout << "PASS - Sff files equivalent for allowed mismatch: " << allowedWrong 
       << " minimum correlation: " << minCorrelation << endl;
  return 0;
}
