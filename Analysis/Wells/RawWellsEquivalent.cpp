/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <assert.h>
#include <iostream>
#include <stdlib.h>

#include "Utils.h"
#include "NumericalComparison.h"
#include "RawWells.h"
#include "OptArgs.h"
#include "IonVersion.h"

using namespace std;

NumericalComparison<double> CompareWells(const string &queryFile, const string &goldFile, 
					 float epsilon, double maxAbsVal) {
  
  NumericalComparison<double> compare(epsilon);
  string queryDir, queryWells, goldDir, goldWells;
  FillInDirName(queryFile, queryDir, queryWells);
  FillInDirName(goldFile, goldDir, goldWells);

  RawWells queryW(queryDir.c_str(), queryWells.c_str());
  RawWells goldW(goldDir.c_str(), goldWells.c_str());
  
  struct WellData goldData;
  goldData.flowValues = NULL;
  struct WellData queryData;
  queryData.flowValues = NULL;
  cout << "Opening query." << endl;
  queryW.OpenForRead();
  cout << "Opening gold." << endl;
  goldW.OpenForRead();
  unsigned int numFlows = goldW.NumFlows();
  while( !queryW.ReadNextRegionData(&queryData) ) {
    assert(!goldW.ReadNextRegionData(&goldData));
    for (unsigned int i = 0; i < numFlows; i++) {
      if (isfinite(queryData.flowValues[i]) && isfinite(goldData.flowValues[i]) && 
	  (fabs(queryData.flowValues[i]) < maxAbsVal && fabs(goldData.flowValues[i]) < maxAbsVal)) {
	compare.AddPair(queryData.flowValues[i], goldData.flowValues[i]);
      }
    }
  }
  const SampleStats<double> ssX = compare.GetXStats();
  const SampleStats<double> ssY = compare.GetYStats();
  cout << "query values: "  << ssX.GetMean() << " +/- "  << ssX.GetSD() << endl;
  cout << "gold values: "  << ssY.GetMean() << " +/- "  << ssY.GetSD() << endl;
  return compare;
}

int main(int argc, const char *argv[]) {

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string queryFile, goldFile;
  double epsilon;
  bool help = false;
  bool version = false;
  int allowedWrong = 0;
  double maxAbsVal = 0;
  double minCorrelation = 1;
  opts.GetOption(queryFile, "", 'q', "query-wells");
  opts.GetOption(goldFile, "", 'g', "gold-wells");
  opts.GetOption(epsilon, "0.0", 'e', "epsilon");
  opts.GetOption(allowedWrong, "0", 'm', "max-mismatch");
  opts.GetOption(minCorrelation, "1", 'c', "min-cor");
  opts.GetOption(maxAbsVal, "1e3", '-', "max-val");
  opts.GetOption(help, "false", 'h', "help");
  opts.GetOption(version, "false", 'v', "version");
  opts.CheckNoLeftovers();
  
  if (version) {
  	fprintf (stdout, "%s", IonVersion::GetFullVersion("RawWellsEquivalent").c_str());
  	exit(0);
  }
  
  if (queryFile.empty() || goldFile.empty() || help) {
    cout << "RawWellsEquivalent - Check to see how similar two wells files are to each other" << endl 
	 << "options: " << endl
	 << "   -g,--gold-wells    trusted wells to compare against." << endl
	 << "   -q,--query-wells   new wells to check." << endl
	 << "   -e,--epsilon       maximum allowed difference to be considered equivalent." << endl 
	 << "   -m,--max-mixmatch  maximum number of non-equivalent entries to allow." << endl
	 << "   -c,--min-cor       minimum correlation allowed to be considered equivalent." << endl
	 << "      --max-val       maximum absolute value considered (avoid extreme values)." << endl
	 << "   -h,--help          this message." << endl
	 << "" << endl 
         << "usage: " << endl
	 << "   RawWellsEquivalent -e 10 --query-wells query.wells --gold-wells gold.wells " << endl;
    exit(1);
  }

  NumericalComparison<double> compare = CompareWells(queryFile, goldFile, epsilon, maxAbsVal);
  cout << compare.GetCount() << " total values. " << endl
       << compare.GetNumSame() << " (" << (100.0 * compare.GetNumSame())/compare.GetCount() <<  "%) are equivalent. " << endl
       << compare.GetNumDiff() << " (" << (100.0 * compare.GetNumDiff())/compare.GetCount() <<  "%) are not equivalent. " << endl 
       << "Correlation of: " << compare.GetCorrelation() << endl;

  if((compare.GetCount() - allowedWrong) > compare.GetNumSame() || 
     (compare.GetCorrelation() < minCorrelation && compare.GetCount() != compare.GetNumSame())) {
     cout << "Wells files not equivalent for allowed mismatch: " << allowedWrong 
     << " minimum correlation: " << minCorrelation << endl;
     return 1;
  }
  cout << "Wells files equivalent for allowed mismatch: " << allowedWrong 
       << " minimum correlation: " << minCorrelation << endl;
  return 0;
}
