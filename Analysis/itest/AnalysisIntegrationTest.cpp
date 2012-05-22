/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <stdlib.h>

using namespace std;

#include <fstream>
#include <errno.h>
#include "FileEquivalent.h"
#include "OptArgs.h"
//#include "FlowDiffStats.h"
#include "Utils.h"
#include "NumericalComparison.h"

string mConfigFile;
string mPrefix;

bool RunAnalysis(const std::string &name, const std::string &dataDir,
		 const std::string &gold, const std::string &test,
		 const std::string &analysisExe, 
		 NumericalComparison<double> &wellsCompare,
                 vector<SffComparison> &sffCompare) {
	//Unused parameter generates compiler warning, so...
	cout << name << endl;
	
  // Have to do this stuff with prefixes as we have to cd to the directory of interest
  string prefix = mPrefix;
  if (!prefix.empty()) {
    prefix += "/";
  }

  mode_t nmask;

  nmask = S_IXUSR | S_IRUSR | S_IWUSR | /* owner read write */
    S_IXGRP | S_IRGRP | S_IWGRP | /* group read write */
    S_IXOTH | S_IROTH | S_IWOTH;            /* other read */
  string dir = prefix + test;
  int r = mkdir(dir.c_str(), nmask); 
  if (r != 0 && errno != EEXIST) {
    cerr << "Couldn't mkdir " << test << endl;
    exit(1);
  }

  string command = "cd " + test + " && " + prefix + analysisExe + "  --region-size=50x50 --bkg-effort-level=17 --well-stat-file wellStats.txt -c 46 --libraryKey=TCAG --no-subdir " + prefix + dataDir + " > analysis.log 2>&1 " ;
  r = 0;
  //  cout << "Running command: " << command << endl;
  r = system(command.c_str());
  if (r != 0) {
    return false;
  }

  wellsCompare = CompareRawWells(prefix + test + "/1.wells", prefix + gold + "/1.wells", .01, 10000);
  int found, missing, goldOnly;
  sffCompare = SFFInfo::CompareSFF(prefix + test + "/rawlib.sff", prefix + gold + "/rawlib.sff", 10, found, missing, goldOnly);
  return true;
}

TEST(AnalysisIntegrationTest, AnalysisCropTest) {
  //  EXPECT_EQ(mConfigFile, "holla");
  ifstream config;
  config.open(mConfigFile.c_str(), ifstream::in);
  if (!config.good()) {
    cerr << "Couldn't open file: " << mConfigFile << endl;
    exit(1);
  }
  string line;
  //bool found = false;
  vector<string> words;
  while(getline(config, line)) {
    if (line.size() > 1 && line[0] == '#') 
      continue;
    split(line,'\t',words);
    NumericalComparison<double> wells;
    vector<SffComparison> sff;
    bool ranOk = RunAnalysis(words[0], words[1], words[2], words[3], words[4], wells, sff);
    if (!ranOk) {
      ADD_FAILURE() << "Failure running analysis for " << words[0] << endl;
    }
    else {
      if (wells.GetNumDiff() > 0) {
	ADD_FAILURE() << "Wells File Failure: " << words[0] << " - " << wells.GetNumDiff() << " different "
		      << wells.GetCorrelation() << " correlation" << endl;
      }
      for (size_t i = 0; i < sff.size(); i++) {
          if (sff[i].GetNumDiff() > 0) {
              ADD_FAILURE() << "SFF File Failure: " << words[0] << " - " << sff[i].GetNumDiff() << " different "
                << sff[i].GetCorrelation() << " correlation" << endl;
          }
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  OptArgs opts;
  opts.ParseCmdLine(argc, (const char **) argv);
  opts.GetOption(mConfigFile, "", '-', "analysis-test-config");
  opts.GetOption(mPrefix, "", '-', "prefix");
  return RUN_ALL_TESTS();
}
