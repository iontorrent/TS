/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <sstream>
#include <vector>
#include <string>

#include <gtest/gtest.h>
#include "FlowDiffStats.h"
#include "GenomeDiffStats.h"
#include "Utils.h"

using namespace std;


TEST(FlowDiffStats_Test, CompareFlowsTest) {
  vector<int> g = char2Vec<int>("001100101210200100100110011002010011011");
  //                             ||||||  ||||||||||||||||||||||||||||||| 
  vector<int> r = char2Vec<int>("001100211210200100100110011002010011011");
  GenomeDiffStats fds(7);
  fds.CompareFlows("r10|c10", g, r);
  //  fds.PrintSumStat(4, cout);
  EXPECT_EQ(fds.GetCount(4,1,2), 1);
  EXPECT_EQ(fds.GetCount(4,0,1), 1);
  EXPECT_EQ(fds.GetCount(4,1,0), 0);
  EXPECT_EQ(fds.GetCount(4,2,2), 3);
}

TEST(FlowDiffStats_Test, CompareFlowsTest2) {
  vector<int> g = char2Vec<int>("1010010100110020211110101002001100100102");
  //                             |||||||||||||||||||||||||||| | ||||||||| 
  vector<int> r = char2Vec<int>("1010010100110020211110101002100100100102");
  GenomeDiffStats fds(7);
  fds.CompareFlows("r10|c10", g, r);
  //  fds.PrintSumStat(4, cout);
  EXPECT_EQ(fds.GetCount(4,1,2), 0);
  EXPECT_EQ(fds.GetCount(4,0,1), 1);
  EXPECT_EQ(fds.GetCount(4,1,0), 1);
  EXPECT_EQ(fds.GetCount(4,2,2), 4);
}

TEST(FlowDiffStats_Test, FillInPairedFlowsTest) {
  string g = "GATCAAAGCGCCAGACTTCCT---G";
  string r = "G-TC--GGC-CCAGACTTCCTTCGG";
  GenomeDiffStats fds(7); 
  vector<int> gVec;
  vector<int> rVec;
  int num = g.length();
  fds.FillInPairedFlows(num, gVec, rVec, g, r, 40);
  vector<int> gGold = char2Vec<int>("0001010010100301001100200101011020201001");
  vector<int> rGold = char2Vec<int>("0001000010110001001000200101011020202012");
  for (size_t i = 0; i < gGold.size(); i++) {
    EXPECT_EQ(gGold[i], gVec[i]);
    EXPECT_EQ(rGold[i], rVec[i]);
  }
}

TEST(FlowDiffStats_Test, NameParseTest) {
  GenomeDiffStats fds(7); 
  string name1 = "r12|c20";
  string name2 = "blah|:13:21";
  int row, col;
  fds.GetRowColFromName(name1, row, col);
  cout << name1 << " -> " << row << "," << col << endl;
  EXPECT_EQ(row, 12);
  EXPECT_EQ(col, 20);

  fds.GetRowColFromName(name2, row, col);
  cout << name2 << " -> " << row << "," << col << endl;
  EXPECT_EQ(row, 13);
  EXPECT_EQ(col, 21);
}
