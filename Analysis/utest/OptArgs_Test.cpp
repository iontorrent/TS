/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include "OptArgs.h"
#include "Utils.h"

using namespace std;

TEST(OptArgs_Test, BasicTest) {
  const char *argv[] = {"test-prog", "--mult-double", "0.0,1.0", "--mult-int", "1,-10,0", "--hello", "world", "-b", "true", "-d", "2.0", "-i", "5", "--unchecked","fun","trailing1", "trailing2"};
  int argc = ArraySize(argv);
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string hello = "junk";
  int i = -1;
  double d = -1.0;
  bool b = false;
  string notCalled = "non-specified";
  vector<string> unchecked;
  vector<double> multDouble;
  vector<int> multInt;
  opts.GetOption(hello, "junk", 's', "hello");
  opts.GetOption(i, "-1", 'i', "int-opt");
  opts.GetOption(d, "-1.0", 'd', "double-opt");
  opts.GetOption(b, "false", 'b', "bool-opt");
  opts.GetOption(multDouble, "", '-', "mult-double");
  opts.GetOption(multInt, "", '-', "mult-int");
  opts.GetUncheckedOptions(unchecked);
  EXPECT_EQ(hello, "world");
  EXPECT_EQ(multDouble.size(), 2);
  EXPECT_EQ(multDouble[0], 0.0);
  EXPECT_EQ(multDouble[1], 1.0);
  EXPECT_EQ(multInt.size(), 3);
  EXPECT_EQ(multInt[0], 1);
  EXPECT_EQ(multInt[1], -10);
  EXPECT_EQ(multInt[2], 0);
  EXPECT_EQ(b, true);
  EXPECT_EQ(d, 2.0);
  EXPECT_EQ(i, 5);
  EXPECT_EQ(unchecked[0], "unchecked");
  vector<string> leftover;
  opts.GetLeftoverArguments(leftover);
  ASSERT_EQ(leftover.size(), 2);
  EXPECT_EQ(leftover[0], "trailing1");
  EXPECT_EQ(leftover[1], "trailing2");
}

TEST(OptArgs_DeathTest, BadTypesTest) {
  const char *argv[] = {"test-prog", "--mult-double", "0.0a,1.0,", "--hello", "world", "-b", "bad", "-d", "true", "-i", "x", "--unchecked","fun","trailing1", "trailing2"};
  int argc = ArraySize(argv);
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string hello = "junk";
  int i = -1;
  double d = -1.0;
  bool b = false;
  string notCalled = "non-specified";
  vector<double> multDouble;
  opts.GetOption(hello, "junk", 's', "hello");
  EXPECT_DEATH(opts.GetOption(i, "-1", 'i', "int-opt"), ".*");
  EXPECT_DEATH(opts.GetOption(d, "-1.0", 'd', "double-opt"), ".*");
  EXPECT_DEATH(opts.GetOption(b, "false", 'b', "bool-opt"), ".*");
  EXPECT_DEATH(opts.GetOption(multDouble, "", '-', "mult-double"), ".*");
  EXPECT_EQ(hello, "world");
  vector<string> leftover;
  opts.GetLeftoverArguments(leftover);
  ASSERT_EQ(leftover.size(), 2);
  EXPECT_EQ(leftover[0], "trailing1");
  EXPECT_EQ(leftover[1], "trailing2");
}


