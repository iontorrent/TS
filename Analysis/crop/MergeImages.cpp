/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>

#include "MergeAcq.h"
#include "Acq.h"
#include "OptArgs.h"

using namespace std;

void usage() {
  cout << "MergeImages - Merge two dat files into a single larger image by putting once" << endl;
  cout << "on top of the other" << endl;
  cout << "" << endl;
  cout << "usage: MergeImages --bottom bottom.dat --top top.dat --merged merged.dat" << endl;
  cout << "  --bottom dat file to stitch onto bottom in merged dat" << endl;
  cout << "  --top dat file to stich onto top in merged dat" << endl;
  cout << "  --merged name of file to output merged to." << endl;
    exit(1);
}

int main(int argc, const char *argv[]) {
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  bool help;
  string topFile, bottomFile, outFile;
  opts.GetOption(topFile, "", '-', "top");
  opts.GetOption(bottomFile, "", '-', "bottom");
  opts.GetOption(outFile, "", '-', "merged");
  opts.GetOption(help, "false", 'h', "help");
  if (help || argc == 1) {
    usage();
  }
  ION_ASSERT(!topFile.empty() && !bottomFile.empty() && !outFile.empty(), 
             "Need top, bottom and merged files. use --help for details.");
  MergeAcq merger;
  Image top;
  Image bottom;
  Image combo;
  cout << "Loading images." << endl;
  ION_ASSERT(top.LoadRaw(topFile.c_str()), "Couldn't load file.");
  ION_ASSERT(bottom.LoadRaw(bottomFile.c_str()), "Couldn't load file.");
  merger.SetFirstImage(&bottom);
  merger.SetSecondImage(&top, bottom.GetRows(), 0); // starting vertically raised but columns the same.
  cout << "Merging." << endl;
  merger.Merge(combo);
  Acq acq;
  cout << "Saving. " << endl;
  acq.SetData(&combo);
  acq.WriteVFC(outFile.c_str(), 0, 0, combo.GetCols(), combo.GetRows());
  cout << "Done." << endl;
  return 0;
}
