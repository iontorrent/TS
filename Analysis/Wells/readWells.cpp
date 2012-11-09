/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "IonErr.h"
#include "OptArgs.h"
#include "RawWells.h"

using namespace std;

void usage() {
  cout << "readWells - a little application to extract data from wells files.  Main purpose is" << endl;
  cout << "to serve as an example of how to use the RawWells interface." << endl;
  cout << endl;
  cout << "The wells to read can be specified either as a set of arbitrary well" << endl;
  cout << "coordinates or as a rectangular region." << endl;
  cout << "" << endl;
  cout << "Options:" << endl;
  cout << "  --col     : comma-separated list of col coordinates to read" << endl;
  cout << "  --row     : comma-separated list of row coordinates to read" << endl;
  cout << "  --min-col : lower column bound for reading rectangular area" << endl;
  cout << "  --max-col : upper column bound for reading rectangular area" << endl;
  cout << "  --min-row : lower row bound for reading rectangular area" << endl;
  cout << "  --max-row : upper row bound for reading rectangular area" << endl;
  cout << "  --help    : this help message" << endl;
  cout << "" << endl;
}

int main(int argc, const char *argv[]) {

  vector<string> wellFiles;
  vector<int32_t> col;
  vector<int32_t> row;
  int32_t minCol, maxCol, minRow, maxRow;
  bool help;

  OptArgs opts;  
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(col,             "",      '-', "col");
  opts.GetOption(row,             "",      '-', "row");
  opts.GetOption(minCol,          "-1",    '-', "min-col");
  opts.GetOption(maxCol,          "-1",    '-', "max-col");
  opts.GetOption(minRow,          "-1",    '-', "min-row");
  opts.GetOption(maxRow,          "-1",    '-', "max-row");
  opts.GetOption(help,            "false", 'h', "help");
  opts.GetLeftoverArguments(wellFiles);
  if(help) {
    usage();
    return(0);
  }
  if (wellFiles.empty()) {
    usage();
    return(1);
  }

  if(col.size()>0 || row.size() > 0) {
    if(col.size() != row.size())
      ION_ABORT("--col and --row should each specify the same number of values\n");
  } else {
    if(minCol==-1 && minRow==-1 && maxCol==-1 && maxRow==-1) {
      usage();
      ION_ABORT("Must specify a set of well coordinates\n");
    } else {
      if(minCol > maxCol || minCol < 0) {
        usage();
        ION_ABORT("Invalid column bounds\n");
      } else if(minRow > maxRow || minRow < 0) {
        usage();
        ION_ABORT("Invalid row bounds\n");
      }
    }
  
    for(int32_t iCol=minCol; iCol < maxCol; iCol++) {
      for(int32_t iRow=minRow; iRow < maxRow; iRow++) {
        col.push_back(iCol);
        row.push_back(iRow);
      }
    }
  }

  for(unsigned int iFile=0; iFile < wellFiles.size(); iFile++) {
    //string wellDir,wellFile;
    //pathSplit(wellFiles[iFile],wellDir,wellFile);
    //RawWells wells(wellDir.c_str(), wellFile.c_str());
    RawWells wells(wellFiles[iFile].c_str(),0,0);
    wells.SetSubsetToLoad(&col[0], &row[0], col.size());
    wells.OpenForRead();
    uint64_t nFlow = wells.NumFlows();
    cout << "#nFlow=" << nFlow << endl;
    string flowOrder = wells.FlowOrder();
    cout << "#flowOrder=" << flowOrder << endl;
    for(unsigned int iWell=0; iWell<col.size(); iWell++) {
      cout << col[iWell] << "\t" << row[iWell];
      const WellData *w = wells.ReadXY(col[iWell],row[iWell]);
      for(unsigned int iFlow=0; iFlow<nFlow; iFlow++) {
        cout << "\t" << setprecision(3) << w->flowValues[iFlow];
      }
      cout << endl;
    }
  }

  return(0);
}
