/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "OptArgs.h"
#include "Image.h"

using namespace std;

void usage() {
  cout << "readDat - a little application to extract data from a dat file.  Main purpose is" << endl;
  cout << "to serve as an example of how to use the Image::LoadSlice() API to read a dat" << endl;
  cout << "file." << endl;
  cout << "" << endl;
}

int main(int argc, const char *argv[]) {

  vector<string> datFiles;
  vector<unsigned int> col;
  vector<unsigned int> row;
  int minCol, maxCol, minRow, maxRow;
  bool uncompress;
  bool doNormalize;
  int normStart;
  int normEnd;
  bool XTCorrect;
  bool help;
  double baselineMinTime,baselineMaxTime;
  double loadMinTime,loadMaxTime;

  OptArgs opts;  
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(col,             "",      '-', "col");
  opts.GetOption(row,             "",      '-', "row");
  opts.GetOption(minCol,          "-1",    '-', "min-col");
  opts.GetOption(maxCol,          "-1",    '-', "max-col");
  opts.GetOption(minRow,          "-1",    '-', "min-row");
  opts.GetOption(maxRow,          "-1",    '-', "max-row");
  opts.GetOption(uncompress,      "false", '-', "uncompress");
  opts.GetOption(doNormalize,     "false", '-', "normalize");
  opts.GetOption(normStart,       "5",     '-', "norm-start");
  opts.GetOption(normEnd,         "20",    '-', "norm-end");
  opts.GetOption(XTCorrect,       "true",  '-', "xtcorrect");
  opts.GetOption(baselineMinTime, "0",     '-', "baseline-min-time");
  opts.GetOption(baselineMaxTime, "-1",    '-', "baseline-max-time");
  opts.GetOption(loadMinTime,     "0",     '-', "load-min-time");
  opts.GetOption(loadMaxTime,     "-1",    '-', "load-max-time");
  opts.GetOption(help,            "false", 'h', "help");
  opts.GetLeftoverArguments(datFiles);
  //opts.CheckNoLeftovers();
  if(help) {
    usage();
    return(0);
  }
  if (datFiles.empty()) {
    usage();
    return(1);
  }

  bool returnSignal=true;
  bool returnMean=true;
  bool returnSD=true;
  bool returnLag=true;
  unsigned int nCol,nRow,nFrame;
  vector<unsigned int> colOut;
  vector<unsigned int> rowOut;
  vector< vector<double> > frameStart;
  vector< vector<double> > frameEnd;
  vector< vector< vector<short> > > signal;
  vector< vector<short> > mean;
  vector< vector<short> > sd;
  vector< vector<short> > lag;

  Image i;
  if(!i.LoadSlice(datFiles,col,row,minCol,maxCol,minRow,maxRow,returnSignal,returnMean,returnSD,returnLag, uncompress,doNormalize,normStart,normEnd,XTCorrect,baselineMinTime,baselineMaxTime,loadMinTime,loadMaxTime,nCol,nRow,colOut,rowOut,nFrame,frameStart,frameEnd,signal,mean,sd,lag)) {
    cerr << "Problem loading raw data" << endl;
    return(1);
  }

  // frameStarts
  cout << "NA\tNA\tNA\tNA";
  for(unsigned int iDat=0; iDat < datFiles.size(); iDat++) {
    for(unsigned int iFrame=0; iFrame < frameStart[iDat].size(); iFrame++) {
      cout << "\t" << setprecision(3) << frameStart[iDat][iFrame];
    }
  }
  cout << endl;

  // frameEnds
  cout << "NA\tNA\tNA\tNA";
  for(unsigned int iDat=0; iDat < datFiles.size(); iDat++) {
    for(unsigned int iFrame=0; iFrame < frameEnd[iDat].size(); iFrame++) {
      cout << "\t" << setprecision(3) << frameEnd[iDat][iFrame];
    }
  }
  cout << endl;

  // signal values
  unsigned int nWell = colOut.size();
  for(unsigned int iWell=0; iWell<nWell; iWell++) {
    cout << colOut[iWell];
    cout << "\t" << rowOut[iWell];
    for(unsigned int iDat=0; iDat < datFiles.size(); iDat++) {
      if(returnMean)
        cout << "\t" << mean[iDat][iWell];
      else
        cout << "\tNA";
      if(returnSD)
        cout << "\t" << sd[iDat][iWell];
      else
        cout << "\tNA";
      if(returnSignal) {
        for(unsigned int iFrame=0; iFrame < signal[iDat][iWell].size(); iFrame++) {
          cout << "\t" << signal[iDat][iWell][iFrame];
        }
      }
    }
    cout << endl;
  }

  return(0);
}
