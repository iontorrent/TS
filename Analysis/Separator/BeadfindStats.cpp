/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <vector>

#include "BFReference.h"
#include "DifferentialSeparator.h"
#include "IonErr.h"
#include "Image.h"
#include "T0Calc.h"
#include "MaskFunctions.h"
#include "ChipIdDecoder.h"

using namespace std;

void CalcBfT0(DifSepOpt &opts, std::vector<float> &t0vec, Mask &mask, int gridMod,
              const std::string &file, int &nRow, int &nCol, SpatialContext &loc_context) {
  string bfFile = file;
  Image img;
  img.SetImgLoadImmediate (false);
  img.SetIgnoreChecksumErrors (opts.ignoreChecksumErrors);
  bool loaded =   img.LoadRaw(bfFile.c_str());
  if (!loaded) { ION_ABORT ("Couldn't load file: " + bfFile); }
  const RawImage *raw = img.GetImage(); 
  nRow = raw->rows;
  nCol = raw->cols;
  t0vec.resize(nRow * nCol,0);
  mask.Init (nCol,nRow,MaskEmpty);
  loc_context.chip_len_x =  raw->cols;
  loc_context.chip_offset_x = img.GetImage()->chip_offset_x;
  loc_context.chip_len_y = raw->rows;
  loc_context.chip_offset_y = img.GetImage()->chip_offset_y;
  loc_context.rows = raw->rows;
  loc_context.cols = raw->cols;
  img.FilterForPinned(&mask, MaskEmpty, false);
  T0Calc t0;
  t0.SetWindowSize(3);
  t0.SetMinFirstHingeSlope(opts.beadfindBfThreshold[0]/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(opts.beadfindBfThreshold[1]/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(opts.beadfindBfThreshold[2]/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(opts.beadfindBfThreshold[3]/raw->baseFrameRate);
  t0.SetDebugLevel(opts.outputDebug);
  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStepY, opts.t0MeshStepX, opts.nCores);
  int *timestamps = raw->timestamps;
  t0.SetTimeStamps(timestamps, frames);
  T0Prior prior;
  prior.mTimeEnd = 3000;
  t0.SetGlobalT0Prior(prior);
  t0.CalcAllSumTrace(data);
  t0.CalcT0FromSum();
  t0.CalcIndividualT0(t0vec, opts.useMeshNeighbors, gridMod);
  for (size_t i = 0; i < t0vec.size(); i++) {
    if (t0vec[i] > 0) {
      int tmpT0End = img.GetFrame(t0vec[i]-img.GetFlowOffset());
      int tmpT0Start = tmpT0End - 1;
      //      assert(tmpT0Start >= 0);
      if (tmpT0Start < 0)
        t0vec[i] = 0;
      else {
        float t0Frames = (t0vec[i] - raw->timestamps[tmpT0Start])/(raw->timestamps[tmpT0End]-raw->timestamps[tmpT0Start])*(raw->compToUncompFrames[tmpT0End] - raw->compToUncompFrames[tmpT0Start]) + raw->compToUncompFrames[tmpT0Start];
        t0Frames = max(0.0f,t0Frames);
        t0Frames = min(t0Frames, frames - 1.0f);
        t0vec[i] = t0Frames;
      }
    }
  }
  if (opts.outputDebug > 0) {
    string refFile = opts.outData + ".reference_bf_t0." + file + ".txt";
    ofstream out(refFile.c_str());
    t0.WriteResults(out);
    out.close();
  }
  img.Close();
}

int main(int argc, const char *argv[]) {

  if (argc != 4) {
    cout << "BeadfindStats - Utility program to dump out the beadfind statistics." << endl;
    cout << "arguments are path to dat, h5 file to output to and region size." << endl;
    cout << "   usage: " << endl;
    cout << "     BeafindStats beadfind_pre_0001.dat out.h5 100" << endl;
    exit(1);
  }
  string rootDir = argv[1];
  string h5File = argv[2];
  int size = atoi(argv[3]);
  string bfFile = rootDir + "/beadfind_pre_0001.dat";
  string explog_path = rootDir + "/explog.txt";
  string bf2File = rootDir  + "/beadfind_pre_0003.dat";
  if (!isFile(explog_path.c_str())) {
    explog_path = rootDir + "/../explog.txt";
  }
  const char *id = GetChipId(explog_path.c_str());
  ChipIdDecoder::SetGlobalChipId(id);
  DifSepOpt opts;
  opts.useMeshNeighbors = 0;
  string sId = id;
  int withinGrid = 0;
  if(sId.find("900") != string::npos) {
    opts.useMeshNeighbors = 3;
    withinGrid = size;
    opts.t0MeshStepX = 10;
    opts.t0MeshStepY = 10;
  }
  else {
    opts.useMeshNeighbors = 2;
    opts.t0MeshStepX = size;
    opts.t0MeshStepY = size;
  }
  
  opts.beadfindBfThreshold = std::vector<float>({-5, 300, -20000, -10});
  
  
  Mask mask;
  int row = 0, col = 0;
  std::vector<float> t0;
  std::vector<float> t02;
  SpatialContext loc_context;
  cout << "Step is: " << size << endl;

  CalcBfT0(opts, t0, mask, withinGrid, bfFile.c_str(), row, col, loc_context);
  CalcBfT0(opts, t02, mask, withinGrid, bf2File.c_str(), row, col, loc_context);
  for (size_t i = 0; i < t0.size(); i++) {
    if (t0[i] > 0 && t02[i] > 0) {
      t0[i] = (t0[i] + t02[i])/2.0;
    }
    else {
      t0[i] = max(t0[i], t02[i]);
    }
  }

  SetExcludeMask(loc_context, &mask, ChipIdDecoder::GetChipType(), mask.H(), mask.W(), std::string(""), false);
  size_t numWells = mask.H() * mask.W();
  int excludeWells = 0;
  for (size_t i = 0; i < numWells; i++) {
    if((mask[i] & MaskExclude) != 0) {
      excludeWells++;
    }
  }
  cout << "Found: " << excludeWells << " excluded wells." << endl;
  BFReference reference;
  reference.Init (mask.H(), mask.W(),
                  size, size,
                  .95, .98);
  reference.SetT0(t0);
  reference.SetRegionSize (size, size);
  reference.SetNumEmptiesPerRegion(opts.percentReference * size * size);
  reference.SetIqrOutlierMult(opts.iqrMult);
  reference.SetDoRegional (opts.useMeshNeighbors == 0);
  H5File h5file(h5File);
  h5file.Open(true);
  h5file.Close();
  reference.SetDebugH5(h5File);
  reference.CalcReference (bfFile, mask, BFReference::BFIntMaxSd);
  return 0;
}
