/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <fstream>

#include "SynchDatSerialize.h"
#include "Image.cpp"
#define D '\t'

using namespace std;
std::ostream & operator<<(std::ostream &o, const TraceChunk& tc) {
  o << tc.mRowStart << D << tc.mColStart << D << tc.mFrameStart << D << tc.mFrameStep << D << tc.mHeight << D << tc.mWidth << D << tc.mDepth << D << tc.mT0 << D << tc.mTMidNuc << D << tc.mSigma << D << tc.mBaseFrameRate << D << tc.mChipRow << D << tc.mChipCol << D << tc.mOrigFrames << D << tc.mStartDetailedTime << D << tc.mStopDetailedTime << D << tc.mLeftAvg;
  return o;
}

int main(int argc, char * argv[]) {
  if(argc != 4) {
    cout << "Usage:\n  DumpSynchDat <in.sdat> <regionsout.txt> <dataout.txt>\n" << endl;
    exit(1);
  }
  TraceChunkSerializer serializer;
  SynchDat sdat;
  cout << "Reading sdat." << endl;
  serializer.SetRecklessAbandon(true);
  serializer.Read(argv[1], sdat);
  cout << "Writing text." << endl;
  ofstream o(argv[3]);
  cout << "Compressor: " << serializer.GetCompressionType() << endl;
  cout << "Rows: " << sdat.NumRow() << endl;
  cout << "Cols: " << sdat.NumCol() << endl;
  int idx = (int)(sdat.mChunks.mRowBin*.25/sdat.mChunks.mRowBin) * sdat.mChunks.mColBin + (int)(sdat.mChunks.mColBin*.25/sdat.mChunks.mColBin);
  TraceChunk &chunk = sdat.GetChunk(idx);
  cout << "For region: " << idx << " T0: " << chunk.mT0 << " Tmid: " << chunk.mTMidNuc << " Sigma: " << chunk.mSigma << endl;

  int keyCount = sdat.GetCount();
  string key, value;
  cout << endl << "Meta data:" << endl;
  for (int i = 0; i < keyCount; i++) {
    sdat.GetEntry(i, key, value);
    cout << key << " = " << value << endl;
  }
  ofstream regOut(argv[2]);
  for (size_t i = 0; i < sdat.mChunks.mBins.size(); i++) {
    regOut << sdat.mChunks.mBins[i] << endl;
  }
  regOut.close();
  cout << "Seconds: ";
  for (size_t i = 0; i < sdat.mChunks.mBins[0].mTimePoints.size(); i++) {
    cout << sdat.mChunks.mBins[0].mTimePoints[i] << ", ";
  }
  cout << endl;
  for (size_t row = 0; row < sdat.NumRow(); row++) {
    for (size_t col = 0; col < sdat.NumCol(); col++) {
      o << row << D << col << D << sdat.GetT0(row, col);
      for (int frame = 0; frame < sdat.GetFrames(); frame++) {
        o << D << sdat.AtWell(row, col, frame);
      }
      o << endl;
    }
  }
  o.close();
  cout << "Finished." << endl;
}
