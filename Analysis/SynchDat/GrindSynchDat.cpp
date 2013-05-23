/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <fstream>
#include <string>
#include "hdf5.h"
#include "SynchDatSerialize.h"
#include "Image.h"
#include "Utils.h"
#include "DeltaCompFstSmX.h"
#include "DeltaCompFst.h"

#define D '\t'

using namespace std;

void ReadGrindSDat(const char *filename, int numGrinds) {
  SynchDat *sdat = new SynchDat;
  
  ClockTimer timer;
  size_t processMicroSec = 0;
  size_t hdf5MicroSec = 0;
  size_t compressMicroSec = 0;
  TraceChunkSerializer *serializer = new TraceChunkSerializer();;
  cout << "Reading sdat." << endl;
  for(int i = 0; i < numGrinds; i++) {
    MemoryUsage("Before Read: " + ToStr(i));
    serializer->Read(filename, *sdat);
    processMicroSec += serializer->computeMicroSec;
    hdf5MicroSec += serializer->ioMicroSec;
    compressMicroSec += serializer->compressMicroSec;
    MemoryUsage("Before close: " + ToStr(i));
    sdat->Close();
    MemoryUsage("After Read: " + ToStr(i));
  }
  size_t usec = timer.GetMicroSec();
  cout << "Total time: " << usec / (1e3) << " milli seconds." << endl;
  cout << "Read took: " << usec / (1e3 * numGrinds) << " milli seconds per sdat." << endl;
  cout << "Read took: " << processMicroSec / (1e3 * numGrinds) << " milli seconds per sdat compute." << endl;
  cout << "Read took: " << hdf5MicroSec / (1e3 * numGrinds) << " milli seconds per sdat hdf5." << endl;
  cout << "Read took: " << compressMicroSec / (1e3 * numGrinds) << " milli seconds per sdat decompressing." << endl;
  MemoryUsage("Before cleaning up.");
  delete serializer;
  delete sdat;


  cout << "Cleaning up..." <<endl;
  MemoryUsage("After cleaning up");
}

void WriteGrindSDat(const char *filename, int numGrinds) {
  MemoryUsage("Before starting");
  TraceChunkSerializer *serializer = new TraceChunkSerializer();
  SynchDat *sdat = new SynchDat();
  MemoryUsage("After new");
  size_t processMicroSec = 0;
  size_t hdf5MicroSec = 0;
  size_t openMicroSec = 0;
  size_t compressMicroSec = 0;
  cout << "Reading sdat." << endl;
  serializer->SetRecklessAbandon(true);
  MemoryUsage("Before read");
  serializer->Read(filename, *sdat);
  MemoryUsage("After read");
  const char *fileOut = "_tmpfile.sdat";
  ClockTimer timer;
  cout << "Writing sdats." << endl;
  //  DeltaCompFstSmX *delta = new DeltaCompFstSmX();
  MemoryUsage("before delta");
  DeltaCompFst *delta = new DeltaCompFst();
  MemoryUsage("after delta");
  //TraceNoCompress *delta = new TraceNoCompress();
  serializer->SetCompressor(delta);
  for(int i = 0; i < numGrinds; i++) {
    remove(fileOut);
    MemoryUsage("Before write: " + ToStr(i));
    serializer->Write(fileOut, *sdat);
    MemoryUsage("After write: " + ToStr(i));
    processMicroSec += serializer->computeMicroSec;
    openMicroSec += serializer->openMicroSec;
    hdf5MicroSec += serializer->ioMicroSec;
    compressMicroSec += serializer->compressMicroSec;
  }

  size_t usec = timer.GetMicroSec();
  cout << "Total time: " << usec / (1e3) << " milli seconds." << endl;
  cout << "Write took: " << usec / (1e3 * numGrinds) << " milli seconds per sdat." << endl;
  cout << "Write took: " << processMicroSec / (1e3 * numGrinds) << " milli seconds per sdat compute." << endl;
  cout << "Write took: " << hdf5MicroSec / (1e3 * numGrinds) << " milli seconds per sdat hdf5." << endl;
  cout << "Write took: " << openMicroSec / (1e3 * numGrinds) << " milli seconds per opening sdat hdf5." << endl;
  cout << "Write took: " << compressMicroSec / (1e3 * numGrinds) << " milli seconds per sdat compressing." << endl;
  SynchDat *copy = new SynchDat();
  serializer->Read(fileOut, *copy);
  size_t nWells = sdat->NumCol() * sdat->NumRow();
  size_t diff = 0;
  size_t obs = 0;
  for (size_t wIx = 0; wIx < nWells; wIx++) {
    for (size_t n = 0; n < sdat->NumFrames(wIx); n++) {
      if (sdat->At(wIx, n) != copy->At(wIx, n)) {
	diff++;
      }
      obs++;
    }
  }
  MemoryUsage("Before close.");
  copy->Close();
  sdat->Close();
  sleep(1);
  MemoryUsage("Before delete.");
  delete sdat;
  delete copy;
  delete serializer;
  cout << "Got: " << diff << " differences for "  << obs << " observations." << endl;
  sleep(1);
  MemoryUsage("After done.");
  H5close();
  sleep(1);
  MemoryUsage("After h5close.");
}

void ReadGrindDat(const char *filename, int numGrinds) {
  Image img;
  ClockTimer timer;
  cout << "Reading image." << endl;
  for(int i = 0; i < numGrinds; i++) {
    img.LoadRaw(filename);
    img.Close();
  }
  size_t usec = timer.GetMicroSec();
  cout << "Total time: " << usec / (1e3) << " milli seconds." << endl;
  cout << "Took: " << usec / (1e3 * numGrinds) << " milli seconds per dat." << endl;
}

int main(int argc, char * argv[]) {
  if(argc != 5) {
    cout << "Usage:\n  GrindSyncDat filename filetype {read|write} numtimes" << endl;
    exit(1);
  }
  string fileType = argv[2];
  string ioType = argv[3];
  int numTimes = atoi(argv[4]);
  if (fileType == "sdat" && ioType == "read") {
    ReadGrindSDat(argv[1], numTimes);
  }
  else if (fileType == "sdat" && ioType == "write") {
    WriteGrindSDat(argv[1], numTimes);
  }
  else if (fileType == "dat" && ioType == "read") {
    ReadGrindDat(argv[1], numTimes);
  }
  else {
    cout << "Don't recognize arguments." << endl;
    exit(1);
  }
  cout << "Finished" << endl;
}
