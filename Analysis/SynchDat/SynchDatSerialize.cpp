/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TraceChunk.h"
#include "H5File.h"
#include "IonErr.h"
#include "SynchDatSerialize.h"
#include "VencoLossless.h"
#include "SvdDatCompress.h"
//#include "SvdDatCompressPlus.h"
#include "FlowChunk.h"
#include "Image.h"
#include "DeltaComp.h" 
#include "DeltaCompFst.h" 
#include "DeltaCompFstSmX.h" 
#include "IonImageSem.h"
#define INFO_KEYS "info_keys"
#define INFO_VALUES "info_values"
#define SYNCHDATSERIALIZE_VERSION_KEY "sdat_version"
#define SYNCHDATSERIALIZE_VERSION_VALUE "2.0"

//#include "SvdDatCompressPlusPlus.h"
ion_semaphore_t *sdatSemPtr;

using namespace std;

TraceChunkSerializer::TraceChunkSerializer() {
  mCompressor = NULL;
  mNumChunks = 0;
  mChunks = NULL;
  mRecklessAbandon = false;
  mRetryInterval = 15;  // 15 seconds wait time.
  mTotalTimeout = 100;  // 100 seconds before giving up.
  mDebugMsg = false;
  mUseSemaphore = false;
  computeMicroSec = ioMicroSec = openMicroSec = compressMicroSec = 0;
}

TraceChunkSerializer::~TraceChunkSerializer() {
  if (mChunks != NULL) {
    delete [] mChunks;
  }
  if (mCompressor != NULL) {
    delete mCompressor;
    mCompressor = NULL;
  }
}

void TraceChunkSerializer::DecompressFromReading(const struct FlowChunk *chunks, GridMesh<TraceChunk> &dataMesh) {
  compressMicroSec = 0;
  for (size_t bIx = 0; bIx < dataMesh.GetNumBin(); bIx++) {
    TraceChunk &tc = dataMesh.GetItem(bIx);
    const struct FlowChunk &fc = chunks[bIx];
    if (mCompressor == NULL) {
      if (mDebugMsg) { cout << "Got compression type: " << chunks[bIx].CompressionType << endl;}
      mCompressor = CompressorFactory::MakeCompressor((TraceCompressor::CodeType)chunks[bIx].CompressionType);
    }
    ION_ASSERT(chunks[bIx].CompressionType == (size_t)mCompressor->GetCompressionType(), "Wrong compression type: " + ToStr(chunks[bIx].CompressionType) + " vs: " + ToStr(mCompressor->GetCompressionType()));
    tc.mRowStart = fc.RowStart;
    tc.mColStart = fc.ColStart;
    tc.mFrameStart = fc.FrameStart;
    tc.mFrameStep = fc.FrameStep;
    tc.mChipRow = fc.ChipRow;
    tc.mChipCol = fc.ChipCol;
    tc.mChipFrame = fc.ChipFrame;
    tc.mStartDetailedTime = fc.StartDetailedTime;
    tc.mStopDetailedTime = fc.StopDetailedTime;
    tc.mLeftAvg = fc.LeftAvg;
    tc.mOrigFrames = fc.OrigFrames;
    tc.mT0 = fc.T0;
    tc.mSigma = fc.Sigma;
    tc.mTMidNuc = fc.TMidNuc;
    tc.mHeight = fc.Height;
    tc.mWidth = fc.Width;
    tc.mDepth = fc.Depth;
    tc.mBaseFrameRate = fc.BaseFrameRate;
    size_t outsize = fc.Height * fc.Width * fc.Depth;
    tc.mData.resize(outsize);
    tc.mTimePoints.resize(tc.mDepth);
    float * tmp = (float *)fc.DeltaFrame.p;
    copy(tmp,tmp+fc.Depth, tc.mTimePoints.begin());
    ClockTimer timer;
    mCompressor->Decompress(tc, (int8_t *)fc.Data.p, fc.Data.len);
    compressMicroSec += timer.GetMicroSec();
    outsize = fc.Height * fc.Width * fc.Depth;
  }
}
  
void TraceChunkSerializer::SetCompressor(TraceCompressor *compressor) {
  if (mCompressor != NULL) {
    delete mCompressor;
    mCompressor = NULL;
  }
  mCompressor = compressor; 
}

void TraceChunkSerializer::ArrangeDataForWriting(GridMesh<TraceChunk> &dataMesh, struct FlowChunk *chunks) {

  if (dataMesh.GetNumBin() == 0) {
    return;
  }
  size_t maxSize = dataMesh.mBins[0].mDepth * dataMesh.mBins[0].mHeight * dataMesh.mBins[1].mWidth * 3;
  int8_t *compressed = new int8_t[maxSize];
  compressMicroSec = 0;
  for (size_t bIx = 0; bIx < dataMesh.GetNumBin(); bIx++) {
    TraceChunk &tc = dataMesh.GetItem(bIx);
    struct FlowChunk &fc = chunks[bIx];
    fc.CompressionType = mCompressor->GetCompressionType();
    fc.RowStart = tc.mRowStart;
    fc.ColStart = tc.mColStart;
    fc.FrameStart = tc.mFrameStart;
    fc.FrameStep = tc.mFrameStep;
    fc.ChipRow = tc.mChipRow;
    fc.ChipCol = tc.mChipCol;
    fc.ChipFrame = tc.mChipFrame;
    fc.StartDetailedTime = tc.mStartDetailedTime;
    fc.StopDetailedTime = tc.mStopDetailedTime;
    fc.LeftAvg = tc.mLeftAvg;
    fc.OrigFrames = tc.mOrigFrames;
    fc.T0 = tc.mT0;
    fc.Sigma = tc.mSigma;
    fc.TMidNuc = tc.mTMidNuc;
    fc.Height = tc.mHeight;
    fc.Width = tc.mWidth;
    fc.Depth = tc.mDepth;
    fc.BaseFrameRate = tc.mBaseFrameRate;
    size_t outsize;
    ClockTimer timer;
    mCompressor->Compress(tc, &compressed, &outsize, &maxSize);
    compressMicroSec += timer.GetMicroSec();
    //cout <<"Doing: " << fc.CompressionType << " Bytes per wells: " << outsize/(float) (tc.mHeight * tc.mWidth) <<" Compression ratio: "<< tc.mData.size()*2/(float)outsize << endl;
    fc.Data.p = (int8_t *)malloc(outsize*sizeof(int8_t));
    memcpy(fc.Data.p, compressed, outsize*sizeof(int8_t));
    fc.Data.len = outsize;
    if (0 == outsize) {
      cout << "How can there be zero blocks." << endl;
    }
    float * tmp = (float *)malloc(tc.mTimePoints.size() * sizeof(float));
    copy(tc.mTimePoints.begin(), tc.mTimePoints.end(), tmp);
    fc.DeltaFrame.p = tmp;
    fc.DeltaFrame.len = tc.mTimePoints.size() * sizeof(float);
  }
  delete [] compressed;
}


void InitSdatReadSem(void) {
  sdatSemPtr = NULL;
}


bool TraceChunkSerializer::Read(H5File &h5, GridMesh<TraceChunk> &dataMesh) {
  hid_t dataset = H5Dopen2(h5.GetFileId(), "FlowChunk", H5P_DEFAULT);
  //    hid_t datatype  = H5Dget_type(dataset);     /* datatype handle */
  static pthread_once_t onceControl = PTHREAD_ONCE_INIT;
  int err = pthread_once(&onceControl, InitSdatReadSem);
  if (err != 0) { cout << "Error with pthread once." << endl; }               
  hid_t dataspace = H5Dget_space(dataset);
  int rank = H5Sget_simple_extent_ndims(dataspace);
  std::vector<hsize_t> dims;
  dims.resize(rank);
  int status = H5Sget_simple_extent_dims(dataspace, &dims[0], NULL);
  if (mChunks != NULL) {
    delete [] mChunks;
  }
  mChunks = new FlowChunk[dims[0]];
  mNumChunks = dims[0];
  hid_t fcDataSpace = H5Screate_simple(1, &dims[0], NULL);
  hid_t charArrayType = H5Tvlen_create (H5T_NATIVE_CHAR);
  hid_t charArrayType2 = H5Tvlen_create (H5T_NATIVE_CHAR);
  hid_t fcType = H5Tcreate(H5T_COMPOUND, sizeof(struct FlowChunk));
  H5Tinsert(fcType, "CompressionType", HOFFSET(struct FlowChunk, CompressionType), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ChipRow", HOFFSET(struct FlowChunk, ChipRow), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ChipCol", HOFFSET(struct FlowChunk, ChipCol), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ChipFrame", HOFFSET(struct FlowChunk, ChipFrame), H5T_NATIVE_B64);
  H5Tinsert(fcType, "RowStart", HOFFSET(struct FlowChunk, RowStart), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ColStart", HOFFSET(struct FlowChunk, ColStart), H5T_NATIVE_B64);
  H5Tinsert(fcType, "FrameStart", HOFFSET(struct FlowChunk, FrameStart), H5T_NATIVE_B64);
  H5Tinsert(fcType, "FrameStep", HOFFSET(struct FlowChunk, FrameStep), H5T_NATIVE_B64);
  H5Tinsert(fcType, "Height", HOFFSET(struct FlowChunk, Height), H5T_NATIVE_B64);
  H5Tinsert(fcType, "Width", HOFFSET(struct FlowChunk, Width), H5T_NATIVE_B64);
  H5Tinsert(fcType, "Depth", HOFFSET(struct FlowChunk, Depth), H5T_NATIVE_B64);
  H5Tinsert(fcType, "OrigFrames", HOFFSET(struct FlowChunk, OrigFrames), H5T_NATIVE_B64);
  H5Tinsert(fcType, "StartDetailedTime", HOFFSET(struct FlowChunk, StartDetailedTime), H5T_NATIVE_INT);
  H5Tinsert(fcType, "StopDetailedTime", HOFFSET(struct FlowChunk, StopDetailedTime), H5T_NATIVE_INT);
  H5Tinsert(fcType, "LeftAvg", HOFFSET(struct FlowChunk, LeftAvg), H5T_NATIVE_INT);
  H5Tinsert(fcType, "T0", HOFFSET(struct FlowChunk, T0), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "Sigma", HOFFSET(struct FlowChunk, Sigma), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "TMidNuc", HOFFSET(struct FlowChunk, TMidNuc), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "BaseFrameRate", HOFFSET(struct FlowChunk, BaseFrameRate), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "DeltaFrame", HOFFSET(struct FlowChunk, DeltaFrame), charArrayType2);
  H5Tinsert(fcType, "Data", HOFFSET(struct FlowChunk, Data), charArrayType);
  ClockTimer timer;
  IonImageSem::Take();
  status = H5Dread(dataset, fcType, H5S_ALL, H5S_ALL, H5P_DEFAULT, mChunks);
  IonImageSem::Give();
  ioMicroSec = timer.GetMicroSec();
  ION_ASSERT(status == 0, "Couldn' read dataset");
  timer.StartTimer();
  dataMesh.Init(mChunks[0].ChipRow, mChunks[0].ChipCol, mChunks[0].Height, mChunks[0].Width);
  ION_ASSERT(dataMesh.GetNumBin() == mNumChunks, "Didn't get number of chunks expected");
  DecompressFromReading(mChunks, dataMesh);
  computeMicroSec = timer.GetMicroSec();
  timer.StartTimer();
  status = H5Dvlen_reclaim(fcType, fcDataSpace, H5P_DEFAULT, mChunks);
  delete [] mChunks;
  mChunks = NULL;
  H5Tclose(fcType);
  H5Tclose(charArrayType);
  H5Tclose(charArrayType2);
  H5Sclose(fcDataSpace);
  H5Dclose(dataset);
  ioMicroSec += timer.GetMicroSec();
  return status == 0;
}

bool TraceChunkSerializer::Write(H5File &h5, GridMesh<TraceChunk> &dataMesh) {

  mNumChunks = dataMesh.GetNumBin();
  ClockTimer timer;
  mChunks = (struct FlowChunk *) malloc(sizeof(struct FlowChunk) * mNumChunks);
  ArrangeDataForWriting(dataMesh, mChunks);
  computeMicroSec = timer.GetMicroSec();
  hsize_t dims1[1];
  dims1[0] = mNumChunks;
  hid_t fcDataSpace = H5Screate_simple(1, dims1, NULL);
  hid_t charArrayType = H5Tvlen_create (H5T_NATIVE_CHAR);
  hid_t charArrayType2 = H5Tvlen_create (H5T_NATIVE_CHAR);
  hid_t fcType = H5Tcreate(H5T_COMPOUND, sizeof(struct FlowChunk));
  H5Tinsert(fcType, "CompressionType", HOFFSET(struct FlowChunk, CompressionType), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ChipRow", HOFFSET(struct FlowChunk, ChipRow), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ChipCol", HOFFSET(struct FlowChunk, ChipCol), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ChipFrame", HOFFSET(struct FlowChunk, ChipFrame), H5T_NATIVE_B64);
  H5Tinsert(fcType, "RowStart", HOFFSET(struct FlowChunk, RowStart), H5T_NATIVE_B64);
  H5Tinsert(fcType, "ColStart", HOFFSET(struct FlowChunk, ColStart), H5T_NATIVE_B64);
  H5Tinsert(fcType, "FrameStart", HOFFSET(struct FlowChunk, FrameStart), H5T_NATIVE_B64);
  H5Tinsert(fcType, "FrameStep", HOFFSET(struct FlowChunk, FrameStep), H5T_NATIVE_B64);
  H5Tinsert(fcType, "Height", HOFFSET(struct FlowChunk, Height), H5T_NATIVE_B64);
  H5Tinsert(fcType, "Width", HOFFSET(struct FlowChunk, Width), H5T_NATIVE_B64);
  H5Tinsert(fcType, "Depth", HOFFSET(struct FlowChunk, Depth), H5T_NATIVE_B64);
  H5Tinsert(fcType, "OrigFrames", HOFFSET(struct FlowChunk, OrigFrames), H5T_NATIVE_B64);
  H5Tinsert(fcType, "StartDetailedTime", HOFFSET(struct FlowChunk, StartDetailedTime), H5T_NATIVE_INT);
  H5Tinsert(fcType, "StopDetailedTime", HOFFSET(struct FlowChunk, StopDetailedTime), H5T_NATIVE_INT);
  H5Tinsert(fcType, "LeftAvg", HOFFSET(struct FlowChunk, LeftAvg), H5T_NATIVE_INT);
  H5Tinsert(fcType, "T0", HOFFSET(struct FlowChunk, T0), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "Sigma", HOFFSET(struct FlowChunk, Sigma), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "TMidNuc", HOFFSET(struct FlowChunk, TMidNuc), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "BaseFrameRate", HOFFSET(struct FlowChunk, BaseFrameRate), H5T_NATIVE_FLOAT);
  H5Tinsert(fcType, "DeltaFrame", HOFFSET(struct FlowChunk, DeltaFrame), charArrayType2);
  H5Tinsert(fcType, "Data", HOFFSET(struct FlowChunk, Data), charArrayType);
  timer.StartTimer();
  hid_t dataset = H5Dcreate2(h5.GetFileId(), "FlowChunk", fcType, fcDataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status = H5Dwrite(dataset, fcType, H5S_ALL, H5S_ALL, H5P_DEFAULT, mChunks);
  status = H5Dvlen_reclaim(fcType, fcDataSpace, H5P_DEFAULT, mChunks);
  //  //  delete [] mChunks;
  free(mChunks);
  mChunks = NULL;
  ION_ASSERT(status == 0, "Couldn't write dataset");
  H5Tclose(fcType);
  H5Tclose(charArrayType);
  H5Tclose(charArrayType2);
  H5Sclose(fcDataSpace);
  H5Dclose(dataset);
  ioMicroSec += timer.GetMicroSec();
  return status == 0;
}

bool TraceChunkSerializer::Read(const char *filename, SynchDat &data) {
  data.Clear();
  if (!H5File::IsH5File(filename)) {
    return false;
  }
  bool result = true;
  try {
    if (!mRecklessAbandon) {
      uint32_t waitTime = mRetryInterval;
      int32_t timeOut = mTotalTimeout;
      //--- Wait up to 3600 seconds for a file to be available
      while ( timeOut > 0 )
        {
          //--- Is the file we want available?
          if ( Image::ReadyToLoad ( filename ) ) {
            break;
          }
          //DEBUG
          fprintf ( stdout, "Waiting to load %s\n", filename );
          sleep ( waitTime );
          timeOut -= waitTime;
        }
    }
    /* Turn off error printing as we're not sure this is actually an hdf5 file. */
    H5File h5(filename);
    result = h5.OpenForReading();

    if (result) {
      h5.SetReadOnly(true);
      result &= Read(h5, data.GetMesh());
      ReadInfo(h5, data);
    }
    h5.Close();
  }
  catch (...) {
    result = false;
  }
  return result;
}

void TraceChunkSerializer::ReadInfo(H5File &h5, SynchDat &sdat) {
  sdat.mInfo.Clear();
  vector<string> keys;
  vector<string> values;
  h5.ReadStringVector(INFO_KEYS, keys);
  h5.ReadStringVector(INFO_VALUES, values);
  if (keys.size() != values.size()) { ION_ABORT("Keys and Values don't match in size."); }
    
  for (size_t i = 0; i < keys.size(); i++) {
    if (!sdat.mInfo.SetValue(keys[i], values[i])) { 
      ION_ABORT("Error: Could not set key: " + keys[i] + " with value: " + values[i]); 
    }
  }
}

bool TraceChunkSerializer::Write(const char *filename, SynchDat &data) {
  data.mInfo.SetValue(SYNCHDATSERIALIZE_VERSION_KEY, SYNCHDATSERIALIZE_VERSION_VALUE);
  int numEntries = data.mInfo.GetCount();
  vector<string> keys(numEntries);
  vector<string> values(numEntries);
  vector<const char *> keysBuff(numEntries);
  vector<const char *> valuesBuff(numEntries);
  for (int i = 0; i < numEntries; i++) {
    data.mInfo.GetEntry(i, keys[i], values[i]);
    keysBuff[i] = keys[i].c_str();
    valuesBuff[i] = values[i].c_str();
  }
  ClockTimer timer;
  H5File h5(filename);
  bool ok = h5.OpenNew();
  openMicroSec = timer.GetMicroSec();
  if (!ok) { return ok; }
  timer.StartTimer();
  h5.WriteStringVector(INFO_KEYS, &keysBuff[0], keysBuff.size());
  h5.WriteStringVector(INFO_VALUES, &valuesBuff[0], keysBuff.size());
  ioMicroSec = timer.GetMicroSec();
  ok =  Write(h5, data.GetMesh());
  timer.StartTimer();
  h5.Close();
  H5garbage_collect();
  ioMicroSec += timer.GetMicroSec();
  return ok;
}

TraceCompressor *CompressorFactory::MakeCompressor(TraceCompressor::CodeType type) {
  if (type == TraceCompressor::None) { return new TraceNoCompress(); }
  if (type == TraceCompressor::LosslessVenco) { return new VencoLossless(); }
  if (type == TraceCompressor::LossySvdDat) { return new SvdDatCompress(); }
  //if (type == TraceCompressor::LossySvdDatPlus) { return new SvdDatCompressPlus(); }
  //  if (type == TraceCompressor::LossySvdDatPlusPlus) { return new SvdDatCompressPlusPlus(); }
  if (type == TraceCompressor::DeltaComp) { return new DeltaComp(); }
  if (type == TraceCompressor::DeltaCompFst) { return new DeltaCompFst(); }
  if (type == TraceCompressor::DeltaCompFstSmX) { return new DeltaCompFstSmX(); }
  ION_ABORT("Unrecognized compression type: " + ToStr(type));
  return NULL; 
}

TopCoderCompressor::TopCoderCompressor() {
  mCompressor = NULL;
}

TopCoderCompressor::~TopCoderCompressor() {}

void TopCoderCompressor::ToTopCoder(TraceChunk &chunk, std::vector<int> &output) {
  size_t row = chunk.mHeight;
  size_t col = chunk.mWidth;
  size_t frames = chunk.mDepth;
  size_t size= row * col * frames + 3;
  if (size % 2 != 0) {
    size++;
  }
  int offset = 3;
  output.resize(size);
  std::fill(output.begin(), output.end(), 0.0f);
  uint16_t *out = (uint16_t *)(&output.front());
  out[0] = (uint16_t) col;
  out[1] = (uint16_t) row;
  out[2] = (uint16_t) frames;
  for (size_t rIx = 0; rIx < row; rIx++) {
    for (size_t cIx = 0; cIx < col; cIx++) {
      for (size_t fIx = 0; fIx < frames; fIx++) {
        size_t idx = ToIdx(row, col, frames, rIx, cIx, fIx) + offset;
        uint16_t x = round(chunk.At(rIx+chunk.mRowStart, cIx+chunk.mColStart, fIx+chunk.mFrameStart));
        out[idx] = x;
      }
    }
  }
}

void TopCoderCompressor::FromTopCoder(const std::vector<int> &input, TraceChunk &chunk) {
  uint16_t *in = (uint16_t *)(&input.front());
  size_t col = in[0];
  size_t row = in[1];
  size_t frames = in[2];
  int offset = 3;
  ION_ASSERT(chunk.mHeight = row, "Rows don't match expected.");
  ION_ASSERT(chunk.mWidth = col, "Cols don't match expected.");
  ION_ASSERT(chunk.mDepth = frames, "Frames don't match expected.");
  chunk.mData.resize(row * col * frames);
  std::fill(chunk.mData.begin(), chunk.mData.end(), 0.0f);
  for (size_t rIx = 0; rIx < row; rIx++) {
    for (size_t cIx = 0; cIx < col; cIx++) {
      for (size_t fIx = 0; fIx < frames; fIx++) {
        size_t idx = ToIdx(row, col, frames, rIx, cIx, fIx);
        uint16_t x = in[idx + offset];
        chunk.At(rIx + chunk.mRowStart, cIx + chunk.mColStart, fIx + chunk.mFrameStart) = x;
      }
    }
  }
}

void TopCoderCompressor::Compress(TraceChunk &tc, int8_t **compressed, size_t *outsize, size_t *maxsize) {
  vector<int> data;
  ToTopCoder(tc, data);
  TraceChunk test = tc;
  std::fill(test.mData.begin(), test.mData.end(), 0.0f);
  FromTopCoder(data, test);
  vector<int> dcComp = mCompressor->compress(data);
  if(*compressed != NULL) {
    free(*compressed);
  }
  *compressed = (int8_t *) malloc(dcComp.size() * sizeof(int));
  *outsize = dcComp.size() * sizeof(int);
  memcpy(*compressed, &dcComp[0], *outsize);
}

void TopCoderCompressor::Decompress(TraceChunk &tc, const int8_t *compressed, size_t size) {
  vector<int> data;
  tc.mData.resize(tc.mHeight * tc.mWidth * tc.mDepth);
  std::fill(tc.mData.begin(), tc.mData.end(), 0.0f);
  data.resize(ceil(size/(sizeof(int) * 1.0f)));
  std::fill(data.begin(), data.end(), 0.0f);
  memcpy(&data[0], compressed, size);
  vector<int> ready = mCompressor->decompress(data);
  ION_ASSERT(tc.mHeight = ready[1], "Rows don't match expected.");
  ION_ASSERT(tc.mWidth = ready[0], "Cols don't match expected.");
  ION_ASSERT(tc.mDepth = ready[2], "Frames don't match expected.");  
  FromTopCoder(ready, tc);//.mData, tc.mHeight, tc.mWidth, tc.mDepth);
}

int TopCoderCompressor::GetCompressionType() { return mCompressor->GetCompressionType(); }
