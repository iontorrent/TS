/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "TraceChunk.h"
#include "H5File.h"
#include "IonErr.h"
using namespace std;

struct FlowChunk {
  size_t CompressionType;
  size_t ChipRow, ChipCol, ChipFrame;
  size_t RowStart, ColStart, FrameStart, FrameStep;
  size_t Height, Width, Depth; // row, col, frame
  size_t OrigFrames;
  int StartDetailedTime, StopDetailedTime, LeftAvg;
  float T0;
  float Sigma;
  float TMidNuc;
  hvl_t Data;
};

TraceChunkSerializer::TraceChunkSerializer() {
  mCompressor = NULL;
  mNumChunks = 0;
  mChunks = NULL;
}

TraceChunkSerializer::~TraceChunkSerializer() {
  if (mChunks != NULL) {
    delete [] mChunks;
  }
}


void TraceChunkSerializer::DecompressFromReading(const struct FlowChunk *chunks, GridMesh<TraceChunk> &dataMesh) {
  for (size_t bIx = 0; bIx < dataMesh.GetNumBin(); bIx++) {
    TraceChunk &tc = dataMesh.GetItem(bIx);
    ION_ASSERT(chunks[bIx].CompressionType == mCompressor->GetCompressionType(), "Wrong compression type.");
    const struct FlowChunk &fc = chunks[bIx];
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
    size_t outsize = fc.Height * fc.Width * fc.Depth;
    tc.mData.resize(outsize);
    mCompressor->Decompress(tc, (int8_t *)fc.Data.p, fc.Data.len);
    outsize = fc.Height * fc.Width * fc.Depth;
  }
}
  
void TraceChunkSerializer::ArrangeDataForWriting(GridMesh<TraceChunk> &dataMesh, struct FlowChunk *chunks) {
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
    int8_t *data = NULL;
    size_t outsize;
    mCompressor->Compress(tc, &data, &outsize);
    fc.Data.p = data;
    fc.Data.len = outsize;
  }
}

void TraceChunkSerializer::Read(const char *filename, GridMesh<TraceChunk> &dataMesh) {
  H5File h5(filename);
  h5.Open(false);
  hid_t dataset = H5Dopen2(h5.GetFileId(), "FlowChunk", H5P_DEFAULT);
  //    hid_t datatype  = H5Dget_type(dataset);     /* datatype handle */
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
  H5Tinsert(fcType, "Data", HOFFSET(struct FlowChunk, Data), charArrayType);
  status = H5Dread(dataset, fcType, H5S_ALL, H5S_ALL, H5P_DEFAULT, mChunks);
  ION_ASSERT(status == 0, "Couldn' read dataset");
  dataMesh.Init(mChunks[0].ChipRow, mChunks[0].ChipCol, mChunks[0].Height, mChunks[0].Width);
  ION_ASSERT(dataMesh.GetNumBin() == mNumChunks, "Didn't get number of chunks expected");
  DecompressFromReading(mChunks, dataMesh);
  status = H5Dvlen_reclaim(fcType, fcDataSpace, H5P_DEFAULT, mChunks);
  delete [] mChunks;
  mChunks = NULL;
  H5Tclose(fcType);
  H5Sclose(fcDataSpace);
  H5Dclose(dataset);
  h5.Close();    
}


void TraceChunkSerializer::Write(const char *filename, GridMesh<TraceChunk> &dataMesh) {
  H5File h5(filename);
  h5.Open(true);

  mNumChunks = dataMesh.GetNumBin();
  mChunks = new struct FlowChunk[mNumChunks];
  ArrangeDataForWriting(dataMesh, mChunks);
  hsize_t dims1[1];
  dims1[0] = mNumChunks;
     
  hid_t fcDataSpace = H5Screate_simple(1, dims1, NULL);
  hid_t charArrayType = H5Tvlen_create (H5T_NATIVE_CHAR);
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
  H5Tinsert(fcType, "Data", HOFFSET(struct FlowChunk, Data), charArrayType);
  hid_t dataset = H5Dcreate2(h5.GetFileId(), "FlowChunk", fcType, fcDataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status = H5Dwrite(dataset, fcType, H5S_ALL, H5S_ALL, H5P_DEFAULT, mChunks);
  status = H5Dvlen_reclaim(fcType, fcDataSpace, H5P_DEFAULT, mChunks);
  delete [] mChunks;
  mChunks = NULL;
  ION_ASSERT(status == 0, "Couldn't write dataset");
  H5Tclose(fcType);
  H5Tclose(charArrayType);
  H5Sclose(fcDataSpace);
  H5Dclose(dataset);
  h5.Close();    
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
  fill(output.begin(), output.end(), 0.0f);
  uint16_t *out = (uint16_t *)(&output.front());
  out[0] = (uint16_t) col;
  out[1] = (uint16_t) row;
  out[2] = (uint16_t) frames;
  size_t count = 0;
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
  fill(chunk.mData.begin(), chunk.mData.end(), 0.0f);
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

void TopCoderCompressor::Compress(TraceChunk &tc, int8_t **compressed, size_t *outsize, size_t *maxSize) {
  vector<int> data;
  ToTopCoder(tc, data);
  TraceChunk test = tc;
  fill(test.mData.begin(), test.mData.end(), 0.0f);
  FromTopCoder(data, test);
  vector<int> dcComp = mCompressor->compress(data);
  if (dcComp.size() > *maxSize) {
    ReallocBuffer(dcComp.size() * sizeof(int), compressed, maxSize);
  }
  *outsize = dcComp.size() * sizeof(int);
  memcpy(*compressed, &dcComp[0], *outsize);
}

void TopCoderCompressor::Decompress(TraceChunk &tc, const int8_t *compressed, size_t size) {
  vector<int> data;
  tc.mData.resize(tc.mHeight * tc.mWidth * tc.mDepth);
  fill(tc.mData.begin(), tc.mData.end(), 0.0f);
  data.resize(ceil(size/(sizeof(int) * 1.0f)));
  fill(data.begin(), data.end(), 0.0f);
  memcpy(&data[0], compressed, size);
  vector<int> ready = mCompressor->decompress(data);
  ION_ASSERT(tc.mHeight = ready[1], "Rows don't match expected.");
  ION_ASSERT(tc.mWidth = ready[0], "Cols don't match expected.");
  ION_ASSERT(tc.mDepth = ready[2], "Frames don't match expected.");  
  FromTopCoder(ready, tc);//.mData, tc.mHeight, tc.mWidth, tc.mDepth);
}

int TopCoderCompressor::GetCompressionType() { return mCompressor->GetCompressionType(); }
