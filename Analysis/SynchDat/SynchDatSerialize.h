/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SYNCHDATSERIALIZE_H
#define SYNCHDATSERIALIZE_H
#include <semaphore.h>
#include "SynchDat.h"

/* shared mem sem.h */
typedef struct sdat_read_semaphore {
    pthread_mutex_t lock;
    pthread_cond_t nonzero;
    unsigned count;
    pthread_t owner;
} sdat_read_semaphore_t;

class H5File;
/** 
 * Interface for classes that take a chunk of data and compress/decompress it.
 */
class TraceCompressor {

public:
  enum CodeType {
    None = 0,
    Delicato = 1,
    LosslessVenco = 2,
    LossyCelebrandil = 3,
    LossySvdDat = 4,
    //    LossySvdDatPlus = 5,
    //    LossySvdDatPlusPlus = 6,
    DeltaComp = 7,
    DeltaCompFst = 8,
    DeltaCompFstSm = 9,
    DeltaCompFstSmX = 9
  };

  virtual ~TraceCompressor() {}

  virtual int GetCompressionType()  = 0;

  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outSize, size_t *maxSize) = 0;

  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) = 0;

  virtual void ReallocBuffer(size_t newSize, int8_t **compressed, size_t *maxSize) {
    ION_ASSERT(newSize > *maxSize, "Shouldn't be resizing");
    if (*compressed != NULL) {
      delete [] (*compressed);
    }
    *compressed = new int8_t[newSize];
    *maxSize = newSize;
  }

  size_t ToIdx(size_t rows, size_t cols, size_t frames, size_t r, size_t c, size_t f) {
    return (r * cols + c) * frames + f;
  }

};

/** 
 * Interface for classes that take a chunk of data and compress/decompress it.
 */
class TraceNoCompress : public TraceCompressor {

public:
  virtual int GetCompressionType() { return 0; }

  virtual void Compress(TraceChunk &chunk, int8_t **compressed, size_t *outSize, size_t *maxSize) {
    size_t row = chunk.mHeight;
    size_t col = chunk.mWidth;
    size_t frames = chunk.mDepth;
    size_t elements = row * col * frames;
    if (elements > *maxSize) { ReallocBuffer(elements * sizeof(short), compressed, maxSize); }
    *outSize = elements*2;
    // memcpy(*compressed, &chunk.mData[0], *outSize);
    uint8_t *c = (uint8_t *) *compressed;
    int16_t *s = &chunk.mData[0];
    for (size_t i = 0; i < elements; i++) {
      *(c++) = (*s) & 0x00FF;
      *(c++) = (*s) >> 8;
      s++;
    }
    /* size_t pos = 0; */
    /* for (size_t i = 0; i < elements; i++) { */
    /*   short s = chunk.mData[i]; */
    /*   (*compressed)[pos++] = s & 0x000F; */
    /*   (*compressed)[pos++] = s >> 8; */
    /* } */
    /* for (size_t i = 0; i < elements; i+=2) { */
    /*   short s = (short) (chunk.At(i/2) + .5); */
    /*   (*compressed)[i] = s & 0x000F; */
    /*   (*compressed)[i+1] = s >> 8; */
    /* } */
  }

  virtual void Decompress(TraceChunk &chunk, const int8_t *compressed, size_t size) {
    size_t row = chunk.mHeight;
    size_t col = chunk.mWidth;
    size_t frames = chunk.mDepth;
    size_t elements = row * col * frames;
    chunk.mData.resize(elements);
    const int8_t *s = compressed;
    int16_t *c = &chunk.mData[0];
    //    memcpy(c, s, elements*2);
    for (size_t i = 0; i < elements; i++) {
      (*c) = (uint8_t)*s;
      s++;
      (*c) |= (((uint8_t)*s) << 8);
      c++;
      s++;
    }
  }
};

/**
 * Compressor class that calls the TopCoder interfaces of 
 * std::vector<int> compress(const std::vector<int> &input);
 * std::vector<int> decompress(const std::vector<int> &compressed);
 */
class TopCoderCompressor : public TraceCompressor {
public:
  TopCoderCompressor();
  virtual ~TopCoderCompressor();
  virtual int GetCompressionType();
  void SetCompressor(DatCompression *dc) { mCompressor = dc; }
  void ToTopCoder(TraceChunk &chunk, std::vector<int> &output);
  void FromTopCoder(const std::vector<int> &input, TraceChunk &chunk);
  virtual void Compress(TraceChunk &tc, int8_t **compressed, size_t *outsize, size_t *maxsize);
  virtual void Decompress(TraceChunk &tc, const int8_t *compressed, size_t size);
private:
  std::vector<int> mData;
  DatCompression *mCompressor;
};

/**
 * Factory for handling the construction of compressors/decompresors
 * so we can make it easy to try different ones as time goes on.
 */
class CompressorFactory {
public:

  ~CompressorFactory();
  static TraceCompressor *MakeCompressor(TraceCompressor::CodeType type);
  
};

/** 
 * Write collections of trace chunks to disk and read them back. 
 */ 
class TraceChunkSerializer {
  
public:  
  TraceChunkSerializer();

  ~TraceChunkSerializer();

  void DecompressFromReading(const struct FlowChunk *chunks, GridMesh<TraceChunk> &dataMesh);
  
  void ArrangeDataForWriting(GridMesh<TraceChunk> &dataMesh, struct FlowChunk *chunks);

  bool Read(H5File &h5, GridMesh<TraceChunk> &dataMesh);
  bool Write(H5File &h5, GridMesh<TraceChunk> &dataMesh);

  bool Read(const char *filename, SynchDat &data);
  bool Write(const char *filename, SynchDat &data);
  void ReadInfo(H5File &h5, SynchDat &sdat);
  int GetCompressionType() { return mCompressor->GetCompressionType(); }
  void SetCompressor(TraceCompressor *compressor) {mCompressor = compressor; }
  // If reckless then will die if file isn't there, otherwise will patiently wait for file to appear
  void SetRecklessAbandon(bool reckless) { mRecklessAbandon = reckless; }
  void SetTimeout ( int _total_timeout,int _retry_interval ) {
    mRetryInterval = _retry_interval;
    mTotalTimeout = _total_timeout;
  }
  void SetUseSemaphore(bool use) { mUseSemaphore = use; }
  static sdat_read_semaphore_t *sdatSemPtr;
  bool mUseSemaphore;
  TraceCompressor *mCompressor;
  struct FlowChunk *mChunks;
  size_t mNumChunks;
  bool mRecklessAbandon; // So named from image class, should we wait to see if a file appears?
  int mRetryInterval;
  int mTotalTimeout;
  bool mDebugMsg;
  size_t computeMicroSec;
  size_t ioMicroSec;
  size_t openMicroSec;
  size_t compressMicroSec;
};

#endif // SYNCHDATSERIALIZE_H
