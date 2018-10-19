/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cinttypes>
#include "RawWells.h"
#include "Mask.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include "Utils.h"
#include "LinuxCompat.h"
#include "IonErr.h"
#include "RawWellsV1.h"
#define WELLS "wells"
#define RANKS "ranks"
#define RESERROR "reserrors"
#define INFO_KEYS "info_keys"
#define INFO_VALUES "info_values"
#define FLOW_ORDER "FLOW_ORDER"

using namespace std;
#define RAWWELLS_VERSION_KEY "RAWWELLS_VERSION"
#define RAWWELLS_VERSION_VALUE "2"


RWH5DataSet::RWH5DataSet()
{
  Clear();
}

void RWH5DataSet::Clear()
{
  mGroup = "/";
  mName = "/";
  mDataset = EMPTY;
  mDatatype = EMPTY;
  mDataspace = EMPTY;
  mSaveAsUShort = false;
  mLower = -5.0;
  mUpper = 28.0;
}

RWH5DataSet::~RWH5DataSet()
{
  Close();
}

void RWH5DataSet::Close()
{
  if ( mDataset != EMPTY )
  {
    H5Sclose ( mDataspace );
    mDataspace = EMPTY;
    H5Tclose ( mDatatype );
    mDatatype = EMPTY;
    H5Dclose ( mDataset );
    mDataset = EMPTY;
  }
}

int RawWellsWriter::WriteWellsData(RWH5DataSet &dataSet, WellChunk &chunk, float *data) {
  hsize_t count[3];       /* size of the hyperslab in the file */
  hsize_t offset[3];      /* hyperslab offset in the file */
  hsize_t count_out[3];   /* size of the hyperslab in memory */
  hsize_t offset_out[3];  /* hyperslab offset in memory */
  hsize_t dimsm[3];       /* memory space dimensions */
  int error = 0;
  /* file offsets. */
  offset[0] = chunk.rowStart;
  offset[1] = chunk.colStart;
  offset[2] = chunk.flowStart;
  count[0] = chunk.rowHeight;
  count[1] = chunk.colWidth;
  count[2] = chunk.flowDepth;
  /* Select correct blocks (hyperslab) of hdf5 file. */
  herr_t status = 0;
  status = H5Sselect_hyperslab ( dataSet.mDataspace, H5S_SELECT_SET, offset, NULL, count, NULL );
  if ( status < 0 ) {
	  ION_WARN ( "Couldn't read wells dataspace for: " + dataSet.mName + ToStr ( chunk.rowStart ) + "," + ToStr ( chunk.colStart ) + "," +
                ToStr ( chunk.rowHeight ) + "," + ToStr ( chunk.colWidth ) + " x " + ToStr ( chunk.flowDepth ) );
	  return 1;
  }
  hid_t memspace;
  /* Define the memory dataspace.  */
  dimsm[0] = count[0];
  dimsm[1] = count[1];
  dimsm[2] = count[2];
  memspace = H5Screate_simple ( 3, dimsm, NULL );

  offset_out[0] = 0;
  offset_out[1] = 0;
  offset_out[2] = 0;
  count_out[0] = chunk.rowHeight;
  count_out[1] = chunk.colWidth;
  count_out[2] = chunk.flowDepth;
  status = H5Sselect_hyperslab ( memspace, H5S_SELECT_SET, offset_out, NULL,
                                         count_out, NULL );
  if ( status < 0 ) {
    ION_WARN ( "Couldn't read wells hyperslab for: " +
                ToStr ( chunk.rowStart ) + "," + ToStr ( chunk.colStart ) + "," +
                ToStr ( chunk.rowHeight ) + "," + ToStr ( chunk.colWidth ) + " x " +
                ToStr ( chunk.flowDepth ) );
    return 1;
  }

  status = -1;
  if(dataSet.mSaveAsUShort)
  {
    WellsConverter converter(dataSet.mLower, dataSet.mUpper);
    size_t dataSize = chunk.rowHeight * chunk.colWidth * chunk.flowDepth;
    unsigned short* buffer = new unsigned short[dataSize];
    unsigned short* buffer2 = buffer;
    float* data2 = data;
    for(size_t i = 0; i < dataSize; ++i, ++buffer2, ++data2)
    {
      *buffer2 = converter.FloatToUInt16(*data2);
	}
    status = H5Dwrite ( dataSet.mDataset, H5T_NATIVE_USHORT, memspace, dataSet.mDataspace, H5P_DEFAULT, buffer);
    delete [] buffer;
    buffer = NULL;
  }
  else
  {
    status = H5Dwrite ( dataSet.mDataset, H5T_NATIVE_FLOAT, memspace, dataSet.mDataspace, H5P_DEFAULT, data);
  }

  if ( status < 0 ) {
    ION_WARN ( "ERROR - Unsuccessful write to file: " +
                ToStr ( chunk.rowStart ) + "," + ToStr ( chunk.colStart ) + "," +
                ToStr ( chunk.rowHeight ) + "," + ToStr ( chunk.colWidth ) + " x " +
                ToStr ( chunk.flowStart ) + "," + ToStr ( chunk.flowDepth ) + "\t" + dataSet.mName );
    return 1;
  }
  H5Sclose(memspace);
  return error;
}

ChunkFlowData::ChunkFlowData()
{
  indexes = NULL;
  flowData = NULL;
  dsBuffer = NULL;
  spaceSize = 0;
  numFlows = 0;
  bufferSize = 0;
  lastFlow = false;
}

ChunkFlowData::ChunkFlowData(unsigned int colXrow, unsigned int flows, unsigned int bufSize)
{
  indexes = new int32_t[colXrow];
  flowData = new float[colXrow * flows];
  dsBuffer = new float[bufSize];
  spaceSize = colXrow;
  numFlows = flows;
  bufferSize = bufSize;
  lastFlow = false;
}

ChunkFlowData::~ChunkFlowData() 
{
  if(indexes) 
  {
    delete [] indexes;
    indexes = NULL;
  }
  if(flowData) 
  {
    delete [] flowData;
    flowData = NULL;
  }
  if(dsBuffer) 
  {
    delete [] dsBuffer;
    dsBuffer = NULL;
  }
}

void ChunkFlowData::clearBuffer() 
{
  if(dsBuffer) 
  {
    memset(dsBuffer, 0, bufferSize * sizeof(float));
  }
}

SemQueue::SemQueue() 
{
  pthread_mutex_init(&mMutex, NULL);
  sem_init(&mSemout, 0, 0);
  mMaxSize = 0;
}

SemQueue::SemQueue(unsigned int maxSize) 
{
  pthread_mutex_init(&mMutex, NULL);
  sem_init(&mSemin, 0, maxSize);
  sem_init(&mSemout, 0, 0);
  mMaxSize = maxSize;
}

SemQueue::~SemQueue() 
{
  pthread_mutex_destroy(&mMutex);
  sem_destroy(&mSemin);
  sem_destroy(&mSemout);
}

void SemQueue::init(unsigned int maxSize) 
{
  sem_init(&mSemin, 0, maxSize);
  mMaxSize = maxSize;
}

size_t SemQueue::size() 
{
  size_t sz = 0;
  pthread_mutex_lock(&mMutex);
  sz = mQueue.size();
  pthread_mutex_unlock(&mMutex);

  return sz;
}

void SemQueue::clear() 
{
  while(!mQueue.empty()) 
  {
    ChunkFlowData* item = mQueue.front();
    mQueue.pop();

    delete item;
    item = NULL;
  }
}

void SemQueue::enQueue(ChunkFlowData* item) 
{
  sem_wait(&mSemin);

  int qs;
  pthread_mutex_lock(&mMutex);
  qs = mQueue.size();
  pthread_mutex_unlock(&mMutex);

  while(qs >= mMaxSize) 
  {
    usleep(10);

    pthread_mutex_lock(&mMutex);
    qs = mQueue.size();
    pthread_mutex_unlock(&mMutex);
  }

  pthread_mutex_lock(&mMutex);
  mQueue.push(item);
  pthread_mutex_unlock(&mMutex);

  sem_post(&mSemout);
}

ChunkFlowData* SemQueue::deQueue() 
{
  ChunkFlowData* item = NULL;

  sem_wait(&mSemout);

  pthread_mutex_lock(&mMutex);
  if(!mQueue.empty()) 
  {
    item = mQueue.front();
    mQueue.pop();
  }
  pthread_mutex_unlock(&mMutex);

  sem_post(&mSemin);

  return item;
}

RawWells::RawWells ( const char *experimentPath, const char *rawWellsName, int rows, int cols )
{
  mSaveAsUShort = false;
  Init ( experimentPath, rawWellsName, rows, cols, 0 );
}

RawWells::RawWells ( const char *experimentPath, const char *rawWellsName )
{
  mSaveAsUShort = false;
  Init ( experimentPath, rawWellsName, 0, 0, 0 );
}

RawWells::RawWells ( const char *wellsFilePath, int rows, int cols )
{
  mSaveAsUShort = false;
  Init ( "", wellsFilePath, rows, cols, 0 );
}

RawWells::RawWells ( const char *experimentPath, const char *rawWellsName, bool saveAsUShort, float lower, float upper )
{
  mSaveAsUShort = saveAsUShort;
  mLower = lower;
  mUpper = upper;
  Init ( experimentPath, rawWellsName, 0, 0, 0 );
}

RawWells::~RawWells()
{
  Close();
  // JZ add for multithreading
  CleanupHdf5();
}

void RawWells::Init ( const char *experimentPath, const char *rawWellsName, int rows, int cols, int flows )
{
  mSaveCopies = false;
  mSaveRes = false;
  mConvertWithCopies = true;
  mIsLegacy = false;
  WELL_NOT_LOADED = -1;
  WELL_NOT_SUBSET = -2;
  mWellChunkSizeRow  = 50;
  mWellChunkSizeCol  = 50;
  mWellChunkSizeFlow = 60;
  mStepSize = 100;
  mCurrentRegionRow = mCurrentRegionCol = 0;
  mCurrentRow = 0;
  mCurrentCol = 0;
  mFirsttimeGetRegionData = true;
  mDirectory = experimentPath;
  mFileLeaf = rawWellsName;
  if ( mDirectory != "" )
  {
    mFilePath = mDirectory + "/" + mFileLeaf;
  }
  else
  {
    mFilePath = mFileLeaf;
  }
  mCurrentWell = 0;
  mCompression = 3;
  SetRows ( rows );
  SetCols ( cols );
  SetFlows ( flows );
  data.flowValues = NULL;
  //  mWriteOnClose = false;
  mHFile = RWH5DataSet::EMPTY;
  mInfo.SetValue ( RAWWELLS_VERSION_KEY, RAWWELLS_VERSION_VALUE );

}

void RawWells::CreateEmpty ( int numFlows, const char *flowOrder, int rows, int cols )
{
  SetRows ( rows );
  SetCols ( cols );
  CreateEmpty ( numFlows, flowOrder );
  //  mWriteOnClose = true;
}

void RawWells::CreateEmpty ( int numFlows, const char *flowOrder )
{
  SetFlows ( numFlows );
  SetFlowOrder ( flowOrder );
  CreateBuffers();
  //  mWriteOnClose = true;
}

void RawWells::SetRegion ( int rowStart, int height, int colStart, int width )
{
  SetChunk ( rowStart, height, colStart, width, 0, NumFlows() );
}

void RawWells::OpenForWrite()
{
  //  mWriteOnClose = true;
  CreateBuffers();
  if ( mHFile == RWH5DataSet::EMPTY )
  {
    mHFile = H5Fcreate ( mFilePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
  }
  if ( mHFile < 0 )
  {
    ION_ABORT ( "Couldn't create file: " + mFilePath );
  }
  OpenWellsForWrite();
  //OpenResForWrite();
}

void RawWells::WriteFlowgram ( size_t flow, size_t x, size_t y, float val )
{
  uint64_t idx = ToIndex ( x,y );
  Set ( idx, flow, val );
}

void RawWells::WriteFlowgram(size_t flow, size_t x, size_t y, float val, float copies)
{
  uint64_t idx = ToIndex ( x,y );
  Set ( idx, flow, val );

  if(mSaveCopies && mWellsCopies.size() > 0 && idx < mWellsCopies.size())
  {
    if(mWellsCopies[idx] < 0)
    {
      mWellsCopies[idx] = copies;
    }
  }
}

void RawWells::WriteFlowgram(size_t flow, size_t x, size_t y, float val, float copies, float resError)
{
  uint64_t idx = ToIndex ( x,y );
  Set ( idx, flow, val );
  SetRes ( idx, flow, resError );

  if(mSaveCopies && mWellsCopies.size() > 0 && idx < mWellsCopies.size())
  {
    if(mWellsCopies[idx] < 0)
    {
      mWellsCopies[idx] = copies;
    }
  }
}

void RawWells::WriteFlowgramWithRes ( size_t flow, size_t x, size_t y, float val, float resError )
{
  uint64_t idx = ToIndex ( x,y );
  Set ( idx, flow, val );
  SetRes ( idx, flow, resError );
}

bool RawWells::OpenForReadLegacy()
{
  //  mWriteOnClose = false;
  RawWellsV1 orig ( mDirectory.c_str(), mFileLeaf.c_str() );
  bool failed = orig.OpenForRead ( false );
  if ( failed )
  {
    return failed;
  }
  WellData o;
  o.flowValues = ( float * ) malloc ( orig.NumFlows() * sizeof ( float ) );
  mFlowOrder = orig.FlowOrder();
  if ( !mInfo.SetValue ( FLOW_ORDER, mFlowOrder ) )
  {
    ION_ABORT ( "Error - Could not set flow order." );
  }
  mFlowData.resize ( ( uint64_t ) orig.NumWells() * orig.NumFlows() );
  fill ( mFlowData.begin(), mFlowData.end(), 0.0f );
  mIndexes.resize ( orig.NumWells() );
  fill ( mIndexes.begin(), mIndexes.end(), WELL_NOT_LOADED );
  mRankData.resize ( orig.NumWells() );
  fill ( mRankData.begin(), mRankData.end(), 0 );
  SetFlows ( orig.NumFlows() );
  mRows = 0;
  mCols = 0;
  mCurrentWell = 0;
  size_t count = 0;
  while ( !orig.ReadNextData ( &o ) )
  {
    mRows = std::max ( ( size_t ) o.y, mRows );
    mCols = std::max ( ( size_t ) o.x, mCols );
    mRankData[count] = o.rank;
    copy ( o.flowValues, o.flowValues + mFlows, mFlowData.begin() + count * mFlows );
    mIndexes[count] = count;
    count++;
  }

  mRows += 1; // Number is 1 higher than largest index.
  mCols += 1;
  SetRows ( mRows );
  SetCols ( mCols );
  SetFlows ( mFlows );
  orig.Close();
  mIsLegacy = true;
  return ( false ); // no error
}


void RawWells::ReadWell ( int wellIdx, WellData *_data )
{
  IndexToXY ( wellIdx, _data->x, _data->y );
  if ( mZeros.empty() )
  {
    mZeros.resize ( mFlows );
  }
  _data->rank = mRankData[wellIdx];
  if ( mIndexes[wellIdx] == WELL_NOT_LOADED )
  {
    ION_ABORT ( "Well: " + ToStr ( wellIdx ) + " is not loaded." );
  }
  if ( mIndexes[wellIdx] >= 0 )
  {
    for ( size_t i = 0; i < mFlows; i++ )
    {
      mZeros[i] = At ( wellIdx, i );
    }
    _data->flowValues = &mZeros[0];
  }
  else if ( mIndexes[wellIdx] == WELL_NOT_SUBSET )
  {
    fill ( mZeros.begin(), mZeros.end(), 0.0f );
    _data->flowValues = &mZeros[0];
  }
  else
  {
    ION_ABORT ( "Don't recognize index: " + ToStr ( mIndexes[wellIdx] ) );
  }
}

bool RawWells::InChunk ( size_t row, size_t col )
{
  return ( row >= mChunk.rowStart && row < mChunk.rowStart + mChunk.rowHeight &&
           col >= mChunk.colStart && col < mChunk.colStart + mChunk.colWidth );
}

size_t RawWells::GetNextRegionData()
{
  bool reload = false;
  if ( mFirsttimeGetRegionData && !mIsLegacy )
  {
    SetChunk ( mCurrentRegionRow, min ( max ( mRows - mCurrentRegionRow, (size_t) 0 ), mWellChunkSizeRow ),
               mCurrentRegionCol, min ( max ( mCols - mCurrentRegionCol, (size_t) 0 ), mWellChunkSizeCol ),
               0, mFlows );
    ReadWells();
  }
  mFirsttimeGetRegionData = false;
  mCurrentCol++;
  if ( mCurrentCol >= mWellChunkSizeCol || mCurrentRegionCol + mCurrentCol >= mCols )
  {
    mCurrentCol = 0;
    mCurrentRow++;
    if ( mCurrentRow >= mWellChunkSizeRow || mCurrentRegionRow + mCurrentRow >= mRows )
    {
      reload = true;
      mCurrentRow = 0;
      mCurrentRegionCol += mWellChunkSizeCol;
      if ( mCurrentRegionCol >= mCols )
      {
        mCurrentRegionCol = 0;
        mCurrentRegionRow += mWellChunkSizeRow;
      }
    }
  }
  /* Load up the right region... */
  size_t row = mCurrentRegionRow + mCurrentRow;
  size_t col = mCurrentRegionCol + mCurrentCol;
  if ( reload && mCurrentRegionRow < mRows && !InChunk ( row,col ) )
  {
    SetChunk ( mCurrentRegionRow, min ( max ( mRows - mCurrentRegionRow, (size_t) 0 ), mWellChunkSizeRow ),
               mCurrentRegionCol, min ( max ( mCols - mCurrentRegionCol, (size_t) 0 ), mWellChunkSizeCol ),
               0, mFlows );
    ReadWells();
  }
  return ToIndex ( col, row );
}

const WellData *RawWells::ReadNextRegionData()
{
  bool val = ReadNextRegionData ( &data );
  if ( val )
  {
    return &data;
  }
  return NULL;
}

bool RawWells::ReadNextRegionData ( WellData *_data )
{
  size_t well = GetNextRegionData();
  if ( well >= NumWells() )
  {
    return true;
  }
  ReadWell ( well,_data );
  return ( _data == NULL );
}

/**
 * Not thread safe. One call to this function will overwrite the
 * previous.  Don't use this function if you can help it
 */
const WellData *RawWells::ReadXY ( int x, int y )
{
  int well = ToIndex ( x,y );
  ReadWell ( well, &data );
  return &data;
}

void RawWells::CreateBuffers()
{
  if ( !mInfo.SetValue ( FLOW_ORDER, mFlowOrder ) )
  {
    ION_ABORT ( "Error - Could not set flow order." );
  }
  mRankData.resize ( NumRows() * NumCols() );
  fill ( mRankData.begin(), mRankData.end(), 0 );
  InitIndexes();
  uint64_t wellCount = 0;
  for ( size_t i = 0; i < mIndexes.size(); i++ )
  {
    if ( mIndexes[i] >= 0 )
    {
      wellCount++;
    }
  }
  mFlowData.resize ( ( uint64_t ) wellCount * mChunk.flowDepth );
  fill ( mFlowData.begin(), mFlowData.end(), -1.0f );
}

void RawWells::WriteToHdf5 ( const std::string &file )
{
  if ( mHFile == RWH5DataSet::EMPTY )
  {
    mHFile = H5Fcreate ( file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
  }
  if ( mHFile < 0 )
  {
    ION_ABORT ( "Couldn't create file: " + mFilePath );
  }

  OpenWellsForWrite();
  WriteWells();
  //OpenResForWrite();
  //WriteRes();
  WriteRanks();
  WriteInfo();
  CleanupHdf5();
}


void RawWells::WriteWells()
{
	//printf("WriteWells %d  %d %d %d \n", mChunk.flowDepth, mChunk.flowStart, mChunk.rowStart, mChunk.colStart);
  writeTimer.StartInterval();
  mInputBuffer.resize ( mStepSize*mStepSize*mChunk.flowDepth );
  fill ( mInputBuffer.begin(), mInputBuffer.end(), 0 );
  WellChunk chunk;
  uint32_t currentRowStart = mChunk.rowStart, 
           currentRowEnd = mChunk.rowStart + min ( mChunk.rowHeight, mStepSize );
  uint32_t currentColStart = mChunk.colStart, 
           currentColEnd = mChunk.colStart + min ( mChunk.colWidth, mStepSize );
  RawWellsWriter writer;
  /* For each chunk of data pad out the non-live (non subset) wells with
     zeros before writing to disk. */
  for ( currentRowStart = 0, currentRowEnd = mStepSize;
        currentRowStart < mChunk.rowStart + mChunk.rowHeight;
        currentRowStart = currentRowEnd, currentRowEnd += mStepSize )
  {
    currentRowEnd = min ( ( uint32_t ) ( mChunk.rowStart + mChunk.rowHeight ), currentRowEnd );
    for ( currentColStart = 0, currentColEnd = mStepSize;
          currentColStart < mChunk.colStart + mChunk.colWidth;
          currentColStart = currentColEnd, currentColEnd += mStepSize )
    {
      currentColEnd = min ( ( uint32_t ) ( mChunk.colStart + mChunk.colWidth ), currentColEnd );
      chunk.rowStart = currentRowStart;
      chunk.rowHeight = currentRowEnd - currentRowStart;
      chunk.colStart = currentColStart;
      chunk.colWidth = currentColEnd - currentColStart;
      chunk.flowStart = mChunk.flowStart;
      chunk.flowDepth = mChunk.flowDepth;
      
      mInputBuffer.resize ( ( uint64_t ) ( currentRowEnd - currentRowStart ) *
                            ( uint64_t ) ( currentColEnd - currentColStart ) *
                            ( uint64_t ) mChunk.flowDepth );
      int idxCount = 0;
      for ( size_t row = currentRowStart; row < currentRowEnd; row++ )
      {
        for ( size_t col = currentColStart; col < currentColEnd; col++ )
        {
          int idx = ToIndex ( col, row );
          for ( size_t fIx = mChunk.flowStart; fIx < mChunk.flowStart + mChunk.flowDepth; fIx++ )
          {
            uint64_t ii = idxCount * mChunk.flowDepth + fIx - mChunk.flowStart;
            assert ( ii < mInputBuffer.size() );
            mInputBuffer[ii] = At ( idx, fIx );
          }
          idxCount++;
        }
      }
      // printf("before writer.WriteWellsData mInputBuffer.size() - %d  \n",mInputBuffer.size() );
      // cout << mWells.mName <<endl;
      int error = writer.WriteWellsData(mWells, chunk, &mInputBuffer[0]);
      if ( error < 0 )
      {
        ION_ABORT ( "ERROR - Unsuccessful write to file: " +
                    ToStr ( mChunk.rowStart ) + "," + ToStr ( mChunk.colStart ) + "," +
                    ToStr ( mChunk.rowHeight ) + "," + ToStr ( mChunk.colWidth ) + " x " +
                    ToStr ( mChunk.flowStart ) + "," + ToStr ( mChunk.flowDepth ) + "\t" + 
                    mFilePath );
      }
    }
  }
  writeTimer.EndInterval();
}

void RawWells::WriteRes()
{
  writeTimer.StartInterval();
  mInputBuffer1.resize ( mStepSize*mStepSize*mChunk.flowDepth );
  fill ( mInputBuffer1.begin(), mInputBuffer1.end(), 0 );
  WellChunk chunk;
  uint32_t currentRowStart = mChunk.rowStart, 
           currentRowEnd = mChunk.rowStart + min ( mChunk.rowHeight, mStepSize );
  uint32_t currentColStart = mChunk.colStart, 
           currentColEnd = mChunk.colStart + min ( mChunk.colWidth, mStepSize );
  RawWellsWriter writer;
  /* For each chunk of data pad out the non-live (non subset) wells with
     zeros before writing to disk. */
  for ( currentRowStart = 0, currentRowEnd = mStepSize;
        currentRowStart < mChunk.rowStart + mChunk.rowHeight;
        currentRowStart = currentRowEnd, currentRowEnd += mStepSize )
  {
    currentRowEnd = min ( ( uint32_t ) ( mChunk.rowStart + mChunk.rowHeight ), currentRowEnd );
    for ( currentColStart = 0, currentColEnd = mStepSize;
          currentColStart < mChunk.colStart + mChunk.colWidth;
          currentColStart = currentColEnd, currentColEnd += mStepSize )
    {
      currentColEnd = min ( ( uint32_t ) ( mChunk.colStart + mChunk.colWidth ), currentColEnd );
      chunk.rowStart = currentRowStart;
      chunk.rowHeight = currentRowEnd - currentRowStart;
      chunk.colStart = currentColStart;
      chunk.colWidth = currentColEnd - currentColStart;
      chunk.flowStart = mChunk.flowStart;
      chunk.flowDepth = mChunk.flowDepth;
      
      mInputBuffer1.resize ( ( uint64_t ) ( currentRowEnd - currentRowStart ) *
                            ( uint64_t ) ( currentColEnd - currentColStart ) *
                            ( uint64_t ) mChunk.flowDepth );
      int idxCount = 0;
      for ( size_t row = currentRowStart; row < currentRowEnd; row++ )
      {
        for ( size_t col = currentColStart; col < currentColEnd; col++ )
        {
          int idx = ToIndex ( col, row );
          for ( size_t fIx = mChunk.flowStart; fIx < mChunk.flowStart + mChunk.flowDepth; fIx++ )
          {
            uint64_t ii = idxCount * mChunk.flowDepth + fIx - mChunk.flowStart;
            assert ( ii < mInputBuffer1.size() );
            mInputBuffer1[ii] =  ResAtWithoutChecking( idx, fIx );
          }
          idxCount++;
        }
      }
      // printf("before writer.WriteWellsData mInputBuffer1.size() - %d  \n",mInputBuffer1.size() );
      // cout << mResErr.mName <<endl;
      int error = writer.WriteWellsData(mResErr, chunk, &mInputBuffer1[0]);
      if ( error < 0 )
      {
        ION_ABORT ( "ERROR - Unsuccessful write residuals to file: " +
                    ToStr ( mChunk.rowStart ) + "," + ToStr ( mChunk.colStart ) + "," +
                    ToStr ( mChunk.rowHeight ) + "," + ToStr ( mChunk.colWidth ) + " x " +
                    ToStr ( mChunk.flowStart ) + "," + ToStr ( mChunk.flowDepth ) + "\t" + 
                    mFilePath );
      }
    }
  }
  writeTimer.EndInterval();
}

void RawWells::OpenWellsForWrite()
{
  mIsLegacy = false;
  mWells.mName = WELLS;
  // Create wells data space
  hsize_t dimsf[3];
  dimsf[0] = mRows;
  dimsf[1] = mCols;
  dimsf[2] = mFlows;
  mWells.mDataspace = H5Screate_simple ( 3, dimsf, NULL );
  mWells.mDatatype = H5Tcopy ( H5T_NATIVE_FLOAT );

  // create 

  // Setup the chunking values, this is important for performance
  hsize_t cdims[3];
  cdims[0] = std::min ( mWellChunkSizeRow, mRows );
  cdims[1] = std::min ( mWellChunkSizeCol, mCols );
  cdims[2] = std::min ( mWellChunkSizeFlow, mFlows );
  hid_t plist;
  plist = H5Pcreate ( H5P_DATASET_CREATE );
  assert ( ( cdims[0]>0 ) && ( cdims[1] >0 ) && ( cdims[2]>0 ) );
  H5Pset_chunk ( plist, 3, cdims );
  if ( mCompression > 0 )
  {
    H5Pset_deflate ( plist, mCompression );
  }
  hid_t dapl;
  dapl = H5Pcreate ( H5P_DATASET_ACCESS );
  if(mSaveAsUShort)
  {
    mWells.mDatatype = H5Tcopy ( H5T_NATIVE_USHORT );
    mWells.mDataset = H5Dcreate2 ( mHFile, mWells.mName.c_str(), H5T_NATIVE_USHORT, mWells.mDataspace,
                                 H5P_DEFAULT, plist, dapl );
    mWells.mSaveAsUShort = mSaveAsUShort;
    mWells.mLower = mLower;
    mWells.mUpper = mUpper;
    hsize_t dimsa[1];
    dimsa[0] = 1;
    hid_t dataspace = H5Screate_simple( 1, dimsa, NULL );
    hid_t attrLower = H5Acreate( mWells.mDataset, "convert_low", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT );
    H5Awrite(attrLower, H5T_NATIVE_FLOAT, &mLower);
    H5Aclose(attrLower);
    hid_t attrUpper = H5Acreate( mWells.mDataset, "convert_high", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT );
    H5Awrite(attrUpper, H5T_NATIVE_FLOAT, &mUpper);
    H5Aclose(attrUpper);
    H5Sclose(dataspace);
  }
  else
  {
    mWells.mDatatype = H5Tcopy ( H5T_NATIVE_FLOAT );
    mWells.mDataset = H5Dcreate2 ( mHFile, mWells.mName.c_str(), H5T_NATIVE_FLOAT, mWells.mDataspace,
                                 H5P_DEFAULT, plist, dapl );
  }
}

void RawWells::OpenResForWrite()
{
  MemUsage ("OpenResForWrite");
  mResErr.mName = RESERROR;
  // Create residual data space
  hsize_t dimsf[3];
  dimsf[0] = mRows;
  dimsf[1] = mCols;
  dimsf[2] = mFlows;
  mResErr.mDataspace = H5Screate_simple ( 3, dimsf, NULL );
  mResErr.mDatatype = H5Tcopy ( H5T_NATIVE_FLOAT );
  // create 
  // Setup the chunking values, this is important for performance
  hsize_t cdims[3];
  cdims[0] = std::min ( mWellChunkSizeRow, mRows );
  cdims[1] = std::min ( mWellChunkSizeCol, mCols );
  cdims[2] = std::min ( mWellChunkSizeFlow, mFlows );
  hid_t plist;
  plist = H5Pcreate ( H5P_DATASET_CREATE );
  assert ( ( cdims[0]>0 ) && ( cdims[1] >0 ) && ( cdims[2]>0 ) );
  H5Pset_chunk ( plist, 3, cdims );
  if ( mCompression > 0 )
  {
    H5Pset_deflate ( plist, mCompression );
  }
  hid_t dapl;
  dapl = H5Pcreate ( H5P_DATASET_ACCESS );


  // have to add msaveUshort
  mResErr.mDatatype = H5Tcopy ( H5T_NATIVE_FLOAT );
  mResErr.mDataset = H5Dcreate2 ( mHFile, mResErr.mName.c_str(), H5T_NATIVE_FLOAT, mResErr.mDataspace,
                                 H5P_DEFAULT, plist, dapl );
}

void RawWells::WriteRanks()
{
  mRanks.mName = RANKS;
  // Create wells data space
  hsize_t dimsf[2];
  dimsf[0] = mRows;
  dimsf[1] = mCols;
  mRanks.mDataspace = H5Screate_simple ( 2, dimsf, NULL );
  mRanks.mDatatype = H5Tcopy ( H5T_NATIVE_INT );

  // Setup the chunking values
  hsize_t cdims[2];
  cdims[0] = std::min ( ( size_t ) 50, mRows );
  cdims[1] = std::min ( ( size_t ) 50, mCols );

  hid_t plist;
  plist = H5Pcreate ( H5P_DATASET_CREATE );
  assert ( ( cdims[0]>0 ) && ( cdims[1] >0 ) );
  H5Pset_chunk ( plist, 2, cdims );
  if ( mCompression > 0 )
  {
    H5Pset_deflate ( plist, mCompression );
  }

  mRanks.mDataset = H5Dcreate2 ( mHFile, mRanks.mName.c_str(), mRanks.mDatatype, mRanks.mDataspace,
                                 H5P_DEFAULT, plist, H5P_DEFAULT );
  herr_t status = H5Dwrite ( mRanks.mDataset, mRanks.mDatatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mRankData[0] );
  if ( status < 0 )
  {
    ION_ABORT ( "ERROR - Unsuccessful write to file: " + mFilePath );
  }
  mRanks.Close();
}

void RawWells::SetSaveCopies(bool saveCopies)
{
    mSaveCopies = saveCopies;
    if(mSaveCopies)
    {
        bool closeFile = false;
        if ( mHFile == RWH5DataSet::EMPTY )
        {
            mHFile = H5Fopen ( mFilePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
            closeFile = true;
        }

        H5G_info_t group_info;
        group_info.nlinks = 0;
        if(H5Gget_info(mHFile, &group_info) < 0)
        {
            mSaveCopies = false;
            if(closeFile)
            {
                H5Fclose(mHFile);
                mHFile = RWH5DataSet::EMPTY;
            }

            cerr << "RawWells::SetSaveCopies() ERROR: fail to H5Gget_info" << endl;
            return;
        }
        else
        {
            for(unsigned int i = 0; i < group_info.nlinks; ++i)
            {
                size_t size = H5Gget_objname_by_idx(mHFile, i, NULL, 0);
                std::unique_ptr<char[]> name( new char[size+1]);
                if(H5Gget_objname_by_idx(mHFile, i, name.get(), size + 1) >= 0)
                {
                    if(strncmp(name.get(), "wells_copies", size+1)==0)
                    {
                        mSaveCopies = false;
                        if(closeFile)
                        {
                            H5Fclose(mHFile);
                            mHFile = RWH5DataSet::EMPTY;
                        }

                        cout << "RawWells::SetSaveCopies() WARNING: wells_copies already exists, skip writing wells copies." << endl;
                        return;
                    }
                }
            }
        }
        if(closeFile)
        {
            H5Fclose(mHFile);
            mHFile = RWH5DataSet::EMPTY;
        }
    }

    if(mSaveCopies)
    {
        mWellsCopies.resize(mRows * mCols, -1);
    }
}

void RawWells::SetConvertWithCopies(bool withCopies)
{
	mConvertWithCopies = withCopies;
}

void RawWells::WriteWellsCopies()
{  
  if(mWellsCopies.empty())
  {
	cerr << "RawWells WARNING: There is no wells copies data. Can not save dataset wells_copies." << endl;
  }

  if(!mSaveCopies)
  {
	cerr << "RawWells WriteWellsCopies() WARNING: Saving copies flag is set to false, skip writing wells_copies." << endl;
	return;
  }

	mSaveCopies = false;

    bool closeFile = false;
    if ( mHFile == RWH5DataSet::EMPTY )
    {
      mHFile = H5Fopen ( mFilePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
      closeFile = true;
    }

	H5G_info_t group_info;
    group_info.nlinks = 0;
	if(H5Gget_info(mHFile, &group_info) < 0)
	{
	  if(closeFile)
      {
        H5Fclose(mHFile);
        mHFile = RWH5DataSet::EMPTY;
      }

	  cerr << "RawWells::WriteWellsCopies() ERROR: fail to H5Gget_info, skip writing wells_copies" << endl;
	  return;
	}
	else
	{
	  for(unsigned int i = 0; i < group_info.nlinks; ++i)
	  {
        size_t size = H5Gget_objname_by_idx(mHFile, i, NULL, 0);
        std::unique_ptr<char[]> name( new char[size+1]);
        if(H5Gget_objname_by_idx(mHFile, i, name.get(), size + 1) >= 0)
	    {
          if(strncmp(name.get(), "wells_copies", size+1)==0)
		  {
              if(closeFile)
              {
                H5Fclose(mHFile);
                mHFile = RWH5DataSet::EMPTY;
              }

              cout << "RawWells::WriteWellsCopies() WARNING: wells_copies already exists, skip writing wells_copies." << endl;
              return;
		  }
	    }
	  }
    }

    hsize_t dimsf[2];
    dimsf[0] = mRows;
    dimsf[1] = mCols;
    hid_t dataspace = H5Screate_simple ( 2, dimsf, NULL );

    // Setup the chunking values
    hsize_t cdims[2];
    cdims[0] = std::min ( ( size_t ) 50, mRows );
    cdims[1] = std::min ( ( size_t ) 50, mCols );

    hid_t plist = H5Pcreate ( H5P_DATASET_CREATE );
    assert ( ( cdims[0]>0 ) && ( cdims[1] >0 ) );
    H5Pset_chunk ( plist, 2, cdims );
    if ( mCompression > 0 )
    {
      H5Pset_deflate ( plist, mCompression );
    }

    hid_t dsCopies = H5Dcreate2 ( mHFile, "wells_copies", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, plist, H5P_DEFAULT );
    if ( dsCopies < 0 )
    {
	  H5Sclose (dataspace);
      H5Pclose (plist);
	  cerr << "RawWells::WriteWellsCopies() ERROR: Fail to create dataset wells_copies." << endl;
    }
	else
	{
      herr_t status = H5Dwrite ( dsCopies, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mWellsCopies[0] );
      if ( status < 0 )
      {
	    cerr << "RawWells::WriteWellsCopies() ERROR: Fail to write dataset wells_copies." << endl;
      }
      H5Dclose (dsCopies);
	  H5Sclose (dataspace);
      H5Pclose (plist);
	}

    if(closeFile)
    {
      H5Fclose(mHFile);
      mHFile = RWH5DataSet::EMPTY;
    }
}

void RawWells::WriteStringVector ( hid_t h5File, const std::string &name, const char **values, int numValues )
{
  RWH5DataSet set;
  set.mName = name;
  // Create info dataspace
  hsize_t dimsf[1];
  dimsf[0] = numValues;
  set.mDataspace = H5Screate_simple ( 1, dimsf, NULL );
  set.mDatatype = H5Tcopy ( H5T_C_S1 );
  herr_t status = 0;
  status = H5Tset_size ( set.mDatatype, H5T_VARIABLE );
  if ( status < 0 )
  {
    ION_ABORT ( "Couldn't set string type to variable in set: " + name );
  }

  hid_t memType = H5Tcopy ( H5T_C_S1 );
  status = H5Tset_size ( memType, H5T_VARIABLE );
  if ( status < 0 )
  {
    ION_ABORT ( "Couldn't set string type to variable in set: " + name );
  }

  // Setup the chunking values
  hsize_t cdims[1];
  cdims[0] = std::min ( 100, numValues );

  hid_t plist;
  plist = H5Pcreate ( H5P_DATASET_CREATE );
  assert ( ( cdims[0]>0 ) );
  H5Pset_chunk ( plist, 1, cdims );
  if ( mCompression > 0 )
  {
    H5Pset_deflate ( plist, mCompression );
  }

  set.mDataset = H5Dcreate2 ( h5File, set.mName.c_str(), set.mDatatype, set.mDataspace,
                              H5P_DEFAULT, plist, H5P_DEFAULT );
  status = H5Dwrite ( set.mDataset, set.mDatatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values );
  if ( status < 0 )
  {
    ION_ABORT ( "ERROR - Unsuccessful write to file: " + mFilePath + " dataset: " + set.mName );
  }
  set.Close();
}

void RawWells::ReadStringVector ( hid_t h5File, const std::string &name, std::vector<std::string> &strings )
{
  strings.clear();
  RWH5DataSet set;

  set.mDataset = H5Dopen2 ( h5File, name.c_str(), H5P_DEFAULT );
  if ( set.mDataset < 0 )
  {
    ION_ABORT ( "Could not open data set: " + name );
  }


  set.mDatatype  = H5Dget_type ( set.mDataset );  /* datatype handle */

  //size_t size  = H5Tget_size ( set.mDatatype );
  set.mDataspace = H5Dget_space ( set.mDataset ); /* dataspace handle */
  int rank = H5Sget_simple_extent_ndims ( set.mDataspace );
  hsize_t dims[rank];           /* dataset dimensions */
  int status_n = H5Sget_simple_extent_dims ( set.mDataspace, dims, NULL );
  if ( status_n<0 )
  {
    H5Sclose ( set.mDataspace );
    ION_ABORT ( "Internal Error in H5Sget_simple_extent_dims - ReadStringVector" );
  }

  char **rdata = ( char** ) malloc ( dims[0] * sizeof ( char * ) );

  hid_t memType = H5Tcopy ( H5T_C_S1 ); // C style strings
  herr_t status = H5Tset_size ( memType, H5T_VARIABLE ); // Variable rather than fixed length
  if ( status < 0 )
  {
    ION_ABORT ( "Error setting string length to variable." );
  }

  // Read the strings
  status = H5Dread ( set.mDataset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata );
  if ( status < 0 )
  {
    ION_ABORT ( "Error reading " + name );
  }

  // Put into output string
  strings.resize ( dims[0] );
  for ( size_t i = 0; i < dims[0]; i++ )
  {
    strings[i] = rdata[i];
  }

  status = H5Dvlen_reclaim ( memType, set.mDataspace, H5P_DEFAULT, rdata );
  free ( rdata );
  set.Close();
}

void RawWells::WriteInfo()
{
  int numEntries = mInfo.GetCount();
  vector<string> keys ( numEntries );
  vector<string> values ( numEntries );
  vector<const char *> keysBuff ( numEntries );
  vector<const char *> valuesBuff ( numEntries );
  for ( int i = 0; i < numEntries; i++ )
  {
    mInfo.GetEntry ( i, keys[i], values[i] );
    keysBuff[i] = keys[i].c_str();
    valuesBuff[i] = values[i].c_str();
  }

  WriteStringVector ( mHFile, INFO_KEYS, &keysBuff[0], keysBuff.size() );
  WriteStringVector ( mHFile, INFO_VALUES, &valuesBuff[0], keysBuff.size() );
}

bool RawWells::OpenMetaData()
{
  OpenForIncrementalRead();
  return true;
}

bool RawWells::OpenForRead ( bool memmap_dummy )
{
  //  mWriteOnClose = false;
  H5E_auto2_t old_func;
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2 ( H5E_DEFAULT, &old_func, &old_client_data );
  H5Eset_auto2 ( H5E_DEFAULT, NULL, NULL );
  if ( mHFile == RWH5DataSet::EMPTY )
  {
    mHFile = H5Fopen ( mFilePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  }
  if ( mHFile >= 0 )
  {
    mIsLegacy = false;
    mCurrentWell = 0;
    OpenWellsToRead();
    ReadWells();
    //ReadRes();
    ReadRanks();
    ReadInfo();
  }
  else if ( mHFile < 0 )
  {
    bool failed = OpenForReadLegacy();
    if ( failed )
    {
      ION_ABORT ( "Couldn't open: " + mFilePath + " to read." );
    }
  }
  H5Eset_auto2 ( H5E_DEFAULT, old_func, old_client_data );
  return false;
}

bool RawWells::OpenForReadWrite()
{
  //  mWriteOnClose = false;
  H5E_auto2_t old_func;
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2 ( H5E_DEFAULT, &old_func, &old_client_data );
  H5Eset_auto2 ( H5E_DEFAULT, NULL, NULL );
  if ( mHFile == RWH5DataSet::EMPTY )
  {
    mHFile = H5Fopen ( mFilePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
  }
  if ( mHFile >= 0 )
  {
    mIsLegacy = false;
    mCurrentWell = 0;
    OpenWellsToRead();
    //OpenResToRead();
    ReadWells();
    //ReadRes();
    ReadRanks();
    ReadInfo();
  }
  else if ( mHFile < 0 )
  {
    ION_ABORT ( "OpenForReadWrite() - Not supported for legacy wells files." );
  }
  H5Eset_auto2 ( H5E_DEFAULT, old_func, old_client_data );
  InitIndexes();
  return false;
}

void RawWells::WriteLegacyWells()
{
  WellHeader hdr;
  hdr.flowOrder = NULL;
  int flowCnt = NumFlows();
  FILE *fp = NULL;
  fopen_s ( &fp, mFilePath.c_str(), "wb" );
  const char *flowOrder = FlowOrder();
  if ( fp )
  {
    // generate and write the header
    hdr.numWells = NumWells();
    hdr.numFlows = flowCnt;
    hdr.flowOrder = ( char * ) malloc ( hdr.numFlows );
    for ( int i=0;i<hdr.numFlows;i++ )
    {
      hdr.flowOrder[i] = flowOrder[i%flowCnt];
    }
    fwrite ( &hdr.numWells, sizeof ( hdr.numWells ), 1, fp );
    fwrite ( &hdr.numFlows, sizeof ( hdr.numFlows ), 1, fp );
    fwrite ( hdr.flowOrder, sizeof ( char ), hdr.numFlows, fp );

    // write the well data, initially all zeros, to create the file
    int dataBlockSize = sizeof ( unsigned int ) + 2 * sizeof ( unsigned short );
    int flowValuesSize = sizeof ( float ) * hdr.numFlows;
    uint64_t offset;

    float* wellPtr = NULL;
    for ( uint64_t y=0;y<mRows;y++ )
    {
      for ( uint64_t x=0;x<mCols;x++ )
      {
        data.rank = 0;//data.rank = i;
        data.x = ( int ) x;
        data.y = ( int ) y;
        fwrite ( &data, dataBlockSize, 1, fp );
        offset = ToIndex ( x,y );
        wellPtr = &mFlowData[mIndexes[offset] * NumFlows() ];
        fwrite ( wellPtr, flowValuesSize, 1, fp );
      }
    }
    free ( hdr.flowOrder );
    fclose ( fp );
    fp = NULL;
  }
}

void RawWells::OpenWellsToRead()
{
  mWells.Close();
  mWells.mName = WELLS;
  mWells.mDataset = H5Dopen2 ( mHFile, mWells.mName.c_str(), H5P_DEFAULT );
  if ( mWells.mDataset < 0 )
  {
    ION_ABORT ( "Could not open data set: " + mWells.mName +
                " in file: " + mFilePath );
  }

  hsize_t dims_out[3];           /* dataset dimensions */
  mWells.mDatatype  = H5Dget_type ( mWells.mDataset );  /* datatype handle */

  mSaveAsUShort = false;
  if(H5Aexists( mWells.mDataset, "convert_low" ) > 0)
  {
    hid_t attrLower = H5Aopen( mWells.mDataset, "convert_low", H5P_DEFAULT );
    H5Aread( attrLower, H5T_NATIVE_FLOAT, &mLower ); 
    mWells.mLower = mLower;
    mSaveAsUShort = true;
    mWells.mSaveAsUShort = mSaveAsUShort;
  }
  if(H5Aexists( mWells.mDataset, "convert_high" ) > 0)
  {
    hid_t attrUpper = H5Aopen( mWells.mDataset, "convert_high", H5P_DEFAULT );
    H5Aread( attrUpper, H5T_NATIVE_FLOAT, &mUpper ); 
    mWells.mUpper = mUpper;
    mSaveAsUShort = true;
    mWells.mSaveAsUShort = mSaveAsUShort;
  }

  //size_t size  = H5Tget_size ( mWells.mDatatype );
  mWells.mDataspace = H5Dget_space ( mWells.mDataset ); /* dataspace handle */
  int rank = H5Sget_simple_extent_ndims ( mWells.mDataspace );
  if ( rank!=3 )
  {
    ION_ABORT ( "Internal Error: rank wrong for H5Sget_simple_extent_ndims in OpenWellsToRead" );
  }
  int status_n = H5Sget_simple_extent_dims ( mWells.mDataspace, dims_out, NULL );
  if ( status_n<0 )
  {
    ION_ABORT ( "Internal Error: H5Sget_simple_extent_dims in OpenWellsToRead" );
  }
  mRows = dims_out[0];
  mCols = dims_out[1];
  mFlows = dims_out[2];
  // Assume whole chip if region if region isn't set
  if ( mChunk.rowHeight <= 0 )
  {
    mChunk.rowStart = 0;
    mChunk.rowHeight = mRows;
  }
  if ( mChunk.colWidth <= 0 )
  {
    mChunk.colStart = 0;
    mChunk.colWidth = mCols;
  }
  if ( mChunk.flowDepth <= 0 )
  {
    mChunk.flowStart = 0;
    mChunk.flowDepth = mFlows;
  }

  // Read the chunk size from the wells file creation property list
  hid_t create_plist = H5Dget_create_plist( mWells.mDataset );
  hsize_t cdims[3]; // We checked above that the rank is 3.
  int rval = H5Pget_chunk ( create_plist, 3, cdims );
  if (rval < 0) {
    ION_WARN ( "RawWells::OpenWellsToRead: Unable to read wells file chunk size." );
  } else {
    mWellChunkSizeRow  = cdims[0];
    mWellChunkSizeCol  = cdims[1];
    mWellChunkSizeFlow = cdims[2];
  }
  H5Pclose( create_plist );

  mWellsCopies2.resize(mRows * mCols, 1.0);
  if(mSaveAsUShort && mConvertWithCopies)
  {
        H5G_info_t group_info;
        group_info.nlinks = 0;
        if(H5Gget_info(mHFile, &group_info) < 0)
        {
            cerr << "RawWells::OpenWellsToRead() WARNING: fail to H5Gget_info" << endl;
        }
        else
        {
            for(unsigned int i = 0; i < group_info.nlinks; ++i)
            {
                size_t size = H5Gget_objname_by_idx(mHFile, i, NULL, 0);
                std::unique_ptr<char[]> name( new char[size+1]);
                if(H5Gget_objname_by_idx(mHFile, i, name.get(), size + 1) < 0)
                {
                    cerr << "RawWells::OpenWellsToRead() WARNING: 1.wells files does not have wells_copies saved." << endl;
                }
                else
                {
                    if(strncmp(name.get(), "wells_copies", size+1) == 0)
                    {
                        hid_t ds = H5Dopen2(mHFile, "wells_copies", H5P_DEFAULT);
                        if(ds < 0)
                        {
                            cerr << "RawWells::OpenWellsToRead() WARNING: 1.wells files does not have wells_copies." << endl;
                        }
                        else
                        {
                            hid_t dataSpace = H5Dget_space(ds);
                            if(dataSpace < 0)
                            {
                                H5Dclose(ds);
                                cerr << "RawWells::OpenWellsToRead() WARNING: fail to H5Dget_space for dataset wells_copies." << endl;
                            }
                            else
                            {
                                hssize_t dsSize = H5Sget_simple_extent_npoints(dataSpace);
                                H5Sclose(dataSpace);
                                if(dsSize != (hssize_t)(mRows * mCols))
                                {
                                    H5Dclose(ds);
                                    cerr << "RawWells::OpenWellsToRead() WARNING: dataset wells_copies size is " << dsSize << ", it is different from mRows * mCols = " << mRows * mCols << endl;
                                }
                                else
                                {
                                    herr_t ret = H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mWellsCopies2[0]);
                                    H5Dclose(ds);
                                    if(ret < 0)
                                    {
                                        mWellsCopies2.resize(mRows * mCols, 1.0);
                                        cerr << "RawWells::OpenWellsToRead() WARNING: failto load dataset wells_copies." << endl;
                                    }
                                }
                            }
                        }
                        break;
                    }                  
                }
            }
       }
  }
  InitIndexes();
}

void RawWells::OpenResToRead()
{
  mResErr.Close();
  mResErr.mName = RESERROR;
  mResErr.mDataset = H5Dopen2 ( mHFile, mResErr.mName.c_str(), H5P_DEFAULT );
  if ( mResErr.mDataset < 0 )
  {
    ION_ABORT ( "Could not open data set: " + mResErr.mName +
                " in file: " + mFilePath );
  }

  hsize_t dims_out[3];           /* dataset dimensions */
  mResErr.mDatatype  = H5Dget_type ( mResErr.mDataset );  /* datatype handle */

  //mSaveAsUShort = false;
  if(H5Aexists( mResErr.mDataset, "convert_low" ) > 0)
  {
    hid_t attrLower = H5Aopen( mResErr.mDataset, "convert_low", H5P_DEFAULT );
    H5Aread( attrLower, H5T_NATIVE_FLOAT, &mLower );
    mResErr.mLower = mLower;
    //mSaveAsUShort = true;
    //mResErr.mSaveAsUShort = mSaveAsUShort;
  }
  if(H5Aexists( mResErr.mDataset, "convert_high" ) > 0)
  {
    hid_t attrUpper = H5Aopen( mResErr.mDataset, "convert_high", H5P_DEFAULT );
    H5Aread( attrUpper, H5T_NATIVE_FLOAT, &mUpper );
    mResErr.mUpper = mUpper;
    //mSaveAsUShort = true;
    //mResErr.mSaveAsUShort = mSaveAsUShort;
  }

  //size_t size  = H5Tget_size ( mWells.mDatatype );
  mResErr.mDataspace = H5Dget_space ( mResErr.mDataset ); /* dataspace handle */
  int rank = H5Sget_simple_extent_ndims ( mResErr.mDataspace );
  if ( rank!=3 )
  {
    ION_ABORT ( "Internal Error: rank wrong for H5Sget_simple_extent_ndims in OpenResToRead" );
  }
  int status_n = H5Sget_simple_extent_dims ( mResErr.mDataspace, dims_out, NULL );
  if ( status_n<0 )
  {
    ION_ABORT ( "Internal Error: H5Sget_simple_extent_dims in OpenResToRead" );
  }

  // Read the chunk size from the wells file creation property list
  hid_t create_plist = H5Dget_create_plist( mResErr.mDataset );
  hsize_t cdims[3]; // We checked above that the rank is 3.
  int rval = H5Pget_chunk ( create_plist, 3, cdims );
  if (rval < 0) {
    ION_WARN ( "RawWells::OpenResToRead: Unable to read wells file chunk size." );
  } else {
    mWellChunkSizeRow  = cdims[0];
    mWellChunkSizeCol  = cdims[1];
    mWellChunkSizeFlow = cdims[2];
  }
  H5Pclose( create_plist );

//  mWellsCopies2.resize(mRows * mCols, 1.0);
//  if(mSaveAsUShort && mConvertWithCopies)
//  {
//    char name[10];
//    string sName;
//	H5G_info_t group_info;
//    group_info.nlinks = 0;
//	if(H5Gget_info(mHFile, &group_info) < 0)
//	{
//		cerr << "RawWells::OpenWellsToRead() WARNING: fail to H5Gget_info" << endl;
//	}
//	else
//	{
//		for(unsigned int i = 0; i < group_info.nlinks; ++i)
//		{
//		  int size = H5Gget_objname_by_idx(mHFile, i, NULL, 0);
//		  if(H5Gget_objname_by_idx(mHFile, i, name, size + 1) < 0)
//		  {
//			cerr << "RawWells::OpenWellsToRead() WARNING: 1.wells files does not have wells_copies saved." << endl;
//		  }
//		  else
//		  {
//			sName = name;
//			if(sName == "wells_copies")
//			{
//			  break;
//			}
//		  }
//		}
//
//		if(sName == "wells_copies")
//		{
//		  hid_t ds = H5Dopen2(mHFile, "wells_copies", H5P_DEFAULT);
//		  if(ds < 0)
//		  {
//			cerr << "RawWells::OpenWellsToRead() WARNING: 1.wells files does not have wells_copies." << endl;
//		  }
//		  else
//		  {
//			hid_t dataSpace = H5Dget_space(ds);
//			if(dataSpace < 0)
//			{
//			  H5Dclose(ds);
//			  cerr << "RawWells::OpenWellsToRead() WARNING: fail to H5Dget_space for dataset wells_copies." << endl;
//			}
//			else
//			{
//			  hssize_t dsSize = H5Sget_simple_extent_npoints(dataSpace);
//			  H5Sclose(dataSpace);
//			  if(dsSize != (hssize_t)(mRows * mCols))
//			  {
//				H5Dclose(ds);
//				cerr << "RawWells::OpenWellsToRead() WARNING: dataset wells_copies size is " << dsSize << ", it is different from mRows * mCols = " << mRows * mCols << endl;
//			  }
//			  else
//			  {
//				  herr_t ret = H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mWellsCopies2[0]);
//				  H5Dclose(ds);
//				  if(ret < 0)
//				  {
//					mWellsCopies2.resize(mRows * mCols, 1.0);
//					cerr << "RawWells::OpenWellsToRead() WARNING: failto load dataset wells_copies." << endl;
//				  }
//			   }
//			}
//		  }
//		}
//	}
// }

  InitIndexes();
}

void RawWells::OpenForIncrementalRead()
{
  //  mWriteOnClose = false;
  H5E_auto2_t old_func;
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2 ( H5E_DEFAULT, &old_func, &old_client_data );
  H5Eset_auto2 ( H5E_DEFAULT, NULL, NULL );
  mHFile = H5Fopen ( mFilePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  if ( mChunk.rowHeight <= 0 )
  {
    mChunk.rowStart = 0;
    mChunk.rowHeight = 1;
  }
  if ( mChunk.colWidth <= 0 )
  {
    mChunk.colStart = 0;
    mChunk.colWidth = 1;
  }
  if ( mChunk.flowDepth <= 0 )
  {
    mChunk.flowStart = 0;
    mChunk.flowDepth = 1;
  }
  if ( mHFile >= 0 )
  {
    mIsLegacy = false;
    mCurrentWell = 0;
    OpenWellsToRead();
    //OpenResToRead();
    ReadRanks();
    ReadInfo();
  }
  else if ( mHFile < 0 )
  {
    ION_WARN ( "Legacy wells file format, loading into memory." );
    bool failed = OpenForReadLegacy();
    if ( failed )
    {
      ION_ABORT ( "Couldn't open: " + mFilePath + " to read." );
    }
    SetChunk ( 0,mRows, 0, mCols, 0, mFlows );
  }
  H5Eset_auto2 ( H5E_DEFAULT, old_func, old_client_data );
}

void RawWells::ReadWells()
{
  if ( mIsLegacy )
  {
    return;
  }
  vector<float> inputBuffer; // 100x100
  uint64_t stepSize = (uint64_t)mWellChunkSizeRow*(uint64_t)mWellChunkSizeCol;
  inputBuffer.resize ( stepSize*mChunk.flowDepth,0.0f );
  // Assume whole chip if region if region isn't set
  //  int currentWellStart = 0, currentWellEnd = min(stepSize,NumWells());
  uint32_t currentRowStart = 0, currentRowEnd = min ( mWellChunkSizeRow, mRows );
  uint32_t currentColStart = 0, currentColEnd = min ( mWellChunkSizeCol, mCols );

  size_t wellsInSubset = 0;
  for ( size_t i = 0; i < mIndexes.size(); i++ )
  {
    if ( mIndexes[i] >= 0 )
    {
      wellsInSubset++;
    }
  }

  mFlowData.resize ( ( uint64_t ) wellsInSubset * mChunk.flowDepth );
  fill ( mFlowData.begin(), mFlowData.end(), -1.0f );
  for ( currentRowStart = mChunk.rowStart, currentRowEnd = mChunk.rowStart + min ( mWellChunkSizeRow, mChunk.rowHeight );
        currentRowStart < mRows && currentRowStart < mChunk.rowStart + mChunk.rowHeight;
        currentRowStart = currentRowEnd, currentRowEnd += mWellChunkSizeRow )
  {
    currentRowEnd = min ( ( uint32_t ) mRows, currentRowEnd );
    for ( currentColStart = mChunk.colStart, currentColEnd = mChunk.colStart + min ( mWellChunkSizeCol, mChunk.colWidth );
          currentColStart < mCols && currentColStart < mChunk.colStart + mChunk.colWidth;
          currentColStart = currentColEnd, currentColEnd += mWellChunkSizeCol )
    {
      currentColEnd = min ( ( uint32_t ) mCols, currentColEnd );
      // Don't go to disk unless we actually have a well to load.
      if ( WellsInSubset ( currentRowStart, currentRowEnd, currentColStart, currentColEnd ) )
      {

        hsize_t count[3];       /* size of the hyperslab in the file */
        hsize_t offset[3];      /* hyperslab offset in the file */
        hsize_t count_out[3];   /* size of the hyperslab in memory */
        hsize_t offset_out[3];  /* hyperslab offset in memory */
        offset[0] = currentRowStart;
        offset[1] = currentColStart;
        offset[2] = 0;
        count[0] = currentRowEnd - currentRowStart;
        count[1] = currentColEnd - currentColStart;
        count[2] = mChunk.flowDepth;
        herr_t status = 0;
        status = H5Sselect_hyperslab ( mWells.mDataspace, H5S_SELECT_SET, offset, NULL,
                                       count, NULL );
        if ( status < 0 )
        {
          ION_ABORT ( "Couldn't read wells dataspace for: " +
                      ToStr ( mChunk.rowStart ) + "," + ToStr ( mChunk.colStart ) + "," +
                      ToStr ( mChunk.rowHeight ) + "," + ToStr ( mChunk.colWidth ) + " x " +
                      ToStr ( mChunk.flowDepth ) );
        }

        hsize_t     dimsm[3];              /* memory space dimensions */
        hid_t       memspace;
        /*
         * Define the memory dataspace.
         */
        dimsm[0] = count[0];
        dimsm[1] = count[1];
        dimsm[2] = count[2];
        memspace = H5Screate_simple ( 3,dimsm,NULL );

        offset_out[0] = 0;
        offset_out[1] = 0;
        offset_out[2] = 0;
        count_out[0] = currentRowEnd - currentRowStart;
        count_out[1] = currentColEnd - currentColStart;
        count_out[2] = mChunk.flowDepth;
        // ClockTimer timer;
        // timer.StartTimer();
        status = H5Sselect_hyperslab ( memspace, H5S_SELECT_SET, offset_out, NULL,
                                       count_out, NULL );
        if ( status < 0 )
        {
          ION_ABORT ( "Couldn't read wells hyperslab for: " +
                      ToStr ( mChunk.rowStart ) + "," + ToStr ( mChunk.colStart ) + "," +
                      ToStr ( mChunk.rowHeight ) + "," + ToStr ( mChunk.colWidth ) + " x " +
                      ToStr ( mChunk.flowDepth ) );
        }

        inputBuffer.resize ( ( uint64_t ) ( currentRowEnd - currentRowStart ) * ( currentColEnd - currentColStart ) * mFlows );
        status = -1;
        if(mSaveAsUShort)
        {
          WellsConverter converter(mLower, mUpper);
          vector<unsigned short> inputBuffer2;
          inputBuffer2.resize ( inputBuffer.size(), 0 );
          status = H5Dread ( mWells.mDataset, H5T_NATIVE_USHORT, memspace, mWells.mDataspace,
                           H5P_DEFAULT, &inputBuffer2[0] );

		  vector<float>::iterator iter = inputBuffer.begin();
		  vector<unsigned short>::iterator iter2 = inputBuffer2.begin();
          for(; iter2 != inputBuffer2.end(); ++iter, ++iter2)
          {
            *iter = converter.UInt16ToFloat(*iter2);
	      }
        }
		else
        {
          status = H5Dread ( mWells.mDataset, H5T_NATIVE_FLOAT, memspace, mWells.mDataspace,
                           H5P_DEFAULT, &inputBuffer[0] );
        }

        if ( status < 0 )
        {
          ION_ABORT ( "Couldn't read wells dataset for file: "  + mFilePath + " dims: " + ToStr ( mChunk.rowStart ) + "," +
                      ToStr ( mChunk.colStart ) + "," + ToStr ( mChunk.rowHeight ) + "," +
                      ToStr ( mChunk.colWidth ) + " x " + ToStr ( mChunk.flowDepth ) );
        }
        // std::cout << "Just read: " << currentRowStart  << "," << currentRowEnd << "," <<
        //   currentColStart << "," << currentColEnd << " x " <<
        //   mChunk.flowDepth << " in: " << timer.GetMicroSec() << "um seconds" << endl;

        uint64_t localCount = 0;
        for ( size_t row = currentRowStart; row < currentRowEnd; row++ )
        {
          for ( size_t col = currentColStart; col < currentColEnd; col++ )
          {
            int idx = ToIndex ( col, row );
            if ( mIndexes[idx] >= 0 )
            {
              for ( size_t flow = mChunk.flowStart; flow < mChunk.flowStart + mChunk.flowDepth; flow++ )
              {
				float val = inputBuffer[localCount * mChunk.flowDepth + flow];
				if(mSaveAsUShort && mConvertWithCopies)
				{
					if(mWellsCopies2[ row * mCols + col] > 0)
						val *= mWellsCopies2[ row * mCols + col];
					else
						val = -1.0;
				}
                Set ( row, col, flow, val );
              }
            }
            localCount++;
          }
        }
      }
    }
  }
}

void RawWells::ReadRes()
{
  if ( mIsLegacy )
  {
    return;
  }
  vector<float> inputBuffer1; // 100x100
  uint64_t stepSize = (uint64_t)mWellChunkSizeRow*(uint64_t)mWellChunkSizeCol;
  inputBuffer1.resize ( stepSize*mChunk.flowDepth,0.0f );
  // Assume whole chip if region if region isn't set
  //  int currentWellStart = 0, currentWellEnd = min(stepSize,NumWells());
  uint32_t currentRowStart = 0, currentRowEnd = min ( mWellChunkSizeRow, mRows );
  uint32_t currentColStart = 0, currentColEnd = min ( mWellChunkSizeCol, mCols );

  size_t wellsInSubset = 0;
  for ( size_t i = 0; i < mIndexes.size(); i++ )
  {
    if ( mIndexes[i] >= 0 )
    {
      wellsInSubset++;
    }
  }

  mResData.resize ( ( uint64_t ) wellsInSubset * mChunk.flowDepth );
  fill ( mResData.begin(), mResData.end(), -1.0f );
  for ( currentRowStart = mChunk.rowStart, currentRowEnd = mChunk.rowStart + min ( mWellChunkSizeRow, mChunk.rowHeight );
        currentRowStart < mRows && currentRowStart < mChunk.rowStart + mChunk.rowHeight;
        currentRowStart = currentRowEnd, currentRowEnd += mWellChunkSizeRow )
  {
    currentRowEnd = min ( ( uint32_t ) mRows, currentRowEnd );
    for ( currentColStart = mChunk.colStart, currentColEnd = mChunk.colStart + min ( mWellChunkSizeCol, mChunk.colWidth );
          currentColStart < mCols && currentColStart < mChunk.colStart + mChunk.colWidth;
          currentColStart = currentColEnd, currentColEnd += mWellChunkSizeCol )
    {
      currentColEnd = min ( ( uint32_t ) mCols, currentColEnd );
      // Don't go to disk unless we actually have a well to load.
      if ( WellsInSubset ( currentRowStart, currentRowEnd, currentColStart, currentColEnd ) )
      {

        hsize_t count[3];       /* size of the hyperslab in the file */
        hsize_t offset[3];      /* hyperslab offset in the file */
        hsize_t count_out[3];   /* size of the hyperslab in memory */
        hsize_t offset_out[3];  /* hyperslab offset in memory */
        offset[0] = currentRowStart;
        offset[1] = currentColStart;
        offset[2] = 0;
        count[0] = currentRowEnd - currentRowStart;
        count[1] = currentColEnd - currentColStart;
        count[2] = mChunk.flowDepth;
        herr_t status = 0;
        status = H5Sselect_hyperslab ( mResErr.mDataspace, H5S_SELECT_SET, offset, NULL,
                                       count, NULL );
        if ( status < 0 )
        {
          ION_ABORT ( "Couldn't read wells dataspace for: " +
                      ToStr ( mChunk.rowStart ) + "," + ToStr ( mChunk.colStart ) + "," +
                      ToStr ( mChunk.rowHeight ) + "," + ToStr ( mChunk.colWidth ) + " x " +
                      ToStr ( mChunk.flowDepth ) );
        }

        hsize_t     dimsm[3];              /* memory space dimensions */
        hid_t       memspace;
        /*
         * Define the memory dataspace.
         */
        dimsm[0] = count[0];
        dimsm[1] = count[1];
        dimsm[2] = count[2];
        memspace = H5Screate_simple ( 3,dimsm,NULL );

        offset_out[0] = 0;
        offset_out[1] = 0;
        offset_out[2] = 0;
        count_out[0] = currentRowEnd - currentRowStart;
        count_out[1] = currentColEnd - currentColStart;
        count_out[2] = mChunk.flowDepth;
        // ClockTimer timer;
        // timer.StartTimer();
        status = H5Sselect_hyperslab ( memspace, H5S_SELECT_SET, offset_out, NULL,
                                       count_out, NULL );
        if ( status < 0 )
        {
          ION_ABORT ( "Couldn't read residual  hyperslab for: " +
                      ToStr ( mChunk.rowStart ) + "," + ToStr ( mChunk.colStart ) + "," +
                      ToStr ( mChunk.rowHeight ) + "," + ToStr ( mChunk.colWidth ) + " x " +
                      ToStr ( mChunk.flowDepth ) );
        }

        inputBuffer1.resize ( ( uint64_t ) ( currentRowEnd - currentRowStart ) * ( currentColEnd - currentColStart ) * mFlows );
        status = -1;
        status = H5Dread ( mResErr.mDataset, H5T_NATIVE_FLOAT, memspace, mResErr.mDataspace,
                           H5P_DEFAULT, &inputBuffer1[0] );

        if ( status < 0 )
        {
          ION_ABORT ( "Couldn't read wells dataset for file: "  + mFilePath + " dims: " + ToStr ( mChunk.rowStart ) + "," +
                      ToStr ( mChunk.colStart ) + "," + ToStr ( mChunk.rowHeight ) + "," +
                      ToStr ( mChunk.colWidth ) + " x " + ToStr ( mChunk.flowDepth ) );
        }
        // std::cout << "Just read: " << currentRowStart  << "," << currentRowEnd << "," <<
        //   currentColStart << "," << currentColEnd << " x " <<
        //   mChunk.flowDepth << " in: " << timer.GetMicroSec() << "um seconds" << endl;

        uint64_t localCount = 0;
        for ( size_t row = currentRowStart; row < currentRowEnd; row++ )
        {
          for ( size_t col = currentColStart; col < currentColEnd; col++ )
          {
            int idx = ToIndex ( col, row );
            if ( mIndexes[idx] >= 0 )
            {
              for ( size_t flow = mChunk.flowStart; flow < mChunk.flowStart + mChunk.flowDepth; flow++ )
              {
				float val = inputBuffer1[localCount * mChunk.flowDepth + flow];
                SetRes ( row, col, flow, val );
              }
            }
            localCount++;
          }
        }
      }
    }
  }
}

bool RawWells::WellsInSubset ( uint32_t currentRowStart, uint32_t currentRowEnd,
                               uint32_t currentColStart, uint32_t currentColEnd )
{
  for ( size_t row = currentRowStart; row < currentRowEnd; row++ )
  {
    for ( size_t col = currentColStart; col < currentColEnd; col++ )
    {
      int idx = ToIndex ( col, row );
      if ( mIndexes[idx] == -3 || mIndexes[idx] >= 0 )
      {
        return true;
      }
    }
  }
  return false;
}

float RawWells::At ( size_t row, size_t col, size_t flow ) const
{
  return At ( ToIndex ( col,row ), flow );
}

float RawWells::At ( size_t well, size_t flow ) const
{
  if ( mIndexes[well] == WELL_NOT_LOADED )
  {
    ION_ABORT ( "Well: " + ToStr ( well ) + " is not loaded." );
  }
  if ( flow < mChunk.flowStart || flow >= mChunk.flowStart + mChunk.flowDepth )
  {
    ION_ABORT ( "Flow: " + ToStr ( flow ) + " is not loaded in range: " +
                ToStr ( mChunk.flowStart ) + " to " + ToStr ( mChunk.flowStart+mChunk.flowDepth ) );
  }
  if ( mIndexes[well] >= 0 )
  {
    uint64_t ii = ( uint64_t ) mIndexes[well] * mChunk.flowDepth + flow - mChunk.flowStart;
    assert ( ii < mFlowData.size() );
    return mFlowData[ii];
  }
  if ( mIndexes[well] == WELL_NOT_SUBSET )
  {
    return 0;
  }
  ION_ABORT ( "Don't recognize index type: " + ToStr ( mIndexes[well] ) );
  return 0;
}

float RawWells::ResAtWithoutChecking ( size_t row, size_t col, size_t flow ) const
{
  return ResAtWithoutChecking ( ToIndex ( col,row ), flow );
}

float RawWells::AtWithoutChecking ( size_t row, size_t col, size_t flow ) const
{
  return AtWithoutChecking ( ToIndex ( col,row ), flow );
}

float RawWells::AtWithoutChecking ( size_t well, size_t flow ) const
{
  uint64_t ii = ( uint64_t ) mIndexes[well] * mChunk.flowDepth + flow - mChunk.flowStart;
  return mFlowData[ii];
}

float RawWells::ResAtWithoutChecking ( size_t well, size_t flow ) const
{
  uint64_t ii = ( uint64_t ) mIndexes[well] * mChunk.flowDepth + flow - mChunk.flowStart;
  return mResData[ii];
}

void RawWells::ReadRanks()
{
  mRanks.Close();
  mRanks.mName = RANKS;
  mRanks.mDataset = H5Dopen2 ( mHFile, mRanks.mName.c_str(), H5P_DEFAULT );
  if ( mRanks.mDataset < 0 )
  {
    ION_ABORT ( "Could not open data set: " + mRanks.mName );
  }

  hsize_t dims_out[2];           /* dataset dimensions */
  mRanks.mDatatype  = H5Dget_type ( mRanks.mDataset );  /* datatype handle */
  
  //size_t size  = H5Tget_size ( mRanks.mDatatype );
  mRanks.mDataspace = H5Dget_space ( mRanks.mDataset ); /* dataspace handle */
  int rank = H5Sget_simple_extent_ndims ( mRanks.mDataspace );
  if ( rank!=2 )
  {
    ION_ABORT ( "Internal Error: dimension wrong for ReadRanks" );
  }
  int status_n = H5Sget_simple_extent_dims ( mRanks.mDataspace, dims_out, NULL );
  if ( status_n<0 )
  {
    ION_ABORT ( "Internal Error: H5Sget_simple_extent_dims in ReadRanks" );
  }
  hsize_t count[2];       /* size of the hyperslab in the file */
  hsize_t offset[2];      /* hyperslab offset in the file */
  hsize_t count_out[2];   /* size of the hyperslab in memory */
  hsize_t offset_out[2];  /* hyperslab offset in memory */
  offset[0] = 0;
  offset[1] = 0;
  count[0] = dims_out[0];
  count[1] = dims_out[1];
  herr_t status = 0;
  status = H5Sselect_hyperslab ( mRanks.mDataspace, H5S_SELECT_SET, offset, NULL,
                                 count, NULL );
  if ( status < 0 )
  {
    ION_ABORT ( "Couldn't read ranks dataspace." );
  }

  hsize_t     dimsm[2];              /* memory space dimensions */
  hid_t       memspace;
  /*
   * Define the memory dataspace.
   */
  dimsm[0] = dims_out[0];
  dimsm[1] = dims_out[1];
  memspace = H5Screate_simple ( 2,dimsm,NULL );

  offset_out[0] = 0;
  offset_out[1] = 0;
  count_out[0] = dims_out[0];
  count_out[1] = dims_out[1];
  status = H5Sselect_hyperslab ( memspace, H5S_SELECT_SET, offset_out, NULL,
                                 count_out, NULL );
  if ( status < 0 )
  {
    ION_ABORT ( "Couldn't read ranks hyperslap." );
  }

  mRankData.resize ( count_out[0] * count_out[1] );
  fill ( mRankData.begin(), mRankData.end(), 0 );
  status = H5Dread ( mRanks.mDataset, H5T_NATIVE_FLOAT, memspace, mRanks.mDataspace,
                     H5P_DEFAULT, &mRankData[0] );
  if ( status < 0 )
  {
    ION_ABORT ( "Couldn't read ranks dataset." );
  }
  mRanks.Close();
}

void RawWells::ReadInfo()
{
  mInfo.Clear();
  vector<string> keys;
  vector<string> values;
  ReadStringVector ( mHFile, INFO_KEYS, keys );
  ReadStringVector ( mHFile, INFO_VALUES, values );
  if ( keys.size() != values.size() )
  {
    ION_ABORT ( "Keys and Values don't match in size." );
  }

  for ( size_t i = 0; i < keys.size(); i++ )
  {
    if ( !mInfo.SetValue ( keys[i], values[i] ) )
    {
      ION_ABORT ( "Error: Could not set key: " + keys[i] + " with value: " + values[i] );
    }
  }
  if ( !mInfo.GetValue ( FLOW_ORDER, mFlowOrder ) )
  {
    ION_ABORT ( "Error - Flow order is not set." );
  }
}

void RawWells::CleanupHdf5()
{
  mRanks.Close();
  mWells.Close();
  mInfoKeys.Close();
  mInfoValues.Close();
  if ( mHFile != RWH5DataSet::EMPTY )
  {
    H5Fclose ( mHFile );
    mHFile = RWH5DataSet::EMPTY;
  }
}

void RawWells::CloseWithoutCleanupHdf5()
{
  mChunk.rowStart = 0;
  mChunk.rowHeight = 0;
  mChunk.colStart = 0;
  mChunk.colWidth = 0;
  mChunk.flowStart = 0;
  mChunk.flowDepth = 0;
  mFlowData.resize ( 0 );
  mResData.resize ( 0 );
  mIndexes.resize ( 0 );
  mWriteSubset.resize ( 0 );
}

void RawWells::Close()
{
  CloseWithoutCleanupHdf5();
  CleanupHdf5();
}

void RawWells::GetRegion ( int &rowStart, int &height, int &colStart, int &width )
{
  rowStart = mChunk.rowStart;
  height = mChunk.rowHeight;
  colStart = mChunk.colStart;
  width = mChunk.colWidth;
}

void RawWells::SetFlowOrder ( const std::string &flowOrder )
{
  mFlowOrder.resize ( NumFlows() );
  for ( size_t i = 0; i < NumFlows(); i++ )
  {
    mFlowOrder.at ( i ) = flowOrder.at ( i % flowOrder.length() );
  }
}

void RawWells::IndexToXY ( size_t index, unsigned short &x, unsigned short &y ) const
{
  y = ( index / NumCols() );
  x = ( index % NumCols() );
}

void RawWells::IndexToXY ( size_t index, size_t &x, size_t &y ) const
{
  y = ( index / NumCols() );
  x = ( index % NumCols() );
}

void RawWells::Set ( size_t idx, size_t flow, float val )
{
  if ( mIndexes[idx] >= 0 )
  {
    uint64_t ii = ( uint64_t ) mIndexes[idx] * mChunk.flowDepth + flow - mChunk.flowStart;
    if (! ( ii < mFlowData.size())) printf(" %zu mIndexes[%zu], ii:%" PRIu64 " =  %d * %zu + %zu - %zu\n",  mFlowData.size(), idx, ii, mIndexes[idx], mChunk.flowDepth, flow, mChunk.flowStart );
    assert ( ii < mFlowData.size() );
    mFlowData[ii] = val;
	return;
  }

  if ( mIndexes[idx] == WELL_NOT_LOADED )
  {
    ION_ABORT ( "Well: " + ToStr ( idx ) + " is not loaded." );
  }
  if ( mIndexes[idx] == WELL_NOT_SUBSET )
  {
    ION_ABORT ( "Well: " + ToStr ( idx ) + " is not is specified subset." );
  }
  if ( flow < mChunk.flowStart || flow >= mChunk.flowStart + mChunk.flowDepth )
  {
    ION_ABORT ( "Flow: " + ToStr ( flow ) + " is not loaded in range: " +
                ToStr ( mChunk.flowStart ) + " to " + ToStr ( mChunk.flowStart+mChunk.flowDepth ) );
  }

  ION_ABORT ( "Don't recognize index: " + ToStr ( mIndexes[idx] ) );
}

void RawWells::SetRes ( size_t idx, size_t flow, float val )
{
  if ( mIndexes[idx] >= 0 )
  {
    uint64_t ii = ( uint64_t ) mIndexes[idx] * mChunk.flowDepth + flow - mChunk.flowStart;
    if (! ( ii < mResData.size())) printf(" %zu mIndexes[%zu], ii:%" PRIu64 " =  %d * %zu + %zu - %zu\n",  mResData.size(), idx, ii, mIndexes[idx], mChunk.flowDepth, flow, mChunk.flowStart );
    assert ( ii < mResData.size() );
    mResData[ii] = val;
  return;
  }
}

void RawWells::SetSubsetToLoad ( const std::vector<int32_t> &xSubset, const std::vector<int32_t> &ySubset )
{
  mWriteSubset.resize ( xSubset.size() );
  for ( size_t i = 0; i < mWriteSubset.size();i++ )
  {
    mWriteSubset[i] = ToIndex ( xSubset[i], ySubset[i] );
  }
}

void RawWells::SetSubsetToWrite ( const std::vector<int32_t> &subset )
{
  mWriteSubset = subset;
}

void RawWells::SetSubsetToLoad ( int32_t *xSubset, int32_t *ySubset, int count )
{
  mWriteSubset.resize ( count );
  for ( size_t i = 0; i < mWriteSubset.size();i++ )
  {
    mWriteSubset[i] = ToIndex ( xSubset[i], ySubset[i] );
  }
}

void RawWells::InitIndexes()
{
  mIndexes.resize ( NumRows() * NumCols() );
  fill ( mIndexes.begin(), mIndexes.end(), WELL_NOT_LOADED );

  uint64_t count = 0;
  if ( mWriteSubset.empty() )
  {
    for ( size_t row = mChunk.rowStart; row < mChunk.rowStart + mChunk.rowHeight; row++ )
    {
      for ( size_t col = mChunk.colStart; col < mChunk.colStart + mChunk.colWidth; col++ )
      {
        mIndexes[ToIndex ( col,row ) ] = count++;
      }
    }
  }
  else
  {
    for ( size_t row = mChunk.rowStart; row < mChunk.rowStart + mChunk.rowHeight; row++ )
    {
      for ( size_t col = mChunk.colStart; col < mChunk.colStart + mChunk.colWidth; col++ )
      {
        mIndexes[ToIndex ( col,row ) ] = WELL_NOT_SUBSET;
      }
    }
    for ( size_t i = 0; i < mWriteSubset.size(); i++ )
    {
      size_t x,y;
      IndexToXY ( mWriteSubset[i], x, y );
      if ( x >= mChunk.colStart && x < ( mChunk.colStart + mChunk.colWidth ) &&
           y >= mChunk.rowStart && y < ( mChunk.rowStart + mChunk.rowHeight ) )
      {
        mIndexes[ToIndex ( x,y ) ] = count++;
      }
    }
  }
  mFlowData.resize ( count * mChunk.flowDepth );
  fill ( mFlowData.begin(), mFlowData.end(), -1.0f );
  //Chao- need to find another default value for res data
  //mResData.resize ( count * mChunk.flowDepth );
  //fill ( mResData.begin(), mResData.end(), -100000.0f );
}

void RawWells::SetChunk ( size_t rowStart, size_t rowHeight,
                          size_t colStart, size_t colWidth,
                          size_t flowStart, size_t flowDepth )
{

  // cout << "Setting chunk: " << rowStart << "," << rowHeight << " "
  //      << colStart << "," << colWidth << " "
  //      << flowStart << "," << flowDepth << endl;
  /* Sanity check */
  if ( !mIsLegacy )
  {
    ION_ASSERT ( rowHeight < 10000 &&
                 colWidth < 10000 &&
                 flowDepth < 10000, "Illegal chunk." );
    mChunk.rowStart = rowStart;
    mChunk.rowHeight = rowHeight;
    mChunk.colStart = colStart;
    mChunk.colWidth = colWidth;
    mChunk.flowStart = flowStart;
    mChunk.flowDepth = flowDepth;
    InitIndexes();
  }
}

void RawWells::OpenExistingWellsForOneChunk(int start_of_chunk, int chunk_depth)
{
    SetChunk (0, NumRows(), 0, NumCols(), start_of_chunk, chunk_depth);
    OpenForReadWrite();
}

void RawWells::OpenExistingWellsForOneChunkWithoutReopenHdf5(int start_of_chunk, int chunk_depth)
{
    SetChunk (0, NumRows(), 0, NumCols(), start_of_chunk, chunk_depth);
    InitIndexes();
}

ChunkyWells::ChunkyWells( 
    const char *experimentPath, 
    const char *rawWellsName,
    int _idealChunkSize,
    int _startingFlow,
    int _endingFlow
  ) :
  RawWells( experimentPath, rawWellsName ),
  idealChunkSize( _idealChunkSize ),
  startingFlow( _startingFlow ),
  endingFlow( _endingFlow )
{
  // Now that we're initialized, let's set up the first flow.
  StartChunk( startingFlow );
  pthread_mutex_init(&mMutex4Pending, NULL);
}

ChunkyWells::~ChunkyWells()
{
  pthread_mutex_destroy(&mMutex4Pending);
}

void ChunkyWells::StartChunk( int chunkStart )
{
  // Well... How much of the chunk can we actually get to?
  int chunkDepth = min( idealChunkSize, endingFlow - chunkStart );

  // Okay. Let's do that much.
  OpenExistingWellsForOneChunk( chunkStart, chunkDepth );
}

void ChunkyWells::StartChunkWithoutReopenHdf5( int chunkStart )
{
  // Well... How much of the chunk can we actually get to?
  int chunkDepth = min( idealChunkSize, endingFlow - chunkStart );

  // Okay. Let's do that much.
  OpenExistingWellsForOneChunkWithoutReopenHdf5( chunkStart, chunkDepth );
}

void ChunkyWells::MarkFlowComplete( int flow )
{
  // For now, we'll do this serially.
  if ( static_cast<size_t>(flow) + 1 != mChunk.flowStart + mChunk.flowDepth )
  {
    return;
  }

  // It's the end of the current chunk, so write it out.
  WriteWells();
  //WriteRes();
  Close();

  // If we're not going to do another flow, return now.
  if ( flow == endingFlow - 1 )
  {
    return;
  }

  // Get ready for the next chunk.
  StartChunk( flow + 1 );
}

void ChunkyWells::DoneUpThroughFlow( int flow )
{
  int nextFlow = mChunk.flowStart + mChunk.flowDepth;

  // Did we finish our chunk?
  if (flow + 1 < nextFlow )
  {
    // Nope. We don't care.
    return;
  }

  // Ooh... This worked.
  // Write it out. (Everyone else can keep running, in a buffer).
  cout << "Writing wells file up to flow " << nextFlow << " from thread " << pthread_self() << endl;

  WriteWells();
  //WriteRes();

  Close();

  // Advance to the next flow.
  // If we're not going to do another flow, return now.
  if ( nextFlow == endingFlow )
  {
    return;
  }

  // Get ready for the next chunk.
  StartChunk( nextFlow );

  // Grab everything we've saved, then let the jobs continue.
  vector< FlowGram > localFlowgrams;
  swap( localFlowgrams, pendingFlowgrams );
  vector< float > localCopies;
  if(mSaveCopies)
  {
	swap( localCopies, pendingCopies );
  }

  // We can be just another thread writing stuff out.
  // As an added bonus, if we do things this way, we can go through the flowgram filter again,
  // in case we've gotten more than one chunk ahead of ourselves.
  int k = -1;
  if(localCopies.size() == localFlowgrams.size())
  {
	  k = 0;
  }
  for( size_t i = 0 ; i < localFlowgrams.size() ; ++i )
  {
    const FlowGram & fg = localFlowgrams[i];

	if(k >= 0)
	{
		WriteFlowgram( fg.flow, fg.x, fg.y, fg.val, localCopies[k], fg.res);
		++k;
	}
	else
	{
		WriteFlowgramWithRes( fg.flow, fg.x, fg.y, fg.val, fg.res);
	}
  }
}

void ChunkyWells::DoneUpThroughFlow( int flow, SemQueue & packQueue, SemQueue & writeQueue )
{
  int nextFlow = mChunk.flowStart + mChunk.flowDepth;

  // need not dequeue if we have not reached flow depth worth of flows
  if (flow + 1 < nextFlow) {
    return;
  }

  ChunkFlowData* chunkData = NULL;
  while(NULL == chunkData)
  {
    chunkData = packQueue.deQueue();
  }

  if ( nextFlow == endingFlow )
  {
    chunkData->lastFlow = true;
  }

  chunkData->wellChunk = mChunk;

  copy(mFlowData.begin(), mFlowData.end(), chunkData->flowData);
  //copy(mResData.begin(), mResData.end(), chunkData->resData);
  copy(mIndexes.begin(), mIndexes.end(), chunkData->indexes);

  writeQueue.enQueue(chunkData);

  CloseWithoutCleanupHdf5();

  if ( nextFlow == endingFlow )
  {
    return;
  }

  // Get ready for the next chunk.
  StartChunkWithoutReopenHdf5( nextFlow );

  // Grab everything we've saved, then let the jobs continue.
  vector< FlowGram > localFlowgrams;
  swap( localFlowgrams, pendingFlowgrams );
  vector< float > localCopies;
  if(mSaveCopies)
  {
	swap( localCopies, pendingCopies );
  }

  // We can be just another thread writing stuff out.
  // As an added bonus, if we do things this way, we can go through the flowgram filter again,
  // in case we've gotten more than one chunk ahead of ourselves.
  int k = -1;
  if(localCopies.size() == localFlowgrams.size())
  {
	  k = 0;
  }
  for( size_t i = 0 ; i < localFlowgrams.size() ; ++i )
  {
    const FlowGram & fg = localFlowgrams[i];

	if(k >= 0)
	{
		WriteFlowgram( fg.flow, fg.x, fg.y, fg.val, localCopies[k], fg.res );
		++k;
	}
	else
	{
		WriteFlowgramWithRes( fg.flow, fg.x, fg.y, fg.val, fg.res );
	}
  }
}

void ChunkyWells::WriteFlowgram( size_t flow, size_t x, size_t y, float val )
{
  if ( flow < mChunk.flowStart + mChunk.flowDepth )
  {
    RawWells::WriteFlowgram( flow, x, y, val );
    return;
  }

  // Save it.
  FlowGram fg;
  fg.flow = flow;
  fg.x = x;
  fg.y = y;
  fg.val = val;

  pthread_mutex_lock(&mMutex4Pending);
  pendingFlowgrams.push_back( fg );
  pthread_mutex_unlock(&mMutex4Pending);
}

void ChunkyWells::WriteFlowgram( size_t flow, size_t x, size_t y, float val, float copies )
{
  if ( flow < mChunk.flowStart + mChunk.flowDepth )
  {
    RawWells::WriteFlowgram( flow, x, y, val, copies );
    return;
  }

  // Save it.
  FlowGram fg;
  fg.flow = flow;
  fg.x = x;
  fg.y = y;
  fg.val = val;

  pthread_mutex_lock(&mMutex4Pending);
  pendingFlowgrams.push_back( fg );
  pendingCopies.push_back( copies );
  pthread_mutex_unlock(&mMutex4Pending);
}

//with residual, with copies
void ChunkyWells::WriteFlowgram( size_t flow, size_t x, size_t y, float val, float copies, float resError )
{
  if ( flow < mChunk.flowStart + mChunk.flowDepth )
  {
    RawWells::WriteFlowgram( flow, x, y, val, copies, resError );
    return;
  }

  // Save it.
  FlowGram fg;
  fg.flow = flow;
  fg.x = x;
  fg.y = y;
  fg.val = val;
  fg.res = resError;

  pthread_mutex_lock(&mMutex4Pending);
  pendingFlowgrams.push_back( fg );
  pendingCopies.push_back( copies );
  pthread_mutex_unlock(&mMutex4Pending);
}

// with residual, without copies
void ChunkyWells::WriteFlowgramWithRes(size_t flow, size_t x, size_t y, float val, float resError)
{
  if ( flow < mChunk.flowStart + mChunk.flowDepth )
  {
    RawWells::WriteFlowgramWithRes( flow, x, y, val, resError );
    return;
  }

  // Save it.
  FlowGram fg;
  fg.flow = flow;
  fg.x = x;
  fg.y = y;
  fg.val = val;
  fg.res = resError;

  pthread_mutex_lock(&mMutex4Pending);
  pendingFlowgrams.push_back( fg );
  pthread_mutex_unlock(&mMutex4Pending);
}
