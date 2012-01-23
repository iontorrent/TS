/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RawWells.h"
#include "Mask.h"
#include "Separator.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include "Utils.h"
#include "LinuxCompat.h"
#include "IonErr.h"
#include "dbgmem.h"
#include "RawWellsV1.h"
#define WELLS "wells"
#define RANKS "ranks"
#define INFO_KEYS "info_keys"
#define INFO_VALUES "info_values"
#define FLOW_ORDER "FLOW_ORDER"

using namespace std;
#define RAWWELLS_VERSION_KEY "RAWWELLS_VERSION"
#define RAWWELLS_VERSION_VALUE "2"


RWH5DataSet::RWH5DataSet() {
  Clear();
}

void RWH5DataSet::Clear() {
  mGroup = "/";
  mName = "/";
  mDataset = EMPTY;
  mDatatype = EMPTY;
  mDataspace = EMPTY;
}

RWH5DataSet::~RWH5DataSet() {
  Close();
}

void RWH5DataSet::Close() {
  if (mDataset != EMPTY) {
    H5Sclose(mDataspace);
    mDataspace = EMPTY;
    H5Tclose(mDatatype);
    mDatatype = EMPTY;
    H5Dclose(mDataset);
    mDataset = EMPTY;
  }
}

bool Info::GetValue(const std::string &key, std::string &value) const {
  if (mMap.find(key) == mMap.end()) {
    return false;
  }
  value =  mValues[mMap.find(key)->second];
  return true;
}
  
bool Info::SetValue(const std::string &key, const std::string &value) {
  if (mMap.find(key) != mMap.end()) {
    mValues[mMap[key]] = value;
  }
  else {
    mKeys.push_back(key);
    mValues.push_back(value);
    mMap[key] = int(mKeys.size()) - 1;
  }
  return true;
}

bool Info::GetEntry(int index, std::string &key, std::string &value) const {
  if (index >= (int) mKeys.size() || index < 0) {
    ION_ABORT("ERROR - Index: " + ToStr(index) + " not valid for max index: " + ToStr(mKeys.size() -1));
    return false;
  }
  key = mKeys[index];
  value = mValues[index];
  return true;
}

int Info::GetCount() const {
  return (int) mValues.size();
}

bool Info::KeyExists(const std::string &key) const {
  if (mMap.find(key) != mMap.end()) {
    return true;
  }
  return false;
}

void Info::Clear() {
  mKeys.clear();
  mValues.clear();
  mMap.clear();
}

RawWells::RawWells(const char *experimentPath, const char *rawWellsName, int rows, int cols) {
  Init(experimentPath, rawWellsName, rows, cols, 0);
}

RawWells::RawWells(const char *experimentPath, const char *rawWellsName) {
  Init(experimentPath, rawWellsName, 0, 0, 0);
}

RawWells::RawWells(const char *wellsFilePath, int rows, int cols) { 
  Init("", wellsFilePath, rows, cols, 0);
}

RawWells::~RawWells() {
  Close();
}

void RawWells::Init(const char *experimentPath, const char *rawWellsName, int rows, int cols, int flows) {
  mIsLegacy = false;
  WELL_NOT_LOADED = -1;
  WELL_NOT_SUBSET = -2;
  mWellChunkSize = 50;
  mFlowChunkSize = 60;
  mStepSize = 100;
  mCurrentRegionRow = mCurrentRegionCol = 0;
  mCurrentRow = 0;
  mCurrentCol = -1;
  mDirectory = experimentPath;
  mFileLeaf = rawWellsName;
  if (mDirectory != "") {
    mFilePath = mDirectory + "/" + mFileLeaf;
  }
  else {
    mFilePath = mFileLeaf;
  }
  mCurrentWell = 0;
  mCompression = 3;
  SetRows(rows);
  SetCols(cols);
  SetFlows(flows);
  data.flowValues = NULL;
  //  mWriteOnClose = false;
  mHFile = RWH5DataSet::EMPTY;
  mInfo.SetValue(RAWWELLS_VERSION_KEY, RAWWELLS_VERSION_VALUE);

}

void RawWells::CreateEmpty(int numFlows, const char *flowOrder, int rows, int cols) {
  SetRows(rows);
  SetCols(cols);
  CreateEmpty(numFlows, flowOrder);
  //  mWriteOnClose = true;
}

void RawWells::CreateEmpty(int numFlows, const char *flowOrder) { 
  SetFlows(numFlows);
  SetFlowOrder(flowOrder);
  CreateBuffers();
  //  mWriteOnClose = true;
}

void RawWells::SetRegion(int rowStart, int height, int colStart, int width) {
  SetChunk(rowStart, height, colStart, width, 0, NumFlows());
}

void RawWells::OpenForWrite() {
  //  mWriteOnClose = true;
  CreateBuffers();
  if (mHFile == RWH5DataSet::EMPTY) {
    mHFile = H5Fcreate(mFilePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }
  if (mHFile < 0) { ION_ABORT("Couldn't create file: " + mFilePath); }
  OpenWellsForWrite();
}

void RawWells::WriteFlowgram(size_t flow, size_t x, size_t y, float val) {
  uint64_t idx = ToIndex(x,y);
  Set(idx, flow, val);
}

bool RawWells::OpenForReadLegacy() {
  //  mWriteOnClose = false;
  RawWellsV1 orig(mDirectory.c_str(), mFileLeaf.c_str());
  bool failed = orig.OpenForRead(false);
  if (failed) {
    return failed;
  }
  WellData o;
  o.flowValues = (float *) malloc (orig.NumFlows() * sizeof(float));
  mFlowOrder = orig.FlowOrder();
  if (!mInfo.SetValue(FLOW_ORDER, mFlowOrder)) { ION_ABORT("Error - Could not set flow order."); }
  mFlowData.resize((uint64_t)orig.NumWells() * orig.NumFlows());
  fill(mFlowData.begin(), mFlowData.end(), 0.0f);
  mIndexes.resize(orig.NumWells());
  fill(mIndexes.begin(), mIndexes.end(), WELL_NOT_LOADED);
  mRankData.resize(orig.NumWells());
  fill(mRankData.begin(), mRankData.end(), 0);
  SetFlows(orig.NumFlows());
  mRows = 0;
  mCols = 0;
  mCurrentWell = 0;
  size_t count = 0;
  while(!orig.ReadNextData(&o)) {
    mRows = std::max((uint64_t)o.y, mRows);
    mCols = std::max((uint64_t)o.x, mCols);
    mRankData[count] = o.rank;
    copy(o.flowValues, o.flowValues + mFlows, mFlowData.begin() + count * mFlows);
    mIndexes[count] = count;
    count++;
  }

  mRows += 1; // Number is 1 higher than largest index.
  mCols += 1; 
  SetRows(mRows);
  SetCols(mCols);
  SetFlows(mFlows);
  orig.Close();
  mIsLegacy = true;
  return (false); // no error
}
  

void RawWells::ReadWell(int wellIdx, WellData *_data) {
  IndexToXY(wellIdx, _data->x, _data->y);
  if (mZeros.empty()) { mZeros.resize(mFlows); }
  _data->rank = mRankData[wellIdx];
  if (mIndexes[wellIdx] == WELL_NOT_LOADED) {
    ION_ABORT("Well: " + ToStr(wellIdx) + " is not loaded.");
  }
  if (mIndexes[wellIdx] >= 0) {
    for (size_t i = 0; i < mFlows; i++) {
      mZeros[i] = At(wellIdx, i);
    }
    _data->flowValues = &mZeros[0];
  }
  else if (mIndexes[wellIdx] == WELL_NOT_SUBSET) {    
    fill(mZeros.begin(), mZeros.end(), 0.0f); 
    _data->flowValues = &mZeros[0];
  }
  else {
    ION_ABORT("Don't recognize index: " + ToStr(mIndexes[wellIdx]));
  }
}

bool RawWells::InChunk(size_t row, size_t col) {
  return (row >= mChunk.rowStart && row < mChunk.rowStart + mChunk.rowHeight &&
          col >= mChunk.colStart && col < mChunk.colStart + mChunk.colWidth);
}

size_t RawWells::GetNextRegionData() {
  mCurrentCol++;
  bool reload = false;
  if (mCurrentCol == 0 && !mIsLegacy) {
    SetChunk(mCurrentRegionRow, min(max(mRows - mCurrentRegionRow,0ul), mWellChunkSize),
             mCurrentRegionCol, min(max(mCols - mCurrentRegionCol,0ul), mWellChunkSize),
             0, mFlows);
    ReadWells();
  }
  if (mCurrentCol >= mWellChunkSize || mCurrentRegionCol + mCurrentCol >= mCols) {
    mCurrentCol = 0;
    mCurrentRow++;
    if (mCurrentRow >= mWellChunkSize || mCurrentRegionRow + mCurrentRow >= mRows) {
      reload = true;
      mCurrentRow = 0;
      mCurrentRegionCol += mWellChunkSize;
      if (mCurrentRegionCol >= mCols) {
        mCurrentRegionCol = 0;
        mCurrentRegionRow += mWellChunkSize;
      }
    }
  }
  /* Load up the right region... */
  size_t row = mCurrentRegionRow + mCurrentRow;
  size_t col = mCurrentRegionCol + mCurrentCol;
  if (reload && mCurrentRegionRow < mRows && !InChunk(row,col)) { 
        SetChunk(mCurrentRegionRow, min(max(mRows - mCurrentRegionRow,0ul), mWellChunkSize),
                 mCurrentRegionCol, min(max(mCols - mCurrentRegionCol,0ul), mWellChunkSize),
                 0, mFlows);
        ReadWells();
  }
  return ToIndex(col, row);
}

const WellData *RawWells::ReadNextRegionData() {
  bool val = ReadNextRegionData(&data);
  if (val) {
    return &data;
  }
  return NULL;
}

bool RawWells::ReadNextRegionData(WellData *_data) {
  size_t well = GetNextRegionData();
  if (well >= NumWells()) {
    return true;
  }
  ReadWell(well,_data);
  return (_data == NULL);
}

/** 
 * Not thread safe. One call to this function will overwrite the
 * previous.  Don't use this function if you can help it
 */
const WellData *RawWells::ReadXY(int x, int y) {
  int well = ToIndex(x,y);
  ReadWell(well, &data);
  return &data;
}

void RawWells::CreateBuffers() { 
  if (!mInfo.SetValue(FLOW_ORDER, mFlowOrder)) { ION_ABORT("Error - Could not set flow order."); }
  mRankData.resize(NumRows() * NumCols());
  fill(mRankData.begin(), mRankData.end(), 0);
  InitIndexes();
  uint64_t wellCount = 0;
  for (size_t i = 0; i < mIndexes.size(); i++) {
    if (mIndexes[i] >= 0) {
      wellCount++;
    }
  }
  mFlowData.resize((uint64_t)wellCount * mChunk.flowDepth);
  fill(mFlowData.begin(), mFlowData.end(), -1.0f);
}

void RawWells::WriteToHdf5(const std::string &file) {
  if (mHFile == RWH5DataSet::EMPTY) {
    mHFile = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }
  if (mHFile < 0) { ION_ABORT("Couldn't create file: " + mFilePath); }
  OpenWellsForWrite();
  WriteWells();
  WriteRanks();
  WriteInfo();
  CleanupHdf5();
}


void RawWells::WriteWells() {
  mInputBuffer.resize(mStepSize*mStepSize*mChunk.flowDepth);
  fill(mInputBuffer.begin(), mInputBuffer.end(), 0);
  uint32_t currentRowStart = mChunk.rowStart, currentRowEnd = mChunk.rowStart + min(mChunk.rowHeight, mStepSize);
  uint32_t currentColStart = mChunk.colStart, currentColEnd = mChunk.colStart + min(mChunk.colWidth, mStepSize);
  hsize_t count[3];       /* size of the hyperslab in the file */
  hsize_t offset[3];      /* hyperslab offset in the file */
  hsize_t count_out[3];   /* size of the hyperslab in memory */
  hsize_t offset_out[3];  /* hyperslab offset in memory */
  hsize_t dimsm[3];       /* memory space dimensions */

  /* For each chunk of data pad out the non-live (non subset) wells with 
     zeros before writing to disk. */
  for (currentRowStart = 0, currentRowEnd = mStepSize; 
       currentRowStart < mChunk.rowStart + mChunk.rowHeight; 
       currentRowStart = currentRowEnd, currentRowEnd += mStepSize) {
    currentRowEnd = min((uint32_t)(mChunk.rowStart + mChunk.rowHeight), currentRowEnd);
    for (currentColStart = 0, currentColEnd = mStepSize; 
         currentColStart < mChunk.colStart + mChunk.colWidth;
         currentColStart = currentColEnd, currentColEnd += mStepSize) {
      currentColEnd = min((uint32_t)(mChunk.colStart + mChunk.colWidth), currentColEnd);

      /* file offsets. */
      offset[0] = currentRowStart;
      offset[1] = currentColStart;
      offset[2] = mChunk.flowStart;
      count[0] = currentRowEnd - currentRowStart;
      count[1] = currentColEnd - currentColStart;
      count[2] = mChunk.flowDepth;
      herr_t status = 0;
      status = H5Sselect_hyperslab(mWells.mDataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
      if (status < 0) { ION_ABORT("Couldn't read wells dataspace for: " + ToStr(mChunk.rowStart) + "," + ToStr(mChunk.colStart) + "," + 
                                  ToStr(mChunk.rowHeight) + "," + ToStr(mChunk.colWidth) + " x " + ToStr(mChunk.flowDepth)); }
      
      hid_t       memspace;
      /*
       * Define the memory dataspace.
       */
      dimsm[0] = count[0];
      dimsm[1] = count[1];
      dimsm[2] = count[2];
      memspace = H5Screate_simple(3,dimsm,NULL);
      
      offset_out[0] = 0;
      offset_out[1] = 0;
      offset_out[2] = 0;
      count_out[0] = currentRowEnd - currentRowStart;
      count_out[1] = currentColEnd - currentColStart;
      count_out[2] = mChunk.flowDepth;
      // ClockTimer timer;
      // timer.StartTimer();
      status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
                                   count_out, NULL);
      if (status < 0) { ION_ABORT("Couldn't read wells hyperslab for: " + 
                                  ToStr(mChunk.rowStart) + "," + ToStr(mChunk.colStart) + "," + 
                                  ToStr(mChunk.rowHeight) + "," + ToStr(mChunk.colWidth) + " x " + 
                                  ToStr(mChunk.flowDepth)); }
      
      mInputBuffer.resize((uint64_t) (currentRowEnd - currentRowStart) * 
                          (uint64_t) (currentColEnd - currentColStart) * 
                          (uint64_t) mChunk.flowDepth);
      int idxCount = 0;
      for (size_t row = currentRowStart; row < currentRowEnd; row++) {
        for (size_t col = currentColStart; col < currentColEnd; col++) {
          int idx = ToIndex(col, row);
          for (size_t fIx = mChunk.flowStart; fIx < mChunk.flowStart + mChunk.flowDepth; fIx++) {
            uint64_t ii = idxCount * mChunk.flowDepth + fIx - mChunk.flowStart;
            assert(ii < mInputBuffer.size());
            mInputBuffer[ii] = At(idx, fIx);
          }
          idxCount++;
        }
      }
      status = H5Dwrite(mWells.mDataset, H5T_NATIVE_FLOAT, memspace, mWells.mDataspace,
                        H5P_DEFAULT, &mInputBuffer[0]);
      if (status < 0) { 
        ION_ABORT("ERRROR - Unsuccessful write to file: " + 
                  ToStr(mChunk.rowStart) + "," + ToStr(mChunk.colStart) + "," + 
                  ToStr(mChunk.rowHeight) + "," + ToStr(mChunk.colWidth) + " x " + 
                  ToStr(mChunk.flowStart) + "," + ToStr(mChunk.flowDepth) + "\t" + mFilePath); }
    }
  }
}

void RawWells::OpenWellsForWrite() {
  mIsLegacy = false;
  mWells.mName = WELLS;
  // Create wells data space
  hsize_t dimsf[3]; 
  dimsf[0] = mRows;
  dimsf[1] = mCols;
  dimsf[2] = mFlows;
  mWells.mDataspace = H5Screate_simple(3, dimsf, NULL);
  mWells.mDatatype = H5Tcopy(H5T_NATIVE_FLOAT);
    
  // Setup the chunking values, this is important for performance
  hsize_t cdims[3];
  cdims[0] = std::min(mWellChunkSize, mRows);
  cdims[1] = std::min(mWellChunkSize, mCols);
  cdims[2] = std::min(mFlowChunkSize, mFlows);
  hid_t plist;
  plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist, 3, cdims);
  if (mCompression > 0) {
    H5Pset_deflate(plist, mCompression);
  }
  hid_t dapl;
  dapl = H5Pcreate (H5P_DATASET_ACCESS);
  mWells.mDataset = H5Dcreate2(mHFile, mWells.mName.c_str(), mWells.mDatatype, mWells.mDataspace,
			       H5P_DEFAULT, plist, dapl);
}

void RawWells::WriteRanks() {
  mRanks.mName = RANKS;
  // Create wells data space
  hsize_t dimsf[2]; 
  dimsf[0] = mRows;
  dimsf[1] = mCols;
  mRanks.mDataspace = H5Screate_simple(2, dimsf, NULL);
  mRanks.mDatatype = H5Tcopy(H5T_NATIVE_INT);
    
  // Setup the chunking values
  hsize_t cdims[2];
  cdims[0] = std::min((uint64_t)50, mRows);
  cdims[1] = std::min((uint64_t)50, mCols);

  hid_t plist;
  plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist, 2, cdims);
  if (mCompression > 0) {
    H5Pset_deflate(plist, mCompression);
  }

  mRanks.mDataset = H5Dcreate2(mHFile, mRanks.mName.c_str(), mRanks.mDatatype, mRanks.mDataspace,
			       H5P_DEFAULT, plist, H5P_DEFAULT);
  herr_t status = H5Dwrite(mRanks.mDataset, mRanks.mDatatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mRankData[0]);
  if (status < 0) { ION_ABORT("ERRROR - Unsuccessful write to file: " + mFilePath); }
  mRanks.Close();
}

void RawWells::WriteStringVector(hid_t h5File, const std::string &name, const char **values, int numValues) {
  RWH5DataSet set;
  set.mName = name;
  // Create info dataspace
  hsize_t dimsf[1]; 
  dimsf[0] = numValues;
  set.mDataspace = H5Screate_simple(1, dimsf, NULL);
  set.mDatatype = H5Tcopy(H5T_C_S1);
  herr_t status = 0;
  status = H5Tset_size(set.mDatatype, H5T_VARIABLE);
  if (status < 0) { ION_ABORT("Couldn't set string type to variable in set: " + name); }
  
  hid_t memType = H5Tcopy(H5T_C_S1);
  status = H5Tset_size(memType, H5T_VARIABLE);
  if (status < 0) { ION_ABORT("Couldn't set string type to variable in set: " + name); }
    
  // Setup the chunking values
  hsize_t cdims[1];
  cdims[0] = std::min(100, numValues);

  hid_t plist;
  plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist, 1, cdims);
  if (mCompression > 0) {
    H5Pset_deflate(plist, mCompression);
  }

  set.mDataset = H5Dcreate2(h5File, set.mName.c_str(), set.mDatatype, set.mDataspace,
			    H5P_DEFAULT, plist, H5P_DEFAULT);
  status = H5Dwrite(set.mDataset, set.mDatatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
  if (status < 0) { ION_ABORT("ERRROR - Unsuccessful write to file: " + mFilePath + " dataset: " + set.mName); }
  set.Close();
}

void RawWells::ReadStringVector(hid_t h5File, const std::string &name, std::vector<std::string> &strings) {
  strings.clear();
  RWH5DataSet set;
  
  set.mDataset = H5Dopen2(h5File, name.c_str(), H5P_DEFAULT);
  if (set.mDataset < 0) { ION_ABORT("Could not open data set: " + name); }
  int status_n, rank;

  set.mDatatype  = H5Dget_type(set.mDataset);     /* datatype handle */
  size_t size;
  size  = H5Tget_size(set.mDatatype);
  set.mDataspace = H5Dget_space(set.mDataset);    /* dataspace handle */
  rank = H5Sget_simple_extent_ndims(set.mDataspace);
  hsize_t dims[rank];           /* dataset dimensions */
  status_n = H5Sget_simple_extent_dims(set.mDataspace, dims, NULL);

  char **rdata = (char**) malloc(dims[0] * sizeof (char *));

  hid_t memType = H5Tcopy(H5T_C_S1); // C style strings
  herr_t status = H5Tset_size(memType, H5T_VARIABLE); // Variable rather than fixed length
  if (status < 0) { ION_ABORT("Error setting string length to variable."); }

  // Read the strings 
  status = H5Dread(set.mDataset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
  if (status < 0) { ION_ABORT("Error reading " + name); }

  // Put into output string
  strings.resize(dims[0]);
  for (size_t i = 0; i < dims[0]; i++) {
    strings[i] = rdata[i];
  }
  
  status = H5Dvlen_reclaim(memType, set.mDataspace, H5P_DEFAULT, rdata);
  free(rdata);
  set.Close();
}

void RawWells::WriteInfo() {
  int numEntries = mInfo.GetCount();
  vector<string> keys(numEntries);
  vector<string> values(numEntries);
  vector<const char *> keysBuff(numEntries);
  vector<const char *> valuesBuff(numEntries);
  for (int i = 0; i < numEntries; i++) {
    mInfo.GetEntry(i, keys[i], values[i]);
    keysBuff[i] = keys[i].c_str();
    valuesBuff[i] = values[i].c_str();
  }
  WriteStringVector(mHFile, INFO_KEYS, &keysBuff[0], keysBuff.size());
  WriteStringVector(mHFile, INFO_VALUES, &valuesBuff[0], keysBuff.size());
}

bool RawWells::OpenMetaData() {
  OpenForIncrementalRead();
  return true;
}

bool RawWells::OpenForRead(bool memmap_dummy) {
  //  mWriteOnClose = false;
  H5E_auto2_t old_func; 
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2( H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto2( H5E_DEFAULT, NULL, NULL);
  if (mHFile == RWH5DataSet::EMPTY) {
    mHFile = H5Fopen(mFilePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  }
  if (mHFile >= 0) {
    mIsLegacy = false;
    mCurrentWell = 0;
    OpenWellsToRead();
    ReadWells();
    ReadRanks();
    ReadInfo();
  }
  else if (mHFile < 0) {
    bool failed = OpenForReadLegacy();
    if (failed) {
      ION_ABORT("Couldn't open: " + mFilePath + " to read.");
    }
  }
  H5Eset_auto2( H5E_DEFAULT, old_func, old_client_data);
  return false;
}

void RawWells::WriteLegacyWells() {
  WellHeader hdr;
  hdr.flowOrder = NULL;
  int flowCnt = NumFlows();
  FILE *fp = NULL;
  fopen_s(&fp, mFilePath.c_str(), "wb");
  const char *flowOrder = FlowOrder();
  if (fp) {
    // generate and write the header
    hdr.numWells = NumWells();
    hdr.numFlows = flowCnt;
    hdr.flowOrder = (char *)malloc(hdr.numFlows);
    for(int i=0;i<hdr.numFlows;i++) {
      hdr.flowOrder[i] = flowOrder[i%flowCnt];
    }
    fwrite(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
    fwrite(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
    fwrite(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);

    // write the well data, initially all zeros, to create the file
    int dataBlockSize = sizeof(unsigned int) + 2 * sizeof(unsigned short);
    int flowValuesSize = sizeof(float) * hdr.numFlows;
    uint64_t offset;

    float* wellPtr = NULL;
    for (uint64_t y=0;y<mRows;y++) {
      for (uint64_t x=0;x<mCols;x++) {
        data.rank = 0;//data.rank = i;
        data.x = (int)x;
        data.y = (int)y;
        fwrite(&data, dataBlockSize, 1, fp);
        offset = ToIndex(x,y);
        wellPtr = &mFlowData[mIndexes[offset] * NumFlows()];
        fwrite(wellPtr, flowValuesSize, 1, fp);
      }
    }
    free(hdr.flowOrder);
    fclose(fp);
    fp = NULL;
  }
}

void RawWells::OpenWellsToRead() {
  mWells.Close();
  mWells.mName = WELLS;
  mWells.mDataset = H5Dopen2(mHFile, mWells.mName.c_str(), H5P_DEFAULT);
  if (mWells.mDataset < 0) { ION_ABORT("Could not open data set: " + mWells.mName + 
                                       " in file: " + mFilePath); } 
  int status_n, rank;
  hsize_t dims_out[3];           /* dataset dimensions */
  mWells.mDatatype  = H5Dget_type(mWells.mDataset);     /* datatype handle */
  size_t size;
  size  = H5Tget_size(mWells.mDatatype);
  mWells.mDataspace = H5Dget_space(mWells.mDataset);    /* dataspace handle */
  rank = H5Sget_simple_extent_ndims(mWells.mDataspace);
  status_n = H5Sget_simple_extent_dims(mWells.mDataspace, dims_out, NULL);
  mRows = dims_out[0];
  mCols = dims_out[1];
  mFlows = dims_out[2];
  // Assume whole chip if region if region isn't set
  if (mChunk.rowHeight <= 0) {
    mChunk.rowStart = 0;
    mChunk.rowHeight = mRows;
  }
  if (mChunk.colWidth <= 0) {
    mChunk.colStart = 0;
    mChunk.colWidth = mCols;
  }
  if (mChunk.flowDepth <= 0) {
    mChunk.flowStart = 0;
    mChunk.flowDepth = mFlows;
  }
  InitIndexes();
}

void RawWells::OpenForIncrementalRead() {
  //  mWriteOnClose = false;
  H5E_auto2_t old_func; 
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2( H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto2( H5E_DEFAULT, NULL, NULL);
  mHFile = H5Fopen(mFilePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (mHFile >= 0) {
    mIsLegacy = false;
    mCurrentWell = 0;
    OpenWellsToRead();
    ReadRanks();
    ReadInfo();
  }
  else if (mHFile < 0) {
    ION_WARN("Legacy wells file format, loading into memory.");
    bool failed = OpenForReadLegacy();
    if (failed) {
      ION_ABORT("Couldn't open: " + mFilePath + " to read.");
    }
    SetChunk(0,mRows, 0, mCols, 0, mFlows);
  }
  H5Eset_auto2( H5E_DEFAULT, old_func, old_client_data);
}

void RawWells::ReadWells() {
  if (mIsLegacy) {
    return;
  }
  vector<float> inputBuffer; // 100x100
  uint64_t stepSize = mWellChunkSize;
  inputBuffer.resize(stepSize*stepSize*mChunk.flowDepth,0.0f);
  // Assume whole chip if region if region isn't set
  //  int currentWellStart = 0, currentWellEnd = min(stepSize,NumWells());
  uint32_t currentRowStart = 0, currentRowEnd = min(stepSize,(uint64_t)mRows);
  uint32_t currentColStart = 0, currentColEnd = min(stepSize,(uint64_t)mCols);

  size_t wellsInSubset = 0;
  for (size_t i = 0; i < mIndexes.size(); i++) {
    if (mIndexes[i] >= 0) {
      wellsInSubset++;
    }
  }
  mFlowData.resize((uint64_t)wellsInSubset * mChunk.flowDepth);
  fill(mFlowData.begin(), mFlowData.end(), -1.0f);
  for (currentRowStart = mChunk.rowStart, currentRowEnd = mChunk.rowStart + min((size_t)stepSize, mChunk.rowHeight); 
       currentRowStart < mRows && currentRowStart < mChunk.rowStart + mChunk.rowHeight; 
       currentRowStart = currentRowEnd, currentRowEnd += stepSize) {
    currentRowEnd = min((uint32_t)mRows, currentRowEnd);
    for (currentColStart = mChunk.colStart, currentColEnd = mChunk.colStart + min((size_t)stepSize, mChunk.colWidth); 
         currentColStart < mCols && currentColStart < mChunk.colStart + mChunk.colWidth;
         currentColStart = currentColEnd, currentColEnd += stepSize) {
      currentColEnd = min((uint32_t)mCols, currentColEnd);
      // Don't go to disk unless we actually have a well to load.
      if (WellsInSubset(currentRowStart, currentRowEnd, currentColStart, currentColEnd)) {

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
        status = H5Sselect_hyperslab(mWells.mDataspace, H5S_SELECT_SET, offset, NULL,
                                     count, NULL);
        if (status < 0) { 
          ION_ABORT("Couldn't read wells dataspace for: " + 
                    ToStr(mChunk.rowStart) + "," + ToStr(mChunk.colStart) + "," + 
                    ToStr(mChunk.rowHeight) + "," + ToStr(mChunk.colWidth) + " x " + 
                    ToStr(mChunk.flowDepth)); 
        }
        
        hsize_t     dimsm[3];              /* memory space dimensions */
        hid_t       memspace;
        /*
         * Define the memory dataspace.
         */
        dimsm[0] = count[0];
        dimsm[1] = count[1];
        dimsm[2] = count[2];
        memspace = H5Screate_simple(3,dimsm,NULL);
        
        offset_out[0] = 0;
        offset_out[1] = 0;
        offset_out[2] = 0;
        count_out[0] = currentRowEnd - currentRowStart;
        count_out[1] = currentColEnd - currentColStart;
        count_out[2] = mChunk.flowDepth;
        // ClockTimer timer;
        // timer.StartTimer();
        status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
                                     count_out, NULL);
        if (status < 0) { ION_ABORT("Couldn't read wells hyperslab for: " + 
                                    ToStr(mChunk.rowStart) + "," + ToStr(mChunk.colStart) + "," + 
                                    ToStr(mChunk.rowHeight) + "," + ToStr(mChunk.colWidth) + " x " + 
                                    ToStr(mChunk.flowDepth)); 
        }
        
        inputBuffer.resize((uint64_t) (currentRowEnd - currentRowStart) * (currentColEnd - currentColStart) * mFlows);
        status = H5Dread(mWells.mDataset, H5T_NATIVE_FLOAT, memspace, mWells.mDataspace,
                         H5P_DEFAULT, &inputBuffer[0]);
        if (status < 0) { 
          ION_ABORT("Couldn't read wells dataset for file: "  + mFilePath + " dims: " + ToStr(mChunk.rowStart) + "," + 
                    ToStr(mChunk.colStart) + "," + ToStr(mChunk.rowHeight) + "," + 
                    ToStr(mChunk.colWidth) + " x " + ToStr(mChunk.flowDepth)); 
        }
        // std::cout << "Just read: " << currentRowStart  << "," << currentRowEnd << "," << 
        //   currentColStart << "," << currentColEnd << " x " << 
        //   mChunk.flowDepth << " in: " << timer.GetMicroSec() << "um seconds" << endl;
        uint64_t localCount = 0;
        for (size_t row = currentRowStart; row < currentRowEnd; row++) {
          for (size_t col = currentColStart; col < currentColEnd; col++) {
            int idx = ToIndex(col, row);
            if (mIndexes[idx] >= 0) {
              for (size_t flow = mChunk.flowStart; flow < mChunk.flowStart + mChunk.flowDepth; flow++) {
                Set(row, col, flow, inputBuffer[localCount * mChunk.flowDepth + flow]);
              }
            }
            localCount++;
          }
        }
      }
    }
  }
}

bool RawWells::WellsInSubset(uint32_t currentRowStart, uint32_t currentRowEnd, 
                             uint32_t currentColStart, uint32_t currentColEnd) {
  for (size_t row = currentRowStart; row < currentRowEnd; row++) {
    for (size_t col = currentColStart; col < currentColEnd; col++) {
      int idx = ToIndex(col, row);
      if (mIndexes[idx] == -3 || mIndexes[idx] >= 0) {
        return true;
      }
    }
  }
  return false;
}

float RawWells::At(size_t row, size_t col, size_t flow) const {
  return At(ToIndex(col,row), flow);
}

float RawWells::At(size_t well, size_t flow) const {
  if (mIndexes[well] == WELL_NOT_LOADED) {
    ION_ABORT("Well: " + ToStr(well) + " is not loaded.");
  }
  if (flow < mChunk.flowStart || flow >= mChunk.flowStart + mChunk.flowDepth) {
    ION_ABORT("Flow: " + ToStr(flow) + " is not loaded in range: " + 
              ToStr(mChunk.flowStart) + " to " + ToStr(mChunk.flowStart+mChunk.flowDepth));
  }
  if (mIndexes[well] >= 0) {
    uint64_t ii = (uint64_t) mIndexes[well] * mChunk.flowDepth + flow - mChunk.flowStart;
    assert(ii < mFlowData.size());
    return mFlowData[ii];
  }
  if (mIndexes[well] == WELL_NOT_SUBSET) {
    return 0;
  }
  ION_ABORT("Don't recognize index type: " + ToStr(mIndexes[well]));
  return 0;
}

void RawWells::ReadRanks() {
  mRanks.Close();
  mRanks.mName = RANKS;
  mRanks.mDataset = H5Dopen2(mHFile, mRanks.mName.c_str(), H5P_DEFAULT);
  if (mRanks.mDataset < 0) { ION_ABORT("Could not open data set: " + mRanks.mName); }
  int status_n, rank;
  hsize_t dims_out[2];           /* dataset dimensions */
  mRanks.mDatatype  = H5Dget_type(mRanks.mDataset);     /* datatype handle */
  size_t size;
  size  = H5Tget_size(mRanks.mDatatype);
  mRanks.mDataspace = H5Dget_space(mRanks.mDataset);    /* dataspace handle */
  rank = H5Sget_simple_extent_ndims(mRanks.mDataspace);
  status_n = H5Sget_simple_extent_dims(mRanks.mDataspace, dims_out, NULL);

  hsize_t count[2];       /* size of the hyperslab in the file */
  hsize_t offset[2];      /* hyperslab offset in the file */
  hsize_t count_out[2];   /* size of the hyperslab in memory */
  hsize_t offset_out[2];  /* hyperslab offset in memory */
  offset[0] = 0;
  offset[1] = 0;
  count[0] = dims_out[0];
  count[1] = dims_out[1];
  herr_t status = 0;
  status = H5Sselect_hyperslab(mRanks.mDataspace, H5S_SELECT_SET, offset, NULL,
			       count, NULL);
  if (status < 0) { ION_ABORT("Couldn't read ranks dataspace."); }
  
  hsize_t     dimsm[2];              /* memory space dimensions */
  hid_t       memspace;
  /*
   * Define the memory dataspace.
   */
  dimsm[0] = dims_out[0];
  dimsm[1] = dims_out[1];
  memspace = H5Screate_simple(2,dimsm,NULL);
  
  offset_out[0] = 0;
  offset_out[1] = 0;
  count_out[0] = dims_out[0];
  count_out[1] = dims_out[1];
  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
			       count_out, NULL);
  if (status < 0) { ION_ABORT("Couldn't read ranks hyperslap."); }

  mRankData.resize(count_out[0] * count_out[1]);
  fill(mRankData.begin(), mRankData.end(), 0);
  status = H5Dread(mRanks.mDataset, H5T_NATIVE_FLOAT, memspace, mRanks.mDataspace,
		   H5P_DEFAULT, &mRankData[0]);
  if (status < 0) { ION_ABORT("Couldn't read ranks dataset."); }
  mRanks.Close();
}

void RawWells::ReadInfo() {
  mInfo.Clear();
  vector<string> keys;
  vector<string> values;
  ReadStringVector(mHFile, INFO_KEYS, keys);
  ReadStringVector(mHFile, INFO_VALUES, values);
  if (keys.size() != values.size()) { ION_ABORT("Keys and Values don't match in size."); }
    
  for (size_t i = 0; i < keys.size(); i++) {
    if (!mInfo.SetValue(keys[i], values[i])) { 
      ION_ABORT("Error: Could not set key: " + keys[i] + " with value: " + values[i]); 
    }
  }
  if (!mInfo.GetValue(FLOW_ORDER, mFlowOrder)) { ION_ABORT("Error - Flow order is not set."); }
}

void RawWells::CleanupHdf5() {
  mRanks.Close();
  mWells.Close();
  mInfoKeys.Close();
  mInfoValues.Close();
  if (mHFile != RWH5DataSet::EMPTY) {
    H5Fclose(mHFile);
    mHFile = RWH5DataSet::EMPTY;
  }
}

void RawWells::Close() {
  mChunk.rowStart = 0;
  mChunk.rowHeight = 0;
  mChunk.colStart = 0;
  mChunk.colWidth = 0;
  mChunk.flowStart = 0;
  mChunk.flowDepth = 0;
  mFlowData.resize(0);
  mIndexes.resize(0);
  mWriteSubset.resize(0);
  CleanupHdf5();
}

void RawWells::GetRegion(int &rowStart, int &height, int &colStart, int &width) {
  rowStart = mChunk.rowStart;
  height = mChunk.rowHeight;
  colStart = mChunk.colStart;
  width = mChunk.colWidth;
}

void RawWells::SetFlowOrder(const std::string &flowOrder) { 
  mFlowOrder.resize(NumFlows());
  for (size_t i = 0; i < NumFlows(); i++) {
    mFlowOrder.at(i) = flowOrder.at( i % flowOrder.length()); 
  }
}

void RawWells::IndexToXY(size_t index, unsigned short &x, unsigned short &y) const { 
  y = (index / NumCols());
  x = (index % NumCols()); 
}

void RawWells::IndexToXY(size_t index, size_t &x, size_t &y) const {
  y = (index / NumCols());
  x = (index % NumCols());
}

void RawWells::Set(size_t idx, size_t flow, float val) {
  if (mIndexes[idx] == WELL_NOT_LOADED) {
    ION_ABORT("Well: " + ToStr(idx) + " is not loaded.");
  }
  if (mIndexes[idx] == WELL_NOT_SUBSET) {
    ION_ABORT("Well: " + ToStr(idx) + " is not is specified subset.");
  }
  if (flow < mChunk.flowStart || flow >= mChunk.flowStart + mChunk.flowDepth) {
    ION_ABORT("Flow: " + ToStr(flow) + " is not loaded in range: " + 
              ToStr(mChunk.flowStart) + " to " + ToStr(mChunk.flowStart+mChunk.flowDepth));
  }
  if (mIndexes[idx] >= 0) {
    uint64_t ii = (uint64_t) mIndexes[idx] * mChunk.flowDepth + flow - mChunk.flowStart;
    assert(ii < mFlowData.size());
    mFlowData[ii] = val;
  }
  else {
    ION_ABORT("Don't recognize index: " + ToStr(mIndexes[idx]));
  }
}

void RawWells::SetSubsetToLoad(const std::vector<int32_t> &xSubset, const std::vector<int32_t> &ySubset) {
  mWriteSubset.resize(xSubset.size());
  for (size_t i = 0; i < mWriteSubset.size();i++) {
    mWriteSubset[i] = ToIndex(xSubset[i], ySubset[i]);
  }
}

void RawWells::SetSubsetToWrite(const std::vector<int32_t> &subset) {
  mWriteSubset = subset;
}

void RawWells::SetSubsetToLoad(int32_t *xSubset, int32_t *ySubset, int count) {
  mWriteSubset.resize(count);
  for (size_t i = 0; i < mWriteSubset.size();i++) {
    mWriteSubset[i] = ToIndex(xSubset[i], ySubset[i]);
  }
}

void RawWells::InitIndexes() {
  mIndexes.resize(NumRows() * NumCols());
  fill(mIndexes.begin(), mIndexes.end(), WELL_NOT_LOADED);

  uint64_t count = 0;
  if (mWriteSubset.empty()) {
    for ( size_t row = mChunk.rowStart; row < mChunk.rowStart + mChunk.rowHeight; row++) {
      for (size_t col = mChunk.colStart; col < mChunk.colStart + mChunk.colWidth; col++) {
        mIndexes[ToIndex(col,row)] = count++;
      }
    }
  }
  else {
    for ( size_t row = mChunk.rowStart; row < mChunk.rowStart + mChunk.rowHeight; row++) {
      for (size_t col = mChunk.colStart; col < mChunk.colStart + mChunk.colWidth; col++) {
        mIndexes[ToIndex(col,row)] = WELL_NOT_SUBSET;
      }
    }
    for (size_t i = 0; i < mWriteSubset.size(); i++) {
      size_t x,y;
      IndexToXY(mWriteSubset[i], x, y);
      if (x >= mChunk.colStart && x < (mChunk.colStart + mChunk.colWidth) &&
          y >= mChunk.rowStart && y <  (mChunk.rowStart + mChunk.rowHeight)) {
        mIndexes[ToIndex(x,y)] = count++;
      }
    }
  }
  mFlowData.resize(count * mChunk.flowDepth);
  fill(mFlowData.begin(), mFlowData.end(), -1.0f);
}

void RawWells::SetChunk(size_t rowStart, size_t rowHeight, 
                        size_t colStart, size_t colWidth,
                        size_t flowStart, size_t flowDepth) {

  // cout << "Setting chunk: " << rowStart << "," << rowHeight << " " 
  //      << colStart << "," << colWidth << " "
  //      << flowStart << "," << flowDepth << endl;
  /* Sanity check */
  if (!mIsLegacy) {
    ION_ASSERT(rowHeight < 10000 &&
               colWidth < 10000 &&
               flowDepth < 10000, "Illegal chunk.");
    mChunk.rowStart = rowStart;
    mChunk.rowHeight = rowHeight;
    mChunk.colStart = colStart;
    mChunk.colWidth = colWidth;
    mChunk.flowStart = flowStart;
    mChunk.flowDepth = flowDepth;
    InitIndexes();
  }
}
  
