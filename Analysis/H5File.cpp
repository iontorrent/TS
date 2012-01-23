/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include "H5File.h"

using namespace std;

void H5DataSet::Init(size_t id) {
  mId = id;
  mDataSet = H5_NULL;
  mGroup = H5_NULL;
  mDataspace = H5_NULL;
  mDatatype = H5_NULL; // @todo - do we need both mType and mDataType?
  mParent = NULL;
  mRank = 0;
  mCompression = 0;
  mType = 0;
  mSize = 0;
}

void H5DataSet::SetDataspace(int rank, hsize_t *dims, hsize_t *chunking, int type) {
  mRank = rank;
  mDims.resize(rank);
  mChunking.resize(rank);
  mType = type;
  for (size_t i = 0; i < mDims.size(); i++) {
    mDims[i] = dims[i];
    mChunking[i] = chunking[i];
  }
}

bool H5DataSet::CreateDataSet() {
  ION_ASSERT(mDataSet == H5_NULL, "DataSet already appears open.");
  mDataspace = H5Screate_simple(mRank, &mDims[0], NULL);
  mDatatype = H5Tcopy(mType);
  mSize = H5Tget_size(mDatatype);
  hid_t plist;
  plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist, mRank, &mChunking[0]);
  if (mCompression > 0) {
    H5Pset_deflate(plist, mCompression);
  }
  hid_t dapl;
  dapl = H5Pcreate (H5P_DATASET_ACCESS);
  hid_t group = mGroup;
  if (mGroup == H5_NULL) {
    group = mParent->GetFileId();
  }
  mDataSet = H5Dcreate2(group, mName.c_str(), mDatatype, mDataspace,
                        H5P_DEFAULT, plist, dapl);
  if (mDataSet < 0) {
    Init(0);
    return false;
  }
  return true;
}

bool H5DataSet::OpenDataSet() {
  ION_ASSERT(mDataSet == H5_NULL, "DataSet already appears open.");
  hid_t group = mGroup;
  if (mGroup == H5_NULL) {
    group = mParent->GetFileId();
  }
  mDataSet = H5Dopen2(group, mName.c_str(), H5P_DEFAULT);
  if( mDataSet < 0) {
    Init(0);
    return false;
  }
  mDatatype  = H5Dget_type(mDataSet);     /* datatype handle */
  mType = mDatatype;
  mSize  = H5Tget_size(mDatatype);
  mDataspace = H5Dget_space(mDataSet);
  int rank = H5Sget_simple_extent_ndims(mDataspace);
  mDims.resize(rank);
  mRank = rank;
  int status = H5Sget_simple_extent_dims(mDataspace, &mDims[0], NULL);
  ION_ASSERT(status >= 0, "Couldn't get data for dataset: '" + mName + "'")
    return true;
}

void H5DataSet::Close() {
  // Tell parent that owns this child to close and cleanup.
  mParent->CloseDataSet(GetId());
}

void H5DataSet::CloseDataSet() {
  if (mDataSet != H5_NULL) {
    H5Dclose(mDataSet);
    mDataSet = H5_NULL;
    H5Sclose(mDataspace);
    mDataspace = H5_NULL;
    H5Tclose(mDatatype);
    mDatatype = H5_NULL;
    if (mGroup != mParent->GetFileId()) {
      H5Gclose(mGroup);
      mGroup = H5_NULL;
    }
  }  
  mRank = 0;
  mDims.resize(0);
  mChunking.resize(0);
  mCompression = 0;
  mType = 0;
  mName = "";
}

void H5DataSet::GetSelectionSpace(const size_t *starts, const size_t *ends,
                                  hid_t &memspace) {
  hsize_t count[mRank];       /* size of the hyperslab in the file */
  hsize_t offset[mRank];      /* hyperslab offset in the file */
  hsize_t count_out[mRank];   /* size of the hyperslab in memory */
  hsize_t offset_out[mRank];  /* hyperslab offset in memory */
  for (size_t i = 0; i < mRank; i++) {
    //@todo - check that we are within our ranges here
    offset[i] = starts[i];
    count_out[i] = count[i] = ends[i] - starts[i];
    offset_out[i] = 0;
  }
  herr_t status = 0;
  status = H5Sselect_hyperslab(mDataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
  ION_ASSERT(status >= 0, "Couldn't select hyperslab from dataspace for read.");
  memspace = H5Screate_simple(mRank,count_out,NULL);
  status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL,
                               count_out, NULL);
  ION_ASSERT(status >= 0, "Couldn't select hyperslab from memspace for read.");
}

H5File::H5File() {
  Init();
}

void H5File::Init() {
  mCurrentId = 0;
  mHFile = H5DataSet::H5_NULL;
}

hid_t H5File::GetOrCreateGroup(hid_t location, const std::string &groupPath, std::vector<hid_t> &freeList) {
  std::string first =  groupPath;
  std::string second;
  size_t pos = groupPath.find('/');
  if ( pos == string::npos) {     // base case, no '/'s left
    hid_t g = H5Gopen(location, groupPath.c_str(), H5P_DEFAULT);
    if (g <= 0) {
      g = H5Gcreate(location, groupPath.c_str(), 0,  H5P_DEFAULT, H5P_DEFAULT);
    }
    return g;
  }
  else { // recurse down a level
    first = groupPath.substr(0,pos);
    second = groupPath.substr(pos+1, groupPath.length());
    hid_t g = location;
    if (pos > 0) {
      g = H5Gopen(location, first.c_str(), H5P_DEFAULT);
      freeList.push_back(g);
    }
    if (g <= 0) {
      g = H5Gcreate(location, first.c_str(), 0,  H5P_DEFAULT, H5P_DEFAULT);
      freeList.push_back(g);
    }
    if (second.length() > 0) {
      return GetOrCreateGroup(g, second, freeList);
    }
    else {
      return g;
    }
  }
  return H5DataSet::H5_NULL;
}

hid_t H5File::GetOrCreateGroup(hid_t location, const std::string &groupPath) {
  
  /* Save old error handler */
  herr_t (*old_func)(hid_t, void *);
  void *old_client_data;
  
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  
  /* Turn off error handling */
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
  
  std::vector<hid_t> groups;
  hid_t group = GetOrCreateGroup(location, groupPath, groups);
  
  for (size_t i = 0; i < groups.size(); i++) {
    H5Gclose(groups[i]);
  }
  
  /* Restore previous error handler */
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
  return group;
}

void H5File::FillInGroupFromName(const std::string &path, std::string &groupName, std::string &dsName) {
  size_t pos = path.rfind('/');
  if (pos == string::npos) {
    groupName = "/";
    dsName = path;
  }
  else {
    groupName = path.substr(0,pos);
    dsName = path.substr(pos+1, path.length() - pos - 1);
  }
}

void H5File::Open(bool overwrite) {
  bool isH5 =  IsH5File(mName);
  if (isH5 && !overwrite) {
    mHFile = H5Fopen(mName.c_str(), H5F_ACC_RDWR,  H5P_DEFAULT);
  }
  else if(overwrite) {
    mHFile = H5Fcreate(mName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }
  if (mHFile < 0) {
    ION_ABORT("Couldn't open file: " + mName);
  }
}

bool H5File::IsH5File(const std::string &path) {
  H5E_auto2_t old_func; 
  void *old_client_data;
  hid_t suspectFile;
  bool isH5 = false;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2( H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto2( H5E_DEFAULT, NULL, NULL);
  suspectFile = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (suspectFile >= 0) {
    isH5 = true;
    H5Fclose(suspectFile);
  }
  H5Eset_auto2( H5E_DEFAULT, old_func, old_client_data);
  return isH5;
}

H5DataSet * H5File::CreateDataSet(const std::string &name,
                                  hsize_t rank,
                                  hsize_t *dims,
                                  hsize_t *chunking,
                                  int compression,
                                  hid_t type) {
  size_t id = mCurrentId;
  mCurrentId++;
  H5DataSet *ds = new H5DataSet(id);
  string groupName, dsName;
  FillInGroupFromName(name, groupName, dsName);
  hid_t group = GetOrCreateGroup(mHFile, groupName);
  ds->SetGroup(group);
  ds->SetParent(this);
  ds->SetName(dsName);
  ds->SetCompression(compression);
  ds->SetDataspace(rank, dims, chunking, type);
  if (ds->CreateDataSet()) {
    mDSMap[id] = ds;
    return mDSMap[id];
  }
  delete ds;
  return NULL;
}

H5DataSet * H5File::OpenDataSet(const std::string &name) {
  size_t id = mCurrentId;
  mCurrentId++;
  H5DataSet *ds = new H5DataSet(id);
  string groupName, dsName;
  FillInGroupFromName(name, groupName, dsName);
  hid_t group = GetOrCreateGroup(mHFile, groupName);
  ds->SetGroup(group);
  ds->SetParent(this);
  ds->SetName(dsName);
  /* Save old error handler */
  herr_t (*old_func)(hid_t, void *);
  void *old_client_data;
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  /* Turn off error handling */
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
  if (ds->OpenDataSet()) {
    mDSMap[id] = ds;
    H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
    return mDSMap[id];
  }
  /* Restore previous error handler */
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
  delete ds;
  return NULL;
}

void H5File::Close() {
  while (!mDSMap.empty()) {
    std::map<size_t, H5DataSet *>::iterator i = mDSMap.begin();
    i->second->Close();
  }
  if (mHFile != H5DataSet::H5_NULL) {
    H5Fclose(mHFile);
    mHFile = H5DataSet::H5_NULL;
  }
}

void H5File::CloseDataSet(size_t id) {
  std::map<size_t, H5DataSet *>::iterator i = mDSMap.find(id);
  ION_ASSERT(i != mDSMap.end(), "Couldn't find id: " + ToStr(id));
  i->second->CloseDataSet();
  delete i->second;
  mDSMap.erase(i);
}

bool H5File::SplitH5FileGroup(const std::string &hdf5FilePath, 
                              std::string &diskPath,
                              std::string &h5Path) {
  size_t pos = hdf5FilePath.rfind(':');
  if (pos == std::string::npos) {
    return false;
  }
  diskPath = hdf5FilePath.substr(0,pos);
  h5Path = hdf5FilePath.substr(pos+1,hdf5FilePath.length() - (pos+1));
  TrimString(diskPath);
  TrimString(h5Path);
  if (!(diskPath.empty() || h5Path.empty())) {
    return true;
  }
  return false;
}
