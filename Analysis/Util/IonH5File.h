/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef H5FILE_H
#define H5FILE_H

#include <vector>
#include <string>
#include <map>
#include <stdio.h>

#include "hdf5.h"
#include "Utils.h"
#include "IonErr.h"
#include "DataCube.h"
#include "BkgModel/BkgMagicDefines.h"

#define MAXRANK 3

class H5File;

/**
 * Wrapper to a serialized data set on disk in hdf5 format.
 * 
 * Most of the functionality is set by the H5File class which acts as
 * a factory for data sets. Don't delete pointers that come from this
 * class. Call Close() instead when done or delete the parent H5File.
 */ 
class H5DataSet {

 public:

  /* Accessors. */
  hid_t GetGroup() { return mGroup; }
  int GetCompression() const { return mCompression; }
  H5File * GetParent() const { return mParent; }
  const std::string & GetName() const { return mName; }

  /** Tell parent to cleanup the resources associated with dataset */
  void Close();

  /** 
   * Assumes that the T* is the same size as specified by starts and
   * ends that starts and ends are contingous in memory. */
  template<typename T> 
    void WriteRangeData(const size_t *starts, const size_t *ends, const T *data);

  /** Utility function to write an entire vector. */
    template<typename T>
      void WriteVector(std::vector<T> &vec);

  /** Utility function to read an entire vector. */
  template<typename T>
  void ReadVector(std::vector<T> &vec);

  template<typename T>
  void WriteDataCube(DataCube<T> &cube);

  /** Utility function to read an entire data cube. */
    template<typename T>
    void ReadDataCube(DataCube<T> &cube);

  /** 
   * Assumes that the T* is the same size as specified by starts and
   * ends that starts and ends are contingous in memory. */
  template<typename T> 
    void ReadRangeData(const size_t *starts, const size_t *ends, 
                       size_t size,  T *data);

  const static int H5_NULL = -1; ///< Filler value for non-initialized fields

  size_t GetRank() { return mRank; }
  std::vector<hsize_t> &GetDims() { return mDims; }
 private:
  /* Most of state is set by factory. */
  friend class H5File;

  /* Don't use. */
  H5DataSet() { Init(0); }; 

  /* only used by H5File factory */
  H5DataSet(size_t id) { Init(id); } 

  /* Create object and initialize variables to default. */
  void Init(size_t id);

  /* Setup for the state of the dataset. */
  void SetDataspace(int rank, const hsize_t dims[], const hsize_t chunking[], hid_t type);
  bool CreateDataSet();
  bool OpenDataSet();
  void GetSelectionSpace(const size_t *starts, const size_t *ends, hid_t &memspace);
  void SetGroup(hid_t group) {mGroup = group; }
  void SetCompression(int level) { mCompression = level; }
  void SetParent(H5File *p) { mParent = p; }
  void SetName(const std::string &s) { mName = s; }
  void ExtendIfNeeded(const size_t *ends);
  void Extend(const size_t *ends, bool onlyBigger);

  /* Clean up our resources. Only called by H5File factory. */
  void CloseDataSet();
   size_t GetId() { return mId; }

  size_t mId;        ///< Unique id for this dataset used by the H5File parent
  std::string mName; ///< Name of the dataset inside the file
  hid_t mDataSet;    ///< hdf5 identifier, H5_NULL if not open
  H5File *mParent;   ///< Pointer to the file obect that created this dataset
  size_t mRank;      ///< Rank of the space of data, 1 for vector, 2 for matrix, 3 for cube, etc.
  int mCompression;  ///< Compression level, 0 for no compression
  hid_t mType;         ///< Type of data like H5T_NATIVE_FLOAT
  hid_t mDataspace;  ///< H5 dataspace identifier
  hid_t mGroup;      ///< H5 datagroup identifier, will be closed if not the root file.
  hid_t mDatatype;   ///< H5 data type identifier
  size_t mSize;      ///< Size in bytes of hdf5 datatype in RAM
  std::vector<hsize_t> mDims; ///< Size of data space in each dimension. e.g. 10,2 for 10x2 matrix
  std::vector<hsize_t> mChunking; ///< How to organize on disk

public:
  hid_t getDataSetId() { return mDataSet; }

};

/**
 * Handles opening and closing of the physical file on disk.
 * also owns and tracks all the memory and resources for the
 * individual datasets. 
 */
class H5File {

 public:
  /** Constructor. */
  H5File();

  /** Constructor. */
  H5File(const std::string &file) { Init(); SetFile(file); }

  /** Initialize to standard defaults. */
  void Init();

  /** Destructor. */
  ~H5File() { Close(); }

  /** Open the hdf5 file as a readonly file. */
  void SetReadOnly(bool readOnly) { mReadOnly = readOnly; }

  /** Utility function to open group (like mkdir -p /path/to/file) */
  static hid_t GetOrCreateGroup(hid_t location, const std::string &groupPath);

  /** Utility function to determine if file is hdf5. */
  static bool IsH5File(const std::string &path);
 
  /** Split on last ':' in /path/to/file:/path/in/hdf5 */
  static bool SplitH5FileGroup(const std::string &hdf5FilePath, 
                               std::string &diskPath,
                               std::string &h5Path);
  
  /** Utility function to read a data cube in one go to an hdf5 file. */
  template <typename T>
    static bool ReadDataCube(const std::string &hdf5FilePath, DataCube<T> &mat);

  /** Utility function to write a vector in one go to an hdf5 file. */
  template <typename T> 
    static bool WriteVector(const std::string &hdf5FilePath, std::vector<T> &vec, bool overwrite=false);

  /** Utility function to read a vector in one go to an hdf5 file. */
  template <typename T> 
    static bool ReadVector(const std::string &hdf5FilePath, std::vector<T> &vec);
  
  /** Check if a dataset exists in an hdf5 file. */
  template <typename T> 
    static bool DatasetExist(const std::string &hdf5FilePath);
  
  /** Set the file name on the OS filesystem. */
  void SetFile(const std::string &path) { mName = path; }
  /** What is the name of our file on OS filesystem. */
  const std::string &GetFile() const { return mName; }
  /** hdf5 file id */
  hid_t GetFileId() const { return mHFile; }
  /** Open file, blindly overwriting. */
  bool OpenNew();
  /** Open file, if overwrite is true delete any current file with name. */
  bool Open(bool overwrite = false);

  /** Open file, if overwrite is true delete any current file with name. */
  bool OpenForReading( void );

  /** Close file, cleanup datesets and associated resources. */
  void Close();

  /* Let compiler figure out types for us. */
  hid_t GetH5Type(float f) { return H5T_NATIVE_FLOAT; }
  hid_t GetH5Type(int i) { return H5T_NATIVE_INT; }
  hid_t GetH5Type(short s) { return H5T_NATIVE_SHORT; }
  hid_t GetH5Type(double d) { return H5T_NATIVE_DOUBLE; }
  hid_t GetH5Type(char c) { return H5T_NATIVE_CHAR; }

  /** 
   * Create a dataset using dimensions and type from vector.  Memory
   * owned by HFile, don't delete when finished, call Close()
   */
  template<typename T>
    H5DataSet * CreateDataSet(const std::string &name, std::vector<T> &vec, int compression);

  /** 
   * Create a dataset using dimensions and type from a cube.  Memory
   * owned by HFile, don't delete when finished, call Close()
   */
  template<typename T>
  H5DataSet * CreateDataSet(const std::string &name, DataCube<T> &mat, int compression);

  /** 
   * Make a new dataset inside the h5 file with properties
   * specified. Memory owned by HFile, don't delete when finished,
   * call Close()
   */
  H5DataSet * CreateDataSet(const std::string &name,
                            hsize_t rank,
                            const hsize_t dims[],
                            const hsize_t chunking[],
                            int compression,
                            hid_t type);
  hid_t CreateAttribute(const hid_t obj_id, const char *name, const char *msg);
  herr_t WriteAttribute(const hid_t attr_id, const char *msg);
  void makeParamNames(const char * param_prefix, int nParams, std::string &outStr);

  /** 
   * Open an existing dataset inside of h5 file with giving path
   * name. Memory owned by HFile, don't delete when finished, call
   * Close()
   */
  H5DataSet * OpenDataSet(const std::string &name);

  /** Close a dataset with a specified id. */
  void CloseDataSet(size_t id);

  /** 
   * Create a dataset in a file and set chunking using chunksize
   * Overwrites the existing file if overwrite=true
   * Closes the file after creating the dataset
   */
  template<typename T>
    static void CreateDataSetOnly(const std::string &hdf5FilePath, std::vector<int>& chunksize, int compression, bool overwrite, T t=T(0));

  /** 
   * Write data to dataset in a file at dimensions between starts and ends
   */
  template<typename T> 
    static void WriteRangeData(const std::string &hdf5FilePath, const size_t *starts, const size_t *ends, const T *data);

  /** 
   * Append data to dataset in a file along the first dimension
   */
  template<typename T> 
    static void Append(const std::string &hdf5FilePath, std::vector<T>& data);


  /** Break a path into a group and data set leaf. /path/to/dataset -> /path/to, dataset */
  static void FillInGroupFromName(const std::string &path, std::string &groupName, std::string &dsName);
  /** Recurse into path creating new groups as necessary. */
  static hid_t GetOrCreateGroup(hid_t location, const std::string &groupPath, std::vector<hid_t> &freeList);
  void WriteString ( const std::string &name, const char *value);
  void WriteStringVector(const std::string &name, const char **values, int numValues);
  void ReadStringVector(const std::string &name, std::vector<std::string> &strings);
  size_t mCurrentId;  ///< Current id for next dataset opened/created
  std::string mName;  ///< Path to file.
  bool mReadOnly;     ///< Are we read only
  hid_t mHFile;       ///< Id for hdf5 file operations.
  std::map<size_t, H5DataSet *> mDSMap; ///< Map of ids to the datasets owned by this file
};

template<typename T> 
void H5DataSet::WriteRangeData(const size_t *starts, const size_t *ends, const T *data) {
  ION_ASSERT(mDataSet != H5_NULL, "DataSet: " + mName + " doesn't appear open.");    
  hid_t memspace;
  ExtendIfNeeded(ends);
  GetSelectionSpace(starts, ends, memspace);
  int status = H5Dwrite(mDataSet, mDatatype, memspace, mDataspace, H5P_DEFAULT, data);
  ION_ASSERT(status >= 0, "Couldn't write data.");
}

template<typename T>
void H5DataSet::WriteDataCube(DataCube<T> &cube) {
  size_t starts[3];
  size_t ends[3];
  size_t xStart, xEnd, yStart, yEnd, zStart, zEnd;
  cube.GetRange(xStart, xEnd, yStart, yEnd, zStart, zEnd);
  starts[0] = xStart;
  ends[0] = xEnd;
  starts[1] = yStart;
  ends[1] = yEnd;
  starts[2] = zStart;
  ends[2] = zEnd;
  //size_t size = (xEnd - xStart) * (yEnd - yStart) * (zEnd - zStart);
  WriteRangeData(starts, ends, cube.GetMemPtr());
}


template<typename T>
void H5DataSet::ReadDataCube(DataCube<T> &cube) {
  ION_ASSERT(mRank == 3, "Can't read a matrix when rank isn't 2");
  cube.Init(mDims[0],mDims[1],mDims[2]);
  cube.AllocateBuffer();
  size_t starts[3];
  size_t ends[3];
  starts[0] = starts[1] = starts[2] = 0;
  ends[0] = mDims[0];
  ends[1] = mDims[1];
  ends[2] = mDims[2];
  size_t size = mDims[0]*mDims[1]*mDims[2];
  ReadRangeData(starts, ends, size, cube.GetMemPtr());
}


/** 
 * Assumes that the T* is the same size as specified by starts and
 * ends that starts and ends are contiguous in memory. */
template<typename T> 
void H5DataSet::ReadRangeData(const size_t *starts, const size_t *ends, 
                              size_t size,  T *data) {
  ION_ASSERT(mDataSet != H5_NULL, "DataSet: " + mName + " doesn't appear open.");
  hid_t memspace;
  GetSelectionSpace(starts, ends, memspace);
  int status = H5Dread(mDataSet, mDatatype, memspace, mDataspace, H5P_DEFAULT, data);
  if( status < 0 )
      H5Eprint( H5E_DEFAULT, stderr );
  ION_ASSERT(status >= 0, "Couldn't read data.");
}


// Assumes we will be writing in chunk sizes that matches vector
template<typename T>
H5DataSet * H5File::CreateDataSet(const std::string &name, std::vector<T> &vec, int compression) {
  hsize_t rank = 1;
  hsize_t dims[rank];
  hsize_t chunking[rank];
  chunking[0] = dims[0] = vec.size();
  T t = 0;
  hid_t type = GetH5Type(t);
  H5DataSet *ds = CreateDataSet(name, rank, dims, chunking, compression, type);
  return ds;
}

template<typename T>
H5DataSet * H5File::CreateDataSet(const std::string &name, DataCube<T> &cube, int compression) {
  hsize_t rank = 3;
  hsize_t dims[rank];
  hsize_t chunking[rank];
  chunking[0] = dims[0] = cube.GetNumX();
  chunking[1] = dims[1] = cube.GetNumY();
  chunking[2] = dims[2] = cube.GetNumZ();
  // @todo - make this settable 
  chunking[0] = std::min((int)chunking[0], 60);
  chunking[1] = std::min((int)chunking[1], 60);
  chunking[2] = std::min((int)chunking[2], 60);
  T t = 0;
  hid_t type = GetH5Type(t);
  H5DataSet *ds = CreateDataSet(name, rank, dims, chunking, compression, type);
  return ds;
}

template <typename T>
bool H5File::ReadDataCube(const std::string &hdf5FilePath, DataCube<T> &cube) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.OpenForReading();
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  ION_ASSERT(ds != NULL, "Couldn't open dataset: " + h5path);
  ds->ReadDataCube(cube);
  ds->Close();
  h5File.Close();
  return true;
}


template <typename T> 
 bool H5File::ReadVector(const std::string &hdf5FilePath, std::vector<T> &vec) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.OpenForReading();
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  ION_ASSERT(ds != NULL, "Couldn't open dataset: " + h5path);
  ds->ReadVector(vec);
  ds->Close();
  h5File.Close();
  return true;
}

template<typename T>
void H5DataSet::ReadVector(std::vector<T> &vec) {
  ION_ASSERT(mRank == 1, "Can't read a vector when rank isn't 1");
  vec.resize(mDims[0]);
  size_t starts[1];
  size_t ends[1];
  starts[0] = 0;
  ends[0] = vec.size();
  size_t size = ends[0];
  ReadRangeData(starts, ends, size, &vec[0]);
}

template <typename T> 
 bool H5File::DatasetExist(const std::string &hdf5FilePath) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open();  
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  bool exists;
  if (ds!=NULL){
    ds->Close();
    exists = true;
  }  
  else
    exists = false;
  
  h5File.Close();
  return exists;
}

template <typename T> 
bool  H5File::WriteVector(const std::string &hdf5FilePath, std::vector<T> &vec, bool overwrite) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open(overwrite);
  H5DataSet *ds = h5File.CreateDataSet(h5path, vec, 3);
  ION_ASSERT(ds != NULL, "Couldn't make dataset: " + h5path);
  ds->WriteVector(vec);
  ds->Close();
  h5File.Close();  return true;
}

template<typename T>
void H5DataSet::WriteVector(std::vector<T> &vec) {
  size_t starts[1];
  size_t ends[1];
  starts[0] = 0;
  ends[0] = vec.size();
  WriteRangeData(starts, ends, &vec[0]);
}

/** just create a dataset, don't write anything, close after done
 * initial dimensions are set by chunksize, first dimension has zero length
 *  expectation is that writes to this dataset will be done in hyperslabs
 * defined by a range of values in the 1st dimension only
 */
template<typename T>
void H5File::CreateDataSetOnly(const std::string &hdf5FilePath, std::vector<int>& chunksize, int compression, bool overwrite, T t) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open(overwrite);
  hsize_t rank = chunksize.size();
  hsize_t dims[rank];
  hsize_t chunking[rank];
  size_t size[rank];
  for (unsigned int i=0; i<rank; i++){
    size[i] = chunking[i] = dims[i] = chunksize[i];
  }
  // T t = 0;
  hid_t type = h5File.GetH5Type(t);
  H5DataSet *ds = h5File.CreateDataSet(h5path, rank, dims, chunking, compression, type);
  
  // make the 1st dimension zero
  size[0] = 0;
  ds->Extend((const size_t *)size, false);
  ds->Close();
  h5File.Close();
}

/** write data to a hdf5 dataset starting at indices "starts" and ending at
 * indices "ends". data is contiguous in memory and the write starts at data[0]
 * the hdf5 dataset must be chunked as it will be extended to include dimensions
 * "ends". Writing in multiples of the chunk size might improve performance.
 */
template<typename T> 
void H5File::WriteRangeData(const std::string &hdf5FilePath, const size_t *starts, const size_t *ends, const T *data) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open();
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  ION_ASSERT(ds != NULL, "Couldn't open dataset: " + h5path);

  ds->WriteRangeData(&starts, &ends, &data[0]);
  ds->Close();
  h5File.Close();
}


/** append data to a hdf5 dataset using the 1st dimension.
 * data is contiguous in memory and the
 * write starts at data[0].  The hdf5 dataset must be chunked as it
 * will be extended in the 1st dimension.
 */
template<typename T> 
void H5File::Append(const std::string &hdf5FilePath, std::vector<T>& data)
{
  size_t n=data.size();
  if (n==0) // nothing to write
    return;

  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open();
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  ION_ASSERT(ds != NULL, "Couldn't open dataset: " + h5path);

  unsigned int rank = ds->mRank;
  size_t starts[rank]; // starting indices for write
  size_t ends[rank];   // ending indices for write

  hsize_t iStart = 1;  // starting linearized index to write
  hsize_t otherDimLength = 1;
  for (unsigned int i=0; i<ds->mDims.size(); i++){
    iStart = iStart * ds->mDims[i];
    if (i>0){
      otherDimLength = otherDimLength * ds->mDims[i];
      starts[i] = 0;
    }
  }
  starts[0] = ds->mDims[0];

  hsize_t iEnd = iStart + n -1; // ending linearized index for write
  assert(otherDimLength>0);

  // @TODO GetSelectionSpace assumes the in-memory data length fits exactly
  // into a (rank-1) dim'l space
  assert( (n/otherDimLength)*otherDimLength == n );

  // given an array [A][B][C], index i = (a*B*C) + (b*C) +c
  // a = i/(B*C)
  // b = (i - a*B*C)/C
  // c = (i - a*B*C - b*C)
  if (rank == 1){
    ends[0] = iEnd;
  }
  else if (rank==2){
    hsize_t B = ds->mDims[1];
    ends[0] = iEnd/B;
    ends[1] = (iEnd - ends[0]*B);
  }
  else if (rank == 3){
    hsize_t B = ds->mDims[1];
    hsize_t C = ds->mDims[2];
    ends[0] = iEnd/(B*C);
    ends[1] = (iEnd - ends[0]*B*C)/C;
    ends[2] = (iEnd - ends[0]*B*C -ends[1]*C);
  }
  else {
    assert (false);
  }

  for (unsigned int i=0; i<rank; i++)
    ends[i]++; // see GetSelectionSpace
  ds->WriteRangeData(starts, ends, &data[0]);

  std::string mName = ds->mName;
  if (rank ==1)
    fprintf(stdout, "H5: Writing %d to %s from [%d] to [%d]\n", (int)n, mName.c_str(), (int)starts[0], (int)ends[0]);
  else if (rank==2)
    fprintf(stdout, "H5: Writing %d to %s from [%d, %d] to [%d, %d]\n", (int)n, mName.c_str(), (int)starts[0], (int)starts[1], (int)ends[0], (int)ends[1]);
  else
    fprintf(stdout, "H5: Writing %d to %s from [%d, %d, %d] to [%d, %d, %d]\n", (int)n, mName.c_str(), (int)starts[0], (int)starts[1], (int)starts[2], (int)ends[0], (int)ends[1], (int)ends[2]);
  
  ds->Close();
  h5File.Close();
}
#endif // H5FILE_H

