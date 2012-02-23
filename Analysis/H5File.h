/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef H5FILE_H
#define H5FILE_H

#include <vector>
#include <string>
#include <map>
#include <armadillo>

#include "hdf5.h"
#include "Utils.h"
#include "IonErr.h"
#include "DataCube.h"

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

  /** Utility function to write an entire matrix. */
  template<typename T>
  void WriteMatrix(arma::Mat<T> &mat);

  /** Utility function to read an entire matrix. */
  template<typename T>
  void ReadMatrix(arma::Mat<T> &mat);

  template<typename T>
  void WriteDataCube(DataCube<T> &cube);

  /** 
   * Assumes that the T* is the same size as specified by starts and
   * ends that starts and ends are contingous in memory. */
  template<typename T> 
    void ReadRangeData(const size_t *starts, const size_t *ends, 
                       size_t size,  T *data);

  const static int H5_NULL = -1; ///< Filler value for non-initialized fields

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
  void SetDataspace(int rank, const hsize_t dims[], const hsize_t chunking[], int type);
  bool CreateDataSet();
  bool OpenDataSet();
  void GetSelectionSpace(const size_t *starts, const size_t *ends, hid_t &memspace);
  void SetGroup(hid_t group) {mGroup = group; }
  void SetCompression(int level) { mCompression = level; }
  void SetParent(H5File *p) { mParent = p; }
  void SetName(const std::string &s) { mName = s; }

  /* Clean up our resources. Only called by H5File factory. */
  void CloseDataSet();
  size_t GetId() { return mId; }

  size_t mId;        ///< Unique id for this dataset used by the H5File parent
  std::string mName; ///< Name of the dataset inside the file
  hid_t mDataSet;    ///< hdf5 identifier, H5_NULL if not open
  H5File *mParent;   ///< Pointer to the file obect that created this dataset
  size_t mRank;      ///< Rank of the space of data, 1 for vector, 2 for matrix, 3 for cube, etc.
  int mCompression;  ///< Compression level, 0 for no compression
  int mType;         ///< Type of data like H5T_NATIVE_FLOAT
  hid_t mDataspace;  ///< H5 dataspace identifier
  hid_t mGroup;      ///< H5 datagroup identifier, will be closed if not the root file.
  hid_t mDatatype;   ///< H5 data type identifier
  size_t mSize;      ///< Size in bytes of hdf5 datatype in RAM
  std::vector<hsize_t> mDims; ///< Size of data space in each dimension. e.g. 10,2 for 10x2 matrix
  std::vector<hsize_t> mChunking; ///< How to organize on disk
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

  /** Utility function to open group (like mkdir -p /path/to/file) */
  static hid_t GetOrCreateGroup(hid_t location, const std::string &groupPath);

  /** Utility function to determine if file is hdf5. */
  static bool IsH5File(const std::string &path);
 
  /** Split on last ':' in /path/to/file:/path/in/hdf5 */
  static bool SplitH5FileGroup(const std::string &hdf5FilePath, 
                               std::string &diskPath,
                               std::string &h5Path);
  
  /** Utility function to write a matrix in one go to an hdf5 file. */
  template <typename T> 
  static bool WriteMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat, bool overwrite=false);

  /** Utility function to read a matrix in one go to an hdf5 file. */
  template <typename T> 
  static bool ReadMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat);
  
  /** Set the file name on the OS filesystem. */
  void SetFile(const std::string &path) { mName = path; }
  /** What is the name of our file on OS filesystem. */
  const std::string &GetFile() const { return mName; }
  /** hdf5 file id */
  hid_t GetFileId() const { return mHFile; }

  /** Open file, if overwrite is true delete any current file with name. */
  void Open(bool overwrite = false);
  /** Close file, cleanup datesets and associated resources. */
  void Close();

  /* Let compiler figure out types for us. */
  hid_t GetH5Type(float f) { return H5T_NATIVE_FLOAT; }
  hid_t GetH5Type(int i) { return H5T_NATIVE_INT; }
  hid_t GetH5Type(short s) { return H5T_NATIVE_SHORT; }
  hid_t GetH5Type(double d) { return H5T_NATIVE_DOUBLE; }

  /** 
   * Create a dataset using dimensions and type from matrix.  Memory
   * owned by HFile, don't delete when finished, call Close()
   */
  template<typename T>
  H5DataSet * CreateDataSet(const std::string &name, arma::Mat<T> &mat, int compression);

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
  
  /** 
   * Open an existing dataset inside of h5 file with giving path
   * name. Memory owned by HFile, don't delete when finished, call
   * Close()
   */
  H5DataSet * OpenDataSet(const std::string &name);

  /** Close a dataset with a specified id. */
  void CloseDataSet(size_t id);

 private:

  /** Break a path into a group and data set leaf. /path/to/dataset -> /path/to, dataset */
  static void FillInGroupFromName(const std::string &path, std::string &groupName, std::string &dsName);
  /** Recurse into path creating new groups as necessary. */
  static hid_t GetOrCreateGroup(hid_t location, const std::string &groupPath, std::vector<hid_t> &freeList);

  size_t mCurrentId;  ///< Current id for next dataset opened/created
  std::string mName;  ///< Path to file.
  hid_t mHFile;       ///< Id for hdf5 file operations.
  std::map<size_t, H5DataSet *> mDSMap; ///< Map of ids to the datasets owned by this file
};

template<typename T> 
void H5DataSet::WriteRangeData(const size_t *starts, const size_t *ends, const T *data) {
  ION_ASSERT(mDataSet != H5_NULL, "DataSet: " + mName + " doesn't appear open.");    
  hid_t memspace;
  GetSelectionSpace(starts, ends, memspace);
  int status = H5Dwrite(mDataSet, mDatatype, memspace, mDataspace, H5P_DEFAULT, data);
  ION_ASSERT(status >= 0, "Couldn't write data.");
}

template<typename T>
void H5DataSet::WriteMatrix(arma::Mat<T> &mat) {
  size_t starts[2];
  size_t ends[2];
  starts[0] = starts[1] = 0;
  ends[0] = mat.n_rows;
  ends[1] = mat.n_cols;
  //size_t size = mat.n_rows * mat.n_cols;
  arma::Mat<T> m = trans(mat); // armadillo is stored column major, we want row major...
  WriteRangeData(starts, ends, m.memptr());
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
void H5DataSet::ReadMatrix(arma::Mat<T> &mat) {
  ION_ASSERT(mRank == 2, "Can't read a matrix when rank isn't 2");
  mat.set_size(mDims[1],mDims[0]);
  size_t starts[2];
  size_t ends[2];
  starts[0] = starts[1] = 0;
  ends[0] = mat.n_cols;
  ends[1] = mat.n_rows;
  size_t size = mat.n_rows * mat.n_cols;
  ReadRangeData(starts, ends, size, mat.memptr());
  mat = trans(mat); // armadillo is stored column major, we want row major...
}

/** 
 * Assumes that the T* is the same size as specified by starts and
 * ends that starts and ends are contingous in memory. */
template<typename T> 
void H5DataSet::ReadRangeData(const size_t *starts, const size_t *ends, 
                              size_t size,  T *data) {
  ION_ASSERT(mDataSet != H5_NULL, "DataSet: " + mName + " doesn't appear open.");
  hid_t memspace;
  GetSelectionSpace(starts, ends, memspace);
  int status = H5Dread(mDataSet, mDatatype, memspace, mDataspace, H5P_DEFAULT, data);
  ION_ASSERT(status >= 0, "Couldn't read data.");
}

template<typename T>
H5DataSet * H5File::CreateDataSet(const std::string &name, arma::Mat<T> &mat, int compression) {
  hsize_t rank = 2;
  hsize_t dims[rank];
  hsize_t chunking[rank];
  chunking[0] = dims[0] = mat.n_rows;
  chunking[1] = dims[1] = mat.n_cols;
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
bool H5File::ReadMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open();
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  ION_ASSERT(ds != NULL, "Couldn't open dataset: " + h5path);
  ds->ReadMatrix(mat);
  ds->Close();
  return true;
}

template <typename T> 
bool  H5File::WriteMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat, bool overwrite) {
  std::string file, h5path;
  bool okPath = SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open(overwrite);
  H5DataSet *ds = h5File.CreateDataSet(h5path, mat, 3);
  ION_ASSERT(ds != NULL, "Couldn't make dataset: " + h5path);
  ds->WriteMatrix(mat);
  ds->Close();
  return true;
}

#endif // H5FILE_H
