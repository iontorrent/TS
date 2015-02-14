/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef H5ARMA_H
#define H5ARMA_H

#include <armadillo>
#include "H5File.h"

class H5Arma {
 public:
  /** Utility function to write an entire matrix. */
  template<typename T>
    static void WriteMatrix(H5DataSet &ds, arma::Mat<T> &mat);
  
  template<typename T>
    static void WriteMatrix(H5DataSet &ds, arma::Mat<T> &mat, int col);
  
  /** Utility function to read an entire matrix. */
  template<typename T>
    static void ReadMatrix(H5DataSet &ds, arma::Mat<T> &mat);
  
  /** Utility function to write a matrix in one go to an hdf5 file. */
  template <typename T> 
    static void WriteMatrix(H5File &h5, const std::string &hdf5FilePath, arma::Mat<T> &mat);
  
  /** 
   * Create a dataset using dimensions and type from matrix.  Memory
   * owned by HFile, don't delete when finished, call Close()
   */
  template<typename T>
    static H5DataSet * CreateDataSet(H5File &h5, const std::string &name, arma::Mat<T> &mat, int compression);

    /** Utility function to write a matrix in one go to an hdf5 file. */
  template <typename T> 
    static bool WriteMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat, bool overwrite);

  /** Utility function to read a matrix in one go to an hdf5 file. */
  template <typename T> 
    static bool ReadMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat);
  
};

template<typename T>
H5DataSet * H5Arma::CreateDataSet(H5File &h5, const std::string &name, arma::Mat<T> &mat, int compression) {
  hsize_t rank = 2;
  hsize_t dims[rank];
  hsize_t chunking[rank];
  chunking[0] = std::min(1000u, mat.n_rows);
  dims[0] = mat.n_rows;
  chunking[1] = std::min(1000u, mat.n_cols);
  dims[1] = mat.n_cols;
  T t = 0;
  hid_t type = h5.GetH5Type(t);
  H5DataSet *ds = h5.CreateDataSet(name, rank, dims, chunking, compression, type);
  return ds;
}

template<typename T>
void H5Arma::WriteMatrix(H5DataSet &ds, arma::Mat<T> &mat) {
  size_t starts[2];
  size_t ends[2];
  starts[0] = starts[1] = 0;
  ends[0] = mat.n_rows;
  ends[1] = mat.n_cols;
  //size_t size = mat.n_rows * mat.n_cols;
  arma::Mat<T> m = trans(mat); // armadillo is stored column major, we want row major...
  ds.WriteRangeData(starts, ends, m.memptr());
}

template<typename T>
void H5Arma::WriteMatrix(H5DataSet &ds, arma::Mat<T> &mat, int col) {
  size_t starts[2];
  size_t ends[2];
  starts[0] = 0; 
  ends[0] = mat.n_rows;
  starts[1] = col;
  ends[1] = col+1;
  arma::Mat<T> m = trans(mat); // armadillo is stored column major, we want row major...
  ds.WriteRangeData(starts, ends, m.memptr());
}

template<typename T>
void H5Arma::ReadMatrix(H5DataSet &ds, arma::Mat<T> &mat) {
  ION_ASSERT(ds.GetRank() == 2, "Can't read a matrix when rank isn't 2");
  std::vector<hsize_t> &dims = ds.GetDims();
  mat.set_size(dims[1],dims[0]);
  size_t starts[2];
  size_t ends[2];
  starts[0] = starts[1] = 0;
  ends[0] = mat.n_cols;
  ends[1] = mat.n_rows;
  size_t size = mat.n_rows * mat.n_cols;
  ds.ReadRangeData(starts, ends, size, mat.memptr());
  mat = trans(mat); // armadillo is stored column major, we want row major...
}

template <typename T> 
bool H5Arma::ReadMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat) {
  std::string file, h5path;
  bool okPath = H5File::SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.OpenForReading();
  H5DataSet *ds = h5File.OpenDataSet(h5path);
  ION_ASSERT(ds != NULL, "Couldn't open dataset: " + h5path);
  ReadMatrix(*ds, mat);
  ds->Close();
  h5File.Close();
  return true;
}

template <typename T>
void  H5Arma::WriteMatrix(H5File &h5, const std::string &h5path, arma::Mat<T> &mat) {
  H5DataSet *ds = CreateDataSet(h5, h5path, mat, 0);
  ION_ASSERT(ds != NULL, "Couldn't make dataset: " + h5path);
  WriteMatrix(*ds, mat);
}

template <typename T> 
bool  H5Arma::WriteMatrix(const std::string &hdf5FilePath, arma::Mat<T> &mat, bool overwrite) {
  std::string file, h5path;
  bool okPath = H5File::SplitH5FileGroup(hdf5FilePath, file, h5path);
  ION_ASSERT(okPath, "Could not find valid ':' to split on in path: '" + hdf5FilePath + "'");
  H5File h5File(file);
  h5File.Open(overwrite);
  H5DataSet *ds = CreateDataSet(h5File, h5path, mat, 3);
  ION_ASSERT(ds != NULL, "Couldn't make dataset: " + h5path);
  WriteMatrix(*ds, mat);
  ds->Close();
  h5File.Close();
  return true;
}



#endif // H5ARMA_H
