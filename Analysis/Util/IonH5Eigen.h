/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef H5EIGEN_H
#define H5EIGEN_H

#include <Eigen/Dense>
#include "IonH5File.h"

#define EigenMat Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>

class H5Eigen {
 public:
  /** Utility function to write an entire matrix. */
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static void WriteMatrix(H5DataSet &ds,  EigenMat &mat);
  
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static void WriteMatrix(H5DataSet &ds, EigenMat &mat, int col);
  
  /** Utility function to read an entire matrix. */
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static void ReadMatrix(H5DataSet &ds, EigenMat &mat);
  
  /** Utility function to write a matrix in one go to an hdf5 file. */
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static void WriteMatrix(H5File &h5, const std::string &hdf5FilePath, EigenMat &mat);
  
  /** 
   * Create a dataset using dimensions and type from matrix.  Memory
   * owned by HFile, don't delete when finished, call Close()
   */
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static H5DataSet * CreateDataSet(H5File &h5, const std::string &name, EigenMat &mat, int compression);

    /** Utility function to write a matrix in one go to an hdf5 file. */
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static bool WriteMatrix(const std::string &hdf5FilePath, EigenMat &mat, bool overwrite);

  /** Utility function to read a matrix in one go to an hdf5 file. */
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    static bool ReadMatrix(const std::string &hdf5FilePath, EigenMat &mat);
  
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
H5DataSet * H5Eigen::CreateDataSet(H5File &h5, const std::string &name, EigenMat &mat, int compression) {
  hsize_t rank = 2;
  hsize_t dims[rank];
  hsize_t chunking[rank];
  chunking[0] = std::min(1000u, (unsigned int)mat.rows());
  dims[0] = mat.rows();
  chunking[1] = std::min(1000u, (unsigned int)mat.cols());
  dims[1] = mat.cols();
  _Scalar t = 0;
  hid_t type = h5.GetH5Type(t);
  H5DataSet *ds = h5.CreateDataSet(name, rank, dims, chunking, compression, type);
  return ds;
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void H5Eigen::WriteMatrix(H5DataSet &ds, EigenMat &mat) {
  size_t starts[2];
  size_t ends[2];
  starts[0] = starts[1] = 0;
  ends[0] = mat.rows();
  ends[1] = mat.cols();
  //size_t size = mat.rows() * mat.cols();
  EigenMat m = mat.transpose(); // armadillo is stored column major, we want row major...
  ds.WriteRangeData(starts, ends, m.data());
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void H5Eigen::WriteMatrix(H5DataSet &ds, EigenMat &mat, int col) {
  size_t starts[2];
  size_t ends[2];
  starts[0] = 0; 
  ends[0] = mat.rows();
  starts[1] = col;
  ends[1] = col+1;
  EigenMat m = mat.transpose(); // armadillo is stored column major, we want row major...
  ds.WriteRangeData(starts, ends, m.data());
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void H5Eigen::ReadMatrix(H5DataSet &ds, EigenMat &mat) {
  ION_ASSERT(ds.GetRank() == 2, "Can't read a matrix when rank isn't 2");
  std::vector<hsize_t> &dims = ds.GetDims();
  mat.resize(dims[1],dims[0]);
  size_t starts[2];
  size_t ends[2];
  starts[0] = starts[1] = 0;
  ends[0] = mat.cols();
  ends[1] = mat.rows();
  size_t size = ends[0] * ends[1];
  ds.ReadRangeData(starts, ends, size, mat.data());
  mat = mat.transpose(); // armadillo is stored column major, we want row major...
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
bool H5Eigen::ReadMatrix(const std::string &hdf5FilePath, EigenMat &mat) {
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

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void  H5Eigen::WriteMatrix(H5File &h5, const std::string &h5path, EigenMat &mat) {
  H5DataSet *ds = CreateDataSet(h5, h5path, mat, 3);
  ION_ASSERT(ds != NULL, "Couldn't make dataset: " + h5path);
  WriteMatrix(*ds, mat);
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> 
bool  H5Eigen::WriteMatrix(const std::string &hdf5FilePath, EigenMat &mat, bool overwrite) {
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

#endif // H5EIGEN_H
