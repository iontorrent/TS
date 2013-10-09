/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "H5Replay.h"
#include <assert.h>

using namespace std;


// *********************************************************************
// Dataset utility class

ReplayH5DataSet::ReplayH5DataSet() {
  // mGroup = "/";
  mName = NULL;
  mDataset = EMPTY;
  mDatatype = EMPTY;
  mDataspace = EMPTY;
  mPList = EMPTY;
  mStatus = -1;
  mRank = 0;
  mCompression = 0;
  for(int i=0; i<3; i++){
    mDims[i] = 0;
    mChunkDims[i] = 0;
    maxdims[i] = H5S_UNLIMITED;
  }
  mCreated = false;
}

ReplayH5DataSet::~ReplayH5DataSet() {
  Close();
  
  if (mName != NULL)
    free(mName);

}


void ReplayH5DataSet::Init(const char *datasetname, hid_t datatype,
			    int rank)
{
  mName = strdup(datasetname);
  mDatatype = H5Tcopy(datatype);  // set to EMPTY on Close()
  assert (mDatatype >=0 );
  assert( rank <= 3 );
  mRank = rank;
}
  
bool ReplayH5DataSet::IsCreated(hid_t hFile)
{
  // returns true if the dataset mName exists in the HDF5 file
  // @TODO currently only checks the leaf of mName e.g., so /foo/bar
  //  would always return true if /baz/bar existed
  if (!mCreated)
  {
    // turn off hdf5 error reporting (default to stdout)
    H5E_auto2_t func;
    void *client_data;
    H5Eget_auto2(H5E_DEFAULT, &func, &client_data);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    htri_t status;
    // @TODO tests only the leaf of mName
    // will error out if a path does not exist?
    status = H5Lexists(hFile, mName, H5P_DEFAULT);
    if (status > 0 )
      mCreated = true;

    // restore error reporting
    H5Eset_auto2(H5E_DEFAULT, func, client_data);
  }
  return (mCreated);
}

/* // not used...
void CreateSimpleDataset(hid_t hFile, hsize_t *dims)
{
  if (!IsCreated(hFile))
  {
    // create a non-extensible dataset with dimension matching dims
    mDataspace = H5Screate_simple (mRank, chunk_dims, NULL );
    // mDataspace = H5Screate_simple (mRank, dims, NULL );
    mPList = H5Pcreate (H5P_DATASET_CREATE);
    assert( mStatus >= 0 );
    if (mCompression > 0) {
      H5Pset_deflate(mPList, mCompression);
    }
    hid_t dapl = H5Pcreate (H5P_DATASET_ACCESS);
    hid_t linkpl = H5Pcreate(H5P_LINK_CREATE);
    herr_t status = H5Pset_create_intermediate_group(linkpl, 1);
    assert (status >= 0);
    mDataset = H5Dcreate2(hFile, mName, mDatatype, mDataspace, linkpl,
			 mPList, dapl);
    assert( mDataset >= 0 );  // @TODO not threadsafe, does this matter?
    H5Pclose(dapl);
    H5Pclose(linkpl);
    mCreated = true;
    fprintf(stdout, "H5: Created dataset: %s\n", mName);
  }
  } */

void ReplayH5DataSet::CreateDataset(hid_t hFile, hsize_t *chunk_dims)
{
  if (!IsCreated(hFile))
  {
    // create an extensible dataset with initial dimensions matching chunk_dims
    mDataspace = H5Screate_simple (mRank, chunk_dims, maxdims );
    // mDataspace = H5Screate_simple (mRank, chunk_dims, NULL );
    mPList = H5Pcreate (H5P_DATASET_CREATE);
    mStatus = H5Pset_chunk (mPList, mRank, chunk_dims);
    assert( mStatus >= 0 );
    if (mCompression > 0) {
      H5Pset_deflate(mPList, mCompression);
    }
    hid_t dapl = H5Pcreate (H5P_DATASET_ACCESS);
    hid_t linkpl = H5Pcreate(H5P_LINK_CREATE);
    herr_t status = H5Pset_create_intermediate_group(linkpl, 1);
    assert (status >= 0);
    mDataset = H5Dcreate2(hFile, mName, mDatatype, mDataspace, linkpl,
			 mPList, dapl);
    assert( mDataset >= 0 );  // @TODO not threadsafe, does this matter?
    H5Pclose(dapl);
    H5Pclose(linkpl);
    mCreated = true;
    fprintf(stdout, "H5: Created dataset: %s of rank %d with chunk dims = [", mName, (int)mRank);
    for (int i=0; i<mRank; i++)
      fprintf(stdout, " %llu ", chunk_dims[i]);
    fprintf(stdout, "]\n");
  }
}


void ReplayH5DataSet::Close() {
  if (IsOpen()) {
    H5Pclose(mPList);
    mPList = EMPTY;
    H5Sclose(mDataspace);
    mDataspace = EMPTY;
    H5Tclose(mDatatype);
    mDatatype = EMPTY;
    H5Dclose(mDataset);
    mDataset = EMPTY;
  }
}


void ReplayH5DataSet::Open(hid_t hFile)
{
  if (hFile >= 0) {
    mDataset = H5Dopen(hFile, mName, H5P_DEFAULT); // dataset handle
    assert (mDataset>=0);  // fail ungracefully

    mDataspace = H5Dget_space (mDataset);
    assert (mDataspace>=0);  // fail ungracefully

    mDatatype  = H5Dget_type(mDataset);                 // datatype handle
    assert (mDatatype>=0);  // fail ungracefully

    // mClass = H5Tget_class(mDatatype);                    // class
    mSize = H5Tget_size(mDatatype);
    assert (mSize>0);         // fail ungracefully    

    mRank = H5Sget_simple_extent_ndims(mDataspace);
    assert (mRank>=0);         // fail ungracefully;
    assert(mRank <= 3);                     // see comment in class definition

    mStatus  = H5Sget_simple_extent_dims(mDataspace, mDims, NULL);
    assert (mStatus>=0);         // fail ungracefully;

    mPList = H5Dget_create_plist (mDataset);
    assert (mPList>=0);  // fail ungracefully

    mLayout = H5Pget_layout (mPList);
    assert (mLayout>=0);  // fail ungracefully
    if (H5D_CHUNKED == mLayout)
    {
      int rankChunk = H5Pget_chunk(mPList, mRank, mChunkDims);
      assert(rankChunk == mRank);
    }
  }
}


bool ReplayH5DataSet::IsOpen()
{
  if (mDataset != EMPTY)
    return (true);
  else
    return (false);
}


void ReplayH5DataSet::Extend(hsize_t *size)
{
  hsize_t dims[3];
  int rank = H5Sget_simple_extent_dims(mDataspace, dims, NULL);
  assert ( rank == mRank );
  bool extend = false;
  for (int i=0; i<mRank; i++){
    size[i] = (size[i] < dims[i]) ? dims[i] : size[i]; // set_extent will shrink
    if (size[i]>dims[i])
      extend = true;
  }
  herr_t status;
  if (extend){
    status = H5Dset_extent(mDataset, size);

    fprintf(stdout, "H5: Extending %s from [\n", mName);
    for (int i=0; i<mRank; i++)
      fprintf(stdout, " %llu ", dims[i]);
    fprintf(stdout, "] to [ ");
    for (int i=0; i<mRank; i++)
      fprintf(stdout, " %llu ", size[i]);
    fprintf(stdout, "]\n");

    assert (status >= 0);
    // redefine the dataspace
    H5Sclose(mDataspace);
    mDataspace = H5Dget_space(mDataset);
    assert (mDataspace >= 0);
  }
}


// the core of the replay class
pthread_mutex_t H5Replay::h5_mutex = PTHREAD_MUTEX_INITIALIZER;

H5Replay::H5Replay(std::string& h5File, const char *datasetname) {
  Init(h5File, datasetname);
}
H5Replay::H5Replay(std::string& h5File, const char *datasetname, hid_t dsnType, unsigned int rank) {
  // useful dsnTypes: H5T_NATIVE_INT, H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE,
  // H5T_NATIVE_CHAR
  // assert ( rank <=3 );  // bigger ranks not supported
  Init(h5File, datasetname, dsnType, rank);
}

H5Replay::~H5Replay() {
  Close();
}

// close a file and its dataset
void H5Replay::Close()
{
  if (mHFile != ReplayH5DataSet::EMPTY) {

    mRDataset.Close();

    herr_t status = H5Fclose(mHFile);
    assert(status >= 0);
    mHFile = ReplayH5DataSet::EMPTY;
  }

  if (locked){
    // fprintf(stdout, "H5Replay::Close unlock on %u\n", (unsigned int)pthread_self());
    // fflush(stdout);
    pthread_mutex_unlock(&h5_mutex);
  }
  locked = false;
}

void H5Replay::Init(std::string& h5File, const char *datasetname)
{
  SetReplayBkgModelDataFile(h5File);  // setup location of record/replay h5 file
  mHFile = ReplayH5DataSet::EMPTY;

  locked = false;

  if (datasetname != NULL)
  {
    mRDataset.mName=strdup(datasetname);
  }
}

void H5Replay::Init(std::string& h5File, const char *datasetname, hid_t dsnType, unsigned int rank)
{
  SetReplayBkgModelDataFile(h5File);  // setup location of record/replay h5 file
  mHFile = ReplayH5DataSet::EMPTY;

  locked = false;

  if (datasetname != NULL)
  {
    mRDataset.Init(datasetname, dsnType, rank);
  }
}

void H5Replay::SetReplayBkgModelDataFile(std::string& file)
{
  if ( file.empty() ){
    char buffer[2000];
    char *out = getcwd(buffer, 2000);
    assert( out != NULL);
    mFilePath = ToStr(buffer);
    if (mFilePath[mFilePath.length()-1] != '/')
      mFilePath += '/';
    std::stringstream ss;
    ss << getpid();
    mFilePath = mFilePath + "dumpfile_" + ss.str() + ".h5";
  }
  else
    mFilePath = file;
}


// *********************************************************************
// reader specific functions

H5ReplayReader::H5ReplayReader(std::string& h5file)
  : H5Replay(h5file, NULL)
{
  CheckValid();
}

/**
 * reader to an already existing dataset in an already existing file
 */
H5ReplayReader::H5ReplayReader(std::string& h5File, const char *datasetname)
  : H5Replay(h5File, datasetname)
{
  Open();         // test we can open the file

  // test the dataset datasetname exists in the file
  if (!mRDataset.IsCreated(mHFile) )
    { ION_ABORT("Couldn't open file: " + mFilePath +
		"with dataset: " + std::string(datasetname)); }

  Close();        // close it down
}

/**
 * open a file read only, leaves open
 */
void H5ReplayReader::Open()
{
  pthread_mutex_lock(&h5_mutex);
  locked = true;
  // fprintf(stdout, "H5ReplayReader::Open lock on %u\n", (unsigned int)pthread_self());
  // fflush(stdout);
  if (mHFile == ReplayH5DataSet::EMPTY) {
    // open file for reading with default property list
    mHFile = H5Fopen(mFilePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  }
  if (mHFile < 0)  { ION_ABORT("Couldn't open file: " + mFilePath); }
  // fprintf(stdout, "H5: opened file: %s with dataset: %s\n", mFilePath.c_str(), mRDataset.mName);
  if (( mHFile >= 0 ) & mRDataset.IsCreated(mHFile) ) {
    mRDataset.Open(mHFile);
  }
}

/**
 * check that the file exists and is valid, returns true if valid
 * gives an HDF5 error if not
 */
bool H5ReplayReader::CheckValid()
{
  // throws an error if not valid @TODO decide whether to return false instead
  Open();
  // @TODO check the directory structure in the future
  Close();
  return(true);
}

/**
 * Return an immutable non-closable native datatype matching this datatype
 */
hid_t H5ReplayReader::GetType()
{
  Open();
  hid_t retval;
  hid_t dt = mRDataset.GetDatatype();
  if ( H5Tequal(dt, H5T_NATIVE_CHAR) > 0)
    retval = H5T_NATIVE_CHAR;
  else if ( H5Tequal(dt, H5T_NATIVE_SCHAR) > 0)
    retval = H5T_NATIVE_SCHAR;
  else if ( H5Tequal(dt, H5T_NATIVE_UCHAR) > 0)
    retval = H5T_NATIVE_UCHAR;
  else if ( H5Tequal(dt, H5T_NATIVE_SHORT) > 0)
    retval = H5T_NATIVE_SHORT;
  else if ( H5Tequal(dt, H5T_NATIVE_USHORT) > 0)
    retval = H5T_NATIVE_USHORT;
  else if ( H5Tequal(dt, H5T_NATIVE_INT) > 0)
    retval = H5T_NATIVE_INT;
  else if ( H5Tequal(dt, H5T_NATIVE_UINT) > 0)
    retval = H5T_NATIVE_UINT;
  else if ( H5Tequal(dt, H5T_NATIVE_LONG) > 0)
    retval = H5T_NATIVE_LONG;
  else if ( H5Tequal(dt, H5T_NATIVE_ULONG) > 0)
    retval = H5T_NATIVE_ULONG;
  else if ( H5Tequal(dt, H5T_NATIVE_LLONG) > 0)
    retval = H5T_NATIVE_LLONG;
  else if ( H5Tequal(dt, H5T_NATIVE_ULLONG) > 0)
    retval = H5T_NATIVE_ULLONG;
  else if ( H5Tequal(dt, H5T_NATIVE_FLOAT) > 0)
    retval = H5T_NATIVE_FLOAT;
  else if ( H5Tequal(dt, H5T_NATIVE_DOUBLE) > 0)
    retval = H5T_NATIVE_DOUBLE;
  else if ( H5Tequal(dt, H5T_NATIVE_LDOUBLE) > 0)
    retval = H5T_NATIVE_LDOUBLE;
  else if ( H5Tequal(dt, H5T_NATIVE_B8) > 0)
    retval = H5T_NATIVE_B8;
  else if ( H5Tequal(dt, H5T_NATIVE_B16) > 0)
    retval = H5T_NATIVE_B16;
  else if ( H5Tequal(dt, H5T_NATIVE_B32) > 0)
    retval = H5T_NATIVE_B32;
  else if ( H5Tequal(dt, H5T_NATIVE_B64) > 0)
    retval = H5T_NATIVE_B64;
  else
    retval = -1;  // don't know about this datatype
  Close();
  return(retval);
}
  
    

// *********************************************************************
// recorder specific functions


/**
 * recorder in a file, create if necessary, dataset not specified
 */
H5ReplayRecorder::H5ReplayRecorder(std::string& h5File)
  : H5Replay(h5File, NULL) {}

/**
 * recorder to an already existing dataset
 */
H5ReplayRecorder::H5ReplayRecorder(std::string& h5File, char *datasetname)
  : H5Replay(h5File, datasetname)
{
  if (datasetname != NULL){
    Open();
    Close();
  }
}

/**
 * recorder to a possibly new dataset in a possibly new file
 * If it already exists type and rank will be ignored
 */
H5ReplayRecorder::H5ReplayRecorder(std::string& h5File, char *datasetname, hid_t dsnType, unsigned int rank)
  : H5Replay(h5File, datasetname, dsnType, rank) {}

/**
 * create a file for write, truncates; if the file is already created, just return
 */
void H5ReplayRecorder::CreateFile()
{
  pthread_mutex_lock(&h5_mutex);
  locked = true;
  if (FileNotCreated(mFilePath)) {
    mHFile = H5Fcreate(mFilePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    fprintf(stdout, "H5: Created background data recorder file: %s\n", mFilePath.c_str());
  }
  
  if (mHFile < 0) { ION_ABORT("Couldn't create file: " + mFilePath); }
  Close();
}

/**
 * create a dataset if it is not already created, open & close file
 */
void H5ReplayRecorder::CreateDataset(vector<hsize_t>& chunk_dims)
{
  Open();
  assert( (mRDataset.IsInitialized())
	  & (mRDataset.GetRank() == chunk_dims.size()));
  mRDataset.CreateDataset(mHFile, &chunk_dims[0]);
  Close();
}

/**
 * open a file for write, leave open
 */
void H5ReplayRecorder::Open()
{
  pthread_mutex_lock(&h5_mutex);
  locked = true;
  // fprintf(stdout, "H5ReplayRecorder::Open lock on %u\n", (unsigned int)pthread_self());
  // fflush(stdout);
  if (mHFile == ReplayH5DataSet::EMPTY) {
    // open file for read/write with default property list
    mHFile = H5Fopen(mFilePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  }
  if (mHFile < 0)  { ION_ABORT("Couldn't open file: " + mFilePath); }
  if (mHFile >= 0) {
    if ( mRDataset.IsCreated(mHFile) ) {
      mRDataset.Open(mHFile);
    }
  }
}

/**
 * extend a dataset, open & close file
 */
void H5ReplayRecorder::ExtendDataSet(vector<hsize_t>& extension)
{
  Open();
  assert( mRDataset.IsCreated(mHFile) & (mRDataset.GetRank() == extension.size()));
  mRDataset.Extend(&extension[0]);
  Close();
}

bool H5ReplayRecorder::FileNotCreated(std::string& h5File)
{
  FILE * pFile;
  pFile = fopen(h5File.c_str(), "r+");
  if (pFile != NULL) { // file must exist for read and update
    fclose (pFile);
    return (false);
  }
  return (true); // file may not exist, or may not be readable or writeable
}
