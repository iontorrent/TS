/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <string.h>
#include "IonH5File.h"
#include "Utils.h"

using namespace std;

bool Info::GetValue ( const std::string &key, std::string &value ) const
{
  if ( mMap.find ( key ) == mMap.end() )
  {
    return false;
  }
  value =  mValues[mMap.find ( key )->second];
  return true;
}

bool Info::SetValue ( const std::string &key, const std::string &value )
{
  if ( mMap.find ( key ) != mMap.end() )
  {
    mValues[mMap[key]] = value;
  }
  else
  {
    mKeys.push_back ( key );
    mValues.push_back ( value );
    mMap[key] = int ( mKeys.size() ) - 1;
  }
  return true;
}

bool Info::GetEntry ( int index, std::string &key, std::string &value ) const
{
  if ( index >= ( int ) mKeys.size() || index < 0 )
  {
    ION_ABORT ( "ERROR - Index: " + ToStr ( index ) + " not valid for max index: " + ToStr ( mKeys.size() -1 ) );
    return false;
  }
  key = mKeys[index];
  value = mValues[index];
  return true;
}

int Info::GetCount() const
{
  return ( int ) mValues.size();
}

bool Info::KeyExists ( const std::string &key ) const
{
  if ( mMap.find ( key ) != mMap.end() )
  {
    return true;
  }
  return false;
}

void Info::Clear()
{
  mKeys.clear();
  mValues.clear();
  mMap.clear();
}

void H5DataSet::Init ( size_t id )
{
  mId = id;
  mDataSet = H5_NULL;
  mGroup = H5_NULL;
  mDataspace = H5_NULL;
  mDatatype = H5_NULL; // @todo - do we need both mType and mDataType?
  mParent = NULL;
  mRank = 0;
  mCompression = 5;
  mType = 0;
  mSize = 0;
}

void H5DataSet::SetDataspace ( int rank, const hsize_t dims[], const hsize_t chunking[], hid_t type )
{
  mRank = rank;
  mDims.resize ( rank );
  mChunking.resize ( rank );
  mType = type;
  for ( size_t i = 0; i < mDims.size(); i++ )
  {
    mDims[i] = dims[i];
    mChunking[i] = chunking[i];
  }
}

bool H5DataSet::CreateDataSet()
{
  ION_ASSERT ( mDataSet == H5_NULL, "DataSet already appears open." );
  // create an extensible dataset with initial dimensions matching mDims
  std::vector<hsize_t> maxdims ( mRank, H5S_UNLIMITED );
  mDataspace = H5Screate_simple ( mRank, &mDims[0], &maxdims[0] );
  // mDataspace = H5Screate_simple(mRank, &mDims[0], NULL);
  mDatatype = H5Tcopy ( mType );
  mSize = H5Tget_size ( mDatatype );
  hid_t plist;
  plist = H5Pcreate ( H5P_DATASET_CREATE );
  H5Pset_chunk ( plist, mRank, &mChunking[0] );
  if ( mCompression > 0 )
  {
    H5Pset_deflate ( plist, mCompression );
  }
  hid_t dapl = H5Pcreate ( H5P_DATASET_ACCESS );
  hid_t group = mGroup;
  if ( mGroup == H5_NULL )
  {
    group = mParent->GetFileId();
  }
  mDataSet = H5Dcreate2 ( group, mName.c_str(), mDatatype, mDataspace,
                          H5P_DEFAULT, plist, dapl );
  H5Pclose (plist);
  if ( mDataSet < 0 )
  {
    Init ( 0 );
    return false;
  }
  return true;
}

bool H5DataSet::OpenDataSet()
{
  ION_ASSERT ( mDataSet == H5_NULL, "DataSet already appears open." );
  hid_t group = mGroup;
  if ( mGroup == H5_NULL )
  {
    group = mParent->GetFileId();
  }
  mDataSet = H5Dopen2 ( group, mName.c_str(), H5P_DEFAULT );
  if ( mDataSet < 0 )
  {
    Init ( 0 );
    return false;
  }
  mDatatype  = H5Dget_type ( mDataSet );  /* datatype handle */
  mType = mDatatype;
  mSize  = H5Tget_size ( mDatatype );
  mDataspace = H5Dget_space ( mDataSet );
  int rank = H5Sget_simple_extent_ndims ( mDataspace );
  mDims.resize ( rank );
  mRank = rank;
  int status = H5Sget_simple_extent_dims ( mDataspace, &mDims[0], NULL );
  ION_ASSERT ( status >= 0, "Couldn't get data for dataset: '" + mName + "'" )
  return true;
}

void H5DataSet::Close()
{
  // Tell parent that owns this child to close and cleanup.
  mParent->CloseDataSet ( GetId() );
}

void H5DataSet::CloseDataSet()
{
  if ( mDataSet != H5_NULL )
  {
    H5Dclose ( mDataSet );
    mDataSet = H5_NULL;
    H5Sclose ( mDataspace );
    mDataspace = H5_NULL;
    H5Tclose ( mDatatype );
    mDatatype = H5_NULL;
    if ( mGroup != mParent->GetFileId() && mGroup >= 0 ) // mgroup is -1 for root, can't close that one
    {
      H5Gclose ( mGroup );
      mGroup = H5_NULL;
    }
  }
  mRank = 0;
  mDims.resize ( 0 );
  mChunking.resize ( 0 );
  mCompression = 0;
  mType = 0;
  mName = "";
}

// starts[], ends[]-1 define the hyperslab in the file
// e.g. a 4x2 matrix written 0,0 will have 8 elements with starts=[0,0],
// ends=[4,2], so ordering is: [0,0], [0,1], [1,0] [1,1], ..., [3,0], [3,1]
// memory hyperslab assumed contiguous starting at 0 and has size exactly
// the product of the dimensions of the file hyperslab
void H5DataSet::GetSelectionSpace ( const size_t *starts, const size_t *ends,
                                    hid_t &memspace )
{
  hsize_t count[mRank];        /* size of the hyperslab in the file */
  hsize_t offset[mRank];       /* hyperslab offset in the file */
  unsigned int memRank = 1;    /* rank of hyperslab in memory */
  hsize_t count_out[memRank];  /* size of the hyperslab in memory */
  hsize_t offset_out[memRank]; /* hyperslab offset in memory */
  count_out[0] = 1;
  offset_out[0] = 0;
  for ( size_t i = 0; i < mRank; i++ )
  {
    //@todo - check that we are within our ranges here
    offset[i] = starts[i];
    count[i] = ends[i] - starts[i];
    count_out[0] = count_out[0] * count[i];
  }
  herr_t status = 0;
  status = H5Sselect_hyperslab ( mDataspace, H5S_SELECT_SET, offset, NULL, count, NULL );
  ION_ASSERT ( status >= 0, "Couldn't select hyperslab from dataspace for read." );
  memspace = H5Screate_simple ( memRank,count_out,NULL );
  status = H5Sselect_hyperslab ( memspace, H5S_SELECT_SET, offset_out, NULL,
                                 count_out, NULL );
  ION_ASSERT ( status >= 0, "Couldn't select hyperslab from memspace for read." );
}

// when incrementally adding to a data set with bounds outside
// of the existing dataset, extend to encompass the new data
// Only works with chunked datasets
void H5DataSet::ExtendIfNeeded ( const size_t *size )
{
  Extend ( size, true );
}

void H5DataSet::Extend ( const size_t *size, bool onlyBigger )
{
  hsize_t dims[mRank];
  hsize_t newsize[mRank];
  unsigned int rank = H5Sget_simple_extent_dims ( mDataspace, dims, NULL );
  ION_ASSERT ( rank == mRank, "rank of dataset in file doesn't match expected rank" );
  bool extend = false;
  for ( unsigned int i=0; i<mRank; i++ )
  {
    ION_ASSERT ( mDims[i] == dims[i], "Dimension in dataset mismatch with expected" );
    if ( onlyBigger )
    {
      newsize[i] = ( size[i] < dims[i] ) ? dims[i] : size[i]; // to >= existing dims
      if ( newsize[i]>dims[i] )
        extend = true;
    }
    else
    {
      newsize[i] = size[i];  // this can shrink the dataset
      extend = true;
    }
  }
  herr_t status;
  if ( extend )
  {
    status = H5Dset_extent ( mDataSet, newsize );
    ION_ASSERT ( status >= 0, "Extending dataset in file failed" );
    if ( rank ==1 )
      fprintf ( stdout, "H5: Extending %s from [%d] to [%d]\n", mName.c_str(), ( int ) dims[0], ( int ) newsize[0] );
    else if ( rank==2 )
      fprintf ( stdout, "H5: Extending %s from [%d, %d] to [%d, %d]\n", mName.c_str(), ( int ) dims[0], ( int ) dims[1], ( int ) newsize[0], ( int ) newsize[1] );
    else
      fprintf ( stdout, "H5: Extending %s from [%d, %d, %d] to [%d, %d, %d]\n", mName.c_str(), ( int ) dims[0], ( int ) dims[1], ( int ) dims[2], ( int ) newsize[0], ( int ) newsize[1], ( int ) newsize[2] );

    // redefine the dataspace
    H5Sclose ( mDataspace );
    mDataspace = H5Dget_space ( mDataSet );
    ION_ASSERT ( mDataspace >= 0, "Couldn't reopen dataset after extending" );
    unsigned int rank = H5Sget_simple_extent_dims ( mDataspace, dims, NULL );
    assert ( rank == mRank );
    for ( unsigned int i=0; i<mRank; i++ )
      mDims[i] = dims[i];
  }
}

H5File::H5File()
{
  Init();
}

void H5File::Init()
{
  mReadOnly = false;
  mCurrentId = 0;
  mHFile = H5DataSet::H5_NULL;
}

hid_t H5File::GetOrCreateGroup ( hid_t location, const std::string &groupPath, std::vector<hid_t> &freeList )
{
  std::string first =  groupPath;
  std::string second;
  size_t pos = groupPath.find ( '/' );
  if ( pos == string::npos )      // base case, no '/'s left
  {
    hid_t g = H5Gopen ( location, groupPath.c_str(), H5P_DEFAULT );
    if ( g <= 0 )
    {
      g = H5Gcreate ( location, groupPath.c_str(), 0,  H5P_DEFAULT, H5P_DEFAULT );
    }
    return g;
  }
  else   // recurse down a level
  {
    first = groupPath.substr ( 0,pos+1 );
    second = groupPath.substr ( pos+1, groupPath.length() );
    hid_t g = location;
    g = H5Gopen ( location, first.c_str(), H5P_DEFAULT );
    freeList.push_back ( g );
    if ( g <= 0 )
    {
      g = H5Gcreate ( location, first.c_str(), 0,  H5P_DEFAULT, H5P_DEFAULT );
      freeList.push_back ( g );
    }
    if ( second.length() > 0 )
    {
      return GetOrCreateGroup ( g, second, freeList );
    }
    else
    {
      return g;
    }
  }
  return H5DataSet::H5_NULL;
}

hid_t H5File::GetOrCreateGroup ( hid_t location, const std::string &groupPath )
{

  /* Save old error handler */
  herr_t ( *old_func ) ( hid_t, void * );
  void *old_client_data;

  H5Eget_auto ( H5E_DEFAULT, &old_func, &old_client_data );

  /* Turn off error handling */
  H5Eset_auto ( H5E_DEFAULT, NULL, NULL );

  std::vector<hid_t> groups;
  hid_t group = GetOrCreateGroup ( location, groupPath, groups );

  for ( size_t i = 0; i < groups.size(); i++ )
  {
    H5Gclose ( groups[i] );
  }

  /* Restore previous error handler */
  H5Eset_auto ( H5E_DEFAULT, old_func, old_client_data );
  return group;
}

void H5File::FillInGroupFromName ( const std::string &path, std::string &groupName, std::string &dsName )
{
  size_t pos = path.rfind ( '/' );
  if ( pos == string::npos )
  {
    groupName = "/";
    dsName = path;
  }
  else
  {
    groupName = path.substr ( 0,pos );
    dsName = path.substr ( pos+1, path.length() - pos - 1 );
  }
}

bool H5File::Open ( bool overwrite )
{
  bool isH5 =  IsH5File ( mName );
  if ( isH5 && !overwrite && mReadOnly )
  {
    mHFile = H5Fopen ( mName.c_str(), H5F_ACC_RDONLY,  H5P_DEFAULT );
  }
  else if ( isH5 && !overwrite )
  {
    mHFile = H5Fopen ( mName.c_str(), H5F_ACC_RDWR,  H5P_DEFAULT );
  }
  else if ( overwrite || !isFile ( mName.c_str() ) )
  {
    mHFile = H5Fcreate ( mName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
  }
  if ( mHFile < 0 )
  {
    //ION_ABORT("Couldn't open file: " + mName);
    return false;
  }
  return true;
}

bool H5File::OpenNew()
{
  mHFile = H5Fcreate ( mName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
  if ( mHFile < 0 )
  {
    //ION_ABORT("Couldn't open file: " + mName);
    return false;
  }
  return true;
}

bool H5File::OpenForReading ( void )
{
  mHFile = H5Fopen ( mName.c_str(), H5F_ACC_RDONLY,  H5P_DEFAULT );
  if ( mHFile < 0 )
  {
    return false;
    //    ION_ABORT("Couldn't open file: " + mName);
  }
  return true;
}

bool H5File::IsH5File ( const std::string &path )
{
  hid_t suspectFile;
  bool isH5 = false;
  H5E_auto2_t old_func;
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2 ( H5E_DEFAULT, &old_func, &old_client_data );
  H5Eset_auto2 ( H5E_DEFAULT, NULL, NULL );
  suspectFile = H5Fopen ( path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  if ( suspectFile >= 0 )
  {
    isH5 = true;
    H5Fclose ( suspectFile );
  }
  H5Eset_auto2 ( H5E_DEFAULT, old_func, old_client_data );
  return isH5;
}

H5DataSet * H5File::CreateDataSet ( const std::string &name,
                                    hsize_t rank,
                                    const hsize_t dims[],
                                    const hsize_t chunking[],
                                    int compression,
                                    hid_t type )
{
  size_t id = mCurrentId;
  mCurrentId++;
  H5DataSet *ds = new H5DataSet ( id );
  string groupName, dsName;
  FillInGroupFromName ( name, groupName, dsName );
  hid_t group = GetOrCreateGroup ( mHFile, groupName );
  ds->SetGroup ( group );
  ds->SetParent ( this );
  ds->SetName ( dsName );
  ds->SetCompression ( compression );
  ds->SetDataspace ( rank, dims, chunking, type );
  if ( ds->CreateDataSet() )
  {
    mDSMap[id] = ds;
    return mDSMap[id];
  }
  delete ds;
  return NULL;
}

H5DataSet * H5File::OpenDataSet ( const std::string &name )
{
  size_t id = mCurrentId;
  mCurrentId++;
  H5DataSet *ds = new H5DataSet ( id );
  string groupName, dsName;
  FillInGroupFromName ( name, groupName, dsName );
  hid_t group = GetOrCreateGroup ( mHFile, groupName );
  ds->SetGroup ( group );
  ds->SetParent ( this );
  ds->SetName ( dsName );
  /* Save old error handler */
  herr_t ( *old_func ) ( hid_t, void * );
  void *old_client_data;
  H5Eget_auto ( H5E_DEFAULT, &old_func, &old_client_data );
  /* Turn off error handling */
  H5Eset_auto ( H5E_DEFAULT, NULL, NULL );
  if ( ds->OpenDataSet() )
  {
    mDSMap[id] = ds;
    H5Eset_auto ( H5E_DEFAULT, old_func, old_client_data );
    return mDSMap[id];
  }
  /* Restore previous error handler */
  H5Eset_auto ( H5E_DEFAULT, old_func, old_client_data );
  delete ds;
  return NULL;
}

void H5File::Close()
{
  while ( !mDSMap.empty() )
  {
    std::map<size_t, H5DataSet *>::iterator i = mDSMap.begin();
    i->second->Close();
    //(i->second) = H5DataSet::H5_NULL;
  }
  if ( mHFile != H5DataSet::H5_NULL )
  {
    H5Fclose ( mHFile );
    mHFile = H5DataSet::H5_NULL;
  }
}

void H5File::CloseDataSet ( size_t id )
{
  std::map<size_t, H5DataSet *>::iterator i = mDSMap.find ( id );
  ION_ASSERT ( i != mDSMap.end(), "Couldn't find id: " + ToStr ( id ) );
  i->second->CloseDataSet();
  delete i->second;
  mDSMap.erase ( i );
}

bool H5File::SplitH5FileGroup ( const std::string &hdf5FilePath,
                                std::string &diskPath,
                                std::string &h5Path )
{
  size_t pos = hdf5FilePath.rfind ( ':' );
  if ( pos == std::string::npos )
  {
    return false;
  }
  diskPath = hdf5FilePath.substr ( 0,pos );
  h5Path = hdf5FilePath.substr ( pos+1,hdf5FilePath.length() - ( pos+1 ) );
  TrimString ( diskPath );
  TrimString ( h5Path );
  if ( ! ( diskPath.empty() || h5Path.empty() ) )
  {
    return true;
  }
  return false;
}
hid_t H5File::CreateAttribute ( const hid_t obj_id, const char *name, const char *msg = NULL )
{

  // 1.  create DataSpace
  //hsize_t dims = 2;
  //hid_t space_id = H5Screate_simple(1, &dims, NULL);
  hid_t space_id = H5Screate ( H5S_SCALAR );

  // 2. make attribute (need to call writeAttribute to assign values to it later)
  hid_t attr_id = -1; // negative value is fail
  try
  {
    hid_t datatype = H5Tcopy ( H5T_C_S1 ); // cannot use H5T_C_S1 directly??
    size_t sz = strlen ( msg );
    if ( sz <= 0 )
      sz = 1;
    herr_t status = H5Tset_size ( datatype,sz );  // need set_size() before Create(), but size has to be positive!!
    if ( status<0 )
    {
      ION_ABORT ( "Internal Error: H5Tset_size bad in CreateAttribute" );
    }

    attr_id = H5Acreate ( obj_id, name, datatype, space_id, H5P_DEFAULT, H5P_DEFAULT );
    if ( msg != NULL )
    {
      //WriteAttribute(attr_id,msg);
      status = H5Awrite ( attr_id,datatype,msg );
      H5Aclose ( attr_id );
    }
    H5Tclose ( datatype );
    H5Sclose ( space_id );
  }
  catch ( ... )
  {
    cerr <<"EXCEPTION: Attribute " << name << " cannot be created in H5File::CreateAttribute()" << endl;
  }

  // 3. return attr_id
  return attr_id;

}



herr_t H5File::WriteAttribute ( const hid_t attr_id, const char *msg = NULL )
{
  hid_t datatype = H5Tcopy ( H5T_C_S1 );
  herr_t status = H5Tset_size ( datatype, strlen ( msg ) );
  status = H5Awrite ( attr_id,datatype,msg );
  status = H5Aclose ( attr_id );
  status = H5Tclose ( datatype );
  return status;
}



void H5File::makeParamNames ( const char * param_prefix, int nParams, std::string &outStr )
{
  stringstream ss;

  for ( int i=0; i<nParams; i++ )
  {
    if ( i>0 )
      ss << ", ";
    ss << param_prefix << i;
  }
  outStr = ss.str(); // ss needs to be const to avoid this get destroyed or re-allocated??
}

void H5File::WriteStringVector ( const std::string &name, const char **values, int numValues )
{
  H5DataSet set;
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

  set.mDataSet = H5Dcreate2 ( mHFile, set.mName.c_str(), set.mDatatype, set.mDataspace,
                              H5P_DEFAULT, plist, H5P_DEFAULT );
  status = H5Dwrite ( set.mDataSet, set.mDatatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values );
  if ( status < 0 )
  {
    ION_ABORT ( "ERRROR - Unsuccessful write to file: " + mName + " dataset: " + set.mName );
  }
  //  set.Close();
  H5Sclose ( set.mDataspace );
  H5Tclose ( set.mDatatype );
  H5Tclose ( memType );
  H5Dclose ( set.mDataSet );
  H5Pclose (plist);
}

// void H5File::WriteString ( const std::string &name, const char *value ) {
//   H5DataSet set;
//   set.mName = name;
//   // Create info dataspace
//   hsize_t dimsf[1];
//   dimsf[0] = 1;
//   set.mDataspace = H5Screate_simple ( 1, dimsf, NULL );
//   set.mDatatype = H5Tcopy ( H5T_C_S1 );
//   herr_t status = 0;
//   status = H5Tset_size ( set.mDatatype, H5T_VARIABLE );
//   if ( status < 0 ) {
//     ION_ABORT ( "Couldn't set string type to variable in set: " + name );
//   }

//   hid_t memType = H5Tcopy ( H5T_C_S1 );
//   status = H5Tset_size ( memType, H5T_VARIABLE );
//   if ( status < 0 ) {
//     ION_ABORT ( "Couldn't set string type to variable in set: " + name );
//   }

//   // Setup the chunking values
//   hsize_t cdims[1];
//   cdims[0] = std::min ( 100, strlen(value));

//   //  hid_t plist;
//   //  plist = H5Pcreate ( H5P_DATASET_CREATE );
//   hid_t plist;
//   plist = H5Pcreate ( H5P_DATASET_CREATE );
//   H5Pset_chunk ( plist, 1, cdims );

//   set.mDataSet = H5Dcreate2 ( mHFile, set.mName.c_str(), set.mDatatype, set.mDataspace,
//                               H5P_DEFAULT, plist, H5P_DEFAULT );
//   status = H5Dwrite ( set.mDataSet, set.mDatatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, value );
//   if ( status < 0 ) {
//     ION_ABORT ( "ERRROR - Unsuccessful write to file: " + mName + " dataset: " + set.mName );
//   }
//   //  set.Close();
//   H5Sclose ( set.mDataspace );
//   H5Tclose ( set.mDatatype );
//   H5Tclose ( memType );
//   H5Dclose ( set.mDataSet );
//   // H5Pclose (plist);
// }

void H5File::WriteString ( const std::string &name, const char *value ) {
  H5DataSet set;
  set.mName = name;
  // Create info dataspace
  set.mDataspace = H5Screate (H5S_SCALAR);
  set.mDatatype = H5Tcopy (H5T_C_S1);
  herr_t status = 0;
  status = H5Tset_size ( set.mDatatype, strlen(value) + 1 );
  if ( status < 0 ) {
    ION_ABORT ( "Couldn't set string type to variable in set: " + name );
  }

  hid_t memType = H5Tcopy ( H5T_C_S1 );
  status = H5Tset_size ( memType, strlen(value) + 1 );
  if ( status < 0 ) {
    ION_ABORT ( "Couldn't set string type to variable in set: " + name );
  }

  // // Setup the chunking values
  // hsize_t cdims[1];
  // cdims[0] = std::min ( (size_t)100, strlen(value));

  // //  hid_t plist;
  // //  plist = H5Pcreate ( H5P_DATASET_CREATE );
  // hid_t plist;
  // plist = H5Pcreate ( H5P_DATASET_CREATE );
  // H5Pset_chunk ( plist, 1, cdims );

  set.mDataSet = H5Dcreate2 ( mHFile, set.mName.c_str(), set.mDatatype, set.mDataspace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
  status = H5Dwrite ( set.mDataSet, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value );
  if ( status < 0 ) {
    ION_ABORT ( "ERRROR - Unsuccessful write to file: " + mName + " dataset: " + set.mName );
  }
  //  set.Close();
  H5Sclose ( set.mDataspace );
  H5Tclose ( set.mDatatype );
  H5Tclose ( memType );
  H5Dclose ( set.mDataSet );
  // H5Pclose (plist);
}


void H5File::ReadStringVector ( const std::string &name, std::vector<std::string> &strings )
{
  strings.clear();
  H5DataSet set;

  set.mDataSet = H5Dopen2 ( mHFile, name.c_str(), H5P_DEFAULT );
  if ( set.mDataSet < 0 )
  {
    ION_ABORT ( "Could not open data set: " + name );
  }


  set.mDatatype  = H5Dget_type ( set.mDataSet );  /* datatype handle */

  //size_t size  = H5Tget_size ( set.mDatatype );
  set.mDataspace = H5Dget_space ( set.mDataSet ); /* dataspace handle */
  int rank = H5Sget_simple_extent_ndims ( set.mDataspace );
  hsize_t dims[rank];           /* dataset dimensions */
  int status_n = H5Sget_simple_extent_dims ( set.mDataspace, dims, NULL );
  if ( status_n<0 )
  {
    ION_ABORT ( "Internal Error: H5File::ReadStringVector H5Sget_simple_extent_dims" );
  }
  char **rdata = ( char** ) malloc ( dims[0] * sizeof ( char * ) );

  hid_t memType = H5Tcopy ( H5T_C_S1 ); // C style strings
  herr_t status = H5Tset_size ( memType, H5T_VARIABLE ); // Variable rather than fixed length
  if ( status < 0 )
  {
    ION_ABORT ( "Error setting string length to variable." );
  }

  // Read the strings
  status = H5Dread ( set.mDataSet, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata );
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
  H5Sclose ( set.mDataspace );
  H5Tclose ( set.mDatatype );
  H5Dclose ( set.mDataSet );
  //  set.Close();
}


