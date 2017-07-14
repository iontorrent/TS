/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
 *
 * ClonalFilterWrapper.cu
 *
 *  Created on: Oct 21, 2014
 *      Author: jakob siegel
 */

#include "ClonalFilterWrapper.h"
#include "ClonalFilter/mixed.h"
#include "hdf5.h"

#define MIXED_PPF_CUTOFF 0.84f


static hid_t SetupCompression(hsize_t dims[1]);
static void WriteDSet(
    hid_t       file_id,
    hid_t       dataspace_id,
    hid_t       dcpl,
    hid_t       type_id,
    hid_t       mem_type_id,
    const char* name,
    const void* data);

ClonalFilterWrapper::ClonalFilterWrapper(unsigned short * bfMask, LayoutCubeWithRegions<unsigned short> & BeadStateMask, LayoutCubeWithRegions<float> & ClonalFilterCube )
{

  pBeadStateMask = &BeadStateMask;
  pClonalFilterCube = &ClonalFilterCube;
  pBfMask = bfMask;
  ImgRegParams irp = pBeadStateMask->getParams();
  pClonalFilterCube->setRWStrideZ();
  size_t idx = 0;
  for( size_t iy = 0; iy< irp.getImgH(); iy++){
    for( size_t ix = 0; ix< irp.getImgW(); ix++){
      if(pBfMask[idx++] & MaskLive){
        unsigned short maskvalue = pBeadStateMask->getAt(ix,iy);
        if( (maskvalue & BkgMaskRandomSample) && !(maskvalue & BkgMaskBadRead)){
          pClonalFilterCube->setRWPtr(ix,iy); //planes: KeyNorm, PPF, SSQ
          float ppfval = pClonalFilterCube->read();
          if(ppfval < MIXED_PPF_CUTOFF){
            col.push_back (ix);
            row.push_back (iy);
            ppf.push_back (ppfval);
            ssq.push_back (pClonalFilterCube->read());
            nrm.push_back (pClonalFilterCube->read());
          } // below cutoff
        } // random sample & not bad read
      } // live bead
    } // x
  }// y
}

void ClonalFilterWrapper::DumpPPFSSQ(const char * results_folder){
  string fname = string (results_folder) + "/BkgModelFilterData.txt";
  ofstream out (fname.c_str());
  assert (out);

  deque<int>::const_iterator   r = row.begin();
  deque<int>::const_iterator   c = col.begin();
  deque<float>::const_iterator p = ppf.begin();
  deque<float>::const_iterator s = ssq.begin();
  deque<float>::const_iterator n = nrm.begin();
  for (; p!=ppf.end(); ++r, ++c, ++p, ++s, ++n)
  {
    out << setw (6) << *r
        << setw (6) << *c
        << setw (8) << setprecision (2) << fixed << *p
        << setw (8) << setprecision (2) << fixed << *s
        << setw (8) << setprecision (2) << fixed << *n
        << endl;
  }
}


void ClonalFilterWrapper::ApplyClonalFilter (const PolyclonalFilterOpts & opts)
{
  clonal_filter filter;
  filter_counts counts;
  make_filter (filter, counts, ppf, ssq, opts);

  ImgRegParams irp = pBeadStateMask->getParams();
  pClonalFilterCube->setRWStrideZ();
  for(size_t i=0; i < irp.getImgSize(); i++ ){
    unsigned short maskValue = pBfMask[i];
    if(maskValue & MaskLive){
      if(maskValue & MaskLib){
        //pClonalFilterCube->setRWPtrIdx(i);
        float bead_ppf = pClonalFilterCube->getAtIdx(i,PolyPpf);
        float bead_ssq = pClonalFilterCube->getAtIdx(i,PolySsq);
        if(!filter.is_clonal(bead_ppf,bead_ssq,opts.mixed_stringency))
            (*pBeadStateMask)[i] |= BkgMaskPolyClonal;
      }
    } // live bead
  }
}

/////////////
//static stuff

void ClonalFilterWrapper::DumpPPFSSQtoH5(const char * results_folder)
{
  vector<int16_t> row;
  vector<int16_t> col;
  vector<float>   ppf;
  vector<float>   ssq;
  vector<float>   nrm;

  ImgRegParams irp = pBeadStateMask->getParams();
  pClonalFilterCube->setRWStrideZ();
  size_t idx = 0;
  for( size_t iy = 0; iy< irp.getImgH(); iy++){
    for( size_t ix = 0; ix< irp.getImgW(); ix++){
      if(pBfMask[idx++] & MaskLive){
        unsigned short maskvalue = pBeadStateMask->getAt(ix,iy);
        if(maskvalue & BkgMaskRandomSample)
        {
          pClonalFilterCube->setRWPtr(ix,iy); //planes: KeyNorm, PPF, SSQ
          col.push_back (ix);
          row.push_back (iy);
          ppf.push_back (pClonalFilterCube->read());
          ssq.push_back (pClonalFilterCube->read());
          nrm.push_back (pClonalFilterCube->read());
        } // random sample
      } // live bead
    } //x
  }//y

  string fname = string (results_folder) + "/BkgModelFilterData.h5";
  assert(row.size() > 0);
  SaveH5(fname.c_str(), row, col, ppf, ssq, nrm);
}


void ClonalFilterWrapper::SaveH5(
    const char*            fname,
    const vector<int16_t>& row,
    const vector<int16_t>& col,
    const vector<float>&   ppf,
    const vector<float>&   ssq,
    const vector<float>&   nrm)
{
  hsize_t dims[1];
  dims[0] = row.size();

  hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  assert(file_id >= 0);

  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
  assert(dataspace_id >= 0);

  hid_t dcpl = SetupCompression(dims);

  WriteDSet(file_id, dataspace_id, dcpl, H5T_STD_I16BE,  H5T_NATIVE_SHORT, "row", row.data());
  WriteDSet(file_id, dataspace_id, dcpl, H5T_STD_I16BE,  H5T_NATIVE_SHORT, "col", col.data());
  WriteDSet(file_id, dataspace_id, dcpl, H5T_IEEE_F32BE, H5T_NATIVE_FLOAT, "ppf", ppf.data());
  WriteDSet(file_id, dataspace_id, dcpl, H5T_IEEE_F32BE, H5T_NATIVE_FLOAT, "ssq", ssq.data());
  WriteDSet(file_id, dataspace_id, dcpl, H5T_IEEE_F32BE, H5T_NATIVE_FLOAT, "nrm", nrm.data());

  herr_t status = H5Sclose(dataspace_id);
  assert(status >= 0);

  status = H5Fclose(file_id);
  assert(status >= 0);
}



void ClonalFilterWrapper::UpdateMask()
{
  ImgRegParams irp = pBeadStateMask->getParams();
  for (size_t well=0; well<irp.getImgSize(); ++well)
  {
    if(pBfMask[well] & MaskLive){
      if(pBfMask[well] & MaskLib){
        unsigned short maskValue = (*pBeadStateMask)[well];
        if(maskValue & BkgMaskBadRead)  // set in bfMask instead
          pBfMask[well] |= MaskFilteredBadKey;  //
        else if(pClonalFilterCube->getAtIdx(well,PolyPpf) >= MIXED_PPF_CUTOFF)
          pBfMask[well] |=  MaskFilteredBadResidual;
        else if( maskValue & BkgMaskPolyClonal)
          pBfMask[well] |=  MaskFilteredBadPPF;
      }
    }
  }
}


//HDF5 static setup functions
static hid_t SetupCompression(hsize_t dims[1])
{
    htri_t avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
    assert(avail);

    unsigned int filter_info = 0;
    herr_t status = H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
    assert( (status >= 0) && (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) && (filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED));

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_deflate(dcpl, 9);
    assert(status >= 0);

    status = H5Pset_chunk(dcpl, 1, dims);
    assert(status >= 0);

    return dcpl;
}

static void WriteDSet(
    hid_t       file_id,
    hid_t       dataspace_id,
    hid_t       dcpl,
    hid_t       type_id,
    hid_t       mem_type_id,
    const char* name,
    const void* data)
{
    hid_t dataset_id = H5Dcreate(file_id, name, type_id, dataspace_id, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    assert(dataset_id >= 0);

    herr_t status = H5Dwrite(dataset_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    assert(status >= 0);

    status = H5Dclose(dataset_id);
    assert(status >= 0);
}




