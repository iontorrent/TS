/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ClonalFilter.h"
#include "ClonalFilter/mixed.h"
#include "hdf5.h"

using namespace std;

void ApplySuperMixedFilter(Mask& mask, std::vector<RegionalizedData *>& sliced_chip);
void ApplyClonalFilter(Mask& mask, std::vector<RegionalizedData *>& sliced_chip,  const deque<float>& ppf, const deque<float>& ssq, const PolyclonalFilterOpts & opts);
void UpdateMask(Mask& mask, std::vector<RegionalizedData *>& sliced_chip);
void GetFilterTrainingSample(deque<int>& row, deque<int>& col, deque<float>& ppf, deque<float>& ssq, deque<float>& nrm, std::vector<RegionalizedData *>& sliced_chip);
void DumpPPFSSQ (const char* results_folder, const deque<int>& row, const deque<int>& col, const deque<float>& ppf, const deque<float>& ssq, const deque<float>& nrm);
void DumpPPFSSQtoH5 (const char* results_folder, std::vector<RegionalizedData *>& sliced_chip);

static void SaveH5(
    const char*            fname,
    const vector<int16_t>& row,
    const vector<int16_t>& col,
    const vector<float>&   ppf,
    const vector<float>&   ssq,
    const vector<float>&   nrm);

static hid_t SetupCompression(hsize_t dims[1]);

static void WriteDSet(
    hid_t       file_id,
    hid_t       dataspace_id,
    hid_t       dcpl,
    hid_t       type_id,
    hid_t       mem_type_id,
    const char* name,
    const void* data);

static void AttemptClonalFilter(Mask& mask, const char* results_folder, std::vector<RegionalizedData *>& sliced_chip, const PolyclonalFilterOpts & opts);

void ApplyClonalFilter (Mask& mask, const char* results_folder, std::vector<RegionalizedData *>& sliced_chip, const PolyclonalFilterOpts & opts)
{
  // Never give up just because filter failed.
  try{
    AttemptClonalFilter(mask,results_folder,sliced_chip, opts);
  }catch(exception& e){
      cerr << "NOTE: clonal filter failed."
           << e.what()
           << endl;
  }catch(...){
      cerr << "NOTE: clonal filter failed." << endl;
  }
}

static void AttemptClonalFilter(Mask& mask, const char* results_folder, std::vector<RegionalizedData *>& sliced_chip, const PolyclonalFilterOpts & opts)
{
  deque<int>   row;
  deque<int>   col;
  deque<float> ppf;
  deque<float> ssq;
  deque<float> nrm;
  GetFilterTrainingSample (row, col, ppf, ssq, nrm, sliced_chip);
  DumpPPFSSQ(results_folder, row, col, ppf, ssq, nrm);
  DumpPPFSSQtoH5(results_folder, sliced_chip);
  if (opts.filter_extreme_ppf_only){
    std::cout << "************ Filtering for extreme ppf at flow 79" << std::endl;
    ApplySuperMixedFilter(mask, sliced_chip);
  }
  else {
    ApplyClonalFilter (mask, sliced_chip, ppf, ssq, opts);
  } 
  UpdateMask(mask, sliced_chip);
}

void UpdateMask(Mask& mask, std::vector<RegionalizedData *>& sliced_chip)
{
  unsigned int numRegions = sliced_chip.size();
  for (unsigned int rgn=0; rgn<numRegions; ++rgn)
  {
    RegionalizedData& local_patch     = *sliced_chip[rgn];
    int       numWells  = local_patch.GetNumLiveBeads();
    int       rowOffset = local_patch.region->row;
    int       colOffset = local_patch.region->col;

    for (int well=0; well<numWells; ++well)
    {
      BeadParams& bead  = local_patch.GetParams (well);
      bead_state&  state = *bead.my_state;

      // Record clonal reads in mask:
      int row = rowOffset + bead.y;
      int col = colOffset + bead.x;
      if(mask.Match(col, row, MaskLib)){
        if(state.bad_read)
          mask.Set(col, row, MaskFilteredBadKey);
        else if(state.ppf >= mixed_ppf_cutoff())
          mask.Set(col, row, MaskFilteredBadResidual);
        else if(not state.clonal_read)
          mask.Set(col, row, MaskFilteredBadPPF);
      }
    }
  }
}

void ApplyClonalFilter (Mask& mask, std::vector<RegionalizedData *>& sliced_chip, const deque<float>& ppf, const deque<float>& ssq, const PolyclonalFilterOpts & opts)
{
  clonal_filter filter;
  filter_counts counts;
  make_filter (filter, counts, ppf, ssq, opts);

  unsigned int numRegions = sliced_chip.size();
  for (unsigned int rgn=0; rgn<numRegions; ++rgn)
  {
    RegionalizedData& local_patch     = *sliced_chip[rgn];
    int       numWells  = local_patch.GetNumLiveBeads();
    int       rowOffset = local_patch.region->row;
    int       colOffset = local_patch.region->col;

    for (int well=0; well<numWells; ++well)
    {
      BeadParams& bead  = local_patch.GetParams (well);
      bead_state&  state = *bead.my_state;

      int row = rowOffset + bead.y;
      int col = colOffset + bead.x;
      if(mask.Match(col, row, MaskLib))
        state.clonal_read = filter.is_clonal (state.ppf, state.ssq, opts.mixed_stringency);
      else if(mask.Match(col, row, MaskTF))
        state.clonal_read = true;
    }
  }
}

void ApplySuperMixedFilter (Mask& mask, std::vector<RegionalizedData *>& sliced_chip)
{
  unsigned int numRegions = sliced_chip.size();
  for (unsigned int rgn=0; rgn<numRegions; ++rgn)
  {
    RegionalizedData& local_patch     = *sliced_chip[rgn];
    int       numWells  = local_patch.GetNumLiveBeads();
    int       rowOffset = local_patch.region->row;
    int       colOffset = local_patch.region->col;

    for (int well=0; well<numWells; ++well)
    {
      BeadParams& bead  = local_patch.GetParams (well);
      bead_state&  state = *bead.my_state;

      int row = rowOffset + bead.y;
      int col = colOffset + bead.x;
      if(mask.Match(col, row, MaskLib))
        if (state.ppf < mixed_ppf_cutoff())
          state.clonal_read = true;
        else
          state.clonal_read = false;
      else if(mask.Match(col, row, MaskTF))
        state.clonal_read = true;
    }
  }
}


void GetFilterTrainingSample (deque<int>& row, deque<int>& col, deque<float>& ppf, deque<float>& ssq, deque<float>& nrm, std::vector<RegionalizedData *>& sliced_chip)
{
  unsigned int numRegions = sliced_chip.size();
  for (unsigned int r=0; r<numRegions; ++r)
  {
    int numWells  = sliced_chip[r]->GetNumLiveBeads();
    int rowOffset = sliced_chip[r]->region->row;
    int colOffset = sliced_chip[r]->region->col;
    for (int well=0; well<numWells; ++well)
    {
      BeadParams bead;
      sliced_chip[r]->GetParams (well, &bead);
      const bead_state& state = *bead.my_state;
      if (state.random_samp and state.ppf<mixed_ppf_cutoff() and not state.bad_read)
      {
        row.push_back (rowOffset + bead.y);
        col.push_back (colOffset + bead.x);
        ppf.push_back (state.ppf);
        ssq.push_back (state.ssq);
        nrm.push_back (state.key_norm);
      }
    }
  }
}

void DumpPPFSSQ (const char* results_folder, const deque<int>& row, const deque<int>& col, const deque<float>& ppf, const deque<float>& ssq, const deque<float>& nrm)
{
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

void DumpPPFSSQtoH5 (const char* results_folder, std::vector<RegionalizedData *>& sliced_chip)
{
  vector<int16_t> row;
  vector<int16_t> col;
  vector<float>   ppf;
  vector<float>   ssq;
  vector<float>   nrm;

  unsigned int numRegions = sliced_chip.size();
  for (unsigned int r=0; r<numRegions; ++r)
  {
    int numWells  = sliced_chip[r]->GetNumLiveBeads();
    int rowOffset = sliced_chip[r]->region->row;
    int colOffset = sliced_chip[r]->region->col;
    for (int well=0; well<numWells; ++well)
    {
      BeadParams bead;
      sliced_chip[r]->GetParams (well, &bead);
      const bead_state& state = *bead.my_state;
      if (state.random_samp)
      {
        row.push_back (rowOffset + bead.y);
        col.push_back (colOffset + bead.x);
        ppf.push_back (state.ppf);
        ssq.push_back (state.ssq);
        nrm.push_back (state.key_norm);
      }
    }
  }

  string fname = string (results_folder) + "/BkgModelFilterData.h5";
  SaveH5(fname.c_str(), row, col, ppf, ssq, nrm);
}

static void SaveH5(
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

static hid_t SetupCompression(hsize_t dims[1])
{
    htri_t avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
    assert(avail);

    unsigned int filter_info = 0;
    herr_t status = H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
    assert(status >= 0 and filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED and filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED);

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

