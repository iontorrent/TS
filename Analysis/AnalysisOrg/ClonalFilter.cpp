/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ClonalFilter.h"
#include "mixed.h"

using namespace std;
void ApplyClonalFilter(Mask& mask, std::vector<RegionalizedData *>& sliced_chip,  const deque<float>& ppf, const deque<float>& ssq);
void UpdateMask(Mask& mask, std::vector<RegionalizedData *>& sliced_chip);
void GetFilterTrainingSample(deque<int>& row, deque<int>& col, deque<float>& ppf, deque<float>& ssq, deque<float>& nrm, std::vector<RegionalizedData *>& sliced_chip);
void DumpPPFSSQ (const char* results_folder, const deque<int>& row, const deque<int>& col, const deque<float>& ppf, const deque<float>& ssq, const deque<float>& nrm);
//void AttemptClonalFilter(Mask& mask, const char* results_folder, RegionalizedData *sliced_chip[], int numRegions);

void AttemptClonalFilter(Mask& mask, const char* results_folder, std::vector<RegionalizedData *>& sliced_chip);

void ApplyClonalFilter (Mask& mask, const char* results_folder, std::vector<RegionalizedData *>& sliced_chip, bool doClonalFilter, int flow)
{
  int applyFlow = ceil(1.0*mixed_last_flow() / NUMFB) * NUMFB - 1;
  if (flow == applyFlow and doClonalFilter)
  {
    // Never give up just because filter failed.
    try{
      AttemptClonalFilter(mask,results_folder,sliced_chip);
    }catch(exception& e){
        cerr << "NOTE: clonal filter failed."
             << e.what()
             << endl;
    }catch(...){
        cerr << "NOTE: clonal filter failed." << endl;
    }

  }
}

void AttemptClonalFilter(Mask& mask, const char* results_folder, std::vector<RegionalizedData *>& sliced_chip)
{
      deque<int>   row;
    deque<int>   col;
    deque<float> ppf;
    deque<float> ssq;
    deque<float> nrm;
    GetFilterTrainingSample (row, col, ppf, ssq, nrm, sliced_chip);
    DumpPPFSSQ(results_folder, row, col, ppf, ssq, nrm);
    ApplyClonalFilter (mask, sliced_chip, ppf, ssq);
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
      bead_params& bead  = local_patch.GetParams (well);
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

void ApplyClonalFilter (Mask& mask, std::vector<RegionalizedData *>& sliced_chip, const deque<float>& ppf, const deque<float>& ssq)
{
  clonal_filter filter;
  filter_counts counts;
  make_filter (filter, counts, ppf, ssq, false); // I dislike verbosity on trunk

  unsigned int numRegions = sliced_chip.size();
  for (unsigned int rgn=0; rgn<numRegions; ++rgn)
  {
    RegionalizedData& local_patch     = *sliced_chip[rgn];
    int       numWells  = local_patch.GetNumLiveBeads();
    int       rowOffset = local_patch.region->row;
    int       colOffset = local_patch.region->col;

    for (int well=0; well<numWells; ++well)
    {
      bead_params& bead  = local_patch.GetParams (well);
      bead_state&  state = *bead.my_state;

      int row = rowOffset + bead.y;
      int col = colOffset + bead.x;
      if(mask.Match(col, row, MaskLib))
        state.clonal_read = filter.is_clonal (state.ppf, state.ssq);
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
      bead_params bead;
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
