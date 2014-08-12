/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SlicedPrequel.h"
#include "H5File.h"

using namespace std;

void WriteBeadFindForBkgModel (std::string& h5file,
                               std::vector<float>& smooth_t0_est,
                               std::vector<float>& tauB, std::vector<float>& tauE,
                               std::vector<RegionTiming> &region_timing);

void LoadBeadFindForBkgModel (std::string& h5file,
                              std::vector<float>& smooth_t0_est,
                              std::vector<float>& tauB, std::vector<float>& tauE,
                              std::vector<RegionTiming> &region_timing);

SlicedPrequel::SlicedPrequel()
{
  num_regions = 0;
}

void SlicedPrequel::FileLocations(std::string &analysisLocation)
{
  bfFile  = analysisLocation + "beadfind.h5";
  bfMaskFile  = analysisLocation + "bfmask.bin";
  bfStatsFile = analysisLocation + "bfmask.stats";
}

void SlicedPrequel::Allocate(int _num_regions)
{
  num_regions = _num_regions;
  region_list.resize(num_regions);
  region_timing.resize(num_regions);
}

SlicedPrequel::~SlicedPrequel()
{
  region_list.clear();
  region_timing.clear();
  num_regions = 0;
}

void SlicedPrequel::SetRegions(int _num_regions, int rows, int cols,
			  int regionXsize, int regionYsize)
{
    Allocate(_num_regions);
    RegionHelper::SetUpRegions(region_list, rows, cols, regionXsize, regionYsize);
}

void WriteBeadFindForBkgModel (std::string& h5file, std::vector<float>& smooth_t0_est, std::vector<float>& tauB, std::vector<float>& tauE,
			       std::vector<RegionTiming> &region_timing)
{
  unsigned int totalRegions = region_timing.size();
  std::vector<float> t_mid_nuc (totalRegions);
  std::vector<float> t0_frame (totalRegions);
  std::vector<float> t_sigma (totalRegions);
  for (unsigned int i=0; i<totalRegions; i++)
  {
    t_mid_nuc[i] = region_timing[i].t_mid_nuc;
    t_sigma[i] = region_timing[i].t_sigma;
    t0_frame[i] = region_timing[i].t0_frame;
  }

  string h5_t0_est = h5file + ":/beadfind/t0_est";
  H5File::WriteVector (h5_t0_est, smooth_t0_est, true);

  string h5_tauB = h5file + ":/beadfind/tauB";
  H5File::WriteVector (h5_tauB, tauB, false);

  string h5_t_mid_nuc = h5file + ":/beadfind/t_mid_nuc";
  H5File::WriteVector (h5_t_mid_nuc, t_mid_nuc, false);

  string h5_t0_frame = h5file + ":/beadfind/t0_frame";
  H5File::WriteVector (h5_t0_frame, t0_frame, false);

  string h5_t_sigma = h5file + ":/beadfind/t_sigma";
  H5File::WriteVector (h5_t_sigma, t_sigma, false);

  string h5_tauE = h5file + ":/beadfind/tauE";
  H5File::WriteVector (h5_tauE, tauE, false);

}

void LoadBeadFindForBkgModel (std::string &h5file, std::vector<float>& smooth_t0_est, std::vector<float>& tauB, std::vector<float>& tauE,
			      std::vector<RegionTiming> &region_timing)
{
  string h5_t_mid_nuc = h5file + ":/beadfind/t_mid_nuc";
  std::vector<float> t_mid_nuc;
  H5File::ReadVector (h5_t_mid_nuc, t_mid_nuc);

  string h5_t_sigma = h5file + ":/beadfind/t_sigma";
  std::vector<float> t_sigma;
  H5File::ReadVector (h5_t_sigma, t_sigma);

  std::vector<float> t0_frame;
  string h5_t0_frame = h5file + ":/beadfind/t0_frame";
  H5File::ReadVector (h5_t0_frame, t0_frame);

  unsigned int totalRegions = region_timing.size();
  assert (t_mid_nuc.size() == totalRegions);

  for (unsigned int i=0; i<totalRegions; i++)
  {
    region_timing[i].t_mid_nuc = t_mid_nuc[i];
    region_timing[i].t_sigma = t_sigma[i];
    region_timing[i].t0_frame = t0_frame[i];
  }

  string h5_t0_est = h5file + ":/beadfind/t0_est";
  H5File::ReadVector (h5_t0_est, smooth_t0_est);

  string h5_tauB = h5file + ":/beadfind/tauB";
  H5File::ReadVector (h5_tauB, tauB);

  string h5_tauE = h5file + ":/beadfind/tauE";
  H5File::ReadVector (h5_tauE, tauE);

}

void SlicedPrequel::WriteBeadFindForSignalProcessing()
{
    WriteBeadFindForBkgModel (bfFile,smooth_t0_est, tauB, tauE, region_timing);
}

void SlicedPrequel::LoadBeadFindForSignalProcessing(bool load)
{
  if (load){
    // load the mask and other data required for Background Modeling
    cout << "Loading " + bfFile + " for Background Model\n";
    LoadBeadFindForBkgModel (bfFile, smooth_t0_est, tauB, tauE, region_timing);
  }
}

void SlicedPrequel::RestrictRegions(std::vector<int>& region_list)
{
  if (region_list.empty())  // do nothing
    return;

  std::vector<unsigned int>regions_to_use;
  for(size_t i = 0; i < region_list.size(); ++i)
  {
	  if(region_list[i] >= 0 && region_list[i] < num_regions)
	  {
		  regions_to_use.push_back((unsigned int)region_list[i]);
	  }
  }

  Elide (regions_to_use);
}

void SlicedPrequel::Elide(std::vector<unsigned int>& regions_to_use)
{
  size_t newsize = regions_to_use.size();
  // if (newsize == region_list.size())
  //   return;

  // code assumes sorted region indices
  sort(regions_to_use.begin(), regions_to_use.end());

  std::vector<Region> tmp_region_list(newsize);
  std::vector<RegionTiming> tmp_region_timing(newsize);

  size_t nbeads = 0;
  for (size_t i=0; i < newsize; ++i){
    int region_to_include = regions_to_use[i];
    tmp_region_list[i] = region_list[region_to_include];
    tmp_region_timing[i] = region_timing[region_to_include];
    nbeads += tmp_region_list[i].h * tmp_region_list[i].w;
  }

  std::vector<float> tmp_smooth_t0_est(nbeads, 0);
  std::vector<float> tmp_tauB(nbeads, 0);
  std::vector<float> tmp_tauE(nbeads, 0);
  int ix_src = 0;
  int ix_dest = 0;
  int ix_region_to_use = 0;
  for (size_t i=0; i < region_list.size(); i++) {
    int rsize = region_list[i].h * region_list[i].w;
    if ( i == regions_to_use[ix_region_to_use] ) {
      ix_region_to_use++;
      fprintf (stdout, "Restricting to region %d\n", (int)i);
      for (int ibd = 0; ibd < rsize; ibd++) {
	tmp_smooth_t0_est[ix_dest] = smooth_t0_est[ix_src];
	tmp_tauB[ix_dest] = tauB[ix_src];
	tmp_tauE[ix_dest] = tauE[ix_src];
	ix_dest++;
	ix_src++;
      }
    }
    else {
      ix_src += rsize;
    }
  }
  region_list.assign(tmp_region_list.begin(), tmp_region_list.end());
  region_timing.assign(tmp_region_timing.begin(), tmp_region_timing.end());
  smooth_t0_est.assign(tmp_smooth_t0_est.begin(), tmp_smooth_t0_est.end());
  tauB.assign(tmp_tauB.begin(), tmp_tauB.end());
  tauB.assign(tmp_tauE.begin(),tmp_tauE.end());
  num_regions = newsize;
}
