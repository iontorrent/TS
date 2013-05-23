/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef GLOBALWRITER_H
#define GLOBALWRITER_H

#include <stdio.h>
#include <set>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "FlowBuffer.h"
#include "Mask.h"
#include "Region.h"
#include "RegionTracker.h"
#include "BeadParams.h"
#include "BeadTracker.h"

class BkgTrace;

class BkgDataPointers; // forward declaration to avoid including <armadillo> which is in BkgDataPointers.h
class RawWells;
class PinnedInFlow;
class TimeCompression;
class EmphasisClass;
class RegionalizedData;

// things referred to globally
class extern_links
{
  public:

  // mask that indicate where beads/pinned wells are located
  // shared across threads, but pinning is only up to flow 0
  Mask        *bfmask;  // global information on bead type and layout
  int16_t *washout_flow;
  // array to keep track of flows for which empty wells are unpinned
  // shared across threads
  PinnedInFlow *pinnedInFlow;  // not regional, set when creating this object

  // the actual output of this whole process
  RawWells    *rawWells;

  // name out output directory
  std::string dirName;

  BkgDataPointers *mPtrs;

  extern_links(){
    bfmask = NULL;
    pinnedInFlow = NULL;
    rawWells = NULL;
    mPtrs = NULL;
    washout_flow = NULL;
  };

  void FillExternalLinks(Mask *_bfmask, PinnedInFlow *_pinnedInFlow, RawWells *_rawWells, int16_t *_washout_flow){
    rawWells = _rawWells;
    pinnedInFlow = _pinnedInFlow;
    bfmask = _bfmask;
    washout_flow = _washout_flow;
  };
  void SetHdf5Pointer(BkgDataPointers *_my_hdf5)
  {
    mPtrs = _my_hdf5;
  }
  void MakeDirName(char *_results_folder){
    dirName = _results_folder;
    //(char *) malloc (strlen (_results_folder) +1);
    //strncpy (dirName, _results_folder, strlen (_results_folder) +1); //@TODO: really?  why is this a local copy??
  };
  void DeLink()
  {
    //if (dirName!=NULL)
    //  free( dirName);
    bfmask = NULL;
    pinnedInFlow = NULL;
    //dirName = NULL;
    mPtrs = NULL;
  };

  void WriteBeadParameterstoDataCubes (int iFlowBuffer, bool last, Region *region, BeadTracker &my_beads, flow_buffer_info &my_flow, BkgTrace &my_trace);
  void WriteOneBeadToDataCubes(bead_params &p, int flow, int iFlowBuffer, int last, int ibd, Region *region, BkgTrace &my_trace);
  void WriteRegionParametersToDataCubes(RegionalizedData *my_region_data);
  
  ~extern_links(){
    DeLink();
  }
  void DumpTimeCompressionH5 ( int reg, TimeCompression &time_c );
  void DumpRegionFitParamsH5 (int region_ndx, int flow, reg_params &rp  );
  void DumpRegionOffsetH5(int reg, int col, int row );
  void DumpEmptyTraceH5 ( int reg, RegionalizedData &my_region_data );
  void DumpRegionInitH5 (int reg, RegionalizedData &my_region_data);
  void DumpDarknessH5(int reg, reg_params &my_rp);
  void DumpDarkMatterH5(int reg, TimeCompression &time_c, RegionTracker &my_reg_tracker);
  void DumpTimeAndEmphasisByRegionH5 (int reg, TimeCompression &time_c, EmphasisClass &emphasis_data);
  void SendErrorVectorToHDF5 (bead_params *p, error_track &err_t, Region *region, flow_buffer_info &my_flow);
  void SendPredictedToHDF5 (int ibd, float *block_signal_predicted, RegionalizedData &my_region_data);
  void SendCorrectedToHDF5 (int ibd, float *block_signal_corrected, RegionalizedData &my_region_data);
  void SendXtalkToHDF5 (int ibd, float *block_signal_xtalk, RegionalizedData &my_region_data);
  void WriteDebugBeadToRegionH5 ( RegionalizedData *my_region_data );
  void WriteAnswersToWells (int iFlowBuffer, Region *region, RegionTracker *my_regions, BeadTracker &my_beads, flow_buffer_info &my_flow);
    void DumpRegionNucShape( reg_params &rp, int region_ndx, int iBlk, int &i_param);
       void DumpRegionalBuffering( reg_params &rp, int region_ndx, int iBlk, int &i_param);
       void DumpRegionalEnzymatics( reg_params &rp, int region_ndx, int iBlk,int  &i_param);
  void SpecificRegionParamDump ( reg_params &rp, int region_ndx, int iBlk);
 private:

  // Serialization section
  friend class boost::serialization::access;
  template<typename Archive>
    void save(Archive& ar, const unsigned version) const {
      ar &
	bfmask &
	pinnedInFlow &
	// rawWells &
	// mPtrs &    // ??
	dirName;
      fprintf(stdout, "Serialization in GlobalWriter: need to rebuild rawWells, mPtrs\n");
  }
  template<typename Archive>
    void load(Archive& ar, const unsigned version) {
      ar &
	bfmask &
	pinnedInFlow &
	// rawWells & // ??
	// mPtrs &    // ??
	dirName;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};



#endif // GLOBALWRITER_H
