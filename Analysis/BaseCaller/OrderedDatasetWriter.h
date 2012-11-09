/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedDatasetWriter.h
//! @ingroup  BaseCaller
//! @brief    OrderedDatasetWriter. Thread-safe, barcode-friendly SFF/BAM writer with deterministic order

#ifndef ORDEREDDATASETWRITER_H
#define ORDEREDDATASETWRITER_H

#include <string>
#include <vector>
#include <deque>

#include "api/BamWriter.h"

#include "BaseCallerUtils.h"
#include "BarcodeDatasets.h"

using namespace std;
using namespace BamTools;


//! @brief    Content of an SFF read, essentially a c++ version of sff_t
//! @ingroup  BaseCaller

struct SFFEntry {
  string            name;                     //!< Read name
  vector<uint16_t>  flowgram;                 //!< Corrected ionogram where onemer equals 100
  vector<uint8_t>   flow_index;               //!< Flow increment for each base. Integrate for incorporating flow per base.
  vector<char>      bases;                    //!< Base calls
  vector<uint8_t>   quality;                  //!< Quality values in phred scale without any offset.
  int               n_bases;                  //!< Number of called bases
  int32_t           clip_qual_left;           //!< First base of quality-trimmed read. 1-based. 0 means no trimming.
  int32_t           clip_qual_right;          //!< Last base of quality-trimmed read. 1-based. 0 means no trimming.
  int32_t           clip_adapter_left;        //!< First base of adapter-trimmed read. 1-based. 0 means no trimming.
  int32_t           clip_adapter_right;       //!< Last base of adapter-trimmed read. 1-based. 0 means no trimming.
  int               barcode_id;               //!< Barcode index. 1-based. 0 means unclassified/no barcode.
  int               barcode_n_errors;         //!< Number of base mismatches in barcode sequence

  int               clip_adapter_flow;

  //! @brief  Swap content with another SFFEntry object
  //! @param  w The other object
  void  swap(SFFEntry &w);
};


//! @brief    Thread-safe writer class for SFF that guarantees deterministic read order
//! @ingroup  BaseCaller

class OrderedDatasetWriter {
public:
  //! Constructor.
  OrderedDatasetWriter();
  //! Destructor.
  ~OrderedDatasetWriter();

  //! @brief  Open SFF file for writing.
  //! @param  sff_filename    Filename of the SFF file to create.
  //! @param  num_regions     Number of regions to expect.
  //! @param  num_flows       Number of flows.
  //! @param  flow_order      Flow order object, also stores number of flows
  //! @param  key             Key sequence.
  void Open(const string& base_directory, BarcodeDatasets& datasets, int num_regions, const ion::FlowOrder& flow_order, const string& key,
      const string& basecaller_name, const string& basecalller_version, const string& basecaller_command_line,
      const string& production_date, const string& platform_unit);

  //! @brief  Drop off a region-worth of reads for writing. Write opportunistically.
  //! @param  region          Index of the region being dropped off.
  //! @param  region_reads    SFF entries from this region.
  void WriteRegion(int region, deque<SFFEntry> &region_reads);

  //! Update SFF header and close.
  void Close(BarcodeDatasets& datasets, bool quiet = false);

  void PrintStats();

  const vector<uint64_t>&  qv_histogram() const { return qv_histogram_; }

private:

  void PhysicalWriteRegion(int iRegion);


  int                       num_datasets_;          //!< How many files are being generated
  int                       num_barcodes_;          //!< Just in case: how many barcodes are there
  map<int,int>              map_barcode_to_dataset_;//!< Which file should a given barcode saved to?
  map<int,string>           map_barcode_id_to_rg_;


  int                       num_regions_;           //!< Total number of regions to expect
  int                       num_regions_written_;   //!< Number of regions physically written thus far
  vector<bool>              region_ready_;          //!< Which regions are ready for writing?
  vector<deque<SFFEntry> >  region_dropbox_;        //!< Reads for regions that are ready for writing
  pthread_mutex_t           dropbox_write_mutex_;   //!< Mutex controlling access to the dropbox
  pthread_mutex_t           sff_write_mutex_;       //!< Mutex controlling sff writing

  vector<int>               num_reads_;             //!< Number of reads written, per dataset
  vector<string>            bam_filename_;

  map<int,int>              num_reads_per_barcode_;             //!< Number of reads written, per read group
  map<int,int>              num_bases_per_barcode_;             //!< Number of bases written, per read group
  map<int,int>              num_Q20_bases_per_barcode_;         //!< Number of >=Q30 bases written per read group
  map<int,vector<int> >     num_barcode_errors_;                //!< Number of reads with N base errors in barcode
  vector<uint64_t>          qv_histogram_;

  vector<BamWriter *>       bam_writer_;

};



#endif // ORDEREDDATASETWRITER_H



