/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedRegionSFFWriter.h
//! @ingroup  BaseCaller
//! @brief    OrderedRegionSFFWriter. Thread-safe SFF writer with deterministic order

#ifndef ORDEREDREGIONSFFWRITER_H
#define ORDEREDREGIONSFFWRITER_H

#include <string>
#include <vector>
#include <deque>

#include "BaseCallerUtils.h"
#include "file-io/sff_definitions.h"

using namespace std;

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

  //! @brief  Swap content with another SFFEntry object
  //! @param  w The other object
  void  swap(SFFEntry &w);
};


//! @brief    Thread-safe writer class for SFF that guarantees deterministic read order
//! @ingroup  BaseCaller

class OrderedRegionSFFWriter {
public:
  //! Constructor.
  OrderedRegionSFFWriter();
  //! Destructor.
  ~OrderedRegionSFFWriter();

  //! @brief  Open SFF file for writing.
  //! @param  sff_filename    Filename of the SFF file to create.
  //! @param  num_regions     Number of regions to expect.
  //! @param  num_flows       Number of flows.
  //! @param  flow_order      Flow order object, also stores number of flows
  //! @param  key             Key sequence.
  void Open(const string& sff_filename, int num_regions, const ion::FlowOrder& flow_order, const string& key);

  //! @brief  Drop off a region-worth of reads for writing. Write opportunistically.
  //! @param  region          Index of the region being dropped off.
  //! @param  region_reads    SFF entries from this region.
  void WriteRegion(int region, deque<SFFEntry> &region_reads);

  //! Update SFF header and close.
  void Close();

  //! Return number of reads physically written.
  int num_reads() { return num_reads_; }

private:

  void PhysicalWriteRegion(int iRegion);

  int                       num_reads_;             //!< Number of reads physically written thus far
  int                       num_regions_;           //!< Total number of regions to expect
  int                       num_regions_written_;   //!< Number of regions physically written thus far
  vector<bool>              region_ready_;          //!< Which regions are ready for writing?
  vector<deque<SFFEntry> >  region_dropbox_;        //!< Reads for regions that are ready for writing
  pthread_mutex_t           dropbox_write_mutex_;   //!< Mutex controlling access to the dropbox
  pthread_mutex_t           sff_write_mutex_;       //!< Mutex controlling sff writing
  sff_file_t                *sff_file_;             //!< "Native" sff file struct
  sff_t                     *sff_;                  //!< "Native" sff read struct
};



#endif // ORDEREDREGIONSFFWRITER_H
