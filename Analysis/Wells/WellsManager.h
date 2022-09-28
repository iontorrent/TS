/* Copyright (C) 2019 Thermo Fisher Scientific. All Rights Reserved */
#ifndef WELLSMANAGER_H
#define WELLSMANAGER_H

#include <vector>
#include "RawWells.h"
#include "ReadClassMap.h"
#include "WellsNormalization.h"

using namespace std;

// ============================================================================
// Class WellsManager
// Manage Access to multiple wells files from the same chip

class WellsManager {

  vector<RawWells>           raw_wells_;
  vector<WellsNormalization> wells_norm_;
  vector<KeySequence>        keys_;

  unsigned int               chunk_size_row_;
  unsigned int               chunk_size_col_;
  unsigned int               chunk_size_flow_;

  bool                       do_norm_;
  bool                       compress_multi_taps_;
  string                     chip_type_;
  string                     flow_order_;
  unsigned int               num_flows_;
  bool                       verbose_;
  string                     norm_method_;
  //unsigned int               num_verbose_;

public:

  ion::FlowOrder const      *ion_flow_order;    //!< Pointer to flow order object
  ReadClassMap   const      *read_class_map;

  WellsManager(const vector<string> & wells_file_names, bool verbose);
  WellsManager( WellsManager *rw);

  int NumWells() const
  { return raw_wells_.size(); };

  void OpenForIncrementalRead();


  void Close();

  void LoadChunk(size_t rowStart,  size_t rowHeight,
                 size_t colStart,  size_t colWidth,
                 size_t flowStart, size_t flowDepth,
				 pthread_mutex_t *mutex=NULL);

  void SetWellsContext(
        ion::FlowOrder const  *flow_order,
        const vector<KeySequence>  &keys,
        ReadClassMap const *rcm,
        const string &norm_method,
        bool  compress_multi_taps);

  void NaNcheck(const size_t &row, const size_t &col, const int &num_flows, vector<float> & wells_measurements) const;

  // Access functions

  void GetMeasurements(size_t row, size_t col, vector<float> & wells_measurements) const;
  void GetMeasurements(size_t row, size_t col, int num_flows, vector<float> & wells_measurements) const;

  void GetMeasurementsOneWell(const size_t &row, const size_t &col, const int &num_flows, vector<float> & wells_measurements) const;
  void GetMeasurementsWellsAvrg(const size_t &row, const size_t &col, const int &num_flows, vector<float> & wells_measurements) const;

  const char * FlowOrder()       const { return flow_order_.c_str(); }
  unsigned int NumFlows()        const { return num_flows_; }
  string       ChipType()        const { return chip_type_; }
  const vector<KeySequence> Keys() const { return  keys_; }
  RawWells *   Wells0()          { return &raw_wells_.at(0); }

  unsigned int H5ChunkSizeRow()  const { return chunk_size_row_; }
  unsigned int H5ChunkSizeCol()  const { return chunk_size_col_; }
  unsigned int H5ChunkSizeFlow() const { return chunk_size_flow_; }

  // Functions to get the averaged signal values

};

#endif // WELLSMANAGER_H
