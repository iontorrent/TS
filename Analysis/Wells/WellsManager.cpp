/* Copyright (C) 2019 Thermo Fisher Scientific. All Rights Reserved */


#include "WellsManager.h"


WellsManager::WellsManager(const vector<string> &wells_file_names, bool verbose)
    : chunk_size_row_(0), chunk_size_col_(0), chunk_size_flow_(0),
      do_norm_(false), compress_multi_taps_(true),
      chip_type_("unknown"), num_flows_(0), verbose_(verbose),
      ion_flow_order(NULL), read_class_map(NULL)
{
  raw_wells_.resize(wells_file_names.size());
  if (verbose) {
    cout << "WellsManager:" << endl;
    cout << "Loading " << wells_file_names.size() << " wells files." << endl;
  }

  for (unsigned int i=0; i<wells_file_names.size(); ++i){

    raw_wells_.at(i).Init("", wells_file_names.at(i).c_str(), 0, 0, 0);
    if (!raw_wells_[i].OpenMetaData()) {
      cerr << "Failed to retrieve metadata from " << wells_file_names[i] << endl;
      exit (EXIT_FAILURE);
    }

    if (i==0){
      raw_wells_[0].GetH5ChunkSize(chunk_size_row_, chunk_size_col_, chunk_size_flow_);
      if (raw_wells_[0].KeyExists("ChipType"))
        raw_wells_[0].GetValue("ChipType", chip_type_);

      num_flows_  = raw_wells_[0].NumFlows();
      flow_order_ = raw_wells_[0].FlowOrder();

      if (verbose) {
        cout << " - ChipType      : " << chip_type_ << endl;
        cout << " - Num Flows     : " << num_flows_ << endl;
        cout << " - H5 Chunk Size : " << chunk_size_row_ << "x"
                                      << chunk_size_col_ << "x"
                                      <<  chunk_size_flow_ << endl << endl;
      }
    }
    // Check consistency of arguments
    else {

      unsigned int chunk_rows, chunk_cols, chunk_flows;
      raw_wells_[i].GetH5ChunkSize(chunk_rows, chunk_cols, chunk_flows);
      if (chunk_rows  != chunk_size_row_ or
          chunk_cols  != chunk_size_col_ or
          chunk_flows != chunk_size_flow_)
        cerr << "WellsManager WARNING: File " << wells_file_names[i]
             << " has a different H5chunk size!" << endl;

      string chip_type = "unknown";
      if (raw_wells_[i].KeyExists("ChipType"))
        raw_wells_[i].GetValue("ChipType", chip_type);
      if (chip_type != chip_type_){
        cerr << "WellsManager ERROR: File " << wells_file_names[i]
             << " has a different chip type : " << chip_type << endl;
        exit (EXIT_FAILURE);
      }

      if (raw_wells_[0].NumFlows() != num_flows_){
        cerr << "WellsManager ERROR: File " << wells_file_names[i]
             << " has a different number of flows!" << endl;
        exit (EXIT_FAILURE);
      }

      string flow_order(raw_wells_[0].FlowOrder());
      if (flow_order != flow_order_){
        cerr << "WellsManager ERROR: File " << wells_file_names[i]
             << " has a different flow order!" << endl;
        exit (EXIT_FAILURE);
      }
    }

  } // -- End loop over wells files

  if((!chip_type_.empty()) && chip_type_[0] == 'P') chip_type_[0] = 'p';

};

// ----------------------------------------------------------------------

void WellsManager::Close()
{
  for (unsigned int i=0; i<raw_wells_.size(); ++i)
    raw_wells_[i].Close();
}

void WellsManager::OpenForIncrementalRead()
{
  for (unsigned int i=0; i<raw_wells_.size(); ++i)
    raw_wells_[i].OpenForIncrementalRead();
}

void WellsManager::LoadChunk(size_t rowStart,  size_t rowHeight,
                             size_t colStart,  size_t colWidth,
                             size_t flowStart, size_t flowDepth)
{
  for (unsigned int i=0; i<raw_wells_.size(); ++i){
    raw_wells_[i].SetChunk(rowStart, rowHeight, colStart, colWidth, flowStart, flowDepth);
    raw_wells_[i].ReadWells();
    if (do_norm_){
      wells_norm_[i].CorrectSignalBias(keys_);
      wells_norm_[i].DoKeyNormalization(keys_);
    }
  }
}

// ----------------------------------------------------------------------

void WellsManager::SetWellsContext(
      ion::FlowOrder const  *flow_order,
      const vector<KeySequence>  &keys,
      ReadClassMap const *rcm,
      const string &norm_method,
      bool compress_multi_taps)
{
  ion_flow_order = flow_order;
  keys_ = keys;
  read_class_map = rcm;
  compress_multi_taps_ = compress_multi_taps;

  if (norm_method == "off"){
    do_norm_ = false;
    wells_norm_.clear();
  }
  else {
    do_norm_ = true;
    wells_norm_.resize(raw_wells_.size());
    for (unsigned int i=0; i<raw_wells_.size(); ++i){
      wells_norm_.at(i).SetFlowOrder(flow_order, norm_method);
      wells_norm_.at(i).SetWells(&raw_wells_.at(i), rcm, i);
    }
  }
}

// ----------------------------------------------------------------------

void WellsManager::GetMeasurements(size_t row, size_t col, vector<float>  & wells_measurements) const
{
  GetMeasurements(row, col, ion_flow_order->num_flows(), wells_measurements);
}

void WellsManager::GetMeasurements(size_t row, size_t col, int num_flows, vector<float>  & wells_measurements) const
{
  // The code for one wells file keeps the legacy implementation
  // and would return any wells signal, also those that only
  // underwent partial signal processing
  if (raw_wells_.size() == 1)
    GetMeasurementsOneWell(row, col, num_flows, wells_measurements);
  // In contrast to that, for multiple wells files, the code only returns
  // signal values if at least one well is unfiltered, fully processed.
  else
    GetMeasurementsWellsAvrg(row, col, num_flows, wells_measurements);

}

// ----------------------------------------------------------------------
// Sanity check. If there are NaNs in this read, print warning

void WellsManager::NaNcheck(const size_t &row, const size_t &col, const int &num_flows, vector<float> & wells_measurements) const
{
  vector<unsigned int> nanflow;
  for (int flow = 0; flow < num_flows; ++flow) {
    if (!isnan(wells_measurements[flow]))
      continue;
    wells_measurements[flow] = 0.0;
    nanflow.push_back(flow);
  }

  if (nanflow.size() > 0) {
    fprintf(stderr, "ERROR: BaseCaller read NaNs from wells file, x=%lu y=%lu flow=%u", col, row, nanflow[0]);
    for (unsigned int flow=1; flow < nanflow.size(); flow++) {
      fprintf(stderr, ",%u", nanflow[flow]);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }
}

// ----------------------------------------------------------------------
// Pretty much the implementation that was in BaseCaller.cpp & PhaseEstimator.cpp before
void WellsManager::GetMeasurementsOneWell(const size_t &row, const size_t &col, const int &num_flows, vector<float> & wells_measurements) const
{
  wells_measurements.resize(num_flows, 0.0);

  // Multi-tap compression. wells file access through .At
  if (compress_multi_taps_) {
    int sig_idx = 0;
    for (int flow = 0; flow < num_flows; ++flow){
      if (flow>0 and ion_flow_order->nuc_at(flow-1)== ion_flow_order->nuc_at(flow)){
        wells_measurements[flow] = 0.0;
        wells_measurements[sig_idx] += raw_wells_[0].At(row,col,flow);
      }
      else {
        sig_idx = flow;
        wells_measurements[flow] = raw_wells_[0].At(row,col,flow);
      }
    }
  }
  else //*/

  for (int flow = 0; flow < num_flows; ++flow)
    wells_measurements[flow] = raw_wells_[0].At(row,col,flow);

  NaNcheck(row, col, num_flows, wells_measurements);
}

// ----------------------------------------------------------------------

void WellsManager::GetMeasurementsWellsAvrg(const size_t &row, const size_t &col, const int &num_flows, vector<float> & wells_measurements) const
{
  wells_measurements.assign(num_flows, 0.0);
  float copy_count_total = 0.0;

  for (unsigned int iwell=0; iwell< raw_wells_.size(); ++iwell){
    // use read class map to determine whether to use the signal from this wells file
    if (not read_class_map->UseWells((int)col, (int)row, iwell))
      continue;

    float copy_count = raw_wells_[iwell].GetCopyCount(row, col);
    if (copy_count <= 0.0)
      continue;
    copy_count_total += copy_count;

    // Differentiate between the case where WellsNormalization
    // is enabled  - we load key normalized values
    // is disabled - we load values scaled by copy count
    for (int flow = 0; flow < num_flows; ++flow){
      if (do_norm_){
        // abuse possible -therefore the at function
        wells_measurements.at(flow) += copy_count * raw_wells_[iwell].At(row,col,flow);
      }
      else{
        wells_measurements.at(flow) += raw_wells_[iwell].At(row,col,flow);
      }
    }
  }
  NaNcheck(row, col, num_flows, wells_measurements);
}
