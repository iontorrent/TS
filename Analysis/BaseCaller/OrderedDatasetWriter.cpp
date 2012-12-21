/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedDatasetWriter.cpp
//! @ingroup  BaseCaller
//! @brief    OrderedDatasetWriter. Thread-safe, barcode-friendly BAM writer with deterministic order

#include "OrderedDatasetWriter.h"

#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#include "api/SamHeader.h"
#include "api/BamAlignment.h"

using namespace std;

class ThousandsSeparator : public numpunct<char> {
protected:
    string do_grouping() const { return "\03"; }
};


OrderedDatasetWriter::OrderedDatasetWriter()
{
  num_regions_ = 0;
  num_regions_written_ = 0;
  pthread_mutex_init(&dropbox_mutex_, NULL);
  pthread_mutex_init(&write_mutex_, NULL);
  pthread_mutex_init(&delete_mutex_, NULL);
}

OrderedDatasetWriter::~OrderedDatasetWriter()
{
  pthread_mutex_destroy(&dropbox_mutex_);
  pthread_mutex_destroy(&write_mutex_);
  pthread_mutex_destroy(&delete_mutex_);
}


void OrderedDatasetWriter::Open(const string& base_directory, BarcodeDatasets& datasets, int num_regions, const ion::FlowOrder& flow_order, const string& key,
    const string& basecaller_name, const string& basecalller_version, const string& basecaller_command_line,
    const string& production_date, const string& platform_unit, bool save_filtered_reads)
{
  num_regions_ = num_regions;
  num_regions_written_ = 0;
  region_ready_.assign(num_regions_+1,false);
  region_dropbox_.clear();
  region_dropbox_.resize(num_regions_);

  qv_histogram_.assign(50,0);

  num_datasets_ = datasets.num_datasets();
  num_barcodes_ = datasets.num_barcodes();
  num_read_groups_ = datasets.num_read_groups();
  num_reads_.resize(num_datasets_,0);
  bam_filename_.resize(num_datasets_);

  save_filtered_reads_ = save_filtered_reads;

  read_group_name_.resize(num_read_groups_);
  read_group_dataset_.assign(num_read_groups_, -1);
  read_group_num_Q20_bases_.assign(num_read_groups_,0);
  read_group_num_barcode_errors_.resize(num_read_groups_);

  for (int rg = 0; rg < num_read_groups_; ++rg) {
    read_group_name_[rg] = datasets.read_group_name(rg);
    read_group_num_barcode_errors_[rg].assign(3,0);
  }

  // New filtering and trimming accounting (per read group)

  read_group_stats_.resize(num_read_groups_);

  bam_writer_.resize(num_datasets_, NULL);

  for (int ds = 0; ds < num_datasets_; ++ds) {

    // Set up BAM header

    bam_filename_[ds] = base_directory + "/" + datasets.dataset(ds)["basecaller_bam"].asString();

    SamHeader sam_header;
    sam_header.Version = "1.4";
    sam_header.SortOrder = "unsorted";

    SamProgram sam_program("bc");
    sam_program.Name = basecaller_name;
    sam_program.Version = basecalller_version;
    sam_program.CommandLine = basecaller_command_line;
    sam_header.Programs.Add(sam_program);

    for (Json::Value::iterator rg = datasets.dataset(ds)["read_groups"].begin(); rg != datasets.dataset(ds)["read_groups"].end(); ++rg) {
      string read_group_name = (*rg).asString();
      Json::Value& read_group_json = datasets.read_groups()[read_group_name];

      read_group_dataset_[datasets.read_group_name_to_id(read_group_name)] = ds;

      SamReadGroup read_group (read_group_name);
      read_group.FlowOrder = flow_order.full_nucs();

      read_group.KeySequence          = key;
      read_group.KeySequence          += read_group_json.get("barcode_sequence","").asString();
      read_group.KeySequence          += read_group_json.get("barcode_adapter","").asString();

      read_group.ProductionDate       = production_date;
      read_group.Sample               = read_group_json.get("sample","").asString();
      read_group.Library              = read_group_json.get("library","").asString();
      read_group.Description          = read_group_json.get("description","").asString();
      read_group.PlatformUnit         = read_group_json.get("platform_unit","").asString();
      read_group.SequencingCenter     = datasets.json().get("sequencing_center","").asString();
      read_group.SequencingTechnology = "IONTORRENT";

      sam_header.ReadGroups.Add(read_group);
    }

    // Open Bam for writing

    RefVector empty_reference_vector;
    bam_writer_[ds] = new BamWriter();
    bam_writer_[ds]->SetCompressionMode(BamWriter::Compressed);
    //bam_writer_[ds]->SetCompressionMode(BamWriter::Uncompressed);
    bam_writer_[ds]->Open(bam_filename_[ds], sam_header, empty_reference_vector);
  }

}




void OrderedDatasetWriter::Close(BarcodeDatasets& datasets, const string& dataset_nickname)
{

  for (;num_regions_written_ < num_regions_; num_regions_written_++) {
    PhysicalWriteRegion(num_regions_written_);
    region_dropbox_[num_regions_written_].clear();
  }

  for (int ds = 0; ds < num_datasets_; ++ds) {
    if (bam_writer_[ds]) {
      bam_writer_[ds]->Close();
      delete bam_writer_[ds];
    }

    datasets.dataset(ds)["read_count"] = num_reads_[ds];
    for (Json::Value::iterator rg = datasets.dataset(ds)["read_groups"].begin(); rg != datasets.dataset(ds)["read_groups"].end(); ++rg) {
      string read_group_name = (*rg).asString();
      Json::Value& read_group_json = datasets.read_groups()[read_group_name];
      int rg_index = datasets.read_group_name_to_id(read_group_name);
      read_group_json["read_count"]  = (Json::UInt64)read_group_stats_[rg_index].num_reads_final_;
      read_group_json["total_bases"] = (Json::UInt64)read_group_stats_[rg_index].num_bases_final_;
      read_group_json["Q20_bases"]   = (Json::UInt64)read_group_num_Q20_bases_[rg_index];
      read_group_json["barcode_errors_hist"][0] = (Json::UInt64)read_group_num_barcode_errors_[rg_index][0];
      read_group_json["barcode_errors_hist"][1] = (Json::UInt64)read_group_num_barcode_errors_[rg_index][1];
      read_group_json["barcode_errors_hist"][2] = (Json::UInt64)read_group_num_barcode_errors_[rg_index][2];
    }

    if (!dataset_nickname.empty())
      printf("%s: Generated %s with %d reads\n", dataset_nickname.c_str(), bam_filename_[ds].c_str(), num_reads_[ds]);
  }

  for (int rg = 0; rg < num_read_groups_; ++rg)
    combined_stats_.MergeFrom(read_group_stats_[rg]);
  if (!dataset_nickname.empty())
    combined_stats_.PrettyPrint(dataset_nickname);
}


void OrderedDatasetWriter::WriteRegion(int region, deque<ProcessedRead> &region_reads)
{
  // Deposit results in the dropbox
  pthread_mutex_lock(&dropbox_mutex_);
  region_dropbox_[region].swap(region_reads);
  region_ready_[region] = true;
  pthread_mutex_unlock(&dropbox_mutex_);

  // Attempt writing duty
  if (pthread_mutex_trylock(&write_mutex_))
    return;
  int num_regions_deleted = num_regions_written_;
  while (true) {
    pthread_mutex_lock(&dropbox_mutex_);
    bool cannot_write = !region_ready_[num_regions_written_];
    pthread_mutex_unlock(&dropbox_mutex_);
    if (cannot_write)
      break;
    PhysicalWriteRegion(num_regions_written_);
    num_regions_written_++;
  }
  pthread_mutex_unlock(&write_mutex_);

  // Destroy written reads, outside of mutex block
  if (pthread_mutex_trylock(&delete_mutex_))
    return;
  while (num_regions_deleted < num_regions_written_)
    region_dropbox_[num_regions_deleted++].clear();
  pthread_mutex_unlock(&delete_mutex_);
}


void OrderedDatasetWriter::PhysicalWriteRegion(int region)
{
  for (deque<ProcessedRead>::iterator entry = region_dropbox_[region].begin(); entry != region_dropbox_[region].end(); entry++) {

    // Step 1: Read filtering and trimming accounting

    read_group_stats_[entry->read_group_index].AddRead(entry->filter);

    // Step 2: Should this read be saved?

    if (entry->filter.is_filtered and not save_filtered_reads_)
      continue;

    int target_file_idx = read_group_dataset_[entry->read_group_index];
    if (target_file_idx < 0) // Read group not assigned to a dataset?
      continue;

    // Step 3: Other misc stats

    num_reads_[target_file_idx]++;

    for (int base = 0; base < (int)entry->bam.Qualities.length(); ++base) {
      int quality = entry->bam.Qualities[base] - 33;
      if (quality >= 20)
        read_group_num_Q20_bases_[entry->read_group_index]++;
      qv_histogram_[min(quality,49)]++;
    }

    int n_errors = max(0,min(2,entry->barcode_n_errors));
    read_group_num_barcode_errors_[entry->read_group_index][n_errors]++;

    // Actually write out the read

    entry->bam.AddTag("RG","Z", read_group_name_[entry->read_group_index]);
    entry->bam.AddTag("PG","Z", string("bc"));
    bam_writer_[target_file_idx]->SaveAlignment(entry->bam);
  }
}



