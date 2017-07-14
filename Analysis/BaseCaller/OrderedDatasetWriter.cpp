/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedDatasetWriter.cpp
//! @ingroup  BaseCaller
//! @brief    OrderedDatasetWriter. Thread-safe, barcode-friendly BAM writer with deterministic order

#include "OrderedDatasetWriter.h"

#include <stddef.h>
#include <algorithm>
#include <sstream>
#include <stdio.h>

#include "BarcodeDatasets.h"

#include "api/BamAlignment.h"

using namespace std;


// ------------------------------------------------------------------

OrderedDatasetWriter::OrderedDatasetWriter()
{
  num_regions_           = 0;
  num_regions_written_   = 0;
  num_barcodes_          = 0;
  num_read_groups_       = 0;
  num_datasets_          = 0;
  save_filtered_reads_   = false;
  compress_bam_          = true;
  num_bamwriter_threads_ = 0;
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

// ----------------------------------------------------------------------------

void OrderedDatasetWriter::Open(const string& base_directory, BarcodeDatasets& datasets, int read_class_idx,
     int num_regions, const ion::FlowOrder& flow_order, const string& key, const vector<string> & bead_adapters,
     int num_bamwriter_threads, const Json::Value & basecaller_json, vector<string>& comments,
     MolecularTagTrimmer& tag_trimmer, bool trim_barcodes, bool compress_bam)
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
  compress_bam_ = compress_bam;

  // A negative read group index indicates untrimmed/unfiltered bam files (w. library key) and we save all reads
  if (read_class_idx < 0) {
    save_filtered_reads_ = true;
    read_class_idx = 0;
  }
  else
    save_filtered_reads_ = false;

  read_group_name_.resize(num_read_groups_);
  read_group_dataset_.assign(num_read_groups_, -1);
  read_group_num_Q20_bases_.assign(num_read_groups_,0);
  read_group_barcode_filt_zero_err_.assign(num_read_groups_, 0);
  read_group_barcode_adapter_rejected_.assign(num_read_groups_, 0);
  read_group_num_barcode_errors_.resize(num_read_groups_);
  read_group_barcode_distance_hist_.resize(num_read_groups_);
  read_group_barcode_bias_.resize(num_read_groups_);

  for (int rg = 0; rg < num_read_groups_; ++rg) {
    read_group_name_[rg] = datasets.read_group_name(rg);
    read_group_num_barcode_errors_[rg].assign(3,0);
    read_group_barcode_bias_[rg].assign(datasets.GetBCmaxFlows(),0.0);
    read_group_barcode_distance_hist_[rg].assign(5,0);

  }

  // New filtering and trimming accounting (per read group)

  read_group_stats_.resize(num_read_groups_);
  for (int rg=0; rg<num_read_groups_; rg++)
	read_group_stats_[rg].SetBeadAdapters(bead_adapters);
  combined_stats_.SetBeadAdapters(bead_adapters);

  bam_writer_.resize(num_datasets_, NULL);
  sam_header_.resize(num_datasets_);
  num_bamwriter_threads_ = num_bamwriter_threads;

  for (int ds = 0; ds < num_datasets_; ++ds) {

    // Set up BAM header

    bam_filename_[ds] = base_directory + "/" + datasets.dataset(ds)["basecaller_bam"].asString();

    SamHeader& sam_header = sam_header_[ds];
    sam_header.Version = "1.4";
    sam_header.SortOrder = "unsorted";

    SamProgram sam_program("bc");
    sam_program.Name        = "BaseCaller";
    sam_program.Version     = basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["git_hash"].asString();
    sam_program.CommandLine = basecaller_json["BaseCaller"]["command_line"].asString();
    sam_header.Programs.Add(sam_program);

    for (Json::Value::iterator rg = datasets.dataset(ds)["read_groups"].begin(); rg != datasets.dataset(ds)["read_groups"].end(); ++rg) {
      string read_group_name = (*rg).asString();
      Json::Value& read_group_json = datasets.read_groups()[read_group_name];

      read_group_dataset_[datasets.read_group_name_to_id(read_group_name)] = ds;

      SamReadGroup read_group (read_group_name);

      read_group.FlowOrder            = flow_order.full_nucs();
      read_group.KeySequence          = key;
      if (trim_barcodes){ // We only add the barcode info to the key sequence if we hard clipped it
        read_group.KeySequence          += read_group_json.get("barcode_sequence","").asString();
        read_group.KeySequence          += read_group_json.get("barcode_adapter","").asString();
      }

      read_group.ProductionDate       = basecaller_json["BaseCaller"]["start_time"].asString();
      read_group.Sample               = read_group_json.get("sample","").asString();
      read_group.Library              = read_group_json.get("library","").asString();
      read_group.Description          = read_group_json.get("description","").asString();
      read_group.PlatformUnit         = read_group_json.get("platform_unit","").asString();
      read_group.SequencingCenter     = datasets.json().get("sequencing_center","").asString();
      read_group.SequencingTechnology = "IONTORRENT";

      // Add custom tags: Structure of tags per read group XXX
      if (datasets.IsLibraryDataset()) {
        MolTag my_tags = tag_trimmer.GetReadGroupTags(read_group_name);
        AddCustomReadGroupTag(read_group, "zt", my_tags.prefix_mol_tag);
        AddCustomReadGroupTag(read_group, "yt", my_tags.suffix_mol_tag);
      }

      sam_header.ReadGroups.Add(read_group);
    }

    for(size_t i = 0; i < comments.size(); ++i)
      sam_header.Comments.push_back(comments[i]);
  }

}

// ----------------------------------------------------------------------------

void  OrderedDatasetWriter::AddCustomReadGroupTag (SamReadGroup & read_group, const string& tag_name, const string& tag_body)
{
  // Nothing to do if body is empty
  if (tag_name.empty() or tag_body.empty())
    return;

  CustomHeaderTag my_custom_tag;
  my_custom_tag.TagName  = tag_name;
  my_custom_tag.TagValue = tag_body;

  read_group.CustomTags.push_back(my_custom_tag);
}

// ----------------------------------------------------------------------------

void OrderedDatasetWriter::Close(BarcodeDatasets& datasets, const string& dataset_nickname)
{

  for (;num_regions_written_ < num_regions_; num_regions_written_++) {
    PhysicalWriteRegion(num_regions_written_);
    region_dropbox_[num_regions_written_].clear();
  }

  for (int ds = 0; ds < num_datasets_; ++ds) {
    if (bam_writer_[ds]) {
      if (!dataset_nickname.empty())
        printf("%s: Generated %s with %d reads\n", dataset_nickname.c_str(), bam_filename_[ds].c_str(), num_reads_[ds]);
      bam_writer_[ds]->Close();
      delete bam_writer_[ds];
    }
    else {
      if (!dataset_nickname.empty())
    	printf("%s: No reads for %s\n", dataset_nickname.c_str(), bam_filename_[ds].c_str());
    }

    datasets.dataset(ds)["read_count"] = num_reads_[ds];
    for (Json::Value::iterator rg = datasets.dataset(ds)["read_groups"].begin(); rg != datasets.dataset(ds)["read_groups"].end(); ++rg) {
      string read_group_name = (*rg).asString();
      Json::Value& read_group_json = datasets.read_groups()[read_group_name];
      int rg_index = datasets.read_group_name_to_id(read_group_name);
      read_group_json["read_count"]  = (Json::UInt64)read_group_stats_.at(rg_index).num_reads_final_;
      read_group_json["total_bases"] = (Json::UInt64)read_group_stats_.at(rg_index).num_bases_final_;
      read_group_json["Q20_bases"]   = (Json::UInt64)read_group_num_Q20_bases_[rg_index];

      // Log barcode statistics only for barcode read groups
      if (read_group_json.isMember("barcode_sequence")) {
        read_group_json["barcode_match_filtered"] = (Json::UInt64)read_group_barcode_filt_zero_err_[rg_index];
        read_group_json["barcode_adapter_filtered"] = (Json::UInt64)read_group_barcode_adapter_rejected_[rg_index];

        for (unsigned int iflow=0; iflow < read_group_barcode_bias_[rg_index].size(); iflow++) {
          Json::Value av_bias_json(read_group_barcode_bias_[rg_index].at(iflow) / max(read_group_stats_.at(rg_index).num_reads_final_,(int64_t)1));
    	  read_group_json["barcode_bias"][iflow] = av_bias_json;
        }
        for (unsigned int ibin=0; ibin < read_group_barcode_distance_hist_[rg_index].size(); ibin++)
          read_group_json["barcode_distance_hist"][ibin] = (Json::UInt64)read_group_barcode_distance_hist_[rg_index].at(ibin);
        for (unsigned int ierr=0; ierr < read_group_num_barcode_errors_[rg_index].size(); ierr++)
          read_group_json["barcode_errors_hist"][ierr] = (Json::UInt64)read_group_num_barcode_errors_[rg_index][ierr];
      }
    }
  }

  for (int rg = 0; rg < num_read_groups_; ++rg)
    combined_stats_.MergeFrom(read_group_stats_.at(rg));
  combined_stats_.ComputeAverages();
  if (!dataset_nickname.empty())
    combined_stats_.PrettyPrint(dataset_nickname);
}

// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------

void OrderedDatasetWriter::PhysicalWriteRegion(int region)
{
  for (deque<ProcessedRead>::iterator entry = region_dropbox_[region].begin(); entry != region_dropbox_[region].end(); ++entry) {

    // Step 1: Read filtering and trimming accounting

    read_group_stats_.at(entry->read_group_index).AddRead(entry->filter);

    // Step 2: Should this read be saved?

    if (entry->filter.is_filtered and not save_filtered_reads_)
      continue;

    int target_file_idx = read_group_dataset_.at(entry->read_group_index);
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

    // Number of barcode base errors
    int n_errors = max(0,min(2,entry->barcode_n_errors));
    read_group_num_barcode_errors_[entry->read_group_index].at(n_errors)++;
    // Transfer barcode bias vector
    if (read_group_barcode_bias_[entry->read_group_index].size() < entry->barcode_bias.size())
      read_group_barcode_bias_[entry->read_group_index].resize(entry->barcode_bias.size(),0.0);
    for (unsigned int iflow=0; iflow<entry->barcode_bias.size(); iflow++)
      read_group_barcode_bias_[entry->read_group_index].at(iflow) += entry->barcode_bias.at(iflow);
    // Barcode signal distance histogram [binned to 0.2 intervals]
    int n_hist = min((int)(5.0*entry->barcode_distance), 4);
    read_group_barcode_distance_hist_.at(entry->read_group_index).at(n_hist)++;
    // 0-error filtered barcodes
    if (entry->barcode_filt_zero_error >= 0)
    	read_group_barcode_filt_zero_err_.at(entry->barcode_filt_zero_error)++;
    // Account for filtered reads due to barcode adapter rejection
    if (entry->barcode_adapter_filtered >= 0)
           read_group_barcode_adapter_rejected_.at(entry->barcode_adapter_filtered)++;


    // Actually write out the read

    entry->bam.AddTag("RG","Z", read_group_name_[entry->read_group_index]);
    entry->bam.AddTag("PG","Z", string("bc"));

    if (not bam_writer_[target_file_idx]) {
      // Open Bam for writing
      RefVector empty_reference_vector;
      bam_writer_[target_file_idx] = new BamWriter();
      if (compress_bam_)
        bam_writer_[target_file_idx]->SetCompressionMode(BamWriter::Compressed);
      else
        bam_writer_[target_file_idx]->SetCompressionMode(BamWriter::Uncompressed);
      bam_writer_[target_file_idx]->SetNumThreads(num_bamwriter_threads_);

      if (not bam_writer_[target_file_idx]->Open(bam_filename_[target_file_idx], sam_header_[target_file_idx], empty_reference_vector)) {
        cerr << "BaseCaller IO error: Failed to create bam file " << bam_filename_[target_file_idx] << endl;
        exit(EXIT_FAILURE);
      }
    }
    if (not bam_writer_[target_file_idx]->SaveAlignment(entry->bam)){
      cerr << "BaseCaller IO error: Failed to write to bam file " << bam_filename_[target_file_idx] << endl;
      exit(EXIT_FAILURE);
    }
  }
}



