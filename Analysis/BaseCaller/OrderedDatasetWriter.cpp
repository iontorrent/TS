/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedDatasetWriter.cpp
//! @ingroup  BaseCaller
//! @brief    OrderedDatasetWriter. Thread-safe, barcode-friendly SFF/BAM writer with deterministic order

#include "OrderedDatasetWriter.h"

#include <stdio.h>
#include <stdlib.h>

#include "api/SamHeader.h"
#include "api/BamAlignment.h"

using namespace std;


OrderedDatasetWriter::OrderedDatasetWriter()
{
  num_regions_ = 0;
  num_regions_written_ = 0;
  pthread_mutex_init(&dropbox_write_mutex_, NULL);
  pthread_mutex_init(&sff_write_mutex_, NULL);
}

OrderedDatasetWriter::~OrderedDatasetWriter()
{
  pthread_mutex_destroy(&dropbox_write_mutex_);
  pthread_mutex_destroy(&sff_write_mutex_);
}


void OrderedDatasetWriter::Open(const string& base_directory, BarcodeDatasets& datasets, int num_regions, const ion::FlowOrder& flow_order, const string& key,
    const string& basecaller_name, const string& basecalller_version, const string& basecaller_command_line,
    const string& production_date, const string& platform_unit)
{
  num_regions_ = num_regions;
  num_regions_written_ = 0;
  region_ready_.assign(num_regions_+1,false);
  region_dropbox_.clear();
  region_dropbox_.resize(num_regions_);

  qv_histogram_.assign(50,0);

  num_datasets_ = datasets.num_datasets();
  num_barcodes_ = datasets.num_barcodes();
  num_reads_.resize(num_datasets_,0);
  bam_filename_.resize(num_datasets_);

  map_barcode_to_dataset_.clear();

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

      int rg_index = datasets.read_group_name_to_id(read_group_name);
      map_barcode_id_to_rg_[rg_index] = read_group_name;
      map_barcode_to_dataset_[rg_index] = ds;
      num_reads_per_barcode_[rg_index] = 0;
      num_bases_per_barcode_[rg_index] = 0;
      num_Q20_bases_per_barcode_[rg_index] = 0;
      num_barcode_errors_[rg_index].assign(3,0);

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




void OrderedDatasetWriter::Close(BarcodeDatasets& datasets, bool quiet)
{

  for (;num_regions_written_ < num_regions_; num_regions_written_++)
    PhysicalWriteRegion(num_regions_written_);

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
      read_group_json["read_count"] = num_reads_per_barcode_[rg_index];
      read_group_json["total_bases"] = num_bases_per_barcode_[rg_index];
      read_group_json["Q20_bases"] = num_Q20_bases_per_barcode_[rg_index];
      read_group_json["barcode_errors_hist"][0] = num_barcode_errors_[rg_index][0];
      read_group_json["barcode_errors_hist"][1] = num_barcode_errors_[rg_index][1];
      read_group_json["barcode_errors_hist"][2] = num_barcode_errors_[rg_index][2];
    }

    if (!quiet)
      printf("Generated %s with %d reads\n", bam_filename_[ds].c_str(), num_reads_[ds]);

  }
}


void OrderedDatasetWriter::WriteRegion(int region, deque<SFFEntry> &region_reads)
{
  // Deposit results in the dropbox
  pthread_mutex_lock(&dropbox_write_mutex_);
  region_dropbox_[region].swap(region_reads);
  region_ready_[region] = true;
  pthread_mutex_unlock(&dropbox_write_mutex_);

  // Attempt writing duty
  if (pthread_mutex_trylock(&sff_write_mutex_))
    return;
  while (true) {
    pthread_mutex_lock(&dropbox_write_mutex_);
    bool cannot_write = !region_ready_[num_regions_written_];
    pthread_mutex_unlock(&dropbox_write_mutex_);
    if (cannot_write)
      break;
    PhysicalWriteRegion(num_regions_written_);
    num_regions_written_++;
  }
  pthread_mutex_unlock(&sff_write_mutex_);
}


void OrderedDatasetWriter::PhysicalWriteRegion(int region)
{
  for (deque<SFFEntry>::iterator entry = region_dropbox_[region].begin(); entry != region_dropbox_[region].end(); entry++) {

    // write

    int target_file_idx = map_barcode_to_dataset_[entry->barcode_id];
    num_reads_[target_file_idx]++;
    num_reads_per_barcode_[entry->barcode_id]++;

    BamAlignment bam_alignment;

    int clip_start = 0;
    if (entry->clip_qual_left > 0)
      clip_start = max(clip_start,(int)entry->clip_qual_left-1);
    if (entry->clip_adapter_left > 0)
      clip_start = max(clip_start,(int)entry->clip_adapter_left-1);

    int clip_end = entry->n_bases;
    if (entry->clip_qual_right > 0)
      clip_end = min(clip_end,(int)entry->clip_qual_right);
    if (entry->clip_adapter_right > 0)
      clip_end = min(clip_end,(int)entry->clip_adapter_right);

    bam_alignment.SetIsMapped(false);
    bam_alignment.Name = entry->name;
    bam_alignment.QueryBases.reserve(entry->n_bases);
    bam_alignment.Qualities.reserve(entry->n_bases);
    for (int base = clip_start; base < clip_end; ++base) {
      bam_alignment.QueryBases.push_back(entry->bases[base]);
      bam_alignment.Qualities.push_back(entry->quality[base] + 33);
      num_bases_per_barcode_[entry->barcode_id]++;
      if (entry->quality[base]>=20)
        num_Q20_bases_per_barcode_[entry->barcode_id]++;
      int clipped_quality = min((int)entry->quality[base],49);
      qv_histogram_[clipped_quality]++;
    }

    int clip_flow = 0;
    for (int base = 0; base <= clip_start and base < entry->n_bases; ++base)
      clip_flow += entry->flow_index[base];
    if (clip_flow > 0)
      clip_flow--;

    bam_alignment.AddTag("RG","Z", map_barcode_id_to_rg_[entry->barcode_id]);
    bam_alignment.AddTag("PG","Z", string("bc"));
    bam_alignment.AddTag("ZF","i", clip_flow);
    bam_alignment.AddTag("FZ", entry->flowgram);

    // This should be optional
    if (entry->clip_adapter_right > 0) {
      bam_alignment.AddTag("ZA", "i", entry->clip_adapter_right - clip_start);
      bam_alignment.AddTag("ZG", "i", entry->clip_adapter_flow);
    }

    bam_writer_[target_file_idx]->SaveAlignment(bam_alignment);

    int n_errors = max(0,min(2,entry->barcode_n_errors));
    num_barcode_errors_[entry->barcode_id][n_errors]++;


  }
  region_dropbox_[region].clear();
}




void  SFFEntry::swap(SFFEntry &w)
{
  int x;
  int32_t y;
  x = n_bases; n_bases = w.n_bases; w.n_bases = x;
  y = clip_qual_left; clip_qual_left = w.clip_qual_left; w.clip_qual_left = y;
  y = clip_qual_right; clip_qual_right = w.clip_qual_right; w.clip_qual_right = y;
  y = clip_adapter_left; clip_adapter_left = w.clip_adapter_left; w.clip_adapter_left = y;
  y = clip_adapter_right; clip_adapter_right = w.clip_adapter_right; w.clip_adapter_right = y;
  x = clip_adapter_flow; clip_adapter_flow = w.clip_adapter_flow; w.clip_adapter_flow = x;
  x = barcode_id; barcode_id = w.barcode_id; w.barcode_id = x;
  name.swap(w.name);
  flowgram.swap(w.flowgram);
  flow_index.swap(w.flow_index);
  bases.swap(w.bases);
  quality.swap(w.quality);
}



