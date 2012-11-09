/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedRegionSFFWriter.cpp
//! @ingroup  BaseCaller
//! @brief    OrderedRegionSFFWriter. Thread-safe SFF writer with deterministic order

#include "OrderedRegionSFFWriter.h"

#include <stdio.h>
#include <stdlib.h>

#include "file-io/sff.h"
#include "file-io/sff_file.h"
#include "file-io/sff_header.h"
#include "file-io/sff_read_header.h"
#include "file-io/sff_read.h"

using namespace std;


OrderedRegionSFFWriter::OrderedRegionSFFWriter()
{
  sff_file_ = NULL;
  sff_ = NULL;
  num_reads_ = 0;
  num_regions_ = 0;
  num_regions_written_ = 0;
  pthread_mutex_init(&dropbox_write_mutex_, NULL);
  pthread_mutex_init(&sff_write_mutex_, NULL);
}

OrderedRegionSFFWriter::~OrderedRegionSFFWriter()
{
  pthread_mutex_destroy(&dropbox_write_mutex_);
  pthread_mutex_destroy(&sff_write_mutex_);
}


void OrderedRegionSFFWriter::Open(const string& sff_filename, int num_regions, const ion::FlowOrder& flow_order, const string& key)
{
  num_reads_ = 0;
  num_regions_ = num_regions;
  num_regions_written_ = 0;
  region_ready_.assign(num_regions_+1,false);
  region_dropbox_.clear();
  region_dropbox_.resize(num_regions_);

  sff_header_t *sff_header = sff_header_init1(num_reads_, flow_order.num_flows(), (char*)flow_order.c_str(), (char*)key.c_str());
  sff_file_ = sff_fopen((char*)sff_filename.c_str(), "wb", sff_header, NULL);
  sff_header_destroy(sff_header);

  sff_ = sff_init1();
  sff_->gheader = sff_file_->header;
  sff_->rheader->name = ion_string_init(0);
  sff_->read->bases = ion_string_init(0);
  sff_->read->quality = ion_string_init(0);
}


void OrderedRegionSFFWriter::Close()
{
  for (;num_regions_written_ < num_regions_; num_regions_written_++)
    PhysicalWriteRegion(num_regions_written_);

  fseek(sff_file_->fp, 0, SEEK_SET);
  sff_file_->header->n_reads = num_reads_;
  sff_header_write(sff_file_->fp, sff_file_->header);

  sff_fclose(sff_file_);

  free(sff_->read->bases);
  free(sff_->read->quality);
  free(sff_->read);
  sff_->read = NULL;
  free(sff_->rheader->name);
  sff_->rheader->name = NULL;
  sff_destroy(sff_);
}


void OrderedRegionSFFWriter::WriteRegion(int region, deque<SFFEntry> &region_reads)
{
  // Deposit results in the dropbox
  pthread_mutex_lock(&dropbox_write_mutex_);
  region_dropbox_[region].clear();
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


void OrderedRegionSFFWriter::PhysicalWriteRegion(int region)
{
  for (deque<SFFEntry>::iterator entry = region_dropbox_[region].begin(); entry != region_dropbox_[region].end(); entry++) {

    // initialize the header
    sff_->rheader->name_length = entry->name.length();
    sff_->rheader->name->s = (char *)entry->name.c_str();
    sff_->rheader->n_bases = entry->n_bases;
    sff_->rheader->clip_qual_left = entry->clip_qual_left;
    sff_->rheader->clip_qual_right = entry->clip_qual_right;
    sff_->rheader->clip_adapter_left = entry->clip_adapter_left;
    sff_->rheader->clip_adapter_right = entry->clip_adapter_right;

    // initialize the read
    sff_->read->flowgram = &(entry->flowgram[0]);
    sff_->read->flow_index = &(entry->flow_index[0]);
    sff_->read->bases->s = &(entry->bases[0]);
    sff_->read->quality->s = (char *)&(entry->quality[0]);

    // write
    sff_write(sff_file_, sff_);

    num_reads_++;
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
  x = barcode_id; barcode_id = w.barcode_id; w.barcode_id = x;
  name.swap(w.name);
  flowgram.swap(w.flowgram);
  flow_index.swap(w.flow_index);
  bases.swap(w.bases);
  quality.swap(w.quality);
}



