/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     OrderedVCFWriter.h
//! @ingroup  VariantCaller
//! @brief    Thread-safe and out-of-order-safe VCF writer

#ifndef ORDEREDVCFWRITER_H
#define ORDEREDVCFWRITER_H

#include <fstream>
#include <deque>
#include <pthread.h>
#include <Variant.h>
#include <errno.h>

#include "VcfFormat.h"
#include "InputStructures.h"

using namespace std;


class OrderedVCFWriter {
public:
  OrderedVCFWriter() {
    num_slots_ = 0;
    num_slots_written_ = 0;
    slot_ready_.push_back(false);
    suppress_no_calls_ = true;
    pthread_mutex_init(&slot_mutex_, NULL);
    pthread_mutex_init(&write_mutex_, NULL);
  }
  ~OrderedVCFWriter() {
    pthread_mutex_destroy(&slot_mutex_);
    pthread_mutex_destroy(&write_mutex_);
  }


  void Initialize(const string& output_vcf, const ExtendParameters& parameters, ReferenceReader& ref_reader, const SampleManager& sample_manager, bool use_molecular_tag = false) {

    string filtered_vcf;
    size_t pos = output_vcf.rfind(".");
    if (pos != string::npos)
      filtered_vcf = output_vcf.substr(0, pos);
    else
      filtered_vcf = output_vcf;
    filtered_vcf += "_filtered.vcf";

    output_vcf_stream_.open(output_vcf.c_str());
    if (not output_vcf_stream_.is_open()) {
      cerr << "ERROR: Cannot open output vcf file " << output_vcf << " : " << strerror(errno) << endl;
      exit(1);
    }
    filtered_vcf_stream_.open(filtered_vcf.c_str());
    if (not filtered_vcf_stream_.is_open()) {
      cerr << "ERROR: Cannot open filtered vcf file " << filtered_vcf << " : " << strerror(errno) << endl;
      exit(1);
    }
    suppress_no_calls_ = parameters.my_controls.suppress_no_calls;

    string vcf_header = getVCFHeader(&parameters, ref_reader, sample_manager.sample_names_, sample_manager.primary_sample_, use_molecular_tag);
    output_vcf_stream_ << vcf_header << endl;
    filtered_vcf_stream_ << vcf_header << endl;
    variant_initializer_.parseHeader(vcf_header);
  }

  vcf::VariantCallFile& VariantInitializer() { return variant_initializer_; }

  void Close() {

    while (num_slots_written_ < num_slots_) {
      for (deque<VariantCandidate>::iterator current_variant = slot_dropbox_[num_slots_written_].begin();
          current_variant != slot_dropbox_[num_slots_written_].end(); ++current_variant) {
        if (current_variant->variant.isFiltered and !current_variant->variant.isHotSpot and suppress_no_calls_)
          filtered_vcf_stream_ << current_variant->variant << endl;
        else
          output_vcf_stream_ << current_variant->variant << endl;
      }
      slot_dropbox_[num_slots_written_].clear();
      num_slots_written_++;

    }
    output_vcf_stream_.close();
    filtered_vcf_stream_.close();
  }

  int ReserveSlot() {
    pthread_mutex_lock(&slot_mutex_);
    int my_slot = num_slots_;
    ++num_slots_;
    slot_ready_.push_back(false);
    slot_dropbox_.push_back(deque<VariantCandidate>());
    pthread_mutex_unlock(&slot_mutex_);
    return my_slot;
  }

  void WriteSlot(int slot, deque<VariantCandidate> &variant_batch) {
    // Deposit results in the dropbox
    pthread_mutex_lock(&slot_mutex_);
    slot_dropbox_[slot].swap(variant_batch);
    slot_ready_[slot] = true;
    pthread_mutex_unlock(&slot_mutex_);

    // Attempt writing duty
    if (pthread_mutex_trylock(&write_mutex_))
      return;
    while (true) {
      pthread_mutex_lock(&slot_mutex_);
      bool cannot_write = !slot_ready_[num_slots_written_];
      pthread_mutex_unlock(&slot_mutex_);
      if (cannot_write)
        break;
      for (deque<VariantCandidate>::iterator current_variant = slot_dropbox_[num_slots_written_].begin();
          current_variant != slot_dropbox_[num_slots_written_].end(); ++current_variant) {
        if (current_variant->variant.isFiltered and !current_variant->variant.isHotSpot and suppress_no_calls_)
          filtered_vcf_stream_ << current_variant->variant << endl;
        else
          output_vcf_stream_ << current_variant->variant << endl;
      }
      slot_dropbox_[num_slots_written_].clear();
      num_slots_written_++;
    }
    pthread_mutex_unlock(&write_mutex_);
  }

private:
  int                           num_slots_;             //! Total number of slots reserved so far
  int                           num_slots_written_;     //! Number of slots physically written so far
  deque<bool>                   slot_ready_;            //! Which slots are ready for writing?
  deque<deque<VariantCandidate> >   slot_dropbox_;      //! Slots for variants that are ready for writing
  pthread_mutex_t               slot_mutex_;            //! Mutex controlling access to the dropbox
  pthread_mutex_t               write_mutex_;           //! Mutex controlling VCF writing
  ofstream                      output_vcf_stream_;     //! Main output VCF file
  ofstream                      filtered_vcf_stream_;   //! Filtered VCF file
  bool                          suppress_no_calls_;     //! If false, filtered variants also go to main VCF
  vcf::VariantCallFile          variant_initializer_;   //! Fake writer to initialize new Variant objects
};



#endif // ORDEREDVCFWRITER_H
