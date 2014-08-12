/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ReferenceReader.h
//! @ingroup  VariantCaller
//! @brief    Memory-mapped reader for fasta+fai reference

#ifndef REFERENCEREADER_H
#define REFERENCEREADER_H

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

using namespace std;

#define FAST_TO_UPPER(c)  ((c)&0x5F)

class ReferenceReader {
public:
  ReferenceReader () : initialized_(false), ref_handle_(0), ref_mmap_(0) {}
  ~ReferenceReader () { Cleanup(); }

  void Initialize(const string& fasta_filename) {
    Cleanup();

    ref_handle_ = open(fasta_filename.c_str(),O_RDONLY);
    if (ref_handle_ < 0) {
      cerr << "ERROR: Cannot open fasta file " << fasta_filename << " : " << strerror(errno) << endl;
      exit(1);
    }

    fstat(ref_handle_, &ref_stat_);
    ref_mmap_ = (char *)mmap(0, ref_stat_.st_size, PROT_READ, MAP_SHARED, ref_handle_, 0);

    string fai_filename = fasta_filename + ".fai";
    FILE *fai = fopen(fai_filename.c_str(), "r");
    if (!fai) {
      cerr << "ERROR: Cannot open fasta index file " << fai_filename << " : " << strerror(errno) << endl;
      exit(1);
    }

    char line[1024], chrom_name[1024];
    while (fgets(line, 1024, fai) != NULL) {
      Reference ref_entry;
      long chr_start;
      if (5 != sscanf(line, "%1020s\t%ld\t%ld\t%d\t%d", chrom_name, &ref_entry.size, &chr_start,
                      &ref_entry.bases_per_line, &ref_entry.bytes_per_line))
        continue;
      ref_entry.chr = chrom_name;
      ref_entry.start = ref_mmap_ + chr_start;
      ref_entry.begin_ = ref_entry.iter(0);
      ref_entry.end_ = ref_entry.iter(ref_entry.size);
      ref_index_.push_back(ref_entry);
      ref_map_[ref_entry.chr] = (int) ref_index_.size() - 1;
    }
    fclose(fai);
    initialized_ = true;
  }

  bool initialized() const { return initialized_; }
  int chr_count() const { return (int)ref_index_.size(); }
  const char *chr(int idx) const { return ref_index_[idx].chr.c_str(); }
  const string& chr_str(int idx) const { return ref_index_[idx].chr; }
  char base(int chr_idx, long pos) const { return ref_index_[chr_idx].base(pos); }
  long chr_size(int idx) const { return ref_index_[idx].size; }

  int chr_idx(const char *chr_name) const {
    string string_chr(chr_name);
    map<string,int>::const_iterator I = ref_map_.find(string_chr);
    if (I != ref_map_.end())
      return I->second;
    I = ref_map_.find("chr"+string_chr);
    if (I != ref_map_.end())
      return I->second;
    if (string_chr == "MT") {
      I = ref_map_.find("chrM");
      if (I != ref_map_.end())
        return I->second;
    }
    return -1;
  }


  class iterator {
  public:
    iterator() : pos_(0), start_of_line_(0), end_of_line_(0), bytes_per_line_(0), gap_size_(0) {}

    iterator(const char *pos, const char *end_of_line, int bases_per_line, int bytes_per_line)
      : pos_(pos), start_of_line_(end_of_line-bases_per_line), end_of_line_(end_of_line),
        bytes_per_line_(bytes_per_line), gap_size_(bytes_per_line-bases_per_line) {}

    void operator++() {
      ++pos_;
      if (pos_ == end_of_line_) {
        pos_ += gap_size_;
        start_of_line_ += bytes_per_line_;
        end_of_line_ += bytes_per_line_;
      }
    }
    void operator--() {
      --pos_;
      if (pos_ < start_of_line_) {
        pos_ -= gap_size_;
        start_of_line_ -= bytes_per_line_;
        end_of_line_ -= bytes_per_line_;
      }
    }
    void operator+=(unsigned int delta) {
      pos_ += delta;
      while (pos_ >= end_of_line_) {
        pos_ += gap_size_;
        start_of_line_ += bytes_per_line_;
        end_of_line_ += bytes_per_line_;
      }
    }
    void operator-=(unsigned int delta) {
      pos_ -= delta;
      while(pos_ < start_of_line_) {
        pos_ -= gap_size_;
        start_of_line_ -= bytes_per_line_;
        end_of_line_ -= bytes_per_line_;
      }
    }
    char operator*() const {
      return FAST_TO_UPPER(*pos_);
    }
    bool operator==(const iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const iterator& other) const { return pos_ != other.pos_; }
    bool operator< (const iterator& other) const { return pos_ < other.pos_; }
    bool operator> (const iterator& other) const { return pos_ > other.pos_; }
    bool operator>=(const iterator& other) const { return pos_ >= other.pos_; }

  private:
    const char *  pos_;
    const char *  start_of_line_;
    const char *  end_of_line_;
    int           bytes_per_line_;
    int           gap_size_;
  };

  iterator iter(int chr_idx, long pos) const {  return ref_index_[chr_idx].iter(pos); }
  const iterator& begin(int chr_idx) const {  return ref_index_[chr_idx].begin(); }
  const iterator& end(int chr_idx) const {  return ref_index_[chr_idx].end(); }

  string substr(int chr_idx, long pos, long len) const {
    string s;
    s.reserve(len+1);
    iterator I = iter(chr_idx,pos);
    while (len and I < end(chr_idx)) {
      s.push_back(*I);
      ++I;
      --len;
    }
    return s;
  }

private:
  void Cleanup() {
    if (initialized_) {
      munmap(ref_mmap_, ref_stat_.st_size);
      close(ref_handle_);
      ref_index_.clear();
      ref_map_.clear();
      initialized_ = false;
    }
  }

  struct Reference {
    string            chr;
    long              size;
    const char *      start;
    int               bases_per_line;
    int               bytes_per_line;
    iterator          begin_;
    iterator          end_;

    char base(long pos) const {
      if (pos < 0 or pos >= size)
        return 'N';
      long ref_line_idx = pos / bases_per_line;
      long ref_line_pos = pos % bases_per_line;
      return toupper(start[ref_line_idx*bytes_per_line + ref_line_pos]);
    }

    iterator iter(long pos) const {
      if (pos < 0 or pos > size)
        pos = size;
      long ref_line_idx = pos / bases_per_line;
      long ref_line_pos = pos % bases_per_line;
      return iterator(start + ref_line_idx*bytes_per_line + ref_line_pos,
          start + ref_line_idx*bytes_per_line + bases_per_line,
          bases_per_line, bytes_per_line);
    }
    const iterator& begin() const { return begin_; }
    const iterator& end() const { return end_; }
  };



  bool                initialized_;
  int                 ref_handle_;
  struct stat         ref_stat_;
  char *              ref_mmap_;
  vector<Reference>   ref_index_;
  map<string,int>     ref_map_;

};


#endif // REFERENCEREADER_H
