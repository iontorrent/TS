/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#include "tvcutils.h"

#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <stdlib.h>
#include <errno.h>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <cstring>
#include <boost/math/distributions/poisson.hpp>
#include "viterbi.h"
#include "ReferenceReader.h"
#include <Variant.h>
#include "TargetsManager.h"

#include "OptArgs.h"
#include "IonVersion.h"
#include "json/json.h"
#include "unify_vcf.h"

#define DEFAULT_STDIN_PARAM "stdin"
#define UNIFY_VARIANTS "[unify_variants]"
#define GT "GT"
#define DP "DP"
#define MAX_DP "MAX_DP"
#define MIN_DP "MIN_DP"

void build_index(const string &path_to_gz);

using namespace std;



void UnifyVcfHelp()
{
  printf ("\n");
  printf ("tvcutils %s-%s (%s) - Miscellaneous tools used by Torrent Variant Caller plugin and workflow.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   tvcutils unify_vcf [options]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -t,--novel-tvc-vcf             FILE       small variants VCF file produced by TVC (required)\n");
  printf ("  -i,--novel-assembly-vcf        FILE       long indels VCF file produced by TVCASSEMBLY (optional)\n");
  printf ("  -a,--hotspot-annotation-vcf    FILE       hotspot annotations VCF file (optional)\n");
  printf ("  -a,--hotspot-annotation-vcf    FILE       hotspot annotations VCF file (optional)\n");
  printf ("  -f,--reference-fasta           FILE       FASTA file containing reference genome (required)\n");
  printf ("  -b,--target-file               FILE       target regions BED file (optional)\n");
  printf ("  -o,--output-vcf                FILE       output merged and annotated VCF or VCF.GZ file (required)\n");
  printf ("  -j,--tvc-metrics               FILE       JSON file with tvc metrics (optional)\n");
  printf ("  -d,--input-depth               FILE       output of samtools depth. if provided, will cause generation of gvcf (optional, stdin ok)\n");
  printf ("  -m,--min-depth                 INT        minimum coverage depth in GVCF output (optional)\n");
  printf ("\n");
}





CoverageInfoEntry* CoverageInfoEntry::parse(const char * line, const ReferenceReader& r) {
  CoverageInfoEntry* entry = new CoverageInfoEntry(r);
  string s;
  stringstream ss(line);

  getline(ss, s, '\t');

  if (ss.eof()) return NULL;
  entry->sequenceName = s;
  getline(ss, s, '\t');
  if (ss.eof()) return NULL;
  entry->position = atol(s.c_str());
  while (!ss.eof()) {
    getline(ss, s, '\t');
    entry->cov += strtoul(s.c_str(), NULL, 10);
  }
  return entry;
}

// BGZF_stream methods implementations
template <typename T>
bgzf_stream &bgzf_stream::operator<<(T& data) {
  stringstream ss;
  ss << data;
  this->data += ss.str();
  while (write_buffer() != -1) { }
  return *this;
}

int bgzf_stream::write(int length) {
  int written = _bgzf_write(bgzf, data.c_str(), length);
  data.erase(data.begin(), data.begin() + written);
  return written;
}

void bgzf_stream::flush() {
  while (!data.empty()) {
    write(data.length());
  }
}

int bgzf_stream::write_buffer() {
  while (data.length() >= WINDOW_SIZE) {
    return write(WINDOW_SIZE);
  }
  return -1;
}

// PriorityQueue methods implementations
bool PriorityQueue::get_next_variant(vcf::Variant* current) {
  // parse VCF line by line
  if (current->vcf->_done) return false;
  bool has_next = current->vcf->getNextVariant(*current);
  if (!has_next) return false;
  map<string, vector<string> >::iterator a = current->info.find("FR");
  if (a == current->info.end()) {
    current->info["FR"].push_back(".");
  }
  return has_next;
}

void PriorityQueue::trim_variant(vcf::Variant* variant) {
  bool flag = true;
  while (flag && variant->ref.length() > 1) {
    string::iterator ref = variant->ref.end() - 1;
    for (vector<string>::iterator alt = variant->alt.begin(); alt != variant->alt.end(); alt++) {
      if (*(alt->end()-1) != *ref || alt->length() == 1) flag = false;
    }
    if (flag) {
      variant->ref.erase(ref, variant->ref.end());
      for (vector<string>::iterator alt = variant->alt.begin(); alt != variant->alt.end(); alt++) {
        vector<string>& omapalt = variant->info["OMAPALT"];
        for (vector<string>::iterator o = omapalt.begin(); o != omapalt.end(); o++)
          if (*o == *alt) o->erase(o->end()-1, o->end());
        alt->erase(alt->end()-1, alt->end());
      }
    }
  }
}

void PriorityQueue::left_align_variant(vcf::Variant* variant) {
  if (!left_align_enabled || variant->alt.empty() || variant->alt.size() > 1) return;

  string alt = variant->alt[0];
  int idx = reference_reader.chr_idx(variant->sequenceName.c_str());

  while(variant->position > 1) {
    if (variant->ref[variant->ref.length() - 1] != alt[alt.length() - 1])
      break;
    char pad = reference_reader.base(idx, variant->position - 2);
    variant->ref = pad + variant->ref.substr(0, variant->ref.length() - 1);
    alt = pad + alt.substr(0, alt.length() - 1);

    // left align of hotspots might cause issue of unsync alt and OMAPALT fields
    vector<string>& oalt = variant->info["OMAPALT"];
    if (oalt.size() == 1) {
      string& oval = oalt[0];
      oval = pad + oval.substr(0, oval.length() - 1);
    }
    variant->position--;
  }
  variant->alt[0] = alt;
}

// VcfOrederedMerger methods implementations
VcfOrderedMerger::VcfOrderedMerger(string& novel_tvc,
                                   string& assembly_tvc,
                                   string& hotspot_tvc,
                                   string& output_tvc,
                                   string& path_to_json,
                                   string& input_depth,
                                   string& gvcf_output,
                                   const ReferenceReader& r,
                                   TargetsManager& tm,
                                   size_t w = 1, size_t min_dp = 0, bool la = false)
                                   : depth_in(NULL), gvcf_out(NULL), reference_reader(r), targets_manager(tm),
                                   left_align_enabled(la),
                                   window_size(w), minimum_depth(min_dp),
                                   novel_queue(novel_tvc, *this, r, w, la, true),
                                   assembly_queue(assembly_tvc, *this, r, w, la, true),
                                   hotspot_queue(hotspot_tvc, *this, r, w, la),
                                   bgz_out(output_tvc),
                                   current_cov_info(NULL) {
  if (!input_depth.empty()) {
    depth_in = (istream *)((input_depth == DEFAULT_STDIN_PARAM) ? &cin : new ifstream(input_depth.c_str()));
    gvcf_out = new bgzf_stream(gvcf_output);
    next_cov_entry();
  }
  write_header(novel_queue.variant_call_file(), path_to_json);
  current_target = targets_manager.merged.begin();
}

VcfOrderedMerger::~VcfOrderedMerger() {
  if (depth_in && depth_in != &cin) delete depth_in;
  if (gvcf_out) delete gvcf_out;
}

template <typename T>
int VcfOrderedMerger::variant_cmp(const T* v1, const vcf::Variant* v2) const {
  int idx_v1 = reference_reader.chr_idx(v1->sequenceName.c_str());
  int idx_v2 = reference_reader.chr_idx(v2->sequenceName.c_str());
  return compare(idx_v1, v1->position, idx_v2, v2->position);
}

template<typename T>
bool VcfOrderedMerger::is_within_target_region(T *variant) {
  long pos = variant->position;
  int chr_idx = reference_reader.chr_idx(variant->sequenceName.c_str());
  while (current_target != targets_manager.merged.end() && chr_idx > current_target->chr) {
    current_target++;
  }
  if (current_target == targets_manager.merged.end() || chr_idx != current_target->chr) return false;
  while (current_target != targets_manager.merged.end() && pos > current_target->end) {
    current_target++;
  }
  return current_target != targets_manager.merged.end()
         && current_target->chr == chr_idx
         && pos >= current_target->begin
         && pos <= current_target->end;
}

void VcfOrderedMerger::perform() {
  // merging ordered files loop
  while (novel_queue.has_value() && assembly_queue.has_value()) {
    vcf::Variant* current;
    int cmp = variant_cmp(novel_queue.current(), assembly_queue.current());
    current = cmp == -1 ? assembly_queue.current() : novel_queue.current();
    if (!cmp) {
        current = merge_overlapping_variants();
    }
    process_and_write_vcf_entry(current);
    if (cmp >= 0) novel_queue.next();
    if (cmp <= 0) assembly_queue.next();
  }

  // write out remaining entries
  while (novel_queue.has_value()) {
    process_and_write_vcf_entry(novel_queue.current());
    novel_queue.next();
  }
  while (assembly_queue.has_value()) {
    process_and_write_vcf_entry(assembly_queue.current());
    assembly_queue.next();
  }
  while (hotspot_queue.has_value()) {
    blacklist_check();
  }
  while (current_target != targets_manager.merged.end()) {
    gvcf_finish_region();
    current_target++;
  }
}

void VcfOrderedMerger::write_header(vcf::VariantCallFile& vcf, string json_path) {
  extend_header(vcf);
  parse_parameters_from_json(json_path, vcf);
  // write out the header to output
  bgz_out << vcf.header << "\n";

  if (gvcf_out) {
    vcf.addHeaderLine("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the variant described in this record\">");
    vcf.addHeaderLine("##INFO=<ID=" MIN_DP ",Number=1,Type=Integer,Description=\"Minimum coverage depth in the region\">");
    vcf.addHeaderLine("##INFO=<ID=" MAX_DP ",Number=1,Type=Integer,Description=\"Maximum coverage depth in the region\">");
    vcf.addHeaderLine("##FORMAT=<ID=" MIN_DP ",Number=1,Type=Integer,Description=\"Minimum coverage depth in the region\">");
    vcf.addHeaderLine("##FORMAT=<ID=" MAX_DP ",Number=1,Type=Integer,Description=\"Maximum coverage depth in the region\">");
    *gvcf_out << vcf.header << "\n";
  }
}

void VcfOrderedMerger::transfer_fields(map<string, vector<string> >& original_field,
                                       map<string, vector<string> >& new_field,
                                       map<string, int>& field_types,
                                       list<pair<int, int> >& map_assembly_to_all,
                                       unsigned int final_num_alleles) {
  for (map<string, vector<string> >::iterator info = new_field.begin(); info != new_field.end(); info++) {
    if (field_types[info->first] == -2) { // Transfer A-field
      if (original_field.find(info->first) == original_field.end())
        original_field[info->first] = vector<string>();
      vector<string>& novel_info_value = original_field[info->first];
      while (novel_info_value.size() < final_num_alleles)
        novel_info_value.push_back(".");

      for (list<pair<int, int> >::iterator index_pair = map_assembly_to_all.begin();
           index_pair != map_assembly_to_all.end(); index_pair++) {
        string& novel_info_indexed_value = novel_info_value[index_pair->second];
        if (novel_info_indexed_value.empty() || novel_info_indexed_value == ".")
          novel_info_indexed_value = info->second[index_pair->first];
      }
    } else { // Transfer non-A field
      if (original_field.find(info->first) == original_field.end() || original_field[info->first].empty() || original_field[info->first][0] == ".")
        original_field[info->first] = info->second;
    }
  }
  for (map<string, vector<string> >::iterator info = original_field.begin(); info != original_field.end(); info++) {
    map<string, vector<string> >::iterator nov = new_field.find(info->first);
    if (nov == new_field.end()) {
      if (field_types[info->first] == -2) { // Span A-field
        while (info->second.size() < final_num_alleles)
          info->second.push_back(".");
      }
    }
  }
}

vcf::Variant* VcfOrderedMerger::merge_overlapping_variants() {
  vcf::Variant& novel_v = *novel_queue.current();
  vcf::Variant& assembly_v = *assembly_queue.current();

  string gt = novel_queue.current()->samples[novel_queue.current()->sampleNames[0]]["GT"][0];
  if (gt == "./." || gt == "0/0") {
    // Logging novel and indel merge
    cout << UNIFY_VARIANTS " Advanced merge of IndelAssembly variant " << assembly_queue.current()->sequenceName
         << ":" << assembly_queue.current()->position << endl;
  } else {
    // Logging skipping event
    cout << UNIFY_VARIANTS " Skipping IndelAssembly variant " << assembly_queue.current()->sequenceName
         << ":" << assembly_queue.current()->position << endl;
    return novel_queue.current();
  }

  span_ref_and_alts();

  // combine alt sequences

  list<pair<int, int> > map_novel_to_all;
  for (vector<string>::iterator alt = novel_v.alt.begin(); alt != novel_v.alt.end(); alt++){
    vector<string>::iterator nov = find(assembly_v.alt.begin(), assembly_v.alt.end(), *alt);
    map_novel_to_all.push_back(make_pair(alt - novel_v.alt.begin(), nov - assembly_v.alt.begin()));
    if (nov == assembly_v.alt.end())
      assembly_v.alt.push_back(*alt);
  }

  for (list<pair<int, int> >::iterator i = map_novel_to_all.begin(); i != map_novel_to_all.end(); i++)
    cout << i->first << ":" << i->second << endl;

  // Transfer INFO fields
  transfer_fields(assembly_v.info, novel_v.info, novel_queue.variant_call_file().infoCounts, map_novel_to_all, assembly_v.alt.size());

  // Transfer FORMAT fields
  assembly_queue.current()->format = novel_queue.current()->format;
  for (map<string, map<string, vector<string> > >::iterator format = novel_v.samples.begin();
       format != novel_v.samples.end(); format++) {
    transfer_fields(assembly_v.samples[format->first], format->second, novel_queue.variant_call_file().formatCounts, map_novel_to_all, assembly_v.alt.size());
  }
  return assembly_queue.current();
}

void VcfOrderedMerger::span_ref_and_alts() {
  long extension_length = novel_queue.current()->ref.length() - assembly_queue.current()->ref.length();
  // extend alt sequences (span to the same length everywhere)
  if (extension_length > 0) {
    // extend assembly alt sequences
    string extension_str = novel_queue.current()->ref.substr(assembly_queue.current()->ref.length());
    vector<string> extended_alts;
    for (vector<string>::iterator alt = assembly_queue.current()->alt.begin(); alt != assembly_queue.current()->alt.end(); alt++){
      extended_alts.push_back(*alt + extension_str);
    }
    assembly_queue.current()->alt = extended_alts;
    assembly_queue.current()->ref = novel_queue.current()->ref;
  } else if (extension_length < 0) {
    // extend novel alt sequences
    string extension_str = assembly_queue.current()->ref.substr(novel_queue.current()->ref.length());
    vector<string> extended_alts;
    for (vector<string>::iterator alt = novel_queue.current()->alt.begin(); alt != novel_queue.current()->alt.end(); alt++){
      extended_alts.push_back(*alt + extension_str);
    }
    novel_queue.current()->alt = extended_alts;
    novel_queue.current()->ref = assembly_queue.current()->ref;
  }
}

void VcfOrderedMerger::generate_novel_annotations(vcf::Variant* variant) {
  if (variant->alt.empty()) return;
  for (vector<string>::iterator alt = variant->alt.begin(); alt != variant->alt.end(); alt++) {
    long i = alt - variant->alt.begin();
    long opos = variant->position;
    string oref = variant->ref;
    string temp_alt = *alt;

    // trim identical ends
    string::iterator orefi = oref.end();
    string::iterator alti = temp_alt.end();
    for (; *orefi == *alti; orefi--, alti--) { }

    if (alti + 1 < temp_alt.end()) temp_alt.erase(alti + 1, temp_alt.end());
    if (orefi + 1 < oref.end()) oref.erase(orefi + 1, oref.end());

    // trim identical beginnings
    orefi = oref.begin();
    alti = temp_alt.begin();
    for (; *orefi == *alti && *alti != '-'; orefi++, alti++) {
      opos++;
    }
    if (distance(temp_alt.begin(), alti) > 0) temp_alt.erase(temp_alt.begin(), alti);
    if (distance(oref.begin(), orefi) > 0) oref.erase(oref.begin(), orefi);

    if (oref.empty()) oref = "-";
    if (temp_alt.empty()) temp_alt = "-";
    stringstream ss;
    ss<<opos;

    push_value_to_vector(variant->info["OID"], i, ".");
    push_value_to_vector(variant->info["OPOS"], i, ss.str());
    push_value_to_vector(variant->info["OREF"], i, oref);
    push_value_to_vector(variant->info["OALT"], i, temp_alt);
    push_value_to_vector(variant->info["OMAPALT"], i, *alt);
  }

}

void VcfOrderedMerger::merge_annotation_into_vcf(vcf::Variant* merged_entry) {
  string annotation_ref_extension;
  long record_ref_extension = 0;
  if (hotspot_queue.current()->ref.length() > merged_entry->ref.length()) {
    record_ref_extension = hotspot_queue.current()->ref.length() - merged_entry->ref.length();
  }
  if (hotspot_queue.current()->ref.length() < merged_entry->ref.length()) {
    annotation_ref_extension = merged_entry->ref.substr(hotspot_queue.current()->ref.length());
  }

  // Form blacklist
  map<string, string> blacklist;
  map<string, vector<string> >::iterator bstr = hotspot_queue.current()->info.find("BSTRAND");
  if (bstr != hotspot_queue.current()->info.end())
    for (vector<string>::iterator key = hotspot_queue.current()->alt.begin(), v = bstr->second.begin();
         key != hotspot_queue.current()->alt.end() && v != bstr->second.end(); key++, v++) {
      blacklist[*key] = *v;
    }

  vector<string> filtered_oid;

  for (vector<string>::iterator oid = hotspot_queue.current()->info["OID"].begin(),
           opos = hotspot_queue.current()->info["OPOS"].begin(),
           oref = hotspot_queue.current()->info["OREF"].begin(), oalt = hotspot_queue.current()->info["OALT"].begin(),
           omapalt = hotspot_queue.current()->info["OMAPALT"].begin();
       oid != hotspot_queue.current()->info["OID"].end() && opos != hotspot_queue.current()->info["OPOS"].end() &&
       oref != hotspot_queue.current()->info["OREF"].end() && oalt != hotspot_queue.current()->info["OALT"].end() &&
       omapalt != hotspot_queue.current()->info["OMAPALT"].end();
       oid++, opos++, oref++, oalt++, omapalt++) {
    if (!blacklist.empty() && blacklist[*omapalt] != ".")
      continue;
    filtered_oid.push_back(*oid);

    if (record_ref_extension) {
      if (hotspot_queue.current()->ref.substr(hotspot_queue.current()->ref.length() - record_ref_extension) !=
          omapalt->substr(omapalt->length() - record_ref_extension)) {
        cout << UNIFY_VARIANTS << " Hotspot annotation " << merged_entry->sequenceName
        << ":" << merged_entry->position << ", allele " << *omapalt << " not eligible for shortening.\n";
        continue;
      }
      *omapalt = omapalt->substr(0, omapalt->length() - record_ref_extension);
    }
    *omapalt = *omapalt + annotation_ref_extension;

    vector<string>::iterator omapalti =
        find(merged_entry->info["OMAPALT"].begin(), merged_entry->info["OMAPALT"].end(), *omapalt);

    if (omapalti == merged_entry->info["OMAPALT"].end()) {
      cout << UNIFY_VARIANTS << " Hotspot annotation " << merged_entry->sequenceName
      << ":" << merged_entry->position << ", allele " << *omapalt << " not found in merged variant file.\n";
      continue;
    }
    if (oref->length() >= 1 && oalt->length() >= 1 && (*oref)[0] == (*oalt)[0]) {
      *oref = oref->substr(1);
      *oalt = oalt->substr(1);
      long p = atol(opos->c_str());
      stringstream ss;
      ss<<++p;
      *opos = ss.str();
    }
    if (oref->empty()) {
      *oref = "-";
    }
    if(oalt->empty()) {
      *oalt = "-";
    }
    long idx = omapalti - merged_entry->info["OMAPALT"].begin();

    if (merged_entry->info["OID"][idx] == ".") {
      merged_entry->info["OID"][idx] = *oid;
      merged_entry->info["OPOS"][idx] = *opos;
      merged_entry->info["OREF"][idx] = *oref;
      merged_entry->info["OALT"][idx] = *oalt;
      merged_entry->info["OMAPALT"][idx] = *omapalt;
    } else {
      merged_entry->info["OID"].push_back(*oid);
      merged_entry->info["OPOS"].push_back(*opos);
      merged_entry->info["OREF"].push_back(*oref);
      merged_entry->info["OALT"].push_back(*oalt);
      merged_entry->info["OMAPALT"].push_back(*omapalt);
    }
  }
  if (!filtered_oid.empty()) {
    merged_entry->id = join(filtered_oid, ";");
  }
}

void VcfOrderedMerger::process_and_write_vcf_entry(vcf::Variant* current) {
  generate_novel_annotations(current);
  process_annotation(current);
  if (is_within_target_region(current)) {
    gvcf_process(current->sequenceName.c_str(), current->position);
    gvcf_out_variant(current);
    bgz_out  << *current << "\n";

  } else cout << UNIFY_VARIANTS " Skipping " << current->sequenceName << ":" << current->position
              << " outside of target regions.\n";
}

void VcfOrderedMerger::gvcf_out_variant(vcf::Variant *current) {
  if (gvcf_out) {
      while (current_cov_info && current_cov_info->position - current->position < (int)current->ref.length()) next_cov_entry();
      *gvcf_out << *current << "\n";
    }
}

void VcfOrderedMerger::gvcf_process(int chr, long position) {
  if (!gvcf_out || !current_cov_info) return;

  if (compare(current_cov_info->chr(), current_cov_info->position, chr, position) == -1) return;

  while (current_cov_info && compare(current_cov_info->chr(), current_cov_info->position, chr, position) == 1) {
    vector<size_t> depth_values;
    long pos = current_cov_info->position, end;
    int prev_chr = current_cov_info->chr();

    while (current_cov_info && compare(current_cov_info->chr(), current_cov_info->position, chr, position) == 1) {
      if (current_cov_info->cov <= minimum_depth) {
        next_cov_entry();
        break;
      }

      depth_values.push_back(current_cov_info->cov);
      end = current_cov_info->position;
      prev_chr = current_cov_info->chr();
      next_cov_entry();
      if (!current_cov_info || prev_chr != current_cov_info->chr() || current_cov_info->position - end != 1) break;
    }
    if (depth_values.empty()) {
      continue;
    }

    markov_chain<size_t> ch(depth_values.begin(), depth_values.end());
    long p = pos;
    for (vector<pair<depth_info<size_t>, long> >::reverse_iterator it = ch.ibegin(); it != ch.iend(); it++) {
      vcf::Variant gvcf_entry = generate_gvcf_entry(novel_queue.variant_call_file(), prev_chr,
                                                    it->first, p, pos + it->second);
      *gvcf_out << gvcf_entry << "\n";
      p = pos + it->second + 1;
    }
  }
}

void VcfOrderedMerger::gvcf_finish_region() {
  gvcf_process(current_target->chr, current_target->end + 1);
}

void VcfOrderedMerger::gvcf_process(const char * seq_name, long pos) {
  gvcf_process(reference_reader.chr_idx(seq_name), pos);
}

vcf::Variant VcfOrderedMerger::generate_gvcf_entry(vcf::VariantCallFile& current, int chr,
                                                   depth_info<size_t> depth, long pos, long end) const {
  stringstream ss;
  string gt = GT, dp = DP, min_dp = MIN_DP, max_dp = MAX_DP;
  vcf::Variant gvcf_entry(current);

  ss << end;
  push_info_field(gvcf_entry, ss.str(), "END");

  gvcf_entry.sequenceName = reference_reader.chr(chr);
  gvcf_entry.position = pos;
  gvcf_entry.addFormatField(gt);
  gvcf_entry.addFormatField(dp);
  gvcf_entry.addFormatField(min_dp);
  gvcf_entry.addFormatField(max_dp);
  gvcf_entry.alt.push_back(".");
  gvcf_entry.id = ".";
  gvcf_entry.filter = "PASS";
  gvcf_entry.sampleNames = current.sampleNames;
  gvcf_entry.quality = .0;
  gvcf_entry.ref = reference_reader.base(chr, pos);

  push_format_field(gvcf_entry, "0/0", gt);

  push_value_to_entry(gvcf_entry, depth.dp, dp);
  push_value_to_entry(gvcf_entry, depth.min_dp, min_dp);
  push_value_to_entry(gvcf_entry, depth.max_dp, max_dp);

  return gvcf_entry;
}

void VcfOrderedMerger::next_cov_entry() {
  if (current_cov_info != NULL) delete current_cov_info;
  if (depth_in->eof()) {
    current_cov_info = NULL;
    return;
  }
  string line;
  getline(*depth_in, line);
  if (line.empty()) {
    current_cov_info = NULL;
    return;
  }
  current_cov_info = CoverageInfoEntry::parse(line, reference_reader);
}

void VcfOrderedMerger::process_annotation(vcf::Variant* current) {
  int cmp;
  if (hotspot_queue.has_value()) do {
    cmp = variant_cmp(current, hotspot_queue.current());
    if (cmp == 1) return;
    if (cmp == 0) {
      merge_annotation_into_vcf(current);
      hotspot_queue.next();
      return;
    }
      blacklist_check();
    } while (cmp == -1 && hotspot_queue.has_value());
}

void VcfOrderedMerger::blacklist_check() {
  map<string, vector<string> >::iterator bstr = hotspot_queue.current()->info.find("BSTRAND");
  bool is_chimera = true;
  if (bstr != hotspot_queue.current()->info.end()) {
      vector<string>::iterator dot = find(bstr->second.begin(), bstr->second.end(), ".");
      if (dot != bstr->second.end())
        is_chimera = false;
    } else is_chimera = false;
  if (!is_chimera)
      cout << UNIFY_VARIANTS << " Hotspot annotation " << hotspot_queue.current()->sequenceName << ":"
      << hotspot_queue.current()->position << " not found in merged variant file.\n";
  hotspot_queue.next();
}

void PriorityQueue::open_vcf_file(vcf::VariantCallFile& vcf, string filename, bool parse_samples) {
  // Open VCF file for parsing
  if (!enabled) return;
  vcf.parseSamples = parse_samples;
  vcf.open(filename);
  if (!vcf.is_open()) {
    cerr << "ERROR: Could not open file : " << filename << " : " << strerror(errno) << endl;
    exit(1);
  }
  vcf._done = false;
  next();
}

void PriorityQueue::next() {
  if (!enabled) return;
  while (size() < _size) {
    ComparableVcfVariant* v = new ComparableVcfVariant(merger, file, _vc++);
    if (get_next_variant(v)) {
      trim_variant(v);
      left_align_variant(v);
      push(v);
    } else {
      _size = 0;
      delete_variant(v);
      break;
    }
  }
  delete_current();
  if (empty()) {
    _current = NULL;
    return;
  }
  _current = top();

  pop();
}

// ComparableVcfVariant wrapper implementation
bool ComparableVcfVariant::operator<(ComparableVcfVariant& b) {
  int r = merger.variant_cmp(this, &b);
  // StableSorting
  if (r == 0) { r = -comp(_t, b._t); }
  return r == -1;
}

// Common procedures (functions)
void push_value_to_vector(vector<string>& v, long index, const string& entry) {
  if (index < 0 || (unsigned)index >= v.size())
    v.push_back(entry);
  else
    v[index] = entry;
}

template <typename T>
int comp(const T& i1, const T& i2) {
  if (i1 == i2) return 0;
  if (i1 < i2) return 1;
  return -1;
}

template <typename T1, typename T2>
int compare(const pair<T1, T2>& p1, const pair<T1, T2>& p2) {
  int cmp = comp(p1.first, p2.first);
  if (cmp) return cmp;
  return comp(p1.second, p2.second);
}

template <typename T1, typename T2>
int compare(const T1& p11, const T2& p12, const T1& p21, const T2& p22) {
  return compare(make_pair(p11, p12), make_pair(p21, p22));
}

void extend_header(vcf::VariantCallFile &vcf) {
  // extend header with new info fields
  vcf.addHeaderLine("##INFO=<ID=OID,Number=.,Type=String,Description=\"List of original Hotspot IDs\">");
  vcf.addHeaderLine("##INFO=<ID=OPOS,Number=.,Type=Integer,Description=\"List of original allele positions\">");
  vcf.addHeaderLine("##INFO=<ID=OREF,Number=.,Type=String,Description=\"List of original reference bases\">");
  vcf.addHeaderLine("##INFO=<ID=OALT,Number=.,Type=String,Description=\"List of original variant bases\">");
  vcf.addHeaderLine("##INFO=<ID=OMAPALT,Number=.,Type=String,Description="
                        "\"Maps OID,OPOS,OREF,OALT entries to specific ALT alleles\">");
}

void parse_parameters_from_json(string filename, vcf::VariantCallFile& vcf) {
  Json::Value root;
  ifstream in(filename.c_str());
  if (in.is_open()){
    in >> root;
    Json::Value metrics = root.get("metrics", Json::nullValue);
    if (metrics != Json::nullValue) {
      stringstream ss;
      ss << "##deamination_metric=";
      ss << metrics.get("deamination_metric", Json::nullValue).asDouble();
      vcf.addHeaderLine(ss.str());
    }
  }
}

void push_value_to_entry(vcf::Variant& gvcf_entry, size_t value, string key) {
  stringstream ss;
  ss << value;
  push_value_to_entry(gvcf_entry, ss.str(), key);
}

void push_value_to_entry(vcf::Variant& gvcf_entry, string value, string key) {
  push_info_field(gvcf_entry, value, key);
  push_format_field(gvcf_entry, value, key);
}

void push_info_field(vcf::Variant& gvcf_entry, string value, string key) {
  gvcf_entry.info[key].push_back(value);
}

void push_format_field(vcf::Variant& gvcf_entry, string value, string key) {
  for (vector<string>::iterator sn = gvcf_entry.sampleNames.begin(); sn != gvcf_entry.sampleNames.end(); sn++){
    gvcf_entry.samples[*sn][key].push_back(value);
  }
}

bool validate_filename_parameter(string filename, string param_name) {
  if (filename.empty()) {
    cerr << "Parameter " << param_name << " is required and empty.\n";
    return false;
  }
  return true;
}

bool check_on_read(string filename, string param_name) {
  ifstream in(filename.c_str());
  if (!in.is_open()) {
    cerr << "Parameter " << param_name << " filename:" << filename << " does not exist or unreadable.\n";
    return false;
  }
  return true;
}

bool check_on_write(string filename, string param_name) {
  ofstream out(filename.c_str());
  if (!out.is_open()) {
    cerr << "Parameter " << param_name << " filename:" << filename << " file is unwritable.\n";
    return false;
  }
  return true;
}

void build_index(const string &path_to_gz) {
  int tab = ti_index_build(path_to_gz.c_str(), &ti_conf_vcf);
  if (tab == -1) cerr << "Tabix index build failed.";
}

int UnifyVcf(int argc, const char *argv[]) {
  unsigned int DEFAULT_WINDOW_SIZE = 10;
  int DEFAULT_MIN_DP = 0;
  bool DEFAULT_LEFT_ALIGN_FLAG = true;
  string DEFAULT_GZ_VCF_EXT = ".vcf.gz";
  string DEFAULT_GVCF_EXT = ".genome.vcf.gz";

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  string novel_param = "novel-tvc-vcf";
  string novel_vcf = opts.GetFirstString('t', novel_param, "");
  string assembly_param = "novel-assembly-vcf";
  string assembly_vcf = opts.GetFirstString('i', assembly_param, "");
  string hotspot_param = "hotspot-annotation-vcf";
  string hotspot_vcf = opts.GetFirstString('a', hotspot_param, "");
  string reference_param = "reference-fasta";
  string reference_fasta = opts.GetFirstString('f', reference_param, "");
  string bed_param = "target-file";
  string bed_file = opts.GetFirstString('b', bed_param, "");
  string json_param = "tvc-metrics";
  string json_path = opts.GetFirstString('j', json_param, "");
  string out_parameter = "output-vcf";
  string output_vcf = opts.GetFirstString('o', out_parameter, "");
  string out_gvcf_parameter = "output-gvcf";
  string input_depth_param = "input-depth";
  string input_depth = opts.GetFirstString('d', input_depth_param, "");

  int min_depth = opts.GetFirstInt('m', "min-depth", 0);
  int window_size = opts.GetFirstInt('w', "window-size", DEFAULT_WINDOW_SIZE);

  // check and fill optional arguments
  unsigned int w = (window_size < 1) ? DEFAULT_WINDOW_SIZE : (unsigned int) window_size;
  size_t minimum_depth = (size_t) max(DEFAULT_MIN_DP, min_depth);

  opts.CheckNoLeftovers();

  // Validataion on mandatory parameters
  if (!validate_filename_parameter(novel_vcf, novel_param) || !check_on_read(novel_vcf, novel_param)
   || !validate_filename_parameter(reference_fasta, reference_param) || !check_on_read(reference_fasta, reference_param)
   || !validate_filename_parameter(output_vcf, out_parameter) || !check_on_write(output_vcf, out_parameter)
   ) {
    UnifyVcfHelp();
    exit(-1);
  }

  // prepare gvcf output path
  string output_gvcf;
  if (!input_depth.empty()) {
    output_gvcf = output_vcf;
    size_t f = output_gvcf.rfind(DEFAULT_GZ_VCF_EXT);
    if (f != string::npos) output_gvcf.replace(f, DEFAULT_GZ_VCF_EXT.length(), DEFAULT_GVCF_EXT);
    else output_gvcf = "";
  }

  // Validation on optional arguments
  if ((!hotspot_vcf.empty() && !check_on_read(hotspot_vcf, hotspot_param)) ||
      (!assembly_vcf.empty() && !check_on_read(assembly_vcf, hotspot_param)) ||
      (!json_path.empty() && !check_on_read(json_path, json_param)) ||
      (!input_depth.empty() && input_depth != DEFAULT_STDIN_PARAM && !check_on_read(input_depth, input_depth_param)) ||
      (!output_gvcf.empty() && !check_on_write(output_gvcf, out_gvcf_parameter))) {
    exit(-1);
  }
  { // block serves as isolation of merging and building tabix index
    // Create ReferenceReader and TargetsManager objects
    ReferenceReader reference_reader;
    reference_reader.Initialize(reference_fasta);

    TargetsManager targets_namager;
    targets_namager.Initialize(reference_reader, bed_file);

    // Prepare merger object
    VcfOrderedMerger merger(novel_vcf, assembly_vcf, hotspot_vcf, output_vcf, json_path, input_depth, output_gvcf,
                            reference_reader, targets_namager, w, minimum_depth,DEFAULT_LEFT_ALIGN_FLAG);

    // Perform merging procedure
    merger.perform();
  }
  // build tabix indices
  build_index(output_vcf);
  if (!output_gvcf.empty()) build_index(output_gvcf);
  return 0;
}
