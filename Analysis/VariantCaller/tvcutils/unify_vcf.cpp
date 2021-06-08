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
#include <list>
#include <map>
#include <deque>
#include <algorithm>
#include <cstring>
#include <boost/math/distributions/poisson.hpp>
#include <boost/algorithm/string.hpp>
#include "viterbi.h"
#include "ReferenceReader.h"
#include <Variant.h>
#include "TargetsManager.h"

#include "OptArgs.h"
#include "IonVersion.h"
#include "MiscUtil.h"
#include "json/json.h"
#include "unify_vcf.h"

#define DEFAULT_STDIN_PARAM "stdin"
#define UNIFY_VARIANTS "[unify_variants]"
#define GT "GT"
#define DP "DP"
#define MAX_DP "MAX_DP"
#define MIN_DP "MIN_DP"

using namespace std;

// Help function for module

void UnifyVcfHelp()
{
  printf ("\n");
  printf ("tvcutils %s-%s (%s) - Miscellaneous tools used by Torrent Variant Caller plugin and workflow.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   tvcutils unify_vcf [options]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -t,--novel-tvc-vcf             FILE        small variants VCF file produced by TVC (required)\n");
  printf ("  -i,--novel-assembly-vcf        FILE        long indels VCF file produced by TVCASSEMBLY (optional)\n");
  printf ("  -a,--hotspot-annotation-vcf    FILE        hotspot annotations VCF file (optional)\n");
  printf ("  -f,--reference-fasta           FILE        FASTA file containing reference genome (required)\n");
  printf ("  -b,--target-file               FILE        target regions BED file (optional)\n");
  printf ("  -o,--output-vcf                FILE        output merged and annotated VCF or VCF.GZ file (required)\n");
  printf ("  -j,--tvc-metrics               FILE        JSON file with tvc metrics (optional)\n");
  printf ("  -d,--input-depth               FILE        output of samtools depth. if provided, will cause generation of gvcf (optional, stdin ok)\n");
  printf ("  -m,--min-depth                 INT         minimum coverage depth in GVCF output (optional)\n");
  printf("\nVCF record filters:\n");
  printf("      --filter-by-target          on/off      Filter vcf records by meta information in the target bed file [on]\n");
  printf("      --hotspot-positions-only    on/off      Report only hotspot vcf records in final output [off]\n");
  printf("      --hotspot-variants-only     on/off      Suppress hotspot records that are no-calls or reference-calls  [off]\n");
  printf("\nAllele Subset annotation:\n");
  printf("      --subset-check              on/off      Enables or disables subset allele subset annotation [on]\n");
  printf("      --subset-scores       INT,INT,INT,INT   Scores for Smith-Waterman aligner: match,mismatch,gap-open,gap-extend  [1,-3,-5,-2]\n");
  printf("      --subset-simple-mnp         on/off      Simplified (faster) subset check for MNPs. [on]\n");
  printf("      --output-truth-alleles      on/off      If on, use PPD and SPD to determine the true allele from alignment [off]\n");
  printf ("\n");
}

// ---------------------------------------------------------------------------------------

void build_index(const string &path_in) {
  char buffer[33];
  int return_code = 0;
  string path = path_in;
  string path_gz = path_in + ".gz";
  // sort
  vector<string> header;
  map<string, string> lines;
  string line;
  ifstream fin(path.c_str());
  string prev_chr = "";
  int order = 0;
  std::map<string, int> chr_order;
  string chr_key = "";
  string line_key = "";
  if (fin.is_open())
  {
    while (getline(fin, line))
    {
      if ((line.length() > 0) and (line[0] == '#')) {header.push_back(line);}
      else {
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of("\t"));
        if (strs.size() > 1) {
          string chr = strs[0];
		  std::map<string, int>::iterator iter = chr_order.find(chr);
		  if (iter == chr_order.end()) {order++; chr_order[chr] = order;}
		  sprintf(buffer, "%d", chr_order[chr]);
		  chr_key = buffer;
		  while (chr_key.length() < 6) {chr_key = "0" + chr_key;}
          string position = strs[1];
		  while (position.length() < 9) {position = "0" + position;}
		  line_key = "";
		  line_key.reserve(line.length());
		  for (unsigned int index = 2; (index < strs.size()); ++index) {line_key = line_key + "\t" + strs[index];}
		  string key = chr_key + ":" + position + line_key;
          lines[key] = line;
        }
      }
    }
    fin.close();
  }
  ofstream fout;
  fout.open(path.c_str());
  for (vector<string>::iterator iter = header.begin(); (iter != header.end()); ++iter) {
	  fout << *iter << endl;
  }
  for (map<string, string>::iterator iter = lines.begin(); (iter != lines.end()); ++iter) {
    fout << iter->second << endl;
  }
  fout.close();
  // compress
  bgzf_stream* gvcf_out;
  gvcf_out = new bgzf_stream(path_gz);
  for (vector<string>::iterator iter = header.begin(); (iter != header.end()); ++iter) {
	  *gvcf_out << *iter << "\n";
  }
  for (map<string, string>::iterator iter = lines.begin(); (iter != lines.end()); ++iter) {
    *gvcf_out << iter->second << "\n";
  }
  if (gvcf_out) {delete gvcf_out;}
  header.clear();
  lines.clear();
  // index
  int tab = ti_index_build(path_gz.c_str(), &ti_conf_vcf);
  if (tab == -1) {cerr << "build_index failed on tabix. " << return_code << endl;}
}

// ---------------------------------------------------------------------------------------

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

// ===========================================================================
// PriorityQueue methods implementations

bool PriorityQueue::get_next_variant(vcf::Variant* current) {
  // parse VCF line by line
  if (current->vcf->_done) return false;
  bool has_next = current->vcf->getNextVariant(*current);
  if (!has_next) return false;
  if (current->sampleNames.size() > 0) {
    map<string, vector<string> >::iterator a = current->samples[current->sampleNames[0]].find("FR");
    if (a == current->samples[current->sampleNames[0]].end()) {
      a = current->info.find("FR");
      if (a == current->info.end()) {
        current->info["FR"].push_back(".");
      }
    }
  }
  else {
    map<string, vector<string> >::iterator a = current->info.find("FR");
    if (a == current->info.end()) {
      current->info["FR"].push_back(".");
    }
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

void VcfOrderedMerger::right_align_variant(vcf::Variant* variant) {
  if (variant->alt.empty() || variant->alt.size() > 1) return;

  string alt = variant->alt[0];
  if (variant->ref[0] != alt[0]) return;
  int idx = reference_reader.chr_idx(variant->sequenceName.c_str());

  while((int) (variant->position+variant->ref.length()) < reference_reader.chr_size(idx)) {
    char pad = reference_reader.base(idx, variant->position-1 + variant->ref.length());
    string temp_ref = variant->ref.substr(1)+pad;
    string temp_alt = alt.substr(1)+pad;
    if (temp_ref[0] != temp_alt[0]) break;
    alt = temp_alt; variant->ref = temp_ref;

    // left align of hotspots might cause issue of unsync alt and OMAPALT fields
    vector<string>& oalt = variant->info["OMAPALT"];
    if (oalt.size() == 1) {
	oalt[0] = oalt[0].substr(1)+pad;
    }
    variant->position++;
  }
  variant->alt[0] = alt;
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
      //trim_variant(v);
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

// =====================================================================================
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
                                   size_t w = 1, size_t min_dp = 0, bool la = false, bool ota = false)
                                   : depth_in(NULL), gvcf_out(NULL), reference_reader(r), targets_manager(tm),
                                   left_align_enabled(la), use_ppd_spd (ota),
                                   window_size(w), minimum_depth(min_dp),
                                   novel_queue(novel_tvc, *this, r, w, false, true),
                                   assembly_queue(assembly_tvc, *this, r, w, true, true), // left align assembly
                                   hotspot_queue(hotspot_tvc, *this, r, w, la),
                                   bgz_out(output_tvc.c_str()),
                                   current_cov_info(NULL),
                                   filter_by_target(true),
                                   hotspot_positions_only(false),
                                   hotspot_variants_only(false),
                                   num_records(0),
                                   num_filtered_records(0),
                                   num_off_target(0) {
  if (!input_depth.empty()) {
    depth_in = (istream *)((input_depth == DEFAULT_STDIN_PARAM) ? &cin : new ifstream(input_depth.c_str()));
    gvcf_out = new ofstream(gvcf_output.c_str());
    next_cov_entry();
  }
  write_header(novel_queue.variant_call_file(), path_to_json);
  current_target = targets_manager.merged.begin();
}


VcfOrderedMerger::~VcfOrderedMerger() {
  if (depth_in && depth_in != &cin) delete depth_in;
  if (gvcf_out) {gvcf_out->close(); delete gvcf_out;}
}

void VcfOrderedMerger::SetVCFrecordFilters(bool filt_by_target, bool hotspot_pos_only, bool hotspot_var_only)
{
  filter_by_target       = filt_by_target;
  hotspot_positions_only = hotspot_pos_only;
  hotspot_variants_only  = hotspot_var_only;
}

// -----------------------------------------------------------------------------------

template <typename T>
int VcfOrderedMerger::variant_cmp(const T* v1, const vcf::Variant* v2) const {
  int idx_v1 = reference_reader.chr_idx(v1->sequenceName.c_str());
  int idx_v2 = reference_reader.chr_idx(v2->sequenceName.c_str());
  return compare(idx_v1, v1->position, idx_v2, v2->position);
}

template <typename T>
int VcfOrderedMerger::variant_cmp_secplus(const T* v1, const vcf::Variant* v2, int p) const {
  int idx_v1 = reference_reader.chr_idx(v1->sequenceName.c_str());
  int idx_v2 = reference_reader.chr_idx(v2->sequenceName.c_str());
  return compare(idx_v1, v1->position, idx_v2, (long) (v2->position+p+v2->ref.length()));
}



// -----------------------------------------------------------------------------------

bool VcfOrderedMerger::too_far(vcf::Variant* v1, vcf::Variant* v2) {
    if (v2 == NULL) return true;
    int com = variant_cmp_secplus(v2, v1, 50);
    return (com == -1);
    /*
    int far = 250; 
    far = v1->ref.length()+50; // now remove the need of constant. The flip side is that too_far is no longer transitive for list ordered by position
			    // If a->position < b->position and too_far(b, v)=true no longer lead to too_far(a,v)=true. 
			    // The above used to hold true when we use a constant. So this cannot be used as stop condition elsewhere in the code.
    v2->position -= far;
    int com = variant_cmp(v1, v2);
    v2->position += far; // return to the original position
    return (com == 1);
    */
}

bool VcfOrderedMerger::too_far_far(vcf::Variant* v1, vcf::Variant* v2) {
    if (v2 == NULL) return true;
    int com = variant_cmp_secplus(v2, v1, 150);
    return (com == -1);
}


// -----------------------------------------------------------------------------------
// position in vcf is 1-based
// targets in bed format are 0-based open ended intervals
// with index conversion [0,x[ becomes ]0,x]

template<typename T>
bool VcfOrderedMerger::is_within_target_region(T *variant) {
  long pos = variant->position;
  long len = variant->ref.length();
  int chr_idx = reference_reader.chr_idx(variant->sequenceName.c_str());
  while (current_target != targets_manager.merged.end() && chr_idx > current_target->chr) {
    current_target++;
  }
  if (current_target == targets_manager.merged.end() || chr_idx != current_target->chr) return false;
  while (current_target != targets_manager.merged.end() && chr_idx == current_target->chr && pos > current_target->end) {
    current_target++;
  }
  return current_target != targets_manager.merged.end()
         && current_target->chr == chr_idx
         && pos >  current_target->begin
         && pos+len-1 <= current_target->end;
}





// -----------------------------------------------------------------------------------

void VcfOrderedMerger::perform() {
  // merging ordered files loop
  while (novel_queue.has_value() && assembly_queue.has_value()) {
    vcf::Variant* current;
    novel_queue.current()->isHotSpot = false; assembly_queue.current()->isHotSpot = true;
    int cmp = variant_cmp(novel_queue.current(), assembly_queue.current());
    current = cmp == -1 ? assembly_queue.current() : novel_queue.current();
    if (cmp < 0) {
	if (find_and_merge_assembly()) {assembly_queue.next(); continue;}
    } else if (cmp == 0) {
	process_and_write_vcf_entry(current); // novel
	if (not find_and_merge_assembly()) {
	    current = merge_overlapping_variants();
	}
	novel_queue.next(); assembly_queue.next();
	continue;
    }
    process_and_write_vcf_entry(current);
    if (cmp >= 0) novel_queue.next();
    if (cmp <= 0) assembly_queue.next();
  }

  // write out remaining entries
  while (novel_queue.has_value()) {
    novel_queue.current()->isHotSpot = false;
    process_and_write_vcf_entry(novel_queue.current());
    novel_queue.next();
  }
  while (assembly_queue.has_value()) {
    assembly_queue.current()->isHotSpot = true;
    if (not find_and_merge_assembly()) process_and_write_vcf_entry(assembly_queue.current());
    assembly_queue.next();
  }
  while (hotspot_queue.has_value()) {
    //blacklist_check(hotspot_queue.current());
    merge_annotation_into_vcf(hotspot_queue.current());
	hotspot_queue.next();
  }
  flush_vcf(NULL); 
  while (current_target != targets_manager.merged.end()) {
    gvcf_finish_region();
    current_target++;
  }
  cout << "VcfOrderedMerger: Wrote " << num_records << " vcf records, num_filtered_records= "
       << num_filtered_records << " , num_off_target=" << num_off_target << endl;
  allele_subset.print_stats();
}

static bool in_gt(vcf::Variant *v, int i, bool &alt_called)
{
	alt_called = false;
         string gt_field;
            stringstream gt(v->samples[v->sampleNames[0]]["GT"][0]);
            while (getline(gt, gt_field, '/')){
                if (gt_field!="."){
                    int called_allele = std::stoi(gt_field);
                    if (called_allele != 0) alt_called = true;
                    if (called_allele == i) {return true;} 
                }
            }
	return false;
}

static bool is_nonrefcall(vcf::Variant *v)
{
    bool alt_called = false;
    bool x = in_gt(v, 1, alt_called);
    if (x) return true;
    return alt_called;
}

static bool allele_in_gt(vcf::Variant *v, int pos, string a, int &idx, string &ref, const ReferenceReader &reference_reader) 
{
    string vl, vr, left, right;
    int rlen = ref.length();
    int chr = reference_reader.chr_idx(v->sequenceName.c_str());
    if (v->position > pos) {
	vl = reference_reader.substr(chr, pos-1, v->position-pos);
    } else if (v->position < pos) {
	left = reference_reader.substr(chr, v->position-1, pos-v->position);
    }
    if (pos+rlen < (int) (v->position+v->ref.length())) {
	right = reference_reader.substr(chr, pos+rlen-1, v->position+v->ref.length()-pos-rlen);
    } else if (pos+rlen > (int) (v->position+v->ref.length())) {
	vr = reference_reader.substr(chr,v->position+v->ref.length()-1, pos+rlen-v->position-v->ref.length());
    }
    /*
    if (v->position >= pos) {
	if (pos+rlen < (int) v->position) return false;
	if (v->position > pos) vl = ref.substr(0, v->position-pos); 
    } else {
	if ((int) (v->position+v->ref.length()) < pos) return false;
	left = v->ref.substr(0, pos-v->position);
    }
    if (pos+rlen < (int) (v->position+v->ref.length())) {
	int len = v->position+v->ref.length()-(pos+rlen);
	right = v->ref.substr(v->ref.length()-len);
    } else if (pos+rlen >(int) (v->position+v->ref.length())) {
	int len = pos+rlen-(v->position+v->ref.length());
	vr = ref.substr(rlen-len);
    }
    */
    unsigned int i;
    for (i = 0; i < v->alt.size(); i++) {
	//cerr << left+a+right << "  " << vl+v->alt[i]+vr << endl;
	if (left+a+right == vl+v->alt[i]+vr) break;
    }
    if (i == v->alt.size()) return false;
    idx = i+1;
    return true;
}    

bool VcfOrderedMerger::find_and_merge_assembly(vcf::Variant *x, bool use_too_far)
{
    list<vcf::Variant>::reverse_iterator it;
    int pos = x->position, reflen = x->ref.length();
    string alt = x->alt[0];
    //cerr << alt << pos << " "  << reflen << endl;
    for (it = variant_list.rbegin(); it != variant_list.rend(); it++) {
	//cerr << "Z " << it->position << it->ref << endl;
	if (use_too_far) {
	    if (too_far(&(*it), x)) break;
	} else {
	    if (variant_cmp(&(*it), x) >= 0) break;
	}
	//cerr << "Not too far " << endl;
	int idx = 0;
	if (allele_in_gt(&(*it), pos, alt, idx, x->ref, reference_reader)) {
	    bool alt_called = false;
	    if (in_gt(&(*it), idx, alt_called)) return true;
	    continue; // just not replacing for now.
	    /*
	    if (alt_called) continue;
	    // replace
	    return true;
	    */
	}
    }
    return false;
}

bool VcfOrderedMerger::find_and_merge_assembly()
{
    return find_and_merge_assembly(assembly_queue.current(), true);
}

// -----------------------------------------------------------------------------------

void VcfOrderedMerger::write_header(vcf::VariantCallFile& vcf, string json_path) {
  extend_header(vcf, allele_subset.check_enabled, filter_by_target);
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

// -----------------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------------

vcf::Variant* VcfOrderedMerger::merge_overlapping_variants() {
  vcf::Variant& novel_v = *novel_queue.current();
  vcf::Variant& assembly_v = *assembly_queue.current();

  string gt = novel_queue.current()->samples[novel_queue.current()->sampleNames[0]]["GT"][0];
  if (false /*(gt == "./." || gt == "0/0"*/) {
    // Logging novel and indel merge
    cout << UNIFY_VARIANTS " Advanced merge of IndelAssembly variant " << assembly_queue.current()->sequenceName
         << ":" << assembly_queue.current()->position << endl;
  } else {
    //intentanl, always try to put the assembly at a different position with padding/
    list<vcf::Variant>::iterator it;
    int bpback = 1;
    bool at_head = true;
    assembly_queue.current()->position -= 1;
    if (not variant_list.empty()) {
      for (it = variant_list.end(), it--; true; it-- ) {
	if (variant_cmp(&(*it), assembly_queue.current()) != 0) { // found a place;
	    at_head = false;
	    break;
	}
        bpback++; assembly_queue.current()->position -= 1;
     	if (it == variant_list.begin()) break;
      }
    }
    if (assembly_queue.current()->position > 0) {
                //padding
                int chr = reference_reader.chr_idx(assembly_queue.current()->sequenceName.c_str());
                string pad = reference_reader.substr(chr, assembly_queue.current()->position-1 , bpback);
                assembly_queue.current()->ref = pad + assembly_queue.current()->ref;
                assembly_queue.current()->alt[0] = pad +  assembly_queue.current()->alt[0];
                // put the assembly at an empty position
		if (at_head) {
		  generate_novel_annotations(assembly_queue.current());
		  variant_list.push_front(*assembly_queue.current()); 
		} else {
                  it++;
                  generate_novel_annotations(assembly_queue.current());
                  variant_list.insert(it, *assembly_queue.current());
		}
     } else {
                // no empty position
                cout  << UNIFY_VARIANTS " Skipping IndelAssembly variant " << assembly_queue.current()->sequenceName
                 << ":" << assembly_queue.current()->position+bpback << endl;
     }
 

    //cout << UNIFY_VARIANTS " Skipping IndelAssembly variant " << assembly_queue.current()->sequenceName
    //     << ":" << assembly_queue.current()->position << endl;
    return novel_queue.current();
  }
  // for now will not reach this part of code anymore ZZ 1/17/18
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

// -----------------------------------------------------------------------------------

bool VcfOrderedMerger::find_match_new(vcf::Variant* merged_entry, vcf::Variant* hotspot, string *omapalt, int &gt_v, int &allele_reads_count, string &adj_omp, long &idx)
{
    // adjust
    string left, right, ll, rr;
    string mid_ref;
    mid_ref.clear();
    int pos = merged_entry->position;
    if (merged_entry->position > hotspot->position) pos = hotspot->position;
    long end_m = merged_entry->position+merged_entry->ref.length(), end_h = hotspot->position+hotspot->ref.length();

    int length = max(end_m, end_h)-pos;
    if (length > (int) (merged_entry->ref.length()+hotspot->ref.length())) { // not overlaping
    	int chr = reference_reader.chr_idx(hotspot->sequenceName.c_str());
	mid_ref = reference_reader.substr(chr, min(end_m, end_h)-1, length-merged_entry->ref.length()-hotspot->ref.length());
    }
    if ( merged_entry->position > hotspot->position) {
	ll.clear(); left =hotspot->ref.substr(0, merged_entry->position-hotspot->position)+mid_ref;
    } else {
	left.clear(); ll = merged_entry->ref.substr(0, hotspot->position-merged_entry->position)+mid_ref;
    }
    if (end_m > end_h) {
	right.clear(); 
	if (mid_ref.size() > 0) rr = mid_ref+merged_entry->ref;
	else rr = merged_entry->ref.substr(end_h-merged_entry->position);
    } else {
	rr.clear(); 
	if (mid_ref.size() > 0) right = mid_ref+hotspot->ref;
	else right = hotspot->ref.substr(end_m-hotspot->position);
    }
    //if (pos == 7578381 and hotspot->position == 7578383) 
	//cerr << pos << merged_entry->ref << hotspot->position << hotspot->ref << "ll=" << ll << "rr=" << rr << " left=" << left << "right=" << right << "mid=" << mid_ref << endl;
    for (vector<string>::iterator omapalti = merged_entry->info["OMAPALT"].begin(); omapalti != merged_entry->info["OMAPALT"].end(); omapalti++) {
	//if (pos == 7578381 and hotspot->position == 7578383) {
	  //  cerr << *omapalti << " " << *omapalt << endl; 
	//}
	if (left+*omapalti+right == ll+*omapalt+rr) {
	    idx = omapalti-merged_entry->info["OMAPALT"].begin()+1;
  	    string gt_field;
  	    stringstream gt(merged_entry->samples[merged_entry->sampleNames[0]]["GT"][0]);
	    gt_v = 4;
  	    while (getline(gt, gt_field, '/')){
		//if (pos == 7578381 and hotspot->position == 7578383) cerr << "gt_field=" << gt_field << endl;
    	    	if (gt_field!="."){
      		    int called_allele = std::stoi(gt_field); // Zero based index for vector access
		    if (called_allele == idx) {gt_v = 1;break;} // 0/A, A/A, A/B
		    else if (called_allele == 0) gt_v = 2; // 0/0
		    else gt_v = 3; // 0/B
		}
	    }
	    //if (pos == 7578381 and hotspot->position == 7578383)  cerr << "gt_v=" <<  gt_v << " " << idx << endl;
	    allele_reads_count = 0;
	    adj_omp = *omapalti;
	    return true;
	}
    }
    return false;
}

void set_annotate(vcf::Variant* merged_entry,vector<string>::iterator oid, vector<string>::iterator opos, vector<string>::iterator oref, vector<string>::iterator oalt, string &adj_omp, long idx)
{
    if (oref->empty()) {
      *oref = "-";
    }
    if(oalt->empty()) {
      *oalt = "-";
    }
    if (merged_entry->info["OID"][idx] == ".") {
      merged_entry->info["OID"][idx] = *oid;
      merged_entry->info["OPOS"][idx] = *opos;
      merged_entry->info["OREF"][idx] = *oref;
      merged_entry->info["OALT"][idx] = *oalt;
      merged_entry->info["OMAPALT"][idx] = /**omapalt*/ adj_omp;
    } else {
      merged_entry->info["OID"].push_back(*oid);
      merged_entry->info["OPOS"].push_back(*opos);
      merged_entry->info["OREF"].push_back(*oref);
      merged_entry->info["OALT"].push_back(*oalt);
      merged_entry->info["OMAPALT"].push_back(adj_omp /**omapalt*/);
    }
    if (merged_entry->id.size() == 0 or merged_entry->id == ".") merged_entry->id = *oid;
    else merged_entry->id += ";"+ *oid;
}

bool VcfOrderedMerger::find_match(vcf::Variant* merged_entry, string &hotspot_ref,vector<string>::iterator oid, vector<string>::iterator opos, vector<string>::iterator oref, vector<string>::iterator oalt, string *omapalt, int record_ref_extension, string &annotation_ref_extension) {

    if (merged_entry == NULL) return false;
    string adj_omp;
    if (record_ref_extension) {
      if ((int)omapalt->length() < record_ref_extension || hotspot_ref.substr(hotspot_ref.length() - record_ref_extension) !=
          omapalt->substr(omapalt->length() - record_ref_extension)) {
	return false;
        cout << UNIFY_VARIANTS << " Hotspot annotation " << merged_entry->sequenceName
        << ":" << merged_entry->position << ", allele " << *omapalt << " not eligible for shortening.\n";
      }
      adj_omp = omapalt->substr(0, omapalt->length() - record_ref_extension);
    } else adj_omp = *omapalt;
    //*omapalt = *omapalt + annotation_ref_extension;
    adj_omp = adj_omp+annotation_ref_extension;

    vector<string>::iterator omapalti =
        find(merged_entry->info["OMAPALT"].begin(), merged_entry->info["OMAPALT"].end(), adj_omp);

    if (omapalti == merged_entry->info["OMAPALT"].end()) {
      return false;
        // match older records.
      cout << UNIFY_VARIANTS << " Hotspot annotation " << merged_entry->sequenceName
      << ":" << merged_entry->position << ", allele " << *omapalt << " not found in merged variant file.\n";
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
      merged_entry->info["OMAPALT"][idx] = /**omapalt*/ adj_omp;
    } else {
      merged_entry->info["OID"].push_back(*oid);
      merged_entry->info["OPOS"].push_back(*opos);
      merged_entry->info["OREF"].push_back(*oref);
      merged_entry->info["OALT"].push_back(*oalt);
      merged_entry->info["OMAPALT"].push_back(adj_omp /**omapalt*/);
    }
    if (merged_entry->id.size() == 0 or merged_entry->id == ".") merged_entry->id = *oid;
    else merged_entry->id += ";"+ *oid;
    return true;
}

// -----------------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------------

void VcfOrderedMerger::generate_novel_annotations(vcf::Variant* variant) {
  if (variant->alt.empty()) return;
  bool use_PPD_SPD = false;
  if (use_ppd_spd and variant->info.find("PPD") != variant->info.end() and variant->info.find("SPD") !=  variant->info.end()) {
	use_PPD_SPD = true;
  }
  for (vector<string>::iterator alt = variant->alt.begin(); alt != variant->alt.end(); alt++) {
    long i = alt - variant->alt.begin();
    long opos = variant->position;
    string oref = variant->ref;
    string temp_alt = *alt;
    if (use_PPD_SPD) {
	int x = stoi(variant->info["PPD"][i]);
	int y = stoi(variant->info["SPD"][i]);
	opos += x;
	int len = oref.size()-x-y;
	oref = oref.substr(x, len);
	len = temp_alt.size()-x-y;
	temp_alt = temp_alt.substr(x, len);	
    } else {
    // trim identical ends
    string::iterator orefi = oref.end();
    string::iterator alti = temp_alt.end();
    for (; *orefi == *alti && orefi != oref.begin() && alti != temp_alt.begin(); orefi--, alti--) { }

    if (alti + 1 < temp_alt.end()) temp_alt.erase(alti + 1, temp_alt.end());
    if (orefi + 1 < oref.end()) oref.erase(orefi + 1, oref.end());
    // trim identical beginnings
    orefi = oref.begin();
    alti = temp_alt.begin();
    for (; *orefi == *alti && *alti != '-' and alti < temp_alt.end() and orefi < oref.end(); orefi++, alti++) {
      opos++;
    }
    if (distance(temp_alt.begin(), alti) > 0) temp_alt.erase(temp_alt.begin(), alti);
    if (distance(oref.begin(), orefi) > 0) oref.erase(oref.begin(), orefi);
    }
    if (oref.empty()) oref = "-";
    if (temp_alt.empty()) temp_alt = "-";
    stringstream ss;
    ss<<opos;

    push_value_to_vector(variant->info["UFR"], i , ".");
    push_value_to_vector(variant->info["OID"], i, ".");
    push_value_to_vector(variant->info["OPOS"], i, ss.str());
    push_value_to_vector(variant->info["OREF"], i, oref);
    push_value_to_vector(variant->info["OALT"], i, temp_alt);
    push_value_to_vector(variant->info["OMAPALT"], i, *alt);
    // We only write a subset info field if there actually are subsets
  }

}


// -----------------------------------------------------------------------------------
// implementation of subset check  and annotation XXX

void VcfOrderedMerger::genotype_to_set(std::set<int> &gt_set, const string &gt_str, char delim)
{
  string gt_field;
  stringstream mysstream(gt_str);
  while (getline(mysstream, gt_field, delim)){
    if (gt_field!="." and gt_field!="0"){
      int called_allele = std::stoi(gt_field)-1; // Zero based index for vector access
      gt_set.insert(called_allele);
    }
  }
}

void VcfOrderedMerger::annotate_subset(vcf::Variant* variant) {

  if (not allele_subset.check_enabled)
    return;
  // Immediately return if this is a NOCALL or not a multi allele variant
  if (variant->filter != "PASS" or variant->alt.size()<2)
    return;

  // unpack called alleles -- First genotype then PPA
  set<int> called_alts;
  string gt_str = variant->samples[variant->sampleNames[0]]["GT"][0];
  genotype_to_set(called_alts, gt_str, '/');

  map<string, vector<string> >::iterator it;
  it = variant->info.find("PPA");
  if (it != variant->info.end()) {
    gt_str = variant->info["PPA"][0];
    genotype_to_set(called_alts, gt_str, '/');
  }

  if (called_alts.size()==0)
    return;

  // We actually have something to test
  vector<string> subset_info;
  bool have_subsets = false;

  // For every alt allele we test it is a strict subset of any called alt alleles
  for (unsigned int aidx=0; aidx<variant->alt.size(); ++aidx){
    string si_field;
    for (std::set<int>::iterator calt=called_alts.begin(); calt!=called_alts.end(); ++calt){
      if ((int)aidx == *calt)
        continue;
      if (allele_subset.is_allele_subset(variant->ref, variant->alt[aidx], variant->alt[*calt])){
        // Option to not label symmetric subsets goes here
        if (allele_subset.ignore_symmetric_subsets and
            allele_subset.is_allele_subset(variant->ref, variant->alt[*calt], variant->alt[aidx]))
        {
          if (allele_subset.debug){
            cout << "IGNORING Ref: " << variant->ref
                      << " Subset: " << variant->alt[aidx]
                    << " Superset: " << variant->alt[*calt] << endl << endl;
          }
          continue;
        }
        else {
          have_subsets = true;
          if (not si_field.empty())
            si_field += "/";
          si_field += std::to_string(*calt+1); // In vcf, make it 1-based again
        }
      }
    }

    if (si_field.empty())
      push_value_to_vector(subset_info, aidx, ".");
    else
      push_value_to_vector(subset_info, aidx, si_field);
  }

  // And if we found subsets we annotate them in the info field
  if (have_subsets)
    variant->info["SUBSET"] = subset_info;
}

// -----------------------------------------------------------------------------------

static bool better(int g, int c, int g1, int c1)
{
    if (g < g1) return false;
    if (g > g1) return true;
    return (c< c1);
}

void VcfOrderedMerger::merge_annotation_into_vcf(vcf::Variant* hotspot)
{
  string annotation_ref_extension;
  long record_ref_extension = 0;

  map<string, string> blacklist;
  map<string, vector<string> >::iterator bstr = hotspot->info.find("BSTRAND");
  if (bstr != hotspot->info.end())
    for (vector<string>::iterator key = hotspot->alt.begin(), v = bstr->second.begin();
         key != hotspot->alt.end() && v != bstr->second.end(); key++, v++) {
      blacklist[*key] = *v;
    }
  for (vector<string>::iterator oid = hotspot->info["OID"].begin(),
           opos = hotspot->info["OPOS"].begin(),
           oref = hotspot->info["OREF"].begin(), oalt = hotspot->info["OALT"].begin(),
           omapalt = hotspot->info["OMAPALT"].begin();
       oid != hotspot->info["OID"].end() && opos != hotspot->info["OPOS"].end() &&
       oref != hotspot->info["OREF"].end() && oalt != hotspot->info["OALT"].end() &&
       omapalt != hotspot->info["OMAPALT"].end();
       oid++, opos++, oref++, oalt++, omapalt++) {
        if (!blacklist.empty() && blacklist[*omapalt] != ".")
          continue;
        bool found = false;
	int gt_v, allele_reads_count;
	int s_gt, s_ac=0;
	long idx, sidx=0;
	string omp, adj_omp; 
        list<vcf::Variant>::reverse_iterator it;
	vcf::Variant *sv = NULL;
        for (it = variant_list.rbegin(); it != variant_list.rend(); it++ ) {
            if (too_far(&(*it), hotspot)) continue; // not assume well ordered.
            if (find_match_new(&(*it), hotspot, &(*omapalt), gt_v, allele_reads_count, adj_omp, idx)) {
		if (it->isHotSpot) allele_reads_count = -1;
		if (not found or (better(s_gt, s_ac, gt_v, allele_reads_count))) {
		    if (found) {
			if (sv->infoFlags.find("HS") != sv->infoFlags.end()) {
			    sv->info["UFR"][sidx-1] = "unknown";
			}
		    }
		    found = true; s_gt = gt_v; sidx = idx, omp = adj_omp; s_ac = allele_reads_count; sv = &(*it);
		} else if (it->infoFlags.find("HS") != sv->infoFlags.end()) {
		    it->info["UFR"][idx-1] = "unknown";
		} 
	    }
        }
        if (not found){
            cout << UNIFY_VARIANTS << " Hotspot annotation " << hotspot->sequenceName
            << ":" << hotspot->position << ", allele " << *omapalt << " not found in merged variant file.\n";
        } else {
	    if (sv->infoFlags.find("HS") == sv->infoFlags.end()) {
		sv->info["UFR"][sidx-1] = "hs";
	    }
	    set_annotate(sv, oid, opos, oref, oalt, omp, sidx-1);
	}
  }
}



void VcfOrderedMerger::merge_annotation_into_vcf(vcf::Variant* merged_entry, vcf::Variant* hotspot) {
  string annotation_ref_extension;
  long record_ref_extension = 0;
  if (merged_entry) {
    if (hotspot->ref.length() > merged_entry->ref.length()) {
      record_ref_extension = hotspot->ref.length() - merged_entry->ref.length();
    }
    if (hotspot->ref.length() < merged_entry->ref.length()) {
      annotation_ref_extension = merged_entry->ref.substr(hotspot->ref.length());
    }
  }

  // Form blacklist
  map<string, string> blacklist;
  map<string, vector<string> >::iterator bstr = hotspot->info.find("BSTRAND");
  if (bstr != hotspot->info.end())
    for (vector<string>::iterator key = hotspot->alt.begin(), v = bstr->second.begin();
         key != hotspot->alt.end() && v != bstr->second.end(); key++, v++) {
      blacklist[*key] = *v;
    }

  //vector<string> filtered_oid;

  for (vector<string>::iterator oid = hotspot->info["OID"].begin(),
           opos = hotspot->info["OPOS"].begin(),
           oref = hotspot->info["OREF"].begin(), oalt = hotspot->info["OALT"].begin(),
           omapalt = hotspot->info["OMAPALT"].begin();
       oid != hotspot->info["OID"].end() && opos != hotspot->info["OPOS"].end() &&
       oref != hotspot->info["OREF"].end() && oalt != hotspot->info["OALT"].end() &&
       omapalt != hotspot->info["OMAPALT"].end();
       oid++, opos++, oref++, oalt++, omapalt++) {
    if (!blacklist.empty() && blacklist[*omapalt] != ".")
      continue;
    //if (merged_entry) filtered_oid.push_back(*oid);
    //ZZ
    if (not find_match(merged_entry, hotspot->ref,oid, opos, oref, oalt, &(*omapalt), record_ref_extension, annotation_ref_extension)) {
	list<vcf::Variant>::reverse_iterator it;
	bool found = false;
	for (it = variant_list.rbegin(); it != variant_list.rend(); it++ ) {
	    if (too_far(&(*it), hotspot)) continue; // not assume well ordered.
	    int padding = hotspot->position-it->position;
	    if (padding < 0) break;
	    if (padding > (int) it->ref.length()) continue; // not contain
	    long record_ref_ext = 0; // new 
	    string annotation_ref_ext; // new
	    string x = it->ref.substr(0,padding)+*omapalt;
	    unsigned int rlen = padding + hotspot->ref.length();
	    string hotspot_ref = it->ref.substr(0,padding)+hotspot->ref;
 	    if (rlen > it->ref.length()) {
      		record_ref_ext = rlen - it->ref.length();
    	    }
    	    if (rlen < it->ref.length()) {
      		annotation_ref_ext = it->ref.substr(rlen);
    	    }
	    if (find_match(&(*it), hotspot_ref, oid, opos, oref, oalt, &x, record_ref_ext, annotation_ref_ext)) {found = true; break;}
	}
	if (not found){ 
	    cout << UNIFY_VARIANTS << " Hotspot annotation " << hotspot->sequenceName
            << ":" << hotspot->position << ", allele " << *omapalt << " not found in merged variant file.\n";
	}
    }
  }
}

// -----------------------------------------------------------------------------------

bool VcfOrderedMerger::filter_VCF_record(vcf::Variant* record) const
{
  bool is_HS = record->infoFlags.find("HS") != record->infoFlags.end();

  // Hotspot positions only filter applied first
  if (hotspot_positions_only and not is_HS)
    return true;

  // Next we filter by target meta data
  if (filter_by_target){
    // vcf->position is 1-based, whereas target bed fiels are 0-based indices.
    int hs_only = targets_manager.ReportHotspotsOnly(*current_target, reference_reader.chr_idx(record->sequenceName.c_str()), record->position-1);
    // In case filtering by target is enabled, we add an info field indicating the target meta information
    record->info["HS_ONLY"].clear();
    record->info["HS_ONLY"].push_back(convertToString(hs_only));
    if (hs_only > 0 and not is_HS)
      return true;
  }

  // Finally filter hotspot lines that are REF of NOCALL
  if (hotspot_variants_only and is_HS){
    // 1) Remove NOCALLs
    if (record->filter != "PASS")
      return true;

    // 2) Remove reference calls
    bool ref_call = true;
    string gt_field;
    for (vector<string>::const_iterator its = record->sampleNames.begin(); its != record->sampleNames.end(); ++its) {
      if (not ref_call)
        break;
      map<string, vector<string> > & sampleOutput = record->samples[*its];
      map<string, vector<string> >::const_iterator itg = sampleOutput.find("GT");
      if (itg == sampleOutput.end())
        return false;

      for (vector<string>::const_iterator itv = itg->second.begin(); itv!=itg->second.end(); ++itv){
        stringstream ss(*itv);
        while (getline(ss, gt_field, '/')){
          if (gt_field == "0" or gt_field == ".")
            continue;
          else
            ref_call = false;
        }
      }
    }
    return ref_call;
  }

  return false;
}

// -----------------------------------------------------------------------------------
// XXX Code below writes lines to the vcf files

void VcfOrderedMerger::flush_vcf(vcf::Variant* latest)
{
  while (not variant_list.empty()) {
    vcf::Variant* current = &(*variant_list.begin());

    if (too_far_far(current, latest)) {
      if (is_within_target_region(current)) { 
	if (current->isHotSpot and  find_and_merge_assembly(current, false)) {
	    variant_list.pop_front();
	    continue;
	}
	unsigned int i;
        bool has_oid = false;
        for (i =0; i < current->info["OID"].size(); i++) {
              if (current->info["OID"][i] != ".") {has_oid = true; break;}
        }
	bool novel_refnocall = false;
        if (current->infoFlags.find("HS") != current->infoFlags.end()) {
                if (not has_oid) {
		    current->infoFlags.erase("HS");  // remove HS flag if all OID's are dot.
		    if (not is_nonrefcall(current))novel_refnocall = true; // when HS is removed, check if it is no call
		}
        } else {
                if (has_oid) current->infoFlags["HS"] = true;
        }

        if (novel_refnocall or filter_VCF_record(current))
          ++num_filtered_records;
        else { // Write out record if not filtered
	  /*
	  if (not has_oid) {
	      current->info["OID"].clear();
      	      current->info["OPOS"].clear();
              current->info["OREF"].clear();
              current->info["OALT"].clear();
              current->info["OMAPALT"].clear();
	  }
	  */

	  bool has_ufr = false;
	  for (i =0; i < current->info["UFR"].size(); i++) 
		if (current->info["UFR"][i] != ".") {has_ufr = true; break;}
	  if (not has_ufr) current->info["UFR"].clear();
    	  gvcf_process(current->sequenceName.c_str(), current->position);
    	  gvcf_out_variant(current);
    	  bgz_out  << *current << "\n";
    	  ++num_records;
        }
      } else {
	bool off_targ = true;
	long pos  = current->position;
        if (current->isHotSpot and (not find_and_merge_assembly(current, false))) {
	    right_align_variant(current);
	    if (current->position != pos) {
	      list<vcf::Variant>::iterator it;
	      current->isHotSpot = false;
	      for (it = variant_list.begin(), it++; it != variant_list.end(); it++) {
		if (variant_cmp(&(*it), current) <= 0) break;
	      } 
	      if (it ==  variant_list.end() or variant_cmp(&(*it), current) != 0) {
		vcf::Variant new_one = *current;
		if (it ==  variant_list.end()) variant_list.push_back(new_one);
		else variant_list.insert(it, new_one);
		off_targ = false; // modified, and see if it is on target
	      }
	    }
	}
	if (off_targ) {
          ++num_off_target;
          cout << UNIFY_VARIANTS " Skipping " << current->sequenceName << ":" << pos
             << " outside of target regions.\n";
	}
      }
      variant_list.pop_front();
    } else break;
  }
  if (latest)
       variant_list.push_back(*latest);
}

// -----------------------------------------------------------------------------------

void VcfOrderedMerger::process_and_write_vcf_entry(vcf::Variant* current) {
  generate_novel_annotations(current);
  process_annotation(current);  // Adds hotspot annotation to entry
  annotate_subset(current);     // Checks if called alleles supersets of others
  flush_vcf(current);           // Filters entries and writes vcf files to file
  return;
}

// -----------------------------------------------------------------------------------

void VcfOrderedMerger::gvcf_out_variant(vcf::Variant *current) {
  if (gvcf_out) {
      int chr = reference_reader.chr_idx(current->sequenceName.c_str());
      while (current_cov_info && current_cov_info->chr() == chr && current_cov_info->position - current->position < (int)current->ref.length()) next_cov_entry();
      *gvcf_out << *current << "\n";
    }
}

// -----------------------------------------------------------------------------------
// Generates coverage lines in gvcf file

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

// -----------------------------------------------------------------------------------

void VcfOrderedMerger::gvcf_finish_region() {
  gvcf_process(current_target->chr, current_target->end + 1);
}

void VcfOrderedMerger::gvcf_process(const char * seq_name, long pos) {
  gvcf_process(reference_reader.chr_idx(seq_name), pos);
}

// -----------------------------------------------------------------------------------

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
  gvcf_entry.ref = reference_reader.base(chr, pos - 1);

  push_format_field(gvcf_entry, "0/0", gt);

  push_value_to_entry(gvcf_entry, depth.dp, dp);
  push_value_to_entry(gvcf_entry, depth.min_dp, min_dp);
  push_value_to_entry(gvcf_entry, depth.max_dp, max_dp);

  return gvcf_entry;
}

// -----------------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------------

void VcfOrderedMerger::process_annotation(vcf::Variant* current) {

  int cmp;
  while (hotspot_queue.has_value()) {
      if (not too_far(hotspot_queue.current(), current)) return;
      merge_annotation_into_vcf(hotspot_queue.current()) ;
      //blacklist_check(hotspot_queue.current()); // does not need, it is checked in above
      hotspot_queue.next();
  }
}
  
// -----------------------------------------------------------------------------------

void VcfOrderedMerger::blacklist_check(vcf::Variant* hotspot) {
  map<string, vector<string> >::iterator bstr = hotspot->info.find("BSTRAND");
  bool is_chimera = true;
  if (bstr != hotspot->info.end()) {
      vector<string>::iterator dot = find(bstr->second.begin(), bstr->second.end(), ".");
      if (dot != bstr->second.end())
        is_chimera = false;
    } else is_chimera = false;
	if (!is_chimera)
		cout << UNIFY_VARIANTS << " Hotspot annotation " << hotspot->sequenceName << ":"
			<< hotspot->position << " not found in merged variant file.\n";
}

// ================================================================================
// ComparableVcfVariant wrapper implementation

bool ComparableVcfVariant::operator<(ComparableVcfVariant& b) {
  int r = merger.variant_cmp(this, &b);
  // StableSorting
  if (r == 0) { 
    //ZZ favor HS tag
    int x = 1, y= 1;
    if (this->infoFlags.find("HS") == this->infoFlags.end()) x = 0;
    if (b.infoFlags.find("HS")== b.infoFlags.end())  y = 0;
    if (x == y) r = -comp(_t, b._t); 
    else if (x < y) r = -1;
    else r = 1;
  }
  return r == -1;
}

// ================================================================================
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

void extend_header(vcf::VariantCallFile &vcf, bool add_subset, bool filter_by_target) {
  // extend header with new info fields
  vcf.addHeaderLine("##INFO=<ID=OID,Number=.,Type=String,Description=\"List of original Hotspot IDs\">");
  vcf.addHeaderLine("##INFO=<ID=OPOS,Number=.,Type=Integer,Description=\"List of original allele positions\">");
  vcf.addHeaderLine("##INFO=<ID=OREF,Number=.,Type=String,Description=\"List of original reference bases\">");
  vcf.addHeaderLine("##INFO=<ID=OALT,Number=.,Type=String,Description=\"List of original variant bases\">");
  vcf.addHeaderLine("##INFO=<ID=OMAPALT,Number=.,Type=String,Description="
                        "\"Maps OID,OPOS,OREF,OALT entries to specific ALT alleles\">");
  vcf.addHeaderLine("##INFO=<ID=UFR,Number=.,Type=String,Description=\"List of ALT alleles with Unexpected Filter thresholds\">");
  if (add_subset)
    vcf.addHeaderLine("##INFO=<ID=SUBSET,Number=A,Type=String,Description="
                        "\"1-based index in ALT list of genotyped allele(s) that are a strict superset\">");
  if (filter_by_target)
    vcf.addHeaderLine("##INFO=<ID=HS_ONLY,Number=1,Type=Integer,Description=\"Corresponding value from target file meta information\">");
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

// ==============================================================================
// XXX Main function

int UnifyVcf(int argc, const char *argv[]) {
  unsigned int DEFAULT_WINDOW_SIZE = 10;
  int DEFAULT_MIN_DP = 0;
  bool DEFAULT_LEFT_ALIGN_FLAG = true;
  string DEFAULT_GZ_VCF_EXT = ".vcf";
  string DEFAULT_GVCF_EXT = ".genome.vcf";

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

  int  min_depth   = opts.GetFirstInt('m', "min-depth", 0);
  int  window_size = opts.GetFirstInt('w', "window-size", DEFAULT_WINDOW_SIZE);

  // VCF record filter settings
  bool  filter_by_target         = opts.GetFirstBoolean ('-', "filter-by-target",       true);
  bool  hotspot_positions_only   = opts.GetFirstBoolean ('-', "hotspot-positions-only", false);
  bool  hotspot_variants_only    = opts.GetFirstBoolean ('-', "hotspot-variants-only",  false);

  // Subset annotation options
  bool subset_debug              = opts.GetFirstBoolean  ('-', "subset-debug",       false);
  bool check_for_subsets         = opts.GetFirstBoolean  ('-', "subset-check",       true);
  bool subset_simple_mnp         = opts.GetFirstBoolean  ('-', "subset-simple-mnp",  true);
  bool ignore_symmetric_subsets  = opts.GetFirstBoolean  ('-', "subset-ignore-symmetric",  true);
  vector<int> subset_scores      = opts.GetFirstIntVector('-', "subset-scores", "1,-3,-5,-2");
  bool true_alleles 		 = opts.GetFirstBoolean  ('-', "output-truth-alleles",  false);

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
                            reference_reader, targets_namager, w, minimum_depth,DEFAULT_LEFT_ALIGN_FLAG, true_alleles);
    // Set filtering options
    merger.SetVCFrecordFilters(filter_by_target, hotspot_positions_only, hotspot_variants_only);
    // Set subset annotation options
    merger.allele_subset.debug                    = subset_debug;
    merger.allele_subset.check_enabled            = check_for_subsets;
    merger.allele_subset.simple_mnp_alignment     = subset_simple_mnp;
    merger.allele_subset.ignore_symmetric_subsets = ignore_symmetric_subsets;
    merger.allele_subset.SetAlignerScores(subset_scores);

    // Perform merging procedure
    merger.perform();
  }
  // build tabix indices
  build_index(output_vcf);
  if (!output_gvcf.empty())
    build_index(output_gvcf);
  return 0;
}

// ==============================================================================
// class AlleleSubsetCheck contains functionality to identify alleles that are strict
// subsets of other alleles
// The method to search for a subset is a two stage conditional multiple sequence alignment
// It's suboptimal compared to a full 3D multiple sequence alignment, but faster.
// TODO: explore vcflib Smith-Waterman instead of Realigner

AlleleSubsetCheck::AlleleSubsetCheck() :
    aligner_(50,10),check_enabled(true), simple_mnp_alignment(true), ignore_symmetric_subsets(false)
{
  // Default to tmap default scores for alignment
  vector<int> def_scores(4, 1);
  def_scores[1] = -3; // mismatch penalty
  def_scores[2] = -5; // gap open penalty
  def_scores[3] = -2; // gap extend penalty
  aligner_.SetScores(def_scores);

  // We do not allow any clipping and we only align in forward direction
  aligner_.SetClipping(0, true);

  reset_counters();
  failure_ = false;
  debug    = false;
}

// --------------------------------------------------------------------
// A fast simplified subset check for MNPs that only allows matches and mismatches as alignment operations
// Otherwise same conditional 2-stage alignment approach as in conditional_alignment_subset_check()
bool AlleleSubsetCheck::mnp_subset_check(const string & ref, const string &subset, const string &super)
{
  int n_edits_ref = 0;
  int n_edits_sup = 0;
  bool rmatch, smatch;

  for (unsigned int i=0; i< subset.length(); ++i){
    rmatch = ref[i]==subset[i];
    smatch = subset[i]==super[i];

    if (not rmatch) {
      ++n_edits_ref;
      if (not smatch)
        return false;
    }
    else if (not smatch)
      ++n_edits_sup;
  }
  return (n_edits_ref>0 and n_edits_sup>0);
}

// --------------------------------------------------------------------
// Check for M cigar operations which can be matches or mismatches

bool AlleleSubsetCheck::is_match(const string & pretty_a, int aidx)
{
  if (aidx>=0 and aidx<(int)pretty_a.length()){
    return (pretty_a[aidx]=='|' or pretty_a[aidx]==' ');
  }
  else {
    return true; // To get boundary conditions at edges right
  }
}

bool AlleleSubsetCheck::have_matches(const string & pretty_1, int idx1, const string & pretty_2, int idx2)
{
  return (is_match(pretty_1, idx1) and is_match(pretty_2, idx2));
}

// --------------------------------------------------------------------
// A fast MNP alignment that only allows matches and mismatches as operations

string AlleleSubsetCheck::get_mnp_pretty_align(const string & allele1, const string & allele2)
{
  if (allele1.length() != allele2.length())
    return "";
  string pretty_a(allele1.length(), '|');
  for (unsigned int i=0; i<allele1.length(); ++i)
    if (allele1[i] != allele2[i])
      pretty_a[i] = ' ';

  return pretty_a;
}

// --------------------------------------------------------------------
// Interface to the outside world and accounting

bool AlleleSubsetCheck::is_allele_subset(const string & ref, const string &subset, const string &super)
{
  if (simple_mnp_alignment and ref.length()==subset.length() and subset.length() == super.length()){
    ++counter_num_mnp_checks;
    if (mnp_subset_check(ref, subset, super)){
      ++counter_num_mnp_subsets;
      return true;
    }
    else
      return false;
  }
  else{
    ++counter_num_align_checks;
    if (conditional_alignment_subset_check(ref, subset, super)){
      ++counter_num_align_subsets;
      return true;
    }
    else
      return false;
  }
}

// --------------------------------------------------------------------
// Exclude a few more entries to avoid calling the aligner too often
// Do a mnp-alignment along ref-superset anchors

bool AlleleSubsetCheck::anchor_sanity_check(const string & ref, const string &subset, const string &super)
{
  unsigned int min_length = min(ref.length(), subset.length());
  min_length = min(min_length, (unsigned int)super.length());

  // Forward check
  unsigned int i=0;
  while (i<min_length and ref[i]==super[i]){
    if (ref[i]!=subset[i])
      return false;
    ++i;
  }

  // Reverse check
  i=1;
  while (i<=min_length and ref[ref.length()-i]==super[super.length()-i]){
    if (ref[ref.length()-i]!=subset[subset.length()-i])
      return false;
    ++i;
  }

  return true;
}

// --------------------------------------------------------------------
// This is the core method of this class to obtain a subset/superset decomposition
// mnp alignment or anchor check are not strictly necessary but only reduce complexity

bool AlleleSubsetCheck::conditional_alignment_subset_check(const string & ref, const string &subset, const string &superset)
{
  if (not anchor_sanity_check(ref, subset, superset))
    return false;

  // Fist stage, get <target> ref - <query> subset alignment

  unsigned int start_position_shift;
  string pretty_ref;
  if (simple_mnp_alignment and ref.length()==subset.length()){
    pretty_ref = get_mnp_pretty_align(ref, subset);
  }
  else{
    aligner_.SetSequences(subset, ref, pretty_ref ,true);
    aligner_.computeSWalignment(cigar_data, md_data, start_position_shift);
    pretty_ref = aligner_.pretty_aln();
  }

  // Second stage, get <target> subset - <query> superset alignment

  string pretty_sup;
  if (simple_mnp_alignment and subset.length()==superset.length()){
    pretty_sup = get_mnp_pretty_align(subset, superset);
  }
  else{
    aligner_.SetSequences(superset, subset, pretty_sup ,true);
    aligner_.computeSWalignment(cigar_data, md_data, start_position_shift);
    pretty_sup = aligner_.pretty_aln();
  }

  // Reconcile alignments to get conditional superset alignment
  string cond_super_aln = get_cond_superset_alignment(pretty_ref, pretty_sup);

  // Debug break point here: XXX
  if (debug and (failure_ or (cond_super_aln.length() > 0))) {
    if (failure_)
      cout << "FAILURE for Ref: " << ref << " \"" << pretty_ref <<  "\" Subset: " << subset << " \"" << pretty_sup << "\" Superset: " << superset << endl;
    else
      cout << "SUPERSET for Ref: " << ref << " \"" << pretty_ref <<  "\" Subset: " << subset << " \"" << pretty_sup << "\" Superset: " << superset << endl;
    cout << "Conditional alignment: Ref: "  << ref << " Superset: " << superset << " Pretty: \"" << cond_super_aln << "\"" << endl << endl;
    //string dinput;
    //getline(cin, dinput);
    //cout << dinput << endl;
    failure_ = false;
  }
  //*/


  if (cond_super_aln.length() > 0)
    return true;
  else
    return false;
}

// --------------------------------------------------------------------
// Function reconciles the two alignment stages and creates a conditional
// ref<->superset alignment if possible. Otherwise returns an empty string.
//
// With events match='|' mismatch=' ' M=' 'or'|' insertion='+' deletion='-',
// to have a valid subset decomposition with independent, disjoint events
// if we only allow the following operations:
// (extension of a gap is NOT a superset of the shorter gap)
//
//      Ref:t <-> q:Sub:t <-> q:Super
//             |           M
//            ' '          |
//           M|||M       M---M
//           M---M       flanking M
//      flanking M       M+++M
//           M+++M       M|||M


string AlleleSubsetCheck::get_cond_superset_alignment(const string & pretty_ref, const string &pretty_super)
{
  string super_aln;
  int ridx = 0;
  int sidx = 0;

  // Iterate over the different alignment operations; one event per loop execution
  while (ridx<(int)pretty_ref.length() or sidx<(int)pretty_super.length()) {
    bool new_event = true;

    // Events that increment both ridx and sidx
    if (ridx<(int)pretty_ref.length() and sidx<(int)pretty_super.length()){

      // 1) Invalid subset operation
      if (pretty_ref[ridx]==' ' and pretty_super[sidx]==' '){
        return "";
      }

      // 2) match event
      if (pretty_ref[ridx]=='|' and pretty_super[sidx] =='|'){
        super_aln += '|';
        ++ridx; ++sidx;
        continue;
      }

      // 3) mismatches event
      if (have_matches(pretty_ref, ridx, pretty_super, sidx)){
        super_aln += ' ';
        ++ridx; ++sidx;
        continue;
      }

      // 4) Deletion in t:Sub <-> q:Super
      while (ridx<(int)pretty_ref.length() and sidx<(int)pretty_super.length() and pretty_super[sidx] =='-'){
        if (pretty_ref[ridx] != '|')
          return "";
        // Enforce left M operation
        if (new_event){
          new_event = false;
          if (not have_matches(pretty_ref, ridx-1, pretty_super, sidx-1))
            return "";
        }
        super_aln += '-';
        ++ridx; ++sidx;
      }
      // Enforce trailing M operation if event was triggered
      if (not new_event){
        if (not have_matches(pretty_ref, ridx, pretty_super, sidx))
          return "";
        continue;
      }

      // 5) Insertion in t:Ref <-> q:Sub
      while  (ridx<(int)pretty_ref.length() and sidx<(int)pretty_super.length() and pretty_ref[ridx] =='+'){
        if (pretty_super[sidx] != '|')
          return "";
        // Enforce left M operation
        if (new_event){
          new_event = false;
          if (not have_matches(pretty_ref, ridx-1, pretty_super, sidx-1))
            return "";
        }
        super_aln += '+';
        ++ridx; ++sidx;
      }
      // Enforce trailing M operation if event was triggered
      if (not new_event){
        if (not have_matches(pretty_ref, ridx, pretty_super, sidx))
          return "";
        continue;
      }

    } // End double incrementing events

    // 6) Deletion in t:Ref <-> q:Sub
    if (ridx<(int)pretty_ref.length() and pretty_ref[ridx] =='-'){
      // Need flanking M cigar operations
      if (not (have_matches(pretty_ref, ridx-1, pretty_super, sidx-1) and is_match(pretty_super, sidx)))
        return "";
      while (ridx<(int)pretty_ref.length() and pretty_ref[ridx] =='-'){
        super_aln += '-';
        ++ridx;
      }
      if (not is_match(pretty_ref, ridx))
        return "";
      continue;
    }

    // 7) Insertion in t:Sub <-> q:Super
    if (sidx<(int)pretty_super.length() and pretty_super[sidx] =='+'){
      // Need flanking M cigar operations
      if (not (have_matches(pretty_ref, ridx-1, pretty_super, sidx-1)and is_match(pretty_ref, ridx)))
        return "";
      while (sidx<(int)pretty_super.length() and pretty_super[sidx] =='+'){
        super_aln += '+';
        ++sidx;
      }
      if (not is_match(pretty_super, sidx))
        return "";
      continue;
    }

    // 8) If we are at this point it means we went through the whole loop
    // without triggering an event - Fail and report
    //cerr << "WARNING AlleleSubsetCheck::get_cond_superset_alignment failed to trigger event for "
    //     << "\"" << pretty_ref << "\":" << ridx << " \"" << pretty_super << "\":" << sidx
    //     << " super_aln=" << super_aln << endl;
    ++counter_failures;
    failure_ = true;
    return "";
  }

  return super_aln;
}

// --------------------------------------------------------------------

void   AlleleSubsetCheck::reset_counters()
{
  counter_num_align_checks  = 0;
  counter_num_align_subsets = 0;
  counter_num_mnp_checks    = 0;
  counter_num_mnp_subsets   = 0;
  counter_failures          = 0;
}

// --------------------------------------------------------------------
void   AlleleSubsetCheck::print_stats()
{
  if (not check_enabled){
    cout << "Allele Subset Annotation Summary: Disabled." << endl;
    return;
  }

  ostringstream table;
  table << endl << "Allele Subset Annotation Summary:" << endl;
  table << setw(23) << " ";
  table << setw(23) << "--------------------" << setw(23) << "--------------------" << endl;
  table << setw(23) << " ";
  table << setw(23) << "Pairs Investigated" << setw(23) << "Subsets found" << endl;
  table << setw(23) << " ";
  table << setw(23) << "--------------------" << setw(23) << "--------------------" << endl;
  table << setw(23) << "MNP alignment";
  table << setw(23) << counter_num_mnp_checks << setw(23) << counter_num_mnp_subsets << endl;
  table << setw(23) << "Smith-Waterman";
  table << setw(23) << counter_num_align_checks << setw(23) << counter_num_align_subsets << endl;

  table << setw(23) << " ";
  table << setw(23) << "--------------------" << setw(23) << "--------------------" << endl;
  table << setw(23) << "Total";
  table << setw(23) << (counter_num_mnp_checks+counter_num_align_checks)
        << setw(23) << (counter_num_mnp_subsets+counter_num_align_subsets) << endl;
  if (counter_failures>0)
    table << "Number of pair failures: " << counter_failures << endl;
  cout << table.str() << endl;
}

