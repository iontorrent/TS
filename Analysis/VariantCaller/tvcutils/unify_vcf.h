/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#ifndef ION_ANALYSIS_UNIFY_VCF_H
#define ION_ANALYSIS_UNIFY_VCF_H

class bgzf_stream;
class VcfOrderedMerger;
class PriorityQueue;
class ComparableVcfVariant;

struct CoverageInfoEntry {
  string sequenceName;
  long position;
  unsigned long cov;
  const ReferenceReader& reference_reader;
  CoverageInfoEntry(const ReferenceReader& reader)
      : sequenceName(""), position(0), cov(0), reference_reader(reader) {  }
  CoverageInfoEntry(const string& chr, const long& pos, const unsigned long& cov, const ReferenceReader& r)
      : sequenceName(chr), position(pos), cov(cov), reference_reader(r) { }
  CoverageInfoEntry(const CoverageInfoEntry& entry)
      : sequenceName(entry.sequenceName), position(entry.position), cov(entry.cov),
        reference_reader(entry.reference_reader)  { }

  int chr() { return reference_reader.chr_idx(sequenceName.c_str()); };

  static CoverageInfoEntry* parse(const char * line, const ReferenceReader& r);
  static CoverageInfoEntry* parse(string line, const ReferenceReader& r) { return parse(line.c_str(), r); }
};

struct CoverageEntryComparator {
  bool operator()(const CoverageInfoEntry& lhs, const CoverageInfoEntry& rhs) { return lhs.cov < rhs.cov; }
};

class bgzf_stream {
  BGZF *bgzf;
  string data;
  unsigned int WINDOW_SIZE;
public:
  bgzf_stream(string& path, unsigned int ws = 64 * 1024)
      : bgzf(NULL), WINDOW_SIZE(ws) { bgzf = _bgzf_open(path.c_str(), "w");data.reserve(2 * WINDOW_SIZE); }

  ~bgzf_stream() { if (bgzf) flush();_bgzf_close(bgzf);  }

  template<class T>
  bgzf_stream &operator<<(T &data);

  void flush();
private:
  int write_buffer();

  int write(int length);
};

class ComparableVcfVariant : public vcf::Variant {
  const VcfOrderedMerger& merger;
  unsigned long _t;
public:
  ComparableVcfVariant(const VcfOrderedMerger & merger, vcf::VariantCallFile& vcf, unsigned long c)
      : vcf::Variant(vcf), merger(merger), _t(c) { }

  bool operator<(ComparableVcfVariant& b);
};

class VariantComparator {
public:
  VariantComparator() {}
  bool operator()(ComparableVcfVariant* lhs, ComparableVcfVariant* rhs) { return (*lhs)<(*rhs); }
};

typedef priority_queue<ComparableVcfVariant*, vector<ComparableVcfVariant*>, VariantComparator> variant_queue;

class PriorityQueue : private variant_queue {
private:
  vcf::VariantCallFile file;
  unsigned long _size, _vc;
  ComparableVcfVariant* _current;
  bool left_align_enabled, enabled;
  VcfOrderedMerger &merger;
  const ReferenceReader &reference_reader;

public:
  PriorityQueue(string& filename,
                VcfOrderedMerger &merger,
                const ReferenceReader &reader,
                unsigned long w = 1,
                bool la = false,
                bool parse_samples = false)
      : variant_queue(VariantComparator()),
        _size(w), _vc(0), _current(NULL), left_align_enabled(la), enabled(!filename.empty()),
        merger(merger), reference_reader(reader) { open_vcf_file(file, filename, parse_samples); }

  ~PriorityQueue() { while (_current) next(); }

  bool has_value() { return _current != NULL; }

  void next();

  vcf::Variant *current() { return reinterpret_cast<vcf::Variant*>(_current); }

  vcf::VariantCallFile& variant_call_file() { return file; }

private:
  void open_vcf_file(vcf::VariantCallFile& vcf, string filename, bool parse_samples = false);

  bool get_next_variant(vcf::Variant* current);

  void trim_variant(vcf::Variant* variant);

  void left_align_variant(vcf::Variant* variant);

  void delete_variant(ComparableVcfVariant* v) { if (v) { delete v;v = NULL; } }

  void delete_current() { delete_variant(_current); }
};

class VcfOrderedMerger {
public:
  VcfOrderedMerger(string& novel_tvc,
                   string& assembly_tvc,
                   string& hotspot_tvc,
                   string& output_tvc,
                   string& path_to_json,
                   string& input_depth,
                   string& gvcf_output,
                   const ReferenceReader& r,
                   TargetsManager& tm,
                   size_t w, size_t min_dp, bool la);

  ~VcfOrderedMerger();

  template <typename T>
  int variant_cmp(const T* v1, const vcf::Variant* v2) const;

  template <typename T>
  bool is_within_target_region(T *variant);

  void perform();
private:

  istream* depth_in;
  bgzf_stream* gvcf_out;
  const ReferenceReader& reference_reader;
  TargetsManager& targets_manager;
  bool left_align_enabled;
  size_t window_size, minimum_depth;
  PriorityQueue novel_queue, assembly_queue, hotspot_queue;
  bgzf_stream bgz_out;
  vector<MergedTarget>::iterator current_target;
  CoverageInfoEntry* current_cov_info;

  void write_header(vcf::VariantCallFile& vcf, string json_path);

  vcf::Variant* merge_overlapping_variants();

  void span_ref_and_alts();

  void generate_novel_annotations(vcf::Variant* variant);

  void merge_annotation_into_vcf(vcf::Variant* merged_entry);

  void process_and_write_vcf_entry(vcf::Variant* current);

  void process_annotation(vcf::Variant* current);

  void blacklist_check();

  void transfer_fields(map<string, vector<string> >& original_field,
                       map<string, vector<string> >& new_field,
                       map<string, int>& field_types,
                       list<pair<int, int> >& map_assembly_to_all,
                       unsigned int final_num_alleles);

  void next_cov_entry();

  vcf::Variant generate_gvcf_entry(vcf::VariantCallFile& current, int chr,
                                   depth_info<size_t> depth, long pos, long end) const;

  void gvcf_process(int chr, long position);

  void gvcf_process(const char * seq_name, long pos);

  void gvcf_out_variant(vcf::Variant *current);

  void gvcf_finish_region();
};

void push_value_to_entry(vcf::Variant& gvcf_entry, size_t value, string key);
void push_value_to_entry(vcf::Variant& gvcf_entry, string value, string key);
void push_info_field(vcf::Variant& gvcf_entry, string value, string key);
void push_format_field(vcf::Variant& gvcf_entry, string value, string key);

template <typename T>
int comp(const T& i1, const T& i2);

template <typename T1, typename T2>
int compare(const pair<T1, T2>& p1, const pair<T1, T2>& p2);

template <typename T1, typename T2>
int compare(const T1& p11, const T2& p12, const T1& p21, const T2& p22);

void push_value_to_vector(vector<string>& v, long index, const string& entry);
void extend_header(vcf::VariantCallFile &vcf);
void parse_parameters_from_json(string filename, vcf::VariantCallFile& vcf);

bool validate_filename_parameter(string filename, string param_name);
bool check_on_read(string filename, string param_name);
bool check_on_write(string filename, string param_name);

#endif //ION_ANALYSIS_UNIFY_VCF_H
