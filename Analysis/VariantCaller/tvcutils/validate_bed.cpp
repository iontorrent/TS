/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "tvcutils.h"

#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <cmath>

#include "OptArgs.h"
#include "IonVersion.h"
#include "json/json.h"
#include "ReferenceReader.h"

using namespace std;


void ValidateBedHelp()
{
  printf ("\n");
  printf ("tvcutils %s-%s (%s) - Miscellaneous tools used by Torrent Variant Caller plugin and workflow.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   tvcutils validate_bed [options]\n");
  printf ("\n");
  printf ("Input selection options (must provide one):\n");
  printf ("     --target-regions-bed        FILE       input is a target regions BED file (required)\n");
  printf ("     --hotspots-bed              FILE       input is a hotspots BED file (required)\n");
  //TODO: printf ("     --hotspots-vcf              FILE       input is a hotspots VCF file\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("     --reference                 FILE       FASTA file containing reference genome (required)\n");
  printf ("     --validation-log            FILE       log file for user-readable warning/error messages [stdout]\n");
  printf ("     --meta-json                 FILE       save validation and file statistics to json file [none]\n");
  printf ("     --unmerged-detail-bed       FILE       output a valid unmerged BED. To be used as input to --primer-trim-bed argument of variant_caller_pipeline.py (recommended) [none]\n");
  printf ("     --unmerged-plain-bed        FILE       output a valid unmerged BED. To be used as input to --region-bed argument of variant_caller_pipeline.py (recommended) [none]\n");
  printf ("     --merged-detail-bed         FILE       output an (almost) valid bedDetail merged BED [none]\n");
  printf ("     --merged-plain-bed          FILE       output a valid plain merged BED [none]\n");
  printf ("     --effective-bed             FILE       output a valid effective BED [none]\n");
  printf ("\n");
}


/**
 *
 * Requirements:
 *  - If input is BED, ignore ANCHOR, fetch correct anchor from fasta
 *  - If VCF, split multi-allelic entries
 *  - Verify reference bases match fasta. Show warning, ignore.
 *  - Verify OBS/ALT are valid bases. Show warning, ignore.
 *  - Verify OBS/ALT != REF. Show warning, ignore.
 *  - Migrate any remaining BED validator checks
 *  - Left align indels
 *  - Output VCF: Produce O* fields
 *  - Output VCF: Combine entries with common start
 *  - Implements 4.0 internal spec https://jira.itw/wiki/pages/viewpage.action?title=BED+File+Formats+and+Examples&spaceKey=TSUD
 *
 * Possibilities:
 *  - With VCF, propagate select INFO fields that may have useful annotations
 *  - Convert chromosome names: 1 -> chr1. Friendly to cosmic, dbsnp
 */
/*


class ReferenceReader {
public:
  ReferenceReader () : initialized_(false), ref_handle_(0), ref_mmap_(0) {}
  ~ReferenceReader () { Cleanup(); }

  bool Initialize(const string& fasta_filename, const string& fai_filename) {
    Cleanup();

    ref_handle_ = open(fasta_filename.c_str(),O_RDONLY);
    if (ref_handle_ < 0) {
      fprintf(stderr, "ERROR: Cannot open %s\n", fasta_filename.c_str());
      return false;
    }

    fstat(ref_handle_, &ref_stat_);
    ref_mmap_ = (char *)mmap(0, ref_stat_.st_size, PROT_READ, MAP_SHARED, ref_handle_, 0);

    FILE *fai = fopen(fai_filename.c_str(), "r");
    if (!fai) {
      fprintf(stderr, "ERROR: Cannot open %s\n", fai_filename.c_str());
      return false;
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
      ref_index_.push_back(ref_entry);
      ref_map_[ref_entry.chr] = (int) ref_index_.size() - 1;
    }
    fclose(fai);
    initialized_ = true;
    return true;
  }

  bool initialized() { return initialized_; }
  int chr_count() { return (int)ref_index_.size(); }
  const char *chr(int idx) { return ref_index_[idx].chr.c_str(); }
  char base(int chr_idx, long pos) { return ref_index_[chr_idx].base(pos); }
  long chr_size(int idx) { return ref_index_[idx].size; }

  int chr_idx(const char *chr_name) {
    string string_chr(chr_name);
    if (ref_map_.find(string_chr) != ref_map_.end())
      return ref_map_[string_chr];
    else if (ref_map_.find("chr"+string_chr) != ref_map_.end())
      return ref_map_["chr"+string_chr];
    else if (string_chr == "MT" and ref_map_.find("chrM") != ref_map_.end())
      return ref_map_["chrM"];
    else
      return -1;
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

    char base(long pos) {
      if (pos < 0 or pos >= size)
        return 'N';
      long ref_line_idx = pos / bases_per_line;
      long ref_line_pos = pos % bases_per_line;
      return toupper(start[ref_line_idx*bytes_per_line + ref_line_pos]);
    }
  };

  bool                initialized_;
  int                 ref_handle_;
  struct stat         ref_stat_;
  char *              ref_mmap_;
  vector<Reference>   ref_index_;
  map<string,int>     ref_map_;

};

*/






enum ErrorLevel {
  kLineCorrect,
  kLineFixed,
  kLineIgnored,
  kFileUnusable
};

enum SuppressGroup {
  kUnsuppressable,
  kLineEnding,
  kStartEndOrder,
  kRegionOrder,
  kDuplicateRegionName,
  kRegionOverlap,
  kMissingGeneId,
  kWrongAnchor,
  kMissingRef
};

struct LogMessage {
  LogMessage() : line(0), column(0), level(kLineCorrect), suppress(kUnsuppressable), filter_message_prefix(0) {}
  int line;
  int column;
  ErrorLevel  level;
  SuppressGroup suppress;
  string name;
  const char *filter_message_prefix;
  string filter_message;
};

struct BedLine {
  // Fields used for both target regions and hotspots
  int           chr_idx;
  long          start;
  long          end;
  string        name;
  string        score;
  string        strand;

  // Target Regions only:
  string        gene_id;
  string        submitted_region;

  // Hotspots only
  string        hotspot_region;
  string        ref;
  string        obs;
  string        anchor;

  // Extra fields: last column (target regions), second last (hotspots)
  deque<string> ion_key;
  deque<string> ion_value;

  // Bookkeeping
  bool          filtered;
  int           line;

  // effective_bed
  long trim_left;
  long trim_right;
};

bool compare_lines (const BedLine& a, const BedLine& b)
{
  if (a.start == b.start)
    return a.end < b.end;
  return a.start < b.start;
}

bool compare_logs (const LogMessage& a, const LogMessage& b)
{
  if (a.line == b.line)
    return a.column < b.column;
  return a.line < b.line;
}


class BedFile {
public:
  BedFile() : num_lines(0), is_bed_detail(false), ion_version(0), num_fields(0), num_standard_fields(0), is_hotspot(false) {}

  int   num_lines;
  bool  is_bed_detail;
  float ion_version;
  int   num_fields;
  int   num_standard_fields;

  bool  is_hotspot;

  deque<string> track_key;
  deque<string> track_value;
  deque<bool>   track_quoted;

  vector<deque<BedLine> > valid_lines;
  deque<LogMessage> log_message;

  LogMessage * log(ErrorLevel level, BedLine *bed_line, const char *message_prefix, const char *message_suffix = 0) {
    log_message.push_back(LogMessage());
    log_message.back().level = level;
    if (bed_line)
      log_message.back().name = bed_line->name;
    log_message.back().filter_message_prefix = message_prefix;
    if (message_suffix)
      log_message.back().filter_message = message_suffix;
    return &log_message.back();
  }

  LogMessage * log_line(ErrorLevel level, SuppressGroup suppress, int line, BedLine *bed_line, const char *message_prefix, const char *message_suffix = 0) {
    log_message.push_back(LogMessage());
    log_message.back().level = level;
    log_message.back().suppress = suppress;
    log_message.back().line = line;
    if (bed_line)
      log_message.back().name = bed_line->name;
    log_message.back().filter_message_prefix = message_prefix;
    if (message_suffix)
      log_message.back().filter_message = message_suffix;
    return &log_message.back();
  }


  LogMessage * log_column(ErrorLevel level, SuppressGroup suppress, int line, int column, BedLine *bed_line, const char *message_prefix, const char *message_suffix = 0) {
    log_message.push_back(LogMessage());
    log_message.back().level = level;
    log_message.back().suppress = suppress;
    log_message.back().line = line;
    log_message.back().column = column;
    if (bed_line)
      log_message.back().name = bed_line->name;
    log_message.back().filter_message_prefix = message_prefix;
    if (message_suffix)
      log_message.back().filter_message = message_suffix;
    return &log_message.back();
  }

  BedFile(const BedFile &bed) {  // deep copy constructor
    num_lines = bed.num_lines;
    is_bed_detail = bed.is_bed_detail;
    ion_version = bed.ion_version;
    num_fields = bed.num_fields;
    num_standard_fields = bed.num_standard_fields;
    is_hotspot = bed.is_hotspot;
    track_key = bed.track_key;
    track_value = bed.track_value;
    track_quoted = bed.track_quoted;
    valid_lines.resize(bed.valid_lines.size());
    for (int chr_idx = 0; chr_idx < (int)bed.valid_lines.size(); ++chr_idx) {
      valid_lines[chr_idx].assign(bed.valid_lines[chr_idx].begin(), bed.valid_lines[chr_idx].end());
    }
    // don't care about log_message
  }
};



bool parse_track_line(char *track_line, int line_number, BedFile& bed)
{
  bool warn_tabs = false;

  // Already verified track line starts with "track "

  track_line += 6;

  while (*track_line) {
    if (*track_line == '\t')
      warn_tabs = true;
    if (isspace(*track_line)) {
      track_line++;
      continue;
    }

    char *key_end = track_line;
    while (*key_end and not isspace(*key_end) and *key_end != '=' and *key_end != '"')
      key_end++;
    if (*key_end != '=') {
      bed.log_line(kFileUnusable, kUnsuppressable, line_number, 0, "Error parsing track line key=value pairs");
      return false;
    }
    bed.track_key.push_back(string(track_line,key_end-track_line));
    track_line = key_end + 1;
    if (*track_line == '"') {
      bed.track_quoted.push_back(true);
      track_line++;
      key_end = track_line;
      while (*key_end and *key_end != '"') {
        if (*key_end == '\t') {
          *key_end = ' ';
          warn_tabs = true;
        }
        key_end++;
      }
      if (*key_end != '"') {
        bed.log_line(kFileUnusable, kUnsuppressable, line_number, 0, "Error parsing track line, no closing quotes");
        return false;
      }
      bed.track_value.push_back(string(track_line,key_end-track_line));
      track_line = key_end + 1;

    } else {
      bed.track_quoted.push_back(false);
      key_end = track_line;
      while (*key_end and !isspace(*key_end)) {
        key_end++;
      }
      bed.track_value.push_back(string(track_line,key_end-track_line));
      track_line = key_end;
    }
  }

  if (warn_tabs)
    bed.log_line(kLineFixed, kUnsuppressable, line_number, 0, "Tabs in track line replaced by spaces");

  bed.is_bed_detail = false;
  bed.ion_version = 0;
  for (unsigned int idx = 0; idx < bed.track_key.size(); ++idx) {
    if (bed.track_key[idx] == "type" and bed.track_value[idx] == "bedDetail") {
      bed.track_quoted[idx] = false;
      bed.is_bed_detail = true;
    }
    if (bed.track_key[idx] == "torrentSuiteVersion" and bed.track_value[idx] == "3.6") {
      bed.log_line(kLineFixed, kUnsuppressable, line_number, 0, "Treating torrentSuiteVersion=3.6 as synonym for ionVersion=4.0 or higher");
      bed.track_quoted[idx] = false;
    }
    if (bed.track_key[idx] == "ionVersion") {
      bed.track_quoted[idx] = false;
      sscanf(bed.track_value[idx].c_str(), "%f", &bed.ion_version);
    }
  }

  if (not bed.is_bed_detail and (bed.ion_version != 0)) {
    bed.log_line(kFileUnusable, kUnsuppressable, line_number, 0, "ionVersion track key is only valid with type=bedDetail");
    return false;
  }

  if (bed.ion_version > 4.0f) {
    bed.log_line(kLineFixed, kUnsuppressable, line_number, 0, "track line has ionVersion higher than current ionVersion=4.0");
  }

  if (bed.ion_version == 0) {
    bed.log_line(kLineFixed, kUnsuppressable, line_number, 0, "track line has no ionVersion number");
  }

  return true;
}


void parse_bed_detail_targets(char *id, char *description, int line_number, int column, BedLine *bed_line, BedFile& bed)
{
  // assert type=bedDetail

  bool invalid_format = false;
  string error_details;

  if (bed.ion_version < 4.0) {

    bed_line->gene_id = description;
    if (bed_line->gene_id == ".")
      bed_line->gene_id.clear();

    size_t found = bed_line->gene_id.find("=");
    if (found != string::npos) {
      invalid_format = true;
      if (not error_details.empty())
	error_details += "; ";
      error_details += "For ionVersion < 4.0 GENE_ID column cannot contain an = sign";
    }
    found = bed_line->gene_id.find(";");
    if (found != string::npos) {
      invalid_format = true;
      if (not error_details.empty())
	error_details += "; ";
      error_details += "For ionVersion < 4.0 GENE_ID column cannot contain a ; character";
    }


    bed_line->submitted_region = id;
    if (bed_line->submitted_region == ".")
      bed_line->submitted_region.clear();

  } else {

    if (description[0] == '.' and description[1] == 0)
      return;

    while (*description) {
      if (*description == ';') {
        description++;
        continue;
      }

      // Parse names
      string key;
      bool invalid_char = false;
      for(; *description and *description != '=' and *description != ';'; ++description) {
        if (isalnum(*description) or *description == '_' or *description == '.' or *description == '-' or *description == '*')
          key.push_back(*description);
        else {
          invalid_format = true;
          invalid_char = true;
        }
      }
      if (invalid_char) {
        if (not error_details.empty())
          error_details += "; ";
        error_details += "invalid character in key " + key;
      }
      if (not *description or *description == ';') {
        invalid_format = true;
        if (not error_details.empty())
          error_details += "; ";
        error_details += "key " + key + " has no value";
        continue;
      }
      description++;

      // parse values
      string value;
      for(; *description and *description != ';'; ++description) {
        if (*description != '=')
          value.push_back(*description);
        else {
          invalid_format = true;
          if (not error_details.empty())
            error_details += "; ";
          error_details += "value for key " + key + " contains = sign";
        }
      }

      if (key == "GENE_ID")
        bed_line->gene_id = value;
      else if (key == "SUBMITTED_REGION")
        bed_line->submitted_region = value;
      else if (key == "TRIM_LEFT") {
	int saved = errno;
	char *tmp;
	errno = 0;
	long val = strtol (value.c_str(), &tmp, 0);
	if (tmp == value.c_str() || *tmp != '\0' || ((val == LONG_MIN || val == LONG_MAX) && errno == ERANGE) || val < 0) {
	  invalid_format = true;
	  error_details += "; ";
          error_details += "value for key " + key + " must be >= 0, is " + value;
	}
	else
	  bed_line->trim_left = val;
	errno = saved;
        bed_line->ion_key.push_back(key);
	string number;
	stringstream strstream;
	strstream << val;
	strstream >> number;
        bed_line->ion_value.push_back(number);
      }
      else if (key == "TRIM_RIGHT") {
	int saved = errno;
	char *tmp;
	errno = 0;
	long val = strtol (value.c_str(), &tmp, 0);
	if (tmp == value.c_str() || *tmp != '\0' || ((val == LONG_MIN || val == LONG_MAX) && errno == ERANGE) || val < 0) {
	  invalid_format = true;
	  error_details += "; ";
          error_details += "value for key " + key + " must be >= 0, is " + value;
	}
	else 
	  bed_line->trim_right = val;
	errno = saved;
        bed_line->ion_key.push_back(key);
	string number;
	stringstream strstream;
	strstream << val;
	strstream >> number;
        bed_line->ion_value.push_back(number);
      }
      else {
        bed_line->ion_key.push_back(key);
        bed_line->ion_value.push_back(value);
      }
    }
  }
  if (invalid_format) {
    bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "Problem parsing description column: ", error_details.c_str());
    bed_line->filtered = true;
  }
}



void parse_bed_detail_hotspots(const char *id, char *description, int line_number, int column, BedLine *bed_line, BedFile& bed,
    ReferenceReader& reference_reader)
{
  // assert type=bedDetail and not torrentSuiteVersion=3.6

  bed_line->hotspot_region = description;

  bool invalid_format = false;
  bool have_ref = false;
  bool have_obs = false;
  bool have_anchor = false;

  if (id[0] == '.' and id[1] == 0)
    id++;

  while (*id and not invalid_format) {
    if (*id == ';') {
      id++;
      continue;
    }

    // Parse names
    string key;
    for(; *id and *id != '=' and *id != ';'; ++id) {
      if (isalnum(*id) or *id == '_' or *id == '.' or *id == '-' or *id == '*')
        key.push_back(*id);
      else {
        invalid_format = true;
        break;
      }
    }
    if (not *id or *id == ';') {
      invalid_format = true;
      break;
    }
    id++;

    // parse values
    string value;
    for(; *id and *id != ';'; ++id) {
      if (*id != '=')
        value.push_back(*id);
      else {
        invalid_format = true;
        break;
      }
    }

    if (key == "REF") {
      bed_line->ref = value;
      have_ref = true;
    } else if (key == "OBS") {
      bed_line->obs = value;
      have_obs = true;
    } else if (key == "ANCHOR") {
      have_anchor = true;
      bed_line->anchor = value;
    } else {
      bed_line->ion_key.push_back(key);
      bed_line->ion_value.push_back(value);
    }
  }

  if (invalid_format) {
    bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "Problem parsing hotspots id column");
    bed_line->filtered = true;
    return;
  }

  for (unsigned int idx = 0; idx < bed_line->ref.size(); ++idx) {
    bed_line->ref[idx] = toupper(bed_line->ref[idx]);
    if (bed_line->ref[idx] != 'A' and bed_line->ref[idx] != 'C' and bed_line->ref[idx] != 'G' and bed_line->ref[idx] != 'T') {
      bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "REF field contains characters other than ATCGatcg");
      bed_line->filtered = true;
      return;
    }
  }

  for (unsigned int idx = 0; idx < bed_line->obs.size(); ++idx) {
    bed_line->obs[idx] = toupper(bed_line->obs[idx]);
    if (bed_line->obs[idx] != 'A' and bed_line->obs[idx] != 'C' and bed_line->obs[idx] != 'G' and bed_line->obs[idx] != 'T') {
      bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "OBS field contains characters other than ATCGatcg");
      bed_line->filtered = true;
      return;
    }
  }

  for (unsigned int idx = 0; idx < bed_line->anchor.size(); ++idx) {
    bed_line->anchor[idx] = toupper(bed_line->anchor[idx]);
    if (bed_line->anchor[idx] != 'A' and bed_line->anchor[idx] != 'C' and bed_line->anchor[idx] != 'G' and bed_line->anchor[idx] != 'T') {
      bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "ANCHOR field contains characters other than ATCGatcg");
      bed_line->filtered = true;
      return;
    }
  }


  string ref_expected;
  string anchor_expected;

  if (bed_line->start)
    anchor_expected += reference_reader.base(bed_line->chr_idx, bed_line->start - 1);

  for (long idx = bed_line->start; idx < bed_line->end; ++idx)
    ref_expected += reference_reader.base(bed_line->chr_idx, idx);


  if (not have_ref) {
    bed.log_column(kLineFixed, kMissingRef, line_number, column, bed_line, "REF field not provided, auto-populating");
    bed_line->ref = ref_expected;
  } else if (ref_expected != bed_line->ref) {
    bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "REF field does not match content of reference fasta file");
    bed_line->filtered = true;
    return;
  }

  if (not have_obs) {
    bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "Mandatory OBS field not found");
    bed_line->filtered = true;
    return;
  }

  if (bed_line->ref == bed_line->obs) {
    bed.log_column(kLineIgnored, kUnsuppressable, line_number, column, bed_line, "Field OBS cannot be the same as REF");
    bed_line->filtered = true;
    return;
  }

  if (not have_anchor) {
    bed_line->anchor = anchor_expected;
  } else if (anchor_expected != bed_line->anchor and not bed_line->anchor.empty()) {
    bed.log_column(kLineFixed, kWrongAnchor, line_number, column, bed_line, "ANCHOR field does not match content of reference fasta file, fixing");
    bed_line->ion_key.push_back("ANCHOR_ORIG");
    bed_line->ion_value.push_back(bed_line->anchor);
    bed_line->anchor = anchor_expected;
  }
}


string validate_name(char *name, int line_number, int column, BedLine *bed_line, BedFile& bed)
{
  string corrected_name;
  corrected_name.reserve(strlen(name));
  bool warning = false;

  for (; *name; ++name) {
    if (isalnum(*name) or *name == '_' or *name == '.' or *name == '-' or *name == ':' or *name == '*')
      corrected_name.push_back(*name);
    else
      warning = true;
  }

  if (warning)
    bed.log_column(kLineFixed, kUnsuppressable, line_number, column, bed_line, "Removed invalid characters");

  return corrected_name;
}



bool load_and_validate_bed(const string& input_file, ReferenceReader& reference_reader, BedFile& bed, Json::Value& meta_json,
    bool is_hotspot)
{

  FILE *input = fopen(input_file.c_str(),"r");
  if (!input) {
    fprintf(stderr,"ERROR: Cannot open %s\n", input_file.c_str());
    return false;
  }

  bed.is_bed_detail = false;
  bed.num_fields = -1;
  bed.num_lines = 0;
  bed.is_hotspot = is_hotspot;
  bool reject_subsequent_tracks = false;

  char line[65536];

  int last_chr = 0;
  long last_start = 0;

  bool line_overflow = false;
  while (fgets(line, 65536, input) != NULL) {

    int line_length = strlen(line);

    bed.num_lines++;
    if (line_length >= 2 and line[line_length-1] == '\n' and line[line_length-2] == '\r')
      bed.log_line(kLineFixed, kLineEnding, bed.num_lines, 0, "DOS line ending");                           // Suppressable

    if (line_length >= 1 and line[line_length-1] == '\r')
      bed.log_line(kLineFixed, kLineEnding, bed.num_lines, 0, "Mac line ending");                           // Suppressable


    if (line[0] and line[line_length-1] != '\n' and line[line_length-1] != '\r' and line_length == 65535) {
      bed.num_lines--;
      line_overflow = true;
      continue;
    }

    if (line_overflow) {
      line_overflow = false;
      bed.log_line(kLineIgnored, kUnsuppressable, bed.num_lines, 0, "Line length exceeds 64K");
      continue;
    }

    // Trim back trailing CR, LF
    while (line_length and (line[line_length-1] == '\n' or line[line_length-1] == '\r'))
      line[--line_length] = 0;

    if (line_length == 0)
      continue;

    if (strncmp(line, "browser", 7) == 0) {
      bed.log_line(kLineIgnored, kUnsuppressable, bed.num_lines, 0, "BED browser line ignored");
      continue;
    }

    if (strncmp(line, "track ", 6) == 0) {
      if (reject_subsequent_tracks) {
        bed.log_line(kLineIgnored, kUnsuppressable, bed.num_lines, 0, "BED track line only allowed in the first line: All lines ignored after this one");
        break;
      }

      if (not parse_track_line(line, bed.num_lines, bed))
        return false;

      reject_subsequent_tracks = true;
      continue;
    }

    reject_subsequent_tracks = true;

    int num_fields = 1;
    char *fields[20];
    fields[0] = line;
    for (char *c = line; *c and num_fields< 20; ++c) {
      if (*c == '\t') {
        *c = 0;
        fields[num_fields++] = c+1;
      }
    }

    if (bed.num_fields == -1) {

      // At this point is_bed_detail is decided

      if (bed.is_bed_detail) {
        if (num_fields < 5 or num_fields > 14) {
          bed.log_line(kFileUnusable, kUnsuppressable, bed.num_lines, 0, "BED file with type=bedDetail must have between 5 and 14 columns");
          return false;
        }
        if (num_fields == 5 and not bed.is_hotspot)
          bed.log_line(kLineFixed, kUnsuppressable, bed.num_lines, 0, "BED file does not have the Region Name column. Names will be automatically assigned as <col1>:<col2>-<col3>.");

        if (num_fields == 5 and bed.is_hotspot)
          bed.log_line(kLineFixed, kUnsuppressable, bed.num_lines, 0, "BED file does not have the Region Name column. Names will be automatically assigned as hotspot<line_no>.");

        if (num_fields > 8)
          bed.log_line(kLineFixed, kUnsuppressable, bed.num_lines, 0, "Only first six and last two BED columns will be used");

      } else {
        if (bed.is_hotspot) {
          bed.log_line(kFileUnusable, kUnsuppressable, bed.num_lines, 0, "Hotspots BED file must have format type=bedDetail");
          return false;
        }
        if (num_fields < 3 or num_fields > 12) {
          bed.log_line(kFileUnusable, kUnsuppressable, bed.num_lines, 0, "BED file without type=bedDetail must have between 3 and 12 columns");
          return false;
        }

        if (num_fields == 3)
          bed.log_line(kLineFixed, kUnsuppressable, bed.num_lines, 0, "BED file does not have the Region Name column. Names will be automatically assigned as <col1>:<col2>-<col3>.");

        if (num_fields > 6)
          bed.log_line(kLineFixed, kUnsuppressable, bed.num_lines, 0, "Only first six BED columns will be used");
      }
      bed.num_fields = num_fields;
      bed.num_standard_fields = bed.is_bed_detail ? num_fields-2 : num_fields;
    }

    if (num_fields != bed.num_fields) {
      bed.log_line(kLineIgnored, kUnsuppressable, bed.num_lines, 0, "Inconsistent number of columns");
      continue;
    }


    // Populate the basic information (first 4 columns)

    int chr_idx = reference_reader.chr_idx(fields[0]);
    if (chr_idx == -1) {
      bed.log_line(kLineIgnored, kUnsuppressable, bed.num_lines, 0, "Unknown chromosome name: ", fields[0]);
      continue;
    }

    bed.valid_lines[chr_idx].push_back(BedLine());
    BedLine& bed_line = bed.valid_lines[chr_idx].back();
    bed_line.chr_idx = chr_idx;
    bed_line.start = strtol(fields[1],NULL,10);
    bed_line.end = strtol(fields[2],NULL,10);
    bed_line.filtered = false;
    bed_line.line = bed.num_lines;
    bed_line.trim_left = 0;
    bed_line.trim_right = 0;

    if (bed.num_standard_fields >= 4)
      bed_line.name = validate_name(fields[3], bed.num_lines, 4, &bed_line, bed);

    if (bed_line.name.empty()) {
      char buffer[1024];
      if (bed.is_hotspot)
        sprintf(buffer, "hotspot%d", bed.num_lines);
      else
        sprintf(buffer, "%s:%ld-%ld", fields[0], bed_line.start+1, bed_line.end);
      bed_line.name = buffer;
      if (bed.num_standard_fields >= 4)
        bed.log_column(kLineFixed, kUnsuppressable, bed.num_lines, 4, &bed_line, "Replacing empty region name column with ", buffer);
    }



    if (bed_line.start < 0 or bed_line.start > reference_reader.chr_size(bed_line.chr_idx)) {
      bed.log_column(kLineIgnored, kUnsuppressable, bed.num_lines, 2, &bed_line, "Region start not in a valid range");
      //bed.valid_lines[chr_idx].pop_back();
      bed_line.filtered = true;
      continue;
    }

    if (bed_line.end < 0 or bed_line.end > reference_reader.chr_size(bed_line.chr_idx)) {
      bed.log_column(kLineIgnored, kUnsuppressable, bed.num_lines, 3, &bed_line, "Region end not in a valid range");
      //bed.valid_lines[chr_idx].pop_back();
      bed_line.filtered = true;
      continue;
    }

    if (bed_line.start > bed_line.end) {
      if (bed.is_hotspot) {
        bed.log_column(kLineIgnored, kUnsuppressable, bed.num_lines, 2, &bed_line, "Region start and end in reverse order");
        //bed.valid_lines[chr_idx].pop_back();
        bed_line.filtered = true;
        continue;
      }
      long x = bed_line.end;
      bed_line.end = bed_line.start;
      bed_line.start = x;
      bed.log_line(kLineFixed, kStartEndOrder, bed.num_lines, &bed_line, "Region start and end in reverse order");      // Suppressable
    }

    if (bed_line.chr_idx < last_chr or (bed_line.chr_idx == last_chr and bed_line.start < last_start))
      bed.log_line(kLineFixed, kRegionOrder, bed.num_lines, &bed_line, "Region out of order");      // Suppressable
    last_chr = bed_line.chr_idx;
    last_start = bed_line.start;

    if (bed.num_standard_fields >= 5) {
      bed_line.score = fields[4];
      if (bed_line.score.empty()) {
        bed.log_column(kLineFixed, kUnsuppressable, bed.num_lines, 5, &bed_line, "Replacing empty column with .");
        bed_line.score = ".";
      }
    } else
      bed_line.score = "0";

    if (bed.num_standard_fields >= 6) {
      bed_line.strand = fields[5];
      if (bed_line.strand != "+" and bed_line.strand != "-") {
        bed.log_column(kLineFixed, kUnsuppressable, bed.num_lines, 6, &bed_line, "Replacing invalid strand string with +");
        bed_line.strand = "+";
      }
    } else
      bed_line.strand = "+";

    if (bed.is_hotspot)
      parse_bed_detail_hotspots(fields[bed.num_standard_fields], fields[bed.num_standard_fields+1], bed.num_lines,
          bed.num_standard_fields+1, &bed_line, bed, reference_reader);

    else if (bed.is_bed_detail)
      parse_bed_detail_targets(fields[bed.num_standard_fields], fields[bed.num_standard_fields+1], bed.num_lines,
          bed.num_standard_fields+2, &bed_line, bed);

  }

  fclose(input);



  // Detect redundant names and get some statistics
  map<string, int>  region_name_map;
  map<string, int>  gene_name_map;
  int num_regions = 0;
  int num_genes = 0;

  for (int chr_idx = 0; chr_idx < (int)bed.valid_lines.size(); ++chr_idx) {
    for (deque<BedLine>::iterator A = bed.valid_lines[chr_idx].begin(); A != bed.valid_lines[chr_idx].end(); ++A) {
      if (A->filtered)
        continue;
      num_regions++;
      if (region_name_map.find(A->name) == region_name_map.end()) {
        region_name_map[A->name] = A->line;
      } else {
        char buffer[1024];
        sprintf(buffer, "%s (previously seen in line %d)", A->name.c_str(), region_name_map[A->name]);
        bed.log_column(kLineFixed, kDuplicateRegionName, A->line, 4, &(*A), "Duplicate region name ", buffer);
      }
      if (not A->gene_id.empty()) {
        if (gene_name_map.find(A->gene_id) == gene_name_map.end()) {
          gene_name_map[A->gene_id] = A->line;
          num_genes++;
        }
      }
    }
  }

  if (bed.is_hotspot) {
    meta_json["num_loci"] = num_regions;

  } else {
    meta_json["num_targets"] = num_regions;
    meta_json["num_genes"] = num_genes;
  }

  if (num_regions == 0) {
    bed.log_line(kFileUnusable, kUnsuppressable, bed.num_lines, 0, "BED file contains no usable regions");
    return false;
  }

  return true;
}





void merge_overlapping_regions(ReferenceReader& reference_reader, BedFile& bed)
{
  for (int chr_idx = 0; chr_idx < (int)bed.valid_lines.size(); ++chr_idx) {

    deque<BedLine>::iterator merged_line = bed.valid_lines[chr_idx].end();
    deque<BedLine>::iterator previous_line = bed.valid_lines[chr_idx].end();

    int merge_count = 0;

    for (deque<BedLine>::iterator A = bed.valid_lines[chr_idx].begin(); A != bed.valid_lines[chr_idx].end(); ++A) {

      if (A->filtered)
        continue;

      if (merged_line == bed.valid_lines[chr_idx].end()) {
        merged_line = A;
        previous_line = A;
        merge_count = 0;
        continue;
      }

      if (merged_line->end <= A->start) { // No overlap
        merged_line = A;
        previous_line = A;
        merge_count = 0;
        continue;
      }

      merge_count++;

      merged_line->end = max(A->end, merged_line->end);
      merged_line->name += "&";
      merged_line->name += A->name;
      merged_line->gene_id += "&";
      merged_line->gene_id += A->gene_id;
      if (not A->submitted_region.empty()) {
          merged_line->submitted_region += "&";
          merged_line->submitted_region += A->submitted_region;
      }
      merged_line->ref += "&";
      merged_line->ref += A->ref;
      merged_line->obs += "&";
      merged_line->obs += A->obs;
      merged_line->anchor += "&";
      merged_line->anchor += A->anchor;
      merged_line->hotspot_region += "&";
      merged_line->hotspot_region += A->hotspot_region;

      for (int dst = 0; dst < (int)merged_line->ion_key.size(); ++dst) {
        merged_line->ion_value[dst] += "&";
        for (int src = 0; src < (int)A->ion_key.size(); ++src) {
          if (A->ion_key[src].empty())
            continue;
          if (A->ion_key[src] == merged_line->ion_key[dst]) {
            merged_line->ion_value[dst] += A->ion_value[src];
            A->ion_key[src].clear();
            break;
          }
        }
      }
      for (int src = 0; src < (int)A->ion_key.size(); ++src) {
        if (A->ion_key[src].empty())
          continue;
        int dst = merged_line->ion_key.size();
        merged_line->ion_key.push_back(A->ion_key[src]);
        merged_line->ion_value.push_back("");
        for (int idx = 0; idx < merge_count; ++idx)
          merged_line->ion_value[dst] += "&";
        merged_line->ion_value[dst] += A->ion_value[src];
      }

      A->filtered = true;
      previous_line = A;
    }
  }
}


bool save_to_bed(const string& output_file, ReferenceReader& reference_reader, BedFile& bed, bool save_bed_detail)
{
  FILE *output = fopen(output_file.c_str(),"w");
  if (!output) {
    fprintf(stderr, "ERROR: Cannot open %s\n", output_file.c_str());
    return false;
  }

  if (save_bed_detail) {
    if (bed.is_hotspot)
      fprintf(output, "track type=bedDetail");
    else
      fprintf(output, "track type=bedDetail ionVersion=4.0");
    for (unsigned int idx = 0; idx < bed.track_key.size(); ++idx) {
      if (bed.track_key[idx] == "type" or bed.track_key[idx] == "torrentSuiteVersion" or bed.track_key[idx] == "ionVersion")
        continue;
      if (bed.track_quoted[idx])
        fprintf(output, " %s=\"%s\"", bed.track_key[idx].c_str(), bed.track_value[idx].c_str());
      else
        fprintf(output, " %s=%s", bed.track_key[idx].c_str(), bed.track_value[idx].c_str());
    }
    fprintf(output, "\n");
  }

  for (int chr_idx = 0; chr_idx < (int)bed.valid_lines.size(); ++chr_idx) {
    for (deque<BedLine>::iterator A = bed.valid_lines[chr_idx].begin(); A != bed.valid_lines[chr_idx].end(); ++A) {
      if (A->filtered)
        continue;

      if (not save_bed_detail) {
        fprintf(output, "%s\t%ld\t%ld\n", reference_reader.chr(A->chr_idx), A->start, A->end);
        continue;
      }

      fprintf(output, "%s\t%ld\t%ld\t%s\t%s\t%s", reference_reader.chr(A->chr_idx), A->start, A->end, A->name.c_str(),
          A->score.c_str(), A->strand.c_str());

      if (bed.is_hotspot) {
        // Hotspots
        fprintf(output, "\tREF=%s;OBS=%s;ANCHOR=%s", A->ref.c_str(), A->obs.c_str(), A->anchor.c_str());
        for (int idx = 0; idx < (int)A->ion_key.size(); ++idx)
          fprintf(output, ";%s=%s", A->ion_key[idx].c_str(), A->ion_value[idx].c_str());
        fprintf(output, "\t%s", A->hotspot_region.c_str());


      } else {
        // Target regions
        fprintf(output, "\t.\t");
        bool has_keys = false;
        if (not A->gene_id.empty()) {
          fprintf(output, "GENE_ID=%s", A->gene_id.c_str());
          has_keys = true;
        }
        if (not A->submitted_region.empty()) {
          if (has_keys)
            fprintf(output, ";");
          fprintf(output, "SUBMITTED_REGION=%s", A->submitted_region.c_str());
          has_keys = true;
        }
        for (int idx = 0; idx < (int)A->ion_key.size(); ++idx) {
          if (has_keys)
            fprintf(output, ";");
          fprintf(output, "%s=%s", A->ion_key[idx].c_str(), A->ion_value[idx].c_str());
          has_keys = true;
        }
        if (not has_keys)
          fprintf(output, ".");
      }
      fprintf(output, "\n");

    }
  }

  fclose(output);
  return true;
}


void save_validation_log(string& input_file, ReferenceReader& reference_reader, BedFile& bed, const string& log_filename)
{
  FILE *output = stdout;
  FILE *output_file = 0;
  if (not log_filename.empty()) {
    output_file = fopen(log_filename.c_str(),"w");
    if (!output_file) {
      fprintf(stderr, "ERROR: Cannot open %s\n", log_filename.c_str());
      return;
    }
    output = output_file;
  }

  int num_warnings = 0;
  int num_errors = 0;
  int num_fatals = 0;


  sort(bed.log_message.begin(), bed.log_message.end(), compare_logs);

  int num_line_ending_warning = 0;
  int num_region_overlap_warning = 0;
  int num_duplicate_region_name_warning = 0;
  int num_missing_ref = 0;
  int num_wrong_anchor = 0;
  for (deque<LogMessage>::iterator L = bed.log_message.begin(); L != bed.log_message.end(); ++L) {
    if (L->suppress == kLineEnding)
      num_line_ending_warning++;
    if (L->suppress == kRegionOverlap)
      num_region_overlap_warning++;
    if (L->suppress == kDuplicateRegionName)
      num_duplicate_region_name_warning++;
    if (L->suppress == kMissingRef)
      num_missing_ref++;
    if (L->suppress == kWrongAnchor)
      num_wrong_anchor++;
  }

  if (num_line_ending_warning > 10)
    fprintf(output, "%s: %d lines have non-unix line endings (showing first 10 warning).\n", input_file.c_str(), num_line_ending_warning);
  if (num_region_overlap_warning > 10)
    fprintf(output, "%s: %d lines overlap another region. (showing first 10 warnings)\n", input_file.c_str(), num_region_overlap_warning);
  if (num_duplicate_region_name_warning > 10)
    fprintf(output, "%s: %d lines have non-unique region names. (showing first 10 warnings)\n", input_file.c_str(), num_duplicate_region_name_warning);
  if (num_missing_ref > 10)
    fprintf(output, "%s: %d lines are missing REF field (showing first 10 warning).\n", input_file.c_str(), num_missing_ref);
  if (num_wrong_anchor > 10)
    fprintf(output, "%s: %d lines have incorrect ANCHOR field (showing first 10 warning).\n", input_file.c_str(), num_wrong_anchor);

  num_line_ending_warning = 0;
  num_region_overlap_warning = 0;
  num_duplicate_region_name_warning = 0;
  num_missing_ref = 0;
  num_wrong_anchor = 0;

  for (deque<LogMessage>::iterator L = bed.log_message.begin(); L != bed.log_message.end(); ++L) {
    if (not L->filter_message_prefix)
      continue;

    if      (L->level == kLineFixed)    num_warnings++;
    else if (L->level == kLineIgnored)  num_errors++;
    else if (L->level == kFileUnusable) num_fatals++;

    if (L->suppress == kLineEnding)
      if (num_line_ending_warning++ >= 10)
        continue;
    if (L->suppress == kRegionOverlap)
      if (num_region_overlap_warning++ >= 10)
        continue;
    if (L->suppress == kDuplicateRegionName)
      if (num_duplicate_region_name_warning++ >= 10)
        continue;
    if (L->suppress == kMissingRef)
      if (num_missing_ref++ >= 10)
        continue;
    if (L->suppress == kWrongAnchor)
      if (num_wrong_anchor++ >= 10)
        continue;

    fprintf(output, "%s", input_file.c_str());

    if (L->column)    fprintf(output, ": line %5d: column %2d: ", L->line, L->column);
    else if (L->line) fprintf(output, ": line %5d:            ", L->line);
    else              fprintf(output, ":                       ");

    if      (L->level == kLineFixed)    fprintf(output, "Warning  ");
    else if (L->level == kLineIgnored)  fprintf(output, "ERROR    ");
    else if (L->level == kFileUnusable) fprintf(output, "FATAL    ");

    if (not L->name.empty())
      fprintf(output, "[%s] ", L->name.c_str());
    fprintf(output, "%s%s", L->filter_message_prefix, L->filter_message.c_str());

    if      (L->level == kLineFixed)    fprintf(output, "\n");
    else if (L->level == kLineIgnored)  fprintf(output, " (line discarded)\n");
    else if (L->level == kFileUnusable) fprintf(output, " (file unusable)\n");
  }

  if (num_fatals)
    fprintf(output, "%s: Validation failed\n\n", input_file.c_str());
  else
    fprintf(output, "%s: Validation successful with %d warnings and %d errors\n\n",
        input_file.c_str(), num_warnings, num_errors);

  if (output_file)
    fclose(output_file);
}








int ValidateBed(int argc, const char *argv[])
{
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  string target_regions_bed       = opts.GetFirstString ('-', "target-regions-bed", "");
  string hotspots_bed             = opts.GetFirstString ('-', "hotspots-bed", "");
  string hotspots_vcf             = opts.GetFirstString ('-', "hotspots-vcf", "");
  string reference                = opts.GetFirstString ('-', "reference", "");
  string validation_log           = opts.GetFirstString ('-', "validation-log", "");
  string meta_json                = opts.GetFirstString ('-', "meta-json", "");
  string unmerged_detail_bed      = opts.GetFirstString ('-', "unmerged-detail-bed", "");
  string unmerged_plain_bed       = opts.GetFirstString ('-', "unmerged-plain-bed", "");
  string merged_detail_bed        = opts.GetFirstString ('-', "merged-detail-bed", "");
  string merged_plain_bed         = opts.GetFirstString ('-', "merged-plain-bed", "");
  string effective_bed            = opts.GetFirstString ('-', "effective-bed", "");
  opts.CheckNoLeftovers();

  string input_mode;
  string input_file;
  string input_file_basename;

  if (not target_regions_bed.empty() and hotspots_bed.empty() and hotspots_vcf.empty()) {
    input_mode = "target_regions";
    input_file = target_regions_bed;

  } else if (target_regions_bed.empty() and not hotspots_bed.empty() and hotspots_vcf.empty()) {
    input_mode = "hotspots_bed";
    input_file = hotspots_bed;

  } else if (target_regions_bed.empty() and hotspots_bed.empty() and not hotspots_vcf.empty()) {
    input_mode = "hotspots_vcf";
    input_file = hotspots_vcf;

  } else {
    ValidateBedHelp();
    return 1;
  }

  input_file_basename = input_file;
  if (string::npos != input_file_basename.find_last_of("/"))
    input_file_basename.erase(0, input_file_basename.find_last_of("/")+1);

  // Populate chromosome list from reference.fai, use mmap to fetch the entire reference

  ReferenceReader reference_reader;
  if (not reference.empty()) {
    reference_reader.Initialize(reference);
  }


  // Load input BED or load input VCF, group by chromosome

  Json::Value meta(Json::objectValue);

  BedFile bed;
  bed.valid_lines.resize(reference_reader.chr_count());


  if (input_mode == "target_regions") {

    if (not load_and_validate_bed(input_file, reference_reader, bed, meta, false)) {
      save_validation_log(input_file_basename, reference_reader, bed, validation_log);
      return 1;
    }

  } else if (input_mode == "hotspots_bed") {

    if (not load_and_validate_bed(input_file, reference_reader, bed, meta, true)) {
      save_validation_log(input_file_basename, reference_reader, bed, validation_log);
      return 1;
    }

  } else {
  }

  for (int chr_idx = 0; chr_idx < (int)bed.valid_lines.size(); ++chr_idx)
    sort(bed.valid_lines[chr_idx].begin(), bed.valid_lines[chr_idx].end(), compare_lines);

  if (not unmerged_detail_bed.empty())
    save_to_bed(unmerged_detail_bed, reference_reader, bed, true);

  if (not unmerged_plain_bed.empty())
    save_to_bed(unmerged_plain_bed, reference_reader, bed, false);


  if (not effective_bed.empty())
  {
    BedFile bed1 = bed;  // should be deep copy, merge_overlapping_regions modifies BedFile arg

    for (int chr_idx = 0; chr_idx < (int)bed1.valid_lines.size(); ++chr_idx) {
      deque<BedLine>::iterator A = bed1.valid_lines[chr_idx].begin();
      while(A != bed1.valid_lines[chr_idx].end()) {
	// cout << A->start << ", " << A->trim_left  << ", " << A->end << ", " << A->trim_right << endl;
	A->start = A->start + A->trim_left;
	A->end = A->end - A->trim_right;
	if (A->end <= A->start){
	  //cout << A->start << " " << A->end << endl;
	  A = bed1.valid_lines[chr_idx].erase(A);
	}
	else {
	  A++;
	}
      }
      sort(bed1.valid_lines[chr_idx].begin(), bed1.valid_lines[chr_idx].end(), compare_lines);
    }
    merge_overlapping_regions(reference_reader, bed1);

    bed1.track_key.push_back("tvc_effective");
    bed1.track_quoted.push_back(true);
    bed1.track_value.push_back("1");

    save_to_bed(effective_bed, reference_reader, bed1, true);
  }

  merge_overlapping_regions(reference_reader, bed);

  if (not merged_detail_bed.empty())
    save_to_bed(merged_detail_bed, reference_reader, bed, true);

  if (not merged_plain_bed.empty())
    save_to_bed(merged_plain_bed, reference_reader, bed, false);

  save_validation_log(input_file_basename, reference_reader, bed, validation_log);


  if (input_mode == "target_regions") {
    long num_bases = 0;
    for (int chr_idx = 0; chr_idx < (int)bed.valid_lines.size(); ++chr_idx) {
      for (deque<BedLine>::iterator A = bed.valid_lines[chr_idx].begin(); A != bed.valid_lines[chr_idx].end(); ++A) {
        if (not A->filtered)
          num_bases += A->end - A->start;
      }
    }
    meta["num_bases"] = (Json::Int64)num_bases;
  }

  if (not meta_json.empty()) {
    ofstream out(meta_json.c_str(), ios::out);
    if (out.good()) {
      out << meta.toStyledString();
    } else {
      fprintf(stderr, "ERROR: unable to write to '%s'\n", meta_json.c_str());
      return 1;
    }
  }

  return 0;




  /*

  save_to_bed(false);
  save_to_bed(true);

  merge_overlapping_lines();

  save_to_bed(false);
  save_to_bed(true);
  */

  /*


  if (!input_bed_filename.empty()) {

    FILE *input = fopen(input_bed_filename.c_str(),"r");
    if (!input) {
      fprintf(stderr,"ERROR: Cannot open %s\n", input_bed_filename.c_str());
      return 1;
    }

    char line2[65536];

    int line_number = 0;
    bool line_overflow = false;
    while (fgets(line2, 65536, input) != NULL) {
      if (line2[0] and line2[strlen(line2)-1] != '\n' and strlen(line2) == 65535) {
        line_overflow = true;
        continue;
      }
      line_number++;
      if (line_overflow) {
        line_overflow = false;
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Malformed hotspot BED line: line length exceeds 64K";
        continue;
      }

      if (strncmp(line2, "browser", 7) == 0)
        continue;

      if (strncmp(line2, "track", 5) == 0) {
        if (string::npos != string(line2).find("allowBlockSubstitutions=true"))
          allow_block_substitutions = true;
        continue;
      }

      char *current_chr = strtok(line2, "\t\r\n");
      char *current_start = strtok(NULL, "\t\r\n");
      char *current_end = strtok(NULL, "\t\r\n");
      char *current_id = strtok(NULL, "\t\r\n");
      char *penultimate = strtok(NULL, "\t\r\n");
      char *ultimate = strtok(NULL, "\t\r\n");
      for (char *next = strtok(NULL, "\t\r\n"); next; next = strtok(NULL, "\t\r\n")) {
        penultimate = ultimate;
        ultimate = next;
      }

      if (!current_chr or !current_start or !current_end or !current_id or !penultimate or !ultimate) {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Malformed hotspot BED line: expected at least 6 fields";
        continue;
      }

      Allele allele;


      allele.chr_idx = reference_reader.chr_idx(current_chr);
      if (allele.chr_idx == -1) {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Unknown chromosome name: ";
        line_status.back().filter_message = current_chr;
        continue;
      }

      allele.pos = strtol(current_start,NULL,10);
      allele.id = current_id;

      char *current_ref = NULL;
      char *current_alt = NULL;
      for (char *next = strtok(penultimate, ";"); next; next = strtok(NULL, ";")) {
        if (strncmp(next,"REF=",4) == 0)
          current_ref = next;
        else if (strncmp(next,"OBS=",4) == 0)
          current_alt = next;
      }
      if (!current_ref or !current_alt) {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Malformed hotspot BED line: REF and OBS fields required in penultimate column";
        continue;
      }
      for (char *pos = current_ref+4; *pos; ++pos)
        allele.ref += toupper(*pos);
      for (char *pos = current_alt+4; *pos; ++pos)
        allele.alt += toupper(*pos);
      allele.filtered = false;
      line_status.push_back(LineStatus(line_number));
      allele.line_status = &line_status.back();
      allele.opos = allele.pos;
      allele.oref = allele.ref;
      allele.oalt = allele.alt;
      alleles[allele.chr_idx].push_back(allele);
      line_status.back().allele = &alleles[allele.chr_idx].back();
    }

    fclose(input);
  }


  if (!input_vcf_filename.empty()) {

    FILE *input = fopen(input_vcf_filename.c_str(),"r");
    if (!input) {
      fprintf(stderr,"ERROR: Cannot open %s\n", input_vcf_filename.c_str());
      return 1;
    }

    char line2[65536];
    int line_number = 0;
    bool line_overflow = false;
    while (fgets(line2, 65536, input) != NULL) {
      if (line2[0] and line2[strlen(line2)-1] != '\n' and strlen(line2) == 65535) {
        line_overflow = true;
        continue;
      }
      line_number++;
      if (line_overflow) {
        line_overflow = false;
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Malformed hotspot VCF line: line length exceeds 64K";
        continue;
      }

      if (strncmp(line2, "##allowBlockSubstitutions=true", 30) == 0) {
        allow_block_substitutions = true;
        continue;
      }
      if (line2[0] == '#')
        continue;

      char *current_chr = strtok(line2, "\t\r\n");
      char *current_start = strtok(NULL, "\t\r\n");
      char *current_id = strtok(NULL, "\t\r\n");
      char *current_ref = strtok(NULL, "\t\r\n");
      char *current_alt = strtok(NULL, "\t\r\n");

      if (!current_chr or !current_start or !current_id or !current_ref or !current_alt) {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Malformed hotspot VCF line: expected at least 5 fields";
        continue;
      }

      int chr_idx = reference_reader.chr_idx(current_chr);
      if (chr_idx == -1) {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Unknown chromosome name: ";
        line_status.back().filter_message = current_chr;
        continue;
      }

      for (char *pos = current_ref; *pos; ++pos)
        *pos = toupper(*pos);
      for (char *pos = current_alt; *pos; ++pos)
        *pos = toupper(*pos);


      for (char *sub_alt = strtok(current_alt,","); sub_alt; sub_alt = strtok(NULL,",")) {

        Allele allele;
        allele.chr_idx = chr_idx;
        allele.ref = current_ref;
        allele.alt = sub_alt;
        allele.pos = strtol(current_start,NULL,10)-1;
        allele.id = current_id;
        if (allele.id == ".")
          allele.id = "hotspot";

        allele.filtered = false;
        line_status.push_back(LineStatus(line_number));
        allele.line_status = &line_status.back();
        allele.opos = allele.pos;
        allele.oref = allele.ref;
        allele.oalt = allele.alt;
        alleles[allele.chr_idx].push_back(allele);
        line_status.back().allele = &alleles[allele.chr_idx].back();
      }
    }

    fclose(input);
  }

  // Process by chromosome:
  //   - Verify reference allele
  //   - Left align
  //   - Sort
  //   - Filter for block substitutions, write

  FILE *output_vcf = NULL;
  if (!output_vcf_filename.empty()) {
    output_vcf = fopen(output_vcf_filename.c_str(), "w");
    if (!output_vcf) {
      fprintf(stderr,"ERROR: Cannot open %s for writing\n", output_vcf_filename.c_str());
      return 1;
    }
    fprintf(output_vcf, "##fileformat=VCFv4.1\n");
    if (allow_block_substitutions)
      fprintf(output_vcf, "##allowBlockSubstitutions=true\n");
    fprintf(output_vcf, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
  }
  FILE *output_bed = NULL;
  if (!output_bed_filename.empty()) {
    output_bed = fopen(output_bed_filename.c_str(), "w");
    if (!output_bed) {
      fprintf(stderr,"ERROR: Cannot open %s for writing\n", output_bed_filename.c_str());
      if (output_vcf)
        fclose(output_vcf);
      return 1;
    }
    if (allow_block_substitutions)
      fprintf(output_bed, "track name=\"hotspot\" type=bedDetail allowBlockSubstitutions=true\n");
    else
      fprintf(output_bed, "track name=\"hotspot\" type=bedDetail\n");
  }


  for (int chr_idx = 0; chr_idx < reference_reader.chr_count(); ++chr_idx) {

    for (deque<Allele>::iterator A = alleles[chr_idx].begin(); A != alleles[chr_idx].end(); ++A) {

      // Invalid characters

      bool valid = true;
      for (const char *c = A->ref.c_str(); *c ; ++c)
        if (*c != 'A' and *c != 'C' and *c != 'G' and *c != 'T')
          valid = false;
      for (const char *c = A->alt.c_str(); *c ; ++c)
        if (*c != 'A' and *c != 'C' and *c != 'G' and *c != 'T')
          valid = false;
      if (not valid) {
        A->filtered = true;
        A->line_status->filter_message_prefix = "REF and/or ALT contain characters other than ACGT: ";
        A->line_status->filter_message = "REF = " + A->ref + " ALT = " + A->alt;
        continue;
      }

      // Filter REF == ALT

      if (A->ref == A->alt) {
        A->filtered = true;
        A->line_status->filter_message_prefix = "REF and ALT alleles equal";
        continue;
      }

      // Confirm reference allele.

      string ref_expected;
      for (int idx = 0; idx < (int) A->ref.size(); ++idx)
        ref_expected += reference_reader.base(chr_idx, A->pos + idx);
      if (A->ref != ref_expected) {
        A->filtered = true;
        A->line_status->filter_message_prefix = "Provided REF allele does not match reference: ";
        A->line_status->filter_message = "Expected " + ref_expected + ", found " + A->ref;
        continue;
      }

      // Trim

      int ref_start = 0;
      int ref_end = A->ref.size();
      int alt_end = A->alt.size();

      // Option 1: trim all trailing bases

      //while(ref_end and alt_end and A->ref[ref_end-1] == A->alt[alt_end-1]) {
      //  --ref_end;
      //  --alt_end;
      //}

      // Option 2: trim all leading basees

      //while (ref_start < ref_end and ref_start < alt_end and A->ref[ref_start] == A->alt[ref_start])
      //  ++ref_start;


      // Option 3: trim anchor base if vcf

      if (!input_vcf_filename.empty()) {
        if (ref_end and alt_end and (ref_end == 1 or alt_end == 1) and A->ref[0] == A->alt[0])
          ref_start = 1;
      }

      A->pos += ref_start;
      A->ref = A->ref.substr(ref_start, ref_end-ref_start);
      A->alt = A->alt.substr(ref_start, alt_end-ref_start);
      ref_end -= ref_start;
      alt_end -= ref_start;

      // Left align
      if (left_alignment) {
        while (A->pos > 0) {
          char nuc = reference_reader.base(chr_idx,A->pos-1);
          if (ref_end > 0 and A->ref[ref_end-1] != nuc)
            break;
          if (alt_end > 0 and A->alt[alt_end-1] != nuc)
            break;
          A->ref = string(1,nuc) + A->ref;
          A->alt = string(1,nuc) + A->alt;
          A->pos--;
        }
      }
      A->ref.resize(ref_end);
      A->alt.resize(alt_end);


      // Filter block substitutions: take 1

      if (ref_end > 0 and alt_end > 0 and ref_end != alt_end and not allow_block_substitutions and not filter_bypass) {
        A->filtered = true;
        A->line_status->filter_message_prefix = "Block substitutions not supported";
        continue;
      }

    }



    if (output_bed) {
      // Sort - without anchor base
      sort(alleles[chr_idx].begin(), alleles[chr_idx].end(), compare_alleles);

      // Write
      for (deque<Allele>::iterator I = alleles[chr_idx].begin(); I != alleles[chr_idx].end(); ++I) {
        if (I->filtered)
          continue;
        if (I->pos)
          fprintf(output_bed, "%s\t%ld\t%ld\t%s\t0\t+\tREF=%s;OBS=%s;ANCHOR=%c\tNONE\n",
              reference_reader.chr(chr_idx), I->pos, I->pos + I->ref.size(), I->id.c_str(),
              I->ref.c_str(), I->alt.c_str(), reference_reader.base(chr_idx,I->pos-1));
        else
          fprintf(output_bed, "%s\t%ld\t%ld\t%s\t0\t+\tREF=%s;OBS=%s;ANCHOR=\tNONE\n",
              reference_reader.chr(chr_idx), I->pos, I->pos + I->ref.size(), I->id.c_str(),
              I->ref.c_str(), I->alt.c_str());
      }
    }


    if (output_vcf) {

      // Add anchor base to indels
      for (deque<Allele>::iterator I = alleles[chr_idx].begin(); I != alleles[chr_idx].end(); ++I) {
        if (I->filtered)
          continue;
        if (not I->ref.empty() and not I->alt.empty())
          continue;
        if (I->pos == 0) {
          I->filtered = true;
          I->line_status->filter_message_prefix = "INDELs at chromosome start not supported";
          continue;
        }
        I->pos--;
        I->ref = string(1,reference_reader.base(chr_idx, I->pos)) + I->ref;
        I->alt = string(1,reference_reader.base(chr_idx, I->pos)) + I->alt;
      }

      // Sort - with anchor base
      sort(alleles[chr_idx].begin(), alleles[chr_idx].end(), compare_alleles);


      // Merge alleles, remove block substitutions, write
      for (deque<Allele>::iterator A = alleles[chr_idx].begin(); A != alleles[chr_idx].end(); ) {

        string max_ref;
        deque<Allele>::iterator B = A;
        for (; B != alleles[chr_idx].end() and B->pos == A->pos; ++B)
          if (!B->filtered and max_ref.size() < B->ref.size())
            max_ref = B->ref;

        bool filtered = true;
        for (deque<Allele>::iterator I = A; I != B; ++I) {
          if (I->filtered)
            continue;

          string new_alt = I->alt + max_ref.substr(I->ref.size());

          if (new_alt.size() > 1 and max_ref.size() > 1 and new_alt.size() != max_ref.size() and not allow_block_substitutions and not filter_bypass) {
            I->filtered = true;
            I->line_status->filter_message_prefix = "Block substitutions not supported (post-merge)";
            continue;
          }

          I->ref = max_ref;
          I->alt = new_alt;
          filtered = false;
        }

        if (not filtered) {

          fprintf(output_vcf, "%s\t%ld\t.\t%s\t",
              reference_reader.chr(chr_idx), A->pos+1, max_ref.c_str());

          bool comma = false;
          set<string> unique_alt_alleles;
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            if (unique_alt_alleles.count(I->alt) > 0)
              continue;
            unique_alt_alleles.insert(I->alt);
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%s", I->alt.c_str());
          }

          fprintf(output_vcf, "\t.\t.\tOID=");
          comma = false;
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%s", I->id.c_str());
          }

          fprintf(output_vcf, ";OPOS=");
          comma = false;
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%ld", I->opos+1);
          }

          fprintf(output_vcf, ";OREF=");
          comma = false;
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%s", I->oref.c_str());
          }

          fprintf(output_vcf, ";OALT=");
          comma = false;
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%s", I->oalt.c_str());
          }

          fprintf(output_vcf, ";OMAPALT=");
          comma = false;
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%s", I->alt.c_str());
          }

          fprintf(output_vcf, "\n");
        }

        A = B;
      }
    }
  }



  if (output_bed) {
    fflush(output_bed);
    fclose(output_bed);
  }
  if (output_vcf) {
    fflush(output_vcf);
    fclose(output_vcf);
  }

  */

}






