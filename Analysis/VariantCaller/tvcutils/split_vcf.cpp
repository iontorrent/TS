/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "tvcutils.h"

#include <string>
#include <fstream>
#include <iostream>
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
#include <string.h>

#include "OptArgs.h"
#include "IonVersion.h"
#include "json/json.h"

using namespace std;

const int MAX_LENGTH = 262145;
const int FIRST_SAMPLE_COLUMN = 9;

void SplitVcfHelp()
{
  printf ("\n");
  printf ("tvcutils %s-%s (%s) - Miscellaneous tools used by Torrent Variant Caller plugin and workflow.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   tvcutils split_vcf [options]\n");
  printf ("\n");
  printf ("Input selection options (must provide one):\n");
  printf ("     --input-vcf                 FILE       input is a multisample VCF file (required)\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("     --out-dir                   FILE       Output directory (required)\n");
  printf ("\n");
}

bool validate_filename(const string& input_file, string& out_file) {
  size_t pos = input_file.rfind("/");
  if (pos == string::npos) {pos = 0;}
  out_file = input_file.substr(pos + 1, input_file.size() - 4 - pos - 1); 
  string suffix = input_file.substr(input_file.size() - 4, input_file.size());
  std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
  if (suffix != ".vcf") {
    cerr << "ERROR: Expected .vcf suffix on input-vcf, found	" << input_file << "\n";
    return false;
  }
  return true;
}

bool validate_line(int line_count, char* line, int max_length) {
    int line_length = strlen(line);
    if (line[0] and line[line_length-1] != '\n' and line[line_length-1] != '\r' and line_length == max_length) {
      cerr << "ERROR: Line length exceeds " << max_length << " characters at line " << line_count << ".\n";
      return false;
    }
    if (line_length >= 2 and line[line_length-1] == '\n' and line[line_length-2] == '\r') { // DOS line ending
      line[line_length-1] = 0;
      line[line_length-2] = '\n';
      line_length--;
    }
    if (line_length >= 1 and line[line_length-1] == '\r') { // MAC line ending
      line[line_length-1] = '\n';
    }
    // Trim back trailing CR, LF
    while (line_length and (line[line_length-1] == '\n' or line[line_length-1] == '\r'))
      line[--line_length] = 0;

    return true;
}

void split_line(char *line, const char separator, vector<string>& fields) {
  fields.clear();
  char *start = line;
  for (char *c = line; *c; ++c) {
    if (*c == separator) {
      *c = 0;
      fields.push_back(start);
      start = c + 1;
    }
  }
  fields.push_back(start);
}

void split_field(const string& in, const char separator, vector<string>& fields) {
  fields.clear();
  int start = 0;
  for (int index = 0; (index < (int)in.size()); ++index) {
    if (in[index] == separator) {
      fields.push_back(in.substr(start, index - start));
      start = index + 1;
    }
  }
  fields.push_back(in.substr(start));
}

void get_sample_names(char *line, vector<string>& sample_names) {
  vector<string> fields;
  split_line(line, '\t', fields);
  for (size_t index = FIRST_SAMPLE_COLUMN; (index < fields.size()); ++index) {
    sample_names.push_back(fields[index]);
  }
}

typedef struct stat Stat;

static int do_mkdir(const char *path, mode_t mode)
{
  Stat st;
  int status = 0;

  if (stat(path, &st) != 0) {
    if (mkdir(path, mode) != 0 && errno != EEXIST) status = -1;
  }
  else if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    status = -1;
  }

  return(status);
}

int mkpath(const char *path, mode_t mode)
{
  char *pp;
  char *sp;
  int status;
  char *copypath = strdup(path);

  status = 0;
  pp = copypath;
  while (status == 0 && (sp = strchr(pp, '/')) != 0) {
    if (sp != pp) {
      *sp = '\0';
      status = do_mkdir(copypath, mode);
      *sp = '/';
    }
    pp = sp + 1;
  }
  if (status == 0) status = do_mkdir(path, mode);
  free(copypath);
  return (status);
}

bool setup_output_files(vector<string>& header, vector<string>& sample_names, const string& out_dir, const string& out_file, vector<ofstream*>& out_files) {
  int index = 0;
  char buffer[33];
  for (vector<string>::iterator iter = sample_names.begin(); (iter != sample_names.end()); ++iter) {
    sprintf(buffer, "%d", index + 1);
    string out_filename = out_dir + "/" + out_file + "_" + string(buffer) + ".vcf";
    ofstream *output = new ofstream;
    output->open(out_filename.c_str());
    if (!output->is_open()) {
      cerr << "ERROR: Cannot open	" << out_filename << "\n";
      return false;
    }
    out_files.push_back(output);
    for (vector<string>::iterator header_iter = header.begin(); (header_iter != header.end()); ++header_iter) {
      (*output) << *header_iter << "\n";
    }
    (*output) << "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	" << *iter << "\n";
    index++;
  }
  return true;
}

void close(FILE *input, vector<ofstream *>& out_files) {
  fclose(input);
  for (vector<ofstream *>::iterator iter = out_files.begin(); (iter != out_files.end()); ++iter) {
    if ((*iter) != NULL) {
      if ((*iter)->is_open()) {(*iter)->close();} 
      delete *iter; *iter = NULL;
    }
  }
  out_files.clear();
}

void check_sample(FILE *input, vector<ofstream *>& out_files, string& qual_field, string& info_field, string& tag_field, string& value_field, bool& nocall, bool& nodata) {
    nodata = false;
    nocall = false;
    vector<string> info;
    vector<string> tags;
    vector<string> values;
    split_field(info_field, ';', info);
    split_field(tag_field, ':', tags);
    split_field(value_field, ':', values);
    if (tags.size() != values.size()) {
      cerr << "ERROR: input-vcf sample specific tags and values do not match.\n"; close(input, out_files); exit(1);
    }
    bool gt_found = false;
    bool fr_found = false;
    int tag_index = 0;
    for (vector<string>::iterator tag_iter = tags.begin(); (tag_iter != tags.end()); ++tag_iter) {
        if (*tag_iter == "GT") {
            gt_found = true;
            if ((values[tag_index] == "./.") or (values[tag_index] == ".")) {nocall = true;}
            if (values[tag_index] == ".") {values[tag_index] = "./.";}
        }
        if (*tag_iter == "FR") {
            fr_found = true;
            if (values[tag_index].find("NODATA") != string::npos) {nodata = true;}
            if (values[tag_index] == "NODATA") {values[tag_index] = ".,NODATA";}
        }
        if (*tag_iter == "GQ") {
            qual_field = values[tag_index];
        }
        if ((*tag_iter == "AO") or 
            (*tag_iter == "DP") or 
            (*tag_iter == "MDP") or 
            (*tag_iter == "MAO") or 
            (*tag_iter == "MRO") or 
            (*tag_iter == "MAF") or 
            (*tag_iter == "FAO") or 
            (*tag_iter == "FDP") or 
            (*tag_iter == "FRO") or 
            (*tag_iter == "FSAF") or 
            (*tag_iter == "FSAR") or 
            (*tag_iter == "FSRF") or 
            (*tag_iter == "FSRR") or 
            (*tag_iter == "RO") or 
            (*tag_iter == "SAF") or 
            (*tag_iter == "SAR") or 
            (*tag_iter == "SRF") or 
            (*tag_iter == "SRR")) {
            for (vector<string>::iterator info_iter = info.begin(); (info_iter != info.end()); ++info_iter) {
                if (info_iter->find(*tag_iter + "=") == 0) {
                    *info_iter = *tag_iter + "=" + values[tag_index];
                }
            }
        }
        tag_index++;
    }
    if (!gt_found) {
        cerr << "ERROR: input-vcf missing sample specific genotype tag.\n"; close(input, out_files); exit(1);
    }
    if (!fr_found) {
        //cerr << "ERROR: input-vcf missing sample specific filter reason tag.\n"; close(input, out_files); exit(1);
    }
    info_field = "";
    for (vector<string>::iterator info_iter = info.begin(); (info_iter != info.end()); ++info_iter) {
      if (info_field.length() > 0) {info_field += ";";}
      info_field += *info_iter;
    }

    value_field = "";
    for (vector<string>::iterator value_iter = values.begin(); (value_iter != values.end()); ++value_iter) {
      if (value_field.length() > 0) {value_field += ":";}
      value_field += *value_iter;
    }
}

bool split_vcf(string& input_file, const string& out_dir) {
  if (mkpath(out_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
    cerr << "ERROR: Cannot create output path	" << out_dir << "\n";
    return false;		
  }
  string out_file;
  if (not validate_filename(input_file, out_file)) {return false;}
  FILE *input = fopen(input_file.c_str(), "r");
  if (!input) {
    cerr << "ERROR: Cannot open	" << input_file << "\n";
    return false;
  }

  vector<ofstream *> out_files;
  vector<string> header;
  vector<string> sample_names;
  int field_count = -1;
  int line_count = 0;
  char line[MAX_LENGTH + 1];
  while (fgets(line, MAX_LENGTH + 1, input) != NULL) {
    line_count++;
    if (not validate_line(line_count, line, MAX_LENGTH)) {close(input, out_files); return false;}
    if (strlen(line) == 0) {continue;}

    if (line[0] == '#') {
      string header_line = line;
      if (header_line.substr(0, 13) == "#CHROM	POS	ID") {
        get_sample_names(line, sample_names);
        if (sample_names.size() < 2) {cerr << "ERROR: input-vcf is not a multisample vcf file.\n"; close(input, out_files); return false;}
        field_count = FIRST_SAMPLE_COLUMN + sample_names.size();
        if (not setup_output_files(header, sample_names, out_dir, out_file, out_files)) {close(input, out_files); return false;}
      }
      else {header.push_back(header_line);}
    }
    else {
      vector<string> fields;
      split_line(line, '\t', fields);
      if (field_count == -1) {field_count = fields.size();}

      if ((int)fields.size() != field_count) {
        cerr << "ERROR: Inconsistent number of columns at line " << line_count << ".\n";
        close(input, out_files);
        return false;
      }
      else {
        for (vector<ofstream *>::iterator iter = out_files.begin(); (iter != out_files.end()); ++iter) {
          if ((*iter)->is_open()) {
            for (int index = 0; (index < FIRST_SAMPLE_COLUMN); ++index) {
              bool nocall = false;
              bool nodata = false;
              check_sample(input, out_files, fields[FIRST_SAMPLE_COLUMN - 4], fields[FIRST_SAMPLE_COLUMN - 2], fields[FIRST_SAMPLE_COLUMN - 1], fields[FIRST_SAMPLE_COLUMN + (int)(iter - out_files.begin())], nocall, nodata);
              if (index == 5) {if (nodata) {(**iter) << "0" << "\t";} else {(**iter) << fields[index] << "\t";}}
              else if (index == 6) {if (nocall) {(**iter) << "NOCALL" << "\t";} else {(**iter) << fields[index] << "\t";}}
              else {(**iter) << fields[index] << "\t";}
            }
            (**iter) << fields[FIRST_SAMPLE_COLUMN + (int)(iter - out_files.begin())] << "\n";
          }
          else {
            cerr << "ERROR: input-vcf apprears to be malformed.\n"; close(input, out_files); return false;
          }
        }
      }
    }
  }

  close(input, out_files);

  return true;
}

int SplitVcf(int argc, const char *argv[])
{
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  string input_vcf                = opts.GetFirstString ('-', "input-vcf", "");
  string out_dir                  = opts.GetFirstString ('-', "out-dir", "");
  opts.CheckNoLeftovers();

  if (not input_vcf.empty() and not out_dir.empty()) {
    if (not split_vcf(input_vcf, out_dir)) {return 1;}
  } else {
    SplitVcfHelp();
    return 1;
  }

  return 0;
}