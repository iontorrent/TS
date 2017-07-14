/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <stdio.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <vector>
#include <list>
#include <map>
#include <deque>
#include <set>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>

#include "OptArgs.h"
#include "Utils.h"
#include "IonVersion.h"


using namespace std;


void PrepareHotspotsHelp()
{
  printf ("\n");
  printf ("tvcutils %s-%s (%s) - Miscellaneous tools used by Torrent Variant Caller plugin and workflow.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   tvcutils prepare_hotspots [options]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -b,--input-bed                 FILE       input is a hotspots BED file (either -b or -v required)\n");
  printf ("  -v,--input-vcf                 FILE       input is a hotspots VCF file (either -b or -v required)\n");
  printf ("  -p,--input-real-vcf            FILE       input is a real vcf file that we will process (when this is present -b and -v cannot, but -o must present)\n");
  printf ("  -q,--output-fake-hot-vcf       FILE       output a hotspot file for subset alleles\n"); 
  printf ("  -d,--output-bed                FILE       output left-aligned hotspots in BED format [none]\n");
  printf ("  -o,--output-vcf                FILE       output is a hotspots VCF file. To be used as input to --hotspot-vcf argument of variant_caller_pipeline.py (recommended) [none]\n");
  printf ("  -r,--reference                 FILE       FASTA file containing reference genome (required)\n");
  printf ("  -a,--left-alignment            on/off     perform left-alignment of indels [off]\n");
  printf ("  -s,--allow-block-substitutions on/off     do not filter out block substitution hotspots [on]\n");
  printf ("  -u,--unmerged-bed              FILE       input a target bed file to filter out hotspots that contain a junction of 2 amplicons (optional)\n");
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
 *
 * Possibilities:
 *  - With VCF, propagate select INFO fields that may have useful annotations
 *  - Convert chromosome names: 1 -> chr1. Friendly to cosmic, dbsnp
 */

struct Allele;

struct LineStatus {
  LineStatus(int _line_number) : line_number(_line_number), filter_message_prefix(0), chr_idx(-1),opos(-1) {}
  int line_number;
  const char *filter_message_prefix;
  string filter_message;
//  Allele *allele;
  int chr_idx;
  long opos;
  string id;
};

struct Allele {
  int chr_idx;
  long pos, opos;
  string id;
  string ref, oref;
  string alt, oalt;
  map<string,string>  custom_tags;
  bool filtered;
  LineStatus *line_status;
};

bool compare_alleles (const Allele& a, const Allele& b)
{
  if (a.pos < b.pos)
    return true;
  if (a.pos > b.pos)
    return false;
  if (a.ref.length() < b.ref.length())
    return true;
  if (a.ref.length() > b.ref.length())
    return false;
  return a.alt < b.alt;
  //return a.pos < b.pos;
}


struct Reference {
  string chr;
  long size;
  const char *start;
  int bases_per_line;
  int bytes_per_line;

  char base(long pos) {
    if (pos < 0 or pos >= size)
      return 'N';
    long ref_line_idx = pos / bases_per_line;
    long ref_line_pos = pos % bases_per_line;
    return toupper(start[ref_line_idx*bytes_per_line + ref_line_pos]);
  }
};

class junction_chr {
   public:
        junction_chr() {last = -1; start.clear(); end.clear();ampl_start.clear(); ampl_end.clear();}
    	void add(int b, int e) {
	    if (last == -1) { beg = b; last = e; ampl_start.push_back(b); ampl_end.push_back(e);return;}
	    if (b < beg) { // error
		fprintf(stderr, "unmerged bed file not in order\n");
		exit(1);
	    } else {
		if (b < last and e > last) {
		    //fprintf(stderr, "%d %d\n", b-1, last+1);
		    start.push_back(b-1);
		    end.push_back(last+1);
		}
		if (e > last) { // removed amplicon completed contained in other
		    ampl_start.push_back(b); ampl_end.push_back(e);
		}
	    }
	    beg = b; last = e;
	}
	bool contain(int b, int e) {
	    // binary?
	    if (start.size() == 0) return false;
	    if (b > start.back()) return false;
	    int l = 0, r = start.size()-1;
	    if (start[0] < b) {
	      while (l < r-1) {
		int m = (l+r)/2;
		int x = start[m];
		if (x == b) { l = r = m; break;}
		if (x < b) {
		    l = m;
		} else {
		    r = m;
		}
	      }
	   } else {
	      r = 0;
	   }
	   if (e >= end[r]) return true;
	   return false;
	   // alternative implementation
	   if (start.size() == 0) return false;
	   int ind = find_before(end, e);
	   if (ind == -1) return false;
	   if (start[ind] >= b) return true;
	   return false;
	}
	int find_before(vector<int> & arr, int val) {
	    if (arr[0] > val) return -1;
	    int l = 0, r = arr.size()-1;
	    if (arr[r] <= val) return r;
	    while (l < r-1) {
                int m = (l+r)/2;
                int x = arr[m];
                if (x == val) { return m;}
                if (x < val) {
                    l = m;
                } else {
                    r = m;
                }
            }
	    return l;
	}
	bool contained_in_ampl(int b, int e) {
	    if (ampl_start.size() == 0) return false;
	    int ind = find_before(ampl_start, b);
	    if (ind == -1) return false;
	    if (ampl_end[ind] >= e) return true;
	    return false;
	}
   protected:
    	vector<int> start;
 	vector<int> end;
	vector<int> ampl_start;
	vector<int> ampl_end;
	int last, beg;
};

class junction {
   public:
	junction() {
	    junc_.clear();
	}
	void init(int n) {for (int i = 0; i < n; i++) junc_.push_back(junction_chr());}
   	bool contain(int id, int pos, unsigned int len) {
	    if (id < 0) return false;
	    if (id >= (int) junc_.size()) return false;
	    return junc_[id].contain(pos, pos+len-1);
	}
	bool contained_in_ampl(int id, int pos, unsigned len) {
	    if (junc_.size() == 0) return true;
	    if (id < 0) return false;
            if (id >= (int) junc_.size()) return false;
	    return junc_[id].contained_in_ampl(pos, pos+len-1);
        }
	void add(int id, int beg, int end) {
	    if (id  >= (int) junc_.size()) return;
	    junc_[id].add(beg, end);
	}
   protected:
	vector <junction_chr> junc_;
};

static bool is_mnp_indel(const char *r, int rl, const char *a, int al)
{
    if (rl == al) return true;
    int i, j, k;
    for (i = 0; i < rl and i < al; i++) if (r[i] != a[i]) break;
    for (j = rl-1, k = al-1; j >=i and k >= i; j--, k--) if (r[j] != a[k]) break;
    if (j < i or k < i) return true;
    return false;
}

bool allele_subset(int pos1, const char *ref1, const char *alt1, int pos2, const char *ref2, const char *alt2)
{
    int rlen1 = strlen(ref1);
    int rlen2 = strlen(ref2);
    if (pos1+rlen1 < pos2+rlen2) return false;
    if (pos1 > pos2) return false;
    // pos1 <= pos2
    int alen1 = strlen(alt1), alen2 = strlen(alt2);
    if (strncmp(ref1+pos2-pos1, ref2, rlen2)!=0) return false; // not even right
    const char *s = strstr(alt1, alt2);
    while (s) { 
       	// check left right portion 
     	if (is_mnp_indel(ref1, pos2-pos1, alt1, s-alt1) and is_mnp_indel(ref1+pos2-pos1+rlen2, pos1+rlen1-pos2-rlen2, alt1+(s-alt1+alen2), alen1-(s-alt1+alen2))) return true;
	s =  strstr(s+1, alt2); // check if there are multiple occurence of the small variant
    }
    return false;
}  


class one_vcfline { //ZZ for subset
    public:
	one_vcfline(char *r, char *alt, int p, int g1, int g2, char *line) {
	    strcpy(ref, r);
    	    //split alt into alts
	    char *ss;
	    for (ss = strtok(alt, ","); ss; ss = strtok(NULL, ",")) {
		alts.push_back(string(ss));
	    }
	    pos = p;
	    affected = false;
	    gt1 = g1; gt2 = g2;
	    o_line = string(line);
	}; 
        char ref[1024];
        vector<string> alts;
	int pos;
	bool affected;
	void check_subset(one_vcfline &newline) {
	    unsigned int i, j;
	    bool need_pad = false;
	    for (i = 0; i < newline.alts.size(); ) {
		if (i+1 != newline.gt1 and i+1 != newline.gt2) {i++; continue;}
		for (j = 0; j < alts.size(); j++) {
		    if (j+1 != gt1 and j+1 != gt2) continue;
		    if (allele_subset(pos, ref, alts[j].c_str(), newline.pos, newline.ref, newline.alts[i].c_str())) break;
		    if (newline.pos == pos+1 and strlen(ref) < strlen(newline.ref)) {
			char tmp[1024], tmp2[1024];
			tmp2[1] = tmp[1] = 0; tmp[0] = tmp2[0] = ref[0]; strcat(tmp, newline.ref); strcat(tmp2, newline.alts[i].c_str());
			if (allele_subset(pos, tmp, tmp2, pos, ref, alts[j].c_str())) { need_pad = true;break;}
		    }
		}
		if (j < alts.size()) {
		    if (need_pad) {
			padding_tail(newline.ref+strlen(ref)-1);
		    } 
		    add_one(newline.pos, strlen(newline.ref), newline.alts[i]);
		    newline.remove_ith_alt(i);
		    affected = newline.affected = true;
		    // after erase, no need to do i++;
		} else {
		    i++;
		}
	    }
	};
	void padding_tail(char *addition) {
	    strcat(ref, addition);
	    string s(addition);
	    for (unsigned int i = 0; i < alts.size(); i++) alts[i] += s;
	};
	bool produce_hot_vcf(char *chr, FILE *fp, int &hot_n, FILE *hot_p) { // out to a file
	   if (not affected) {fprintf(fp, "%s", o_line.c_str()); return false;}
	   fprintf(hot_p,"%s\t%d\thotspot_%d\t%s\t", chr,  pos, hot_n, ref);
	   hot_n++;
	   for (unsigned int i = 0; i < alts.size(); i++) {
		if (i != 0) fprintf(hot_p, ",");
		fprintf(hot_p, "%s", alts[i].c_str());
	   }
	   fprintf(hot_p, "\t.\t.\t.\n");
	   return true;
	};
	bool produce_hot_vcf(char *chr, FILE *fp, int &hot_n) {
	    return produce_hot_vcf(chr, fp, hot_n, stdout);
	}
	void remove_ith_alt(int i) {
	   alts.erase(alts.begin()+i); 
	};
	unsigned int gt1, gt2;
    protected:
	void add_one(unsigned int p, unsigned int reflen, string &alt) {
	    string s;
	    unsigned int i;
	    for (i = pos; i < p; i++) s.push_back(ref[i-pos]); //padding front
	    // add middle
	    s += alt;
	    // padding back
	    for (i = p+reflen; i < pos+strlen(ref); i++) s.push_back(ref[i-pos]);
	    alts.push_back(s);
	};
	string o_line;
};


int PrepareHotspots(int argc, const char *argv[])
{
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string input_bed_filename       = opts.GetFirstString ('b', "input-bed", "");
  string input_vcf_filename       = opts.GetFirstString ('v', "input-vcf", "");
  string input_real_vcf_filename  = opts.GetFirstString ('p', "input-real-vcf", "");
  string output_hot_vcf		  = opts.GetFirstString ('q', "output-fake-hot-vcf", "");
  string output_bed_filename      = opts.GetFirstString ('d', "output-bed", "");
  string output_vcf_filename      = opts.GetFirstString ('o', "output-vcf", "");
  string reference_filename       = opts.GetFirstString ('r', "reference", "");
  string unmerged_bed 		  = opts.GetFirstString ('u', "unmerged-bed", "");
  bool left_alignment             = opts.GetFirstBoolean('a', "left-alignment", false);
  bool filter_bypass              = opts.GetFirstBoolean('f', "filter-bypass", false);
  bool allow_block_substitutions  = opts.GetFirstBoolean('s', "allow-block-substitutions", true);
  opts.CheckNoLeftovers();

  if((input_bed_filename.empty() == (input_vcf_filename.empty() and input_real_vcf_filename.empty())) or
      (output_bed_filename.empty() and output_vcf_filename.empty()) or reference_filename.empty()) {
    PrepareHotspotsHelp();
    return 1;
  }
  if ((not input_real_vcf_filename.empty()) and (output_vcf_filename.empty() or not input_vcf_filename.empty())) {
    PrepareHotspotsHelp();
    return 1;
  }


  // Populate chromosome list from reference.fai
  // Use mmap to fetch the entire reference

  int ref_handle = open(reference_filename.c_str(),O_RDONLY);

  struct stat ref_stat;
  fstat(ref_handle, &ref_stat);
  char *ref = (char *)mmap(0, ref_stat.st_size, PROT_READ, MAP_SHARED, ref_handle, 0);


  FILE *fai = fopen((reference_filename+".fai").c_str(), "r");
  if (!fai) {
    fprintf(stderr, "ERROR: Cannot open %s.fai\n", reference_filename.c_str());
    return 1;
  }

  vector<Reference>  ref_index;
  map<string,int> ref_map;
  char line[1024], chrom_name[1024];
  while (fgets(line, 1024, fai) != NULL) {
    Reference ref_entry;
    long chr_start;
    if (5 != sscanf(line, "%1020s\t%ld\t%ld\t%d\t%d", chrom_name, &ref_entry.size, &chr_start,
                    &ref_entry.bases_per_line, &ref_entry.bytes_per_line))
      continue;
    ref_entry.chr = chrom_name;
    ref_entry.start = ref + chr_start;
    ref_index.push_back(ref_entry);
    ref_map[ref_entry.chr] = (int) ref_index.size() - 1;
  }
  fclose(fai);
  junction junc;
  if (!unmerged_bed.empty()) {
    FILE *fp = fopen(unmerged_bed.c_str(), "r");
    if (!fp) {
	fprintf(stderr, "ERROR: Cannot open %s\n", unmerged_bed.c_str());
	return 1;
    }
    char line2[65536];

    junc.init(ref_index.size());
    bool line_overflow = false;
    while (fgets(line2, 65536, fp) != NULL) {
      if (line2[0] and line2[strlen(line2)-1] != '\n' and strlen(line2) == 65535) {
        line_overflow = true;
	continue;
      }
      if (line_overflow) {
        line_overflow = false;
        continue;
      }
     if (strstr(line2, "track")) continue;
      char chr[100];
      int b, e;
      sscanf(line2, "%s %d %d", chr,  &b, &e);
      junc.add(ref_map[chr], b, e);
    }
    fclose(fp);
  }

  // Load input BED or load input VCF, group by chromosome

  deque<LineStatus> line_status;
  vector<deque<Allele> > alleles(ref_index.size());

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

      // OID= table has special meaning
      if (string::npos != string(line2).find("OID=")) {
	line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Bed line contains OID=";
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

      string string_chr(current_chr);
      if (ref_map.find(string_chr) != ref_map.end())
        allele.chr_idx = ref_map[string_chr];
      else if (ref_map.find("chr"+string_chr) != ref_map.end())
        allele.chr_idx = ref_map["chr"+string_chr];
      else if (string_chr == "MT" and ref_map.find("chrM") != ref_map.end())
        allele.chr_idx = ref_map["chrM"];
      else {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Unknown chromosome name: ";
        line_status.back().filter_message = string_chr;
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
        else if (strncmp(next,"ANCHOR=",7) == 0) {
          // ignore ANCHOR
        } else {
          char *value = next;
          while (*value and *value != '=')
            ++value;
          if (*value == '=')
            *value++ = 0;
          allele.custom_tags[next] = value;
        }
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
      // here is the place to check the length of the hotspot cover the amplicon junction. ZZ
      /*
      if (junc.contain(allele.chr_idx, allele.pos, (unsigned int) allele.ref.size())) {
	line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "hotspot BED line contain the complete overlapping region of two amplicon, the variant cannot be detected by tvc";
        continue;
      }
      if (not junc.contained_in_ampl(allele.chr_idx, allele.pos, (unsigned int) allele.ref.size())) {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "hotspot BED line is not contained in any amplicon, the variant cannot be detected by tvc";
        continue;
      }
      */

      allele.filtered = false;
      line_status.push_back(LineStatus(line_number));
      allele.line_status = &line_status.back();
      allele.opos = allele.pos;
      allele.oref = allele.ref;
      allele.oalt = allele.alt;
      alleles[allele.chr_idx].push_back(allele);
      //line_status.back().allele = &alleles[allele.chr_idx].back();
      line_status.back().chr_idx = allele.chr_idx;
      line_status.back().opos = allele.opos;
      line_status.back().id = allele.id;
    }

    fclose(input);
  }



  if (!input_vcf_filename.empty() or !input_real_vcf_filename.empty()) {

    bool real_vcf = false;
    FILE *input;
    FILE *out_real = NULL;
    FILE *out_hot = NULL;
    int fake_ = 0;
    int hn = 1;
    if (!input_real_vcf_filename.empty()) {
	real_vcf = true;
	input = fopen(input_real_vcf_filename.c_str(),"r");
	if (!input) {
	    fprintf(stderr,"ERROR: Cannot open %s\n", input_real_vcf_filename.c_str());
            return 1;
	}
	out_real = fopen(output_vcf_filename.c_str(), "w");
	if (!out_real) {
            fprintf(stderr,"ERROR: Cannot open %s\n", output_vcf_filename.c_str());
            return 1;
        }
	if (!output_hot_vcf.empty()) {
	    out_hot = fopen(output_hot_vcf.c_str(), "w");
	    if (!out_hot) {
		fprintf(stderr,"ERROR: Cannot open %s\n", output_hot_vcf.c_str());
		return 1;
	    } 
   	} else out_hot = stdout;
	fprintf(out_hot, "##fileformat=VCFv4.1\n##allowBlockSubstitutions=true\n#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO\n");
    } else {
        input = fopen(input_vcf_filename.c_str(),"r");
        if (!input) {
            fprintf(stderr,"ERROR: Cannot open %s\n", input_vcf_filename.c_str());
            return 1;
    	}
    }

    char line2[65536];
    char line3[65536];
    int line_number = 0;
    bool line_overflow = false;
    list<one_vcfline> vcflist;

    char last_chr[1024] = "";
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
      if (line2[0] == '#') {
	if (out_real) { fprintf(out_real, "%s", line2);}
        continue;
      }

      if (real_vcf) strcpy(line3, line2);
      char *current_chr = strtok(line2, "\t\r\n");
      char *current_start = strtok(NULL, "\t\r\n");
      char *current_id = strtok(NULL, "\t\r\n");
      char *current_ref = strtok(NULL, "\t\r\n");
      char *current_alt = strtok(NULL, "\t\r\n");
      strtok(NULL, "\t\r\n"); // Ignore QUAL
      strtok(NULL, "\t\r\n"); // Ignore FILTER
      char *current_info = strtok(NULL, "\t\r\n");
      strtok(NULL, "\t\r\n");
      char *gt = strtok(NULL, "\t\r\n");

      if (!current_chr or !current_start or !current_id or !current_ref or !current_alt) {
        line_status.push_back(LineStatus(line_number));
        if (real_vcf) line_status.back().filter_message_prefix = "Malformed real VCF line: expected at least 5 fields";
	else line_status.back().filter_message_prefix = "Malformed hotspot VCF line: expected at least 5 fields";
        continue;
      }


      string string_chr(current_chr);
      int chr_idx = 0;
      if (ref_map.find(string_chr) != ref_map.end())
        chr_idx = ref_map[string_chr];
      else if (ref_map.find("chr"+string_chr) != ref_map.end())
        chr_idx = ref_map["chr"+string_chr];
      else if (string_chr == "MT" and ref_map.find("chrM") != ref_map.end())
        chr_idx = ref_map["chrM"];
      else {
        line_status.push_back(LineStatus(line_number));
        line_status.back().filter_message_prefix = "Unknown chromosome name: ";
        line_status.back().filter_message = string_chr;
        continue;
      }

      for (char *pos = current_ref; *pos; ++pos)
        *pos = toupper(*pos);
      for (char *pos = current_alt; *pos; ++pos)
        *pos = toupper(*pos);


      // Process custom tags
      vector<string>  bstrand;
      vector<string>  hp_max_length;
      string raw_oid;
      string raw_omapalt;
      string raw_oalt;
      string raw_oref;
      string raw_opos;

      if (current_info) {
        string raw_bstrand;
        string raw_hp_max_length;
        for (char *next = strtok(current_info, ";"); next; next = strtok(NULL, ";")) {

          char *value = next;
          while (*value and *value != '=')
            ++value;
          if (*value == '=')
            *value++ = 0;

          if (strcmp(next, "TYPE") == 0)
            continue;
          if (strcmp(next, "HRUN") == 0)
            continue;
          if (strcmp(next, "HBASE") == 0)
            continue;
          if (strcmp(next, "FR") == 0)
            continue;
          if (strcmp(next, "OPOS") == 0) {
	    raw_opos = value;
            continue;
	  }
          if (strcmp(next, "OREF") == 0) {
	    raw_oref = value;
            continue;
	  }
          if (strcmp(next, "OALT") == 0) {
	    raw_oalt = value;
            continue;
	  }
          if (strcmp(next, "OID") == 0) {
            raw_oid = value;
            continue;
          }
          if (strcmp(next, "OMAPALT") == 0) {
            raw_omapalt = value;
            continue;
          }
          if (strcmp(next, "BSTRAND") == 0) {
            raw_bstrand = value;
            continue;
          }
          if (strcmp(next, "hp_max_length") == 0) {
            raw_hp_max_length = value;
            continue;
          }
        }

        if (not raw_bstrand.empty())
          split(raw_bstrand, ',', bstrand);
        if (not raw_hp_max_length.empty())
          split(raw_hp_max_length, ',', hp_max_length);

      }

      if (real_vcf) {
	//fprintf(stderr, "%s\n", gt);
        if (gt == NULL) continue;
	// get gt
	int g1 = atoi(gt), g2;
	gt = strchr(gt, '/');
	if (gt) g2 = atoi(gt+1);
	else {fprintf(stderr, "GT not formatted right\n"); exit(1);}
	//if (g1 == 0 and g2 == 0) continue;
	unsigned int cur_pos = atoi(current_start);
	one_vcfline newline(current_ref, current_alt, cur_pos, g1, g2, line3);
	bool new_chr = false;
	if (strcmp(current_chr, last_chr) != 0) {
	    new_chr = true;
	}
	while (not vcflist.empty()) {
	    if ((not new_chr) and vcflist.front().pos+strlen(vcflist.front().ref) > cur_pos) break;
	    if (vcflist.front().produce_hot_vcf(last_chr, out_real, hn, out_hot)) fake_++;
	    vcflist.pop_front();
	}
	if (new_chr) strcpy(last_chr, current_chr);
	for (list<one_vcfline>::iterator it = vcflist.begin(); it != vcflist.end(); it++) {
	    it->check_subset(newline);
	}
	if (not newline.alts.empty()) vcflist.push_back(newline);
	continue;
      } 
      unsigned int allele_idx = 0;
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

        if (allele_idx < bstrand.size()) {
          if (bstrand[allele_idx] != ".")
            allele.custom_tags["BSTRAND"] = bstrand[allele_idx];
        }

        if (allele_idx < hp_max_length.size()) {
          if (hp_max_length[allele_idx] != ".")
            allele.custom_tags["hp_max_length"] = hp_max_length[allele_idx];
        }

        alleles[allele.chr_idx].push_back(allele);
        //line_status.back().allele = &alleles[allele.chr_idx].back();
        line_status.back().chr_idx = allele.chr_idx;
        line_status.back().opos = allele.opos;
        line_status.back().id = allele.id;
        allele_idx++;
      }
    }

    fclose(input);
    if (real_vcf) {
        while (not vcflist.empty()) {
            if (vcflist.front().produce_hot_vcf(last_chr, out_real, hn, out_hot)) fake_++;
            vcflist.pop_front();
        }
	fclose(out_real);
	fclose(out_hot);
	if (fake_ > 0) 
            return 0;
	else return 1;
    }
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


  for (int chr_idx = 0; chr_idx < (int)ref_index.size(); ++chr_idx) {

    for (deque<Allele>::iterator A = alleles[chr_idx].begin(); A != alleles[chr_idx].end(); ++A) {

      // check bed file
      if (junc.contain(A->chr_idx, A->pos, (unsigned int) A->ref.size())) {
	A->filtered = true;
        A->line_status->filter_message_prefix = "hotspot BED line contain the complete overlapping region of two amplicon, the variant cannot be detected by tvc";
        continue;
      }
      if (not junc.contained_in_ampl(A->chr_idx, A->pos, (unsigned int) A->ref.size())) {
	A->filtered = true;
        A->line_status->filter_message_prefix = "hotspot BED line is not contained in any amplicon, the variant cannot be detected by tvc";
        continue;
      }


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
        ref_expected += ref_index[chr_idx].base(A->pos + idx);
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

      // Option 1: trim all trailing bases;

      //while(ref_end and alt_end and A->ref[ref_end-1] == A->alt[alt_end-1]) {
      //  --ref_end;
      //  --alt_end;
      //}

      // Option 2: trim all leading basees;

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
      if (left_alignment && A->custom_tags.find("BSTRAND") == A->custom_tags.end()) { // black list variant not to be left aligned.
	string trailing;
	int can_do = 0, need_do = 0;
	int ref_end_orig= ref_end, alt_end_orig = alt_end;
	while(ref_end and alt_end and A->ref[ref_end-1] == A->alt[alt_end-1]) {
	    ref_end--; alt_end--;
	} 
	if (ref_end == 0 || alt_end == 0) {
	    can_do = need_do = 1; // indel type, ZZ
	} else {
	    int tmp_start = ref_start;
	    int ref_end_0 = ref_end, alt_end_0 = alt_end; // end after remove trailing match ZZ
	    while (tmp_start < ref_end and tmp_start < alt_end and A->ref[tmp_start] == A->alt[tmp_start])
     		++tmp_start;
	    if (tmp_start == ref_end || tmp_start == alt_end) {
		can_do = 1; need_do = 0; // indel but indel is not at the left. ZZ
	    } else {
		ref_end--; alt_end--;
		while(ref_end and alt_end and A->ref[ref_end-1] == A->alt[alt_end-1]) {
            	    ref_end--; alt_end--;
        	}
		if (ref_end == 0 || alt_end == 0) {
		   // complex with 1 bp MM at right end
		    can_do = need_do = 1;
		    if (ref_end + alt_end == 0) need_do = 0; // SNP
		} else {
		  int tmp_start0 = tmp_start; // start after removing leading matches
		  tmp_start++;
		  while (tmp_start < ref_end_orig and tmp_start < alt_end_orig and A->ref[tmp_start] == A->alt[tmp_start])
			tmp_start++;
		  if (tmp_start >= ref_end_0 || tmp_start >= alt_end_0 || ref_end <= tmp_start0 || alt_end <= tmp_start0) {
			// 1MM plus indel in middle, by definition cannot move the indel left enough to change A->pos
		    	can_do = 1; need_do = 0;
		  } // else real complex 
		}
	    }
	}
	if (!can_do or !need_do) {
	    // do nothing
	    // if !can_do need add some more DP
	    ref_end = ref_end_orig;
	    alt_end = alt_end_orig;
	} else {
	 // left align the indel part, here either ref_end = 0 or alt_end = 0
	  int opos = A->pos;
          while (A->pos > 0) {
            char nuc = ref_index[chr_idx].base(A->pos-1);
            if (ref_end > 0 and A->ref[ref_end-1] != nuc)
              break;
            if (alt_end > 0 and A->alt[alt_end-1] != nuc)
              break;
            A->ref = string(1,nuc) + A->ref;
            A->alt = string(1,nuc) + A->alt;
            A->pos--;
          }
	  if (ref_end != ref_end_orig) {
	    // trailing part is aligned, the whole ref and alt need to be kept. ZZ
	    ref_end = A->ref.size();
	    alt_end = A->alt.size();
	  } 
	  if (junc.contain(chr_idx, A->pos, ref_end) or not junc.contained_in_ampl(chr_idx, A->pos, ref_end)) {
		// after left align the hotspot contain an overlap region, revert to the original ZZ
		if (opos != A->pos) {
		    A->ref.erase(0, opos-A->pos);
		    A->alt.erase(0, opos-A->pos);
		    A->pos = opos;
		    ref_end = ref_end_orig;
		    alt_end = alt_end_orig;
		}
	  }
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
      stable_sort(alleles[chr_idx].begin(), alleles[chr_idx].end(), compare_alleles);

      // Write
      for (deque<Allele>::iterator I = alleles[chr_idx].begin(); I != alleles[chr_idx].end(); ++I) {
        if (I->filtered)
          continue;

        fprintf(output_bed, "%s\t%ld\t%ld\t%s\tREF=%s;OBS=%s",
            ref_index[chr_idx].chr.c_str(), I->pos, I->pos + I->ref.size(), I->id.c_str(),
            I->ref.c_str(), I->alt.c_str());

        for (map<string,string>::iterator C = I->custom_tags.begin(); C != I->custom_tags.end(); ++C)
          fprintf(output_bed, ";%s=%s", C->first.c_str(), C->second.c_str());

        fprintf(output_bed, "\tNONE\n");

        /*
        if (I->pos)
          fprintf(output_bed, "%s\t%ld\t%ld\t%s\t0\t+\tREF=%s;OBS=%s;ANCHOR=%c\tNONE\n",
              ref_index[chr_idx].chr.c_str(), I->pos, I->pos + I->ref.size(), I->id.c_str(),
              I->ref.c_str(), I->alt.c_str(), ref_index[chr_idx].base(I->pos-1));
        else
          fprintf(output_bed, "%s\t%ld\t%ld\t%s\t0\t+\tREF=%s;OBS=%s;ANCHOR=\tNONE\n",
              ref_index[chr_idx].chr.c_str(), I->pos, I->pos + I->ref.size(), I->id.c_str(),
              I->ref.c_str(), I->alt.c_str());
        */
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
        I->ref = string(1,ref_index[chr_idx].base(I->pos)) + I->ref;
        I->alt = string(1,ref_index[chr_idx].base(I->pos)) + I->alt;
      }

      // Sort - with anchor base
      stable_sort(alleles[chr_idx].begin(), alleles[chr_idx].end(), compare_alleles);


      // Merge alleles, remove block substitutions, write
      for (deque<Allele>::iterator A = alleles[chr_idx].begin(); A != alleles[chr_idx].end(); ) {

        string max_ref;
        deque<Allele>::iterator B = A;
        for (; B != alleles[chr_idx].end() and B->pos == A->pos; ++B)
          if (!B->filtered and max_ref.size() < B->ref.size())
            max_ref = B->ref;

        bool filtered = true;
        map<string,set<string> > unique_alts_and_ids;
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

          // Filter alleles with duplicate ALT + ID pairs
          map<string,set<string> >::iterator alt_iter = unique_alts_and_ids.find(new_alt);
          if (alt_iter != unique_alts_and_ids.end()) {
            if (alt_iter->second.count(I->id) > 0) {
              I->filtered = true;
              I->line_status->filter_message_prefix = "Duplicate allele and ID";
              continue;
            }
          }
          unique_alts_and_ids[new_alt].insert(I->id);

          filtered = false;
        }

        if (not filtered) {



          fprintf(output_vcf, "%s\t%ld\t.\t%s\t",
              ref_index[chr_idx].chr.c_str(), A->pos+1, max_ref.c_str());

          bool comma = false;

          map<string,map<string,string> > unique_alts_and_tags;
          set<string> unique_tags;
	  set<string> unique_alt_alleles;

          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            unique_alts_and_tags[I->alt].insert(I->custom_tags.begin(), I->custom_tags.end());
            for (map<string,string>::iterator S = I->custom_tags.begin(); S != I->custom_tags.end(); ++S)
              unique_tags.insert(S->first);
            if (unique_alt_alleles.count(I->alt) > 0)
              continue;
            unique_alt_alleles.insert(I->alt);
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            fprintf(output_vcf, "%s", I->alt.c_str());
          }
	  /*
          for (deque<Allele>::iterator I = A; I != B; ++I) {
            if (I->filtered)
              continue;
            map<string,map<string,string> >::iterator Q = unique_alts_and_tags.find(I->alt);
            if (comma)
              fprintf(output_vcf, ",");
            comma = true;
            if (Q == unique_alts_and_tags.end()) {fprintf(output_vcf, "."); continue;}
            fprintf(output_vcf, "%s", Q->first.c_str());
          }
          */

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

          for (set<string>::iterator S = unique_tags.begin(); S != unique_tags.end(); ++S) {
            fprintf(output_vcf, ";%s=", S->c_str());
            comma=false;
            for (deque<Allele>::iterator I = A; I != B; ++I) {
              if (I->filtered)
                continue;
              map<string,map<string,string> >::iterator Q = unique_alts_and_tags.find(I->alt);
              if (comma)
                fprintf(output_vcf, ",");
              comma = true;
              if (Q == unique_alts_and_tags.end()) {fprintf(output_vcf, "."); continue;}
              map<string,string>::iterator W = Q->second.find(*S);
              if (W == Q->second.end())
                fprintf(output_vcf, ".");
              else
                fprintf(output_vcf, "%s", W->second.c_str());
            }
          }
//            fprintf(output_vcf, ";%s=%s", S->first.c_str(), S->second.c_str());

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


  int lines_ignored = 0;
  for (deque<LineStatus>::iterator L = line_status.begin(); L != line_status.end(); ++L) {
    if (L->filter_message_prefix) {
      if (L->chr_idx >= 0)
        printf("Line %d ignored: [%s:%ld %s] %s%s\n", L->line_number, ref_index[L->chr_idx].chr.c_str(), L->opos+1, L->id.c_str(),
            L->filter_message_prefix, L->filter_message.c_str());
      else
        printf("Line %d ignored: %s%s\n", L->line_number, L->filter_message_prefix, L->filter_message.c_str());
      lines_ignored++;
    }
  }
  printf("Ignored %d out of %d lines\n", lines_ignored, (int)line_status.size());


  munmap(ref, ref_stat.st_size);
  close(ref_handle);

  return 0;
}



