#ifndef ALLELEPARSER_H
#define ALLELEPARSER_H

#include <string>
#include <vector>
#include <map>
#include <deque>
#include <Variant.h>
#include <iostream>
#include <fstream>
#include "ExtendParameters.h"
#include "ReferenceReader.h"
#include "BAMWalkerEngine.h"
#include "SampleManager.h"
#include "HotspotReader.h"
#include "InputStructures.h"
#include "HandleVariant.h"
using namespace std;
#define MAXBLACK 1000
class OrderedVCFWriter;
class CandidateExaminer;
// ====================================================================

class AlleleDetails {
public:
  AlleleDetails() : type(ALLELE_UNKNOWN), chr(0), position(0), ref_length(0), 
      length(0), minimized_prefix(0), minimized_suffix(0),
      repeat_boundary(0), hp_repeat_len (0) , initialized(false), filtered(false), is_hotspot(false), is_black_listed('.'),
      hotspot_params(NULL), coverage(0), coverage_fwd(0), coverage_rev(0), samples(1), param(NULL) {}

  void add_observation(const Allele& observation, int sample_index, bool is_reverse_strand, int _chr, int num_samples, int read_count) {
    if (not initialized) {
      type = observation.type;
      alt_sequence.append(observation.alt_sequence, observation.alt_length);
      position = observation.position;
      chr = _chr;
      ref_length = observation.ref_length;
      initialized = true;
    }
    if (sample_index < 0) return;
    coverage += read_count;
    if ((int)samples.size() != num_samples)
      samples.resize(num_samples);
    samples[sample_index].coverage += read_count;
    if (is_reverse_strand) {
      coverage_rev += read_count;
      samples[sample_index].coverage_rev += read_count;
    } else {
      coverage_fwd += read_count;
      samples[sample_index].coverage_fwd += read_count;
    }
  }

  void add_observation(const Allele& observation, int sample_index, bool is_reverse_strand, int _chr, int num_samples, int read_count, string rawCg) {
	if (not initialized) { raw_cigar = rawCg; }
	else {
		unsigned int s = raw_cigar.size();
		if (s > rawCg.size()) s = rawCg.size();
		for (unsigned int i = 0; i < s; i++) {
		    if (raw_cigar[i] == 'M' and rawCg[i] != 'M') { raw_cigar = rawCg; break;}
		}
	}
	add_observation(observation, sample_index, is_reverse_strand,  _chr, num_samples,  read_count);
  }

  void add_reference_observation(int sample_index, bool is_reverse_strand, int chr_idx_, int read_count) {
    coverage += read_count;
    //if ((int)samples.size() <= sample_index)
    //  samples.resize(sample_index+1);
    samples[sample_index].coverage += read_count;
    if (is_reverse_strand) {
      coverage_rev += read_count;
      samples[sample_index].coverage_rev += read_count;
    } else {
      coverage_fwd += read_count;
      samples[sample_index].coverage_fwd += read_count;
    }
  }

  void initialize_reference(long int _position, int num_samples) {
    type = ALLELE_REFERENCE;
    position = _position;
    ref_length = 1;
    length = 0;
    initialized = true;
    coverage = 0;
    coverage_fwd = 0;
    coverage_rev = 0;
    samples.clear();
    samples.resize(num_samples);
  }


  void add_hotspot(const HotspotAllele& observation, int num_samples, int p, int s) {
    if (not is_hotspot) { // multiple hotspot same allele, present at the left most
    	is_hotspot = true;
    	hotspot_params = &observation;
    	minimized_prefix = p; // special for hotspot
    	minimized_suffix = s; // special for hotspot, will not be unset
    }
    if (not initialized) {
      chr = observation.chr;
      position = observation.pos;
      ref_length = observation.ref_length;
      alt_sequence = observation.alt;
      type = observation.type;
      length = observation.length;
      initialized = true;
    }
    if ((int)samples.size() != num_samples)
      samples.resize(num_samples);
  }

  const char *type_str() {
    if (type == ALLELE_DELETION)    return "del";
    if (type == ALLELE_INSERTION)   return "ins";
    if (type == ALLELE_COMPLEX)     return "complex";
    if (type == ALLELE_SNP)         return "snp";
    if (type == ALLELE_MNP)         return "mnp";
    if (type == ALLELE_REFERENCE)   return "ref";
    return "???";
  }

  struct AlleleCoverage {
    AlleleCoverage() : coverage(0), coverage_fwd(0), coverage_rev(0){}
    long int coverage;
    long int coverage_fwd;
    long int coverage_rev;
  };

  AlleleType              type;                 //! type of the allele
  string                  alt_sequence;         //! allele sequence
  int                     chr;                  //! chromosome
  long int                position;             //! position 0-based against reference
  unsigned int            ref_length;           //! allele length relative to the reference
  int                     length;               //! allele length reported in LEN tag
  int                     minimized_prefix;     //! the two are defining the minimum allele
  int 			  minimized_suffix;
  long int                repeat_boundary;      //! end of homopolymer or tandem repeat for indels
  long int                hp_repeat_len;        //! length of HP for filtering
  bool                    initialized;          //! is allele type info populated?
  bool                    filtered;             //! if true, do not report this allele as candidate
  bool                    is_hotspot;           //! is this allele present in hotspot file?
  char 			  is_black_listed;
  const HotspotAllele *   hotspot_params;       //! if not NULL, points to hotspot-specific parameters struct
  long int                coverage;             //! total allele coverage (across samples)
  long int                coverage_fwd;         //! forward strand allele coverage (across samples)
  long int                coverage_rev;         //! reverse strand allele coverage (across samples)
  string                  raw_cigar; 
  vector<AlleleCoverage>  samples;              //! per-sample coverages
  VariantSpecificParams   *param;
};

// ====================================================================

class AllelePositionCompare {
public:
  bool operator()(const Allele& a, const Allele& b) const {
    if (a.position < b.position)
      return true;
    if (a.position > b.position)
      return false;
/*
    if (alternateLength < other.alternateLength)
      return true;
    if (alternateLength > other.alternateLength)
      return false;
    if (referenceLength < other.referenceLength)
      return true;
    if (referenceLength > other.referenceLength)
      return false;

    int strdelta = strncmp(alternateSequence,other.alternateSequence,min(alternateLength,other.alternateLength));
    return strdelta < 0;
*/

    int strdelta = strncmp(a.alt_sequence,b.alt_sequence,min(a.alt_length,b.alt_length));
    if (strdelta < 0)
      return true;
    if (strdelta > 0)
      return false;
    if (a.alt_length < b.alt_length)
      return true;
    if (a.alt_length > b.alt_length)
      return false;
    return a.ref_length < b.ref_length;
  }
};

class AllelePool {
public:
  struct AlleleInfo {
    string alt;
    int fcov;
    int rcov;
    char bstrand;
  };
  AllelePool() { clear();}

  void add_allele(string alt, string ref, int fc, int rc, char b, int reff, int refr) {
    unsigned int i;
    if (alleles.size() == 0) {
	reffwd = reff; refrev = refr; 
    } else {
	if (reffwd > reff) reffwd = reff;
	if (refrev > refr) refrev = refr;
    }
    if (ref.size() <= refseq.size()) {
	if (ref.size() < refseq.size()) {
	    alt += refseq.substr(ref.size());
	}
	for (i = 0; i < alleles.size(); i++) {
	    if (alt == alleles[i].alt) break;
	}
	if (i <  alleles.size()) return;
    } else {
	string addi = ref.substr(refseq.size());
	for (i = 0; i < alleles.size(); i++) {
	    if (alt == alleles[i].alt+addi) break;
    	}
	if (i <  alleles.size()) return;
	refseq = ref;
	for (i = 0; i < alleles.size(); i++) {
	     alleles[i].alt += addi;
	}
    }
    AlleleInfo n;
    n.alt = alt;
    n.fcov = fc;
    n.rcov = rc;
    n.bstrand = b;
    alleles.push_back(n);
  }
  void clear() { alleles.clear(); refseq.clear();}  
  
  string refseq;
  int reffwd, refrev;
  vector<AlleleInfo> alleles;
}; 


// ====================================================================

class AlleleParser {
public:

   AlleleParser(const ExtendParameters& parameters, const ReferenceReader& ref_reader,
       const SampleManager& sample_manager, OrderedVCFWriter& vcf_writer, HotspotReader& hotspot_reader);
  ~AlleleParser();

  //! basic filters to reject a read
  bool BasicFilters(Alignment& ra, const TargetsManager * const targets_manager = NULL) const;

  //! Populates the allele specific data in the read Alignment object
  void UnpackReadAlleles(Alignment& ra, const TargetsManager * const targets_manager = NULL) const;

  void GenerateCandidates(deque<VariantCandidate>& variant_candidates,
      list<PositionInProgress>::iterator& position_ticket, int& haplotype_length,
	  CandidateExaminer* my_examiner = NULL, const TargetsManager * const targets_manager = NULL);
  bool GetNextHotspotLocation(int& chr, long& position) const;

private:
  //void SetupHotspotsVCF(const string& hotspots_file); // XXX remove me!

  void MakeAllele(deque<Allele>& alleles, AlleleType type, long int pos, int length, const char *alt_sequence) const;

  void PileUpAlleles(int allowed_allele_types, int haplotype_length, bool scan_haplotype,
      list<PositionInProgress>::iterator& position_ticket, int hotspot_window);

  void PileUpAlleles(int pos, int haplotype_length, list<PositionInProgress>::iterator& position_ticket);
  int first_non_empty(vector<HotspotAllele> hotspot) {
	if (hotspot.size() == 0) return -1;
	for (unsigned int i = 0; i < hotspot.size(); i++) {
	    if (hotspot[i].length > 0) return i;
	}
	return -1;
  }
  bool PileUpHotspotOnly( vector<HotspotAllele> hotspot, list<PositionInProgress>::iterator& position_ticket) {
	int j = first_non_empty(hotspot);
	if (j < 0) return false;
	PileUpAlleles(hotspot[j].pos, hotspot[j].ref_length, position_ticket);
	return true;
  }
  void clean_heap_chars() {
	for (unsigned i = 0; i <  heap_chars_.size(); i++) {
	    delete [] heap_chars_[i];
	}
	heap_chars_.clear();
  }
  char *get_heap_chars(const char *s, int len) {
	char *n = new char[len+1];
	strncpy(n, s, len);
	n[len] = 0;
	heap_chars_.push_back(n);
	return n;
  }
  void handle_candidate_list(list<PositionInProgress>::iterator& position_ticket) {
    if (not candidate_list_.is_open()) return;
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (not allele.filtered) {
         candidate_list_ << ref_reader_->chr_str(allele.chr) << '\t' << allele.position+ allele.minimized_prefix+1 << '\t' << ref_reader_->substr(position_ticket->chr,allele.position+ allele.minimized_prefix, allele.ref_length- allele.minimized_prefix) << '\t' << allele.alt_sequence.substr(allele.minimized_prefix) << '\t'  << allele.coverage_fwd << '\t' << allele.coverage_rev << '\t' << ref_pileup_.coverage_fwd << '\t' << ref_pileup_.coverage_rev << endl;
      }
    }
  }
  void handle_black_out(string &refstring) {

    if (not blacked_var_.is_open()) return;
    int pos = 0;
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      pos = allele.position;
      if (allele.is_black_listed != '.') {
        int idx = allele.minimized_prefix;
        if (not add_black(pos+idx, allele.alt_sequence.substr(idx), refstring.substr(idx, allele.ref_length-idx), allele.coverage_fwd,allele.coverage_rev, allele.is_black_listed, ref_pileup_.coverage_fwd, ref_pileup_.coverage_rev )) {
           cerr << "ERROR:  Fail to add black list at position" << pos+idx << endl;
           exit(1);
        }
      }
    }
  }
  void InferAlleleTypeAndLength(AlleleDetails& allele) const;
  bool filtered_by_coverage_novel_allele(AlleleDetails& allele);
  bool is_fake_hotspot(AlleleDetails& allele);
  bool is_fake_hotspot(AlleleDetails& allele, vector<AlleleDetails *>& mnp_list);
  AlleleDetails *subset_of(AlleleDetails& allele, vector<AlleleDetails *>& mnp_list);
  AlleleDetails *subset_of(int ref_length, int pref, int suff, int pos, string alt, vector<AlleleDetails *>& mnp_list);
  void find_mnps(vector<AlleleDetails *>& mnp_list);

  bool get_alt(AlleleDetails& allele, long hint_position, long rlen, string &a_alt);
  bool to_ref(AlleleDetails& allele, long hint_position, long rlen);
  bool decompose_allele(AlleleDetails &allele, long hp, long rlen, int &ab, int &ae, int &alb, int &ale);
  AlleleDetails *find_same_allele(AlleleDetails *allele);

  long ComputeRepeatBoundary(const string& seq, int chr, long position, int max_size, long &hp_repeat_len) const;

  void GenerateCandidateVariant(deque<VariantCandidate>& variant_candidates,
      list<PositionInProgress>::iterator& position_ticket, int& haplotype_length);

  void FillInHotSpotVariant(deque<VariantCandidate>& variant_candidates, vector<HotspotAllele>& hotspot);
  bool FillVariantFlowDisCheck(VariantCandidate &v, string &refstring, list<PositionInProgress>::iterator& position_ticket, bool hotspot_present, int haplotype_length);
  void set_subset(VariantCandidate &v1, VariantCandidate &v, list<int> &co);
  int MakeVariant(deque<VariantCandidate>& variant_candidates, list<PositionInProgress>::iterator& position_ticket, int n, list<int> *alist);
  void BlacklistAlleleIfNeeded(AlleleDetails& allele, int cov, int cov_f, bool b);
  void SegmentBlacklist(int pos, int chr, int total_cov, int total_f_cov);
  void flushblackpos(int idx, size_t pos); 
  int nextblackpos(int i) {
	if (i >= MAXBLACK) return -1;
	int j = i - blackidx;
	if (j < 0) j+=MAXBLACK;
	if (j <nblackpos) {
	    i++;
	    if (i == MAXBLACK) return 0;
	    return i;
	}
	return -1;
  }
  bool add_black(int pos, string alt, string ref, int fc, int rc, char b, int reff, int refr) {
	if (nblackpos == 0) {blackstart = pos;blackidx = 0;}
	int j = pos - blackstart;
	if (j >= MAXBLACK) return false;
	if (j < 0) {
	    nblackpos -= j;
	    blackidx += j;
	    if (blackidx < 0) blackidx+=MAXBLACK;
	    j = blackidx;
	    blackstart = pos;
	} else {
	    if (j+1 > nblackpos) nblackpos = j+1;
	    j += blackidx;
	    if (j >= MAXBLACK) j -= MAXBLACK;
	} 
 	blackedAlleles[j].add_allele(alt, ref, fc, rc, b, reff, refr);
	return true;
  }
	

  // operation parameters
  bool                        only_use_input_alleles_;
  bool                        process_input_positions_only_;
  bool                        use_duplicate_reads_;      // -E --use-duplicate-reads
  int                         use_best_n_alleles_;         // -n --use-best-n-alleles
  int 			      use_best_n_total_alleles_; 
  int                         max_complex_gap_;
  unsigned int                min_mapping_qv_;                    // -m --min-mapping-quality
  float                       read_max_mismatch_fraction_;  // -z --read-max-mismatch-fraction
  int                         read_snp_limit_;            // -$ --read-snp-limit
  long double                 min_alt_fraction_;  // -F --min-alternate-fraction
  long double                 min_indel_alt_fraction_; // Added by SU to reduce Indel Candidates for Somatic
  long double		      min_fake_hotspot_fr_;    // use it to decide if a hotspot allele is considered fake HS.
  int                         min_alt_count_;             // -C --min-alternate-count
  int                         min_alt_total_;             // -G --min-alternate-total
  int                         min_coverage_;             // -! --min-coverage
  int                         allowed_allele_types_;
  int 			      merge_lookahead_;
  bool 			      output_cigar_;
  bool 			      new_hotspot_grouping;
  bool			      coverage_above_minC_;

  // data structures
  const ReferenceReader *     ref_reader_;
  const SampleManager *       sample_manager_;
  OrderedVCFWriter *          vcf_writer_;              //! Only used as Variant factory
  int                         num_samples_;

  HotspotReader *             hotspot_reader_;
  deque<HotspotAllele>        hotspot_alleles_;

  typedef map<Allele,AlleleDetails,AllelePositionCompare>  pileup;
  pileup                      allele_pileup_;
  AlleleDetails               ref_pileup_;
  vector<long int>           coverage_by_sample_;
  long int		     total_cov_;
  //vector<char>                black_list_strand_;
  char                        black_list_strand_; // revert to 4.2
  int                         hp_max_lenght_override_value; //! if not zero then it overrides the maxHPLenght parameter in filtering
  float                       strand_bias_override_value;   //! if below zero then it overrides the strand_bias parameter in filtering
  ofstream       	      blacked_var_;  // output the list of variants black_listed
  ofstream                    candidate_list_;  // output all the candidates generated by freebayes
  AllelePool		      blackedAlleles[MAXBLACK];
  CandidateExaminer*          my_examiner_;
  vector<char *>              heap_chars_;
  int 			      nblackpos;
  size_t 		      blackstart;
  int			      blackidx;
  int                         black_chr;
  long    		      end_cur_ampl_, start_next_ampl_;
};

#endif
