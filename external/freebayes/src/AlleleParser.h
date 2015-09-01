#ifndef ALLELEPARSER_H
#define ALLELEPARSER_H

#include <string>
#include <vector>
#include <map>
#include <deque>
#include <Variant.h>
#include "ExtendParameters.h"
#include "ReferenceReader.h"
#include "BAMWalkerEngine.h"
#include "SampleManager.h"
#include "HotspotReader.h"
#include "InputStructures.h"

using namespace std;

class OrderedVCFWriter;

class AlleleDetails {
public:
  AlleleDetails() : type(ALLELE_UNKNOWN), chr(0), position(0), ref_length(0),
      length(0), minimized_prefix(0),
      repeat_boundary(0), hp_repeat_len (0) , initialized(false), filtered(false), is_hotspot(false),
      hotspot_params(NULL), coverage(0), coverage_fwd(0), coverage_rev(0), samples(1) {}

  void add_observation(const Allele& observation, int sample_index, bool is_reverse_strand, int _chr, int num_samples) {
    if (not initialized) {
      type = observation.type;
      alt_sequence.append(observation.alt_sequence, observation.alt_length);
      position = observation.position;
      chr = _chr;
      ref_length = observation.ref_length;
      initialized = true;
    }
    if (sample_index < 0) return;
    coverage++;
    if ((int)samples.size() != num_samples)
      samples.resize(num_samples);
    samples[sample_index].coverage++;
    if (is_reverse_strand) {
      coverage_rev++;
      samples[sample_index].coverage_rev++;
    } else {
      coverage_fwd++;
      samples[sample_index].coverage_fwd++;
    }
  }

  void add_reference_observation(int sample_index, bool is_reverse_strand, int chr_idx_) {
    coverage++;
    //if ((int)samples.size() <= sample_index)
    //  samples.resize(sample_index+1);
    samples[sample_index].coverage++;
    if (is_reverse_strand) {
      coverage_rev++;
      samples[sample_index].coverage_rev++;
    } else {
      coverage_fwd++;
      samples[sample_index].coverage_fwd++;
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


  void add_hotspot(const HotspotAllele& observation, int num_samples) {
    is_hotspot = true;
    hotspot_params = &observation;
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
    AlleleCoverage() : coverage(0), coverage_fwd(0), coverage_rev(0) {}
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
  int                     minimized_prefix;
  long int                repeat_boundary;      //! end of homopolymer or tandem repeat for indels
  long int                hp_repeat_len;        //! length of HP for filtering
  bool                    initialized;          //! is allele type info populated?
  bool                    filtered;             //! if true, do not report this allele as candidate
  bool                    is_hotspot;           //! is this allele present in hotspot file?
  const HotspotAllele *   hotspot_params;       //! if not NULL, points to hotspot-specific parameters struct
  long int                coverage;             //! total allele coverage (across samples)
  long int                coverage_fwd;         //! forward strand allele coverage (across samples)
  long int                coverage_rev;         //! reverse strand allele coverage (across samples)
  vector<AlleleCoverage>  samples;              //! per-sample coverages
};


class AllelePositionCompare {
public:
  bool operator()(const Allele& a, const Allele& b) {
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




class AlleleParser {
public:

   AlleleParser(const ExtendParameters& parameters, const ReferenceReader& ref_reader,
       const SampleManager& sample_manager, OrderedVCFWriter& vcf_writer, HotspotReader& hotspot_reader);
  ~AlleleParser();

  void BasicFilters(Alignment& ra);
  void RegisterAlignment(Alignment& ra);
  void GenerateCandidates(deque<VariantCandidate>& variant_candidates,
      list<PositionInProgress>::iterator& position_ticket, int& haplotype_length);

  bool GetNextHotspotLocation(int& chr, long& position);

private:
  void SetupHotspotsVCF(const string& hotspots_file);

  void MakeAllele(deque<Allele>& alleles, AlleleType type, long int pos, int length, const char *alt_sequence);

  void PileUpAlleles(int allowed_allele_types, int haplotype_length, bool scan_haplotype,
      list<PositionInProgress>::iterator& position_ticket, int hotspot_window);
  void PileUpAlleles(int pos, int haplotype_length, list<PositionInProgress>::iterator& position_ticket);
  void PileUpHotspotOnly( vector<HotspotAllele> hotspot, list<PositionInProgress>::iterator& position_ticket) {
	if (hotspot.size() == 0) return;
	PileUpAlleles(hotspot[0].pos, hotspot[0].ref_length, position_ticket);
  }
  void InferAlleleTypeAndLength(AlleleDetails& allele);
  long ComputeRepeatBoundary(const string& seq, int chr, long position, int max_size, long &hp_repeat_len);

  void GenerateCandidateVariant(deque<VariantCandidate>& variant_candidates,
      list<PositionInProgress>::iterator& position_ticket, int& haplotype_length);
  void FillInHotSpotVariant(deque<VariantCandidate>& variant_candidates, vector<HotspotAllele>& hotspot);
  void BlacklistAlleleIfNeeded(AlleleDetails& allele);


  // operation parameters
  bool                        only_use_input_alleles_;
  bool                        process_input_positions_only_;
  bool                        use_duplicate_reads_;      // -E --use-duplicate-reads
  int                         use_best_n_alleles_;         // -n --use-best-n-alleles
  int                         max_complex_gap_;
  unsigned int                min_mapping_qv_;                    // -m --min-mapping-quality
  float                       read_max_mismatch_fraction_;  // -z --read-max-mismatch-fraction
  int                         read_snp_limit_;            // -$ --read-snp-limit
  long double                 min_alt_fraction_;  // -F --min-alternate-fraction
  long double                 min_indel_alt_fraction_; // Added by SU to reduce Indel Candidates for Somatic
  int                         min_alt_count_;             // -C --min-alternate-count
  int                         min_alt_total_;             // -G --min-alternate-total
  int                         min_coverage_;             // -! --min-coverage
  int                         allowed_allele_types_;

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
  //vector<char>                black_list_strand_;
  char                        black_list_strand_; // revert to 4.2
  int                         hp_max_lenght_override_value; //! if not zero then it overrides the maxHPLenght parameter in filtering
  float                       strand_bias_override_value;   //! if below zero then it overrides the strand_bias parameter in filtering

};

#endif
