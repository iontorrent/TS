#include "AlleleParser.h"

#include <math.h>

#include "MiscUtil.h"
#include "OrderedVCFWriter.h"



AlleleParser::AlleleParser(const ExtendParameters& parameters, const ReferenceReader& ref_reader,
    const SampleManager& sample_manager, OrderedVCFWriter& vcf_writer, HotspotReader& hotspot_reader)
{

  use_duplicate_reads_ = parameters.useDuplicateReads;
  use_best_n_alleles_ = parameters.useBestNAlleles;
  max_complex_gap_ = parameters.maxComplexGap;
  min_mapping_qv_ = parameters.min_mapping_qv;
  read_max_mismatch_fraction_ = parameters.readMaxMismatchFraction;
  read_snp_limit_ = parameters.read_snp_limit;
  min_alt_fraction_ = parameters.minAltFraction;
  min_indel_alt_fraction_ = parameters.minIndelAltFraction;
  min_alt_count_ = parameters.minAltCount;
  min_alt_total_ = parameters.minAltTotal;
  min_coverage_ = parameters.minCoverage;
  only_use_input_alleles_ = parameters.onlyUseInputAlleles;
  process_input_positions_only_ = parameters.processInputPositionsOnly;

  ref_reader_ = &ref_reader;
  sample_manager_ = &sample_manager;
  vcf_writer_ = &vcf_writer;
  num_samples_ = sample_manager_->num_samples_;

  hotspot_reader_ = &hotspot_reader;

  allowed_allele_types_ = ALLELE_REFERENCE;
  if (parameters.allowSNPs)
    allowed_allele_types_ |= ALLELE_SNP;
  if (parameters.allowIndels)
    allowed_allele_types_ |= ALLELE_INSERTION|ALLELE_DELETION;
  if (parameters.allowMNPs)
    allowed_allele_types_ |= ALLELE_MNP;
  if (parameters.allowComplex)
    allowed_allele_types_ |= ALLELE_COMPLEX;

  // black_list_strand.clear(); black_list_strand.push_back('.');  // revert to 4.2
  black_list_strand = '.';
  hp_max_lenght_override_value = 0;
  strand_bias_override_value = 0.0;

}


AlleleParser::~AlleleParser()
{
}


bool AlleleParser::GetNextHotspotLocation(int& chr, long& position)
{
  if (hotspot_reader_->HasMoreVariants()) {
    chr = hotspot_reader_->next_chr();
    position = hotspot_reader_->next_pos();
    return true;
  }
  return false;
}




void AlleleParser::BasicFilters(Alignment& ra)
{
  if (not ra.alignment.BuildCharData()) {
    cerr << "ERROR: Failed to parse read data for BAM Alignment " << ra.alignment.Name << endl;
    exit(1);
  }
  ra.original_position = ra.alignment.Position;
  ra.end = ra.alignment.GetEndPosition();

  // Basic read filters

  if (not sample_manager_->IdentifySample(ra)) {
    ra.filtered = true;
    ra.evaluator_filtered = true;
    return;
  }
  if (ra.alignment.IsDuplicate() and not use_duplicate_reads_) {
    ra.filtered = true;
    return;
  }
  if (!ra.alignment.IsMapped()) {
    ra.filtered = true;
    ra.evaluator_filtered = true;
    return;
  }
  if (!ra.alignment.IsPrimaryAlignment()) {
    ra.filtered = true;
    ra.evaluator_filtered = true;
    return;
  }
  if (ra.alignment.MapQuality < min_mapping_qv_) {
    ra.filtered = true;
    ra.evaluator_filtered = true;
    return;
  }
  ra.worth_saving = true;

}

void AlleleParser::RegisterAlignment(Alignment& ra)
{

  /*
  if (not ra.alignment.BuildCharData()) {
    cerr << "ERROR: Failed to parse read data for BAM Alignment " << ra.alignment.Name << endl;
    exit(1);
  }
  ra.end = ra.alignment.GetEndPosition();

  // Basic read filters

  if (not sample_manager_->IdentifySample(ra)) {
    ra.filtered = true;
    return;
  }
  if (ra.alignment.IsDuplicate() and not use_duplicate_reads_) {
    ra.filtered = true;
    return;
  }
  if (!ra.alignment.IsMapped()) {
    ra.filtered = true;
    return;
  }
  if (!ra.alignment.IsPrimaryAlignment()) {
    ra.filtered = true;
    return;
  }
  if (ra.alignment.MapQuality < min_mapping_qv_) {
    ra.filtered = true;
    return;
  }
  ra.worth_saving = true;

  */

  // Parse read into alleles and store them in generator-friendly format

  int ref_length = ra.end - ra.alignment.Position;
  ra.refmap_start.reserve(ref_length+1);
  ra.refmap_code.reserve(ref_length+1);
  ra.refmap_has_allele.assign(ref_length,'N');
  ra.refmap_allele.resize(ref_length);

  int mismatch_count = 0;
  ra.snp_count = 0;
  deque<Allele> alleles;
  ReferenceReader::iterator ref_ptr  = ref_reader_->iter(ra.alignment.RefID, ra.alignment.Position);
  int                       ref_pos  = ra.alignment.Position;
  const char *              read_ptr = &ra.alignment.QueryBases[0];

  vector<CigarOp>::const_iterator cigar = ra.alignment.CigarData.begin();
  vector<CigarOp>::const_iterator cigar_end  = ra.alignment.CigarData.end();
  if (cigar_end != cigar) {
    // Make sure indel and soft clip cigar ops at the end are ignored
    while (cigar_end-1 != cigar and ((cigar_end-1)->Type == 'I' or (cigar_end-1)->Type == 'D' or
        (cigar_end-1)->Type == 'S' or (cigar_end-1)->Type == 'N'))
      --cigar_end;
  }

  // Main cigar parsing loop
  for ( ; cigar != cigar_end; ++cigar ) {
    unsigned int cigar_len = cigar->Length;

    // Special precaution. If multiple cigars of same type in a row, merge them right here.
    while ((cigar+1) != cigar_end and (cigar+1)->Type == cigar->Type) {
      cigar_len += (cigar+1)->Length;
      ++cigar;
    }

    if (cigar->Type == 'M') { // match or mismatch

      int length = 0;

      for (int i = 0; i < (int)cigar_len; ++i) {

        // record mismatch if we have a mismatch here
        // when the reference is N, we should always call a mismatch
        if (*read_ptr == *ref_ptr and *ref_ptr != 'N') {
          ra.refmap_start.push_back(read_ptr);
          ra.refmap_code.push_back('M');
          ++ref_pos;
          ++ref_ptr;
          ++read_ptr;
          ++length;
          continue;
        }

        if (length)
          MakeAllele(alleles, ALLELE_REFERENCE, ref_pos - length, length, read_ptr - length);

        // register mismatch
        ++mismatch_count;
        ++ra.snp_count;

        ra.refmap_start.push_back(read_ptr);
        length = 0;
        if (*read_ptr == 'A' or *read_ptr == 'T' or *read_ptr == 'G' or *read_ptr == 'C') {
          ra.refmap_code.push_back('X');
          MakeAllele(alleles, ALLELE_SNP, ref_pos, 1, read_ptr);
        } else {
          ra.refmap_code.push_back('N');
          MakeAllele(alleles, ALLELE_NULL, ref_pos, 1, read_ptr);
        }

        // update positions
        ++ref_pos;
        ++ref_ptr;
        ++read_ptr;
      }

      if (length)
        MakeAllele(alleles, ALLELE_REFERENCE, ref_pos - length, length, read_ptr - length);


    } else if (cigar->Type == 'D') { // deletion

      MakeAllele(alleles, ALLELE_DELETION, ref_pos, cigar_len, read_ptr);

      mismatch_count += cigar_len;

      for (unsigned int i = 0; i < cigar_len; ++i) {
        ra.refmap_start.push_back(read_ptr);
        ra.refmap_code.push_back('D');
      }

      ref_pos += cigar_len;  // update sample position
      ref_ptr += cigar_len;

    } else if (cigar->Type == 'I') { // insertion


      MakeAllele(alleles, ALLELE_INSERTION, ref_pos, cigar_len, read_ptr);

      mismatch_count += cigar_len;

      read_ptr += cigar_len;

    // handle other cigar element types
    } else if (cigar->Type == 'S') { // soft clip, clipped sequence present in the read not matching the reference
      read_ptr += cigar_len;

    } else if (cigar->Type == 'H') { // hard clip on the read, clipped sequence is not present in the read
      // the alignment position is the first non-clipped base.
      // thus, hard clipping seems to just be an indicator that we clipped something
      // here we do nothing
      //sp += l; csp += l;

    } else if (cigar->Type == 'N') { // skipped region in the reference not present in read, aka splice
      for (unsigned int i = 0; i < cigar_len; ++i) {
        ra.refmap_start.push_back(read_ptr);
        ra.refmap_code.push_back('D');
      }
      ref_pos += cigar_len;
      ref_ptr += cigar_len;
    }

  } // end cigar iter loop
  ra.refmap_start.push_back(read_ptr);
  ra.refmap_code.push_back('N');

  // backtracking if we have too many mismatches or if there are no recorded alleles
  if (alleles.empty() or
      ((float) mismatch_count / (float) ra.alignment.QueryBases.size()) > read_max_mismatch_fraction_ or
      ra.snp_count > read_snp_limit_) {
    ra.filtered = true;
    return;
  }

  ra.start = alleles.front().position;
  ra.end = alleles.back().position + alleles.back().ref_length;

  for (deque<Allele>::iterator allele = alleles.begin(); allele != alleles.end(); ++allele) {
    if (allele->type == ALLELE_REFERENCE) {
      for (unsigned int i = 0; i < allele->ref_length; ++i) {
        ra.refmap_has_allele[allele->position - ra.alignment.Position + i] = 'R';
        ra.refmap_allele[allele->position - ra.alignment.Position + i] = *allele;
      }
    } else {
      ra.refmap_has_allele[allele->position - ra.alignment.Position] = 'A';
      ra.refmap_allele[allele->position - ra.alignment.Position] = *allele;
    }
  }

}




void AlleleParser::MakeAllele(deque<Allele>& alleles, AlleleType type, long int pos, int length, const char *alt_sequence)
{

  int ref_length = (type == ALLELE_INSERTION) ? 0 : length;
  int alt_length = (type == ALLELE_DELETION) ? 0 : length;

  Allele new_allele(type, pos, ref_length, alt_length, alt_sequence);

  if (alleles.empty()) {
    // presently, it's unclear how to handle insertions and deletions
    // reported at the beginning of the read.  are these events actually
    // indicative of longer alleles?
    if (type & (ALLELE_INSERTION|ALLELE_DELETION|ALLELE_NULL))
      return;
    alleles.push_back(new_allele);
    return;
  }


  Allele& last_allele = alleles.back();

  // Rule: Null alleles and new reference alleles are unmergable
  if ((new_allele.type & (ALLELE_REFERENCE|ALLELE_NULL)) or last_allele.type == ALLELE_NULL) {
    alleles.push_back(new_allele);
    return;
  }

  if ((allowed_allele_types_ & ALLELE_MNP) and (last_allele.position + last_allele.ref_length == new_allele.position)) {

    // Rule: Form MNP from SNP/MNP + SNP
    if ((last_allele.type & (ALLELE_SNP|ALLELE_MNP)) and new_allele.type == ALLELE_SNP) {
      last_allele.ref_length += new_allele.ref_length;
      last_allele.alt_length += new_allele.alt_length;
      last_allele.type = ALLELE_MNP;
      return;
    }
  }

  if ((allowed_allele_types_ & ALLELE_COMPLEX) and (last_allele.position + last_allele.ref_length == new_allele.position)) {

    // Rule: Form MNP from SNP/MNP + SNP
    // if ((last_allele.type & (ALLELE_SNP|ALLELE_MNP)) and new_allele.type == ALLELE_SNP) {
    //   last_allele.ref_length += new_allele.ref_length;
    //   last_allele.alt_length += new_allele.alt_length;
    //   last_allele.type = ALLELE_MNP;
    //   return;
    // }

    // Rule: Form COMPLEX from SNP/MNP/INS/DEL/COMPLEX + SNP/INS/DEL
    if (last_allele.type != ALLELE_REFERENCE) {
      last_allele.ref_length += new_allele.ref_length;
      last_allele.alt_length += new_allele.alt_length;
      last_allele.type = ALLELE_COMPLEX;
      return;
    }

    // Rule: Form COMPLEX from SNP/MNP/INS/DEL/COMPLEX + ref + SNP/INS/DEL
    if (alleles.size() >= 2 and last_allele.type == ALLELE_REFERENCE and last_allele.ref_length <= (unsigned int)max_complex_gap_) {
      Allele& second_last_allele = *(alleles.end()-2);
      if (not (second_last_allele.type & (ALLELE_REFERENCE|ALLELE_NULL)) and
          (second_last_allele.position + second_last_allele.ref_length == last_allele.position)) {

        second_last_allele.ref_length += last_allele.ref_length + new_allele.ref_length;
        second_last_allele.alt_length += last_allele.alt_length + new_allele.alt_length;
        last_allele.type = ALLELE_COMPLEX;
        alleles.pop_back();
        return;
      }
    }
  }

  // Rule: Insertions and deletions following reference get the first base of the reference
  if (last_allele.type == ALLELE_REFERENCE and (new_allele.type & (ALLELE_INSERTION|ALLELE_DELETION))) {
    if (last_allele.ref_length == 1) {
      alleles.pop_back();
    } else {
      last_allele.ref_length--;
      last_allele.alt_length--;
    }
    new_allele.position--;
    new_allele.alt_sequence--;
    new_allele.ref_length++;
    new_allele.alt_length++;
  }

  alleles.push_back(new_allele);
}





void AlleleParser::PileUpAlleles(int allowed_allele_types, int haplotype_length, bool scan_haplotype,
    list<PositionInProgress>::iterator& position_ticket, int hotspot_window)
{

  allele_pileup_.clear();
  ref_pileup_.initialize_reference(position_ticket->pos, num_samples_);


  if (haplotype_length == 1) {

    //
    // Aggregate observed alleles. Basic, non-haplotype mode
    //

    const Alignment * __restrict__ rai_end = position_ticket->end;
    for (const Alignment * __restrict__ rai = position_ticket->begin; rai != rai_end; rai = rai->next) {
      if (rai->filtered)
        continue;

      int read_pos = position_ticket->pos - rai->alignment.Position;
      if (read_pos < 0 or read_pos >= (int)rai->refmap_has_allele.size())
        continue;

      if (rai->refmap_has_allele[read_pos] == 'R') {
        ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr);

      } else if (rai->refmap_has_allele[read_pos] == 'A') {
        const Allele& allele = rai->refmap_allele[read_pos];
        allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_);
      }
    }


  } else if (scan_haplotype) {

    //
    // Aggregate observed alleles. Preliminary haplotype mode
    //

    Alignment *rai_end = position_ticket->end;
    for (Alignment *rai = position_ticket->begin; rai != rai_end; rai = rai->next) {
      if (rai->filtered)
        continue;

      int read_pos = position_ticket->pos - rai->alignment.Position;
      if (read_pos < 0 or read_pos >= (int)rai->refmap_has_allele.size())
        continue;

      if (rai->refmap_has_allele[read_pos] == 'R') {
        Allele& allele = rai->refmap_allele[read_pos];
        long int start = allele.position;
        long int end = allele.position + allele.ref_length;
        if (start <= position_ticket->pos && end >= position_ticket->pos + haplotype_length) {
          ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr);
          continue;
        }
      }

      for (int i = 0; i < haplotype_length; ++i, ++read_pos) {
        if (read_pos >= (int)rai->refmap_has_allele.size())
          break;

        if (rai->refmap_has_allele[read_pos] == 'A') {
          Allele& allele = rai->refmap_allele[read_pos];
          allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_);
        }
      }
    }


  } else {

    //
    // Aggregate observed alleles. Final haplotype mode
    //

    Alignment *rai_end = position_ticket->end;
    for (Alignment *rai = position_ticket->begin; rai != rai_end; rai = rai->next) {
      if (rai->filtered)
        continue;

      int haplotypeEnd = position_ticket->pos + haplotype_length;
      if (rai->start > position_ticket->pos or rai->end < haplotypeEnd)
        continue;

      int read_start = position_ticket->pos - rai->alignment.Position;
      if (rai->refmap_code[read_start] == 'D')    // isDividedIndel
        continue;

      const char* start_ptr = rai->refmap_start[read_start];
      const char* end_ptr = rai->refmap_start[read_start+haplotype_length];

      Allele allele;
      allele.position = position_ticket->pos;
      allele.ref_length = haplotype_length;
      allele.alt_sequence = start_ptr; // Pointer to the beginning of alternate sequence
      allele.alt_length = end_ptr - start_ptr;

      if (allele.alt_length == 0)
        continue;

      if (allele.ref_length != allele.alt_length) {
        allele.type = ALLELE_COMPLEX; // anything non-reference will do
      } else {
        allele.type = ALLELE_REFERENCE;
        for (int pos = 0; pos < haplotype_length; ++pos) {
          if (rai->refmap_code[read_start+pos] != 'M') {
            allele.type = ALLELE_COMPLEX; // anything non-reference will do
            break;
          }
        }
      }

      if (allele.type == ALLELE_REFERENCE)
        ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr);
      else
        allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_);
    }
  }

  // Calculate coverage by sample

  coverage_by_sample_.resize(num_samples_);
  for (int sample = 0; sample < num_samples_; ++sample)
    coverage_by_sample_[sample] = ref_pileup_.samples[sample].coverage;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& genotype = I->second;
    for (int sample = 0; sample < num_samples_; ++sample)
      coverage_by_sample_[sample] += genotype.samples[sample].coverage;
  }


  //
  // Add hotspots. Identify if there is a indel, mnp, or complex allele
  //

  // Pre-fetch applicable hotspots into hotspot_alleles_
  while (hotspot_reader_->HasMoreVariants()) {
    if ((hotspot_reader_->next_chr() < position_ticket->chr) or
        (hotspot_reader_->next_chr() == position_ticket->chr and hotspot_reader_->next_pos() < position_ticket->pos)) {
      hotspot_reader_->FetchNextVariant();
      continue;
    }

    // if (hotspot_reader_->next_chr() == position_ticket->chr and hotspot_reader_->next_pos() < position_ticket->pos + /*hotspot_window*/ min(haplotype_length,15)) { //revert to 4.2
    if (hotspot_reader_->next_chr() == position_ticket->chr and hotspot_reader_->next_pos() <= position_ticket->pos + hotspot_window) {
      for (size_t i = 0; i < hotspot_reader_->next().size(); i++)
        hotspot_alleles_.push_back(hotspot_reader_->next()[i]);
      hotspot_reader_->FetchNextVariant();
      continue;
    }
    break;
  }

  // black_list_strand.clear();black_list_strand.push_back('.'); revert to 4.2
  black_list_strand = '.';
  hp_max_lenght_override_value = 0;
  strand_bias_override_value = 0.0;
  
  for (deque<HotspotAllele>::iterator a = hotspot_alleles_.begin(); a != hotspot_alleles_.end() /*and a->pos <= position_ticket->pos */; ++a) {
    //if (a->pos != position_ticket->pos)
    //  continue;
     
    //record bad-strand information if provided
    if(a->params.black_strand != '.') {
      // revert to 4.2
      // int pos_offset = a->pos - position_ticket->pos;
      // for(int i=black_list_strand.size();i<=pos_offset;i++) black_list_strand.push_back('.');
      //  if(black_list_strand[pos_offset] == '.')
      //   black_list_strand[pos_offset] = a->params.black_strand;
      // else if(a->params.black_strand != 'H' && black_list_strand[pos_offset] != a->params.black_strand)
      //   black_list_strand[pos_offset] = 'B';
      if(black_list_strand == '.')
	black_list_strand = a->params.black_strand;
      else if(a->params.black_strand != 'H' && black_list_strand != a->params.black_strand)
	black_list_strand = 'B';
      // end reversion to 4.2

      if(a->params.hp_max_length != 0) hp_max_lenght_override_value = a->params.hp_max_length;
      if(a->params.strand_bias > 0.0) strand_bias_override_value = a->params.strand_bias;
      
      continue;
    }
    
    HotspotAllele& allele = *a;
 
    if (not scan_haplotype) {

      int pos_offset = allele.pos - position_ticket->pos;
      if (pos_offset > 0) {
        allele.pos -= pos_offset;
        allele.ref_length += pos_offset;
        allele.alt = ref_reader_->substr(position_ticket->chr, position_ticket->pos, pos_offset) + allele.alt;
      }

      if (allele.ref_length < haplotype_length) {
        allele.alt += ref_reader_->substr(position_ticket->chr,
            allele.pos+allele.ref_length, haplotype_length-allele.ref_length);
        allele.ref_length = haplotype_length;
      }
    }

    allele_pileup_[Allele(allele.type,allele.pos,allele.ref_length,allele.alt.size(),allele.alt.c_str())].add_hotspot(allele, num_samples_);
  }

  //revert to 4.2; remove denovo alleles generated from the bad-strand
  if (black_list_strand != '.') {
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (not allele.is_hotspot) {
	if(black_list_strand == 'F') {
	  if( ((float)allele.coverage_fwd ) / ((float) allele.coverage) > .7)
	    allele.filtered = true;
	} else if(black_list_strand == 'R') {
	  if( ((float)allele.coverage_rev ) / ((float) allele.coverage) > .7 )
	    allele.filtered = true;
	} else if(black_list_strand == 'B') {
	  allele.filtered = true;
	}
      }
    }
  }
  // end reversion to 4.2

  
  if (only_use_input_alleles_) {
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (not allele.is_hotspot)
        allele.filtered = true;
      InferAlleleTypeAndLength(allele);
    }
    return;
  }


  //
  // Apply min_alt_count and min_alt_fraction filters and do pre-work for use_best_n_alleles filter
  //

  vector<long int> quals;
  quals.reserve(allele_pileup_.size()+1);
  quals.push_back(ref_pileup_.coverage);

  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;

    if (allele.alt_sequence.empty()) {
      allele.filtered = true;
      continue;
    }

    InferAlleleTypeAndLength(allele);

    if (not allele.is_hotspot) {

      if (not (allele.type & allowed_allele_types)) {
        allele.filtered = true;
        continue;
      }

      if (allele.coverage < min_alt_total_) {
        allele.filtered = true;
        continue;
      }

      long double min_fraction = (allele.type & (ALLELE_DELETION | ALLELE_INSERTION))
          ? min_indel_alt_fraction_ : min_alt_fraction_;
      // special somatic filter
     /* if (allele.type & (ALLELE_DELETION | ALLELE_INSERTION)){
        // loosen filter for hp repeats 0-3 where we expect high accuracy even on proton
        if (allele.hp_repeat_len>-1){
          min_fraction = 0.007*allele.hp_repeat_len*allele.hp_repeat_len + 0.01; // frequency adjusted by hp length
        }
        if (min_fraction>min_indel_alt_fraction_) //keep existing threshold
          min_fraction = min_indel_alt_fraction_;
      }*/
      bool reject = true;

      for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {

        if (allele.samples[sample_idx].coverage < min_alt_count_)
          continue;
        if (((float) allele.samples[sample_idx].coverage / (float) coverage_by_sample_[sample_idx]) < min_fraction)
          continue;
        reject = false;
        break;
      }
      if (reject) {
        allele.filtered = true;
        continue;
      }
    }

    if (allele.type == ALLELE_SNP)
      quals.push_back(allele.coverage);
  }

  //
  // Apply use_best_n_alleles filter
  //

  if (use_best_n_alleles_ and (int)quals.size() > use_best_n_alleles_) {
    int snps_to_skip = quals.size() - use_best_n_alleles_;
    std::nth_element(quals.begin(), quals.begin()+snps_to_skip, quals.end());

    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (allele.type == ALLELE_SNP and not allele.is_hotspot and allele.coverage < quals[snps_to_skip])
        allele.filtered = true;
    }
  }
}




void AlleleParser::InferAlleleTypeAndLength(AlleleDetails& allele)
{
  string ref_sequence = ref_reader_->substr(allele.chr, allele.position, allele.ref_length);
  int alt_length = allele.alt_sequence.length();
  int ref_length = allele.ref_length;

  while (alt_length > 1 and ref_length > 1 and allele.alt_sequence[alt_length-1] == ref_sequence[ref_length-1]) {
    --alt_length;
    --ref_length;
  }

  allele.minimized_prefix = 0;
  while (allele.minimized_prefix < alt_length-1 and allele.minimized_prefix < ref_length-1
      and allele.alt_sequence[allele.minimized_prefix] == ref_sequence[allele.minimized_prefix])
    ++allele.minimized_prefix;

  int prefix = allele.minimized_prefix;
  while (prefix < alt_length and prefix < ref_length and allele.alt_sequence[prefix] == ref_sequence[prefix])
    ++prefix;

  ref_length -= prefix;
  alt_length -= prefix;

  allele.repeat_boundary = 0;

  if (ref_length == 0 and alt_length == 0) {
    allele.type = ALLELE_REFERENCE;
    allele.length = allele.ref_length;
    return;
  }
  if (ref_length == 0) {
    allele.type = ALLELE_INSERTION;
    allele.length = alt_length;
    allele.repeat_boundary = ComputeRepeatBoundary(allele.alt_sequence.substr(prefix,alt_length),
        allele.chr, allele.position + prefix, 12, allele.hp_repeat_len);
    return;
  }
  if (alt_length == 0) {
    allele.type = ALLELE_DELETION;
    allele.length = ref_length;
    allele.repeat_boundary = ComputeRepeatBoundary(ref_sequence.substr(prefix,ref_length),
        allele.chr, allele.position + prefix, 12, allele.hp_repeat_len);

    return;
  }
  if (ref_length == 1 and alt_length == 1) {
    allele.type = ALLELE_SNP;
    allele.length = 1;
    return;
  }
  if (ref_length == alt_length) {
    allele.type = ALLELE_MNP;
    allele.length = ref_length;
    return;
  }
  allele.type = ALLELE_COMPLEX;
  allele.length = max(ref_length, alt_length);

}


long AlleleParser::ComputeRepeatBoundary(const string& seq, int chr, long position, int max_size, long &my_repeat_len)
{
  int seq_length = seq.size();
  my_repeat_len = 1; // always at least 0/1 - don't go too low in frequency
  max_size = min(max_size, seq_length);

  const ReferenceReader::iterator& chr_begin = ref_reader_->begin(chr);
  const ReferenceReader::iterator& chr_end = ref_reader_->end(chr);
  long chr_size = ref_reader_->chr_size(chr);

  for (int i = 1; i <= max_size and (position+i) <= chr_size; ++i) {
    if (seq_length % i)
      continue;

    // Step 1. Check if repeat present in reference.

    int leftsteps = 0;
    ReferenceReader::iterator pos = ref_reader_->iter(chr, position);
    ReferenceReader::iterator repeat_newstart = pos;
    pos += i;

    while (true) {
      bool match = true;
      for (int j = 0; j < i; ++j) {
        --pos;
        --repeat_newstart;
        if (repeat_newstart < chr_begin or *repeat_newstart != *pos) {
          match = false;
          break;
        }
      }
      if (not match)
        break;
      leftsteps++;
    }

    int rightsteps = 0;

    pos = ref_reader_->iter(chr, position);
    repeat_newstart = pos;
    repeat_newstart += i;
    while (true) {
      bool match = true;
      for (int j = 0; j < i; ++j) {
        if (repeat_newstart >= chr_end or *pos != *repeat_newstart) {
          match = false;
          break;
        }
        ++pos;
        ++repeat_newstart;
      }
      if (not match)
        break;
      rightsteps++;
    }

    if (leftsteps + rightsteps == 0)
      continue;


    // Step 2. Check if indel compatible with repeat

    const char *seqptr = seq.c_str();

    ReferenceReader::iterator unit_start = ref_reader_->iter(chr, position);
    ReferenceReader::iterator unit_end = ref_reader_->iter(chr, position+i);
    ReferenceReader::iterator unitptr = unit_start;

    while (*seqptr) {
      if (*seqptr != *unitptr)
        break;
      ++seqptr;
      ++unitptr;
      if (unitptr == unit_end)
        unitptr = unit_start;
    }
    if (*seqptr == 0 and unitptr == unit_start){
      if (i==1)
        my_repeat_len = rightsteps+leftsteps+1;
      return position + i*(rightsteps+1) + 1;
    }
  }

  return position;
}



void AlleleParser::GenerateCandidates(deque<VariantCandidate>& variant_candidates,
    list<PositionInProgress>::iterator& position_ticket, int& haplotype_length)
{

  hotspot_alleles_.clear();
  haplotype_length = 1;

  char cb = ref_reader_->base(position_ticket->chr,position_ticket->pos);
  if (cb != 'A' && cb != 'T' && cb != 'C' && cb != 'G')
    return;


  GenerateCandidateVariant(variant_candidates, position_ticket, haplotype_length);

  // Fetch left-over hotspots

  long next_pos = min(position_ticket->pos + haplotype_length, position_ticket->target_end);
  while (hotspot_reader_->HasMoreVariants()) {

    if ((hotspot_reader_->next_chr() > position_ticket->chr) or
        (hotspot_reader_->next_chr() == position_ticket->chr and hotspot_reader_->next_pos() >= next_pos))
      break;

    vector<HotspotAllele> hotspot;
    for (size_t i = 0; i < hotspot_reader_->next().size(); i++)
      if(hotspot_reader_->next()[i].params.black_strand == '.')
        hotspot.push_back(hotspot_reader_->next()[i]);
    if (not hotspot.empty())
      FillInHotSpotVariant(variant_candidates, hotspot);

    hotspot_reader_->FetchNextVariant();
  }

}



void AlleleParser::GenerateCandidateVariant(deque<VariantCandidate>& variant_candidates,
    list<PositionInProgress>::iterator& position_ticket, int& haplotype_length)
{
  // Generate candidates
  if (position_ticket->pos == (position_ticket->target_end-1)) // Last base in target
    PileUpAlleles(allowed_allele_types_ & (ALLELE_REFERENCE|ALLELE_SNP), 1, false, position_ticket, 0);
  else
    PileUpAlleles(allowed_allele_types_, 1, false, position_ticket, 0);

  int total_cov = 0;
  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx)
    total_cov += coverage_by_sample_[sample_idx];

  bool hotspot_present = false;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    if (I->second.is_hotspot) {
      hotspot_present = true;
      break;
    }
  }

  if (not hotspot_present) {
    if (process_input_positions_only_) //skip all non-hot-spot positions
      return;

    if (total_cov == 0 || total_cov < min_coverage_)
      return;
  }


  if (not only_use_input_alleles_) {

    // Detect multi-base haplotypes and redo the pileup if needed

    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (not allele.filtered)
        haplotype_length = max(haplotype_length, (int)allele.ref_length);
    }

    if (haplotype_length > 1) {

      // NB: for indels in tandem repeats, if the indel sequence is
      // derived from the repeat structure, build the haplotype
      // across the entire repeat pattern.  This ensures we actually
      // can discriminate between reference and indel/complex
      // alleles in the most common misalignment case.  For indels
      // that match the repeat structure, we have cached the right
      // boundary of the repeat.  We build the haplotype to the
      // maximal boundary indicated by the present alleles.

      int old_haplotype_length = haplotype_length;
      int old_prefix = 0;
      int new_prefix = 0;
      do {
        do {
          old_haplotype_length = haplotype_length;

          PileUpAlleles(allowed_allele_types_, haplotype_length, true, position_ticket, new_prefix);

          for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
            AlleleDetails& allele = I->second;
            if (not allele.filtered) {
              long int hapend = max((long int) (allele.position + allele.ref_length), allele.repeat_boundary);
              haplotype_length = max((long int)haplotype_length, hapend - position_ticket->pos);
            }
          }

        } while (haplotype_length != old_haplotype_length);

        // now re-get the alleles
        PileUpAlleles(allowed_allele_types_, haplotype_length, false, position_ticket, new_prefix);

        old_prefix = new_prefix;
        new_prefix = -1;
        for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
          AlleleDetails& allele = I->second;
          if (allele.filtered)
            continue;
          if (new_prefix == -1)
            new_prefix = allele.minimized_prefix;
          else
            new_prefix = min(new_prefix, allele.minimized_prefix);
        }
        if (new_prefix > old_prefix)
          new_prefix = old_prefix + 1;

        /*
        if (position_ticket->pos >= 121432110 and position_ticket->pos < 121432120) {
          cout << "*********chr=" << position_ticket->chr << " pos=" << position_ticket->pos+1 << " hp=" << haplotype_length << " old_prefix=" << old_prefix << " new_prefix=" << new_prefix << endl;
          for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
            AlleleDetails& allele = I->second;
            if (allele.filtered)
              continue;
            cout << "Allele pos=" << allele.position+1 << " reflen=" << allele.ref_length << " alt=" << allele.alt_sequence << " prefix=" << allele.minimized_prefix << " HS=" << allele.is_hotspot << endl;
          }
        }
        */

      } while (old_prefix != new_prefix);
    }
  }

  // Alleles pileup is complete, now actually finalize the candidates

  int num_genotypes = 0;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    if (not I->second.filtered)
      num_genotypes++;
    if (I->second.is_hotspot)
      hotspot_present = true;
  }
  if (num_genotypes == 0)
    return;


  // Pad alleles to common reference length

  unsigned int common_ref_length = 1;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;
    if (not allele.filtered)
      common_ref_length = max(common_ref_length, allele.ref_length);
  }

  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;
    if (not allele.filtered and allele.ref_length < common_ref_length) {
      allele.alt_sequence += ref_reader_->substr(position_ticket->chr, allele.position+allele.ref_length,
          common_ref_length-allele.ref_length);
      allele.ref_length = common_ref_length;
    }
  }

  // Determine common prefix and suffix to trim away

  int common_prefix = 0;
  int common_suffix = 0;
  bool first = true;

  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;
    if (allele.filtered)
      continue;

    int current_start_pos = 0;
    int current_end_match = 0;
    int current_end_pos_ref = allele.ref_length;
    int current_end_pos_alt = allele.alt_sequence.size();
    ReferenceReader::iterator start_pos_ref = ref_reader_->iter(position_ticket->chr,position_ticket->pos);
    ReferenceReader::iterator end_pos_ref = ref_reader_->iter(position_ticket->chr,position_ticket->pos+allele.ref_length-1);

    while (current_end_pos_ref > 1 and current_end_pos_alt > 1) {
      if (*end_pos_ref != allele.alt_sequence[current_end_pos_alt-1])
        break;
      --current_end_pos_ref;
      --current_end_pos_alt;
      --end_pos_ref;
      ++current_end_match;
    }

    while (current_start_pos < current_end_pos_ref-1 and current_start_pos < current_end_pos_alt-1) {
      if (*start_pos_ref != allele.alt_sequence[current_start_pos])
        break;
      ++current_start_pos;
      ++start_pos_ref;
    }

    if (first) {
      common_prefix = current_start_pos;
      common_suffix = current_end_match;
      first = false;
    } else {
      common_prefix = min(common_prefix,current_start_pos);
      common_suffix = min(common_suffix,current_end_match);
    }
  }


  // Build Variant object

  variant_candidates.push_back(VariantCandidate(vcf_writer_->VariantInitializer()));
  VariantCandidate& candidate = variant_candidates.back();
  vcf::Variant& var = candidate.variant;

  candidate.variant.sequenceName = ref_reader_->chr_str(position_ticket->chr);
  candidate.variant.position = position_ticket->pos + common_prefix + 1;
  candidate.variant.id = ".";
  candidate.variant.filter = ".";
  candidate.variant.quality = 0.0;

  SetUpFormatString(candidate.variant);

  candidate.variant.isHotSpot = hotspot_present;
  if (candidate.variant.isHotSpot)
    candidate.variant.infoFlags["HS"] = true;

  candidate.variant.ref = ref_reader_->substr(position_ticket->chr, position_ticket->pos + common_prefix,
      common_ref_length - common_prefix - common_suffix);

  candidate.variant.info["RO"].push_back(convertToString(ref_pileup_.coverage));
  candidate.variant.info["SRF"].push_back(convertToString(ref_pileup_.coverage_fwd));
  candidate.variant.info["SRR"].push_back(convertToString(ref_pileup_.coverage_rev));

  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
    map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
    format["RO"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage));
    format["SRF"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage_fwd));
    format["SRR"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage_rev));
  }

  total_cov = ref_pileup_.coverage;

  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;

    total_cov += allele.coverage;
    if (allele.filtered)
      continue;

    if (common_prefix or common_suffix)
      candidate.variant.alt.push_back(allele.alt_sequence.substr(common_prefix, allele.alt_sequence.size() - common_suffix - common_prefix));
    else
      candidate.variant.alt.push_back(allele.alt_sequence);

    candidate.variant.info["TYPE"].push_back(allele.type_str());
    candidate.variant.info["LEN"].push_back(convertToString(allele.length));
    candidate.variant.info["AO"].push_back(convertToString(allele.coverage));
    candidate.variant.info["SAF"].push_back(convertToString(allele.coverage_fwd));
    candidate.variant.info["SAR"].push_back(convertToString(allele.coverage_rev));
 //   candidate.variant.info["JUNK"].push_back(convertToString(allele.hp_repeat_len));
     candidate.variant.info["HRUN"].push_back("0");
    if (allele.is_hotspot and allele.hotspot_params)
      candidate.variant_specific_params.push_back(allele.hotspot_params->params);
    else {
    candidate.variant_specific_params.push_back(VariantSpecificParams());
    // candidate.variant_specific_params.back().black_strand = black_list_strand.size()>(unsigned)allele.minimized_prefix ? black_list_strand[allele.minimized_prefix] : '.'; // revert to 4.2
    if(hp_max_lenght_override_value > 0) 
    {
       candidate.variant_specific_params.back().hp_max_length = hp_max_lenght_override_value;
       candidate.variant_specific_params.back().hp_max_length_override = true;
       }
    if(strand_bias_override_value > 0) 
    {
       candidate.variant_specific_params.back().strand_bias = strand_bias_override_value;
       candidate.variant_specific_params.back().strand_bias_override = true;
       
       }
    }

    for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
      map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
      format["AO"].push_back(convertToString(allele.samples[sample_idx].coverage));
      format["SAF"].push_back(convertToString(allele.samples[sample_idx].coverage_fwd));
      format["SAR"].push_back(convertToString(allele.samples[sample_idx].coverage_rev));
    }
  }
  candidate.variant.info["DP"].push_back(convertToString(total_cov));

  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
    map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
    format["DP"].push_back(convertToString(coverage_by_sample_[sample_idx]));
  }

}


void AlleleParser::FillInHotSpotVariant(deque<VariantCandidate>& variant_candidates, vector<HotspotAllele>& hotspot)
{
  // Set position_upper_bound to protect against allele reordering
  if (not variant_candidates.empty()) {
    VariantCandidate& previous_candidate = variant_candidates.back();
    previous_candidate.position_upper_bound = hotspot[0].pos + 1;
  }


  variant_candidates.push_back(VariantCandidate(vcf_writer_->VariantInitializer()));
  VariantCandidate& candidate = variant_candidates.back();

  candidate.variant.isHotSpot = true;
  candidate.variant.sequenceName = ref_reader_->chr_str(hotspot[0].chr);
  candidate.variant.position = hotspot[0].pos + 1;
  candidate.variant.id = ".";
  candidate.variant.filter = ".";
  candidate.variant.quality = 0.0;

  SetUpFormatString(candidate.variant);

  //copy ref and alt alleles from hotspot variant object

  candidate.variant.ref = ref_reader_->substr(hotspot[0].chr, hotspot[0].pos, hotspot[0].ref_length);
  for (size_t i = 0; i < hotspot.size(); i++) {

    const string& altbase = hotspot[i].alt;
    candidate.variant.alt.push_back(altbase);

    switch (hotspot[i].type) {
      case ALLELE_SNP:        candidate.variant.info["TYPE"].push_back("snp"); break;
      case ALLELE_MNP:        candidate.variant.info["TYPE"].push_back("mnp"); break;
      case ALLELE_DELETION:   candidate.variant.info["TYPE"].push_back("del"); break;
      case ALLELE_INSERTION:  candidate.variant.info["TYPE"].push_back("ins"); break;
      case ALLELE_COMPLEX:    candidate.variant.info["TYPE"].push_back("complex"); break;
      default:                candidate.variant.info["TYPE"].push_back("unknown");
    }
    candidate.variant.info["LEN"].push_back(convertToString(hotspot[i].length));
    candidate.variant.info["AO"].push_back("0");
    candidate.variant.info["SAF"].push_back("0");
    candidate.variant.info["SAR"].push_back("0");
    candidate.variant.info["HRUN"].push_back("0");

    candidate.variant_specific_params.push_back(hotspot[i].params);
  }

  candidate.variant.infoFlags["HS"] = true;

  int coverage = 0;
  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
    coverage += coverage_by_sample_[sample_idx];
    map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
    format["DP"].push_back(convertToString(coverage_by_sample_[sample_idx]));
    format["RO"].push_back(convertToString(coverage_by_sample_[sample_idx]));
    format["SRF"].push_back("0");
    format["SRR"].push_back("0");
    for (vector<string>::iterator aa = candidate.variant.alt.begin(); aa != candidate.variant.alt.end(); ++aa) {
      format["AO"].push_back("0");
      format["SAF"].push_back("0");
      format["SAR"].push_back("0");
    }
  }

  candidate.variant.info["DP"].push_back(convertToString(coverage));
  candidate.variant.info["RO"].push_back(convertToString(coverage));
  candidate.variant.info["SRF"].push_back("0");
  candidate.variant.info["SRR"].push_back("0");

}




