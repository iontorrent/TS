#include "AlleleParser.h"

#include <math.h>

#include "MiscUtil.h"
#include "OrderedVCFWriter.h"

//  convert a raw cigar to a real cigar.
static string convert2cigar(string s)
{
    std::ostringstream iss;

    if (s.size() > 0) { 
	char a = s[0];
    	unsigned int i, j;
    	for (i = 1, j = 1; i < s.size(); i++) {
	    if (a != s[i]) {
		iss << j << a;
		j = 1;
		a = s[i];
	    } else j++;
	}
	iss << j << a;
    } else {
	iss << "NAL";
    }
    return iss.str();
}


AlleleParser::AlleleParser(const ExtendParameters& parameters, const ReferenceReader& ref_reader,
    const SampleManager& sample_manager, OrderedVCFWriter& vcf_writer, HotspotReader& hotspot_reader)
{

  use_duplicate_reads_ = parameters.useDuplicateReads;
  use_best_n_alleles_ = parameters.useBestNAlleles;
  use_best_n_total_alleles_ = parameters.useBestNTotalAlleles;
  max_complex_gap_ = parameters.maxComplexGap;
  min_mapping_qv_ = parameters.min_mapping_qv;
  read_max_mismatch_fraction_ = parameters.readMaxMismatchFraction;
  read_snp_limit_ = parameters.read_snp_limit;
  min_alt_fraction_ = parameters.minAltFraction;
  min_indel_alt_fraction_ = parameters.minIndelAltFraction;
  min_alt_count_ = parameters.minAltCount;
  min_alt_total_ = parameters.minAltTotal;
  min_coverage_ = parameters.minCoverage;
  coverage_above_minC_ = false;
  my_examiner_ = NULL;
  new_hotspot_grouping = true;
  only_use_input_alleles_ = parameters.onlyUseInputAlleles;
  process_input_positions_only_ = parameters.processInputPositionsOnly;
  if (not parameters.black_listed.empty()) {
        blacked_var_.open(parameters.black_listed);
        if (not blacked_var_.is_open()) {
            cerr << "Fatal ERROR: black listed file " << parameters.black_listed << "cannot be openned for writing" << endl;
            exit(1);
        }
  }
  if (not parameters.candidate_list.empty()) {
        candidate_list_.open(parameters.candidate_list);
        if (not candidate_list_.is_open()) {
            cerr << "Fatal ERROR: candidate list file " << parameters.candidate_list << "cannot be openned for writing" << endl;
            exit(1);
        }
  }
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
  black_list_strand_ = '.';
  hp_max_lenght_override_value = 0;
  strand_bias_override_value = 0.0;
  merge_lookahead_ = parameters.mergeLookAhead;
  output_cigar_ = parameters.output_allele_cigar;
  nblackpos = 0;
  const ControlCallAndFilters& mc = parameters.my_controls;
  min_fake_hotspot_fr_ = min(min_alt_fraction_, min_indel_alt_fraction_);
  long double x = min(min(mc.filter_snp.min_allele_freq, mc.filter_mnp.min_allele_freq), min(mc.filter_hp_indel.min_allele_freq, mc.filter_hotspot.min_allele_freq));
  min_fake_hotspot_fr_ = min(min_fake_hotspot_fr_, x);
  // cerr << "fake hotspot fre" << min_fake_hotspot_fr_ << endl;
}

// -----------------------------------------------------------------------------

AlleleParser::~AlleleParser()
{
    flushblackpos(black_chr+1, 0);
    if (blacked_var_.is_open()) blacked_var_.close();
    if (candidate_list_.is_open()) candidate_list_.close();
    clean_heap_chars();
}

// -----------------------------------------------------------------------------

bool AlleleParser::GetNextHotspotLocation(int& chr, long& position) const
{
  if (not hotspot_alleles_.empty()) {
	chr = hotspot_alleles_.front().chr;
	position = hotspot_alleles_.front().pos;
	return true;
  }
  if (hotspot_reader_->HasMoreVariants()) {
    chr = hotspot_reader_->next_chr();
    position = hotspot_reader_->next_pos();
    return true;
  }
  return false;
}

// -------------------------------------------------------------------------

bool AlleleParser::BasicFilters(Alignment& ra, const TargetsManager * const targets_manager) const
{
  // Basic read filters

  if (!sample_manager_->IdentifySample(ra.alignment, ra.sample_index, ra.primary_sample)) {
    ra.filtered = true;
    return false;
  }
  if (ra.alignment.IsDuplicate() and not use_duplicate_reads_) {
    ra.filtered = true;
    return false;
  }
  if (!ra.alignment.IsMapped()) {
    ra.filtered = true;
    return false;
  }
  if (!ra.alignment.IsPrimaryAlignment()) {
    ra.filtered = true;
    return false;
  }

  // Filter on mapping_qv
  int eff_min_mapping_qv = min_mapping_qv_;
  bool is_min_mapping_qv_override = false;

  for (vector<int>::iterator target_idx_it = ra.target_coverage_indices.begin(); target_idx_it != ra.target_coverage_indices.end() and targets_manager != NULL; ++target_idx_it){
    if (targets_manager->unmerged[*target_idx_it].amplicon_param.min_mapping_qv_override){
      // Take the largest override (most stringent) if the read covers multiple amplicons.
      eff_min_mapping_qv = is_min_mapping_qv_override? max(eff_min_mapping_qv, targets_manager->unmerged[*target_idx_it].amplicon_param.min_mapping_qv) : targets_manager->unmerged[*target_idx_it].amplicon_param.min_mapping_qv;
      is_min_mapping_qv_override = true;
	}
  }

  if ((int) ra.alignment.MapQuality < eff_min_mapping_qv) {
    ra.filtered = true;
    return false;
  }
  return true;
}

// -------------------------------------------------------------------------

void AlleleParser::UnpackReadAlleles(Alignment& ra, const TargetsManager * const targets_manager) const
{
  // No need to waste time if the read is filtered
  if (ra.filtered)
    return;

  // Parse read into alleles and store them in generator-friendly format
  ra.is_read_allele_unpacked = true;
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

  // Determine is_clean_left_primer and is_clean_right_primer
  if (ra.best_coverage_target_idx >= 0 and targets_manager != NULL){
    // Determine is_clean_left_primer
    int amplicon_begin = targets_manager->unmerged[ra.best_coverage_target_idx].begin;
    int left_primer_len = amplicon_begin - ra.original_position;
    // is_clean_left_primer = true iff the left primer part of the read exactly matches the reference
    ra.is_clean_left_primer = false;
    vector<CigarOp>::const_iterator old_cigar_begin = ra.old_cigar.begin();
    if (left_primer_len > 0 and old_cigar_begin->Type == 'M' and (int) (old_cigar_begin->Length) >= left_primer_len){
      ra.is_clean_left_primer = true;
      ReferenceReader::iterator left_primer_ref_ptr = ref_reader_->iter(ra.alignment.RefID, (long) ra.original_position);
      for (int base_idx = 0; base_idx < left_primer_len; ++base_idx, ++left_primer_ref_ptr){
        if (ra.alignment.QueryBases[base_idx] != *left_primer_ref_ptr){
          ra.is_clean_left_primer = false;
          break;
        }
      }
    }

    // Determine is_clean_right_primer
    int amplicon_end = targets_manager->unmerged[ra.best_coverage_target_idx].end;
    int right_primer_len = ra.original_end_position - amplicon_end;
    // is_clean_right_primer = true iff the right primer part of the read exactly matches the reference
    ra.is_clean_right_primer = false;
    vector<CigarOp>::const_iterator old_cigar_last = ra.old_cigar.end() - 1;
    if (right_primer_len > 0 and old_cigar_last->Type == 'M' and (int) (old_cigar_last->Length) >= right_primer_len){
      ra.is_clean_right_primer = true;
      ReferenceReader::iterator right_primer_ref_ptr = ref_reader_->iter(ra.alignment.RefID, (long) ra.original_end_position - 1);
      for (int rev_base_idx = 0; rev_base_idx < right_primer_len; ++rev_base_idx, --right_primer_ref_ptr){
        if (ra.alignment.QueryBases[(int) ra.alignment.QueryBases.size() - 1 - rev_base_idx] != *right_primer_ref_ptr){
          ra.is_clean_right_primer = false;
          break;
        }
      }
    }
  }


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

    if (cigar->Type == 'M' || cigar->Type == '=') { // match or mismatch

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

  // read_snp_limit override
  int eff_read_snp_limit = read_snp_limit_;
  bool is_read_snp_limit_override = false;

  for (vector<int>::iterator target_idx_it = ra.target_coverage_indices.begin(); target_idx_it != ra.target_coverage_indices.end() and targets_manager != NULL; ++target_idx_it){
    if (targets_manager->unmerged[*target_idx_it].amplicon_param.read_snp_limit_override){
      // Take the minimum override (most stringent) if the read covers multiple amplicons.
      eff_read_snp_limit = is_read_snp_limit_override? min(eff_read_snp_limit, targets_manager->unmerged[*target_idx_it].amplicon_param.read_snp_limit) : targets_manager->unmerged[*target_idx_it].amplicon_param.read_snp_limit;
      is_read_snp_limit_override = true;
	}
  }

  // backtracking if we have too many mismatches or if there are no recorded alleles
  if (alleles.empty() or ra.alignment.QueryBases.size() == 0 or
      (mismatch_count > (int)((float)ra.alignment.QueryBases.size()) * read_max_mismatch_fraction_) or
      ra.snp_count > eff_read_snp_limit) {
    ra.filtered = true;
    return;
  }
  // Update the NM tag since the cigar may be changed.
  ra.alignment.EditTag("NM", "i", mismatch_count); // no mismatch

  ra.start = alleles.front().position;
  ra.end = alleles.back().position + alleles.back().ref_length;

  for (deque<Allele>::iterator allele = alleles.begin(); allele != alleles.end(); ++allele) {
    if (allele->type == ALLELE_REFERENCE) {
      for (unsigned int i = 0; i < allele->ref_length; ++i) {
        if ((unsigned int)(allele->position - ra.alignment.Position + i) >= ra.refmap_allele.size()) {break;}
        ra.refmap_has_allele[allele->position - ra.alignment.Position + i] = 'R';
        ra.refmap_allele[allele->position - ra.alignment.Position + i] = *allele;
      }
    } else {
      if ((unsigned int)(allele->position - ra.alignment.Position) >= ra.refmap_allele.size()) {break;}
      ra.refmap_has_allele[allele->position - ra.alignment.Position] = 'A';
      ra.refmap_allele[allele->position - ra.alignment.Position] = *allele;
    }
  }
}


// -----------------------------------------------------------------------------

void AlleleParser::MakeAllele(deque<Allele>& alleles, AlleleType type, long int pos, int length, const char *alt_sequence) const
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

  if ((allowed_allele_types_ & ALLELE_MNP) and (last_allele.position + (long int)last_allele.ref_length == new_allele.position)) {

    // Rule: Form MNP from SNP/MNP + SNP
    if ((last_allele.type & (ALLELE_SNP|ALLELE_MNP)) and new_allele.type == ALLELE_SNP) {
      last_allele.ref_length += new_allele.ref_length;
      last_allele.alt_length += new_allele.alt_length;
      last_allele.type = ALLELE_MNP;
      return;
    }
  }

  if ((allowed_allele_types_ & ALLELE_COMPLEX) and (last_allele.position + (long int)last_allele.ref_length == new_allele.position)) {

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
          (second_last_allele.position + (long int)second_last_allele.ref_length == last_allele.position)) {

        second_last_allele.ref_length += last_allele.ref_length + new_allele.ref_length;
        second_last_allele.alt_length += last_allele.alt_length + new_allele.alt_length;
        second_last_allele.type = ALLELE_COMPLEX;
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

// -----------------------------------------------------------------------------

void AlleleParser::BlacklistAlleleIfNeeded(AlleleDetails& allele, int total_cov, int total_f_cov, bool decom)
{
  // walk through positions to check hints
  if (hotspot_reader_->hint_empty())
    return;
  int buffer = 100;

  // find the next hint chromosome index >= this allele
  if (allele.is_hotspot) return; // no need to check black list for not spot variants ZZ
  hotspot_reader_->hint_start();
  while (hotspot_reader_->hint_more() &&
	 (hotspot_reader_->hint_chr_index() < allele.chr)) {
    //cout << "Catch up hotspot at " << hotspot_reader_->hint_chr_index() << " " << hotspot_reader_->hint_position() << ", allele pos "<< allele.chr << " " << allele.position << endl;
    hotspot_reader_->hint_pop();
    hotspot_reader_->hint_next();
  }

  // if hint chromosome index matches allele, find the next hint position >= this allele
  int real_var_pos = allele.position+ allele.minimized_prefix;
  while (hotspot_reader_->hint_more() &&
	 (hotspot_reader_->hint_chr_index() == allele.chr) &&
	 (hotspot_reader_->hint_position() < real_var_pos-2)) { //allow small padding
    //cout << "Catch up hotspot at " << hotspot_reader_->hint_position() << ", allele pos " << allele.position << endl;
    if (hotspot_reader_->hint_position() < real_var_pos - buffer) hotspot_reader_->hint_pop();
    hotspot_reader_->hint_next();
  }

  // see if this allele position matches this hint position
  if (hotspot_reader_->hint_more()) {
    int hint_chr_index = hotspot_reader_->hint_chr_index();
    long int hint_position = hotspot_reader_->hint_position();
    hotspot_reader_->hint_store_checkpt();
    //cout << "Test hotspot at " << hotspot_reader_->hint_position() << ", allele pos " << real_var_pos << endl;
    while  ((hint_chr_index == allele.chr) && (hint_position <= real_var_pos)) {
      // filter according to the hint
      //if (not allele.is_hotspot) { // no need to check, it is checked at the beginning
	int pref = hotspot_reader_->hint_prefix(); 
	long int rlen = hotspot_reader_->hint_rlen();
	string alt = hotspot_reader_->hint_alt();
        bool match = true;
        if (pref+hint_position < real_var_pos) goto label_next_black;
	if (hint_position < real_var_pos) {
	    int dif = real_var_pos-hint_position;
	    alt = alt.substr(dif);
	    rlen -= dif;
	}
	if (rlen > 0 and alt.size() > 0) {
	    int arlen = allele.ref_length-allele.minimized_prefix;
	    if (arlen < rlen) {
		string a = allele.alt_sequence.substr(allele.minimized_prefix)+ref_reader_->substr(allele.chr, hint_position+arlen, rlen-arlen);
		if (alt != a) match = false;
	    } else {
		// padding the hotspot allele to match the ref length of allele.
		if (arlen > rlen) alt += ref_reader_->substr(allele.chr, hint_position+rlen, arlen-rlen);
	    	if (alt != allele.alt_sequence.substr(allele.minimized_prefix)) match = false;
	    }
	    if (match) {
		double af, sd;
		if (hotspot_reader_->hint_getAF(af,sd) and allele.coverage > (af+4.0*sd)*(total_cov)){ match= false; break;}
		if (hotspot_reader_->hint_AF_good_allele(allele.coverage_fwd, allele.coverage_rev, total_f_cov, total_cov-total_f_cov)) { 
			match=false; 
			if (hotspot_reader_->hint_value() == SPEC_PARAM)  allele.param = hotspot_reader_->hint_param_ptr();
			break;
		}
	    }
	} 
	if (match)  {
	  switch ( hotspot_reader_->hint_value() )
	  {
	   case FWD_BAD_HINT:
	    if( ((float)allele.coverage_fwd ) / ((float) allele.coverage) > .7) {
	      allele.is_black_listed = 'F';  allele.filtered = true;
	    }
	    break;
	   case REV_BAD_HINT:
	    if( ((float)allele.coverage_rev ) / ((float) allele.coverage) > .7 ) {
	      allele.is_black_listed = 'R';  allele.filtered = true;
	    }
	    break;
	   case BOTH_BAD_HINT:
	   case SPEC_BAD_HINT:
	   case SPEC_PARAM:
	    allele.is_black_listed = 'B'; allele.filtered = true;
	    break;
	   default:
	    break;
	  }
	  break;
	}
   label_next_black:
	hotspot_reader_->hint_next();
	if (not hotspot_reader_->hint_more()) break;
	hint_chr_index = hotspot_reader_->hint_chr_index();
	hint_position = hotspot_reader_->hint_position();
    } // while
  } // if
}

AlleleDetails *AlleleParser::find_same_allele(AlleleDetails *a)
{
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
        AlleleDetails *ap = &(I->second);
	if (ap == a or ap->filtered) continue;
	if (a->alt_sequence == ap->alt_sequence) return ap;
    }
    return NULL;
}

bool AlleleParser::decompose_allele(AlleleDetails &allele, long hp, long rlen, int &ab, int &ae, int &alb, int &ale)
{
    if (allele.raw_cigar.size() == 0) return false;
    ab = 0; alb = 0;
    long pos = allele.position;
    char last = '.';
    while (pos < hp) {
	if (alb >= (int (allele.raw_cigar.size()))) return false;
	char x = allele.raw_cigar[alb];
	if (x == 'M') {
	    ab++; pos++;
	} else if (x == 'D') {
	    pos++;
	} else if (x == 'I') {
	    ab++;
	}
	last = x;
	alb++;
    }
    ae = allele.alt_sequence.size()-1; ale = allele.raw_cigar.size()-1;
    pos = allele.position+allele.ref_length-1;
    while (pos >hp+rlen-1) {
	if (ale < 0) return false;
	char x = allele.raw_cigar[ale];
        if (x == 'M') {
            ae--; pos--;
        } else if (x == 'D') {
            pos--;
        } else if (x == 'I') {
            ae--;
        }
	ale--;
    }
    if (ab > ae or alb > ale) return false; 
    if (last == 'D' and allele.raw_cigar[alb] == 'D') return false; // a deletion is broken in the middle, not a decomposition
    return true;
}

string AlleleParser::get_alt(AlleleDetails &allele, long hp, long rlen)
{
    int ab, ae, alb, ale;
    string s;
    s.clear();
    if (not decompose_allele(allele, hp, rlen, ab, ae, alb, ale)) return s;
    if (hp+rlen > allele.position+allele.ref_length) {
	string extra = ref_reader_->substr(allele.chr, allele.position+allele.ref_length, hp+rlen-(allele.position+allele.ref_length));
	return allele.alt_sequence.substr(ab)+extra;
    }
    return allele.alt_sequence.substr(ab, ae-ab+1);
}

bool AlleleParser::to_ref(AlleleDetails &allele, long hp, long rlen)
{
    int ab, ae, alb, ale;
    if (not decompose_allele(allele, hp, rlen, ab, ae, alb, ale)) return false;
    string ref = ref_reader_->substr(allele.chr, hp, rlen);
    string cig;
    cig.assign(ref.size(), 'M');
    if (hp+rlen >= allele.position+allele.ref_length) {
	int lz = ref.size()-(hp+rlen-(allele.position+allele.ref_length));
        allele.alt_sequence = allele.alt_sequence.substr(0, ab)+ref.substr(0, lz);
	allele.raw_cigar = allele.raw_cigar.substr(0, alb)+cig.substr(0, lz);
    } else {
	allele.alt_sequence = allele.alt_sequence.substr(0, ab)+ref+allele.alt_sequence.substr(ae+1);
	allele.raw_cigar = allele.raw_cigar.substr(0, alb)+cig+allele.raw_cigar.substr(ale+1);
    }
    return true;
}

void AlleleParser::SegmentBlacklist(int pos, int chr, int total_cov, int total_f_cov)
{
  hotspot_reader_->hint_start();
  while (hotspot_reader_->hint_more() &&
         (hotspot_reader_->hint_chr_index() == chr) &&
         (hotspot_reader_->hint_position() < pos)) {
    //cout << "Catch up hotspot at " << hotspot_reader_->hint_position() << ", allele pos " << allele.position << endl;
    hotspot_reader_->hint_next();
  }

  while (hotspot_reader_->hint_more()) {
    int hint_chr_index = hotspot_reader_->hint_chr_index();
    long int hint_position = hotspot_reader_->hint_position();
    long int rlen = hotspot_reader_->hint_rlen();
    string alt = hotspot_reader_->hint_alt();
    if (hint_chr_index != chr) break;

    if (alt.size() > 0) {
      bool need_cont = false;
      int tfc= 0, trc=0, tc=0;
      vector<AlleleDetails*> alist;
      // get all the unfiltered allele that fit the svb allele
      for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
        AlleleDetails& allele = I->second;
	int arlen = allele.ref_length;
	if (allele.filtered) continue;
	if (hint_chr_index == chr and hint_position < allele.position+ arlen) {
	    need_cont = true;
	    if (hint_position >= allele.position) {
		string a = get_alt(allele, hint_position, rlen);
	    	if (a == alt) {
		    tc += allele.coverage; tfc += allele.coverage_fwd; trc += allele.coverage_rev;
		    alist.push_back(&allele);
		}
	    }
	}
      }
      if (not alist.empty()) {
	double af, sd;
	if ((hotspot_reader_->hint_getAF(af,sd) and tc > (af+4.0*sd)*(total_cov)) or 
         	(hotspot_reader_->hint_AF_good_allele(tfc, trc, total_f_cov, total_cov-total_f_cov))) { 
	    alist.clear();
	} else {
	  char b = '.';
          switch ( hotspot_reader_->hint_value() )
          {
           case FWD_BAD_HINT:
            if( ((float)tfc ) / ((float) tc) > .7) b = 'F';
            break;
           case REV_BAD_HINT:
	    if( ((float)trc ) / ((float) tc) > .7) b = 'R';
            break;
           case BOTH_BAD_HINT:
           case SPEC_BAD_HINT:
	    b = 'B';
            break;
           default:
            break;
          }
	  if (b != '.') {
	     for (unsigned i = 0; i < alist.size(); i++) {
		if (alist[i]->is_hotspot) continue;
		string c_alt = alist[i]->alt_sequence, c_cig =  alist[i]->raw_cigar;	
		//cerr << "Segment error " << c_alt << endl;
		if (to_ref(*(alist[i]), hint_position, rlen)) { // change the blacked part to reference
		  AlleleDetails *fp = find_same_allele(alist[i]);
		  if (fp) {
		    //cerr << "Segement found " <<  alist[i]->alt_sequence << " " << fp->alt_sequence << endl;
		    alist[i]->alt_sequence = c_alt; alist[i]->raw_cigar = c_cig;
                    alist[i]->filtered= true;
                    alist[i]->is_black_listed = b;
		    fp->coverage += alist[i]->coverage;
		    fp->coverage_fwd += alist[i]->coverage_fwd;
		    fp->coverage_rev += alist[i]->coverage_rev;

      		    for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
        		fp->samples.at(sample_idx).coverage +=  alist[i]->samples.at(sample_idx).coverage;
		    }
		    //cerr << "Segement coverage " << fp->coverage << " " << alist[i]->coverage << " " << fp->coverage_fwd << " " << fp->coverage_rev << " " << fp->filtered << endl;
		  } else {
		    string ref = ref_reader_->substr(alist[i]->chr, alist[i]->position, alist[i]->ref_length); // ref
		    //cerr << "Segment not found " << ref << " " << alist[i]->alt_sequence << endl;
		    if (ref == alist[i]->alt_sequence) {
			alist[i]->alt_sequence = c_alt; alist[i]->raw_cigar = c_cig;
			alist[i]->filtered= true;
			alist[i]->is_black_listed = b;
		    } else { // modified allele, need to recheck the prefix and suffix and type
			InferAlleleTypeAndLength(*(alist[i]));
		    }
		  }
		} else {
                    alist[i]->filtered= true;
                    alist[i]->is_black_listed = b;
		}
	     } 
	  }
	}
      }

      if (not need_cont) break;
    }
    hotspot_reader_->hint_next();
  }
}
    

// -------------------------------------------------------------------

void AlleleParser::PileUpAlleles(int allowed_allele_types, int haplotype_length, bool scan_haplotype,
    list<PositionInProgress>::iterator& position_ticket, int hotspot_window)
{
  coverage_above_minC_ = false; total_cov_ = 0;
  int max_length = position_ticket->target_end-position_ticket->pos;
  if (max_length > 1000) max_length = 1000; 
  allele_pileup_.clear();
  clean_heap_chars();
  ref_pileup_.initialize_reference(position_ticket->pos, num_samples_);
  if (new_hotspot_grouping) hotspot_window = max(hotspot_window, haplotype_length-1);
  //cout << "hyplolate length" << haplotype_length << endl; // ZZ
  if (haplotype_length == 1) {

    //
    // Aggregate observed alleles. Basic, non-haplotype mode
    //

    const Alignment * __restrict__ rai_end = position_ticket->end;
    for (const Alignment * __restrict__ rai = position_ticket->begin; rai != rai_end; rai = rai->next) {
      if (rai == NULL) {position_ticket->end = NULL; break;}
      if (rai->filtered)
        continue;

      int read_pos = position_ticket->pos - rai->alignment.Position;
      if (read_pos < 0 or read_pos >= (int)rai->refmap_has_allele.size())
        continue;

      if (rai->refmap_has_allele[read_pos] == 'R') {
        ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, rai->read_count);

      } else if (rai->refmap_has_allele[read_pos] == 'A') {
	//const Allele& obs = rai->refmap_allele[read_pos];
	//string tmp1(obs.alt_sequence, obs.alt_length);
        const Allele& allele = rai->refmap_allele[read_pos];
	//string tmp(allele.alt_sequence, allele.alt_length);
	//cout << "Adding observation at " << allele.position <<  ", read_pos " << read_pos << ", alt_length " << allele.alt_length << " from " << tmp1 << " to " << tmp << endl;
	if (allele.position+allele.ref_length-position_ticket->pos > max_length) continue;
	allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_, rai->read_count);
      }
    }
  //cout << "First pass PileUpAlleles" << endl; // ZZ
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (allele.type != ALLELE_REFERENCE) {
        // cout << "New allele at " << allele.position  << ", ref_length " << allele.ref_length << " with " << allele.alt_sequence << endl; // ZZ
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
          ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, rai->read_count);
          continue;
        }
      }

      for (int i = 0; i < haplotype_length; ++i, ++read_pos) {
        if (read_pos >= (int)rai->refmap_has_allele.size())
          break;

        if (rai->refmap_has_allele[read_pos] == 'A') {
          Allele& allele = rai->refmap_allele[read_pos];
	  //string tmp(allele.alt_sequence, allele.alt_length);
	  //cout << "2nd pass Adding observation at " << allele.position <<  ", ref_len " << allele.ref_length << ", alt_length " << allele.alt_length << " to " << tmp << endl; // ZZ
	  if (allele.position+allele.ref_length-position_ticket->pos >= max_length) break; // out of boundary
          allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_, rai->read_count);
        }
      }
    }
    //cout << "Scan_haplotype pass PileUpAlleles" << endl; // ZZ
    /* ZZ
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (allele.type != ALLELE_REFERENCE) {
        cout << "New allele at " << allele.position  << ", ref_length " << allele.ref_length << " with " << allele.alt_sequence << endl;
      }
    }
    */

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
        ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, rai->read_count);
      else {
	//string tmp(allele.alt_sequence, allele.alt_length);
	//cout << "Adding observation at " << allele.position <<  ", read_pos " << read_start << ", alt_length " << allele.alt_length <<  " to " << tmp << endl; // ZZ
	// Generate raw cigar for non reference allele.
	string raw_cigar;
	for (int pos = 0; pos < haplotype_length; ++pos) {
          if (rai->refmap_code[read_start+pos] == 'D') raw_cigar.push_back('D');
	  else {
	    raw_cigar.push_back('M');
	  }
	    // innsertion
	    int j = rai->refmap_start[read_start+pos+1]-rai->refmap_start[read_start+pos];
	    j--;
	    while (j > 0) {
		j--;
		raw_cigar.push_back('I');
	    }
  	}
        allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_, rai->read_count, raw_cigar);
      }
    }
    /* ZZ
    cout << "final pass PileUpAlleles" << endl; 
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (allele.type != ALLELE_REFERENCE) {
        cout << "New allele at " << allele.position  << ", ref_length " << allele.ref_length << " with " << allele.alt_sequence << endl;
      }
    }
    */

  }

  // Calculate coverage by sample

  coverage_by_sample_.resize(num_samples_);
  int total_cov = ref_pileup_.coverage;
  int total_f_cov = ref_pileup_.coverage_fwd;
  for (int sample = 0; sample < num_samples_; ++sample)
    coverage_by_sample_[sample] = ref_pileup_.samples.at(sample).coverage;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& genotype = I->second;
    total_cov += genotype.coverage;
    total_f_cov += genotype.coverage_fwd;
    for (int sample = 0; sample < num_samples_; ++sample)
      coverage_by_sample_[sample] += genotype.samples.at(sample).coverage;
  }
  coverage_above_minC_ = (total_cov >= min_coverage_);
  total_cov_ = total_cov; // store for overriding.

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
  black_list_strand_ = '.';
  hp_max_lenght_override_value = 0;
  strand_bias_override_value = 0.0;

  //cout << "start one at " << position_ticket->pos << " " << haplotype_length << "scan " << scan_haplotype << endl;
  
  for (deque<HotspotAllele>::iterator a = hotspot_alleles_.begin(); a != hotspot_alleles_.end() /*and a->pos <= position_ticket->pos */; ++a) {
    //if (a->pos != position_ticket->pos)
    //  continue;
    if (a->pos >= position_ticket->pos+haplotype_length) continue;
    if (a->pos < position_ticket->pos) continue;
    //record bad-strand information if provided
    if(a->params.black_strand != '.') {
      // revert to 4.2
      // int pos_offset = a->pos - position_ticket->pos;
      // for(int i=black_list_strand.size();i<=pos_offset;i++) black_list_strand.push_back('.');
      //  if(black_list_strand[pos_offset] == '.')
      //   black_list_strand[pos_offset] = a->params.black_strand;
      // else if(a->params.black_strand != 'H' && black_list_strand[pos_offset] != a->params.black_strand)
      //   black_list_strand[pos_offset] = 'B';
      if(black_list_strand_ == '.')
	black_list_strand_ = a->params.black_strand;
      else if(a->params.black_strand != 'H' && black_list_strand_ != a->params.black_strand)
	black_list_strand_ = 'B';
      // end reversion to 4.2

      if(a->params.hp_max_length != 0) hp_max_lenght_override_value = a->params.hp_max_length;
      if(a->params.strand_bias > 0.0) strand_bias_override_value = a->params.strand_bias;
      
      continue;
    }
    
    HotspotAllele& allele = *a;
    /*
    if ((unsigned) allele.ref_length == allele.alt.size()) { // will be done once. SNP or mnp, trim down
	string t = ref_reader_->substr(position_ticket->chr, allele.pos, allele.ref_length);
	while (allele.ref_length > 1 and allele.alt[allele.ref_length-1] == t[allele.ref_length-1]) {
		allele.ref_length--; allele.alt.pop_back();
	}
    }
    */
    int pos_offset = 0, suf_offset = allele.suffix_padding;
    //has_hotspot = true; 
    bool changed = false;
    int opos = allele.pos;
    int oref_len = allele.ref_length;
    string old_alt = allele.alt;

    if (not scan_haplotype and haplotype_length > 1) {
      pos_offset = allele.pos - position_ticket->pos;
      if (pos_offset > 0) {
        allele.pos -= pos_offset;
        allele.ref_length += pos_offset;
	changed = true;
        allele.alt = ref_reader_->substr(position_ticket->chr, position_ticket->pos, pos_offset) + allele.alt;
      }

      if (allele.ref_length < haplotype_length) {
        allele.alt += ref_reader_->substr(position_ticket->chr,
            allele.pos+allele.ref_length, haplotype_length-allele.ref_length);

        suf_offset += haplotype_length - allele.ref_length;
        allele.ref_length = haplotype_length;
	changed = true;
      } //else if (allele.ref_length > haplotype_length) continue; // final stage shall not allow hotspot longer than the haplotype
    }
    //cerr << "haplength " << haplotype_length << " pos "  << opos << " " << old_alt << " " << oref_len << " "  << allele.ref_length << " " << allele.pos << " add hot " << allele.alt << pos_offset << " " <<  suf_offset << endl;
    allele_pileup_[Allele(allele.type,allele.pos,allele.ref_length,allele.alt.size(),get_heap_chars(allele.alt.c_str(),allele.alt.size()))].add_hotspot(allele, num_samples_, pos_offset, suf_offset);
    if (changed) {
	allele.pos = opos; allele.ref_length = oref_len; allele.alt = old_alt; // reset to the old, for future calls
    }
  }
// calculate prefix before doing black list ZZ
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      InferAlleleTypeAndLength(allele);
  }
//  end ZZ

  
  if (only_use_input_alleles_) {
    // in this case, no need to check black list since all the variants are filtered anyway.
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (not allele.is_hotspot)
        allele.filtered = true;
     // InferAlleleTypeAndLength(allele); // ZZ, no need to do it again
    }
    return;
  }

  //revert to 4.2; remove denovo alleles generated from the bad-strand
  //if (black_list_strand_ != '.') {  // Need to check black list for varints with non zero minimized_prefix evne if black_list_strand is '.'
  // ZZ: This check can be done at one level up. It is safer to do it here. No black listed allele will escape to outside this routine.
  //     On the other hand, if a part of repeat region is black listed, the 2-while-loop may not be able to expand to the whole repeat region?
  // After test, this is still the safer place, it will miss allele if we filter on the other place. HERE IS CALLED PLACE_1.
    bool decom = false;
    if ((allowed_allele_types & ALLELE_COMPLEX) and (not scan_haplotype)) decom = true;
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;

      //cout << "Blacklist " << allele.alt_sequence << " at " << allele.position+ allele.minimized_prefix << endl;
      BlacklistAlleleIfNeeded(allele, total_cov, total_f_cov, decom);
      //if (allele.is_hotspot) decom = false; // temp turn off segment in hotspot variant, need to turn on
    }
    if (decom) {
	SegmentBlacklist(position_ticket->pos, position_ticket->chr,  total_cov, total_f_cov);
    }
  //}
  // end reversion to 4.2



  //
  // Apply min_alt_count and min_alt_fraction filters and do pre-work for use_best_n_alleles filter
  //

  vector<long int> quals;
  vector<long int> quals_t;

  quals_t.reserve(allele_pileup_.size()+1);
  quals_t.push_back(ref_pileup_.coverage);
  quals.reserve(allele_pileup_.size()+1);
  quals.push_back(ref_pileup_.coverage);

  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;

    if (allele.alt_sequence.empty()) {
      allele.filtered = true;
      continue;
    }

    //InferAlleleTypeAndLength(allele);// ZZ no need to do that now again.

    if (not allele.is_hotspot) {
      if (not (allele.type & allowed_allele_types)) {
	//if (not(allele.type == ALLELE_COMPLEX and final_pass and has_hotspot)) { // in final pass with hotspot, allow complex haplotype // not use it for now.
          allele.filtered = true;
          continue;
	//}
      }
      if (filtered_by_coverage_novel_allele(allele)) {
	allele.filtered = true;
        continue;
      }
    }

    if (allele.type == ALLELE_SNP)
      quals.push_back(allele.coverage);
    else 
      quals_t.push_back(allele.coverage);
  }

  //
  // Apply use_best_n_alleles filter
  //
  //cout << "use best n= " << use_best_n_alleles_ << "qualsszie=" << quals.size(); // ZZ
  if ((use_best_n_alleles_ and scan_haplotype) or haplotype_length > 1) use_best_n_alleles_ += 2;
  if (use_best_n_alleles_ and (int)quals.size() > use_best_n_alleles_) {
    int snps_to_skip = quals.size() - use_best_n_alleles_;
    std::nth_element(quals.begin(), quals.begin()+snps_to_skip, quals.end());
    //cout << "coverage to skip" << quals[snps_to_skip]; // ZZ

    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (allele.type == ALLELE_SNP and not allele.is_hotspot and allele.coverage < quals[snps_to_skip])
        allele.filtered = true;
    }
  }
  if ((use_best_n_alleles_ and scan_haplotype) or haplotype_length > 1) use_best_n_alleles_ -= 2;
  if (use_best_n_total_alleles_ and (int) quals_t.size() > use_best_n_total_alleles_) {
    int alleles_to_skip = quals_t.size() - use_best_n_total_alleles_;
    std::nth_element(quals_t.begin(), quals_t.begin()+alleles_to_skip, quals_t.end());

    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (allele.type != ALLELE_SNP and not allele.is_hotspot and allele.coverage < quals_t[alleles_to_skip])
        allele.filtered = true;
    }
  }
}

bool AlleleParser::filtered_by_coverage_novel_allele(AlleleDetails& allele)
{
      float min_indel_f = min_indel_alt_fraction_, min_alt_f= min_alt_fraction_;
      int minC;
      if (allele.param) {
	if (allele.param->gen_min_coverage_override) {
	   if (total_cov_ < allele.param->gen_min_coverage) return true;
	} else if (not coverage_above_minC_) return true;
	if (allele.param->gen_min_indel_allele_freq_override) min_indel_f = allele.param->gen_min_indel_allele_freq;
	if (allele.param->gen_min_allele_freq_override) min_alt_f= allele.param->gen_min_allele_freq;
      } else if (not coverage_above_minC_) return true;
      if (allele.coverage < min_alt_total_) return true;

      long double min_fraction = (allele.type & (ALLELE_DELETION | ALLELE_INSERTION))
          ? min_indel_f : min_alt_f;
      if (min_fraction == 1.0) return true;
      if (allele.ref_length > 20) min_fraction /= 2.0;
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
      bool print = false;
      //if (allele.position >= 55242460 and  allele.position <= 55242467) { print = true; cerr << allele.position << endl; }

      for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
        //if (print) cerr << "sample_id" << sample_idx << "cov" << allele.samples.at(sample_idx).coverage << "min_co" << min_alt_count_ << endl;
        if (allele.samples.at(sample_idx).coverage < (long int)min_alt_count_)
          continue;
        // make sure zero coverage gets rejected
       //if (print) cerr << "cov" << allele.samples.at(sample_idx).coverage << "fraction=" << min_fraction << "cutoff=" << (long int)(min_fraction * (long double)coverage_by_sample_.at(sample_idx)) << endl;
        if ( allele.samples.at(sample_idx).coverage <= (long int)(min_fraction * (long double)coverage_by_sample_.at(sample_idx)) )
          continue;
        reject = false;
        break;
      }
      if (reject) return true;
      return false;
}

bool AlleleParser::is_fake_hotspot(AlleleDetails& allele)
{
         // if (allele.position >= 55242460 and  allele.position <= 55242467) cerr << "fake" << endl;
    long double s1 = min_indel_alt_fraction_, s2 = min_alt_fraction_;
    min_indel_alt_fraction_ = min_alt_fraction_ = min_fake_hotspot_fr_;

    bool r_v = filtered_by_coverage_novel_allele(allele);
    // return to original val
    min_indel_alt_fraction_ = s1; min_alt_fraction_ = s2;
    return r_v;
}

// -------------------------------------------------------------------------

void AlleleParser::InferAlleleTypeAndLength(AlleleDetails& allele) const
{
  string ref_sequence = ref_reader_->substr(allele.chr, allele.position, allele.ref_length);
  int alt_length = allele.alt_sequence.length();
  int ref_length = allele.ref_length;
  unsigned int end_trim = 0;

  while (alt_length > 1 and ref_length > 1 and allele.alt_sequence[alt_length-1] == ref_sequence[ref_length-1]) {
    if (allele.raw_cigar.size() > 0 and end_trim < allele.raw_cigar.size() and allele.raw_cigar[allele.raw_cigar.size()-1-end_trim] != 'M') break;
    --alt_length;
    --ref_length;
    end_trim++;
  }
  int m_suffix = allele.ref_length - ref_length;
  if (not allele.is_hotspot or m_suffix < allele.minimized_suffix) allele.minimized_suffix = m_suffix;

  int saved_m = allele.minimized_prefix;
  allele.minimized_prefix = 0;
  while (allele.minimized_prefix < alt_length-1 and allele.minimized_prefix < ref_length-1
      and allele.alt_sequence[allele.minimized_prefix] == ref_sequence[allele.minimized_prefix]) {
    if (allele.raw_cigar.size() > 0 and ((unsigned) allele.minimized_prefix) < allele.raw_cigar.size()-1 and (allele.raw_cigar[allele.minimized_prefix] != 'M' or allele.raw_cigar[allele.minimized_prefix+1] != 'M')) break;
    ++allele.minimized_prefix;
  }

  int prefix = allele.minimized_prefix;
  if (allele.is_hotspot and allele.minimized_prefix > saved_m) allele.minimized_prefix = saved_m; // if the alignment indicate smaller trim than hotspot, take the smaller
  while (prefix < alt_length and prefix < ref_length and allele.alt_sequence[prefix] == ref_sequence[prefix]) {
    if (allele.raw_cigar.size() > 0 and ((unsigned) prefix) < allele.raw_cigar.size() and allele.raw_cigar[prefix] != 'M') break;
    ++prefix;
  }

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
  if (ref_length == alt_length and (allele.raw_cigar.size() == 0 or allele.raw_cigar.size() == ref_length+prefix+end_trim)) {
    allele.type = ALLELE_MNP;
    allele.length = ref_length;
    return;
  }
  allele.type = ALLELE_COMPLEX;
  // complex must start with 'M' operation, if it is mismatch, need remove back the padding
  if (allele.minimized_prefix > 0 and allele.alt_sequence[allele.minimized_prefix] != ref_sequence[allele.minimized_prefix]) {
    allele.minimized_prefix--;
    ref_length++; alt_length++;
  }
  allele.length = max(ref_length, alt_length);

}

// -------------------------------------------------------------------------

long AlleleParser::ComputeRepeatBoundary(const string& seq, int chr, long position, int max_size, long &my_repeat_len) const
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

// -------------------------------------------------------------------------

void AlleleParser::GenerateCandidates(deque<VariantCandidate>& variant_candidates,
    list<PositionInProgress>::iterator& position_ticket, int& haplotype_length, CandidateExaminer* my_examiner, const TargetsManager * const targets_manager)
{

  //hotspot_alleles_.clear();
  while (not hotspot_alleles_.empty() and (hotspot_alleles_.front().chr < position_ticket->chr or (hotspot_alleles_.front().chr == position_ticket->chr and hotspot_alleles_.front().pos < position_ticket->pos)))
	hotspot_alleles_.pop_front();
  haplotype_length = 1;

  char cb = ref_reader_->base(position_ticket->chr,position_ticket->pos);
  if (cb != 'A' && cb != 'T' && cb != 'C' && cb != 'G')
    return;

  my_examiner_ = my_examiner; // setting as global as the only entry point shall be fine. ZZ
  start_next_ampl_ = end_cur_ampl_ = position_ticket->target_end+1;
  if (targets_manager) {
	bool possible = targets_manager->FindPossibleBreakIntervalInMerge(position_ticket->chr,  position_ticket->pos, end_cur_ampl_, start_next_ampl_);
	//cerr << "ZZcross " << end_cur_ampl_ << " " << start_next_ampl_ << endl;
  }

  GenerateCandidateVariant(variant_candidates, position_ticket, haplotype_length);
  /* move in GenerateCandidateVariant(), need for removing duplicate novo variant. ZZ
  // Fetch left-over hotspots

  long next_pos = min(position_ticket->pos + haplotype_length, position_ticket->target_end);
  while (hotspot_reader_->HasMoreVariants()) {

    //cout << "Fetching hotspots from  " << position_ticket->pos << " to " << next_pos << endl;
    for (size_t i = 0; i < hotspot_reader_->next().size(); i++){
      HotspotAllele hs = hotspot_reader_->next()[i];
      //cout << "   with alt " << hs.alt << " and ref_length " << hs.ref_length << endl;
    }
    
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
  */

}

static string bstrand(char a)
{
    if (a == 'F') return string("+");
    if (a == 'R') return string("-");
    return string("-+");
}

void AlleleParser::flushblackpos(int chr_ind, size_t pos) 
{
    if (not blacked_var_.is_open()) return;
    if (nblackpos == 0) { black_chr = chr_ind; return;}
    if (chr_ind != black_chr) pos = blackstart+MAXBLACK+10;
    while (blackstart < pos and nblackpos > 0) {
      int tcov = 0;
      AllelePool & ap = blackedAlleles[blackidx];
      if (ap.alleles.size() > 0) {
	tcov = ap.reffwd+ap.refrev;
        blacked_var_ << ref_reader_->chr_str(black_chr) << '\t' << blackstart+1 << "\t.\t" << ap.refseq  << '\t';
        for (unsigned int j = 0; j < ap.alleles.size(); j++) {
            if (j > 0) blacked_var_ << ',';
            blacked_var_ << ap.alleles[j].alt;
	    tcov += ap.alleles[j].rcov+ap.alleles[j].fcov;
        }
        blacked_var_ << '\t' <<  0 << '\t' << "NOCALL" << '\t' << "AO=";
        for (unsigned int j = 0; j < ap.alleles.size(); j++) {
            if (j != 0) blacked_var_ << ',';
            blacked_var_ << ap.alleles[j].fcov+ap.alleles[j].rcov;
        }
        blacked_var_ << ";DP=" << tcov << ";FR=";
        for (unsigned int j = 0; j < ap.alleles.size(); j++) {
	    if (j != 0) blacked_var_ << ',';
	    blacked_var_ << ".&SSEL" << bstrand(ap.alleles[j].bstrand);
	}
        blacked_var_ << ";SAF=";
        for (unsigned int j = 0; j < ap.alleles.size(); j++) {
            if (j != 0) blacked_var_ << ',';
            blacked_var_ << ap.alleles[j].fcov;
        }
        blacked_var_ << ";SAR=";
        for (unsigned int j = 0; j < ap.alleles.size(); j++) {
            if (j != 0) blacked_var_ << ',';
            blacked_var_ << ap.alleles[j].rcov;
        }
        blacked_var_ << ";SRF=" << ap.reffwd << ";SRR=" << ap.refrev << '\t' << "GT:GQ" << '\t' << "./.:0" << endl;
      }
      ap.clear();
      blackidx = nextblackpos(blackidx);
      nblackpos--;
      blackstart++;
    }
    if (nblackpos == 0) {
	blackstart = 0;
	blackidx = 0;
    }
    black_chr = chr_ind;
}

void AlleleParser::set_subset(VariantCandidate &v1, VariantCandidate &v, list<int> &co)
{
    co.sort();
    v1.variant.sequenceName = v.variant.sequenceName;
    v1.variant.position =  v.variant.position;
    v1.variant.isHotSpot = v.variant.isHotSpot;
    v1.variant.ref = v.variant.ref;
    list<int>::iterator it = co.begin();
    int ind = *it;
    int i = 0;
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); I++) {
        if (I->second.filtered) continue;
        if (i < ind) {
            I->second.filtered = true;
            i++; continue;
        }
        v1.variant.isAltHotspot.push_back(v.variant.isAltHotspot[ind]);
        v1.variant.isAltFakeHotspot.push_back(v.variant.isAltFakeHotspot[ind]);
        v1.variant_specific_params.push_back(v.variant_specific_params[ind]);
        v1.variant.alt.push_back(v.variant.alt[ind]);
        v1.variant.alt_orig_padding.push_back(v.variant.alt_orig_padding[ind]);
        it++;
        i++;
        if (it == co.end()) ind = 10000000;
        else ind = *it;
    }
}

// -------------------------------------------------------------------------

void AlleleParser::GenerateCandidateVariant(deque<VariantCandidate>& variant_candidates,
    list<PositionInProgress>::iterator& position_ticket, int& haplotype_length)
{
  // Generate candidates

  new_hotspot_grouping = true; // reset in case it is set due to end of target last run
  flushblackpos(position_ticket->chr, position_ticket->pos);
  int lookahead = merge_lookahead_;
  int lookahead_flow = 2;
  if (lookahead_flow > lookahead-1) lookahead_flow = lookahead-1;
  int not_look_ahead = 0;
  if (position_ticket->target_end > ref_reader_->chr_size(position_ticket->chr)) {
    cerr << "Target is outside the chromosome length" << endl;
    exit(1);
  }
  int max_length = position_ticket->target_end-position_ticket->pos;
  if (max_length > 1000) max_length = 1000;  // shall be enough
  //cout << "length " << max_length << " pos " <<  position_ticket->pos << endl;
  if (max_length <= 0) return; // shall not happen in theory, the lock seems to be not working properly, every thread seem to be doing one more run. fix later.
			      // This will be a gate keeper for now. ZZ 6/22/15
  string refstring;
  refstring = ref_reader_->substr(position_ticket->chr, position_ticket->pos, max_length);
  int scan_length = 1;
  if (position_ticket->pos == (position_ticket->target_end-1)) {// Last base in target
    PileUpAlleles(allowed_allele_types_ & (ALLELE_REFERENCE|ALLELE_SNP), 1, false, position_ticket, 0);
    not_look_ahead = 1;
  } else {
    PileUpAlleles(allowed_allele_types_, 1, false, position_ticket, 0);
    if (not (allowed_allele_types_ & ALLELE_COMPLEX)) {
	not_look_ahead = 1;
    } else if (lookahead == 0) {
	not_look_ahead = 1;
    } else if (lookahead > 0) {
	my_examiner_ = NULL; // if negative, use auto
    } else if (my_examiner_ == NULL) {
	not_look_ahead = 1;
    }
  } 
  if (not_look_ahead or my_examiner_ == NULL) new_hotspot_grouping = false;

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


  int old_prefix = 0;
  int new_prefix = 0;

  if (not only_use_input_alleles_) {

    // Detect multi-base haplotypes and redo the pileup if needed

    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      if (not allele.filtered)
        haplotype_length = max(haplotype_length, (int)allele.ref_length);
    }
    haplotype_length = min(max_length, haplotype_length);


    if (haplotype_length > not_look_ahead) {

      // NB: for indels in tandem repeats, if the indel sequence is
      // derived from the repeat structure, build the haplotype
      // across the entire repeat pattern.  This ensures we actually
      // can discriminate between reference and indel/complex
      // alleles in the most common misalignment case.  For indels
      // that match the repeat structure, we have cached the right
      // boundary of the repeat.  We build the haplotype to the
      // maximal boundary indicated by the present alleles.

      int old_haplotype_length = haplotype_length;
      do {
	bool skip_outter_loop = false;
	long int current_look_up_window = haplotype_length;
        do {
          //old_haplotype_length = haplotype_length;
	  int lkbase = 1;
	  int i, j;
	  if (not_look_ahead or haplotype_length == max_length) 
		lkbase = 0;
	  else {
	    // calculate look ahead
            if (my_examiner_) { 
                // set up the candidate
		if (haplotype_length == 0) break;
		bool cross_next_amplicon_start = false;
 		for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
            		AlleleDetails& allele = I->second;
			//cerr << "allele " << allele.position << " "  << allele.ref_length << " filtered " << allele.filtered << " is_hot " << allele.is_hotspot << endl;
            		if (not allele.filtered) {
			    //if (allele.is_hotspot and is_fake_hotspot(allele)) continue; //does not extend the look up window for fake HS. scan phase, not sure if it is fake
              		    long int hapend = max((long int) (allele.position + allele.ref_length), allele.repeat_boundary);
              		    current_look_up_window = max(current_look_up_window, hapend - position_ticket->pos);
            		}
          	}
		current_look_up_window = min((long int) max_length, current_look_up_window);
	 	if (haplotype_length < current_look_up_window) haplotype_length = current_look_up_window;
		// At this point, ZZ, if (postick, postick+hp_length) contain a starting amplicon position we need set scan_length to start_amp-postick, and do a scan again
		//cerr << "ZZcross " << position_ticket->pos+haplotype_length << "(" << position_ticket->pos << "+" << haplotype_length << ")" << start_next_ampl_ << endl;
		if (position_ticket->pos+haplotype_length > start_next_ampl_) {
		    current_look_up_window = scan_length = start_next_ampl_ - position_ticket->pos;
		    PileUpAlleles(allowed_allele_types_, current_look_up_window, true, position_ticket, new_prefix);
		    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
                        AlleleDetails& allele = I->second;
                        //cerr << "allele " << allele.position << " "  << allele.ref_length << " filtered " << allele.filtered << " is_hot " << allele.is_hotspot << endl;
                        if (not allele.filtered) {
                            //if (allele.is_hotspot and is_fake_hotspot(allele)) continue; //does not extend the look up window for fake HS. scan phase, not sure if it is fake
                            long int hapend = max((long int) (allele.position + allele.ref_length), allele.repeat_boundary);
                            current_look_up_window = max(current_look_up_window, hapend - position_ticket->pos);
                        }
                    }
		    haplotype_length = current_look_up_window;
		    cross_next_amplicon_start = true;
		}
		// ZZ added end

		PileUpAlleles(allowed_allele_types_, current_look_up_window, false, position_ticket, new_prefix);
		VariantCandidate v(vcf_writer_->VariantInitializer());
		//cerr << "place 1 " << max_length <<  " " << current_look_up_window  << " " << position_ticket->pos << " "  << position_ticket->target_end << endl;
		bool exist_allele = FillVariantFlowDisCheck(v, refstring, position_ticket, hotspot_present, current_look_up_window);
		if (not exist_allele) {
		    handle_candidate_list(position_ticket);
                    handle_black_out(refstring);
		    return;
		}
		my_examiner_->SetupVariantCandidate(v);
		// split,
		// if there are some subset can be done now:
		// haplotype_length = the legnth of the first group , then break to get he haplotype alleles in outer loop.ZZ
		//lkbase = my_examiner_->FindLookAheadEnd1()-position_ticket->pos-haplotype_length;
		list<list<int> > allele_groups_ready_to_go;
		vector<int>alleles_on_hold;
		int look_ahead_sliding_win_start, look_ahead_sliding_win_end;
		my_examiner_->LookAheadSlidingWindow0(allele_groups_ready_to_go, alleles_on_hold, look_ahead_sliding_win_start, look_ahead_sliding_win_end, position_ticket->pos+scan_length);
		//cout << position_ticket->pos << " ZZ hapl " << haplotype_length << " " << allele_groups_ready_to_go.size() << " " << alleles_on_hold.size() << endl;
		//printf( "ZZ:lkbase %d pos %ld hapl %d lAE1 %d\n", lkbase, position_ticket->pos, haplotype_length, my_examiner_->FindLookAheadEnd1());
		if (allele_groups_ready_to_go.size() > 0 or current_look_up_window == max_length or cross_next_amplicon_start) {
		    /*
		    long int save_pos = position_ticket->pos;
		    for (list<list<int> >::iterator it= allele_groups_ready_to_go.begin(); it != allele_groups_ready_to_go.end(); it++) {
			int start = -1, end = -1;
			for (list<int>::jt = it.begin(); jt != it.end(); jt++) {
			    pair<int,int> & p = v.variant.alt_orig_padding[*it];
			    if (start < 0) { start = p.first; end = p.second;}
			    else {
				start = min(start, p.first);
				end = min(end, p.second);
			    }
			}
			position_ticket->pos = save_pos+start;
			haplotype_length = v.variant.ref.size()-end-start;
			// generate haplotyped alleles
			PileUpAlleles(allowed_allele_types_, haplotype_length, false, position_ticket, new_prefix);
		 	handle_candidate_list();
			handle_black_out();
			int num_genotypes = 0;
  			for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    			    if (not I->second.filtered)
      				num_genotypes++;
  			}
  			if (num_genotypes == 0) continue;
			MakeVariant(variant_candidates, position_ticket, new_prefix, NULL);
		    }
		    haplotype_length =  position_ticket->pos-save_pos+haplotype_length;
		    position_ticket->pos = save_pos;
		    */
		    handle_candidate_list(position_ticket);
		    handle_black_out(refstring);
		    if (allele_groups_ready_to_go.size() > 0) {
			//cerr << "place 5 " << position_ticket->pos << " " << haplotype_length << " "  << allele_groups_ready_to_go.size() << endl;
		      //int max_to_go = 1;
		      for (list<list<int> >::iterator it= allele_groups_ready_to_go.begin(); it != allele_groups_ready_to_go.end(); it++) {
			int x = MakeVariant(variant_candidates, position_ticket, new_prefix, &(*it));
			//cerr << "ZZ pos=" << x << endl;
			/*
			max_to_go--;
			if (max_to_go <= 0) {
			    x -= position_ticket->pos;
			    haplotype_length = x;
			    return; // type to get more reads 
			}
			*/
		      }
		    }
		    if (alleles_on_hold.size() == 0) return;
		    if (current_look_up_window == max_length or cross_next_amplicon_start ) {
			int max_suf_padding = -1; 
			int max_pre_padding = 1000;
			if (cross_next_amplicon_start) {
			    // ZZ to add check
			    max_pre_padding = scan_length; 
			    if (position_ticket->pos+haplotype_length > end_cur_ampl_) {  max_suf_padding = position_ticket->pos+haplotype_length -end_cur_ampl_;}
			    //cerr << "ZZcross start pad " << position_ticket->pos <<"+"<< haplotype_length << " " << end_cur_ampl_ << " max suf padding:" << max_suf_padding << endl;
			}
			list<int> co;
			for (unsigned int i = 0; i < alleles_on_hold.size(); i++) 
			    if (v.variant.alt_orig_padding[alleles_on_hold[i]].first < max_pre_padding and v.variant.alt_orig_padding[alleles_on_hold[i]].second >= max_suf_padding) co.push_back(alleles_on_hold[i]); // bigger padding mean short allele 
			if (co.size() > 0) { 
                            if (co.size() < 20) {
                                //cerr << "place 6 " << position_ticket->pos << " " << haplotype_length << endl;
                                MakeVariant(variant_candidates, position_ticket, new_prefix, &co);
                            } else {
                                VariantCandidate v1(vcf_writer_->VariantInitializer());
                                set_subset(v1, v, co);
                                //bool exist_allele = FillVariantFlowDisCheck(v, refstring, position_ticket, hotspot_present, current_look_up_window, co);
                                my_examiner_->SetupVariantCandidate(v1);
                                list<list<int> > allele_groups;
                                my_examiner_->SplitCandidateVariant(allele_groups);
				//cerr << "place 6.5 split " << position_ticket->pos << " " << haplotype_length << endl;
                                for (list<list<int> >::iterator it = allele_groups.begin(); it != allele_groups.end(); it++) {
                                    MakeVariant(variant_candidates, position_ticket, new_prefix, &(*it));
                                }
                            }
			}
			if (cross_next_amplicon_start) haplotype_length = scan_length;
			return;
		    }
		    if (look_ahead_sliding_win_start-position_ticket->pos > current_look_up_window)  {
			cerr << "warning the skip ahead may be too much " << look_ahead_sliding_win_start-position_ticket->pos << " " << current_look_up_window<< endl;
		    }
		    haplotype_length = look_ahead_sliding_win_start-position_ticket->pos;
		    return;  // test ZZ, why do the recursion when it is more correct to release
		    if (look_ahead_sliding_win_start == current_look_up_window or haplotype_length >= max_length) return;
		    size_t old_pos = position_ticket->pos;
		    position_ticket->pos = look_ahead_sliding_win_start;
		    GenerateCandidates(variant_candidates, position_ticket, haplotype_length, my_examiner_);
		    haplotype_length += position_ticket->pos-old_pos;
		    position_ticket->pos = old_pos;
		    return;
		}
		lkbase = look_ahead_sliding_win_end - position_ticket->pos-haplotype_length;
		if (lkbase < 0) lkbase = 0; //Look ahead window behind current candidate window, we advance it to that. 
		current_look_up_window = min(max_length, haplotype_length+lkbase);
		scan_length = current_look_up_window;
		PileUpAlleles(allowed_allele_types_, current_look_up_window, true, position_ticket, new_prefix);
		old_haplotype_length = haplotype_length-1;
		continue;
	     } else {
		for (i = haplotype_length, j= 1; j < lookahead_flow and i < max_length-1; i++, lkbase++) {
                    if (refstring[i] != refstring[i+1]) j++;
            	}
            	if (lkbase < lookahead) lkbase = lookahead;
	     }
	  }
          PileUpAlleles(allowed_allele_types_, min(max_length, haplotype_length+lkbase), true, position_ticket, new_prefix);
	  old_haplotype_length = haplotype_length;
          for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
            AlleleDetails& allele = I->second;
            if (not allele.filtered) {
              long int hapend = max((long int) (allele.position + allele.ref_length), allele.repeat_boundary);
              haplotype_length = max((long int)haplotype_length, hapend - position_ticket->pos);
            }
          }
	  haplotype_length = min(haplotype_length, max_length);
	  if (haplotype_length-old_haplotype_length > 200) { 
		if (old_haplotype_length > 2) {
			int dec = (old_haplotype_length-8)/4;
			if (dec < 1) dec = 1;
			old_haplotype_length-= dec;
		}
		if (old_haplotype_length == 1) old_haplotype_length++;
		haplotype_length = old_haplotype_length; // too big a deletion, not deal with. ZZ. Also trim the end base that may be corrupt.
		skip_outter_loop = true;
	  }

        } while (haplotype_length != old_haplotype_length);

        // now re-get the alleles
        PileUpAlleles(allowed_allele_types_, haplotype_length, false, position_ticket, new_prefix);

        old_prefix = new_prefix;
        new_prefix = -1;
        //bool has_hotspot_now = false;
        for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
          AlleleDetails& allele = I->second;
          if (allele.filtered)
            continue;
	  if (allele.is_hotspot) {/*has_hotspot_now = true;*/ new_prefix = old_prefix; break;}
          if (new_prefix == -1)
            new_prefix = allele.minimized_prefix;
          else
            new_prefix = min(new_prefix, allele.minimized_prefix);
        }
	if (skip_outter_loop or new_hotspot_grouping) break; // special case of big del, stop extending.
        if (new_prefix > old_prefix)
          new_prefix = old_prefix + 1;
	/*
        if (position_ticket->pos >= 55242460 and position_ticket->pos < 55242463) {
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

// ZZ, Denote here as PLACE_2.
// this check can be done one level down inside PileUpAlleles() at PLACE_1. It will be safer there. Check this out.
// After checking, mostly due to skipping over "repeat region" after trying, there is chance of missing variants. 
// Revert to PLACE_1
/*
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;

      //cout << "Blacklist " << allele.alt_sequence << " at " << allele.position+ allele.minimized_prefix << endl;
      BlacklistAlleleIfNeeded(allele);
    }
*/
  /*
  if (blacked_var_.is_open()) {
    // ZZ; algorithm for keep a sorted output. The starting position determined by allele.minimized_prefix, so a simple vector keep all the output lines.
    // Just a sort by placing.
    // ZZZ2
    vector<string> alts[haplotype_length+1];
    vector<char> bstr[haplotype_length+1];
    vector<int> for_cov[haplotype_length+1], rev_cov[haplotype_length+1], reflen[haplotype_length+1]; 
    total_cov = ref_pileup_.coverage;
    int pos = 0;
    for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
      AlleleDetails& allele = I->second;
      total_cov += allele.coverage;
      pos = allele.position;
      if (55955440 < pos and 55955450 > pos) {
	fprintf(stderr, "%s %d %d %d %c min_pref=%d reflen=%d %ld\n", allele.alt_sequence.substr(allele.minimized_prefix).c_str(), pos+allele.minimized_prefix+1, haplotype_length, allele.filtered? 0:1, allele.is_black_listed, allele.minimized_prefix, allele.ref_length, position_ticket->pos);
	}
      if (allele.is_black_listed != '.') {
	int idx = allele.minimized_prefix;
	if (idx > haplotype_length) continue;
	alts[idx].push_back(allele.alt_sequence.substr(allele.minimized_prefix));
	for_cov[idx].push_back(allele.coverage_fwd);
	rev_cov[idx].push_back(allele.coverage_rev);
	reflen[idx].push_back(allele.ref_length);
	bstr[idx].push_back(allele.is_black_listed);
      }
    }
    for (int i = 0; i <= haplotype_length; i++) {
      if (alts[i].size() > 0) {
	blacked_var_ << ref_reader_->chr_str(position_ticket->chr) << '\t' << pos+i+1 << "\t.\t" << refstring.substr(pos+i-position_ticket->pos, reflen[i][0]-i)  << '\t';
	for (unsigned int j = 0; j < alts[i].size(); j++) {
	    if (j > 0) blacked_var_ << ',';
	    blacked_var_ << alts[i][j];
	}
	blacked_var_ << '\t' <<  0 << '\t' << "NOCALL" << '\t' << "AO=";
	for (unsigned int j = 0; j < for_cov[i].size(); j++) {
	    if (j != 0) blacked_var_ << ',';
	    blacked_var_ << for_cov[i][j]+rev_cov[i][j]; 
	}
	blacked_var_ << ";DP=" << total_cov << ";FR=.&SSEL" << bstrand(bstr[i][0]); 
	for (unsigned int j = 1; j < for_cov[i].size(); j++) blacked_var_ << ",.&SSEL" << bstrand(bstr[i][j]);
	blacked_var_ << ";SAF=";
	for (unsigned int j = 0; j < for_cov[i].size(); j++) {
            if (j != 0) blacked_var_ << ',';
            blacked_var_ << for_cov[i][j];
        }
	blacked_var_ << ";SAR=";
	for (unsigned int j = 0; j < for_cov[i].size(); j++) {
            if (j != 0) blacked_var_ << ',';
            blacked_var_ << rev_cov[i][j];
        }
	blacked_var_ << ";SRF=" << ref_pileup_.coverage_fwd << ";SRR=" << ref_pileup_.coverage_rev << '\t' << "GT:GQ" << '\t' << "./.:0" << endl; 
      }
    }
  }
  */
  handle_candidate_list(position_ticket);

  // ZZ remove duplicate and also save hotspot for fill in later.

  long next_pos = min(position_ticket->pos + haplotype_length, position_ticket->target_end);
  vector< vector<HotspotAllele> > save_hotspots;
  while (hotspot_reader_->HasMoreVariants()) {
    if ((hotspot_reader_->next_chr() > position_ticket->chr) or
        (hotspot_reader_->next_chr() == position_ticket->chr and hotspot_reader_->next_pos() >= next_pos))
      break;

    vector<HotspotAllele> hotspot1;
    for (size_t i = 0; i < hotspot_reader_->next().size(); i++) 
      if(hotspot_reader_->next()[i].params.black_strand == '.')
        hotspot1.push_back(hotspot_reader_->next()[i]);
    if (not hotspot1.empty()) {
        save_hotspots.push_back(hotspot1);
	vector<HotspotAllele>& hotspot = save_hotspots.back();
	for (unsigned int i = 0; i < hotspot.size(); i++) {
            HotspotAllele& allele = hotspot[i];

	    int pos_offset = allele.pos - position_ticket->pos;
	    int pos = allele.pos;
	    int ref_length = allele.ref_length;
	    string alt;
	    if (pos_offset > 0) {
	        pos -= pos_offset;
	        ref_length += pos_offset;
	        alt = ref_reader_->substr(position_ticket->chr, position_ticket->pos, pos_offset) + allele.alt;
	    } else {
		alt =  allele.alt;
	    }
	    if (ref_length < haplotype_length) {
	        alt += ref_reader_->substr(position_ticket->chr, pos+ref_length, haplotype_length-ref_length);
	        ref_length = haplotype_length;
            } else if (ref_length > haplotype_length) {
		if (allele.suffix_padding < ref_length-haplotype_length) continue;
		string ref = ref_reader_->substr(position_ticket->chr, pos, ref_length);
		int x = alt.size()-1;
		while (ref_length >  haplotype_length) {
		    if (alt[x] != ref[ref_length-1]) break;
		    ref_length--; alt.erase(x, 1); x--;
		}
		if (ref_length > haplotype_length) continue;
	    }	
	    // use the map to find duplicate, maybe we can get the coverage as well for AO. ZZ 6/29/15
	    pileup::iterator x = allele_pileup_.find(Allele(allele.type,pos,ref_length,alt.size(),alt.c_str()));
	    if (x == allele_pileup_.end()) continue;
	    if (x->second.filtered or x->second.is_hotspot) continue;
	    /*
	    if (x->second.is_hotspot) continue; // two hotspot are effective same, do nothing now.
	    x->second.filtered = true;
	    */
	    // instead of remove novel, remove the hotspot allele. marking them as new.
	    x->second.is_hotspot = true;
	    x->second.hotspot_params = &allele;
	    x->second.filtered = false;
	    x->second.is_black_listed = '.';
	    int new_suf = haplotype_length-allele.ref_length+allele.suffix_padding-pos_offset; // TS-17918
	    if (x->second.minimized_prefix >= pos_offset and x->second.minimized_suffix >=  new_suf) { // not change if the alignment shift the hotspot allele
		x->second.minimized_prefix = pos_offset; x->second.minimized_suffix = new_suf;
	    }
	    if (x->second.minimized_suffix < 0) x->second.minimized_suffix = 0;
	    allele.length = -1; 
 	}
	//save_hotspots.push_back(hotspot);
    }
      //FillInHotSpotVariant(variant_candidates, hotspot);
    hotspot_reader_->FetchNextVariant();
  }

  // we are doing output here ZZ. output black_listed varients
  // This is the right place to output since it is not always go through 3 stages of PileUp run in calling PileUp(). In some case, if just SNPs are found, first pass
  // will be just enough, so output at only the haplotype stage is not sufficient to capture all the  effective alleles, both for output black_listed or candidate list
  // Since each run of PileUpAlleles() runs black list, we save a is_black_listed flag, which is new so it is easy to output the alleles here.
  // Also here is right place since you want to output the novel alleles that are duplicated with hotspot allele for the candidate list.
  // --ZZ
  handle_black_out(refstring);

  int num_genotypes = 0;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    if (not I->second.filtered)
      num_genotypes++;
    if (I->second.is_hotspot)
      hotspot_present = true;
  }
  if (num_genotypes == 0) {
    // build hotspot variants before return
    for (unsigned int i = 0; i < save_hotspots.size(); i++) {
      if (not PileUpHotspotOnly(save_hotspots[i], position_ticket)) continue;
      FillInHotSpotVariant(variant_candidates, save_hotspots[i]);
    }

    return;
  }

//ZZ temp
  /*
  pileup temp_pileup;
  temp_pileup.clear();  
  string ref;
  bool notset = true;
  int rlen = 0;
  ref.clear();  
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele1 = I->second;
    if (allele1.filtered) continue;
    if (notset) {
	ref = ref_reader_->substr(position_ticket->chr, allele1.position, allele1.ref_length);
	rlen = allele1.ref_length;
	notset = false;
    }
    int p1, s1;
    get_prefix_suffix(ref, allele1.alt_sequence, p1, s1);
    pileup::iterator J = I;
    ++J;
    for (; J != allele_pileup_.end(); ++J) {
	AlleleDetails& allele2 = J->second;
	if (allele2.filtered) continue;
	// check need for get a new_allele chr
	int p2, s2;
	get_prefix_suffix(ref, allele2.alt_sequence, p2, s2);
	string s; 
	s.clear();
	if (p1+s2 > rlen) {
	    s+= allele2.alt_sequence.substr(0, allele2.alt_sequence.size()-s2);
	    p1-= p1+s2-rlen;
	    s+= allele1.alt_sequence.substr(p1, allele1.alt_sequence.size()-p1);
	} else if (p2+s1 > rlen) {
	    s+= allele1.alt_sequence.substr(0, allele1.alt_sequence.size()-s1);
            p2-= p2+s1-rlen;
            s+= allele2.alt_sequence.substr(p1, allele2.alt_sequence.size()-p2);
	} else continue;
	Allele new_allele(ALLELE_COMPLEX, allele1.position, rlen,s.size(),s.c_str());
	temp_pileup[new_allele].add_observation(new_allele, -1, 0, position_ticket->chr, num_samples_); // add a fake observation?? hope this works for now. ZZ.
    }
  }
  for (pileup::iterator I = temp_pileup.begin(); I != temp_pileup.end(); ++I) {
	const Allele& allele = I->first;
	pileup::iterator J = allele_pileup_.find(allele);
	if (J == allele_pileup_.end()) {
	    allele_pileup_[allele] = I->second;
	} 
  }
  */
  if (my_examiner_ and not_look_ahead == 0) {
                // set up the candidate
      	VariantCandidate v(vcf_writer_->VariantInitializer());
	//cerr << "place 2" << endl;
	bool exist_allele = FillVariantFlowDisCheck(v, refstring, position_ticket, hotspot_present, haplotype_length);
	if (exist_allele) {
	  my_examiner_->SetupVariantCandidate(v);
	  //split
	  list<list<int> > allele_groups;
	  my_examiner_->SplitCandidateVariant(allele_groups);
	  /* the split takes care of grouping hotspot alleles at the same location together.
	  if (new_hotspot_grouping) {
		// regroup orphan hotspots by position
		for (list<list<int> >::iterator it = allele_groups.begin(); it != allele_groups.end(); it++) {
		    if (it->size() > 1) continue;
		    if (v.variant.isAltFakeHotspot[*(it->begin())]) { // is hotspot
			list<list<int> >::iterator it1 = it;
			it1++;
			while (it1 != allele_groups.end()) {
			   // check some
			    if (it1->size() == 1 and v.variant.isAltFakeHotspot[*(it1->begin())]) { // is hotspot and size 1
				// insert into it
				int i = *(it1->begin());
			    	it->insert(it->begin(), i);
				it1 = allele_groups.erase(it1);
			    } else break; 
			}
		    }
		}
	  }
	  */
	  //cerr << "Number of splits " << allele_groups.size() << endl; 
	  //cerr << "place 3" << endl;
	  for (list<list<int> >::iterator it = allele_groups.begin(); it != allele_groups.end(); it++) {
	    MakeVariant(variant_candidates, position_ticket, new_prefix, &(*it));
	  }
	}
  } else {
	// cerr << "place 4" << endl;
      MakeVariant(variant_candidates, position_ticket, new_prefix, NULL);
  }

  for (unsigned int i = 0; i < save_hotspots.size(); i++) {
    if (not PileUpHotspotOnly(save_hotspots[i], position_ticket)) continue;
    FillInHotSpotVariant(variant_candidates, save_hotspots[i]);
  }

}

bool AlleleParser::FillVariantFlowDisCheck(VariantCandidate &v, string &refstring, list<PositionInProgress>::iterator& position_ticket, bool hotspot_present, int haplotype_length)
{
                v.variant.sequenceName = ref_reader_->chr_str(position_ticket->chr);
                v.variant.position = position_ticket->pos + 1;
                v.variant.isHotSpot = hotspot_present;
                unsigned int common_ref_len = haplotype_length; // same as hap-len from previous calculation
		for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    		    AlleleDetails& allele = I->second;
    		    if (allele.filtered) continue; 
		    unsigned int rlen = allele.ref_length + allele.position-position_ticket->pos;
		    if (rlen > refstring.size()) {
			allele.filtered = true;
			continue;
		    }
        	    common_ref_len = max(common_ref_len, rlen);
		}
                v.variant.ref = refstring.substr(0, common_ref_len);
		//cerr << v.variant.position  << " " << v.variant.ref.size() << v.variant.ref << endl;
                //fprintf(stderr, "pos %ld ref %s \n", v.variant.position, v.variant.ref.c_str());
                bool exist_allele = false;
		//cout << "ZZ:isAltFakeHotspot :";
		//cout << "Freebayes pos:" << v.variant.position << v.variant.ref << endl;
                for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
                        AlleleDetails& allele = I->second;
                        if (allele.filtered)
                                {continue;}
                        exist_allele = true;
                        //common_ref_len = max(common_ref_len, allele.ref_length);
                        //v.variant.alt.push_back(allele.alt_sequence);
                        v.variant.isAltHotspot.push_back(allele.is_hotspot); // Indicates the alt "allele" is HS or de novo?
                        if (allele.is_hotspot and is_fake_hotspot(allele)) v.variant.isAltFakeHotspot.push_back(true);
			else v.variant.isAltFakeHotspot.push_back(false);
			//cout << (v.variant.isAltFakeHotspot.back()? "true ":"false ") << allele.position << " " << position_ticket->pos;
                        if (allele.is_hotspot and allele.hotspot_params)
                                v.variant_specific_params.push_back(allele.hotspot_params->params);
                        else
                                v.variant_specific_params.push_back(VariantSpecificParams());
                        //padding
                        int fpad = allele.position-position_ticket->pos;
                        if (fpad == 0)
                            v.variant.alt.push_back(allele.alt_sequence+v.variant.ref.substr(allele.ref_length));
                        else v.variant.alt.push_back(refstring.substr(0, fpad)+allele.alt_sequence+v.variant.ref.substr(fpad+allele.ref_length)); //may need padding front
                        //cerr << v.variant.alt.back() << " " << allele.raw_cigar << endl;
			pair<int, int> p;
			/*if (allele.is_hotspot) p = make_pair(fpad, v.variant.ref.size()-fpad-allele.ref_length);
			else*/  p = make_pair(fpad+allele.minimized_prefix, v.variant.ref.size()-fpad-allele.ref_length+allele.minimized_suffix);
			v.variant.alt_orig_padding.push_back(p);
			//cerr << "FreeBayse " << fpad << " prefix " << p.first << " suffix " << p.second << " allele reflen" << allele.ref_length << " " << allele.minimized_suffix << endl;
                }
		//cout << endl;
		return exist_allele;

}
static void init_al(int &i, list<int>::iterator &it, list<int> *alist)
{
    i = 0;
    if (alist) it = alist->begin();
}

static bool check_allele_filter(AlleleDetails& allele, int &i, list<int>::iterator &it, list<int> *alist)
{
    if (alist == NULL) return (not allele.filtered);
    if (allele.filtered) return false;
    if (it == alist->end()) return false;
    if (i < *it) {
	i++; return false;
    }
    if (i == *it) {
	i++; it++;
	return true;
    }
    // impossible
    cerr << "The split list is not in order, negative or something else is wrong" << endl;
    exit(1);
}

int AlleleParser::MakeVariant(deque<VariantCandidate>& variant_candidates, list<PositionInProgress>::iterator& position_ticket, int new_prefix, list<int> *alist)
{

  // Pad alleles to common reference length
  // ZZ: need to enumerate hyplotype instead of simply padded by reference seq.
  // TODO: first try simple thing, for every pair of alleles, try to add an allele wit
  //       both alternative sequences.

  list<int>::iterator it;
  if (alist) {
	alist->sort();
	if (alist->size() == 0) {
	    cerr << "One split is empty" << endl;
	    exit(1);
	}
  }
  int adj_ro = 0, adj_srr = 0, adj_srf = 0;
  vector<int> sam_adj_ro, sam_adj_srr, sam_adj_srf;
  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
	sam_adj_ro.push_back(0);
	sam_adj_srr.push_back(0);
	sam_adj_srf.push_back(0);
  }
  unsigned int common_ref_length = 1;
  int i;
  init_al(i, it, alist);
  vector<AlleleDetails*> alleles_this;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;
    if (check_allele_filter(allele, i, it, alist)) {
	int suf = allele.minimized_suffix;
	if (allele.raw_cigar.size() > 0 and allele.ref_length == allele.alt_sequence.size() and allele.type == ALLELE_COMPLEX) {  // Looks like MNP but not, with cig
	    int len = allele.ref_length-1-suf;
	    string r = ref_reader_->substr(position_ticket->chr,allele.position+len, 2); // last base before suffix trim and 1 more.
	    if (allele.raw_cigar[allele.raw_cigar.size()-1-suf] != 'M' or allele.alt_sequence[len] != r[0]) { // if trim at the end of an indel
	 	if (suf == 0) { // no trim, need add a padding base to the allele
		    allele.ref_length++;
		    allele.raw_cigar += "M";
		    allele.alt_sequence += r.substr(1,1);
		} else { // if there is trimming, trim one less base
		    allele.minimized_suffix--; 
		}
	    }
	}
      	common_ref_length = max(common_ref_length, allele.ref_length);
    } else if (not allele.filtered) {
	adj_ro += allele.coverage;
	adj_srr += allele.coverage_rev;
	adj_srf += allele.coverage_fwd;
	for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
	    sam_adj_ro[sample_idx] += allele.samples[sample_idx].coverage;
	    sam_adj_srr[sample_idx] += allele.samples[sample_idx].coverage_rev;
	    sam_adj_srf[sample_idx] += allele.samples[sample_idx].coverage_fwd;
	} 
    }
  }
  init_al(i, it, alist);
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;
    if (check_allele_filter(allele, i, it, alist)) {
	if (allele.ref_length < common_ref_length) {
      	    allele.alt_sequence += ref_reader_->substr(position_ticket->chr, allele.position+allele.ref_length,
          	common_ref_length-allele.ref_length);
      	    allele.ref_length = common_ref_length;
	}
	// check redundant
	unsigned int i;
	for (i = 0; i < alleles_this.size(); i++) {
	    if (allele.alt_sequence == alleles_this[i]->alt_sequence) {
		alleles_this[i]->coverage += allele.coverage;
         	alleles_this[i]->coverage_rev += allele.coverage_rev;
        	alleles_this[i]->coverage_fwd += allele.coverage_fwd;
                for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
                    alleles_this[i]->samples[sample_idx].coverage += allele.samples[sample_idx].coverage;
                    alleles_this[i]->samples[sample_idx].coverage_rev += allele.samples[sample_idx].coverage_rev;
                    alleles_this[i]->samples[sample_idx].coverage_fwd += allele.samples[sample_idx].coverage_fwd;
                    allele.samples[sample_idx].coverage = allele.samples[sample_idx].coverage_rev = allele.samples[sample_idx].coverage_fwd = 0;
                }
                allele.coverage = allele.coverage_rev = allele.coverage_fwd = 0; // set all zero, but cannot set filtered. will screw up order
		if (allele.is_hotspot) alleles_this[i]->is_hotspot = true;
		if (alleles_this[i]->raw_cigar.size() == 0 and allele.raw_cigar.size() > 0) alleles_this[i]->raw_cigar = allele.raw_cigar;
		cerr << "Warning duplicate allele: " <<  allele.alt_sequence << " at " << allele.position << endl;
		break;
	    }
	}
	if (i >= alleles_this.size()) alleles_this.push_back(&(I->second));
    }
  }

  // Determine common prefix and suffix to trim away
  // split variants into multiple

  int common_prefix = 0;
  int common_suffix = 0;
  bool first = true;
  bool hotspot_present = false; // recheck
  init_al(i, it, alist);
  /*
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;
    if (not check_allele_filter(allele, i, it, alist))
      continue;
  */
  for (unsigned i = 0; i < alleles_this.size(); i++) {
    AlleleDetails& allele = *(alleles_this[i]);
    if (allele.is_hotspot) hotspot_present = true;

    int current_start_pos = 0;
    int current_end_match = 0;
    int current_end_pos_ref = allele.ref_length;
    int current_end_pos_alt = allele.alt_sequence.size();
    ReferenceReader::iterator start_pos_ref = ref_reader_->iter(position_ticket->chr,position_ticket->pos);
    ReferenceReader::iterator end_pos_ref = ref_reader_->iter(position_ticket->chr,position_ticket->pos+allele.ref_length-1);
    string refstring = ref_reader_->substr(position_ticket->chr, position_ticket->pos, allele.ref_length);

    while (current_end_pos_ref > 1 and current_end_pos_alt > 1) {
      if (*end_pos_ref != allele.alt_sequence[current_end_pos_alt-1] or (allele.raw_cigar.size() > 0 and allele.raw_cigar[allele.raw_cigar.size()-1-current_end_match] != 'M'))
        break;
      --current_end_pos_ref;
      --current_end_pos_alt;
      --end_pos_ref;
      ++current_end_match;
    }
    int delta = current_end_pos_ref-current_end_pos_alt;
    while (current_start_pos < current_end_pos_ref-1 and current_start_pos < current_end_pos_alt-1 and current_start_pos < allele.minimized_prefix) {
      if (*start_pos_ref != allele.alt_sequence[current_start_pos] or (allele.raw_cigar.size() > 0 and (allele.raw_cigar[current_start_pos] != 'M' or allele.raw_cigar[current_start_pos+1] != 'M')))
        break;
      // check left align repeat ZZ.
      /*
      if (delta > 1) {
	if (current_start_pos+delta < current_end_pos_ref-1 and refstring[current_start_pos+delta] ==  allele.alt_sequence[current_start_pos]) break;
      } else if (delta < -1) {
	if (current_start_pos-delta < current_end_pos_alt-1 and *start_pos_ref == allele.alt_sequence[current_start_pos-delta]) break;
      }
      */
      // end checking left align repeat ZZ
      ++current_start_pos;
      ++start_pos_ref;
    }
    // check repeat
    /*
    if (delta > 0) {
	int i;
	if (current_end_pos_ref-current_start_pos> delta) {
	  while (current_start_pos > delta) {
	    for (i = 0; i < delta; i++) {
		if (refstring[current_start_pos+i] != allele.alt_sequence[current_start_pos+i-delta]) break;
	    }
	    if (i >= delta) current_start_pos -=delta;
	    else break;
	  } 
	}
    } else if (delta < 0) {
	int i;
	delta *= -1;
	if (current_end_pos_alt- current_start_pos> delta) {
          while (current_start_pos > delta) {
            for (i = 0; i < delta; i++) {
                if (refstring[current_start_pos+i-delta] != allele.alt_sequence[current_start_pos+i]) break;
            }
            if (i >= delta) current_start_pos-=delta;
            else break;
          }
	}
    }
    */
    //cout << "find prefix " << allele.alt_sequence << " " << current_start_pos << " " << current_end_match << allele.raw_cigar[current_start_pos] << allele.raw_cigar[current_start_pos+1] << allele.minimized_prefix << endl;  
    if (allele.is_hotspot) {
	current_start_pos =  allele.minimized_prefix;
        current_end_match = min(current_end_match, allele.minimized_suffix);
    } else {
        //if (current_end_pos_ref-current_start_pos >= 2 and current_end_pos_alt-current_start_pos >=2 and current_start_pos > 0) current_start_pos--; // complex, mnp need one anchor base, per hotspot convention
    }
    //if (allele.ref_length < allele.alt_sequence.size() and current_start_pos != current_end_pos_ref-1 and current_start_pos > 0) current_start_pos--;
    //cout << "find prefix " << allele.alt_sequence << " " << current_start_pos << " " << current_end_match  << endl;   // ZZ
    if (first) {
      common_prefix = current_start_pos;
      common_suffix = current_end_match;
      first = false;
    } else {
      common_prefix = min(common_prefix,current_start_pos);
      common_suffix = min(common_suffix,current_end_match);
    }
    // cout << " common_prefix/suffix " << common_prefix << " " << common_suffix << endl; // ZZ
  }
  //if (hotspot_present) common_prefix = min(common_prefix, new_prefix); // no trimming the prefix if there is hotspot can be moved inside loop for efficiency. ZZ
  //if (hotspot_present) common_prefix = new_prefix; // has to trim to exact the location , for now ZZ.

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
  //cerr << candidate.variant.position << " " << candidate.variant.ref.size() << endl;

  candidate.variant.info["RO"].push_back(convertToString(ref_pileup_.coverage+adj_ro));
  candidate.variant.info["SRF"].push_back(convertToString(ref_pileup_.coverage_fwd+adj_srf));
  candidate.variant.info["SRR"].push_back(convertToString(ref_pileup_.coverage_rev+adj_srr));

  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
    map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
    format["RO"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage+sam_adj_ro[sample_idx]));
    format["SRF"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage_fwd+sam_adj_srf[sample_idx]));
    format["SRR"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage_rev+sam_adj_srr[sample_idx]));
  }

  int total_cov = ref_pileup_.coverage;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    total_cov += I->second.coverage;
  }
  /*
  init_al(i, it, alist);
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& allele = I->second;

    if (not check_allele_filter(allele, i, it, alist))
      continue;
  */
  for (unsigned i = 0; i < alleles_this.size(); i++) {
    AlleleDetails& allele = *(alleles_this[i]);

    if (common_prefix or common_suffix)
      candidate.variant.alt.push_back(allele.alt_sequence.substr(common_prefix, allele.alt_sequence.size() - common_suffix - common_prefix));
    else
      candidate.variant.alt.push_back(allele.alt_sequence);
    candidate.variant.alt_orig_padding.push_back(make_pair(0,0));
    candidate.variant.isAltHotspot.push_back(allele.is_hotspot); // Indicates the alt "allele" is HS or de novo?
    if (allele.is_hotspot and is_fake_hotspot(allele) ) candidate.variant.isAltFakeHotspot.push_back(true);
    else candidate.variant.isAltFakeHotspot.push_back(false);
    candidate.variant.info["PPD"].push_back(convertToString(allele.minimized_prefix-common_prefix));
    candidate.variant.info["SPD"].push_back(convertToString(allele.minimized_suffix>common_suffix? allele.minimized_suffix-common_suffix: 0));

    if (allele.raw_cigar.size() > 0) {
	if (allele.raw_cigar.size() <= (unsigned int) (common_prefix+common_suffix)) {
	    // raw cigar wrong
	    cerr << "ERROR: raw cigar wrong " << allele.raw_cigar << " common_prefix_suffix " << common_prefix << " " << common_suffix << endl;
    	    exit(1);
	} else {
	    // cerr << allele.raw_cigar << " " << common_prefix << " " << common_suffix << " "; 
	    if (common_suffix > 0) {
		allele.raw_cigar.erase(allele.raw_cigar.size()-common_suffix, common_suffix);
	    }
	    if (common_prefix > 0) {
		allele.raw_cigar.erase(0, common_prefix);
	    }
	    // cerr << allele.raw_cigar << endl;
	}
    }

    candidate.variant.info["TYPE"].push_back(allele.type_str());
    candidate.variant.info["LEN"].push_back(convertToString(allele.length));
    candidate.variant.info["AO"].push_back(convertToString(allele.coverage));
    candidate.variant.info["SAF"].push_back(convertToString(allele.coverage_fwd));
    candidate.variant.info["SAR"].push_back(convertToString(allele.coverage_rev));
    if (output_cigar_) candidate.variant.info["CIGAR"].push_back(convert2cigar(allele.raw_cigar));  // ZZ, placeholder
 //   candidate.variant.info["JUNK"].push_back(convertToString(allele.hp_repeat_len));
     candidate.variant.info["HRUN"].push_back("0");
    if (allele.is_hotspot and allele.hotspot_params)
      candidate.variant_specific_params.push_back(allele.hotspot_params->params);
    else if (allele.param) {
	candidate.variant_specific_params.push_back(*(allele.param));
    } else {
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
  return  candidate.variant.position;
}

void AlleleParser::PileUpAlleles(int pos, int haplotype_length,  list<PositionInProgress>::iterator& position_ticket)
{

  allele_pileup_.clear();
  clean_heap_chars();
  ref_pileup_.initialize_reference(pos, num_samples_);
  int haplotypeEnd = pos + haplotype_length;

    Alignment *rai_end = position_ticket->end;
    for (Alignment *rai = position_ticket->begin; rai != rai_end; rai = rai->next) {
      if (rai->filtered)
        continue;

      if (rai->start > pos or rai->end < haplotypeEnd)
        continue;

      int read_start = pos - rai->alignment.Position;
      // next 4 lines try to capture all possible alt, even the one starting with D, for the case alignment not left align
      int rd = read_start;
      /*
      for (; rd <= read_start+haplotype_length; rd++) {
	if (rai->refmap_code[rd] != 'D') break; 
      }
      if (rd > read_start+haplotype_length) continue;
      */
      // this is to be consistent with novel
      if (rai->refmap_code[read_start] == 'D')    // isDividedIndel
          continue;
      
      const char* start_ptr = rai->refmap_start[rd];
      const char* end_ptr = rai->refmap_start[read_start+haplotype_length];

      Allele allele;
      allele.position = pos;
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
        ref_pileup_.add_reference_observation(rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, rai->read_count);
      else {
        string tmp(allele.alt_sequence, allele.alt_length);
        //cout << "Adding observation at " << allele.position <<  ", read_pos " << read_start << ", alt_length " << allele.alt_length <<  " to " << tmp << endl; // ZZ
        allele_pileup_[allele].add_observation(allele, rai->sample_index, rai->alignment.IsReverseStrand(), position_ticket->chr, num_samples_, rai->read_count);
      }
    }

  coverage_by_sample_.resize(num_samples_);
  for (int sample = 0; sample < num_samples_; ++sample)
    coverage_by_sample_[sample] = ref_pileup_.samples.at(sample).coverage;
  for (pileup::iterator I = allele_pileup_.begin(); I != allele_pileup_.end(); ++I) {
    AlleleDetails& genotype = I->second;
    for (int sample = 0; sample < num_samples_; ++sample)
      coverage_by_sample_[sample] += genotype.samples.at(sample).coverage;
  }

}

// ----------------------------------------------------------------------

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
  int pos = hotspot[0].pos;
  int ref_length = hotspot[0].ref_length;
  for (size_t i = 0; i < hotspot.size(); i++) {
    if (hotspot[i].length < 0) continue;
    const string& altbase = hotspot[i].alt;
    candidate.variant.alt.push_back(altbase);
    candidate.variant.alt_orig_padding.push_back(make_pair(0,0));
    candidate.variant.isAltHotspot.push_back(true); // I am adding a hotspot allele.
    candidate.variant.isAltFakeHotspot.push_back(false);
    HotspotAllele& h_allele = hotspot[i];

    pileup::iterator x = allele_pileup_.find(Allele(h_allele.type,pos,ref_length,h_allele.alt.size(),h_allele.alt.c_str()));
    if (x != allele_pileup_.end()) {
	// find real type
	InferAlleleTypeAndLength(x->second);
	hotspot[i].type = x->second.type;
    } else {
	AlleleDetails a;
	a.add_hotspot(h_allele, num_samples_, 0, 0);
	InferAlleleTypeAndLength(a);
	hotspot[i].type = a.type;
    }
    //cout << "FillInHotspot with " <<  altbase << " at " <<  hotspot[0].pos << ", length " << hotspot[0].ref_length  << endl; 
    switch (hotspot[i].type) {
      case ALLELE_SNP:        candidate.variant.info["TYPE"].push_back("snp"); break;
      case ALLELE_MNP:        candidate.variant.info["TYPE"].push_back("mnp"); break;
      case ALLELE_DELETION:   candidate.variant.info["TYPE"].push_back("del"); break;
      case ALLELE_INSERTION:  candidate.variant.info["TYPE"].push_back("ins"); break;
      case ALLELE_COMPLEX:    candidate.variant.info["TYPE"].push_back("complex"); break;
      default:                candidate.variant.info["TYPE"].push_back("unknown");
    }
    candidate.variant.info["LEN"].push_back(convertToString(hotspot[i].length));
    candidate.variant.info["HRUN"].push_back("0");
    candidate.variant_specific_params.push_back(hotspot[i].params);
// TS-17918
    candidate.variant.info["PPD"].push_back("0");
    candidate.variant.info["SPD"].push_back(convertToString(hotspot[i].suffix_padding));
//

            if (x == allele_pileup_.end()) {
    		candidate.variant.info["AO"].push_back("0");
    		candidate.variant.info["SAF"].push_back("0");
    		candidate.variant.info["SAR"].push_back("0");
		if (output_cigar_) candidate.variant.info["CIGAR"].push_back("NAL");
	  	for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
    		   map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
	      	   format["AO"].push_back("0");
      		   format["SAF"].push_back("0");
      		   format["SAR"].push_back("0");
		}
		continue;
	    }
    AlleleDetails& allele = x->second;
    candidate.variant.info["AO"].push_back(convertToString(allele.coverage));
    candidate.variant.info["SAF"].push_back(convertToString(allele.coverage_fwd));
    candidate.variant.info["SAR"].push_back(convertToString(allele.coverage_rev));
    if (output_cigar_) candidate.variant.info["CIGAR"].push_back(convert2cigar(allele.raw_cigar));
    for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
      map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
      format["AO"].push_back(convertToString(allele.samples[sample_idx].coverage));
      format["SAF"].push_back(convertToString(allele.samples[sample_idx].coverage_fwd));
      format["SAR"].push_back(convertToString(allele.samples[sample_idx].coverage_rev));
    }
  }

  candidate.variant.infoFlags["HS"] = true;

  int coverage = 0;
  for (int sample_idx = 0; sample_idx < num_samples_; ++sample_idx) {
    coverage += coverage_by_sample_[sample_idx];
    map<string, vector<string> >& format = candidate.variant.samples[sample_manager_->sample_names_[sample_idx]];
    format["DP"].push_back(convertToString(coverage_by_sample_[sample_idx]));
    format["RO"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage));
    format["SRF"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage_fwd));
    format["SRR"].push_back(convertToString(ref_pileup_.samples[sample_idx].coverage_rev));
  }

  candidate.variant.info["DP"].push_back(convertToString(coverage));
  candidate.variant.info["RO"].push_back(convertToString(ref_pileup_.coverage));
  candidate.variant.info["SRF"].push_back(convertToString(ref_pileup_.coverage_fwd));
  candidate.variant.info["SRR"].push_back(convertToString(ref_pileup_.coverage_rev));

}




