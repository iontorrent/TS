/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantAssist.cpp
//! @ingroup  VariantCaller
//! @brief    Helpful routines for output of variants

#include "VariantAssist.h"
#include <limits>

MultiBook::MultiBook(){
  _invalid_reads = 0;
  _is_bi_dir_umt = false;
  _num_hyp_not_null = 0;
}

void MultiBook::Allocate(int num_hyp){
  _num_hyp_not_null = num_hyp;
  _my_book.resize(3); // 0: Bi-dir, 1: FWD, 2: REV
  for (unsigned int i_strand=0; i_strand <_my_book.size(); i_strand++){
    _my_book[i_strand].resize(_num_hyp_not_null);
  }

  my_position_bias.resize(_num_hyp_not_null);
  my_position_bias.resize(_num_hyp_not_null);
  for (int i_hyp = 0; i_hyp < _num_hyp_not_null; i_hyp++) {
    my_position_bias[i_hyp].rho = -1.0f;
    my_position_bias[i_hyp].pval = -1.0f;
  }
}



void MultiBook::ResetCounter(){
  _invalid_reads = 0;
  _is_bi_dir_umt = false;
  for (int i_strand = 0; i_strand < 3; ++i_strand){
	_my_book[i_strand].assign(_num_hyp_not_null, 0);
  }
}

void MultiBook::FillBiDirFamBook(const vector<vector<unsigned int> >& alt_fam_indices, const vector<EvalFamily>& my_eval_families, const vector<CrossHypotheses>& my_hypotheses){
	if (not _is_bi_dir_umt){
		return;
	}
	_my_bidir_fam_book.resize(alt_fam_indices.size());
	// Iterator over alternative alleles
	for (int i_alt = 0; i_alt < (int) alt_fam_indices.size(); ++i_alt){
		_my_bidir_fam_book[i_alt].resize(0);
		_my_bidir_fam_book[i_alt].reserve(alt_fam_indices[i_alt].size());
		// Iterate over families that support the alt allele
		for (vector<unsigned int>::const_iterator fam_vec_it = alt_fam_indices[i_alt].begin(); fam_vec_it != alt_fam_indices[i_alt].end(); ++fam_vec_it){
			// Iterate over reads in the family
			BiDirCov my_read_cov_by_strand = {0, 0, 0, 0};
			for (vector<unsigned int>::const_iterator read_idx_it = my_eval_families.at(*fam_vec_it).valid_family_members.begin(); read_idx_it != my_eval_families.at(*fam_vec_it).valid_family_members.end(); ++read_idx_it){
				// most_responsible = -2, -1 means OL, REF, respectively.
				int most_responsible = my_hypotheses.at(*read_idx_it).MostResponsible() - 2;
				int read_counter = my_hypotheses.at(*read_idx_it).read_counter;
				if (most_responsible == -2){
					// I don't count OL reads.
					continue;
				}
				if (my_hypotheses.at(*read_idx_it).strand_key == 0){
					my_read_cov_by_strand.fwd_total_cov += read_counter;
					if (most_responsible == i_alt){
						my_read_cov_by_strand.fwd_var_cov += read_counter;
					}
				}else{
					my_read_cov_by_strand.rev_total_cov += read_counter;
					if (most_responsible == i_alt){
						my_read_cov_by_strand.rev_var_cov += read_counter;
					}
				}
			}
			if (my_read_cov_by_strand.fwd_total_cov + my_read_cov_by_strand.rev_total_cov > 0){
				_my_bidir_fam_book[i_alt].push_back(my_read_cov_by_strand);
			}
		}
	}
}


// strand_id[i] = -1, 0, 1 indicates BIDIR, FWD, REV family (if use mol-tag) or read (if not use mol-tag and it can't be -1).
// my_book[0], my_book[1], my_book[2] indicate the coverage of BIRDIR, FWD, REV based on strand_id.
// If strand_id[i] > -1 (i.e., FWD or REV), then read/family i will contribute to both my_book[0] and my_book[strand_id[i] + 1].
// If strand_id[i] = -1 (i.e., BIDIR), then family i will contribute to my_book[0] only.
// Note that I don't allow mixture of bi-dir UMT and uni-dir UMT.
void MultiBook::AssignStrandToHardClassifiedReads(const vector<int> &strand_id, const vector<int> &read_id, const vector<int> &left, const vector<int> &right)
{
  ResetCounter();
  to_left.resize(read_id.size());
  to_right.resize(read_id.size());
  allele_index.resize(read_id.size());
  int valid_reads = 0;
  // record strand direction for each read with a valid allele
  for (unsigned int i_read = 0; i_read < read_id.size(); i_read++) {
    int most_responsible = read_id[i_read];
    if (most_responsible < 0){
      ++_invalid_reads;
      continue;
    }
    allele_index[valid_reads] = most_responsible;
    to_right[valid_reads] = right[i_read];
    to_left[valid_reads] = left[i_read];
    ++_my_book[0][most_responsible];
    if (strand_id[i_read] >= 0){
	  ++_my_book[strand_id[i_read] + 1][most_responsible];
    }
    else{
      _is_bi_dir_umt = true;
    }
    ++valid_reads;
  }

  to_left.resize(valid_reads);
  to_right.resize(valid_reads);
  allele_index.resize(valid_reads);

  // Sanity check: I don't allow mixture of bi-dir UMT and uni-dir UMT.
  if (_is_bi_dir_umt){
	// FWD cov + REV cov should both equal zero.
    assert(TotalCount(0) == 0 and TotalCount(1) == 0);
  }else{
	// BIDIR cov shoud equal FWD cov + REV cov.
    assert(TotalCount(-1) == (TotalCount(0) + TotalCount(1)));
  }
}

float VariantAssist::median(const std::vector<float>& values)
{
  std::vector<float> v = values;
  std::vector<float>::iterator  i = v.begin();
  std::vector<float>::size_type m = v.size() / 2;

  if (v.size() == 0)
    return std::numeric_limits<float>::quiet_NaN();
    
  if (v.size() == 1)
    return v[0];

  std::nth_element(i, i + m, v.end());
  float m1 = v[m];

  if ( v.size() % 2 == 0 ){   // Even number of elements
    std::nth_element(i, i + m - 1, v.end());
    return float(v[m-1] + m1)/2.0;
  }
  else {                           // Odd number of elements
    return float(v[m]);
  }
}

// References: Knuth Sec. 3.4.2 pp. 137-8
void VariantAssist::randperm(vector<unsigned int> &v, RandSchrange& rand_generator)
{
  unsigned int n = v.size();
  if (n < 1)
    return;

  for(unsigned int i = 0; i < n-1; i++) {
    // generate 0 <= c < n-i
    unsigned int c = (int)((double)rand_generator.Rand())/((double)rand_generator.RandMax + 1) * (n-i);
    // swap
    unsigned int t = v[i];
    v[i] = v[i+c];
    v[i+c] = t;
  }
}

double VariantAssist::partial_sum(const vector<double> &v, size_t n) {
  assert (n <= v.size());
  if(n==0)
    return std::numeric_limits<double>::quiet_NaN();

  double total = 0.0;
  for (size_t i=0; i<n; i++)
      total += v[i];
  return (total) ;
}


// compute the ranks of the values in the vector vals and overwrites vals
// If any values are tied, tiedrank computes their average rank.
// The tied rank is an adjustment for ties required by the nonparametric test
// using Mann-Whitney U (ranksum test) and computation of rho
void VariantAssist::tiedrank(vector<double> &vals)
{
  size_t n = vals.size();
  vector<double *>p(n);

  for(size_t i = 0; i < n; ++i)
  {
    p[i] = &vals[i];
  }
  std::sort(&p[0], &p[0]+n, VariantAssist::mycomparison());
  size_t i = 0;
  while(i < n)
  {
    size_t j = i;
    double tiedrank =  i+1.0;
    while (  (j<(n-1)) && (*p[j] == *p[j+1]) ){
      j++;
      tiedrank += j+1.0;
    }
    tiedrank = tiedrank/(j-i+1);
    for (size_t jj=i; jj<=j; ++jj) {
      *p[jj] = tiedrank;
    }
    i = j+1;
  }
}

// Wilcoxon 2-sample or Mann_Whitney U statistic, 1-sided version
// small values test whether ranks(var) < rank(ref)
double VariantAssist::MannWhitneyU(vector<float> &ref, vector<float> &var, bool debug){
   vector<double> both(ref.size() + var.size());
   for (unsigned int i = 0; i < ref.size(); i++)
     both[i] = ref[i];
   for (unsigned int i = ref.size(); i < both.size(); i++)
     both[i] = var[i - ref.size()];

   VariantAssist::tiedrank(both);

   if (debug) {
     fprintf(stdout, "ranks: ");
    for (int j=0; j< (int)both.size(); j++){
      fprintf(stdout, "%.1f ", (float)both[j]);
    }
    fprintf(stdout, "\n");
   }

   double maxU = (double)ref.size()*(double)var.size() + (double)ref.size()*(((double)ref.size()+1))/2.0;
   double U = maxU - VariantAssist::partial_sum(both, ref.size());
   if ((U < 0) || (U > maxU)) {
     fprintf(stdout, "Warning: overflow in VariantAssist::MannWhitneyU; ");
     fprintf(stdout, "ref.size()=%zu; ", ref.size());
     fprintf(stdout, "var.size()=%zu; ", var.size());
     fprintf(stdout, "both.size()=%zu; ", both.size());
     fprintf(stdout, "partial_sum=%f; ", VariantAssist::partial_sum(both, ref.size()));
     double newU = (U < 0) ? 0 : maxU;
     fprintf(stdout, "U=%f is set to %f\n", U, newU);
     U = newU;
   }
   return(U);
 }

// rho = estimate of P(ref) > var) + 0.5 P(var = ref)
double VariantAssist::MannWhitneyURho(vector<float> &ref, vector<float> &var, bool debug) {
  double U1 =  MannWhitneyU(ref, var, debug);
  double U2 = (double)ref.size()*(double)var.size()-U1; // large U2 means ranks(ref) > ranks(var)

  double rho = U2/((double)ref.size()*(double)var.size());
  if (debug) {
    fprintf(stdout, "U1=%f, U2=%f, rho=%f\n", U1, U2, rho);
  }
  if (rho < 0) {
    fprintf(stdout, "Warning: overflow in VariantAssist::MannWhitneyRho; ");
    fprintf(stdout, "rho=%f is set to 0\n", rho);     
    rho = 0;
  }
  return (rho);
}

float MultiBook::PositionBias(int i_alt) {
  ComputePositionBias(i_alt);
  return(my_position_bias[i_alt].rho);
}

float MultiBook::GetPositionBiasPval(int i_alt) {
  if (my_position_bias[i_alt].pval >= 0)
    return(my_position_bias[i_alt].pval);
  else
    return(-1); // uninitialized
}

float MultiBook::GetPositionBias(int i_alt) {
  if(my_position_bias[i_alt].rho >= 0)
    return(my_position_bias[i_alt].rho);
  else
    return(-1); // uninitialized
}

#define REFERENCE_ALLELE 0
// ignore multialleles, only use valid reads, distribute across reference
// might be worth cache'ing result if computationally expensive
void MultiBook::ComputePositionBias(int i_alt)
{
  // Bootstrap observed positions relative to soft clipped ends
  // to test if significantly different between reference and variant carrying reads
  // If we have count_ref indices for reference reads {r0, r1, ..., r_{count_ref-1}}
  // and count_var indices for variant reads {v0, v1, ..., v_{count_var -1}} that partition
  // all count = count_ref+count_var valid read indices {0, 1, ..., count-1}
  // Metric is Wilcoxon 2-sample U using length from variant position to untrimmed but
  // soft-clipped read end within each group, left and right separate
  //
  // Bootstrap is to generate a random permutation of the read indices and assign NR
  // to reference end positions, NV to end variant positions and calculate the metric
  //
  // Take i=0,..N-1 bootstrap samples of size count
  // assuming Null hypothesis is position on each type of read have the same
  // distribution
  //
  // pval = proportion of times bootstrap positional diff >= observed positional diff
  // rationale is that if this pval < threshold then there is real positional bias
  // 
  // repeat for positional bias to the left and to the right
  // return the minimum of the two pvals

  // calling convention is 0 to number of alt alleles, allele_index is 0 = reference
  assert(i_alt > 0);

  unsigned int count_ref = 0;
  unsigned int count_var = 0;
  unsigned int count = 0;

  vector<float> to_left_ref(to_left.size(),0);
  vector<float> to_left_var(to_left.size(),0);
  vector<float> to_right_ref(to_left.size(),0);
  vector<float> to_right_var(to_left.size(),0);
  vector<unsigned int> ix(to_left.size(),0);

  for (unsigned int i=0; i < to_left.size(); i++){
    if (allele_index[i] == REFERENCE_ALLELE) {
      to_left_ref[count_ref] = to_left[i];
      to_right_ref[count_ref] = to_right[i];
      count_ref++;
      ix[count] = i;
      count++ ;
    }
    if  (allele_index[i] == i_alt) {
      to_left_var[count_var] = to_left[i];
      to_right_var[count_var] = to_right[i];
      count_var++;
      ix[count] = i;
      count++ ;
    }
  }
  to_right_ref.resize(count_ref);
  to_left_ref.resize(count_ref);
  to_right_var.resize(count_var);
  to_left_var.resize(count_var);
  ix.resize(count);

  bool debug = false;

  if (debug) {
    fprintf(stdout, "count=%d, count_ref=%d, count_var=%d\n", (int)count, (int)count_ref, (int)count_var);
    fprintf(stdout, "to_left: ");
    for (int j=0; j< (int)to_left.size(); j++){
      fprintf(stdout, "%d ", (int)to_left[j]);
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "to_right: ");
    for (int j=0; j< (int)to_right.size(); j++){
      fprintf(stdout, "%d ", (int)to_right[j]);
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "i_alt=%d, allele: ", i_alt);
    for (int j=0; j< (int)allele_index.size(); j++){
      fprintf(stdout, "%d ", (int)allele_index[j]);
    }
    fprintf(stdout, "\n");
    fflush(stdout);
  }

  // pval computation
  if ( (count == 0) || (count_ref == 0) || (count_var == 0) ){
    my_position_bias[i_alt].rho = 0.5;
    my_position_bias[i_alt].pval = 1.0;
    return;
  }

  // look for inset reads with a bunch of variants close to an end
  #define to_end_cutoff 10
  float left_var = VariantAssist::median(to_left_var);
  float right_var = VariantAssist::median(to_right_var);

  float left_ref = VariantAssist::median(to_left_ref);
  float right_ref = VariantAssist::median(to_right_ref);

  if (debug) {
    fprintf(stdout, "left_var %f, right_var %f\n", left_var, right_var);
    fprintf(stdout, "left_ref %f, right_ref %f\n", left_ref, right_ref);
    fflush(stdout);
  }

  vector<float>to_ref;
  vector<float>to_var;
  vector<int>to_end;
  if (left_var < right_var) {  // use the left side
    if (left_var > to_end_cutoff){
      my_position_bias[i_alt].rho = 0.5;
      my_position_bias[i_alt].pval = 1.0;
      return;
    }
    if (left_var >= left_ref){ // only look at variants inside
      my_position_bias[i_alt].rho = 0.5;
      my_position_bias[i_alt].pval = 1.0;
      return;
    }

    if (debug)
      fprintf(stdout, "left_var %f < right_var %f\n", left_var, right_var);

    to_ref = to_left_ref;
    to_var = to_left_var;
    to_end = to_left;
  }
  else { // right_var <= left_var, use the right side
    if (right_var > to_end_cutoff){
      my_position_bias[i_alt].rho = 0.5;
      my_position_bias[i_alt].pval = 1.0;
      return;
    }

    if (right_var >= right_ref){ // only look at variants inside
      my_position_bias[i_alt].rho = 0.5;
      my_position_bias[i_alt].pval = 1.0;
      return;
    }

    if (debug)
      fprintf(stdout, "left_var %f >= right_var %f\n", left_var, right_var);
    to_ref = to_right_ref;
    to_var = to_right_var;
    to_end = to_right;
  }


  if (debug) {
    fprintf(stdout, "to_ref: ");
    for (int j=0; j< (int)to_ref.size(); j++){
      fprintf(stdout, "%d ", (int)to_ref[j]);
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "to_var ");
    for (int j=0; j< (int)to_var.size(); j++){
      fprintf(stdout, "%d ", (int)to_var[j]);
    }
    fprintf(stdout, "\n");
  }
  
  double observed = VariantAssist::MannWhitneyURho(to_ref, to_var, debug);

  RandSchrange rand_generator;
  rand_generator.SetSeed(1);

  if (debug) {
    fprintf(stdout, "observed: %f\n", observed);
  }
  long int N = 1000;
  double total = 0;
  for  (long int i=0; i<N; i++){    
    VariantAssist::randperm(ix, rand_generator);
    for (unsigned int ii=0; ii<count_ref; ii++){
      to_ref[ii] = to_end[ix[ii]];
    }
    for (unsigned int ii=count_ref; ii<count; ii++){
      to_var[ii-count_ref] = to_end[ix[ii]];
    }

    if (debug && i<10) {
      fprintf(stdout, "to_ref: ");
      for (int j=0; j< (int)to_ref.size(); j++){
	fprintf(stdout, "%d ", (int)to_ref[j]);
      }
      fprintf(stdout, "\n");
      fprintf(stdout, "to_var ");
      for (int j=0; j< (int)to_var.size(); j++){
	fprintf(stdout, "%d ", (int)to_var[j]);
      }
      fprintf(stdout, "\n");
    }
    double bootstrap =  VariantAssist::MannWhitneyURho(to_ref, to_var, (debug && i<10));
    if (debug && i<10) {
      fprintf(stdout, "bootstrap: %f\n", bootstrap);
    }

    // greater or equal to what we saw?
    if ( bootstrap >= observed )
      total++;
  }
  float pval =  total/N;
  if (debug) {
      fprintf(stdout, "total= %f, pval=%f\n", total, pval);
      fflush(stdout);
  }
  my_position_bias[i_alt].rho = observed;
  my_position_bias[i_alt].pval = pval;
  return;
}


float MultiBook::GetFailedReadRatio() const {
  int total_depth = _invalid_reads + TotalCount(-1);
  float fxx = (total_depth == 0)? 0.0f : (float) _invalid_reads / (float) total_depth;
  return fxx;
}

// STB for BI-DIR UMT.
// 1) STB is the average STB of the families that support the allele
// 2) When calculating the STB of one family, I use "total read counts" instead of the count of "ref + this alt".
float MultiBook::OldStrandBiasBiDirFam_(int i_alt, float tune_bias) const{
	float full_strand_bias = 0.0f;
	if (_my_bidir_fam_book[i_alt].empty()){
		return 0.0f;
	}
	for (vector<BiDirCov>::const_iterator my_cov_it = _my_bidir_fam_book[i_alt].begin(); my_cov_it != _my_bidir_fam_book[i_alt].end(); ++my_cov_it){
		full_strand_bias += fabs(ComputeTransformStrandBias(my_cov_it->fwd_var_cov, my_cov_it->fwd_total_cov, my_cov_it->rev_var_cov, my_cov_it->rev_total_cov, tune_bias));
	}
	full_strand_bias /= (float) _my_bidir_fam_book[i_alt].size();
	float old_style_strand_bias = (full_strand_bias + 1.0f) / 2.0f;
	return old_style_strand_bias;
}

float MultiBook::OldStrandBias(int i_alt, float tune_bias) const {
  if (_is_bi_dir_umt){
	  // STB for Bi-Dir UMT
	  return OldStrandBiasBiDirFam_(i_alt, tune_bias);
  }

  float full_strand_bias = fabs(ComputeTransformStrandBias(_my_book[1][i_alt+1], GetDepth(0,i_alt), _my_book[2][i_alt+1], GetDepth(1,i_alt), tune_bias));
  //@TODO: revert to full range parameters
  // this ugly transformation makes the range 0.5-1, just like the old strand bias
  // so we don't have to change parameter files
  float old_style_strand_bias = (full_strand_bias+1.0f)/2.0f;
  return old_style_strand_bias;
}

float MultiBook::StrandBiasPval(int i_alt, float tune_bias) const {
  if (_is_bi_dir_umt){
	  // Dummy STBP for Bi-Dir UMT
	  return 0.0f;
  }
  float strand_bias_pval = fabs(BootstrapStrandBias(_my_book[1][i_alt+1], GetDepth(0,i_alt), _my_book[2][i_alt+1], GetDepth(1,i_alt), tune_bias));
  return strand_bias_pval;
}
float MultiBook::GetXBias(int i_alt, float tune_xbias) const {
  return(ComputeTunedXBias(_my_book[1][i_alt+1], GetDepth(0,i_alt), _my_book[2][i_alt+1], GetDepth(1,i_alt), tune_xbias));
}

int MultiBook::TotalCount(int strand_key) const {
  int retval = 0;
  for (unsigned int i_hyp = 0; i_hyp < _my_book[0].size(); ++i_hyp){
    retval += GetAlleleCount(strand_key, i_hyp);
  }
  return retval;
}



// pretend beta distribution on each strand
// pretend quantity of interest is variant-frequency difference between strands
// use variance from beta to determine significance
// but we don't care about small deviations even if significant, so add a default variance to prevent going to infinity
// do the usual mild prior to avoid blowing up at zero observations
float ComputeXBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float var_zero){
  float mean_plus = (plus_var+0.5f)/(plus_depth+1.0f);
  float mean_minus = (neg_var+0.5f)/(neg_depth+1.0f);
  float var_plus = (plus_var+0.5f)*(plus_depth-plus_var+0.5f)/((plus_depth+1.0f)*(plus_depth+1.0f)*(plus_depth+2.0f));
  float var_minus = (neg_var+0.5f)*(neg_depth-neg_var+0.5f)/((neg_depth+1.0f)*(neg_depth+1.0f)*(neg_depth+2.0f));
  
  // squared difference in mean frequency, divided by variance of quantity, inflated by potential minimal variance
  return((mean_plus-mean_minus)*(mean_plus-mean_minus)/(var_plus+var_minus+var_zero));
}

float ComputeTunedXBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float proportion_zero){
  float mean_plus = (plus_var+0.5f)/(plus_depth+1.0f);
  float mean_minus = (neg_var+0.5f)/(neg_depth+1.0f);
  float var_plus = (plus_var+0.5f)*(plus_depth-plus_var+0.5f)/((plus_depth+1.0f)*(plus_depth+1.0f)*(plus_depth+2.0f));
  float var_minus = (neg_var+0.5f)*(neg_depth-neg_var+0.5f)/((neg_depth+1.0f)*(neg_depth+1.0f)*(neg_depth+2.0f));
  
  float mean_zero = (plus_var+neg_var+0.5f)/(plus_depth+neg_depth+1.0f);
  float var_zero = proportion_zero*mean_zero*proportion_zero*(1.0f-mean_zero); // variance proportional to frequency to handle near-somatic cases
  
  // squared difference in mean frequency, divided by variance of quantity, inflated by potential minimal variance
  return((mean_plus-mean_minus)*(mean_plus-mean_minus)/(var_plus+var_minus+var_zero));
}

// Compute ratio of variant counts to reads in forward strand
// Compute ratio of variant counts to reads in reverse strand
// strand_bias =  max of these divided by the sum of these
//   will be close to 1 if one of the ratios is close to 0 and the other is not
// return 0 if any strand has 0 reads, or both strands have 0 variant reads
float ComputeStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth){
  float strand_bias;
    long int num = max((long int)plus_var*neg_depth, (long int)neg_var*plus_depth);
  long int denum = (long int)plus_var*neg_depth + (long int)neg_var*plus_depth;
  if (denum == 0)
    strand_bias = 0.0f; //if there is no coverage on one of the strands then there is no bias.
  else {
    strand_bias = (float)num/denum;
  }
  return(strand_bias);
}

// Bootstrap from from both strands to see if observed variant counts on the
// neg strand are significantly different
//
// Take i=0,..N-1 bootstrap samples of size plus_depth and neg_depth
// assuming Null hypothesis is variant reads on each strand have equal
// independent binomial probability p = (plus_var+neg_var)/(plus_depth+neg_depth)
//
// return pval = proportion of times bootstrap strand bias >= observed strand bias
// rationale is that if this pval < threshold then there is real strand bias
float BootstrapStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float tune_fish){
  if ((neg_depth == 0) || (plus_depth == 0))
    return 1.0;

  assert ((neg_depth>0 ) && (plus_depth>0) );

  float observed = fabs(ComputeTransformStrandBias(plus_var, plus_depth, neg_var, neg_depth, tune_fish));
  double p = (plus_var+neg_var)/( (double)(plus_depth + neg_depth));

  RandSchrange rand_generator;
  rand_generator.SetSeed(1);

  long int N = 1000;
  double total = 0;

  for  (long int i=0; i<N; i++){    
    long int bootstrap_plus = 0;
    long int bootstrap_neg = 0;
    // bootstrap the sample
    for (long int j=0; j < plus_depth; j++){
      if ((double)(rand_generator.Rand())/rand_generator.RandMax < p)
	bootstrap_plus++;
    }
    for (long int j=0; j < neg_depth; j++){
      if ((double)(rand_generator.Rand())/rand_generator.RandMax < p)
	bootstrap_neg++;
    }
    // compute the bootstrap ratio given p
    float bootstrap = fabs(ComputeTransformStrandBias(bootstrap_plus, plus_depth, bootstrap_neg, neg_depth, tune_fish));
    // printf("obs=%f, bootstrap=%f, +=%ld, +depth=%ld, -=%ld, -depth=%ld\n", (observed+1.0f)/2.0f, (bootstrap+1.0f)/2.0f, bootstrap_plus, plus_depth, bootstrap_neg, neg_depth);
    // greater or equal to what we saw?
    if ( bootstrap >= observed)
      total++;
  }
  return (float) (total/N);
}

float ComputeTransformStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float tune_fish){
  //  Naive calcualtion with no tuning
  // x =  frequency_plus/frequency_neg  [range: 0 to infinity]
  // f(x) = (1-x)/(1+x) to bring range between -1,1
  // Multiply out and remove common factor 1/frequency_neg to reduce to:
  // f(x) = (frequency_neg - frequency_plus)/(frequency_neg - frequency_plus)
  // Multiply num & denom by 1/(plus_depth*minus_depth)
  // f(x) = (neg_var*plus_depth - plus_var*neg_depth)/(neg_var*plus_depth + plus_var*neg_depth)
  // increase denominator by .001 to avoid divide by 0

  // tuning step fudges extra alleles to try and stabilize low counts
  // the bootstrap step eems to work better instead
  // replace neg_var by neg_var + relative_tune_fish*neg_depth
  // replace pos_var by neg_var + relative_tune_fish*pos_depth
  // replace neg_depth by neg_var + expected_negative
  // replace pos_depth by pos_var + expected_positive

  // take absolute value for filtering
  // tune_fish should be something like 0.5 on standard grounds
  if (tune_fish<0.001f)
    tune_fish = 0.001f;
  
  // need to handle safety factor properly
  // "if we had tune_fish extra alleles of both reference and alternate for each strand, how would we distribute them across the strands preserving the depth ratio we see"
  // because constant values get hit by depth ratio when looking at 0/0
  // to give tune_fish expected alleles on each strand, given constant depth
  float relative_tune_fish = 1.0f- (plus_depth+neg_depth+2.0f*tune_fish)/(plus_depth+neg_depth+4.0f*tune_fish);
  
  // expected extra alleles to see on positive and negative strand
  float expected_positive = relative_tune_fish * plus_depth; 
  float expected_negative = relative_tune_fish * neg_depth;  
  
  // bias calculation based on observed counts plus expected safety level
  float pos_val = (plus_var + expected_positive) * (neg_depth + 2.0f*expected_negative);
  float neg_val = (neg_var + expected_negative) * (plus_depth + 2.0f*expected_positive);
  float strand_bias = (pos_val - neg_val)/(pos_val + neg_val+0.001f);  // what if depth on one strand is 0, then of course we can't detect any bias
  
//  cout << plus_var << "\t" << plus_depth << "\t" << neg_var << "\t" << neg_depth << "\t" << tune_fish << "\t" << strand_bias << endl;
  
  return(strand_bias);
}

// Given (x1, y1) and (x2, y2), interpolate y at x.
double LodAssist::LinearInterpolation(double x, double x1, double y1, double x2, double y2){
    assert((min(x1, x2) <= x) and (x <= max(x1, x2)));
    double alpha = (x - x2) / (x1 - x2);
    return alpha * y1 + (1.0 - alpha) * y2;
}

// calculate log(sum(exp(log_vec))) with better numerical stability
double LodAssist::LogSumFromLogIndividual(const vector<double>& log_vec){
	assert(not log_vec.empty());
	double max_value = log_vec[0];
	double log_sum = 0.0;
	for (vector<double>::const_iterator it = log_vec.begin(); it != log_vec.end(); ++it){
		max_value = max(max_value, *it);
	}
	for (vector<double>::const_iterator it = log_vec.begin(); it != log_vec.end(); ++it){
		log_sum += exp(*it - max_value);
	}
	log_sum = max_value + log(log_sum);
	return log_sum;
}

double LodAssist::LinearToPhread(double x){
	return -10.0 / log(10.0) * log(x);
}

double LodAssist::PhreadToLinear(double x){
	return exp(-0.1 * log(10.0) * x);
}

// Calculate log(1-p) from log(p)
double LodAssist::LogComplementFromLogP(double log_p){
	assert(log_p <= 0);
	if (log_p == 0.0){
		return log(0.0);
	}else if(not isfinite(log_p)){
		return 0.0;
	}
	// Now log(p), log(1-p) are not singular
	double p = exp(log_p);
	double q = 1.0 - p;
	if (q == 1.0){
		// non-zero p gives q = 1 implies floating error. Use 1st order Taylor approximation to calculate log(q) ~ -p
		return -p;
	}else if(q == 0.0){
		// p != 1 gives q = 0 implies floating error. Use 1st order Taylor approximation to calculate log(q) ~ log(-log(p))
		return log(-log_p);
	}
	return log(q);
}

LodAssist::BinomialUtils::BinomialUtils(){
	PreComputeLogFactorial(256);
}

// Calculate log(factorial(0)), ..., log(factorial(precomput_size - 1))
void LodAssist::BinomialUtils::PreComputeLogFactorial(int precompute_size){
	precompute_log_factorial_.assign(precompute_size, 0.0);
	for (int x = 2; x < precompute_size; ++x){
		precompute_log_factorial_[x] = precompute_log_factorial_[x - 1] + log((double) x);
	}
}

// Calculate the log factorial using the Stirling-Names formula, which converges to log(factorial(x)) asymptotically.
// The approximation is very good if x > 20.
double LodAssist::BinomialUtils::StirlingNamesApprox_(int x) const {
	double x_plus = (double) x + 1.0;
	// 1.8378770664093453 = log(2*pi)
    return 0.5 * (1.8378770664093453 - log(x_plus)) + (log(x_plus + x_plus / (12.0 * x_plus * x_plus - 0.1)) - 1.0) * x_plus;
}

double LodAssist::BinomialUtils::LogFactorial(int x) const{
	if (x < (int) precompute_log_factorial_.size()){
		return precompute_log_factorial_[x];
	}
	return StirlingNamesApprox_(x);
}

double LodAssist::BinomialUtils::LogNChooseK(int n, int k) const{
	if (k == 0 or k == n){
		return 0.0;
	}
	assert(k > 0 and k < n);
    return max(LogFactorial(n) - (LogFactorial(k) + LogFactorial(n - k)), 1.0);

}
double LodAssist::BinomialUtils::LogBinomialPmf(int n, int k, double log_p, double log_q) const{
	if (log_q >= 1.0){
		log_q = LogComplementFromLogP(log_p);
	}
	return min(LogNChooseK(n, k) + (double(k) * log_p + (double) (n - k) * log_q), 0.0);
}

double LodAssist::BinomialUtils::LogBinomialCdf(int x, int n, double log_p, double log_q) const{
	if (log_q >= 1.0){
		log_q = LogComplementFromLogP(log_p);
	}
	vector<double> log_pmf_vec(x + 1, 0.0);
	int k = 0;
	for (vector<double>::iterator it = log_pmf_vec.begin(); it != log_pmf_vec.end(); ++it, ++k){
		*it = LogBinomialPmf(n, k, log_p, log_q);
	}
    return min(LogSumFromLogIndividual(log_pmf_vec), 0.0);
}

LodManager::LodManager(){
	min_var_coverage_ = 2;
	min_allele_freq_ = 0.0005;
	min_variant_score_ = LodAssist::LinearToPhread(0.5);
	min_callable_prob_ = 0.98;
	do_smoothing_ = true;
}

void LodManager::SetParameters(int min_var_coverage, double min_allele_freq, double min_variant_score, double min_callable_prob){
	min_var_coverage_ = min_var_coverage;
	min_allele_freq_ = min_allele_freq;
	// min_variant_score < 3.010 is equivalent to min_variant_score = 3.010
	min_variant_score_ = max(min_variant_score, LodAssist::LinearToPhread(0.5));
	min_callable_prob_ = min_callable_prob;
	assert(min_allele_freq_ > 0.0 and min_allele_freq_ < 1.0);
	assert(min_var_coverage_ >= 0);
	assert(min_callable_prob_ > 0.0 and min_callable_prob_ < 1.0);
}

void LodManager::CalculateMinCallableAo_(int dp, int& min_callable_ao, double& qual_plus, double& qual_minus) const{
	// Initialize qual, qual_minus, min_callable_ao to be invalid values.
	qual_plus = -1.0;
	qual_minus = -1.0;
	min_callable_ao = -1;

	if (dp == 0 or dp < min_var_coverage_){
		// The variant can not possibly be called.
		return;
	}
	// Initialization for computing incomplete beta function.
	double log_f_c = log(min_allele_freq_);
	double log_1_minus_f_c = LodAssist::LogComplementFromLogP(log_f_c);
	double log_beta = -log((double) dp + 1.0);
	double beta_inc = max(1.0 - exp(((double) dp + 1.0) * log_1_minus_f_c), 0.0);
	int ro = dp; // ro = dp - ao
	// ao_plus = ao + 1
	for (int ao_plus = 1; ao_plus <= dp + 1; ++ao_plus, --ro){
		// beta_inc = betainc(ao + 1, dp - ao + 1, min_allele_freq_) where betainc is the incomplete beta function.
		qual_plus = LodAssist::LinearToPhread(beta_inc);
		// The event that the variant is callable.
	    if (qual_plus >= min_variant_score_ and ao_plus > min_var_coverage_){
	    	min_callable_ao = ao_plus - 1;
			return;
	    }
	    qual_minus = qual_plus;
		// Update beta_inc and log_beta
        beta_inc -= exp(log_f_c * (double) ao_plus + log_1_minus_f_c * (double) ro - log_beta - log((double) ao_plus));
        log_beta += (log((double) ao_plus) - log((double) ro));
	}
	// There exists no callable ao. Set to be invalid values
	qual_plus = -1.0;
	qual_minus = -1.0;
	min_callable_ao = -1;
}

double LodManager::CalculateCallableProb_(int dp, double af, int min_callable_ao, double qual_plus, double qual_minus) const {
	assert(0.0 <= af and af <= 1.0);
	if (min_callable_ao == 0){
		return 1.0;
	}
	double log_p = log(af);
	double log_q = LodAssist::LogComplementFromLogP(log_p);
	vector<double> log_pmf_vec(min_callable_ao, 0.0);
	int k = 0;
	for (vector<double>::iterator it = log_pmf_vec.begin(); it != log_pmf_vec.end(); ++it, ++k){
		*it = binomial_utils_.LogBinomialPmf(dp, k, log_p, log_q);
	}
	// The reason I don't call binomial_utils.LogBinomialCdf is because I need to do smoothing later.
	double p_callable = 1.0 - exp(min(LodAssist::LogSumFromLogIndividual(log_pmf_vec), 0.0));

	// Condition of not doing smoothing.
	if (qual_plus < 0.0 or qual_minus < 0.0 or qual_minus >= qual_plus or min_callable_ao == min_var_coverage_ or (not do_smoothing_)){
		return p_callable;
	}
	double p_callable_minus = p_callable + exp(log_pmf_vec.back());
	return LodAssist::LinearInterpolation(min_variant_score_, qual_minus, p_callable_minus, qual_plus, p_callable);
}

double LodManager::CalculateLod(int dp) const{
	assert(dp >= 0);
	// No variant can be called anyway given the dp.
	if (dp == 0 or dp < min_var_coverage_){
		return -1.0;
	}
	// Calculate minimum callable ao.
	int min_callable_ao = -1;
	double qual_plus = -1.0;
	double qual_minus = -1.0;
	CalculateMinCallableAo_(dp, min_callable_ao, qual_plus, qual_minus);
	if (min_callable_ao == 0){
		// Shouldn't happen, just in case.
		return 0.0;
	}else if (min_callable_ao < 0){
		return -1.0;
	}
	// Calculate the AF that gives
	const int max_rounds = 10;
	const int num_div = 10;
	double start_af = 0.1 * min(1 / (double) dp, min_allele_freq_);
	double new_start_af = -1.0;
	double end_af = 1.0 - start_af;
	double new_end_af = -1.0;
    double p_callable_new_start = -1.0;
    double p_callable_new_end = -1.0;
	int iter_num = 0;

	for (int round_idx = 0; round_idx < max_rounds; ++round_idx){
	    new_start_af = -1.0;
	    new_end_af = -1.0;
	    p_callable_new_start = -1.0;
	    p_callable_new_end = -1.0;
	    double div_size = (end_af - start_af) / (double) num_div;
	    for (int div_idx = 0; div_idx <= num_div; ++div_idx){
	    	++iter_num;
	    	double af = (div_idx == num_div)? end_af : (start_af + div_idx * div_size); // In case start_af + num_div * div_size != end_af
	    	double p_callable = CalculateCallableProb_(dp, af, min_callable_ao, qual_plus, qual_minus);
	        if (p_callable < min_callable_prob_){
	            new_start_af = af;
	            p_callable_new_start = p_callable;
	        }else if (p_callable > min_callable_prob_){
	            new_end_af = af;
	            p_callable_new_end = p_callable;
	        }else{
	            // Lucky me! Exactly hit min_callable_prob_
	            return af;
	        }
			// Stoping rule in a round
			if (new_start_af >= 0.0 and new_end_af >= 0.0){
				start_af = new_start_af;
				end_af = new_end_af;
				break;
			}
	    }

	    // Stop if we are very close
	    if (p_callable_new_start >= 0 and p_callable_new_end >= 0){
			if (fabs(p_callable_new_start - p_callable_new_end) < 0.001 * min_callable_prob_){
				return LodAssist::LinearInterpolation(min_callable_prob_, p_callable_new_start, new_start_af, p_callable_new_end, new_end_af);
			}
	    }

		// Handle the case of bad initial condition:
		if (new_end_af < 0.0){
			end_af = 0.5 * (end_af + 1.0);
		}
		if (new_start_af < 0.0){
			start_af *= 0.5;
		}
	}
	// Reach the max round, use interpolation.
	if (p_callable_new_start >= 0 and p_callable_new_end >= 0){
		return LodAssist::LinearInterpolation(min_callable_prob_, p_callable_new_start, new_start_af, p_callable_new_end, new_end_af);
	}
	return 1.0;
}
