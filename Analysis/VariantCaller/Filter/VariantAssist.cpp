/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantAssist.cpp
//! @ingroup  VariantCaller
//! @brief    Helpful routines for output of variants

#include "VariantAssist.h"
#include <limits>

MultiBook::MultiBook(){
  invalid_reads = 0;
}

void MultiBook::Allocate(int num_hyp){
  my_book.resize(2);
  for (unsigned int i_strand=0; i_strand <my_book.size(); i_strand++){
    my_book[i_strand].resize(num_hyp);
  }

  my_position_bias.resize(num_hyp);
  my_position_bias.resize(num_hyp);
  for (unsigned int i = 0; i < (unsigned int)num_hyp; i++) {
    my_position_bias[i].rho = -1;
    my_position_bias[i].pval = -1;
  }
}

int MultiBook::NumAltAlleles(){
  return(my_book[0].size()-1); // not counting Ref =0
}

void MultiBook::SetCount(int i_strand, int i_hyp, int count){
  my_book[i_strand][i_hyp] = count;
}

void MultiBook::ResetCounter(){
  invalid_reads = 0;
  for (int i_strand=0; i_strand<2; i_strand++){
    for (unsigned int i_hyp=0; i_hyp<my_book[i_strand].size(); i_hyp++){
      my_book[i_strand][i_hyp]=0;
    }
  }
}

void MultiBook::AssignPositionFromEndToHardClassifiedReads(vector<int> &read_id, vector<int> &left, vector<int> &right)
{
  // record position in read for each read with a valid allele
  to_left.resize(read_id.size());
  to_right.resize(read_id.size());
  allele_index.resize(read_id.size());

  unsigned int count = 0;
  for (unsigned int i_read = 0; i_read < read_id.size(); i_read++) {
     int _allele_index = read_id[i_read];
     if (_allele_index > -1){
       to_right[count] = right[i_read];
       to_left[count] = left[i_read];
       allele_index[count] = _allele_index;
       count++;
     }
  }
  to_left.resize(count);
  to_right.resize(count);
  allele_index.resize(count);
}

void MultiBook::AssignStrandToHardClassifiedReads(vector<bool> &strand_id, vector<int> &read_id)
{
  // reset counter
  ResetCounter();
  // record strand direction for each read with a valid allele
  for (unsigned int i_read = 0; i_read < read_id.size(); i_read++) {
     int _allele_index = read_id[i_read];
     int i_strand = 1;
     if (strand_id[i_read])
       i_strand = 0;
     if (_allele_index>-1){
       my_book[i_strand][_allele_index]+=1;
     }
     if (_allele_index==-1){
       invalid_reads++;
     }
     // otherwise no deal - doesn't count for this alternate at all
   }
}

float VariantAssist::median(std::vector<float>& values)
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
    unsigned int t = v[i]; v[i] = v[i+c]; v[i+c] = t;	/* swap */
  }
}

double VariantAssist::partial_sum(vector<double> &v, size_t n) {
  assert (n <= v.size());
  if(n==0)
    return std::numeric_limits<double>::quiet_NaN();

  double total = 0;
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
     fprintf(stdout, "ref.size()=%lu; ", ref.size());
     fprintf(stdout, "var.size()=%lu; ", var.size());
     fprintf(stdout, "both.size()=%lu; ", both.size());
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

int MultiBook::GetDepth(int i_strand, int i_alt){
  int retval=0;
  if (i_strand<0){
    retval += my_book[0][i_alt+1]+my_book[0][0];
    retval += my_book[1][i_alt+1]+my_book[1][0];
  }else
    retval = my_book[i_strand][i_alt+1]+my_book[i_strand][0];
  return(retval);
}

float MultiBook::GetFailedReadRatio(){
  int tdepth=0;
  for (unsigned int i_strand =0; i_strand<2; i_strand++){
    for (unsigned int i_hyp=0; i_hyp<my_book[i_strand].size(); i_hyp++){
      tdepth+= my_book[i_strand][i_hyp];
    }
  }
  // all divisions need a safety factor
  float retval = 1.0f*invalid_reads/(1.0f*tdepth+1.0f*invalid_reads+0.01f);
  return(retval);
}

float MultiBook::OldStrandBias(int i_alt, float tune_bias){
  float full_strand_bias = fabs(ComputeTransformStrandBias(my_book[0][i_alt+1], GetDepth(0,i_alt), my_book[1][i_alt+1], GetDepth(1,i_alt), tune_bias));
//@TODO: revert to full range parameters
// this ugly transformation makes the range 0.5-1, just like the old strand bias
// so we don't have to change parameter files
float old_style_strand_bias = (full_strand_bias+1.0f)/2.0f;
return(old_style_strand_bias);
}

float MultiBook::StrandBiasPval(int i_alt, float tune_bias){
  float strand_bias_pval = fabs(BootstrapStrandBias(my_book[0][i_alt+1], GetDepth(0,i_alt), my_book[1][i_alt+1], GetDepth(1,i_alt), tune_bias));
  return strand_bias_pval;
}
float MultiBook::GetXBias(int i_alt, float tune_xbias){
  return(ComputeTunedXBias(my_book[0][i_alt+1], GetDepth(0,i_alt), my_book[1][i_alt+1], GetDepth(1,i_alt), tune_xbias));
}

int MultiBook::GetAlleleCount(int strand_key, int i_hyp){
  int retval;
  if (strand_key<0)
    retval = my_book[0][i_hyp]+my_book[1][i_hyp];
  else
    retval= my_book[strand_key][i_hyp];
  return retval;
}

int MultiBook::TotalCount(int strand_key){
  int retval=0;
  if (strand_key<0){
    for (unsigned int i_hyp=0; i_hyp<my_book[0].size(); i_hyp++)
      retval += my_book[0][i_hyp]+my_book[1][i_hyp];
  } else {
    for (unsigned int i_hyp=0; i_hyp<my_book[0].size(); i_hyp++)
      retval += my_book[strand_key][i_hyp];
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

