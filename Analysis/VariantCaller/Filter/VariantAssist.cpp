/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantAssist.cpp
//! @ingroup  VariantCaller
//! @brief    Helpful routines for output of variants

#include "VariantAssist.h"

MultiBook::MultiBook(){
  invalid_reads = 0;
}

void MultiBook::Allocate(int num_hyp){
  my_book.resize(2);
  for (unsigned int i_strand=0; i_strand <my_book.size(); i_strand++){
    my_book.at(i_strand).resize(num_hyp);
  }
}

int MultiBook::NumAltAlleles(){
  return(my_book.at(0).size()-1); // not counting Ref =0
}

void MultiBook::SetCount(int i_strand, int i_hyp, int count){
  my_book.at(i_strand).at(i_hyp) = count;
}

void MultiBook::ResetCounter(){
  invalid_reads = 0;
  for (int i_strand=0; i_strand<2; i_strand++){
    for (unsigned int i_hyp=0; i_hyp<my_book[i_strand].size(); i_hyp++){
      my_book[i_strand][i_hyp]=0;
    }
  }
}

void MultiBook::DigestHardClassifiedReads(vector<bool> &strand_id, vector<int> &read_id){
  // reset counter
  ResetCounter();
  for (unsigned int i_read = 0; i_read < read_id.size(); i_read++) {
     int _allele_index = read_id[i_read];
     int i_strand = 1;
     if (strand_id.at(i_read))
       i_strand = 0;
     if (_allele_index>-1){
       my_book.at(i_strand).at(_allele_index)+=1;
     }
     if (_allele_index==-1){
       invalid_reads++;
     }
       // otherwise no deal - doesn't count for this alternate at all
   }
}


int MultiBook::GetDepth(int i_strand, int i_alt){
  int retval=0;
  if (i_strand<0){
    retval += my_book.at(0).at(i_alt+1)+my_book.at(0).at(0);
    retval += my_book.at(1).at(i_alt+1)+my_book.at(1).at(0);
  }else
    retval = my_book.at(i_strand).at(i_alt+1)+my_book.at(i_strand).at(0);
  return(retval);
}

float MultiBook::GetFailedReadRatio(){
  int tdepth=0;
  for (unsigned int i_strand =0; i_strand<2; i_strand++){
    for (unsigned int i_hyp=0; i_hyp<my_book.at(i_strand).size(); i_hyp++){
      tdepth+= my_book.at(i_strand).at(i_hyp);
    }
  }
  // all divisions need a safety factor
  float retval = 1.0f*invalid_reads/(1.0f*tdepth+1.0f*invalid_reads+0.01f);
  return(retval);
}

float MultiBook::OldStrandBias(int i_alt, float tune_bias){
  float full_strand_bias = fabs(ComputeTransformStrandBias(my_book.at(0).at(i_alt+1), GetDepth(0,i_alt), my_book.at(1).at(i_alt+1), GetDepth(1,i_alt), tune_bias));
//@TODO: revert to full range parameters
// this ugly transformation makes the range 0.5-1, just like the old strand bias
// so we don't have to change parameter files
float old_style_strand_bias = (full_strand_bias+1.0f)/2.0f;
return(old_style_strand_bias);
}

float MultiBook::GetXBias(int i_alt, float tune_xbias){
  return(ComputeTunedXBias(my_book.at(0).at(i_alt+1), GetDepth(0,i_alt), my_book.at(1).at(i_alt+1), GetDepth(1,i_alt), tune_xbias));
}

int MultiBook::GetAlleleCount(int strand_key, int i_hyp){
  int retval;
  if (strand_key<0)
    retval = my_book.at(0).at(i_hyp)+my_book.at(1).at(i_hyp);
  else
    retval= my_book.at(strand_key).at(i_hyp);
  return(retval);
}

int MultiBook::TotalCount(int strand_key){
  int retval=0;
  if (strand_key<0){
    for (unsigned int i_hyp=0; i_hyp<my_book.at(0).size(); i_hyp++)
      retval += my_book.at(0).at(i_hyp)+my_book.at(1).at(i_hyp);
  } else {
    for (unsigned int i_hyp=0; i_hyp<my_book.at(0).size(); i_hyp++)
      retval += my_book.at(strand_key).at(i_hyp);
  }
  return(retval);
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

float ComputeTransformStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float tune_fish){
  // trying to compute f_plus/f_neg
  //  transform to (1-x)/(1+x) to bring range between -1,1
  // take absolute value for filtering
  // tune_fish shoudl be something like 0.5 on standard grounds
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

