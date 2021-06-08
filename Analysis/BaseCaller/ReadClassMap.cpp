/* Copyright (C) 2019 Thermo Fisher Scientific. All Rights Reserved */


#include <iostream>
#include <sstream>
#include <locale>
#include "ReadClassMap.h"


class ThousandsSeparator : public numpunct<char> {
protected:
    string do_grouping() const { return "\03"; }
};

ReadClassMap::ReadClassMap()
    : W(1), H(1), ignore_washouts_(false),
      num_library_(0), num_tf_(0), num_valid_lib_(0),
      num_valid_tf_(0), filter_mask(1,1)
{
};

void ReadClassMap::LoadMaskFiles(const std::vector<std::string> & mask_file_names, bool ignore_washouts)
{
  assert(mask_file_names.size() < 9);
  if (filter_mask.SetMask (mask_file_names.at(0).c_str()))
      exit (EXIT_FAILURE);

  W = filter_mask.W();
  H = filter_mask.H();
  class_map.assign((W * H), 0);
  ignore_washouts_ = ignore_washouts;

  // XXX Verbose output:
  cout << "ReadClassMap:" << endl;
  cout << " - Loaded mask of size W: " <<  W << " , H: " << H
       << " from file " << mask_file_names.at(0) <<endl;
  cout << " - size of class map: " << class_map.size() << endl;


  // Transfer mask information to class map
  //unsigned int num_library    = 0;
  //unsigned int num_tf         = 0;
  unsigned int num_BadKey     = 0;
  unsigned int num_HighPPF    = 0;
  unsigned int num_polyclonal = 0;
  unsigned int num_washout    = 0;

  // Add code to merge the other masks

  for (unsigned int fi=0; fi<mask_file_names.size(); ++fi){

    // reset counters
    num_library_ = num_tf_ = num_valid_lib_ = num_valid_tf_ = 0;
    num_BadKey = num_HighPPF = num_polyclonal = num_washout = 0;

    Mask temp_mask(1,1);
    if (temp_mask.SetMask (mask_file_names.at(fi).c_str()))
      exit (EXIT_FAILURE);

    if (temp_mask.W() != W or temp_mask.H() != H){
      cerr << "ReadClassMap ERROR: Dimensions of mask files do not agree" << endl;
      cerr << "Dim " << mask_file_names.at(0)  << " : " << W << "x" << H << endl;
      cerr << "Dim " << mask_file_names.at(fi) << " : " << temp_mask.W() << "x" << temp_mask.H() << endl;
    }

    // Create Consensus Classification
    for (unsigned int index=0; index<class_map.size(); ++index)
    {
      // Non-live beads keep their initial classification
      if (not temp_mask.Match(index, MaskLive))
        continue;

      bool first_live = (fi==0);
      // Library classification in any run dominates.
      if (temp_mask.Match(index, MaskLib)){
        class_map[index] |= MapLibrary;
        ++num_library_;
        if (not filter_mask.Match(index, MaskLib)){
          first_live = true;
          filter_mask[index] = temp_mask[index];
        }
      }
      if (temp_mask.Match(index, MaskTF)){
        class_map[index] |= MapTF;
        ++num_tf_;
        if (not filter_mask.Match(index, MaskTF)){
          first_live = true;
          filter_mask[index] = temp_mask[index];
        }
      }


      // Washouts only undergo partial signal processing - Ignore
      if (temp_mask.Match(index, MaskWashout)){
        ++num_washout;
        if(ignore_washouts_)
          continue;
      }

      // Propagate unfiltered status
      if (not temp_mask.Match(index, MaskFiltered)){
        SetClassUnfiltered(index);
        temp_mask.Match(index, MaskTF) ? ++num_valid_tf_ : ++num_valid_lib_;
        if (filter_mask.Match(index, MaskWashout)){
          temp_mask[index] |= MaskWashout;
        }
        filter_mask[index] = temp_mask[index];
        // Use signal from this run
        class_map[index] |= (1<<(8+fi));
        continue;
      }

      // In BkgModel you only have one filtering reason set
      // Here we store multiple filter reasons, if they differ between the runs
      bool transfer_filters = first_live or ClassMatch(index, MapFiltered);


      if (temp_mask.Match(index, MaskFilteredBadKey)){
        ++num_BadKey;
        if (transfer_filters){
          class_map[index]   |= MapFilteredBadKey;
          filter_mask[index] |= MaskFilteredBadKey;
        }
      }
      if (temp_mask.Match(index, MaskFilteredBadResidual)){
        ++num_HighPPF;
        if (transfer_filters){
          class_map[index]   |= MapFilteredHighPPF;
          filter_mask[index] |= MapFilteredHighPPF;
        }
      }
      if (temp_mask.Match(index, MaskFilteredBadPPF)){
        ++num_polyclonal;
        if (transfer_filters){
          class_map[index]   |= MapFilteredPolyclonal;
          filter_mask[index] |= MaskFilteredBadPPF;
        }
      }

    } // -- End loop over class map

    // Summary statistics for this mask file
    cout <<  "Summary info for mask " << mask_file_names.at(fi) << endl;
    cout << " - Num Library Wells : " << num_library_   << endl;
    cout << " - Num Test Fragments: " << num_tf_        << endl;
    cout << " - Num Bad Key       : " << num_BadKey     << endl;
    cout << " - Num HighPPF       : " << num_HighPPF    << endl;
    cout << " - Num Polyclonal    : " << num_polyclonal << endl;
    cout << " - Num Washout       : " << num_washout    << endl;
    cout << " - Num Valid Wells   : " << num_valid_lib_+num_valid_tf_ << endl;
    cout << endl;

  }; // -- End Loop over mask files

  cout << endl;
  if (mask_file_names.size() < 2)
    return;

  // Another pass over class map to do accounting & exclude wells that were
  // identified as Library ad TF in different runs  --- Yes, those exist!

  unsigned int num_lib_tf = 0;
  vector<unsigned int> signal_diversity_lib(mask_file_names.size()+1, 0);
  vector<unsigned int> signal_diversity_tf (mask_file_names.size()+1, 0);
  num_library_ = num_tf_ = num_valid_lib_ = num_valid_tf_ = 0;

  for (unsigned int index=0; index<class_map.size(); ++index)
  {
    if (ClassMatch(index, MapLibrary) and ClassMatch(index, MapTF)){
      ++num_lib_tf;
      filter_mask[index] = MaskIgnore;
      class_map[index]   = 0;
    }

    if (ClassMatch(index, MapLibrary)){
      ++num_library_;
      ++signal_diversity_lib.at(getSignalDiversity(index));
      if (IsValidRead(index))
        ++num_valid_lib_;
    }
    else if (ClassMatch(index, MapTF)){
      ++num_tf_;
      ++signal_diversity_tf.at(getSignalDiversity(index));
      if (IsValidRead(index))
        ++num_valid_tf_;
    }
  };

  ostringstream table;
  table.imbue(locale(table.getloc(), new ThousandsSeparator));

  table << endl;
  table << setw(20) << "Read Class Map" << setw(20) << "Library Wells" << setw(20) << "Test Fragments" << endl;

  table << setw(20) << "Total Wells";
  table << setw(20) << num_library_ << setw(20) << num_tf_ << endl;

  table << setw(20) << "Valid Wells";
  table << setw(20) << num_valid_lib_ << setw(20) << num_valid_tf_ << endl;

  table << setw(20) << "Signal Diversity";
  table << setw(20) << "-----------------" << setw(20) << "-----------------" << endl;

  for (unsigned int m=0; m<mask_file_names.size()+1; ++m){
    table << setw(20) << m;
    table << setw(20) << signal_diversity_lib.at(m) << setw(20) << signal_diversity_tf.at(m) << endl;
  }

  table << setw(20) << " ";
  table << setw(20) << "-----------------" << setw(20) << "-----------------" << endl;

  table << "Ignoring " << num_lib_tf << " wells that were classified both as Lib & TF." << endl;
  table << endl;
  cout << table.str();

};

// --------------------------------------------------------

bool ReadClassMap::WriteFilterMask(const std::string mask_filename)
{
  filter_mask.WriteRaw (mask_filename.c_str());
  filter_mask.validateMask();
  return (true);
};

void ReadClassMap::Close()
{

};

