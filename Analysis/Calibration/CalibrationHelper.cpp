/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     CalibrationHelper.cpp
//! @ingroup  Calibration
//! @brief    CalibrationHelper. Classes and methods for the Calibration executable

#include <iomanip>
#include "Utils.h"
#include "MiscUtil.h"
#include "ion_util.h"

#include "CalibrationHelper.h"
#include "FlowAlignment.h"

// -----------------------------------------------------------------------

void PrintHelp_Calibration()
{
  cout << endl;
  cout << "Calibrate homopolymers for a run using training data." << endl;
  cout << "Usage: Calibration [options]"                          << endl << endl;

  cout << "Required Arguments:"                                                   << endl;
  cout << "  -i,--input                     FILE      mapped input BAM file(s)"     << endl;
  cout << "     --block-size                INT,INT   block/chip x,y dimensions"    << endl;
  cout << endl;
  cout << "Optional Arguments:"                                                   << endl;
  cout << "  -o,--output-dir                DIR       output directory                           [.]"        << endl;
  cout << "     --block-offset              INT,INT   block offset coordinates x,y               [0,0]"      << endl;
  cout << "     --num-calibration-regions   INT,INT   number of regions in x,y direction         [2,2]"      << endl;
  cout << "     --num-threads               INT       number of worker threads                   [nCores]"   << endl;
  cout << "     --num-reads-per-thread      INT       reads to load per thread                   [1000]"     << endl;
  cout << "     --flow-window-size          INT       size of a flow window                      [nflows/2]" << endl;
  cout << "     --skip-droop                INT       Disregard droop in prediction generation   [true]"     << endl;
  cout << "     --min-mapping-qv            INT       Minimum mapping quality for reads          [8]"        << endl;
  cout << "     --min-align-length          INT       Minimum target alignment length            [30]"       << endl;
  cout << "     --verbose                   INT       Set verbose level                          [1]"        << endl;
  cout << endl;
};

// =======================================================================
// Functions of CalibrationContext

bool CalibrationContext::InitializeFromOpts(OptArgs &opts)
{

  vector<string> input_bams  = opts.GetFirstStringVector('i', "input", "");
  if (input_bams.empty()) {
    PrintHelp_Calibration();
    return false;
  }

  if (!bam_reader.Open(input_bams)) {
    cerr << "Calibration: Failed to open input bam file(s):" << endl;
    for (unsigned int iBam=0; iBam<input_bams.size(); ++iBam)
      cerr << "    " << input_bams.at(iBam) << endl;
    exit(EXIT_FAILURE);
  }

  max_num_flows = 0;
  DetectFlowOrderzAndKeyFromBam(bam_reader.GetReadGroups());

  chip_subset.InitializeCalibrationRegionsFromOpts(opts);

  string  output_dir   = opts.GetFirstString ('o', "output-dir", ".");
  filename_json        = output_dir + "/Calibration.json";

  num_threads          = opts.GetFirstInt    ('-', "num-threads", max(numCores(), 4));
  num_reads_per_thread = opts.GetFirstInt     ('-', "num-reads-per-thread", 1000);
  flow_window_size     = opts.GetFirstInt     ('-', "flow-window-size", (max_num_flows+1)/2);

  // Some options are not yest included in help since at this point they would not be helpful for users
  load_unmapped        = opts.GetFirstBoolean ('-', "load-unmapped", false);
  skip_droop           = opts.GetFirstBoolean ('-', "skip-droop", true);
  do_flow_alignment    = opts.GetFirstBoolean ('-', "do-flow-alignment", true);
  verbose_level        = opts.GetFirstInt     ('-', "verbose-level", 1);
  debug                = opts.GetFirstBoolean ('-', "debug", false);

  // General filters
  min_mapping_qv       = opts.GetFirstInt     ('-', "min-mapping-qv", 8);
  min_align_length     = opts.GetFirstInt     ('-', "min-align-length", 30);

  // Variables to log some information
  num_reads_in_bam = 0;
  num_mapped_reads = 0;
  num_loaded_reads = 0;
  num_useful_reads = 0;

  Verbose();
  return true;
};


// -----------------------------------------------------------------------
// Mostly borrowed from variant caller

void CalibrationContext::DetectFlowOrderzAndKeyFromBam(const BamTools::SamReadGroupDictionary & read_groups){

    // We only store flow orders that are different from each other in the flow order vector.
    // The same flow order but different total numbers of flow map to the  same flow order object
    // So multiple runs, even with different max flows, point to the same flow order object
    // We assume that the read group name is written in the form <run_id>.<Barcode Name>

    if (read_groups.Size() == 0) {
      cerr << "Calibration ERROR: There are no read groups in the headers of the specified BAM files." << endl;
      exit(EXIT_FAILURE);
    }

    flow_order_vector.clear();
    vector<string> temp_flow_order_vector;
    int num_read_groups = 0;

    for (BamTools::SamReadGroupConstIterator itr = read_groups.Begin(); itr != read_groups.End(); ++itr) {

      num_read_groups++;
      if (itr->ID.empty()){
        cerr << "Calibration ERROR: BAM file has a read group without ID." << endl;
        exit(EXIT_FAILURE);
      }
      // We need a flow order to do flow alignment so throw an error if there is none.
      if (not itr->HasFlowOrder()) {
        cerr << "Calibration ERROR: read group " << itr->ID << " does not have a flow order." << endl;
        exit(EXIT_FAILURE);
      }

      // Check for duplicate read group ID entries and throw an error if one is found
      std::map<string,string>::const_iterator key_it = key_by_read_group.find(itr->ID);
      if (key_it != key_by_read_group.end()) {
        cerr << "Calibration ERROR: Multiple read group entries with ID " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }

      // Store Key Sequence for each read group
      // The read group key in the BAM file contains the full prefix: key sequence + barcode + barcode adapter
      key_by_read_group[itr->ID] = itr->KeySequence;

      // Get run id from read group name: convention <read group name> = <run_id>.<Barcode Name>
      string run_id = itr->ID.substr(0,itr->ID.find('.'));
      if (run_id.empty()) {
        cerr << "Calibration ERROR: Unable to extract run id from read group name " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }

      // Check whether an entry already exists for this run id and whether it is compatible
      // (only one flow order per run is possible)
      std::map<string,int>::const_iterator fo_it = flow_order_index_by_run_id.find(run_id);
      if (fo_it != flow_order_index_by_run_id.end()) {
    	// Flow order for this run id may be equal or a subset of the stored one
        if ((temp_flow_order_vector.at(fo_it->second).length() < itr->FlowOrder.length())
            or (temp_flow_order_vector.at(fo_it->second).substr(0, itr->FlowOrder.length()) != itr->FlowOrder)
            or (num_flows_by_run_id.at(run_id) != (int)(itr->FlowOrder).length()))
        {
          cerr << "TVC ERROR: Flow order information extracted from read group name " << itr->ID
               << " does not match previous entry for run id " << run_id << ": " << endl;
          cerr << "Existing entry : " << temp_flow_order_vector.at(fo_it->second) << endl;
          cerr << "Newly extracted: " << itr->FlowOrder << endl;
          exit(EXIT_FAILURE);
        }
        // Found matching entry and everything is OK.
        continue;
      }

      // New run id: Check whether this flow order is the same or a sub/ superset of an existing flow order
      unsigned int iFO = 0;
      for (; iFO< temp_flow_order_vector.size(); iFO++) {

        // Is the new flow order a subset of an existing flow order?
        if ( temp_flow_order_vector.at(iFO).length() >= itr->FlowOrder.length() ) {
          if (temp_flow_order_vector.at(iFO).substr(0, itr->FlowOrder.length()) == itr->FlowOrder ) {
            flow_order_index_by_run_id[run_id] = iFO;
            num_flows_by_run_id[run_id] = itr->FlowOrder.length();
            break;
          }
          else
            continue;
        }

        // Is the new flow order a superset of an existing flow order?
        if ( temp_flow_order_vector.at(iFO).length() < itr->FlowOrder.length() ) {
          if ( itr->FlowOrder.substr(0, temp_flow_order_vector.at(iFO).length()) == temp_flow_order_vector.at(iFO) ) {
            temp_flow_order_vector.at(iFO) = itr->FlowOrder;
            flow_order_index_by_run_id[run_id] = iFO;
            num_flows_by_run_id[run_id] = itr->FlowOrder.length();
            break;
          }
        }
      }

      // Do we have a new flow order?
      if (iFO == temp_flow_order_vector.size()) {
    	temp_flow_order_vector.push_back(itr->FlowOrder);
    	flow_order_index_by_run_id[run_id] = iFO;
    	num_flows_by_run_id[run_id] = itr->FlowOrder.length();
      }

    } // --- end loop over read groups

    // Now we have amassed all the unique flow orders and can construct the FlowOrder objects
    for (unsigned int iFO=0; iFO < temp_flow_order_vector.size(); iFO++){
      ion::FlowOrder tempIonFlowOrder(temp_flow_order_vector.at(iFO), temp_flow_order_vector.at(iFO).length());
      flow_order_vector.push_back(tempIonFlowOrder);
      max_num_flows = max(max_num_flows, tempIonFlowOrder.num_flows());
    }

}

// -----------------------------------------------------------------------

void CalibrationContext::Verbose()
{
  if (verbose_level > 0) {
    cout << "Calibration Options:" << endl;
    cout << "   Output file            : " << filename_json << endl;
    cout << "   num-treads             : " << num_threads   << endl;
    cout << "   num-reads-per-thread   : " << num_reads_per_thread << endl;
    cout << "   flow-window-size       : " << flow_window_size << endl;
    cout << "   min-mapping-qv         : " << min_mapping_qv   << endl;
    cout << "   min-align-length       : " << min_align_length << endl;
    // Flow orders and keys information
    cout << "   found a total of " << key_by_read_group.size() << " read groups in BAM(s)." << endl;
    cout << "   found a total of " << flow_order_vector.size() << " unique flow orders of max flow lengths: ";
    int iFO=0;
    for (; iFO<(int)flow_order_vector.size()-1; iFO++)
      cout << flow_order_vector.at(iFO).num_flows() << ',';
    cout << flow_order_vector.at(iFO).num_flows() << endl;
  }
}

// -----------------------------------------------------------------------

void CalibrationContext::Close(Json::Value &json)
{
	bam_reader.Close();
	json["num_reads_in_bam"] = (Json::UInt64)num_reads_in_bam;
	json["num_mapped_reads"] = (Json::UInt64)num_mapped_reads;
	json["num_loaded_reads"] = (Json::UInt64)num_loaded_reads;
	json["num_useful_reads"] = (Json::UInt64)num_useful_reads;

	// Print a summary of loaded beads
	if (verbose_level>0) {
	  cout << endl;
	  cout << "Calibration read summary: " << endl;
	  cout << setw(28) << "Number of reads in BAM : " << num_reads_in_bam << endl;
      cout << setw(28) << "Number of mapped reads : " << num_mapped_reads << endl;
      cout << setw(28) << "Number of loaded reads : " << num_loaded_reads << endl;
      cout << setw(28) << "Number of useful reads : " << num_useful_reads << endl;
      cout << endl;
	}
}




// =======================================================================

ReadAlignmentInfo::ReadAlignmentInfo()
{
  Reset();
}

// -----------------------------------------------------------------------

void ReadAlignmentInfo::Reset()
{
  alignment = NULL;
  measurements.clear();
  measurements_length = -1;
  predictions_as_called.clear();
  predictions_ref.clear();
  state_inphase.clear();
  phase_params.clear();

  // Non-Alignment read info
  run_id.clear();
  read_group.clear();
  read_bases.clear();
  well_xy.clear();
  start_flow = -1;
  prefix_flow = -1;
  flow_order_index = -1;

  // Base alignment information
  target_bases.clear();
  query_bases.clear();
  pretty_align.clear();

  // Flow alignment info
  aln_flow_order.clear();
  aligned_qHPs.clear();
  aligned_tHPs.clear();
  align_flow_index.clear();
  pretty_flow_align.clear();

  is_filtered = true;
}

// -----------------------------------------------------------------------

void ReadAlignmentInfo::SetSize(int flow_size)
{
  Reset();
  measurements.reserve(flow_size);
  predictions_as_called.reserve(flow_size);
  predictions_ref.reserve(flow_size);
  state_inphase.reserve(flow_size);

  run_id.reserve(5);
  read_group.reserve(50);
  read_bases.reserve(2*flow_size);

  target_bases.reserve(2*flow_size);
  query_bases.reserve(2*flow_size);
  pretty_align.reserve(2*flow_size);

  aln_flow_order.reserve(2*flow_size);
  aligned_qHPs.reserve(2*flow_size);
  aligned_tHPs.reserve(2*flow_size);
  align_flow_index.reserve(2*flow_size);
  pretty_flow_align.reserve(2*flow_size);
}

// -----------------------------------------------------------------------
// Function Initializes non-alignment related read information:
//         run_id & coordinates from alignment name
//         read group name
//         read bases (query bases from alignment)
//         measurements from ZM tag
//         phasing from ZP tag
//         start flow from ZF tag

bool ReadAlignmentInfo::UnpackReadInfo(BamAlignment* new_alignment, const CalibrationContext& calib_context)
{
  Reset();
  alignment = new_alignment;
  is_filtered = false;

  // Extract run id & read coordinates

  if (not alignment->Name.empty()) {
	run_id  = alignment->Name.substr(0,alignment->Name.find(":"));
    well_xy.resize(2, 0);
    if (not ion_readname_to_xy(alignment->Name.c_str(), &well_xy[0], &well_xy[1]))
      run_id.clear();
  }
  if (run_id.empty()){
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Unable to determine run id or coordinates of read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  if (calib_context.chip_subset.CoordinatesToRegionIdx(well_xy[0],well_xy[1])<0){
    cerr << "Calibration ERROR: Read " << alignment->Name << " is outside of block boundaries." << endl;
    exit(EXIT_FAILURE);
  }

  std::map<string,int>::const_iterator fo_it = calib_context.flow_order_index_by_run_id.find(run_id);
  if (fo_it == calib_context.flow_order_index_by_run_id.end()){
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: No matching flow oder found for read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  flow_order_index = fo_it->second;

  // Retrieve read group information

  if (not alignment->GetTag("RG",read_group)) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: No read group found in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }

  // make sure the read group is present in the header
  std::map<string,string>::const_iterator key_it = calib_context.key_by_read_group.find(read_group);
  if (key_it == calib_context.key_by_read_group.end()){
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: No matching read group found for read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }

  // Get read bases (query)

  read_bases = alignment->QueryBases;
  if (read_bases.empty()) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Ignoring length zero read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }

  // Retrieve measurements from ZM tag

  vector<int16_t> quantized_measurements;
  if (not alignment->GetTag("ZM", quantized_measurements)) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Normalized measurements ZM:tag is not present in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  if ((int)quantized_measurements.size() > calib_context.num_flows_by_run_id.at(run_id)) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Normalized measurements ZM:tag length " << quantized_measurements.size()
           << " exceeds flow order length " << calib_context.num_flows_by_run_id.at(run_id)
           <<" in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  measurements.assign(calib_context.num_flows_by_run_id.at(run_id), 0.0);
  for (size_t counter = 0; counter < quantized_measurements.size(); ++counter)
    measurements[counter] = (float)quantized_measurements[counter]/256;
  measurements_length = quantized_measurements.size();

  // Retrieve phasing parameters from ZP tag

  if (not alignment->GetTag("ZP", phase_params)) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Phasing Parameters ZP:tag is not present in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  if (phase_params.size() != 3) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Phasing Parameters ZP:tag does not have 3 phase parameters in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  if ((phase_params[0] < 0) or (phase_params[0] > 1) or (phase_params[1] < 0) or (phase_params[1] > 1)
      or (phase_params[2] < 0) or (phase_params[2] > 1)) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Phasing Parameters ZP:tag outside of [0,1] range in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }
  if (calib_context.skip_droop)
    phase_params[2] = 0.0f;   // set droop to zero if switch activated

  // Retrieve start flow from BAM

  start_flow = 0;
  if (not alignment->GetTag("ZF", start_flow)) {
    uint8_t start_flow_byte = 0;
    if (not alignment->GetTag("ZF", start_flow_byte)) {
      if (calib_context.verbose_level > 0)
        cerr << "Calibration WARNING: Start Flow ZF:tag not found in read " << alignment->Name << endl;
      is_filtered = true;
      return false;
    }
    start_flow = (int)start_flow_byte;
  }
  if (start_flow == 0) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Start Flow ZF:tag has zero value in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }

  // And what we really want is not the flow of the first read base but the
  // flow of the last prefix base because the alignment might start with a substitution
  // Here: Get flow corresponding to last key or barcode base.

  prefix_flow = 0;
  unsigned int base_idx = 0;
  const ion::FlowOrder & flow_order = calib_context.flow_order_vector.at(flow_order_index);
  const std::string & prefix_bases = key_it->second;

  while ((base_idx < prefix_bases.length()) and (prefix_flow < flow_order.num_flows())) {
    while ((prefix_flow < flow_order.num_flows()) and (flow_order.nuc_at(prefix_flow) != prefix_bases.at(base_idx)))
      prefix_flow++;
    base_idx++;
  }

  return true;
}

// -----------------------------------------------------------------------
// This function extracts the alignment information in the BAM record

bool ReadAlignmentInfo::UnpackAlignmentInfo (const CalibrationContext& calib_context, bool debug) {

  if (is_filtered or (not alignment->IsMapped()))
    return false;

  // *** Extract base space Alignment

  string md_tag;
  if (not alignment->GetTag("MD", md_tag)) {
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: MD tag not found in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }

  // Retrieve base alignment information & create reference sequence from tags
  string pretty_tseq;  // Aligned target sequence including gaps.
  string pretty_qseq;  // Aligned query sequence including gaps
  RetrieveBaseAlignment(alignment->QueryBases, alignment->CigarData, md_tag,
  				target_bases, query_bases, pretty_tseq, pretty_qseq, pretty_align, left_sc, right_sc);

  // Filter based on alignment length
  if (target_bases.length() < calib_context.min_align_length) {
    is_filtered = true;
    return false;
  }

  // *** make sure all our alignment quantities are in read direction

  start_sc = left_sc;
  if (alignment->IsReverseStrand()) {

    start_sc = right_sc;

    ReverseComplementInPlace(read_bases);
    ReverseComplementInPlace(target_bases);
    ReverseComplementInPlace(query_bases);
    ReverseComplementInPlace(pretty_align);
  }

  // Update prefix flow and create flow index vector
  const ion::FlowOrder & flow_order = calib_context.flow_order_vector.at(flow_order_index);
  unsigned int base_idx = 0;

  while ((base_idx < start_sc) and (prefix_flow < (int)flow_order.num_flows())) {
    while ((prefix_flow < flow_order.num_flows()) and (flow_order.nuc_at(prefix_flow) != read_bases.at(base_idx)))
        prefix_flow++;
      base_idx++;
  }

  // *** Do flow alignment

  if (calib_context.do_flow_alignment) {

    //const ion::FlowOrder & flow_order = calib_context.flow_order_vector.at(flow_order_index);

    if (not PerformFlowAlignment(target_bases, query_bases, flow_order.str(), prefix_flow,
              aln_flow_order, aligned_qHPs, aligned_tHPs, align_flow_index, pretty_flow_align))
    {
      if (calib_context.verbose_level > 0)
        cerr << "Calibration WARNING: Flow alignment failed in read " << alignment->Name << endl;
      is_filtered = true;
      return false;
    }
  }

  // -------------------------------------------------
  // XXX print debug info
  if (debug > 0) {
  cout << "----------------------" << endl;
  cout << alignment->Name << " : read group " << read_group << " : run id " << run_id  << " : x " << well_xy[0] << " y " << well_xy[1] << endl;
  cout << "Prefix flow: " << prefix_flow << " Start Flow: " << start_flow << " StartSC: " << start_sc << " leftSC " << left_sc << " rightSC " << right_sc << endl;
  cout << (alignment->IsReverseStrand() ? "Reverse " : "Forward ") << " Cigar: ";
  for (unsigned int iCE=0; iCE<alignment->CigarData.size(); ++iCE){
    cout << alignment->CigarData.at(iCE).Type << alignment->CigarData.at(iCE).Length;
  }
  cout << endl << endl;


  cout << "Base Space Alignment: " << endl;
  cout << "Query:  " << pretty_qseq  << endl;
  cout << "Align:  " << pretty_align << endl;
  cout << "Target: " << pretty_tseq << endl << endl;

  cout << "Flow Space Alignment: " << endl;
  cout << "Fl-Idx: ";
  for (unsigned int iHP=0; iHP<align_flow_index.size(); ++iHP)
    cout << align_flow_index.at(iHP) << " ";
  cout << endl;
  cout << "Fl-Nuc: ";
  for (unsigned int iHP=0; iHP<aln_flow_order.size(); ++iHP)
    cout << aln_flow_order.at(iHP);
  cout << endl;
  cout << "Query:  ";
  for (unsigned int iHP=0; iHP<aligned_qHPs.size(); ++iHP)
    cout << aligned_qHPs.at(iHP);
  cout << endl;
  cout << "Align:  ";
  for (unsigned int iHP=0; iHP<pretty_flow_align.size(); ++iHP)
    cout << pretty_flow_align.at(iHP);
  cout << endl;
  cout << "Target: ";
  for (unsigned int iHP=0; iHP<aligned_tHPs.size(); ++iHP)
    cout << aligned_tHPs.at(iHP);
  cout << endl << endl;

  cout << "Vector Sizes: idx " << align_flow_index.size()
	   << "  Nuc " << aln_flow_order.size() << "  Q " << aligned_qHPs.size() << "  A " << pretty_flow_align.size() << "  T " << aligned_tHPs.size() << endl << endl;
  }
  // ------------------------------------------------- */

  return true;
}

// -----------------------------------------------------------------------
// This function generates the predicted sequences

void ReadAlignmentInfo::GeneratePredictions (vector<DPTreephaser>& treephaser_vector, const CalibrationContext& calib_context){

  if (is_filtered)
    return;

  DPTreephaser & treephaser = treephaser_vector.at(flow_order_index);
  treephaser.SetModelParameters(phase_params[0], phase_params[1], phase_params[2]);

  BasecallerRead read;
  read.SetData(measurements, measurements.size());

  // *** Simulate read as called

  std::map<string,string>::const_iterator key_it = calib_context.key_by_read_group.find(read_group);
  read.sequence.reserve(2*read_bases.length());

  std::copy(key_it->second.begin(), key_it->second.end(), back_inserter(read.sequence));
  std::copy(read_bases.begin(), read_bases.end(), back_inserter(read.sequence));

  treephaser.Simulate(read, measurements.size(), true);
  predictions_as_called.swap(read.prediction);
  state_inphase.swap(read.state_inphase);

  // *** Simulate reference hypothesis
  // The start of the read might be soft clipped in which case we keep the bases as called

  if (alignment->IsMapped()) {
    read.sequence.resize(key_it->second.length() + start_sc);
    std::copy(target_bases.begin(), target_bases.end(), back_inserter(read.sequence));

    treephaser.Simulate(read, measurements.size());
    predictions_ref = read.prediction;
  }

}

// =======================================================================

MultiBamHandler::MultiBamHandler() :
    have_bam_files_(false), current_bam_idx_(0)
{

}

// -----------------------------------------------------------------------

bool  MultiBamHandler::Open(vector<string> bam_names)
{
  Close();
  bam_readers_.assign(bam_names.size(), NULL);

  for (unsigned int bam_idx=0; bam_idx<bam_names.size(); ++bam_idx){
    bam_readers_.at(bam_idx) = new BamTools::BamReader;

    if (not bam_readers_.at(bam_idx)->Open(bam_names.at(bam_idx))) {
      cerr << "MultiBamHandler ERROR: Cannot open BAM file " << bam_names.at(bam_idx) << endl;
      Close();
      return false;
    }

    sam_headers_.push_back(bam_readers_.at(bam_idx)->GetHeader());
    merged_read_groups_.Add(sam_headers_.at(bam_idx).ReadGroups);
  }

  have_bam_files_  = true;
  return true;
}

// -----------------------------------------------------------------------

void  MultiBamHandler::Close()
{
  for (unsigned int bam_idx=0; bam_idx<bam_readers_.size(); ++bam_idx){
    if (bam_readers_.at(bam_idx) != NULL) {
      bam_readers_.at(bam_idx)->Close();
      delete bam_readers_.at(bam_idx);
    }
  }
  bam_readers_.clear();
  sam_headers_.clear();
  merged_read_groups_.Clear();
  current_bam_idx_ = 0;
  have_bam_files_ = false;
}

// -----------------------------------------------------------------------
// We simply take the reads serially out of the different BAM files

bool  MultiBamHandler::GetNextAlignment(BamAlignment & alignment)
{
  if (not have_bam_files_)
    return false;

  bool success = false;

  while ((not success) and (current_bam_idx_ < bam_readers_.size())){
    success = bam_readers_.at(current_bam_idx_)->GetNextAlignment(alignment);
    if (not success){
      current_bam_idx_++;
    }
  }

  if (not success) {
    have_bam_files_ = false;
    return false;
  }
  else
    return true;
}

// =======================================================================

