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
#include "LinearCalibrationModel.h"

// -----------------------------------------------------------------------

void PrintHelp_Calibration()
{
  cout << endl;
  cout << "Calibrate homopolymers for a run using training data." << endl;
  cout << "Usage: Calibration [options]"                          << endl << endl;

  cout << "Required Arguments:"                                                                               << endl;
  cout << "  -i,--input                        FILE      mapped input BAM file(s)"                            << endl;
  cout << "     --block-size                   INT,INT   block/chip x,y dimensions"                           << endl;
  cout << endl;
  cout << "Optional Arguments:"                                                                                << endl;
  cout << "  -o,--output-dir                   DIR       output directory                            [.]"      << endl;
  cout << "     --block-offset                 INT,INT   block offset coordinates x,y                [0,0]"    << endl;
  cout << "     --num-calibration-regions      INT,INT   number of regions in x,y direction          [2,2]"    << endl;
  cout << "     --num-threads                  INT       number of worker threads                    [nCores]" << endl;
  cout << "     --num-reads-per-thread         INT       reads to load per thread                    [1000]"   << endl;
  cout << "     --flow-window-size             INT       target size of a flow window                [250]"    << endl;
  cout << endl; // Fit options
  cout << "     --successive-fit               BOOL      Fit models successively, in order of appl.  [on]"     << endl;
  cout << "     --blind-fit                    BOOL      Fit without using alignment information     [off]"    << endl;
  cout << "     --num-train-iterations         INT       Number of blind training iterations         [5]"      << endl;
  cout << endl; // Alignment options
  cout << "     --min-mapping-qv               INT       Minimum mapping quality for reads           [8]"      << endl;
  cout << "     --min-align-length             INT       Minimum target alignment length             [30]"     << endl;
  cout << "     --load-unmapped                BOOL      Allow loading of unmapped reads             [off]"    << endl;
  cout << "     --do-flow-alignment            BOOL      Do full dynamic programming flow alignment  [off]"    << endl;
  cout << "     --align-fill-gaps              BOOL      Fill alignment sections with many errors    [off]"    << endl;
  cout << "     --align-match-zero             BOOL      Match zeros in light flow alignment         [off]"    << endl;
  cout << endl; // Solver & general
  cout << "     --resolve-clipped-bases        BOOL      Solve hard clipped read prefix              [off]"    << endl;
  cout << "     --skip-droop                   INT       Disregard droop in prediction generation    [true]"   << endl;
  cout << "     --verbose                      INT       Set verbose level                           [1]"      << endl;
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

  // General program options
  chip_subset.InitializeCalibrationRegionsFromOpts(opts);
  string  output_dir   = opts.GetFirstString ('o', "output-dir", ".");
  filename_json        = output_dir + "/Calibration.json";

  num_threads          = opts.GetFirstInt     ('-', "num-threads", max(numCores(), 4));
  num_reads_per_thread = opts.GetFirstInt     ('-', "num-reads-per-thread", 250);
  flow_window_size     = opts.GetFirstInt     ('-', "flow-window-size", 250);
  rand_seed            = opts.GetFirstInt     ('-', "rand-seed", 631);

  // Given a target window length we distribute the number of flows into equal sized windows
  int num_flow_windows = max(1, max_num_flows/flow_window_size);
  flow_window_size = (max_num_flows+num_flow_windows-1)/num_flow_windows;

  // Model fit options
  successive_fit       = opts.GetFirstBoolean ('-', "successive-fit", true);
  blind_fit            = opts.GetFirstBoolean ('-', "blind-fit", false);
  num_train_iterations = opts.GetFirstInt     ('-', "num-train-iterations", 5);  // only used for blind
  if (not blind_fit)
    num_train_iterations = 1;

  // Alignment options
  load_unmapped        = opts.GetFirstBoolean ('-', "load-unmapped", blind_fit);
  do_flow_alignment    = opts.GetFirstBoolean ('-', "do-flow-alignment", false);
  match_zero_flows     = opts.GetFirstBoolean ('-', "align-match-zero", false);
  fill_strange_gaps    = opts.GetFirstDouble  ('-', "align-fill-gaps", -0.5f); // default to not do this

  // General filters
  min_mapping_qv       = opts.GetFirstInt     ('-', "min-mapping-qv", 8);
  min_align_length     = opts.GetFirstInt     ('-', "min-align-length", 30);

  // Solver options
  resolve_clipped_bases= opts.GetFirstBoolean ('-', "resolve-clipped-bases", false);
  skip_droop           = opts.GetFirstBoolean ('-', "skip-droop", true);

  verbose_level        = opts.GetFirstInt     ('-', "verbose", 1);
  debug                = opts.GetFirstBoolean ('-', "debug", false);

  // Set training options for first pass through BAM
  local_fit_linear_model = true;
  local_fit_polish_model = successive_fit ? false : true;

  // Variables to log some information
  num_reads_in_bam = 0;
  num_mapped_reads = 0;
  num_loaded_reads = 0;
  num_useful_reads = 0;

  // Program threading information
  num_model_reads  = 0;
  num_model_writes = 0;
  wait_to_read_model  = false;
  wait_to_write_model = true;

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
    cout << "Calibration Options" << (debug ? " DEBUG MODE:" : ":")<< endl;
    cout << "   Output file            : " << filename_json << endl;
    cout << "   num-threads            : " << num_threads   << endl;
    cout << "   num-reads-per-thread   : " << num_reads_per_thread << endl;
    cout << "   flow-window-size       : " << flow_window_size << endl;
    cout << "   successive-fit         : " << (successive_fit ? "on" : "off") << endl;
    cout << "   blind-fit              : " << (blind_fit ? "on" : "off") << endl;
    cout << "   num-train-iterations   : " << num_train_iterations << endl;
    cout << "   load-unmapped          : " << (load_unmapped ? "on" : "off") << endl;
    cout << "   do-flow-alignment      : " << (do_flow_alignment ? "on" : "off") << endl;
    cout << "   align-match-zero       : " << (match_zero_flows ? "on" : "off") << endl;
    cout << "   align-fill-gaps        : " << fill_strange_gaps << endl;
    cout << "   min-mapping-qv         : " << min_mapping_qv   << endl;
    cout << "   min-align-length       : " << min_align_length << endl;
    cout << "   resolve-clipped-bases  : " << (resolve_clipped_bases ? "on" : "off") << endl;
    cout << "   skip-droop             : " << (skip_droop ? "on" : "off") << endl;
    cout << "   verbose                : " << verbose_level << endl;

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
  LastJsonInfo(json);
}

void CalibrationContext::LastJsonInfo(Json::Value &json){
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
  prefix_bases.clear();
  well_xy.clear();
  start_flow = -1;
  prefix_flow = -1;
  flow_order_index = -1;

  // Base alignment information
  target_bases.clear();
  query_bases.clear();
  pretty_align.clear();
  full_target_bases.clear();
  full_query_bases.clear();
  left_sc = right_sc = start_sc = 0;

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
  full_target_bases.reserve(2*flow_size);
  full_query_bases.reserve(2*flow_size);

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

bool ReadAlignmentInfo::UnpackReadInfo(BamAlignment* new_alignment, vector<DPTreephaser>& treephaser_vector, const CalibrationContext& calib_context)
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
  // Here: Get flow corresponding to last hard clipped base.

  prefix_flow = -1;
  if (not calib_context.resolve_clipped_bases) {

    // Construct hard clipped prefix from tags [KS][ZK][ZT][ZE]
    prefix_bases = key_it->second;
    std::string temp_zk, temp_zt, temp_ze;
    if (alignment->GetTag("ZK", temp_zk))
      prefix_bases += temp_zt;
    if (alignment->GetTag("ZT", temp_zt))
      prefix_bases += temp_zt;
    if (alignment->GetTag("ZE", temp_ze))
      prefix_bases += temp_ze;

    // Get prefix flow from prefix bases

    const ion::FlowOrder & flow_order = calib_context.flow_order_vector.at(flow_order_index);
    if (prefix_bases.length()>0){
      prefix_flow = 0;
      unsigned int base_idx = 0;
      while ((base_idx < prefix_bases.length()) and (prefix_flow < flow_order.num_flows())) {
        while ((prefix_flow < flow_order.num_flows()) and (flow_order.nuc_at(prefix_flow) != prefix_bases.at(base_idx)))
          prefix_flow++;
        base_idx++;
      }
    }

    // Check consistency of prefix_flow and start_flow (we might have a hard clipped region that has not been accounted for)

    char first_read_base = read_bases.at(0);
    if (alignment->IsMapped() and alignment->IsReverseStrand())
      first_read_base = NucComplement(read_bases.at(read_bases.length()-1));

    if (prefix_flow>=0) {
      int check_start_flow = prefix_flow;
      while (check_start_flow < flow_order.num_flows() and  flow_order.nuc_at(check_start_flow) != first_read_base)
        check_start_flow++;
      if (check_start_flow != start_flow) {
        prefix_flow = -1;
        prefix_bases.clear();
      }
    }
  }

  // If desired or if the above check failed, we solve the read prefix

  if (prefix_flow < 0) {

    DPTreephaser & treephaser = treephaser_vector.at(flow_order_index);
    treephaser.SetModelParameters(phase_params[0], phase_params[1], phase_params[2]);

    BasecallerRead master_read;
    master_read.SetData(measurements, measurements.size());
    prefix_flow = GetStartOfMasterRead(treephaser, master_read, calib_context);
    prefix_bases.append(master_read.sequence.begin(), master_read.sequence.end());
  }

  return true;
}

// -----------------------------------------------------------------------

int ReadAlignmentInfo::GetStartOfMasterRead(DPTreephaser & treephaser, BasecallerRead &master_read, const CalibrationContext& calib_context)
{
  // Solve beginning of potentially clipped read
  const ion::FlowOrder & flow_order = calib_context.flow_order_vector.at(flow_order_index);
  int until_flow = min((start_flow+20), calib_context.num_flows_by_run_id.at(run_id));

  treephaser.Solve( master_read, until_flow, 0);

  // StartFlow clipped? Get solved HP length at startFlow.
  unsigned int base = 0;
  int flow = 0;
  unsigned int HPlength = 0;
  while (base < master_read.sequence.size()) {
    while (flow < flow_order.num_flows() and flow_order.nuc_at(flow) != master_read.sequence[base]) {
      flow++;
    }
    if (flow > start_flow or flow == flow_order.num_flows())
      break;
    if (flow == start_flow)
      HPlength++;
    base++;
  }

  // Get HP size at the start of the read as called in Hypotheses[0]
  unsigned int count = 1;
  while (count < read_bases.length() and read_bases.at(count) == read_bases.at(0))
    count++;

  // Adjust the length of the prefix and erase extra solved bases
  if (HPlength>count)
    base -= count;
  else
    base -= HPlength;
  master_read.sequence.erase(master_read.sequence.begin()+base, master_read.sequence.end());

  // Get flow of last prefix base
  int prefix_flow = 0;
  for (unsigned int i_base = 0; i_base < master_read.sequence.size(); i_base++) {
    while (prefix_flow < flow_order.num_flows() and flow_order.nuc_at(prefix_flow) != master_read.sequence[i_base])
      prefix_flow++;
  }

  return prefix_flow;
}


// -----------------------------------------------------------------------
// This function extracts the alignment information in the BAM record

bool ReadAlignmentInfo::UnpackAlignmentInfo (const CalibrationContext& calib_context) {

  if (is_filtered)
    return false;
  if (not alignment->IsMapped() and (not calib_context.blind_fit)) {
    is_filtered = true;
    return false;
  }

  // *** Extract base space Alignment

  string pretty_tseq;  // Aligned target sequence including gaps.
  string pretty_qseq;  // Aligned query sequence including gaps
  bool need_pretty = false;

  // Blind fit is reference independent and relies on the called bases

  if (calib_context.blind_fit) {
    target_bases = query_bases = read_bases; // identical here
  }
  else {

    string md_tag;
    if (not alignment->GetTag("MD", md_tag)) {
      if (calib_context.verbose_level > 0)
        cerr << "Calibration WARNING: MD tag not found in read " << alignment->Name << endl;
      is_filtered = true;
      return false;
    }

    // Retrieve base alignment information & create reference sequence from cigar & MD tag
    RetrieveBaseAlignment(alignment->QueryBases, alignment->CigarData, md_tag,
                          target_bases, query_bases, pretty_tseq, pretty_qseq, pretty_align, left_sc, right_sc);
    need_pretty = calib_context.debug or !calib_context.do_flow_alignment;
  }

  // Filter reads based on alignment length
  if (target_bases.length() < calib_context.min_align_length) {
    is_filtered = true;
    return false;
  }

  // *** make sure all our alignment quantities are in read direction

  start_sc = left_sc;
  if (alignment->IsMapped() and alignment->IsReverseStrand()) {

    start_sc = right_sc;
    RevComplementInPlace(read_bases);
    RevComplementInPlace(target_bases);
    RevComplementInPlace(query_bases);

    if (need_pretty) { // If we need the pretty representation
      RevComplementInPlace(pretty_align);
      RevComplementInPlace(pretty_qseq);
      RevComplementInPlace(pretty_tseq);
    }
  }

  // *** Create full base string including key, barcode plus 5' hard & soft clipped bases

  full_query_bases  = prefix_bases + read_bases.substr(0, start_sc) + query_bases;
  full_target_bases = prefix_bases + read_bases.substr(0, start_sc) + target_bases;

  // Update prefix flow to reflect first last non-aligned base and create flow index vector
  const ion::FlowOrder & flow_order = calib_context.flow_order_vector.at(flow_order_index);
  unsigned int base_idx = 0;

  while ((base_idx < start_sc) and (prefix_flow < (int)flow_order.num_flows())) {
    while ((prefix_flow < flow_order.num_flows()) and (flow_order.nuc_at(prefix_flow) != read_bases.at(base_idx)))
      prefix_flow++;
    base_idx++;
  }

  // *** Crate flow alignment information for comparing calls

  bool align_success = true;

  if (calib_context.blind_fit) {

    NullFlowAlignment(full_target_bases, full_query_bases, flow_order.str(),0,
                      aln_flow_order, aligned_qHPs, aligned_tHPs,align_flow_index,pretty_flow_align);
  }
  else if (calib_context.do_flow_alignment){

    align_success = PerformFlowAlignment(full_target_bases, full_query_bases, flow_order.str(), 0,
                           aln_flow_order, aligned_qHPs, aligned_tHPs, align_flow_index, pretty_flow_align);
  }
  else { // light flow alignment

    string full_aligned_query_bases, full_aligned_target_bases;
    full_aligned_query_bases = prefix_bases + read_bases.substr(0,start_sc) + pretty_qseq;
    full_aligned_target_bases = prefix_bases+ read_bases.substr(0,start_sc) + pretty_tseq;
    align_success = LightFlowAlignment(full_aligned_target_bases, full_aligned_query_bases, flow_order.str(),
                         calib_context.match_zero_flows, calib_context.fill_strange_gaps, aln_flow_order,
                         aligned_qHPs, aligned_tHPs, align_flow_index, pretty_flow_align, full_target_bases);
  }

  if (not align_success){
    if (calib_context.verbose_level > 0)
      cerr << "Calibration WARNING: Flow alignment failed in read " << alignment->Name << endl;
    is_filtered = true;
    return false;
  }

  // -------------------------------------------------
  // XXX print debug info
  if (calib_context.debug) {
    cout << "----------------------" << endl;
    cout << alignment->Name << " : read group " << read_group << " : run id " << run_id  << " : x " << well_xy[0] << " y " << well_xy[1] << endl;
    cout << "Prefix flow: " << prefix_flow << " Start Flow: " << start_flow << " StartSC: " << start_sc << " leftSC " << left_sc << " rightSC " << right_sc << endl;

    if (alignment->IsMapped() and !calib_context.blind_fit) {
      cout << (alignment->IsReverseStrand() ? "Reverse " : "Forward ") << " strand read, Cigar: ";
      for (unsigned int iCE=0; iCE<alignment->CigarData.size(); ++iCE){
        cout << alignment->CigarData.at(iCE).Type << alignment->CigarData.at(iCE).Length;
      }
      cout << endl;

      cout << "Base Space Alignment: " << endl;
      cout << "Query:  " << pretty_qseq  << endl;
      cout << "Align:  " << pretty_align << endl;
      cout << "Target: " << pretty_tseq << endl;
    }

    cout << endl << "Flow Space Alignment: " << endl;
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

    //cout << "Vector Sizes: idx " << align_flow_index.size()
    //     << "  Nuc " << aln_flow_order.size() << "  Q " << aligned_qHPs.size() << "  A " << pretty_flow_align.size() << "  T " << aligned_tHPs.size() << endl << endl;
  }
  // ------------------------------------------------- */

  return true;
}

// -----------------------------------------------------------------------
// This function generates the predicted sequences

void ReadAlignmentInfo::GeneratePredictions (vector<DPTreephaser>& treephaser_vector,  LinearCalibrationModel& linear_model_local){

  if (is_filtered)
    return;

  DPTreephaser & treephaser = treephaser_vector.at(flow_order_index);
  treephaser.SetModelParameters(phase_params[0], phase_params[1], phase_params[2]);
  const vector<vector<vector<float> > > * aPtr = 0;
  const vector<vector<vector<float> > > * bPtr = 0;

  if (linear_model_local.is_enabled() ){  // if we have generated a model, use it
    // Equal calibration opportunity for everybody! (except TFs!)
    //cout <<"Well: " <<well_xy[0] << "\t" <<well_xy[1] <<endl;
    aPtr = linear_model_local.getAs(well_xy[0], well_xy[1]);
    bPtr =linear_model_local.getBs(well_xy[0], well_xy[1]);
    treephaser.SetAsBs(aPtr, bPtr); // Set/delete recalibration model for this read

  }

  BasecallerRead read;
  read.SetData(measurements, measurements.size());

  // *** Simulate read as called

  read.sequence.reserve(2*read_bases.length());
  std::copy(prefix_bases.begin(), prefix_bases.end(), back_inserter(read.sequence));
  std::copy(read_bases.begin(), read_bases.end(), back_inserter(read.sequence));

  treephaser.Simulate(read, measurements.size(), true);
  predictions_as_called.swap(read.prediction);
  state_inphase.swap(read.state_inphase);

  // *** Simulate reference hypothesis
  // The start of the read might be soft clipped in which case we keep the bases as called

  if (alignment->IsMapped()) {
    read.sequence.resize(prefix_bases.length() + start_sc);
    std::copy(target_bases.begin(), target_bases.end(), back_inserter(read.sequence));
    // reference must be simulated without recalibration to regress properly
    treephaser.DisableRecalibration();
    treephaser.Simulate(read, measurements.size());
    predictions_ref = read.prediction;
  } else {
    read.sequence.resize(prefix_bases.length() + start_sc);
    std::copy(target_bases.begin(), target_bases.end(), back_inserter(read.sequence));
    // reference must be simulated without recalibration to regress properly
    treephaser.DisableRecalibration();
    treephaser.Simulate(read, measurements.size());
    predictions_ref = read.prediction;
  }

  // simulate blind fit hypothesis?

}

// =======================================================================

MultiBamHandler::MultiBamHandler() :
  have_bam_files_(false), no_more_data_(true), current_bam_idx_(0), num_bam_passes_(-1)
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
  no_more_data_    = false; // potential available data!
  num_bam_passes_  = 0;
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
  num_bam_passes_  = -1;
  have_bam_files_ = false;
}

// -----------------------------------------------------------------------
// We simply take the reads serially out of the different BAM files

bool  MultiBamHandler::GetNextAlignmentCore(BamAlignment & alignment)
{
  if ((not have_bam_files_) or no_more_data_)
    return false;

  bool success = false;

  while ((not success) and (current_bam_idx_ < bam_readers_.size())){
    success = bam_readers_.at(current_bam_idx_)->GetNextAlignmentCore(alignment);
    if (not success){
      current_bam_idx_++;
    }
  }

  if (not success) {
    no_more_data_ = true;
    return false;
  }
  else
    return true;
}

// -----------------------------------------------------------------------

bool MultiBamHandler::Rewind(void){
  if (not have_bam_files_)
    return(false);

  for (unsigned int bam_idx=0; bam_idx<bam_readers_.size(); ++bam_idx){
    if (bam_readers_.at(bam_idx) != NULL) {
      bam_readers_.at(bam_idx)->Rewind();
    }
  }
  no_more_data_= false; // back to having potential data
  current_bam_idx_ = 0; // back to start of all bam files read sequentially
  ++num_bam_passes_;
  return(true);
}


// =======================================================================

