/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
/* Tool to compare the content of unaligned BAM files produced by the BaseCaller */

#include <string>
#include <ctime>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "json/json.h"

#include "OptArgs.h"
#include "IonVersion.h"
#include "Utils.h"
#include "ion_util.h"
#include "BaseCallerUtils.h"
#include "MiscUtil.h"

#include "api/BamMultiReader.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamWriter.h"
#include "api/BamAlignment.h"

using namespace std;
using namespace BamTools;

int PrintHelp()
{
  cout << " --- Merge the content of ION resequencing bam files with UMTs.\n";
  cout << "Usage: MergeReseqBam [options]" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << " -l,--logfile          FILE       Filename for json log file." << endl;
  cout << " -i,--input-bam        FILE,FILE  Input bam files" << endl;
  cout << " -o,--output-bam       FILE       Output bam for merged results" << endl;
  cout << " -t,--num-threads      INT        Number of threads used by bam writer to do compression. [def. 12]" << endl;
  cout << "    --prefix-mol-tag   STRING     Structure of prefix molecular tag                   {ACGTN bases}" << endl;
  cout << "    --suffix-mol-tag   STRING     Structure of suffix molecular tag                   {ACGTN bases}" << endl;
  cout << "    --max-residual     FLOAT      Maximum excess residual to do a tag update.             [2.0]"     << endl;
  cout << "    --max-warnings     INT        Maximum number of warnings to be printed.               [20]"      << endl;
  cout << " -v,--verbose          BOOL       Print detailed information for wanrings                 [False]"   << endl;
  cout << " -------------------------------------------------------- " << endl;

  return 1;
}


void DumpStartingStateOfProgram (int argc, const char *argv[], time_t analysis_start_time, Json::Value &json)
{
  char my_host_name[128] = { 0 };
  gethostname (my_host_name, 128);
  string command_line = argv[0];
  for (int i = 1; i < argc; i++) {
    command_line += " ";
    command_line += argv[i];
  }

  cout << "---------------------------------------------------------------------" << endl;
  cout << "MergeReseqBam " << IonVersion::GetVersion() << "." << IonVersion::GetRelease()
       << " (" << IonVersion::GetGitHash() << ") (" << IonVersion::GetBuildNum() << ")" << endl;
  cout << "Command line = " << command_line << endl;

  json["host_name"]    = my_host_name;
  json["start_time"]   = get_time_iso_string(analysis_start_time);
  json["version"]      = IonVersion::GetVersion() + "." + IonVersion::GetRelease();
  json["git_hash"]     = IonVersion::GetGitHash();
  json["build_number"] = IonVersion::GetBuildNum();
  json["command_line"] = command_line;
}

// --------------------------------------------------------------------

struct FlowsAndKeys
{
  vector<ion::FlowOrder >   flow_order_vector;
  map<string, int>          flow_order_index_by_run_id;
  map<string, int>          num_flows_by_run_id;
  map<string, string>       key_by_read_group;
};

// --------------------------------------------------------------------

bool ion_readname_to_xy(const string readname, int& x, int& y) {
  // Assume the Ion read format is a 3 field colon separated string of the form
  // "Identifyer":"y (or row) value":"x (or column) value"
  bool valid_read = true;
  int value = 0;
  int group = 0;

  for (unsigned int idx=0; idx < readname.length(); idx++) {

    if (readname.at(idx) == ':') {
      if (group == 1)
        y = value;
      value = 0; // reset
      group++;
    }
    else if ('0' <= readname.at(idx) && readname.at(idx) <= '9') {
      value *= 10;
      value += (int)(readname.at(idx) - '0');
    }
    else if (group > 0)
      valid_read = false;
  }

  if (valid_read and group == 2)
    x = value;
  else {
    x = 0;
    y = 0;
    valid_read = false;
  }
  return valid_read;
}

// --------------------------------------------------------------------
// Return index corresponding to smaller coordinates (in row - column sorting)
// Return zero if reads have same coordinates
// Return -1 if error occurred

int compareWellCoordinates(const string read_name1, const string read_name2) {

  int row1, col1, row2, col2;
  if (not(ion_readname_to_xy(read_name1, col1, row1))) {
    cerr << "MergeReseqBam ERROR parsing coordinates from read name: " << read_name1 << endl;
    return -1;
  }
  if (not(ion_readname_to_xy(read_name2, col2, row2))) {
    cerr << "MergeReseqBam ERROR parsing coordinates from read name: " << read_name1 << endl;
    return -1;
  }
  // Assume sorting by row - column
  if (row1 < row2) {
    return 1;
  }
  else if (row1 > row2) {
    return 2;
  }
  else {
    if (col1 < col2) {
      return 1;
    }
    else if (col1 > col2) {
      return 2;
    }
    else {
      return 0;
    }
  }
}

string getRunID(const string &read_name){
  return read_name.substr(0, read_name.find(":"));
}

// --------------------------------------------------------------------

void SaveJson(const Json::Value & json, const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    cerr << "Unable to write JSON file " + filename_json << endl;
}

// --------------------------------------------------------------------

bool StrictStrctureMatch(const string& mol_tag, const string& tag_struct)
{
  if ((mol_tag.size() != tag_struct.size())){
    return false;
  }

  string::const_iterator tag_it = mol_tag.begin();
  for (string::const_iterator struct_it = tag_struct.begin(); struct_it != tag_struct.end(); ++struct_it, ++tag_it){
    if ((*struct_it != 'N') and (*tag_it != *struct_it)){
      return false;
    }
  }
  return true;
}

// --------------------------------------------------------------------

void RecordTagStructureMatch(vector<unsigned int> & log, const BamTools::BamAlignment& alignment,
    const string& zt_structure, const string& yt_structure)
{
  string tag;
  bool have_tag = alignment.GetTag("ZT", tag);
  if (have_tag and StrictStrctureMatch(tag, zt_structure))
    ++log.at(0);
  have_tag = alignment.GetTag("YT", tag);
  if (have_tag and StrictStrctureMatch(tag, yt_structure))
    ++log.at(1);
}

// --------------------------------------------------------------------

int GetLastIncorporatingFlow(const string & bases, const ion::FlowOrder & flow_order)
{
  unsigned int base_idx = 0;
  int flow = 0;
  while (base_idx < bases.length() and flow < flow_order.num_flows()) {
    while (flow < flow_order.num_flows() and  flow_order.nuc_at(flow) != bases.at(base_idx))
      flow++;
    base_idx++;
  }
  return flow;
}

// --------------------------------------------------------------------

// Flow alignment of a particular sequence in a flow order context
vector<int> Seq2Flow(const string & bases, const ion::FlowOrder & flow_order, int flow=0){
  vector<int> aln;
  unsigned int base_idx = 0;

  while (base_idx < bases.length() and flow < flow_order.num_flows()) {
    int my_hp = 0;
    while (base_idx < bases.length() and flow_order.nuc_at(flow) == bases.at(base_idx)){
      ++my_hp;
      ++base_idx;
    }
    aln.push_back(my_hp);
    ++flow;
  }
  return aln;
}


// --------------------------------------------------------------------

void PrintWarning(const string & warn_text, const string & aln_name, vector<unsigned int> &  max_warnings)
{
  if (max_warnings.at(0) < max_warnings.at(1)){
    ++max_warnings.at(0);
    cerr << warn_text << " " << aln_name << endl;
  }
  if (max_warnings.at(0) == max_warnings.at(1)){
    ++max_warnings.at(0);
    cerr << "MergeReseqBams WARNING: There were more than " << max_warnings.at(1) << " warnings!" << endl;
  }
}

// --------------------------------------------------------------------
// XXX
// The meat

bool UpdateTagInformation(const string& tag_name,
        BamTools::BamAlignment& alignment,
        const string          & new_tag_value,
        const FlowsAndKeys &    flows_and_keys,
        vector<unsigned int> &  max_warnings,
        unsigned int &          num_query_updates,
        unsigned int &          num_high_residual,
        double                  max_residual)
{
  if (tag_name == "YT"){
    alignment.EditTag(tag_name, "Z", new_tag_value);
    return true;
  }

  // --- The following reconciliation block is for ZT tag only

  // Step 1: Unpack read information from alignment

  // Find flow order for this alignment
  std::map<string,int>::const_iterator fo_it = flows_and_keys.flow_order_index_by_run_id.find(getRunID(alignment.Name));
  if (fo_it == flows_and_keys.flow_order_index_by_run_id.end()){
    PrintWarning("MergeReseqBams WARNING: No matching flow oder found for read ", alignment.Name, max_warnings);
    return false;
  }
  const ion::FlowOrder & flow_order = flows_and_keys.flow_order_vector.at(fo_it->second);

  // Get start flow of query
  int zf_flow = 0;
  if (not alignment.GetTag("ZF", zf_flow)){
    PrintWarning("MergeReseqBams WARNING: Start Flow ZF:tag not found in read ", alignment.Name, max_warnings);
    return false;
  }//*/

  // Get ZA and ZE tags
  int za_tag;
  if(not alignment.GetTag("ZA", za_tag)){
    uint32_t u_za = 0;
    if (not alignment.GetTag("ZA", u_za))
      za_tag = -1;
    else
      za_tag = (int) u_za;
  }

  string ze_tag;
  if (not alignment.GetTag("ZE", ze_tag))
    ze_tag.clear();
  int ze_length = ze_tag.length();

  // Get row & column index
  vector<int> well_rowcol(2);
  ion_readname_to_rowcol(alignment.Name.c_str(), &well_rowcol[0], &well_rowcol[1]);

  // Get query
  string read_bases = alignment.QueryBases;
  if (alignment.IsReverseStrand())
    RevComplementInPlace(read_bases);

  string prefix_bases, read_group, zk_tag, org_zt_tag;
  if (not alignment.GetTag("RG",read_group)) {
    PrintWarning("MergeReseqBams WARNING: No read group found in read ", alignment.Name, max_warnings);
    return false;
  }

  std::map<string,string>::const_iterator key_it = flows_and_keys.key_by_read_group.find(read_group);
  if (key_it != flows_and_keys.key_by_read_group.end())
    prefix_bases = key_it->second;
  if (alignment.GetTag("ZK", zk_tag)){
    prefix_bases += zk_tag;
  }
  int prefix_flow = GetLastIncorporatingFlow(prefix_bases, flow_order);

  if (not alignment.GetTag("ZT", org_zt_tag)){
    PrintWarning("MergeReseqBams WARNING: No ZT tag found in read ", alignment.Name, max_warnings);
    return false;
  }

  // Unpack measurements from BAM alignment

  vector<int16_t> quantized_measurements;
  if (not alignment.GetTag("ZM", quantized_measurements)) {
    PrintWarning("MergeReseqBams WARNING: Normalized measurements ZM:tag is not present in read ", alignment.Name, max_warnings);
    return false;
  }
  vector<float> measurements(quantized_measurements.size(), 0.0);
  for (unsigned int flow=0; flow < quantized_measurements.size(); ++flow){
    measurements.at(flow) = quantized_measurements.at(flow) / 256.0f;
  }
  //*/


  // Step 2: Do flow alignment to determine query (& ZF) correction

  int num_extra_bases = 16;
  int num_extra_query = num_extra_bases - ze_tag.length();
  string extra_bases;
  if (ze_tag.length() > 0)
    extra_bases += ze_tag.substr(0, num_extra_bases);
  if (num_extra_query > 0)
    extra_bases += read_bases.substr(0,min(num_extra_query, (int)read_bases.length()));

  vector<int> aln_org_read  = Seq2Flow(org_zt_tag+extra_bases, flow_order, prefix_flow);
  vector<int> aln_new_tag   = Seq2Flow(new_tag_value, flow_order, prefix_flow);

  // Insert residual sanity check here:

  if (measurements.size() < (unsigned int)prefix_flow + aln_new_tag.size()){
    PrintWarning("MergeReseqBams WARNING: Not enough measurements for residual check in read ", alignment.Name, max_warnings);
    return false;
  }

  // Check whether 16 additional bases of query sequence give indeed a longer alignment than the new tag
  if (aln_new_tag.size() >= aln_org_read.size()){
    PrintWarning("MergeReseqBams WARNING: Flow alignment for new tag too short in read ", alignment.Name, max_warnings);
    if ((max_warnings.at(2) == 1) and (max_warnings.at(0) <= max_warnings.at(1))){
      cerr << "prefix flow = " << prefix_flow << endl;
      cerr << "org zt tag: " << org_zt_tag << " : " << extra_bases << endl;
      cerr << "new zt tag: " << new_tag_value << endl;
      cerr << "new flow align: ";
      for (unsigned int flow = 0; flow < aln_new_tag.size(); ++flow){
        cerr << aln_new_tag.at(flow) << " ";
      }
      cerr << endl;
      cerr << "flow order    : ";
        for (unsigned int flow = prefix_flow; flow < prefix_flow+aln_org_read.size(); ++flow){
          cerr << flow_order.nuc_at(flow) << " ";
      }
      cerr << endl;
      cerr << "org flow align: ";
      for (unsigned int flow = 0; flow < aln_org_read.size(); ++flow){
         cerr << aln_org_read.at(flow) << " ";
      }
      cerr << endl;
      cerr << "------------------------------------------------------------------------" << endl;
    }
    return false;
  }

  double residual, total_residual = 0.0;
  double org_residual, total_org_residual = 0.0;

  // Exclude flow 0 since alignment does not reflect the incorporation of the handle
  for (int flow=1; flow<(int)aln_new_tag.size()-1; ++flow){
    residual = measurements.at(prefix_flow+flow) - (double)aln_new_tag.at(flow);
    org_residual = measurements.at(prefix_flow+flow) - (double)aln_org_read.at(flow);
    total_residual     += residual*residual;
    total_org_residual += org_residual*org_residual;
  }

  if ((total_residual-total_org_residual) > max_residual){
    PrintWarning("MergeReseqBams WARNING: Residual too large for read ", alignment.Name, max_warnings);
    if ((max_warnings.at(2) == 1) and (max_warnings.at(0) <= max_warnings.at(1))){
      cerr << "excess residual = " << total_residual << "-" << total_org_residual << " > " << max_residual << endl;
      cerr << " Run ID: " << getRunID(alignment.Name) << " ReadGroup: " << read_group << endl;
      cerr << "prefix flow = " << prefix_flow << " prefix bases: " << prefix_bases << endl;
      cerr << "Flow Order: " << flow_order.str() << endl;

      cerr << "org zt tag: " << org_zt_tag << " : " << extra_bases << endl;
      cerr << "new zt tag: " << new_tag_value << endl;

      cerr << "org flow align: ";
      for (unsigned int flow = 0; flow < aln_new_tag.size(); ++flow){
        cerr << aln_org_read.at(flow) << " ";
      }
      cerr << endl;
      cerr << "flow order    : ";
      for (unsigned int flow = prefix_flow; flow < (unsigned int)prefix_flow+aln_new_tag.size(); ++flow){
        cerr << flow_order.nuc_at(flow) << " ";
      }
      cerr << endl;
      cerr << "new flow align: ";
      for (unsigned int flow = 0; flow < aln_new_tag.size(); ++flow){
        cerr << aln_new_tag.at(flow) << " ";
      }
      cerr << endl;
      cerr << "measurements  : ";
      for (unsigned int flow = prefix_flow; flow < (unsigned int)prefix_flow+aln_new_tag.size(); ++flow){
        cerr << measurements.at(flow) << " ";
      }
      cerr << endl;
      cerr << "residual      : ";
      for (unsigned int flow = 0; flow < aln_new_tag.size()-1; ++flow){
        cerr << measurements.at(prefix_flow+flow) - (double)aln_new_tag.at(flow) << " ";
      }
      cerr << endl;
      cerr << "------------------------------------------------------------------------" << endl;
    }
    ++num_high_residual;
    return false;
  }
  //*/

  int aln_start_flow = aln_new_tag.size()-1;
  int query_base_correction = aln_org_read.at(aln_start_flow) - aln_new_tag.at(aln_start_flow);
  if (query_base_correction <= 0){
    ++aln_start_flow;
    while(aln_org_read.at(aln_start_flow) == 0)
      ++aln_start_flow;
    query_base_correction = aln_org_read.at(aln_start_flow);
  }
  for (int flow=aln_start_flow+1; flow<(int)aln_org_read.size(); ++flow){
    query_base_correction += aln_org_read.at(flow);
  }
  aln_start_flow += prefix_flow;
  query_base_correction -= num_extra_bases;

  bool query_update = query_base_correction != 0;
  string new_qualities;
  // recompute ze tag and query
  string new_ze_tag;
  if (query_base_correction < 0){
    // shift ze tag by query_base_correction bases
    if (ze_length > 0){
      // ze start -- last query_base_correction
      int ze_start   = min(-query_base_correction, ze_length);
      int num_from_ze    = ze_length - ze_start;
      int num_from_query = ze_length - num_from_ze;
      int query_start    =  -query_base_correction - num_from_query;

      new_ze_tag  = ze_tag.substr(ze_start, num_from_ze) +
                    read_bases.substr(query_start, num_from_query);
    }

    read_bases    = read_bases.substr(-query_base_correction, string::npos);
    new_qualities = alignment.Qualities.substr(-query_base_correction, string::npos);
  }
  else if (query_base_correction > 0){
    string extra_read_bases = org_zt_tag.substr(org_zt_tag.length()-query_base_correction, string::npos);
    if (ze_length > 0){
      string new_query_bases;
      new_ze_tag = org_zt_tag.substr(org_zt_tag.length()-query_base_correction, ze_length);
      if ((int)new_ze_tag.length() < ze_length){
        new_ze_tag += ze_tag.substr(0, ze_length-new_ze_tag.length());
        extra_read_bases = ze_tag.substr(ze_length-new_ze_tag.length(), string::npos);
      }
      else{
        extra_read_bases = org_zt_tag.substr(org_zt_tag.length()-query_base_correction+ze_length, string::npos) + ze_tag;
      }
    }

    read_bases    = extra_read_bases + read_bases;
    // need to create dummy quality string for this case
    string dummy_qv(query_base_correction, alignment.Qualities.at(0));
    new_qualities = dummy_qv + alignment.Qualities;
  }
  else {
    if (aln_start_flow != zf_flow){
      PrintWarning("MergeReseqBams WARNING: Flow Alignment does not match ZF:tag in read ", alignment.Name, max_warnings);
      return false;
    }
  }

  if (query_update and alignment.IsMapped()){
    PrintWarning("MergeReseqBams WARNING: Cannot update query for mapped read ", alignment.Name, max_warnings);
    return false;
  }

  // Step 3: Do a final start flow validation before update

  int new_start_flow = GetLastIncorporatingFlow(prefix_bases+new_tag_value+read_bases.substr(0,1), flow_order);
  int new_za_tag  = za_tag + query_base_correction;

  if (new_start_flow != aln_start_flow){
    PrintWarning("MergeReseqBams WARNING: ZF start flow violation in read ", alignment.Name, max_warnings);
    if ((max_warnings.at(2) == 1) and (max_warnings.at(0) <= max_warnings.at(1))){
      cerr << "prefix flow = " << prefix_flow  << " new_start_flow = " << new_start_flow << " zf_from_aln = " << aln_start_flow << endl;
      cerr << "query_base_correction = " << query_base_correction << " read_bases start with " << read_bases.substr(0,16) << endl;
      cerr << "org zt tag: " << org_zt_tag << endl;
      cerr << "new zt tag: " << new_tag_value << endl;

      cerr << "new flow align: ";
      for (unsigned int flow = 0; flow < aln_new_tag.size(); ++flow){
        cerr << aln_new_tag.at(flow) << " ";
      }
      cerr << endl;
      cerr << "flow order    : ";
        for (unsigned int flow = prefix_flow; flow < prefix_flow+aln_org_read.size(); ++flow){
          cerr << flow_order.nuc_at(flow) << " ";
      }
      cerr << endl;
      cerr << "org flow align: ";
      for (unsigned int flow = 0; flow < aln_org_read.size(); ++flow){
        cerr << aln_org_read.at(flow) << " ";
      }
      cerr << endl;
      cerr << "------------------------------------------------------------------------" << endl;
      //*/
    }
    return false;
  }

  // ---- Extra check for ZA

  //if (well_rowcol[0] == 1456 and well_rowcol[1] == 8262){
  if (za_tag > 0 and new_za_tag < (int)read_bases.length()) {
    cerr << "MergeReseqBams WARNING: ZA tag violation for read " << alignment.Name << endl;
    cerr << "prefix flow = " << prefix_flow  << " new_start_flow = " << new_start_flow << " zf_from_aln = " << aln_start_flow << endl;
    cerr << "query_base_correction = " << query_base_correction << " read_bases start with " << read_bases.substr(0,16) << endl;
    cerr << "org zt tag: " << org_zt_tag << endl;
    cerr << "new zt tag: " << new_tag_value << endl;
    cerr << "old za tag: " << za_tag << " new za tag: " << new_za_tag << endl;
    cerr << "Query length: " << read_bases.length() << endl;
    cerr << "------------------------------------------------------------------------" << endl;
    return false;
  }


  // Do the actual tag swap & update
  if (query_update){
    ++num_query_updates;
    alignment.EditTag("ZF", "i", new_start_flow);
    if (za_tag > 0){
      alignment.EditTag("ZA", "i", new_za_tag);
    }
    // No support for mapped reads, so no need to reverse complement
    //if (alignment.IsReverseStrand())
    //  RevComplementInPlace(read_bases);
    alignment.QueryBases = read_bases;
    alignment.Length     = read_bases.length();
    alignment.Qualities  = new_qualities;
    //*/
    //return false;
  }

  alignment.EditTag(tag_name, "Z", new_tag_value);
  return true;
}


// --------------------------------------------------------------------

bool ReconcileTags(
    const string& tag_name,
    BamTools::BamAlignment &alignment1,
    BamTools::BamAlignment &alignment2,
    vector<unsigned int>   &log_strict_match,
    vector<unsigned int>   &log_update,
    const string           &tag_structure,
    const FlowsAndKeys     &flows_and_keys,
    vector<unsigned int>   &max_warnings,
    unsigned int           &num_query_updates,
    unsigned int           &num_high_residual,
    double                  max_residual)
{

  // Compare and unify tags
  bool identical_tags = false;
  string tag1, tag2;
  bool have_tag1 = alignment1.GetTag(tag_name, tag1);
  bool have_tag2 = alignment2.GetTag(tag_name, tag2);
  bool struct_match1 = StrictStrctureMatch(tag1, tag_structure);
  bool struct_match2 = StrictStrctureMatch(tag2, tag_structure);

  // --- tag reconciliation
  int log_idx = 0;
  if (have_tag1){
    if (struct_match1)
      ++log_idx;

    if (not have_tag2){ // Case: Only have tag in Bam 1
      alignment2.AddTag(tag_name, "Z", tag1);
      ++log_update.at(3-2*log_idx);
      ++log_strict_match.at(log_idx);
    }
    else{ // Case: have tags in Bam 1&2

      if (struct_match2)
        log_idx+=2;
      ++log_strict_match.at(log_idx);
      if (tag1 != tag2){ // different tags
        if (struct_match2 and not struct_match1){
          if (UpdateTagInformation(
              tag_name,
              alignment1,
              tag2,
              flows_and_keys,
              max_warnings,
              num_query_updates,
              num_high_residual,
              max_residual))

            ++log_update.at(2);
          else
            ++log_update.at(5);
        }
        else {
          if (not UpdateTagInformation(
              tag_name,
              alignment2,
              tag1,
              flows_and_keys,
              max_warnings,
              num_query_updates,
              num_high_residual,
              max_residual))

            // Try reverse swap if this one fails
            ++log_update.at(5);
          else if (struct_match1){
            if (struct_match2)
              ++log_update.at(4);
            else
              ++log_update.at(1);
          }
          else
            ++log_update.at(3);
        }
      }
      else{
        identical_tags = true;
        ++log_update[0];
      }
    }
  }
  else if (have_tag2) { // Case: Only have tag in Bam 2
    if (struct_match2)
      ++log_strict_match.at(2);
    alignment1.AddTag(tag_name, "Z", tag2);
  }
  return identical_tags;
}

// --------------------------------------------------------------------


void DetectFlowOrderzAndKeyFromBam(const SamHeader &samHeader, FlowsAndKeys &flows_and_keys)

{
    // We only store flow orders that are different from each other in the flow order vector.
    // The same flow order but different total numbers of flow map to the  same flow order object
    // So multiple runs, even with different max flows, point to the same flow order object
    // We assume that the read group name is written in the form <run_id>.<Barcode Name>

    //flows_and_keys.flow_order_vector.clear();
    unsigned int num_fo = flows_and_keys.flow_order_vector.size();
    if (num_fo>0){
      cout<< "INFO: DetectFlowOrderzAndKeyFromBam called with " << num_fo << " flow orders already present." << endl;
    }
    // Init the temp vector so that the function can be called multiple times for different bam files & builds a joint map
    vector<string> temp_flow_order_vector(num_fo);
    for (unsigned int iFO=0; iFO < temp_flow_order_vector.size(); iFO++){
      temp_flow_order_vector.at(iFO) = flows_and_keys.flow_order_vector.at(iFO).str();
    }
    int num_read_groups = 0;

    for (BamTools::SamReadGroupConstIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {

      num_read_groups++;
      if (itr->ID.empty()){
        cerr << "MergeReseqBams ERROR: BAM file has a read group without ID." << endl;
        exit(EXIT_FAILURE);
      }
      // We need a flow order to do variant calling so throw an error if there is none.
      if (not itr->HasFlowOrder()) {
        cerr << "MergeReseqBams ERROR: read group " << itr->ID << " does not have a flow order." << endl;
        exit(EXIT_FAILURE);
      }

      // Check for duplicate read group ID entries and throw an error if one is found
      std::map<string,string>::const_iterator key_it = flows_and_keys.key_by_read_group.find(itr->ID);
      if (key_it != flows_and_keys.key_by_read_group.end()) {
        cerr << "MergeReseqBams ERROR: Multiple read group entries with ID " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }

      // Store Key Sequence for each read group
      // The read group key in the BAM file contains the full prefix: key sequence + barcode + barcode adapter
      flows_and_keys.key_by_read_group[itr->ID] = itr->KeySequence;

      // Get run id from read group name: convention <read group name> = <run_id>.<Barcode Name>
      string run_id = itr->ID.substr(0,itr->ID.find('.'));
      if (run_id.empty()) {
        cerr << "MergeReseqBams ERROR: Unable to extract run id from read group name " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }

      // Check whether an entry already exists for this run id and whether it is compatible
      std::map<string,int>::const_iterator fo_it = flows_and_keys.flow_order_index_by_run_id.find(run_id);
      if (fo_it != flows_and_keys.flow_order_index_by_run_id.end()) {
        // Flow order for this run id may be equal or a subset of the stored one
        if (temp_flow_order_vector.at(fo_it->second).length() < itr->FlowOrder.length()
            or temp_flow_order_vector.at(fo_it->second).substr(0, itr->FlowOrder.length()) != itr->FlowOrder
            or flows_and_keys.num_flows_by_run_id.at(run_id) != (int)(itr->FlowOrder).length())
        {
          cerr << "MergeReseqBams ERROR: Flow order information extracted from read group name " << itr->ID
               << " does not match previous entry for run id " << run_id << ": " << endl;
          cerr << "Exiting entry  : " << temp_flow_order_vector.at(fo_it->second) << endl;
          cerr << "Newly extracted: " << itr->FlowOrder << endl;
          cerr << temp_flow_order_vector.at(fo_it->second) << endl;
          exit(EXIT_FAILURE);
        }
        // Found matching entry and everything is OK.
        continue;
      }

      // New run id: Check whether this flow order is the same or a sub/ superset of an existing flow order
      unsigned int iFO = 0;
      for (; iFO< temp_flow_order_vector.size(); iFO++){
        if ( temp_flow_order_vector.at(iFO) == itr->FlowOrder){
          flows_and_keys.flow_order_index_by_run_id[run_id] = iFO;
          flows_and_keys.num_flows_by_run_id[run_id] = itr->FlowOrder.length();
            break;
        }
      }

      // Do we have a new flow order?
      if (iFO == temp_flow_order_vector.size()) {
        temp_flow_order_vector.push_back(itr->FlowOrder);
        flows_and_keys.flow_order_index_by_run_id[run_id] = iFO;
        flows_and_keys.num_flows_by_run_id[run_id] = itr->FlowOrder.length();
      }

    } // --- end loop over read groups

    // Now we have amassed all the unique flow orders and can construct the FlowOrder objects
    for (unsigned int iFO=num_fo; iFO < temp_flow_order_vector.size(); iFO++){
      ion::FlowOrder tempIonFlowOrder(temp_flow_order_vector.at(iFO), temp_flow_order_vector.at(iFO).length());
      flows_and_keys.flow_order_vector.push_back(tempIonFlowOrder);
    }

    // Verbose output
    cout << "DetectFlowOrderzAndKeyFromBam found a total of " << flows_and_keys.flow_order_vector.size() << " different flow orders of max flow lengths: ";
    int iFO=0;
    for (; iFO<(int)flows_and_keys.flow_order_vector.size()-1; iFO++)
      cout << flows_and_keys.flow_order_vector.at(iFO).num_flows() << ',';
    cout << flows_and_keys.flow_order_vector.at(iFO).num_flows();
    cout << " from parsing " << flows_and_keys.key_by_read_group.size()<< " read groups." << endl;

}


// =========================================================================
// XXX

int main (int argc, const char *argv[])
{
  time_t program_start_time;
  time(&program_start_time);
  Json::Value log_json(Json::objectValue);
  DumpStartingStateOfProgram (argc,argv,program_start_time, log_json["MergeReseqBams"]);

  //
  // Step 1. Process command line options
  //
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  vector<unsigned int> max_warnings(3,0);
  string   log_filename = opts.GetFirstString      ('l', "logfile",      "");
  vector<string> bams   = opts.GetFirstStringVector('i', "input-bam",    "");
  string   output_bam   = opts.GetFirstString      ('o', "output-bam",   "");
  int      num_threads  = opts.GetFirstInt         ('t', "num-threads",  12);
  double   max_residual = opts.GetFirstDouble      ('-', "max-residual", 2.0);
  max_warnings.at(1)    = opts.GetFirstInt         ('-', "max-warnings", 50);
  string prefix_mol_tag = opts.GetFirstString  ('-', "prefix-mol-tag", "TNNNACTNNNTGAT");
  string suffix_mol_tag = opts.GetFirstString  ('-', "suffix-mol-tag", "ATCANNNAGTNNNA");

  if (opts.GetFirstBoolean     ('v', "verbose",   false))
    max_warnings.at(2) = 1;

  // Check arguments
  if (argc < 3)
    return PrintHelp();
  cout << endl;
  if (bams.size() != 2){
    cerr << "MergeReseqBams ERROR: Need 2 bam files to merge." << endl;
    return PrintHelp();
  }
  if (output_bam == ""){
    cerr << "MergeReseqBams ERROR: Need output bam file." << endl;
    return PrintHelp();
  }

  // Command line parsing officially over. Detect unknown options.
      opts.CheckNoLeftovers();

  // ----- Load input bam files
  BamMultiReader     bam_multi_reader;
  SamHeader          merged_header;
  BamReader          bamReader1, bamReader2;
  vector<SamHeader>  headers;

  // Accounting structures for flow information
  FlowsAndKeys flows_and_keys;

  if(!bamReader1.Open(bams.at(0))) {
    cerr << "MergeReseqBams ERROR: failed to open bam file " << bams.at(0) << endl;
    return 1;
  }
  if(!bamReader2.Open(bams.at(1))) {
    bamReader1.Close();
    cerr << "MergeReseqBams ERROR: failed to open bam file " << bams.at(1) << endl;
    return 1;
  }
  headers.push_back(bamReader1.GetHeader());
  headers.push_back(bamReader2.GetHeader());
  bamReader1.Close();
  bamReader2.Close();

  // Use BamMultiReader to merge header
  if (not bam_multi_reader.SetExplicitMergeOrder(BamMultiReader::MergeByName)) {
    cerr << "MergeReseqBams ERROR: Could not set merge order to BamMultiReader::MergeByName" << endl;
    exit(1);
  }
  if (not bam_multi_reader.Open(bams)) {
    cerr << "MergeReseqBams ERROR: Could not open input BAM file(s) : " << bam_multi_reader.GetErrorString() << endl;
    exit(1);
  }
  merged_header = bam_multi_reader.GetHeader();
  bam_multi_reader.Close();
  // Extract flow order and key information for this bam
  DetectFlowOrderzAndKeyFromBam(merged_header, flows_and_keys);

  // Manually merge comment sections of BAM files
  unsigned int num_duplicates = 0;
  unsigned int num_merged = 0;

  for (unsigned int bam_idx = 0; bam_idx < bams.size(); bam_idx++) {
    // Manually merge comment section
    log_json["input-bam"][bam_idx] = bams.at(bam_idx);
    for (unsigned int i_co = 0; i_co < headers.at(bam_idx).Comments.size(); i_co++) {

      // Step 1: Check if this comment is already part of the merged header
      unsigned int m_co = 0;
      while (m_co < merged_header.Comments.size() and merged_header.Comments.at(m_co) != headers.at(bam_idx).Comments.at(i_co))
        m_co++;

      if (m_co < merged_header.Comments.size()){
        num_duplicates++;
        continue;
      }

      // Add comment line to merged header if it is a new one
      num_merged++;
      merged_header.Comments.push_back(headers.at(bam_idx).Comments.at(i_co));
    }
  }

  // Verbose what we did
  //cout << "Merged " << num_merged << " unique comment lines into combined BAM header. Encountered " << num_duplicates << " duplicate comments." << endl;
  // reopen input bams
  if(!bamReader1.Open(bams.at(0))) {
    cerr << "MergeReseqBams ERROR: failed to open bam file " << bams.at(0) << endl;
    return 1;
  }
  if(!bamReader2.Open(bams.at(1))) {
    bamReader1.Close();
    cerr << "MergeReseqBams ERROR: failed to open bam file " << bams.at(1) << endl;
    return 1;
  }

  // ---- setup output BAM IO
  BamTools::BamWriter writer;
  BamTools::RefVector refs   = bamReader1.GetReferenceData();
  bool write_output_bam = output_bam.length() > 0;

  writer.SetNumThreads(num_threads);
  writer.SetCompressionMode(BamWriter::Compressed);
  if (!writer.Open(output_bam, merged_header, refs)) {
    cerr << "CompareIonBam ERROR: failed to open output bam file " << output_bam << endl;
    return 1;
  }

  cout << "Merging BAM 1: " << bams.at(0) << endl;
  cout << "        BAM 2: " << bams.at(1) << endl;
  log_json["MergeReseqBams"]["output_bam"] = output_bam;
  log_json["BAM_1"]["filename"] = bams.at(0);
  log_json["BAM_2"]["filename"] = bams.at(1);

  // Molecular tag structures
  cout << "Using molecular tag structure:" << endl;
  cout << "Prefix: " << prefix_mol_tag << endl;
  cout << "Suffix: " << suffix_mol_tag << endl;
  log_json["MergeReseqBams"]["prefix_tag"] = prefix_mol_tag;
  log_json["MergeReseqBams"]["suffix_tag"] = suffix_mol_tag;
  log_json["MergeReseqBams"]["max_residual"] = max_residual;

  // Setup counters and variables for read comparison
  BamAlignment alignment1, alignment2;
  bool get_next_alignment1, get_next_alignment2, valid_alignment1, valid_alignment2;
  get_next_alignment1 = get_next_alignment2 = true;
  valid_alignment1    = valid_alignment2    = false;

  // Accounting variables
  unsigned int num_reads1, num_reads2, reads_only1, reads_only2, reads_both;
  unsigned int identical_tags, num_query_updates, num_high_residual;
  vector<unsigned int> strict_match_zt(4,0), strict_match_yt(4,0);
  vector<unsigned int> strict_match_b1(2,0), strict_match_b2(2,0);
  vector<unsigned int> zt_update(6,0), yt_update(6,0);
  int          compare_wells;

  // Init variables
  num_reads1 = num_reads2 = reads_only1 = reads_only2 = reads_both = 0;
  identical_tags = num_query_updates = num_high_residual =0;

  // ------- Read comparison loop -------

  while(1) {

    // Step 1: Get reads with same well coordinates to compare against each other
    while(get_next_alignment1 or get_next_alignment2) {
      if (get_next_alignment1) {
        valid_alignment1 = bamReader1.GetNextAlignment(alignment1);
        if (valid_alignment1)
          num_reads1++;
      }
      if (get_next_alignment2) {
        valid_alignment2 = bamReader2.GetNextAlignment(alignment2);
        if (valid_alignment2)
          num_reads2++;
      }

      if (valid_alignment1 and valid_alignment2) { // Yay, have 2 alignments, let's compare them!
        compare_wells = compareWellCoordinates(alignment1.Name, alignment2.Name);
        if (compare_wells == 0) {  // Coordinates agree, let's compare reads
          get_next_alignment1 = false;
          get_next_alignment2 = false;
        } else if (compare_wells > 0) {
          if (compare_wells == 1) { // Read from Bam 1 not in Bam 2
            RecordTagStructureMatch(strict_match_b1, alignment1, prefix_mol_tag, suffix_mol_tag);
            writer.SaveAlignment(alignment1);
            reads_only1++;
            get_next_alignment1 = true;
            get_next_alignment2 = false;
          } else { // Read from Bam 2 not in Bam 1
            RecordTagStructureMatch(strict_match_b2, alignment2, prefix_mol_tag, suffix_mol_tag);
            writer.SaveAlignment(alignment2);
            reads_only2++;
            get_next_alignment1 = false;
            get_next_alignment2 = true;
          }
        } else { // An error occurred while parsing the read names
          return 1;
        }
      }
      else if (not(valid_alignment1 or valid_alignment2)) { // Done here, both files have finished.
        get_next_alignment1 = false;
        get_next_alignment2 = false;
      }
      else { // Reached end of one file but not other.
        get_next_alignment1 = valid_alignment1;
        get_next_alignment2 = valid_alignment2;
        if (valid_alignment1) {
          reads_only1++;
          RecordTagStructureMatch(strict_match_b1, alignment1, prefix_mol_tag, suffix_mol_tag);
          writer.SaveAlignment(alignment1);
        }
        if (valid_alignment2) {
          reads_only2++;
          RecordTagStructureMatch(strict_match_b2, alignment2, prefix_mol_tag, suffix_mol_tag);
          writer.SaveAlignment(alignment2);
        }
      }
    }
    // Exit loop if we're done reading the files
    if (not(valid_alignment1 or valid_alignment2))
      break;

    // Compare and unify tags
    ++reads_both;
    RecordTagStructureMatch(strict_match_b1, alignment1, prefix_mol_tag, suffix_mol_tag);
    RecordTagStructureMatch(strict_match_b2, alignment2, prefix_mol_tag, suffix_mol_tag);

    bool ident_zt = ReconcileTags(
        "ZT",
        alignment1,
        alignment2,
        strict_match_zt,
        zt_update,
        prefix_mol_tag,
        flows_and_keys,
        max_warnings,
        num_query_updates,
        num_high_residual,
        max_residual);

    bool ident_yt = ReconcileTags(
        "YT",
        alignment1,
        alignment2,
        strict_match_yt,
        yt_update,
        suffix_mol_tag,
        flows_and_keys,
        max_warnings,
        num_query_updates,
        num_high_residual,
        max_residual);

    if (ident_zt and ident_yt)
      ++identical_tags;

    // Write output reads
    writer.SaveAlignment(alignment1);
    writer.SaveAlignment(alignment2);

    get_next_alignment1 = true;
    get_next_alignment2 = true;
  }
  // ----- End of read comparison loop -----

  bamReader1.Close();
  bamReader2.Close();
  if (writer.IsOpen())
    writer.Close();

  // Create json log
  log_json["BAM_1"]["num_reads"] = num_reads1;
  log_json["BAM_1"]["unique_reads"] = reads_only1;
  for (unsigned int m=0; m<strict_match_b1.size(); ++m){
    log_json["BAM_1"]["structure_match"][m] = strict_match_b1.at(m);
  }
  log_json["BAM_2"]["num_reads"] = num_reads2;
  log_json["BAM_2"]["unique_reads"] = reads_only2;
  for (unsigned int m=0; m<strict_match_b2.size(); ++m){
    log_json["BAM_2"]["structure_match"][m] = strict_match_b2.at(m);
  }

  log_json["Comparison"]["num_reads"] = reads_both;
  log_json["Comparison"]["identical_tags"] = identical_tags;
  log_json["Comparison"]["update"][0] = "identical tags";
  log_json["Comparison"]["update"][1] = "Update to tag in Bam1 (Bam1 matching structure)";
  log_json["Comparison"]["update"][2] = "Update to tag in Bam2 (Bam2 matching structure)";
  log_json["Comparison"]["update"][3] = "Update to tag in Bam1 (NEITHER matching structure)";
  log_json["Comparison"]["update"][4] = "Update to tag in Bam1 (BOTH matching structure)";
  log_json["Comparison"]["update"][5] = "Tag update failed!";
  for (unsigned int m=0; m<zt_update.size(); ++m)
    log_json["Comparison"]["zt_update"][m] = zt_update.at(m);
  for (unsigned int m=0; m<yt_update.size(); ++m)
    log_json["Comparison"]["yt_update"][m] = yt_update.at(m);
  log_json["Comparison"]["strict match"][0] = "none";
  log_json["Comparison"]["strict match"][1] = "tag from bam1 only";
  log_json["Comparison"]["strict match"][2] = "tag from bam2 only";
  log_json["Comparison"]["strict match"][3] = "tags from both bams";
  for (unsigned int m=0; m<strict_match_zt.size(); ++m)
    log_json["Comparison"]["strict_match_zt"][m] = strict_match_zt.at(m);
  for (unsigned int m=0; m<strict_match_yt.size(); ++m)
    log_json["Comparison"]["strict_match_yt"][m] = strict_match_yt.at(m);
  log_json["Comparison"]["num_query_updates"] = num_query_updates;
  log_json["Comparison"]["num_high_residual"] = num_high_residual;

  time_t program_end_time;
  time(&program_end_time);
  int duration = (int)difftime(program_end_time,program_start_time);
  log_json["MergeReseqBams"]["end_time"] = get_time_iso_string(program_end_time);
  log_json["MergeReseqBams"]["total_duration"] = duration;
  SaveJson(log_json, log_filename);

  // Print summary
  cout << " --- MergeReseqBam Summary --- " << endl;
  cout << "         Processing time: " << duration << " seconds." << endl;
  cout << "                         " << setw(15) << "BAM 1"            << " :" << setw(15) << "BAM 2"            << endl;
  cout << "            Total Reads :" << setw(15) << num_reads1         << " :" << setw(15) << num_reads2         << endl;
  cout << "     Structure Match ZT :" << setw(15) << strict_match_b1[0] << " :" << setw(15) << strict_match_b2[0] << endl;
  cout << "     Structure Match YT :" << setw(15) << strict_match_b1[1] << " :" << setw(15) << strict_match_b2[1] << endl;
  cout << "           Unique Reads :" << setw(15) << reads_only1        << " :" << setw(15) << reads_only2        << endl;
  cout << "         ZT Tag Updates :" << setw(15) << zt_update[2]       << " :" << setw(15) << zt_update[1] +zt_update[3]+zt_update[4] << endl;
  cout << "         YT Tag Updates :" << setw(15) << yt_update[2]       << " :" << setw(15) << yt_update[1] +yt_update[3]+zt_update[4] << endl;
  cout << " ------------------------" << endl;
  cout << "Num. Reads found in both:" << setw(15) << reads_both   << endl;
  cout << "      Identical ZT tags :" << setw(15) << zt_update[0] << endl;
  cout << "      Identical YT tags :" << setw(15) << yt_update[0] << endl;
  cout << "   Identical ZT&YT tags :" << setw(15) << identical_tags << endl;
  cout << " ------------------------" << endl;

  return 0;
}
