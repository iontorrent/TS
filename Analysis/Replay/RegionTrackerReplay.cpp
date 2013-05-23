/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "RegionTrackerReplay.h"

#include <vector>
#include <assert.h>
#include <pthread.h>
#include "FileBkgReplay.h"

using namespace std;

// *********************************************************************
// reader specific functions

RegionTrackerReader::RegionTrackerReader(CommandLineOpts &_clo, int _regionindex)
  : clo(_clo), regionindex(_regionindex)
{
  // specify the file and section to read from
  fileReplayDsn dsninfo;
  std::string dumpfile = clo.sys_context.analysisLocation + "dump.h5";
  std::string key_rp(REGIONTRACKER_REGIONPARAMS);
  dsn info_rp = dsninfo.GetDsn(key_rp);
  reader_rp = new H5ReplayReader(dumpfile, info_rp.dataSetName);
  std::string key_mm(REGIONTRACKER_MISSINGMATTER);
  dsn info_mm = dsninfo.GetDsn(key_mm);
  reader_mm = new H5ReplayReader(dumpfile, info_mm.dataSetName);

  reg_params_length = sizeof(reg_params_H5)/sizeof(float);
  numFlows = clo.flow_context.numTotalFlows;
  flowToIndex.resize(numFlows);
  flowToBlockId.resize(numFlows);

  ReadFlowIndex();
}

RegionTrackerReader::~RegionTrackerReader()
{
  delete reader_rp;
  delete reader_mm;
}


void RegionTrackerReader::Read(int flow)
{
  // region tracker has two components
  // struct reg_params rp and class Halo missing_mass, both all floats

  ReadRegParams(flowToBlockId[flow]);
  ReadMissingMass(flowToBlockId[flow]);
}

void RegionTrackerReader::ReadFlowIndex()
{
  // side-effect is to initialize flowToIndex with hdf5 contents

  // hyperslab layout in file
  // 1 x numFlows
  // start position [0, 0] for flowToIndex
  vector<hsize_t> offset(2);
  offset[0] = 0;
  offset[1] = 0;

  // read out 1 x numFlows
  vector<hsize_t> count(2);
  count[0] = 1;
  count[1] = numFlows;

  // memory layout for each slab we read in 1 x numFlows
  // start position in memory slab is [0]
  vector<hsize_t> offset_out(1);
  offset_out[0] = 0;

  // read in 1 x numFlows
  vector<hsize_t> count_out(1);
  count_out[0] = numFlows;

  fileReplayDsn dsninfo;
  std::string dumpfile = clo.sys_context.analysisLocation + "dump.h5";
  std::string key(FLOWINDEX);
  dsn info = dsninfo.GetDsn(key);
  H5ReplayReader reader = H5ReplayReader(dumpfile, info.dataSetName);
  reader.Read(offset, count,  offset_out, count_out, &flowToIndex[0]);
  fprintf(stdout, "H5 RegionTrackerReader flowToIndex:  region %d: %d, %d, ...\n", regionindex, flowToIndex[0], flowToIndex[1]);

  // start position [1, 0] for flowToBlockId
  offset[0] = 1;
  offset[1] = 0;

  // count and memory layout is the same as for flowToIndex

  reader.Read(offset, count,  offset_out, count_out, &flowToBlockId[0]);
  fprintf(stdout, "H5 RegionTrackerReader flowToBlockId:  region %d: %d, %d, ...\n", regionindex, flowToBlockId[0], flowToBlockId[1]);

}

void RegionTrackerReader::ReadRegParams(int block_id)
{
  // hyperslab layout in file
  // blocks indexed by flows
  // start position [regionindex, block_id, 0]
  vector<hsize_t> offset(3);
  offset[0] = regionindex;
  offset[1] = block_id;
  offset[2] = 0;

  // read out 1 x 1 x reg_params_length
  vector<hsize_t> count(3);
  count[0] = 1;
  count[1] = 1;
  count[2] = reg_params_length;

  // memory layout for each slab we read in 1 x reg_params_length
  // start position in memory slab is [0]
  vector<hsize_t> offset_out(1);
  offset_out[0] = 0;

  // read in 1 x reg_params_length
  vector<hsize_t> count_out(1);
  count_out[0] = reg_params_length;

  reg_params_H5 rp5;
  float *out = (float *)(&rp5);

  reader_rp->Read(offset, count,  offset_out, count_out, out);
  fprintf(stdout, "H5 RegionTrackerReader: region_params: block_id %d, region %d: %f, %f, ...\n", block_id, regionindex, out[0], out[1]);
  reg_params_H5_copyTo_reg_params(rp5, rp);
}

void RegionTrackerReader::ReadMissingMass(int block_id)
{
  int nuc_flow_t = missing_mass.nuc_flow_t;
  assert(nuc_flow_t <= MAX_COMPRESSED_FRAMES*NUMNUC);

  // hyperslab layout in file
  // regions x block_id x nuc_flow_t
  // start position [regionindex, block_id, 0]
  vector<hsize_t> offset(3);
  offset[0] = regionindex;
  offset[1] = block_id;
  offset[2] = 0;

  // read out 1 x 1 x nuc_flow_t
  vector<hsize_t> count(3);
  count[0] = 1;
  count[1] = 1;
  count[2] = nuc_flow_t;

  // memory layout for each slab we read in 1 x nuc_flow_t
  // start position in memory slab is [0]
  vector<hsize_t> offset_out(1);
  offset_out[0] = 0;

  // read in 1 x  nuc_flow_t
  vector<hsize_t> count_out(1);
  count_out[0] = nuc_flow_t;

  float *out = &missing_mass.dark_matter_compensator[0];

  reader_mm->Read(offset, count,  offset_out, count_out, out);
  fprintf(stdout, "H5 RegionTrackerReader dark_matter: block_id %d, region %d: %f, %f, ...\n", block_id, regionindex, out[0], out[1]);
}

// *********************************************************************
// recorder specific functions

RegionTrackerRecorder::RegionTrackerRecorder(CommandLineOpts &_clo, int _regionindex)
  : clo(_clo), regionindex(_regionindex)
{
  // specify the file and section to record to
  fileReplayDsn dsninfo;
  std::string dumpfile = clo.sys_context.analysisLocation + "dump.h5";
  std::string key_rp(REGIONTRACKER_REGIONPARAMS);
  dsn info_rp = dsninfo.GetDsn(key_rp);
  recorder_rp = new H5ReplayRecorder(dumpfile, info_rp.dataSetName, info_rp.dsnType, info_rp.rank);
  std::string key_mm(REGIONTRACKER_MISSINGMATTER);
  dsn info_mm = dsninfo.GetDsn(key_mm);
  recorder_mm = new H5ReplayRecorder(dumpfile, info_mm.dataSetName, info_mm.dsnType, info_mm.rank);

  reg_params_length = sizeof(reg_params_H5)/sizeof(float);
  numFlows = clo.flow_context.numTotalFlows;
  flowToIndex.resize(numFlows);
  flowToBlockId.resize(numFlows);

  Init();
}

RegionTrackerRecorder::~RegionTrackerRecorder()
{
  delete recorder_rp;
  delete recorder_mm;
}

void RegionTrackerRecorder::Init()
{
  int nFlowBlks = ceil(float(numFlows)/NUMFB);

  // Create reg_params dataset
  vector<hsize_t> rp_dims(3);
  rp_dims[0] = 1;
  rp_dims[1] = nFlowBlks;
  rp_dims[2] = sizeof(reg_params_H5)/sizeof(float);
  recorder_rp->CreateDataset(rp_dims);

  // Create missing_matter dataset
  vector<hsize_t> mm_dims(3);
  mm_dims[0] = 1;
  mm_dims[1] = nFlowBlks;
  mm_dims[2] = MAX_COMPRESSED_FRAMES * NUMNUC;
  recorder_mm->CreateDataset(mm_dims);

  // Create and write flow buffer information to the hdf5 file
  WriteFlowInfo();
}


void RegionTrackerRecorder::Write(int flow)
{
  int block_id = flowToBlockId[flow];
  WriteRegParams(block_id);
  WriteMissingMass(block_id);
}

void RegionTrackerRecorder::WriteRegParams(int block_id)
{
  // hyperslab layout on disk
  // regions x flow blocks x each reg_param
  // start position [regionindex, block, 0]
  vector<hsize_t> offset(3);
  offset[0] = regionindex;
  offset[1] = block_id;
  offset[2] = 0;
  // write out a block x reg_param
  vector<hsize_t> count(3);
  count[0] = 1;
  count[1] = 1;
  count[2] = reg_params_length;

  // memory layout of slab we write from is just a reg_param
  // start position in memory slab is [0]
  vector<hsize_t> offset_in(1);
  offset_in[0] = 0;

  // write out the reg_param
  vector<hsize_t> count_in(1);
  count_in[0] = reg_params_length;

  vector<hsize_t> extension(3);
  extension[0] = regionindex+1;
  extension[1] = block_id+1;
  extension[2] = reg_params_length;

  reg_params_H5 rp5;
  reg_params_copyTo_reg_params_H5(rp, rp5);
  float *in = (float *)(&rp5);
  recorder_rp->ExtendDataSet(extension); // extend if necessary
  fprintf(stdout, "H5 RegionTrackerRecorder: region_params: block_id %d, region %d: %f, %f, ...\n", block_id, regionindex, in[0], in[1]);
  recorder_rp->Write(offset, count, offset_in, count_in, in);
}

void RegionTrackerRecorder::WriteMissingMass(int block_id)
{
  int nuc_flow_t = missing_mass.nuc_flow_t;
  assert(nuc_flow_t <= MAX_COMPRESSED_FRAMES*NUMNUC);

  // hyperslab layout on disk
  // regions x flow blocks x dark_matter_compensator data
  // start position [regionindex, block, 0]
  vector<hsize_t> offset(3);
  offset[0] = regionindex;
  offset[1] = block_id;
  offset[2] = 0;
  // write out a block x length(dark_matter_compensator)
  vector<hsize_t> count(3);
  count[0] = 1;
  count[1] = 1;
  count[2] = nuc_flow_t;

  // memory layout of slab we write from is just a dark_matter_compensator
  // start position in memory slab is [0]
  vector<hsize_t> offset_in(1);
  offset_in[0] = 0;

  vector<hsize_t> count_in(1);
  count_in[0] = nuc_flow_t;

  vector<hsize_t> extension(3);
  extension[0] = regionindex+1;
  extension[1] = block_id+1;
  extension[2] = nuc_flow_t;

  recorder_mm->ExtendDataSet(extension); // extend if necessary

  float *in = &missing_mass.dark_matter_compensator[0];
  fprintf(stdout, "H5 RegionTrackerRecorder dark_matter: block_id %d, region %d: %f, %f, ...\n", block_id, regionindex, in[0], in[1]);
  recorder_mm->Write(offset, count, offset_in, count_in, in);
}


void  RegionTrackerRecorder::WriteFlowInfo()
{
  vector<hsize_t> flowindex_dims(2);
  flowindex_dims[0] = 2;
  flowindex_dims[1] = numFlows;
  
  // where should this really be handled??
  vector<int> flowBlockIndex(2*numFlows);
  for (int i=0; i<numFlows; i++){
    int j = i+numFlows;
    flowBlockIndex[i] = i % NUMFB;  // within block index indexed by flow
    flowBlockIndex[j] = i/NUMFB;    // block id indexed by flow
  }

  for (int i=0; i<numFlows; i++){
    flowToIndex[i] = flowBlockIndex[i];
    flowToBlockId[i] = flowBlockIndex[i+numFlows];
  }

  fileReplayDsn dsninfo;
  std::string dumpfile = clo.sys_context.analysisLocation + "dump.h5";
  std::string key(FLOWINDEX);
  dsn info = dsninfo.GetDsn(key);
  H5ReplayRecorder recorder = H5ReplayRecorder(dumpfile, info.dataSetName, info.dsnType, info.rank);
  recorder.CreateDataset(flowindex_dims);

  // hyperslab layout on disk
  // 2 x numFlows
  // 0, 1, 2, ... NUMFB-1, 0, 1, 2, ...
  // 0, 0, 0, ...    0,    1, 1, 1, ... 
  // start position [0,0]
  vector<hsize_t> offset(2);
  offset[0] = 0;
  offset[1] = 0;

  vector<hsize_t> count(2);
  count[0] = 2;
  count[1] = numFlows;

  // memory layout of slab we write from is 2 x numFlows
  // start position in memory slab is [0,0]
  vector<hsize_t> offset_in(2);
  offset_in[0] = 0;
  offset_in[1] = 0;

  vector<hsize_t> count_in(2);
  count_in[0] = 2;
  count_in[1] = numFlows;

  // fprintf(stdout, "H5 buffer, flow %d, region %d: %f, %f, ...\n", flow, regionIndex, newbuff[0], newbuff[1]);
  recorder.Write(offset, count, offset_in, count_in, &flowBlockIndex[0]);
}
