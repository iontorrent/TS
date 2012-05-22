/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include "FileBkgReplay.h"

dsn::dsn(char* _dsn, char* _desc, hid_t _class, unsigned int _rank)
  : dataSetName(_dsn), description(_desc), dsnType(_class), rank(_rank)
{
}

dsn::dsn() : dataSetName(0), description(0), dsnType(-1), rank(0) {}

// structure of the HDF5 dataset names in the replay file
// you can instantiate at any time, read only
// if you want to change the structure of the replay file, do it here
fileReplayDsn::fileReplayDsn()
{
  // dataset name, description, rank
  mMap["version"] = 
    dsn("/info/version",
	"version information for this file",
	H5T_NATIVE_CHAR, 1);
  mMap[std::string("processparameters")] = 
    dsn("/info/processparameters",
	"command that output this file",
	H5T_NATIVE_CHAR, 1);
  // add other types of data later..
  mMap[std::string("fg_bead_DC")] =
    dsn("/bead/fg_bead_DC",
	"bead_DC per bead: chip x by chip y by flow",
	H5T_NATIVE_FLOAT, 3);
  mMap[std::string("amp_multiplier")] = 
    dsn("/bead/amp_multiplier",
	"Amplitude-multiplier per bead: chip x by chip y by flow",
	H5T_NATIVE_FLOAT, 3);
  mMap[std::string("k_rate_multiplier")] = 
    dsn("/bead/k_rate_multiplier",
	"K-rate-multiplier per bead: chip x by chip y by flow",
	H5T_NATIVE_FLOAT, 3);
  mMap[std::string("res_err")] = 
    dsn("/bead/res_error",
	"Residual-error per bead: chip x by chip y by flow",
	H5T_NATIVE_FLOAT, 3);
  mMap[std::string("bead_init_param")] =
    dsn("/bead/bead_init_param",
	"Bead-init-param per bead: chip x by chip y by [Copies, R, dH5Replmult, gain]",
	H5T_NATIVE_FLOAT, 3);
  mMap[std::string("compute_regions")] =
    dsn("/region/compute_regions",
	"region by [index; chip offset x; y; width; height]",
	H5T_NATIVE_INT, 2);
  mMap[PINNEDINFLOW] =
    dsn("/image/pinnedInFlow",
	"Per well: when a well is first pinned in a flow, record the flow.  -1 is never pinned",
	H5T_NATIVE_SHORT, 2);
  mMap[PINSPERFLOW] =
    dsn("/image/pinsPerFlow",
	"Per Flow: total number of pins per flow",
	H5T_NATIVE_INT, 1);
  mMap[REGIONTRACKER_REGIONPARAMS] = 
    dsn("/RegionTracker/region_param",
	"regionindex x block of flows x region_param as a struct of floats",
	H5T_NATIVE_FLOAT, 3);
  mMap[REGIONTRACKER_MISSINGMATTER] = 
    dsn("/RegionTracker/missing_matter",
	"regionindex x block of flows x missing_matter as a struct of floats",
	H5T_NATIVE_FLOAT, 3);
  mMap[FLOWINDEX] = 
    dsn("/flowIndex",
	"block id and within-block flow indexed by flow",
	H5T_NATIVE_INT, 2);
  // regions for emptyTrace currently match computational regions
  mMap[EMPTYTRACE] = 
    dsn("/region/emptyTrace",
	"Empty-trace per region: regionindex by frame by flow",
	H5T_NATIVE_FLOAT, 3);
  mMap[std::string("bg_bead_DC")] =
    dsn("/region/bg_bead_DC",
	"bg_DC per region by regionindex by flow",
	H5T_NATIVE_FLOAT, 2);
 mMap[std::string("")] = 
   dsn();
}

fileReplayDsn::~fileReplayDsn()
{
  mMap.clear();
}

dsn fileReplayDsn::GetDsn(std::string& key)
{
  // returns the dsn associated with key
  // if there is no key, throws an error
  std::map<std::string, dsn>::iterator it = mMap.find(key);
  
  // if (it == mMap.end())
  //   it = mMap.find("");
  assert( it != mMap.end() );
  
  return (it->second );  
}
