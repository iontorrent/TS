/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FILEBKGREPLAY_H
#define FILEBKGREPLAY_H

#include "hdf5.h"
#include <string>
#include <map>

#define EMPTYTRACE "emptyTrace"
#define PINNEDINFLOW "pinnedInFlow"
#define PINSPERFLOW "pinsPerFlow"
#define REGIONTRACKER_REGIONPARAMS "region_param"
#define REGIONTRACKER_MISSINGMATTER "missing_matter"
#define FLOWINDEX "flowIndex"

// layout for the background model replay file
struct dsn {
  dsn(char* _dsn, char* _desc, hid_t _class, unsigned int _rank);
  dsn ();
  char *dataSetName;
  char *description;
  hid_t dsnType;
  unsigned int rank;
};

class fileReplayDsn{
 public:
  fileReplayDsn();
  ~fileReplayDsn();

  dsn GetDsn(std::string& key);

 private:
  std::map<std::string, dsn> mMap;
};

#endif // FILEBKGREPLAY_H
