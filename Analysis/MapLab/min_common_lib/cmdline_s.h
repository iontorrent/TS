/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __cmdline_s__h__
#define __cmdline_s__h__

#include <string>
#include <vector>
#include <map>

typedef std::vector<std::string> Arglist;
typedef std::map<std::string, Arglist> Optdict;

void get_opt (int argc,  const char* const* argv, const char* optdef, Arglist& arglist, Optdict& optdict, const char* const* longopts = NULL);
void parse_options (int argc, const char* const* argv, std::string& progname, Arglist& arglist, Optdict& optdict, const char* optdef = "s:h", const char* const * longopts = NULL);


#endif
