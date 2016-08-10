/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#include <iterator>
#include "parameters_section.h"

bool ParametersSection::writeHelp (std::ostream& ostr)
{
    ostr << "    Section [" << name_.c_str () << "] : " << description_.c_str () << std::endl;
    for (parmap::iterator itr = parameters_.begin (); itr != parameters_.end (); itr ++)
        ostr << "      " << itr->second.name_.c_str () << " : type = " << itr->second.type_.c_str () << ", default = " << itr->second.def_value_.c_str () << " : " << itr->second.description_.c_str () << std::endl;
    return true;
}

bool ParametersSection::hasParameter (const char* name)
{
    parmap::iterator itr = parameters_.find (name);
    if (itr == parameters_.end ()) return false;
    else return true;
}

