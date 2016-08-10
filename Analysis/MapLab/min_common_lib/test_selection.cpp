/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#include "test_selection.h"
#include "test_facet.h"
#include <ostream>
#include <cstring>
#include <iomanip>

bool TestSelection::leaf () const
{
    return branches_.empty ();
}
bool TestSelection::includes (const char* next_level, const TestSelection*& subpath) const
{
    subpath = NULL;
    if (leaf ())
        return true;
    for (SelVec::const_iterator itr = branches_.begin (), sent = branches_.end (); itr != sent; ++itr)
        if (0 == strncasecmp ((*itr).name_.c_str (), next_level, (*itr).name_.length ()))
        {
            subpath = &(*itr);
            return true;
        }
    return false;
}
bool TestSelection::increment (const char* path)
{
    // skip front separator
    size_t seplen = strspn (path, TestFacet::separator);
    path += seplen;
    // find next separator in path
    const char* next_sep = strpbrk (path, TestFacet::separator);
    const char* next_path = NULL;
    std::string  sub;
    if (next_sep)
    {
        sub.assign (path, next_sep - path);
        seplen = strspn (next_sep, TestFacet::separator);
        next_path = next_sep + seplen;
        if (!*next_path)
            next_path = NULL;
    }
    else 
        sub.assign (path);

    for (SelVec::iterator itr = branches_.begin (), sent = branches_.end (); itr != sent; ++itr)
        if ((*itr).name_ == sub && next_path)
            return (*itr).increment (next_path);

    branches_.resize (branches_.size () + 1);
    branches_.back ().name_ = sub;
    if (next_path)
        branches_.back ().increment (next_path);
    return true;
}

void TestSelection::print (std::ostream& ostr, unsigned nest) const
{
    ostr << std::setw (nest * INDENT) << "" << "\"" << name_.c_str () << "\"" << std::setw (0) << std::endl;
    ++ nest;
    for (SelVec::const_iterator itr = branches_.begin (), sent = branches_.end (); itr != sent; ++itr)
        (*itr).print (ostr, nest);
}
