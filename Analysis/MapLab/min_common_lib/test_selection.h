/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_selection_h__
#define __test_selection_h__

#include <vector>
#include <string>

class TestSelection
{
    static unsigned const INDENT = 4;
public:
    typedef std::vector <TestSelection> SelVec;
    std::string name_;
    SelVec branches_;
    bool leaf () const;
    bool includes (const char* next_level, const TestSelection*& subpath) const;
    bool increment (const char* path);
    void print (std::ostream& o, unsigned nest = 0) const;
};
inline std::ostream& operator << (std::ostream& ostr, const TestSelection& ts)
{
    ts.print (ostr);
    return ostr;
}

#endif // __test_selection_h__
