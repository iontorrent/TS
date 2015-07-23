/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __text_files_dir_facet_h__
#define __text_files_dir_facet_h__

#include "test_facet.h"
#include <iostream>

class TestDirWithTextFiles_facet : public TestFacet
{
#if defined (_MSC_VER)
    static const char PSEPARATOR = '\\';
#else
    static const char PSEPARATOR = '/';
#endif
    static const unsigned MIN_STRINGS_DEF = 50;
    static const unsigned MAX_STRINGS_DEF = 100;
    static const unsigned FILE_NO_DEF = 8; 
    std::string base_dir_name_;
    bool owner_;
    bool failed_;
    std::ostream* o_;
    unsigned min_strings_;
    unsigned max_strings_;
    unsigned files_no_;
    bool generate_test_tree_in (const char* basedir);
    bool generate_test_text_file (const char* fname);
protected:
    // generates content of the line and writes to the file
    virtual bool output_line (FILE* f, unsigned  lineno, unsigned maxlines, const char* fname);
public:
    TestDirWithTextFiles_facet (unsigned min_strings = MIN_STRINGS_DEF, unsigned max_strings = MAX_STRINGS_DEF, unsigned files_no = FILE_NO_DEF, std::ostream& o = std::cout);
    TestDirWithTextFiles_facet (const char* basedir, unsigned min_strings = MIN_STRINGS_DEF, unsigned max_strings = MAX_STRINGS_DEF, unsigned files_no = FILE_NO_DEF, std::ostream& o = std::cout);
    ~TestDirWithTextFiles_facet ();
    const char* dirname () const
    {
        return base_dir_name_.c_str ();
    }
    bool failed () const
    {
        return failed_;
    }
    void set_stream (std::ostream& o)
    {
        o_ = &o;
    }
    bool init (const char* basedir = NULL);
    bool cleanup ();
};

#endif // __text_files_dir_facet_h__
