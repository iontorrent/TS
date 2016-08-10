/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_file_utils_h__
#define __test_file_utils_h__
#include <extesting.h>
#include <text_files_dir_facet.h>

class TestPathops : public Test
{
public:
    TestPathops ()
    :
    Test ("PathOps: operations on paths (fileutils.h)")
    {
    }
    bool process ();
};

class TestListdir : public Test
{
public:
    TestListdir ()
    :
    Test ("listdir: directory listings (fileutils.h)")
    {
    }
    bool process ();
};

class TestReadlist : public Test
{
    static const unsigned MIN_STRINGS = 200, MAX_STRINGS = 400, FILES_NO = 5;
public:
    TestReadlist ()
    :
    Test ("ReadList: read first tokens from multiple files into a vector")
    {
    }
    bool process ();
};

class TestFileUtils : public Test
{
    static const unsigned MAX_STRINGS = 100000;
    static const unsigned MIN_STRINGS = 50000;
    static const unsigned FILES_NO = 20;
    TestDirWithTextFiles_facet fixture_;

    TestPathops testPathops;
    TestListdir testListdir;
    TestReadlist testReadlist;
public:
    TestFileUtils ()
    :
    Test ("FileUtils: File utilities (fileutils.h)"),
    fixture_ (MIN_STRINGS, MAX_STRINGS, FILES_NO)
    {
    }

    bool init ();
    bool cleanup ();
    bool process ();
};


#endif // __test_file_utils_h__

