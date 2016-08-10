/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#include "text_files_dir_facet.h"

#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <sstream>
#include "fileutils.h"


bool TestDirWithTextFiles_facet::generate_test_tree_in (const char* basedir)
{
    std::string tempdir = basedir;
    tempdir += PSEPARATOR;
    tempdir += "test_XXXXXX";
    char* buf = (char*) alloca (tempdir.length () + 1);
    strncpy (buf, tempdir.c_str (), tempdir.length ());
    buf [tempdir.length ()] = 0;
    (*o_)  << "FACET: Generating test subtree at " << buf << std::endl;
    const char* dname = mkdtemp (buf);
    if (!dname)
    {
        (*o_) << "FACET ERROR: Could not create temp dir at " << buf << ": " << strerror (errno) << std::endl;;
        return false;
    }
    base_dir_name_ = dname;
    owner_ = true;
    const char* base_fname = "test_text_";
    for (unsigned c = 0; c != files_no_; ++c)
    {
        std::ostringstream fns;
        fns << dname << PSEPARATOR << base_fname << c << ".txt";
        const char* fn = fns.str ().c_str ();
        if (!generate_test_text_file (fn))
            return false;
    }
    return true;
}

bool TestDirWithTextFiles_facet::generate_test_text_file (const char* fname)
{
    unsigned num_strings = min_strings_ + rand () % (max_strings_-min_strings_);
    bool result = true;
    FILE* f = fopen (fname, "w");
    if (!f)
    {
        (*o_)  << "FACET ERROR: Could not open file " << fname << " for writing: "  << strerror (errno) << std::endl;
        result = false;
    }
    for (unsigned sidx = 0; sidx != num_strings && result; ++sidx)
    {
        if (!output_line (f, sidx, num_strings, fname))
        {
            (*o_)  << "FACET ERROR: Failed to write to the file " << fname << ": " << strerror (errno) << std::endl;
            result = false;
        }
    }
    if (fclose (f) != 0)
    {
        (*o_)  << "FACET ERROR: Failed to close file " << fname << ": " << strerror (errno) << std::endl;
        result = false;
    }
    return result;
}

bool TestDirWithTextFiles_facet ::output_line (FILE* f, unsigned  lineno, unsigned maxlines, const char* fname)
{
    return (fprintf (f, "Line_%d out of total %d, in test file %s\n", lineno+1, maxlines, fname) >= 0);
}

static const char* TestDirWithTextFiles_facet_name = "TestDirWithTextFiles";

TestDirWithTextFiles_facet ::TestDirWithTextFiles_facet (unsigned min_strings, unsigned max_strings, unsigned files_no, std::ostream& o)
:
TestFacet (TestDirWithTextFiles_facet_name),
owner_ (false),
failed_ (false),
o_ (&o),
min_strings_ (min_strings),
max_strings_ (max_strings),
files_no_ (files_no)
{
}

TestDirWithTextFiles_facet ::TestDirWithTextFiles_facet (const char* basedir, unsigned min_strings, unsigned max_strings, unsigned files_no, std::ostream& o)
:
TestFacet (TestDirWithTextFiles_facet_name),
owner_ (false),
failed_ (false),
o_ (&o),
min_strings_ (min_strings),
max_strings_ (max_strings),
files_no_ (files_no)
{
    init (basedir);
}

TestDirWithTextFiles_facet ::~TestDirWithTextFiles_facet ()
{
    cleanup ();
}

bool TestDirWithTextFiles_facet ::init (const char* basedir)
{
    failed_ = false;
    if (!basedir)
    {
        std::string tempdir = temp_dir ();
        if (!generate_test_tree_in (tempdir.c_str ()))
            failed_ = true;
    }
    else
        base_dir_name_ = basedir;
    return !failed_;
}

bool TestDirWithTextFiles_facet ::cleanup ()
{
    if (owner_ && base_dir_name_.length ())
    {
        (*o_)  << "FACET: Removing test subtree at " << base_dir_name_ << std::endl;
        std::string command = "rm -rf ";
        command += base_dir_name_;
        if (system (command.c_str ()) != 0)
        {
            (*o_)  << "FACET: Failed removing test subtree at " << base_dir_name_ << ": " << strerror (errno) << std::endl;
            return false;
        }
    }
    return true;
}

