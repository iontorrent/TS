/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_file_utils.h"
#include <fileutils.h>
#include <test_args.h>

bool TestFileUtils :: init ()
{
    fixture_.set_stream (o_);
    bool res = fixture_.init (find_test_arg ("--file-utils-dir")); // NULL as basename is Ok - it directs fixture to automatically choose location (in temp dir)
    if (res)
    {
        add_subord (&testPathops);
        add_subord (&testListdir);
        add_subord (&testReadlist);
    }
    return res;
}
bool TestFileUtils :: cleanup ()
{
    return fixture_.cleanup ();
}
bool TestFileUtils :: process ()
{
    std::string fname = fixture_.dirname ();
    fname += path_separator ();
    fname += "*.txt";
    StrVec sv;
    expand_wildcards (fname.c_str (), sv);
    StrVec::iterator itr = sv.begin ();
    for (; itr != sv.end (); itr ++)
    {
        o_ << (*itr).c_str () << std::endl;
    }
    if (!sv.size ())
    {
        o_ << "No files to test reader" << std::endl;
        return false;
    }

    LineReader lr (sv.front ().c_str ());
    char* ln;
    int no = 0;
    int slen = 0;
    time_t st = time (NULL);
    while ((ln = lr.nextLine ()))
    {
        ++ no;
        slen += strlen (ln);
        //if (strlen (ln) > 100) o_ << ln;
        if (no % 100000 == 0)
        {
            int td = time (NULL) - st;
            double rd = double (slen) / (1024*1024);
            o_ << "\rline " << no << ", " << rd << " Mb, average speed " << rd / (td?td:1) << " Mb/s    " << std::flush;
        }
    }
    time_t et = time (NULL);
    lr.close ();
    return true;
}

bool TestPathops::process ()
{
#ifdef _MSC_VER
    const char* paths [] = {"\\kokos\\banan", "d:\\Encyclopedia\\Data\\Genomes\\E.coli.K12\\proteins\\fasta\\", "", "\\", "\\ananas\\", NULL};
#else
    const char* paths [] = {"/kokos/banan", "~/Encyclopedia/Data/Genomes/E.coli.K12/proteins/fasta/", "", "/", "/ananas/", NULL};
#endif
    const char** path = paths;
    for (; *path != NULL; path ++)
    {
        o_ << "Path " << *path << std::endl;
        StrVec comps = split_path (*path);
        o_ << "path " << *path << ", " << comps.size () << " components" << std::endl;
        StrVec::const_iterator i = comps.begin ();
        for (;i < comps.end (); i ++)
            o_ << "  '" << (*i).c_str () << "'" << std::endl;
        std::string rejoined = join_path (comps);
        o_ << "  Rejoined: " << rejoined.c_str () << std::endl;
    }

    std::string jp = join_path ("kokos", "banan", "ananas", "yabloko", NULL);
    o_ << jp.c_str () << std::endl;
    jp = join_path ("", "kokos", NULL);
    o_ << jp.c_str () << std::endl;
    jp = join_path ("", NULL);
    o_ << jp.c_str () << std::endl;
    jp = join_path ("kokos", NULL);
    o_ << jp.c_str () << std::endl;
    jp = join_path (NULL);
    o_ << jp.c_str () << std::endl;
    return true;
}

bool TestListdir::process ()
{
    typedef std::vector <std::string> StrVec;
    StrVec file_list;
    const char *dir_name = ".";
    file_list = listdir (dir_name);
    o_ << "Listing of directory '" << dir_name << "':" << std::endl;
    StrVec::iterator i = file_list.begin();
    while (i != file_list.end())
    {
        o_ << (*i).c_str () << std::endl;
        i ++;
    }
    return true;
}

bool TestReadlist::process ()
{
    TestDirWithTextFiles_facet facet (MIN_STRINGS, MAX_STRINGS, FILES_NO, o_);
    std::string fname = facet.dirname ();
    fname += path_separator ();
    fname += "*.txt";
    StrVec names;
    expand_wildcards (fname.c_str (), names);
    StrVec ll = read_lists (names);
    return true;
}
