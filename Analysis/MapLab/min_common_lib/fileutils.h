/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __fileutils_h__
#define __fileutils_h__

#include "common_types.h"


void expand_wildcards (const char* expr, std::vector <std::string>& files, bool verbose = true);
void checkCreateOutputFile (const char* output_name, bool overwrite, bool append);

StrVec      listdir (const char *path);    // returns directory listing
StrVec      listdir (const std::string& path);
StrVec      split_path (const char* path);  // splits path into components
StrVec      split_path (const std::string& path);
std::string join_path (StrVec components);  // merges components into path
std::string join_path (const char* comp0, ...); // last component must be NULL
bool        file_exists (const char* name); // checks for file existence
bool        file_exists (const std::string& name);
bool        is_file (const char* fname);
bool        is_file (const std::string& fname);
bool        is_dir (const char* fname);
bool        is_dir (const std::string& fname);
time_t      file_time (const char* fname);
time_t      file_time (const std::string& fname);
unsigned long long file_size (const char* fname);
unsigned long long file_size (const std::string& fname);
unsigned long long get_open_file_size (int fhandle);
char        path_separator ();

std::string temp_dir (const char* tmpdir = NULL);
// warning: make_temp_fname function is HEAVY (reads directory, uses heap-allocated data etc.; also returns std::string)
// warning: make_temp_fname function creates race condition if used simultanously with same prefix on same directory
std::string make_temp_fname (const char* tmpdir = NULL, const char* prefix = NULL);
int make_linked_temp_file (std::string& dest, const char* tmpdir = NULL, const char* prefix = NULL);
int make_temp_file ();


class LineReader
{
private:
    int fhandle_;
    char *buffer_;
    int cur_line_beg_;
    int cur_line_end_;
    int buf_end_;
    char prev_char_;
    long long cur_pos_;

    void nextChunk ();
public:
    LineReader (const char* fname);
    ~LineReader ();
    char* nextLine ();
    bool isOpen () const;
    long long curPos () const;
    void close ();
};

#undef putc
#define WRITE_BUF_SZ 1024*1024*4 // 4Mb
class BufferedWriter
{
    int fhandle;
    char *buffer;
    int bufst;
    int cpos;
public:
    BufferedWriter (const char* fname = NULL, bool append = false);
    ~BufferedWriter ();
    bool open (const char* fname, bool append = false);
    void close ();
    void flush ();
    bool is_open ()
    {
        return (fhandle != -1 && buffer);
    }
    void write (char* buf, unsigned blen)
    {
        while (blen--) putc (*buf++);
    }
    int puts (const char* s)
    {
        int toR = 0;
        while (*s) putc (*s++), toR ++;
        return toR;
    }
    void putc (char c)
    {
        if (cpos + 1 >= WRITE_BUF_SZ) flush ();
        buffer [cpos++] = c;
    }
    int tell ()
    {
        return bufst + cpos;
    }
};


StrVec read_lists (const StrVec& fnames);
StrVec read_list (const char* fname);
StrVec read_list (std::string fname);

#endif

