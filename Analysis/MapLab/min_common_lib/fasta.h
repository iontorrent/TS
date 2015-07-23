/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __fasta_h__
#define __fasta_h__

#include <cstdio>
#include "platform.h"

#define MAX_HDR_LEN 8000L
#define MAX_NAME_LEN 160L
#define MAX_LINE_LEN 8000L
// #define INIT_SEQ_LEN 300000000ULL // 300M
#define INIT_SEQ_LEN 1000000ULL // 1M

class FastaFile
{
    FILE* f_;
    ulonglong tot_len_;
    ulonglong cur_pos_;
    ulonglong prev_pos_;
    ulonglong cur_recstart_;
    unsigned cur_reclen_;
    unsigned seq_no_;
    char *seqbuf_;
    unsigned seq_buf_sz_;
    unsigned seqlen_;
    char hdrbuf_  [MAX_HDR_LEN+1];
    unsigned descr_beg_;
    bool hdr_eol_read_;
    char namebuf_ [MAX_NAME_LEN+1];
    char linebuf_ [MAX_LINE_LEN];
    char* l_;

    void reset ();
    void parse_hdr ();
    void add_seq ();

public:
    FastaFile (ulonglong init_sz = INIT_SEQ_LEN);
    FastaFile (const char* name, ulonglong init_sz = INIT_SEQ_LEN);
    ~FastaFile ();

    bool        open (const char* name);
    bool        close ();
    bool        next ();
    bool        seek (ulonglong off);
    bool        is_open () const {return f_ != NULL;}
    void        fetch_hdr () { parse_hdr (); }

    const char* cur_name () const;
    const char* cur_hdr  () const;
    const char* cur_descr () const;
    const char* cur_seq  () const;
    char*       cur_seq_buf ();
    unsigned    cur_seq_len () const { return seqlen_; }
    unsigned    cur_no  () const;
    ulonglong   tot_len () const;
    ulonglong   cur_pos () const;
    unsigned    cur_reclen () const;
    ulonglong   cur_recstart () const;
};

#endif
