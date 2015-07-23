/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <cstring>
#include <ctype.h>
#include <cstdlib>
#include <cerrno>

#include "fasta.h"
#include "rerror.h"
#include "tracer.h"
#include "myassert.h"

// #include <fileutils.h>
#include "portability.h"

static const char file_not_open [] = "INTERNAL ERROR: Operating on unopen fasta file";

void FastaFile::reset ()
{
    cur_pos_ = 0;
    prev_pos_ = 0;
    *linebuf_ = 0;
    *seqbuf_ = 0;
    *namebuf_ = 0;
    l_ = linebuf_;
}


void FastaFile::parse_hdr ()
{
    if (!*l_) return;
    myassert (*l_ == '>'); // the parse_header should be called only if l_ contains ">" in zero position

    char* p = l_ + 1;

    // find ad copy name
    
    // skip to first non-space
    while (*p && isspace (*p)) p ++;
    // if at the end of the line
    if (*p == 0) 
    {
        descr_beg_ = 0;
        namebuf_ [0] = 0;
        hdrbuf_ [0] = 0;
        return;
    }
    // there is a non-space character - means name has at least one char. Get it.
    // skip to first space not more distant from p1 then MAX_NAME_LEN
    char* pe = p;
    while (*pe && !isspace (*pe) && pe - p < MAX_NAME_LEN) pe ++;
    // copy name into namebuf
    memcpy (namebuf_, p, pe - p);
    namebuf_ [pe - p] = 0; // and zero-terminate it
    descr_beg_ = pe - l_ - 1;
    // advance to the very end of the string
    while (*pe) ++pe;
    --pe; // pe-1 is always valid address since pe 
    // check if end of line was read
    hdr_eol_read_ = (*pe == '\n'); 
    // skip final whitespace
    while (pe > l_ && isspace (*pe))
        --pe;
    // pe now is at last non-whitespace position. Advance to next to make it sentinel
    ++pe;
    // copy header
    unsigned hdrlen = std::min (MAX_HDR_LEN, pe - l_ - 1); // find how much of the geader can be stored
    memcpy (hdrbuf_, l_ + 1, hdrlen); // copy header
    *(hdrbuf_ + hdrlen) = 0; // and zero-terminate it
}

void FastaFile::add_seq ()
{
    char* p = l_;
    while (*p)
    {
        if (!isspace (*p))
        {
            if (seqlen_ == seq_buf_sz_)
            {
                warnlog << "Reallocating FastaReader buffer!" << std::endl;
                // reallocate seqbuf_: increment space twice
                unsigned new_sz = seq_buf_sz_ * 2;
                char* newbuf;
                try
                {
                    newbuf = new char [new_sz + 1];
                }
                catch (std::bad_alloc&)
                {
                    newbuf = NULL;
                }
                if (!newbuf)
                    ers << "ERROR: Not enough memory to hold the sequence" << namebuf_ << ": error allocating " << new_sz + 1 << " bytes" <<  Throw;
                memcpy (newbuf, seqbuf_, seq_buf_sz_);
                delete [] seqbuf_;
                seqbuf_ = newbuf;
                seq_buf_sz_ = new_sz;
            }
            seqbuf_ [seqlen_] = tolower (*p);
            seqlen_ ++;
        }
        p ++;
    }
}

FastaFile::FastaFile (ulonglong init_sz)
:
f_ (NULL)
{
    try
    {
        seqbuf_ = new char [init_sz+1];
    }
    catch (std::bad_alloc&)
    {
        seqbuf_ = NULL;
    }
    if (!seqbuf_)
        ers << "Memory allocation error: unable to allocate " << init_sz + 1 << " bytes" << Throw;

    seq_buf_sz_ = init_sz;
    reset ();
}

FastaFile::FastaFile (const char* name, ulonglong init_sz)
:
f_ (NULL)
{
    try
    {
        seqbuf_ = new char [init_sz+1];
    }
    catch (std::bad_alloc&)
    {
        seqbuf_ = NULL;
    }
    if (!seqbuf_)
        ers << "Memory allocation error: unable to allocate " << init_sz + 1 << " bytes"  << Throw;

    seq_buf_sz_ = init_sz;
    reset ();
    if (!open (name))
    {
        ers << "ERROR: Could not open fasta file" << name << Throw;
    }
}

FastaFile::~FastaFile ()
{
    close ();
    delete [] seqbuf_;
    seq_buf_sz_ = 0;
}

bool FastaFile::open (const char* name)
{
    if (f_) fclose (f_);
    reset ();

    l_ = linebuf_;
    *l_ = 0;
    f_ = fopen (name, "rb");
    if (f_ != NULL)
    {
        fseek (f_, 0, SEEK_END);
        tot_len_ = ftell (f_);
        fseek (f_, 0, SEEK_SET);
        while (*l_ != '>')
        {
            prev_pos_ = cur_pos_;
            if (!(l_ = fgets (linebuf_, MAX_LINE_LEN, f_)))
            {
                if (ferror (f_))
                {
                    ers << "Error reading fasta file: " << strerror (errno) << Throw;
                }
                else
                {
                    fclose (f_);
                    f_ = NULL;
                    *linebuf_ = 0;
                    ers << "Fasta file: " << name << " has no valid records" << Throw;
                }
            }
            cur_pos_ = ftell (f_);
        }
        seq_no_ = unsigned (-1);
        return true;
    }
    return false;
}

bool FastaFile::close ()
{
    if (f_)
    {
        fclose  (f_);
        f_ = NULL;
        return true;
    }
    else
        return false;
}

bool FastaFile::next ()
{
    unsigned ll;
    if (cur_pos_ == tot_len_) 
        return false;
    cur_recstart_ = prev_pos_;
    parse_hdr ();
    while (!hdr_eol_read_)
    {
        l_ = fgets (linebuf_, MAX_LINE_LEN, f_);
        if (!l_)
        {
            cur_pos_ = tot_len_;
            return false;
        }
        ll = strlen (l_);
        if (ll && l_ [ll - 1] == '\n') // theoretically ll always non-zero:  NULL should be returned from fgets if file pointer is at eol
            hdr_eol_read_ = true;
    }
    if (hdr_eol_read_)
    {
        seqlen_ = 0;
        l_ = fgets (linebuf_, MAX_LINE_LEN, f_);
        if (l_)
        {
            do
            {
                prev_pos_ = cur_pos_;
                cur_pos_ = ftell (f_);
                if (*l_ == '>')
                    break;
                add_seq ();
            }
            while ((l_ = fgets (linebuf_, MAX_LINE_LEN, f_)));
        }
        else
        {
            cur_pos_ = tot_len_;
        }
        seq_no_ ++;
        cur_reclen_ = prev_pos_ - cur_recstart_;
        seqbuf_ [seqlen_] = 0;
    }
    return true;
}

bool FastaFile::seek (ulonglong off)
{
    if (!f_) return false;
    if (off >= tot_len_) return false;
    fseek (f_, off, SEEK_SET);
    prev_pos_ = off;
    l_ = fgets (linebuf_, MAX_LINE_LEN, f_);
    if (!l_ || *l_ != '>') ERR ("Invalid offset in fasta file: not at record start");
    cur_pos_ = ftell (f_);
    return true;
}

const char* FastaFile::cur_name ()  const
{
    return namebuf_;
}

const char* FastaFile::cur_hdr  () const
{
    return hdrbuf_;
}

const char* FastaFile::cur_seq  () const
{
    return seqbuf_;
}

char* FastaFile::cur_seq_buf  ()
{
    return seqbuf_;
}

unsigned FastaFile::cur_no  () const
{
    return seq_no_;
}

ulonglong FastaFile::tot_len () const
{
    return tot_len_;
}

ulonglong FastaFile::cur_pos () const
{
    return cur_pos_;
}

ulonglong FastaFile::cur_recstart () const
{
    return cur_recstart_;
}

unsigned FastaFile::cur_reclen () const
{
    return cur_reclen_;
}


#ifdef FASTA_TEST
int main (int argc, char* argv [])
{
    try
    {
        ulonglong opa [100];
        int c = 0;
        if (argc < 2) ERR ("No argument");
        FastaFile f (argv [1]);
        while (f.next () && c < 100)
        {
            ((int*) (opa + c)) [0] = f.cur_recstart ();
            ((int*) (opa + c)) [1] = f.cur_reclen ();
            c ++;
        }

        for (c = 0; c < 100; c ++)
        {
            f.seek (((int*) (opa + c)) [0]);
            f.next ();
            printf (">%s %s\n%s\n", f.cur_name (), f.cur_hdr (), f.cur_seq ());
        }

    }
    catch (Rerror& e)
    {
        printf ("%s\n", (const char*) e);
        return 1;
    }
    return 0;
}
#endif
