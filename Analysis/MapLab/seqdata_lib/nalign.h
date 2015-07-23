/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __nalign_h__
#define __nalign_h__

#include "align_common.h"

#define DEBUG_TRACE 1
class Align
{
    int xref, yref, bstep;
    int max_ylen, max_size;
    int max_w, max_x, max_y;
    bool max_reached;
    int last_w, last_y, last_x;
    bool last_reached;
    char* max_bp;
    char* last_bp;
    char* btrmx;
    ALIGN_VECT* ap;
    int gip, gep, mat, mis;
    bool to_first, to_last;
    int accum_hor_skip;
    int nocost_gap;
    int low_score;

    bool trace_mode;
    std::ostream* logp_;

    /*
    member inner loops used for finding alignment
    fill backtrace matrix and store max score position
    returns the last score (in the topmost computed cell in a column)
    */
    int align_y_loop (register ALIGN_VECT* ap, unsigned char base, char* bp, int x, int y, int len, bool bottom_in, int add_score, bool last_col = false);

public:

    Align ();
    Align (int max_ylen, int max_size, int gip, int gep, int mat, int mis);
    ~Align ();

    void init (int max_ylen, int max_size, int gip, int gep, int mat, int mis);

    void set_scoring (int gip, int gep, int mat, int mis);

    void set_trace (bool trace) { trace_mode = trace; }
    void set_log (std::ostream& log) { logp_ = &log; }

    // calculates best local alignment nucleotide sequence pair
    // returns maximum local alignment score
    int  align (const char* xseq, int xlen, const char* yseq, int ylen, bool unpack = true);

    // calculates best local alignment nucleotide sequence pair
    // on a diagonal band (len, diag +- width)
    // returns maximum local alignment score
    // NOTE: batch xpos, ypos, len should be inside X and Y sequences, width > 0
    int  align_band (const char* xseq, int xlen, const char* yseq, int ylen, int xpos, int ypos, int len, int width, bool unpack = true, int width_right = -1, bool tobeg = false, bool toend = false);

    // follows backtrace matrix, fills BATCH array, returns number of batches
    unsigned  backtrace (BATCH *b_ptr, int max_cnt, bool reverse = false, unsigned width = 0);

    int get_max_x () const { return max_x; }
    int get_max_y () const { return max_y; }
    int get_score () const { return max_reached?max_w:0; }

    int get_last_x () const { return last_x; }
    int get_last_y () const { return last_y; }
    int get_last_score () const { return last_reached?last_w:0; }

    // QWORD encode (); // encodes the (short) alignment into 8-byte value, as 8 1-byte segments: 2bit segment type (match, mismatch, xskip, yskip) + 6-bit segment length (up to 64 bases).

};


#endif // __nalign_h__
