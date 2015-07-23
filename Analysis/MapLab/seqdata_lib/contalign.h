/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __contalign_h__
#define __contalign_h__

#include "align_common.h"

#define DEBUG_TRACE 1

class ContAlign
{
public:
    enum SCALE_TYPE
    {
        SCALE_NONE,
        SCALE_GEP,
        SCALE_GIP_GEP
    };
private:
    int xref, yref, bstep;

    int max_ylen, max_xlen, max_size;

    int max_x, max_y;
    double max_w;
    bool max_reached;

    int last_y, last_x;
    double last_w;
    bool last_reached;

    char* max_bp;
    char* last_bp;
    char* btrmx;
    ALIGN_FVECT* ap;
    double gip, gep, mat, mis;
    bool to_first, to_last;
    int accum_hor_skip;
    double low_score;

    char *xhomo;
    char *yhomo;

    bool trace_mode;
    std::ostream* logp_;
    SCALE_TYPE scale_type_;

    /*
    member inner loops used for finding alignment
    fill backtrace matrix and store max score position
    returns the last score (in the topmost computed cell in a column)
    */
    double align_y_loop (register ALIGN_FVECT* ap, unsigned char base, char* bp, int x, int xdivisor, int y, int len, bool bottom_in, double add_score, bool last_col = false);

public:

    ContAlign ();
    ContAlign (int max_ylen, int max_xlen, int max_size, int gip, int gep, int mat, int mis);
    ~ContAlign ();

    void init (int max_ylen, int max_xlen, int max_size, int gip, int gep, int mat, int mis);

    void set_scoring (int gip, int gep, int mat, int mis);

    void set_trace (bool trace) { trace_mode = trace; }
    void set_log (std::ostream& log) { logp_ = &log; }
    void set_scale (SCALE_TYPE scale_type) { scale_type_ = scale_type; }
    SCALE_TYPE get_scale () const { return scale_type_; }

    // calculates best local alignment nucleotide sequence pair
    // returns maximum local alignment score
    double align (const char* xseq, int xlen, const char* yseq, int ylen, bool unpack = true);

    // calculates best local alignment nucleotide sequence pair
    // on a diagonal band (len, diag +- width)
    // returns maximum local alignment score
    // NOTE: batch xpos, ypos, len should be inside X and Y sequences, width > 0
    double align_band (const char* xseq, int xlen, const char* yseq, int ylen, int xpos, int ypos, int len, int width, bool unpack = true, int width_right = -1, bool tobeg = false, bool toend = false);

    // follows backtrace matrix, fills BATCH array, returns number of batches
    unsigned  backtrace (BATCH *b_ptr, int max_cnt, bool reverse = false, unsigned width = 0);

    int get_max_x () const { return max_x; }
    int get_max_y () const { return max_y; }
    double get_score () const { return max_reached?max_w:0.0; }

    int get_last_x () const { return last_x; }
    int get_last_y () const { return last_y; }
    double get_last_score () const { return last_reached?last_w:0.0; }

    // QWORD encode (); // encodes the (short) alignment into 8-byte value, as 8 1-byte segments: 2bit segment type (match, mismatch, xskip, yskip) + 6-bit segment length (up to 64 bases).

};


#endif // __nalign_h__
