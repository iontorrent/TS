/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __contalign_h__
#define __contalign_h__


#include <ostream>
#include "align_common.h"

class ContAlign
{
public:
    enum SCALE_TYPE
    {
        SCALE_NONE = 0,
        SCALE_GEP = 1,
        SCALE_GIP_GEP = 2
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
    double float_error_bound;
    ALIGN_FVECT* ap;
    double gip, gep, mat, mis;
    bool to_first, to_last;
    int accum_hor_skip;
    double low_score;

    char *xhomo;
    char *yhomo;

    std::basic_streambuf <char>* logbuf_;
    std::ostream* own_log_;
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

    void set_scoring (double gip, double gep, double mat, double mis);

    void set_log (std::ostream& log);
    void set_log (int posix_handle);
    void reset_log ();
    void set_scale (SCALE_TYPE scale_type) { scale_type_ = scale_type; }
    SCALE_TYPE get_scale () const { return scale_type_; }

    // calculates best local alignment nucleotide sequence pair
    // returns maximum local alignment score
    double align (const char* xseq, int xlen, const char* yseq, int ylen);

    // calculates best local alignment nucleotide sequence pair
    // on a diagonal band (len, diag-width:diag+width_right)
    // returns maximum local alignment score
    // NOTE: batch xpos, ypos, len should be inside X and Y sequences, width > 0
    // 02/28/2017 : Changing semantics of tobeg / toend to only apply to X sequence (read in TMAP)
    double align_band (const char* xseq, int xlen, const char* yseq, int ylen, int xpos, int ypos, int len, int width, int width_right = -1, bool tobeg = false, bool toend = false);

    // checks if enough matrix space is present for alignment
    bool can_align (int xlen, int ylen, int width, int width_right);

    // follows backtrace matrix, fills BATCH array, returns number of batches
    unsigned  backtrace (BATCH *b_ptr, int max_cnt, unsigned width = 0);

    int get_max_x () const { return max_x; }
    int get_max_y () const { return max_y; }
    double get_score () const { return max_reached?max_w:0.0; }

    int get_last_x () const { return last_x; }
    int get_last_y () const { return last_y; }
    double get_last_score () const { return last_reached?last_w:0.0; }

};


#endif // __contalign_h__
