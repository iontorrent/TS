/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <limits>
#include "sequtil.h"
#include "contalign.h"
#include <cassert>
#include <boost/concept_check.hpp>
#include <ext/stdio_filebuf.h>
#include <cfloat>

// #define EXPANDED_LOG

#if defined (EXPANDED_LOG)
    #define LOG_UNIT_WIDTH (3+3*9)
#else
    #define LOG_UNIT_WIDTH (3+6)
#endif

typedef union {
  long long ll;
  double d;
} dpbits;

static inline double epsilon (double val)
{
    dpbits test;
    if (val < 0) val = -val;
    test.d = val;
    ++ test.ll;
    return test.d - val;
}

// const double FEB = 0.0000001;
#define FEB float_error_bound
/*
member inner loops used for finding alignment
fill backtrace matrix and store max score position
*/

double ContAlign::align_y_loop (register ALIGN_FVECT* ap, unsigned char base, char* bp, int x, int xdivisor, int y, int len, bool bottom_in, double add_score, bool last_col)
{
    //initialise bottom boundary
    double prev_w, save_w;
#if defined (EXPANDED_LOG)
    double save_h, save_v;
#endif
    if (bottom_in)
    {
        if (to_first)
        {
            prev_w = accum_hor_skip;
            if (!accum_hor_skip) accum_hor_skip -= gip;
            accum_hor_skip -= gep;
        }
        else
        {
            prev_w = add_score;
        }
    }
    else
        prev_w = (ap-1)->w;

    double v = low_score, nv, nh; // works for both to_first and normal mode: just guarantees that cost of coming from the bottom is higher then by diagonal
    double w, pw = prev_w;

    if (logp_)
    {
        (*logp_) << "\n";
        (*logp_) << std::setw (6) << std::right << x << " " << (char) ((base < 4) ? base2char (base) : base) << " ";
        if (y > yref)
            (*logp_) << std::setw (LOG_UNIT_WIDTH * (y-yref)) << std::left << " ";
        (*logp_) << std::flush;
    }

    y += len - 1;
    while (len-- > 0)
    {
        char dir = ALIGN_DIAG;

        //w = max (0, h, v, prev_w + s(x, y))
        w = prev_w + ((ap->r == base || ap->r == 'N') ? mat : mis); // HACK - need proper handling for non-standard IUPACs
        float_error_bound += epsilon (w);
#if defined (EXPANDED_LOG)        
        save_h = ap->h;
#endif
        if (w + FEB < ap->h)   // hack to support left-align
            w = ap->h, dir = ALIGN_LEFT;

#if defined (EXPANDED_LOG)
        save_v = v;
#endif
        if (w + FEB < v)
            w = v, dir = ALIGN_DOWN;
#if 0
        if (logp_ && x >= 12 && x <= 14 && y-len >= 12 && y-len <= 14)
        {
            (*logp_) << x << ":" << y-len << ": float_error_bound = " << std::scientific << std::setprecision (6) << float_error_bound << ", w-v = " << (v-w) << " h-w = " << (ap->h - w) << std::endl;
        }
#endif

        if (w <= 0)
        {
            if (!to_first)
                w = 0, dir = ALIGN_STOP;
            else if (w < 0)
                dir |= ALIGN_ZERO;
        }

        save_w = w;

        // save max score position
        if (w >= max_w)
            max_w = w, max_bp = bp, max_x = x, max_y = y - len, max_reached = true;

        // for semiglobal mode, remember best edge score
        if (last_col && w >= last_w)
            last_w = w, last_bp = bp, last_x = x, last_y = y - len, last_reached = true;

        //save w[x][y] for next pass
        prev_w = ap->w;
        ap->w = pw = w;

        // h = max (w - gip, h) - gep;
        nh = w - ((scale_type_ == SCALE_GIP_GEP) ? gip / xdivisor : gip);
        float_error_bound += epsilon (nh);

        if (nh > ap->h)
            ap->h = nh, dir |= ALIGN_HSKIP;
        ap->h -= ((scale_type_ == SCALE_NONE) ? gep : gep / xdivisor);
        float_error_bound += epsilon (ap->h);

        //v = max (w - gip, v) - gep;
        nv = w - ((scale_type_ == SCALE_GIP_GEP) ? gip / ap->div : gip);
        float_error_bound += epsilon (nv);
        if (nv > v)
            v = nv, dir |= ALIGN_VSKIP;
        v -= ((scale_type_ == SCALE_NONE) ? gep : gep / ap->div); 
        float_error_bound += epsilon (v);

        if (logp_)
        {
            switch (dir&3)
            {
                case ALIGN_DOWN: (*logp_) << "-";  break;
                case ALIGN_LEFT: (*logp_) << "|";  break;
                case ALIGN_DIAG: (*logp_) << "\\"; break;
                case ALIGN_STOP: (*logp_) << "#";  break;
            }

            if (dir & ALIGN_VSKIP)
                (*logp_) << ((dir & ALIGN_ZERO) ? "V" : "v");
            else
                (*logp_) << "-";

            if (dir & ALIGN_HSKIP)
                (*logp_) << ((dir & ALIGN_ZERO) ? "H" : "h");
            else
                (*logp_) << "-";

            #if defined (EXPANDED_LOG)
            (*logp_) << std::setw (9) << std::resetiosflags (std::ios::floatfield) << std::left  << std::showpos << std::setprecision (3) << save_v
                     << std::setw (9) << std::resetiosflags (std::ios::floatfield) << std::left  << std::showpos << std::setprecision (3) << save_w
                     << std::setw (9) << std::resetiosflags (std::ios::floatfield) << std::left  << std::showpos << std::setprecision (3) << save_h;
            #else
            (*logp_) << std::setw (6) << std::resetiosflags (std::ios::floatfield) << std::left  << std::showpos << std::setprecision (3) << save_w;
            #endif
        }

        //save bactrace pointer (4bits / byte)
        *bp++ = dir;
        ap++;
    }
    if (logp_)
    {
        (*logp_) << "(" << y-len << ")" << std::flush;
    }
    return pw;
}


ContAlign::ContAlign ()
:
btrmx (NULL),
ap (NULL),
xhomo (NULL),
yhomo (NULL),
logbuf_ (NULL),
own_log_ (NULL),
logp_ (NULL)
{
}


ContAlign::ContAlign (int max_ylen, int max_xlen, int max_size, int gip, int gep, int mat, int mis)
:
btrmx (NULL),
ap (NULL),
xhomo (NULL),
yhomo (NULL),
logbuf_ (NULL),
own_log_ (NULL),
logp_ (NULL)
{
    init (max_ylen, max_xlen, max_size, gip, gep, mat, mis);
}


ContAlign::~ContAlign ()
{
    delete [] ap;
    delete [] btrmx;
    delete [] xhomo;
    delete [] yhomo;
    reset_log ();
}

void ContAlign::init (int max_ylen, int max_xlen, int max_size, int gip, int gep, int mat, int mis)
{
    if (ap) delete [] ap;
    if (btrmx) delete [] btrmx;
    if (xhomo) delete [] xhomo;
    if (yhomo) delete [] yhomo;

    ContAlign::max_ylen = max_ylen;
    ContAlign::max_xlen = max_xlen;
    ContAlign::max_size = max_size;
    ContAlign::gip = gip;
    ContAlign::gep = gep;
    ContAlign::mis = mis;
    ContAlign::mat = mat;
    ContAlign::logp_ = NULL;
    ContAlign::low_score = -gip - std::max (max_ylen, max_size / max_ylen) * (gep - mis) - 1;

    //allocate running Y-vector and backtrace matrix
    btrmx = new char [max_size];
    std::fill (btrmx, btrmx + max_size, ALIGN_STOP);
    ap = new ALIGN_FVECT [max_ylen];
    yhomo = new char [max_ylen+1];
    xhomo = new char [max_xlen+1];
}

void ContAlign::set_scoring ( double gip, double gep, double mat, double mis)
{
    ContAlign::gip = gip;
    ContAlign::gep = gep;
    ContAlign::mis = mis;
    ContAlign::mat = mat;
    ContAlign::low_score = -gip - std::max (max_ylen, max_size / max_ylen) * gep - 1;
}

static void fill_homo_tract_len (const char* seq, int seq_len, char* dest)
{
    if (seq_len)
    {
        char pbase = *seq, len;
        int ppos = 0, pos;
        for (pos = 1; pos != seq_len; ++pos)
            if (seq [pos] != pbase)
            {
                len = (pos - ppos < std::numeric_limits< char>::max ()) ? (pos- ppos) : std::numeric_limits< char>::max ();
                while (ppos != pos)
                    dest [ppos++] = len;
                pbase = seq [pos];
            }
        len = (pos - ppos < std::numeric_limits< char>::max ()) ? (pos- ppos) : std::numeric_limits< char>::max ();
        while (ppos != pos)
            dest [ppos++] = len;
        dest [ppos] = 1;
    }
 }

/*
calculates best local alignment between pair of nucleotide sequences
returns maximum local alignment score
*/
double ContAlign::align (const char* xseq, int xlen, const char* yseq, int ylen)
{
    int x, y;
    char* bp = btrmx;
    max_reached = false;
    last_reached = false;
    accum_hor_skip = 0;
    to_first = false, to_last = false;

    //check if enough memory allocated for band alignment
    assert (ylen <= max_ylen);
    assert (xlen * ylen <= max_size);
    assert (xlen <= max_xlen);

    bstep = ylen;
    xref = 0, yref = 0;

    // initialize homopolimer size indices
    fill_homo_tract_len (xseq, xlen, xhomo);
    fill_homo_tract_len (yseq, ylen, yhomo);

    //initialize left boundary
    //unpack Y sequence for faster processing
    for (y = 0; y < ylen; y++)
    {
        ap[y].w = 0;
        ap[y].h = low_score;
        ap[y].r = yseq [y];
        ap[y].div = yhomo [y+1];
    }

    //find best local alignment
    double cur_w;
    max_w = last_w = - std::numeric_limits <double>::max ();
    max_bp = btrmx;
    for (x = 0; x < xlen; x++, bp += bstep)
    {
        cur_w = align_y_loop (ap, xseq [x], bp, x, xhomo [x], 0, ylen, true, 0.0);
        // remember X edge terminal scores
        if (last_w < cur_w)
            last_w = cur_w, last_bp = bp + ylen - 1, last_x = x, last_y = ylen-1, last_reached = true;
    }

    if (logp_)
        (*logp_) << std::endl;

    return get_score ();
}


bool ContAlign::can_align (int xlen, int ylen, int width, int width_right)
{
   if (width_right == -1)
        width_right = width;

   xlen = std::max (xlen, ylen);

   if (ylen > max_ylen) 
       return false;
   if (xlen * (width + width_right + 1) > max_size) 
       return false;
   return true;
}


/*
calculates best local alignment between pair of nucleotide sequences
on a diagonal band (len, diag +- width)
returns maximum local alignment score
NOTE: batch xpos, ypos, len should be inside X and Y sequences, width > 0

*/
double ContAlign::align_band (const char* xseq, int xlen, const char* yseq, int ylen, int xpos, int ypos, int len, int width, int width_right, bool tobeg, bool toend)
{
    char* bp = btrmx;
    max_reached = false;
    last_reached = false;
    to_first = tobeg, to_last = toend;
    accum_hor_skip = 0;
    float_error_bound = 0;

    if (width_right == -1)
        width_right = width;

    // check if enough memory allocated for band alignment
    // and if batch variables are sane
    assert (ylen <= max_ylen);
    assert (len * (width + width_right + 1) <= max_size);


    bstep = width + 1 + width_right;
    xref = xpos, yref = ypos;

    // initialize homopolimer size indices
    fill_homo_tract_len (xseq + xref, xlen - xref, xhomo);
    fill_homo_tract_len (yseq + yref, ylen - yref, yhomo);

    // initialize left boundary
    // unpack Y sequence for faster processing
    int ylast = std::min (ylen, yref + len);

    if (logp_)
    {
        (*logp_) << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            (*logp_) << std::setw (LOG_UNIT_WIDTH) << std::left << yy;
        (*logp_) << "\n";
        (*logp_) << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            (*logp_) << std::setw (LOG_UNIT_WIDTH) << std::left << (char) yseq [yy];
        (*logp_) << "\n" << std::flush;
        (*logp_) << std::setw (9) << "";
    }

    // for (int i = std::max (0, yref - width); i < ylast; i++)
    // changed from above line to always have the valid prev_w when y > 0
    double curw = -gip-gep;
    // for (int i = max_ (0, yref - width - 1); i < ylast; i++)
    for (int i = yref; i < ylast; i++)
    {
        ap [i].w = curw;
        if (logp_)
            (*logp_) <<  std::left << std::setw (LOG_UNIT_WIDTH) << std::left  << std::resetiosflags (std::ios::floatfield) << std::setprecision (3) << curw << std::flush;
            // (*logp_) << "   " << std::left << std::setw (6) << std::left  << std::fixed << std::setprecision (4) << curw << std::flush;

        curw -= gep;
        ap[i].h = low_score; // guaranteed to always prevent 'coming from the right'. Works for both tobeg and not.
        ap[i].r = yseq [i];
        ap[i].div = yhomo [i - yref + 1];
    }

    // find best local alignment, save backtrace pointers
    last_w = max_w = - std::numeric_limits <double>::max ();
    max_bp = btrmx, ypos -= width;

    int y_start, y_end, y_len; //, y;
    char *bp_start;
    // double cur_w;
    bool bottom_in = false;
    double add_score;

    bool last_col = false;
    while (len-- > 0)
    {
        if (xpos == xlen)
            break;

        // clip Y vector versus band boundaries
        y_start = std::max (yref, ypos);
        y_end = std::min (ylen, ypos + bstep);

        if (y_start >= y_end)
            break;

        y_len = y_end - y_start;
        bp_start = bp + y_start - ypos;
        bottom_in = (to_first && y_start == yref);
        add_score = 0;
        if (bottom_in)
        {
            // ugly and heavy, optimize later. Also : only works for restricted bands passed from seg_align, not useful for generic search.
            if ((xpos - xref) > (width - width_right))
                add_score = -gip - ((xpos - xref) - (width - width_right)) * gep;
        }

        assert (!last_col);
        if (len  <= 0  || xpos + 1 == xlen || std::max (yref, ypos+1) >= std::min (ylen, ypos + 1 + bstep))
            last_col = true;

        align_y_loop (ap + y_start, xseq [xpos], bp_start, xpos, xhomo [xpos - xref + 1], y_start, y_len, bottom_in, add_score, last_col);

        // record max score at the top of those columns that end at the band edge
        // 02/28/2017: with the change of semantic of firstx / lastx, the lastx means alignment should terminate at the edge of X sequence. The code below terminates at Y edge => commented
        // if (y_end == ylen && last_w <= cur_w)
        //     last_w = cur_w, last_bp = bp_start + y_len - 1, last_x = xpos, last_y = y_end-1, last_reached = true;

        xpos++, ypos++, bp += bstep;
    }

    // force diagonal offset for backtrace matrix
    bstep--;

    if (logp_)
        (*logp_) << std::endl;

    if (to_last)
        return get_last_score ();
    else
        return get_score ();
}

/*
follows backtrace matrix, fills BATCH array, returns number of batches
*/
unsigned ContAlign::backtrace (BATCH *b_ptr, int max_cnt, unsigned width)
{
    if (!to_last  && !max_reached)
        return 0; // this is correct case if alignment area size is zero (due to task or clipping)
    if (to_last && !last_reached)
        return 0;

    char* bp = to_last ? last_bp : max_bp;  // in case of toend, bp is pointing to the upper-right cell in matrix
    int state = *bp & 3;

    int x = to_last ? last_x : max_x;
    int y = to_last ? last_y : max_y;

    int b_len = 1, b_cnt = 0;
    int lowest_y = std::max (0, (int) yref - (int) width);
    BATCH* b_start = b_ptr;
    bool before_first_batch = true;
    int next_state;

    // backtrace from highest score to:

    bool done = false;
    do
    {
        switch (state)
        {
            //follow v-trace down until ALIGN_VSKIP flag set
            case ALIGN_DOWN:
                if (to_last && before_first_batch) // add zero-length segment at the very end of alignment - for scoring purposes
                {
                    assert (b_cnt < max_cnt);
                    b_ptr->xpos = x+1;
                    b_ptr->ypos = y+1;
                    b_ptr->len  = 0;
                    ++b_ptr;
                    ++b_cnt;
                }
                y--;
                if (y >= yref)
                {
                    --bp;
                    if (*bp & ALIGN_VSKIP)
                        state = *bp & 3;
                }
                break;

            //follow h-trace left until ALIGN_HSKIP flag set
            case ALIGN_LEFT:
                --x;
                if (x >= xref)
                {
                    bp -= bstep;
                    if (*bp & ALIGN_HSKIP)
                        state = *bp & 3;
                }
                break;

            //follow diagonal until best score is achieved from v-trace or h-trace
            case ALIGN_DIAG:
                if (x > xref && y > yref)
                {
                  bp -= bstep + 1;
                  next_state = (*bp & 3);
                  if (state != next_state)
                  {
                    state = *bp & 3;
                    assert (b_cnt < max_cnt);
                    ++b_cnt;
                    if (next_state != ALIGN_STOP)
                    {
                        b_ptr->xpos = x;
                        b_ptr->ypos = y;
                        b_ptr->len = b_len;
                    }
                    else
                    {
                        b_ptr->xpos = x-1;
                        b_ptr->ypos = y-1;
                        b_ptr->len = b_len+1;
                    }
                    ++b_ptr;
                    b_len = 0;
                  }
                }
                b_len ++;
                x --, y --;
                break;

            //end of alignment (w[x][y] was set to 0)
            case ALIGN_STOP:
                if (!to_first)
                    done = true;
                break;
        }
        before_first_batch = false;
    }
    while (!done && x >= xref && y >= lowest_y);

    //if alignment ends at the edge of the matrix we get here
    if (state == ALIGN_DIAG)
    {
        assert (b_cnt < max_cnt);
        ++b_cnt;
        b_ptr->xpos = x+1;
        b_ptr->ypos = y+1;
        b_ptr->len = b_len-1;
        ++b_ptr;
    }
    // in to_first mode, if y is not fully covered, add pseudo segment and a gap
    // 02/28/2017: seem to be not needed as only X coverage is now required for to_first
    /*
    if (to_first && y >= yref)
    {
        assert (b_cnt < max_cnt);
        b_ptr->xpos = x+1;
        b_ptr->ypos = yref;
        b_ptr->len  = 0;
        ++b_cnt;
    }
    */
    reverse_inplace<BATCH> (b_start, b_cnt);
    return b_cnt;
}

void ContAlign::set_log (int posix_handle)
{
    reset_log ();
    if (posix_handle >= 0)
    {
        logbuf_ = new __gnu_cxx::stdio_filebuf<char> (posix_handle, std::ios::out);
        own_log_ = new std::ostream (logbuf_);
        logp_ = own_log_;
    }
}
void ContAlign::set_log (std::ostream& log)
{
    reset_log ();
    logp_ = &log;
}
void ContAlign::reset_log ()
{
    delete own_log_;
    own_log_ = NULL;
    delete logbuf_;
    logbuf_ = NULL;
    logp_ = NULL;
}
