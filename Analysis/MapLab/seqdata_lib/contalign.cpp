/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <limits>
#include <rerror.h>
#include <tracer.h>
#include <myassert.h>
#include "sequtil.h"
#include "contalign.h"

/*
member inner loops used for finding alignment
fill backtrace matrix and store max score position
*/

double ContAlign::align_y_loop (register ALIGN_FVECT* ap, unsigned char base, char* bp, int x, int xdivisor, int y, int len, bool bottom_in, double add_score, bool last_col)
{
    //initialise bottom boundary
    double prev_w, save_w;
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

    double v = low_score; // works for both to_first and normal mode: just guarantees that cost of coming from the bottom is higher then by diagonal
    double w, pw = prev_w, hw, vw;

#ifdef DEBUG_TRACE
    if (trace_mode)
    {
        trclog << "\n";
        trclog << std::setw (6) << std::right << x << " " << (char) ((base < 4) ? base2char (base) : base) << " ";
        if (y > yref)
            trclog << std::setw (7 * (y-yref)) << std::left << " ";
        trclog << std::flush;
    }
#endif
    if (logp_)
    {
        (*logp_) << "\n";
        (*logp_) << std::setw (6) << std::right << x << " " << (char) ((base < 4) ? base2char (base) : base) << " ";
        if (y > yref)
            (*logp_) << std::setw (7 * (y-yref)) << std::left << " ";
        (*logp_) << std::flush;
    }

    y += len - 1;
    while (len-- > 0)
    {
        char dir = ALIGN_DIAG;

        //w = max (0, h, v, prev_w + s(x, y))
        w = prev_w + ((ap->r == base || ap->r == 'N') ? mat : mis); // HACK for unpacked variant

        if (w < v)
            w = v, dir = ALIGN_DOWN;

        if (w < ap->h)
            w = ap->h, dir = ALIGN_LEFT;

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
        // --- ?? --- NO! semiglobal is global by Y sequence, so premature termination before reaching maxx Y is forbidden
        // - what a stupid notion! there surely can be a gap at the end!
        // - the notion is not so stupid.
        if (last_col && w >= last_w)
            last_w = w, last_bp = bp, last_x = x, last_y = y - len, last_reached = true;

        //save w[x][y] for next pass
        prev_w = ap->w;
        ap->w = pw = w;

        //h = max (w - gip, h) - gep;
        hw = w - ((scale_type_ == SCALE_GIP_GEP) ? (gip / xdivisor) : gip);
        if (hw > ap->h)
            ap->h = hw, dir |= ALIGN_HSKIP;
        ap->h -= (scale_type_ == SCALE_NONE) ? gep : (gep/xdivisor);

        //v = max (w - gip, v) - gep;
        vw = w - ((scale_type_ == SCALE_GIP_GEP) ? (gip/ap->div) : gip);
        if (vw > v)
            v = vw, dir |= ALIGN_VSKIP;
        v -= (scale_type_ == SCALE_NONE) ? gep : gep/ap->div;

#ifdef DEBUG_TRACE
        if (trace_mode)
        {
            switch (dir&3)
            {
                case ALIGN_DOWN: trclog << "-"; break;
                case ALIGN_LEFT: trclog << "|"; break;
                case ALIGN_DIAG: trclog << "\\"; break;
                case ALIGN_STOP: trclog << "#"; break;
            }
            if (dir & ALIGN_VSKIP)
                trclog << ((dir & ALIGN_ZERO) ? "V" : "v");
            else if (dir & ALIGN_HSKIP)
                trclog << ((dir & ALIGN_ZERO) ? "H" : "h");
            else
                trclog << ((dir & ALIGN_ZERO) ? "o" : " ");;
            trclog << std::setw (5) << std::left  << std::fixed << std::setprecision (1) << save_w;
        }
#endif
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
            else if (dir & ALIGN_HSKIP)
                (*logp_) << ((dir & ALIGN_ZERO) ? "H" : "h");
            else
                (*logp_) << ((dir & ALIGN_ZERO) ? "o" : " ");;
            (*logp_) << std::setw (5) << std::left  << std::fixed << std::setprecision (1) << prev_w;
        }

        //save bactrace pointer (4bits / byte)
        *bp++ = dir;
        ap++;
    }
#ifdef DEBUG_TRACE
    if (trace_mode)
    {
        trclog << "(" << y-len << ")" << std::flush;
    }
#endif
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
trace_mode (false),
logp_ (NULL),
scale_type_ (SCALE_GEP)
{
}


ContAlign::ContAlign (int max_ylen, int max_xlen, int max_size, int gip, int gep, int mat, int mis)
:
btrmx (NULL),
ap (NULL),
xhomo (NULL),
yhomo (NULL),
trace_mode (false),
logp_ (NULL),
scale_type_ (SCALE_GEP)
{
    init (max_ylen, max_xlen, max_size, gip, gep, mat, mis);
}


ContAlign::~ContAlign ()
{
    delete [] ap;
    delete [] btrmx;
    delete [] xhomo;
    delete [] yhomo;
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
    ContAlign::trace_mode = false;
    ContAlign::logp_ = NULL;
    ContAlign::low_score = -gip - std::max (max_ylen, max_size / max_ylen) * (gep - mis) - 1;

    //allocate running Y-vector and backtrace matrix
    btrmx = new char [max_size];
    std::fill (btrmx, btrmx + max_size, ALIGN_STOP);
    ap = new ALIGN_FVECT [max_ylen];
    trclog << "ap allocated at " << (void*) ap << " for " << max_ylen *sizeof (ALIGN_FVECT) << " bytes\n";
    yhomo = new char [max_ylen+1];
    trclog << "yhomo allocated at " << (void*) yhomo << " for " << max_ylen + 1 << " bytes\n" << std::flush;
    xhomo = new char [max_xlen+1];
}

void ContAlign::set_scoring ( int gip, int gep, int mat, int mis)
{
    ContAlign::gip = gip;
    ContAlign::gep = gep;
    ContAlign::mis = mis;
    ContAlign::mat = mat;
    ContAlign::low_score = -gip - std::max (max_ylen, max_size / max_ylen) * gep - 1;
}
/*
calculates best local alignment between pair of nucleotide sequences
returns maximum local alignment score
*/

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

double ContAlign::align (const char* xseq, int xlen, const char* yseq, int ylen, bool unpack)
{
    int x, y;
    char* bp = btrmx;
    max_reached = false;
    last_reached = false;
    accum_hor_skip = 0;
    to_first = false, to_last = false;

    //check if enough memory allocated for band alignment
    if (ylen > max_ylen || xlen * ylen > max_size || xlen > max_xlen)
        ers << "align error: attempting to align sequences longer than declared, xlen = " << xlen << ", ylen = "  << ylen << Throw;

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
        ap[y].r = unpack? get_base (yseq, y) : yseq [y];
        ap[y].div = yhomo [y+1];
    }

    //find best local alignment
    double cur_w;
    max_w = last_w = - std::numeric_limits <double>::max ();
    max_bp = btrmx;
    for (x = 0; x < xlen; x++, bp += bstep)
    {
        cur_w = align_y_loop (ap, unpack ? get_base (xseq, x) : xseq [x+1], bp, x, xhomo [x], 0, ylen, true, 0.0);
        // remember X edge terminal scores
        if (last_w < cur_w)
            last_w = cur_w, last_bp = bp + ylen - 1, last_x = x, last_y = ylen-1, last_reached = true;
    }

#ifdef DEBUG_TRACE
    if (trace_mode)
        trclog << std::endl;
#endif
    if (logp_)
        (*logp_) << std::endl;

    return get_score ();
}



/*
calculates best local alignment between pair of nucleotide sequences
on a diagonal band (len, diag +- width)
returns maximum local alignment score
NOTE: batch xpos, ypos, len should be inside X and Y sequences, width > 0

*/
double ContAlign::align_band (const char* xseq, int xlen, const char* yseq, int ylen, int xpos, int ypos, int len, int width, bool unpack, int width_right, bool tobeg, bool toend)
{
    char* bp = btrmx;
    max_reached = false;
    last_reached = false;
    to_first = tobeg, to_last = toend;
    accum_hor_skip = 0;

    // check if enough memory allocated for band alignment
    // and if batch variables are sane
    if (ylen > max_ylen || (max_size > 0 && len * width > max_size))
        ers << "align error: attempting to batch-align sequences longer than declared, max_ylen = " << max_ylen << ", max_size = " << max_size << ", ylen = " << ylen << ", len = "  << len << ", width = " << width << Throw;

    if (width_right == -1)
        width_right = width;

    bstep = width + 1 + width_right;
    xref = xpos, yref = ypos;

    // initialize homopolimer size indices
    fill_homo_tract_len (xseq + xref, xlen - xref, xhomo);
    fill_homo_tract_len (yseq + yref, ylen - yref, yhomo);

    // initialize left boundary
    // unpack Y sequence for faster processing
    int ylast = std::min (ylen, yref + len);

#ifdef DEBUG_TRACE
    if (trace_mode)
    {
        trclog << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            trclog << std::setw (7) << std::left << yy;
        trclog << "\n";
        trclog << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            trclog << std::setw (7) << std::left << (char) (unpack? get_base (yseq, yy) : yseq [yy]);
        trclog << "\n";
    }
#endif
    if (logp_)
    {
        (*logp_) << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            (*logp_) << std::setw (7) << std::left << yy;
        (*logp_) << "\n";
        (*logp_) << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            (*logp_) << std::setw (7) << std::left << (char) (unpack? get_base (yseq, yy) : yseq [yy]);
        (*logp_) << "\n";
    }

    // for (int i = std::max (0, yref - width); i < ylast; i++)
    // changed from above line to always have the valid prev_w when y > 0
    double curw = -gip-gep;
    // for (int i = max_ (0, yref - width - 1); i < ylast; i++)
#if DEBUG_TRACE
    trclog << "Initializing ap[i]->h for i between " << yref << " and " << ylast << " to " << low_score << "\n" << std::setw (9) << ""; 
#endif
    if (logp_)
        (*logp_) << "Initializing ap[i]->h for i between " << yref << " and " << ylast << " to " << low_score << "\n" << std::setw (9) << "";
    for (int i = yref; i < ylast; i++)
    {
        ap [i].w = curw;
#ifdef DEBUG_TRACE
        if (trace_mode)
            trclog   << " " << std::left << std::setw (6) << std::left  << std::fixed << std::setprecision (1) << curw;
#endif
        if (logp_)
            (*logp_) << " " << std::left << std::setw (6) << std::left  << std::fixed << std::setprecision (1) << curw;

        curw -= gep;
        ap[i].h = low_score; // guaranteed to always prevent 'coming from the right'. Works for both tobeg and not.
        ap[i].r = unpack? get_base (yseq, i) : yseq [i];
        ap[i].div = yhomo [i - yref + 1];
    }

    // find best local alignment, save backtrace pointers
    last_w = max_w = - std::numeric_limits <double>::max ();
    max_bp = btrmx, ypos -= width;

    int y_start, y_end, y_len; //, y;
    char *bp_start;
    double cur_w;
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

        myassert (!last_col);
        if (len  <= 0  || xpos + 1 == xlen || std::max (yref, ypos+1) >= std::min (ylen, ypos + 1 + bstep))
            last_col = true;

        cur_w = align_y_loop (ap + y_start, unpack ? get_base (xseq, xpos) : xseq [xpos], bp_start, xpos, xhomo [xpos - xref + 1], y_start, y_len, bottom_in, add_score, last_col);

        // in case of toend, record every column top, (so that last one will remain after loop end) [checking if loop will end in next iteration is too cumbersome]
        if (y_end == ylen && last_w <= cur_w)
            last_w = cur_w, last_bp = bp_start + y_len - 1, last_x = xpos, last_y = y_end-1, last_reached = true;

        xpos++, ypos++, bp += bstep;
    }

    // force diagonal offset for backtrace matrix
    bstep--;

#ifdef DEBUG_TRACE
    if (trace_mode)
    {
        trclog << "\nmax_reached: " << (max_reached?TRUE_STR:FALSE_STR) << "\n";
        if (max_reached)
            trclog << "  max_w = " << last_w << "\n" << "  max_x = " << max_x << "\n" << "  max_y = " << max_y << "\n";
        if (to_last)
        {
            trclog << "last_reached: " << (last_reached?TRUE_STR:FALSE_STR) << "\n";
            if (last_reached)
                trclog << "  last_w = " << last_w << "\n" << "  last_x = " << last_x << "\n" << "  last_y = " << last_y << "\n";
        }
        trclog << std::endl;
    }
#endif
    if (logp_)
    {
        (*logp_) << "max_reached: " << max_reached << "\n";
        if (max_reached)
            (*logp_) << "  max_w = " << last_w << "\n" << "  max_x = " << max_x << "\n" << "  max_y = " << max_y << "\n";
        if (to_last)
        {
            (*logp_) << "last_reached: " << last_reached << "\n";
            if (last_reached)
                (*logp_) << "  last_w = " << last_w << "\n" << "  last_x = " << last_x << "\n" << "  last_y = " << last_y << "\n";
        }
        (*logp_) << std::endl;
    }

    if (to_last)
        return get_last_score ();
    else
        return get_score ();
}

/*
follows backtrace matrix, fills BATCH array, returns number of batches
*/
unsigned ContAlign::backtrace (BATCH *b_ptr, int max_cnt, bool reverse, unsigned width)
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
                    if (b_cnt == max_cnt)
                        ers << "Not enough space to store all batches (>" << b_cnt-1 << ") of the alignment" << Throw;
                    b_ptr->xpos = x+1;
                    b_ptr->ypos = y+1;
                    b_ptr->len  = 0;
                    b_ptr += (reverse ? -1 : 1);
                    b_cnt ++;
                }
                y--;
                if (y >= yref)
                {
                    bp--;
                    if (*bp & ALIGN_VSKIP)
                        state = *bp & 3;
                }
                break;

            //follow h-trace left until ALIGN_HSKIP flag set
            case ALIGN_LEFT:
                x--;
                if (x >= xref)
                {
                    bp -= bstep;
                    if (*bp & ALIGN_HSKIP)
                        state = *bp & 3;
                }
                break;

            //follow diagonal until best score is achieved from v-trace or h-trace
            case ALIGN_DIAG:
                if (x > xref && y > yref )
                {
                    bp -= bstep + 1;
                    next_state = (*bp & 3);
                    if (state != next_state)
                    {
                        state = *bp & 3;
                        if (b_cnt == max_cnt)
                            ers << "Not enough space to store all batches (>" << b_cnt-1 << ") of the alignment" << Throw;
                        b_cnt ++;
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
                        b_ptr += (reverse ? -1 : 1);
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
        if (b_cnt == max_cnt)
            ers << "Not enough space to store all batches (>" << b_cnt-1 << ") of the alignment" << Throw;
        b_cnt++;
        b_ptr->xpos = x+1;
        b_ptr->ypos = y+1;
        b_ptr->len = b_len-1;
        b_ptr += (reverse ? -1 : 1);
    }
    // in to_first mode, if y is not fully covered, add pseudo segment and a gap
    if (to_first && y >= yref)
    {
        if (b_cnt == max_cnt)
            ers << "Not enough space to store all batches (>" << b_cnt-1 << ") of the alignment" << Throw;
        b_ptr->xpos = x+1;
        b_ptr->ypos = yref;
        b_ptr->len  = 0;
        b_cnt ++;
    }

    if (!reverse) reverse_inplace<BATCH> (b_start, b_cnt);
    return b_cnt;
}
