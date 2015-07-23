/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <limits>
#include <rerror.h>
#include <tracer.h>
#include <myassert.h>
#include "sequtil.h"
#include "nalign.h"

/*
member inner loops used for finding alignment
fill backtrace matrix and store max score position
*/

int Align::align_y_loop (register ALIGN_VECT* ap, unsigned char base, char* bp, int x, int y, int len, bool bottom_in, int add_score, bool last_col)
{
    //initialise bottom boundary
    int prev_w, save_w;
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

    int v = low_score; // works for both to_first and normal mode: just guarantees that cost of coming from the bottom is higher then by diagonal
    int w, pw = prev_w;

#ifdef DEBUG_TRACE
    if (trace_mode)
    {
        trclog << "\n";
        trclog << std::setw (6) << std::right << x << " " << (char) ((base < 4) ? base2char (base) : base) << " ";
        if (y > yref)
            trclog << std::setw (6 * (y-yref)) << std::left << " ";
        trclog << std::flush;
    }
#endif
    if (logp_)
    {
        (*logp_) << "\n";
        (*logp_) << std::setw (6) << std::right << x << " " << (char) ((base < 4) ? base2char (base) : base) << " ";
        if (y > yref)
            (*logp_) << std::setw (6 * (y-yref)) << std::left << " ";
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
        w -= gip;
        if (w > ap->h)
            ap->h = w, dir |= ALIGN_HSKIP;
        ap->h -= gep;

        //v = max (w - gip, v) - gep;
        if (w > v)
            v = w, dir |= ALIGN_VSKIP;
        v -= gep;

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
            trclog << std::setw (4) << std::left << save_w;
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
            (*logp_) << std::setw (4) << std::left << prev_w;
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


Align::Align ()
:
btrmx (NULL),
ap (NULL)
{
}


Align::Align (int max_ylen, int max_size, int gip, int gep, int mat, int mis)
:
btrmx (NULL),
ap (NULL),
trace_mode (false),
logp_ (NULL)
{
    init (max_ylen, max_size, gip, gep, mat, mis);
}


Align::~Align ()
{
    delete [] ap;
    delete [] btrmx;
}

void Align::init (int max_ylen, int max_size, int gip, int gep, int mat, int mis)
{
    if (ap) delete [] ap;
    if (btrmx) delete [] btrmx;

    Align::max_ylen = max_ylen;
    Align::max_size = max_size;
    Align::gip = gip;
    Align::gep = gep;
    Align::mis = mis;
    Align::mat = mat;
    Align::trace_mode = false;
    Align::logp_ = NULL;
    Align::low_score = -gip - std::max (max_ylen, max_size / max_ylen) * (gep - mis) - 1;

    //allocate running Y-vector and backtrace matrix
    btrmx = new char [max_size];
    std::fill (btrmx, btrmx + max_size, ALIGN_STOP);
    ap = new ALIGN_VECT [max_ylen];
}

void Align::set_scoring ( int gip, int gep, int mat, int mis)
{
    Align::gip = gip;
    Align::gep = gep;
    Align::mis = mis;
    Align::mat = mat;
    Align::low_score = -gip - std::max (max_ylen, max_size / max_ylen) * gep - 1;
}
/*
calculates best local alignment between pair of nucleotide sequences
returns maximum local alignment score
*/

int Align::align (const char* xseq, int xlen, const char* yseq, int ylen, bool unpack)
{
    int x, y;
    char* bp = btrmx;
    max_reached = false;
    last_reached = false;
    accum_hor_skip = 0;
    to_first = false, to_last = false;

    //check if enough memory allocated for band alignment
    if (ylen > max_ylen || xlen * ylen > max_size)
        ers << "align error: attempting to align sequences longer than declared, xlen = " << xlen << ", ylen = "  << ylen << Throw;

    bstep = ylen;
    xref = 0, yref = 0;

    //initialize left boundary
    //unpack Y sequence for faster processing
    for (y = 0; y < ylen; y++)
    {
        ap[y].w = 0;
        ap[y].h = low_score;
        ap[y].r = unpack? get_base (yseq, y) : yseq [y];
    }

    //find best local alignment
    int cur_w;
    last_w = std::numeric_limits <int>::min ();
    max_w = std::numeric_limits <int>::min ();
    max_bp = btrmx;
    for (x = 0; x < xlen; x++, bp += bstep)
    {
        cur_w = align_y_loop (ap, unpack ? get_base (xseq, x) : xseq [x], bp, x, 0, ylen, true, 0);
        // remember X edge terminal scores
        if (last_w < cur_w)
            last_w = cur_w, last_bp = bp + ylen - 1, last_x = x, last_y = ylen-1, last_reached = true;
    }

#ifdef DEBUG_TRACE
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
int Align::align_band (const char* xseq, int xlen, const char* yseq, int ylen, int xpos, int ypos, int len, int width, bool unpack, int width_right, bool tobeg, bool toend)
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


    // initialize left boundary
    // unpack Y sequence for faster processing
    int ylast = std::min (ylen, yref + len);

#ifdef DEBUG_TRACE
    if (trace_mode)
    {
        trclog << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            trclog << std::setw (6) << std::left << yy;
        trclog << "\n";
        trclog << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            trclog << std::setw (6) << std::left << (char) (unpack? get_base (yseq, yy) : yseq [yy]);
        trclog << "\n";
        trclog << std::setw (9) << "";
    }
#endif
    if (logp_)
    {
        (*logp_) << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            (*logp_) << std::setw (6) << std::left << yy;
        (*logp_) << "\n";
        (*logp_) << std::setw (9) << "";
        for (int yy = yref; yy != ylast; ++yy)
            (*logp_) << std::setw (6) << std::left << (char) (unpack? get_base (yseq, yy) : yseq [yy]);
        (*logp_) << "\n";
        (*logp_) << std::setw (9) << "";
    }

    // for (int i = std::max (0, yref - width); i < ylast; i++)
    // changed from above line to always have the valid prev_w when y > 0
    int curw = -gip-gep;
    // for (int i = max_ (0, yref - width - 1); i < ylast; i++)
    for (int i = yref; i < ylast; i++)
    {
        ap [i].w = curw;
#ifdef DEBUG_TRACE
        if (trace_mode)
            trclog   << " " << std::left << std::setw (5) << std::left << curw;
#endif
        if (logp_)
            (*logp_) << " " << std::left << std::setw (5) << std::left << curw;

        curw -= gep;
        ap[i].h = low_score; // guaranteed to always prevent 'coming from the right'. Works for both tobeg and not.
        ap[i].r = unpack? get_base (yseq, i) : yseq [i];
    }

    // find best local alignment, save backtrace pointers
    last_w = std::numeric_limits <int>::min ();
    max_w = std::numeric_limits <int>::min ();
    max_bp = btrmx, ypos -= width;

    int y_start, y_end, y_len; //, y;
    char *bp_start;
    int cur_w;
    bool bottom_in = false;
    int add_score;

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

        cur_w = align_y_loop (ap + y_start, unpack ? get_base (xseq, xpos) : xseq [xpos], bp_start, xpos, y_start, y_len, bottom_in, add_score, last_col);

        // in case of toend, record every column top, (so that last one will remain after loop end) [checking if loop will end in next iteration is too cumbersome]
        //if (last_w <= cur_w)
            last_w = cur_w, last_bp = bp_start + y_len - 1, last_x = xpos, last_y = y_end-1, last_reached = true;

        xpos++, ypos++, bp += bstep;
    }

    // force diagonal offset for backtrace matrix
    bstep--;

#ifdef DEBUG_TRACE
    trclog << std::endl;
#endif
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
unsigned Align::backtrace (BATCH *b_ptr, int max_cnt, bool reverse, unsigned width)
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
                bp -= bstep + 1;
                next_state = (*bp & 3);
                if (x > xref && y > yref && state != next_state)
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
