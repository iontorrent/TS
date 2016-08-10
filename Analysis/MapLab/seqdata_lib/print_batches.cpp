/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#include <climits>
#include <iomanip>
#include <runtime_error.h>
#include <myassert.h>

#include "sequtil.h"

#include "print_batches.h"

std::ostream& operator << (std::ostream& ostr, const PRAL& pa)
{
    for (BATCH *b = pa.b_, *sent = pa.b_+pa.bno_; b != sent; ++b)
    {
        if (b != pa.b_) ostr << " ";
        ostr << pa.xoff_ + b->xpos << ":" << pa.yoff_ + b->ypos << ":" << b->len;
    }
    return ostr;
}



#define MAXSTRLEN 256
#define NUMSTRLEN 11

void print_batches (const char* xseq, unsigned xlen, bool xrev, const char* yseq, unsigned ylen, bool yrev, const BATCH *b_ptr, int b_cnt, std::ostream& stream, bool unpack, unsigned xoff, unsigned yoff, unsigned margin, unsigned width, bool zero_based)
{
    unsigned slen = 0;
    unsigned blen = 0;
    unsigned x = b_ptr->xpos;
    unsigned y = b_ptr->ypos;
    unsigned xstart = x;
    unsigned ystart = y;
    unsigned char xc, yc;
    char s[3][MAXSTRLEN];

    myassert (width < MAXSTRLEN);

    stream << std::setiosflags (std::ios::left);
    while (b_cnt > 0)
    {
        xc = unpack ? base2char (get_base (xseq, x)) : xseq [x];
        yc = unpack ? base2char (get_base (yseq, y)) : yseq [y];

        // special handling for (x < b_ptr->xpos && y < b_ptr->ypos)
        // treating as batch with special match symbols

        if (x < b_ptr->xpos && y < b_ptr->ypos)
        {
            s[0][slen] = xc;
            s[2][slen] = yc;
            s[1][slen] = '#';
            x++, y++, slen++;
        }
        // X insert
        else if (x < b_ptr->xpos)
        {
            s[0][slen] = xc;
            s[1][slen] = ' ';
            s[2][slen] = '-';
            x++, slen++;
        }
        // Y insert
        else if (y < b_ptr->ypos)
        {
            s[0][slen] = '-';
            s[1][slen] = ' ';
            s[2][slen] = yc;
            y++, slen++;
        }
        // emit text batch
        else if (blen < b_ptr->len)
        {
            s[0][slen] = xc;
            s[2][slen] = yc;
            s[1][slen] = (toupper (xc) == toupper (yc) || toupper (xc) == 'N' || toupper (yc) == 'N') ? '*' : ' ';
            // s[1][slen] = (xc == yc || xc == 'N' || yc == 'N') ? '*' : ' ';
            x++, y++, slen++, blen++;
        }
        else
            blen = 0, b_cnt--, b_ptr++;

        //print accumulated lines
        if ((slen + NUMSTRLEN > width) || b_cnt <= 0)
        {
            //null terminate all strings
            for (int i = 0; i < 3; i++)
                s[i][slen] = 0;

            unsigned xdisp = (xrev ? xlen - xstart - 1 : xstart) + xoff + (zero_based ? 0 : 1);
            unsigned ydisp = (yrev ? ylen - ystart - 1 : ystart) + yoff + (zero_based ? 0 : 1);
            stream << "\n" << std::setw (margin) << "" << std::setw (NUMSTRLEN) << xdisp  << std::setw (0) << " " << s[0];
            stream << "\n" << std::setw (margin) << "" << std::setw (NUMSTRLEN) << " "    << std::setw (0) << " " << s[1];
            stream << "\n" << std::setw (margin) << "" << std::setw (NUMSTRLEN) << ydisp  << std::setw (0) << " " << s[2];
            stream << "\n";

            xstart = x, ystart = y, slen = 0;
        }
    }
    stream << std::flush;
}

static unsigned decout (unsigned num, char* dest, unsigned destlen) // returns 0 on failure, number of chars on success
{
    unsigned pp = num;
    unsigned decposno = 1;
    while ((pp /= 10)) decposno ++;
    unsigned rval = decposno;
    if (decposno >= destlen) return 0;
    do
    {
        dest [--decposno] = char ('0' + num % 10);
    }
    while ((num /= 10));
    return rval;
}

int print_batches_cigar (const BATCH *b_ptr, int b_cnt, char* dest, unsigned destlen)
{
    unsigned dpos = 0, pl;
    int curb = 0;
    const BATCH *pb = NULL;
    myassert (destlen > 1);
    while (curb <= b_cnt)
    {
        if (pb)
        {
            pl = decout (pb->len, dest + dpos, destlen - dpos);
            if (!pl)
                break;
            dpos += pl;
            if (dpos == destlen - 1)
                break;
            dest [dpos++] = 'M';
            if (curb < b_cnt)
            {
                if (pb->xpos + pb->len < b_ptr->xpos) // skip on x (subject) == gap on y (query)
                {
                    pl = decout (b_ptr->xpos - (pb->xpos + pb->len), dest + dpos, destlen - dpos);
                    if (!pl)
                        break;
                    dpos += pl;
                    if (dpos == destlen - 1)
                        break;
                    dest [dpos++] = 'I';
                }
                if (pb->ypos + pb->len < b_ptr->ypos) // skip on y (query) == gap on x (subject)
                {
                    pl = decout (b_ptr->ypos - (pb->ypos + pb->len), dest + dpos, destlen - dpos);
                    if (!pl)
                        break;
                    dpos += pl;
                    if (dpos == destlen - 1)
                        break;
                    dest [dpos++] = 'D';
                }
            }
        }
        pb = b_ptr;
        b_ptr ++;
        curb ++;
    }
    dest [dpos] = 0;
    return dpos;
}
