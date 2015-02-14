/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "align_batch.h"
#include <cassert>

int align_score (const BATCH* bm, unsigned bno, const char* xseq, const char* yseq, int gip, int gep, int mat, int mis)
{
    int score = 0, gap;
    const char* xp, *yp, *xs;
    for (const BATCH* sent = bm + bno, *prevb = NULL; bm != sent; prevb = bm, ++bm)
    {
        if (prevb)
        {
            assert (bm->xpos >= prevb->xpos + prevb->len);
            gap = bm->xpos - (prevb->xpos + prevb->len);
            if (gap)
                score -= gip + gep*gap;
            assert (bm->ypos >= prevb->ypos + prevb->len);
            gap = bm->ypos - (prevb->ypos + prevb->len);
            if (gap)
                score -= gip + gep*gap;
        }
        for (xp = xseq + bm->xpos, yp = yseq + bm->ypos, xs = xp + bm->len; xp != xs; ++xp, ++yp)
            score += (*xp == *yp) ? mat : mis;
    }
    return score;
}
