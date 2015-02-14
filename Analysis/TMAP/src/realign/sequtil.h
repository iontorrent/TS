/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef SEQUTIL_H
#define SEQUTIL_H


extern const char number2base [];


inline char base2char (unsigned b)
{
    if (b > 4)
        b = 0;
    return number2base [b];
}
inline unsigned get_base (const char *seq, unsigned pos)
{
    return (seq[pos >> 2] >> ((pos & 3) << 1)) & 3;
}

#endif // SEQUTIL_H
