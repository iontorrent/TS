/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __sequtil_h__
#define __sequtil_h__

// #ifndef __sequtil_cpp__
extern const char number2base [];
// #endif 

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



#endif // __sequtil_h__
