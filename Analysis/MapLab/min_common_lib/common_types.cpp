/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#include "common_types.h"
#include <algorithm>

char* lltoa (longlong val, char* buf, int base)
{
    // characters are 0-9A-Z
    int sign = 1;
    if (val < 0) val = -val, sign = -1;
    int pos = 0;
    do
    {
        char c = (char) (val % base);
        if (c <= 9) c += '0';
        else c += 'A';
        buf [pos ++] = c;
        val = val / base;
    }
    while (val > 0);
    if (sign == -1) buf [pos++] = '-';
    buf [pos] = 0;
    std::reverse (buf, buf + pos);
    return buf;
}

char* ulltoa (ulonglong val, char* buf, int base)
{
    // characters are 0-9A-Z
    int pos = 0;
    do
    {
        char c = (char) (val % base);
        if (c <= 9) c += '0';
        else c += 'A';
        buf [pos ++] = c;
        val = val / base;
    }
    while (val > 0);
    buf [pos] = 0;
    std::reverse (buf, buf + pos);
    return buf;
}

ulonglong atoull (const char* strval)
{
    ulonglong rv = 0;
    // skip leading whitespace
    while (*strval == ' ' || *strval == '\t') strval ++;
    // convert string to int
    while (*strval != '\0')
    {
        if (*strval >= '0' && *strval <= '9')
            rv = rv * 10 + *strval++ - '0';
        strval ++;
    }
    return rv;
}
