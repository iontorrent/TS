/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "trace_matrix.h"
#include <iomanip>

namespace genstr
{

static const char* prefix = "      ";

static void print_axis_pg (std::ostream& o, int dim)
{
    o << prefix;
    for (int p = 0; p < dim; p ++)
        if (p % 5 == 0) o << '+';
        else o << '|';
    o << std::endl;
}
static void print_axis_nums (std::ostream& o, int dim)
{
    o << prefix;
    for (int p = 0; p < dim; p += 10)
        o << std::setw (10) << std::left << p;
    o << std::endl;
}

static void print_trace_matrix_axis (std::ostream& o, int dim, bool beg)
{
    if (beg)
    {
        print_axis_nums (o, dim);
        print_axis_pg (o, dim);
    }
    else
    {
        print_axis_pg (o, dim);
        print_axis_nums (o, dim);
    }
}

void TraceMatrix<false>
::print (unsigned p1, unsigned p2, std::ostream& o)
{
    o << std::endl;
    print_trace_matrix_axis (o, dim2_, true);

    for (unsigned pos1 = 0; pos1 < dim1_; pos1 ++)
    {
        o << std::setw (6) << std::left << pos1 << std::setw (0);
        for (unsigned pos2 = 0; pos2 < dim2_; pos2 ++)
        {
            if (p1 == pos1 && p2 == pos2)
                o << '*';
            else
            {
                char c = ' ';
                switch (get (pos1, pos2))
                {
                case ALONG_FIRST:   c = '|'; break;
                case ALONG_SECOND:  c = '-'; break;
                case ALONG_DIAG:    c = '\\'; break;
                default:            c = ' ';
                }
                o << c;
            }
        }
        o << std::endl;
    }
    print_trace_matrix_axis (o, dim2_, false);
}

void TraceMatrix<true>
::print (unsigned p1, unsigned p2, std::ostream& o)
{
    o << std::endl;
    print_trace_matrix_axis (o, len_, true);

    for (unsigned pos1 = 0; pos1 < len_; pos1 ++)
    {
        o << std::setw (6) << std::left << pos1 << std::setw (0);
        for (unsigned pos2 = 0; pos2 < len_; pos2 ++)
        {
            if (p1 == pos1 && p2 == pos2)
                o << '*';
            else
            {
                char c = ' ';
                try
                {
                    switch (get (pos1, pos2))
                    {
                        case ALONG_FIRST:   c = '|'; break;
                        case ALONG_SECOND:  c = '-'; break;
                        case ALONG_DIAG:    c = '\\'; break;
                        default:            c = ' ';
                    }
                }
                catch (OutOfBand&)
                {
                    c = '.';
                }
                o << c;
            }
        }
        o << std::endl;
    }
    print_trace_matrix_axis (o, len_, false);
}

}
