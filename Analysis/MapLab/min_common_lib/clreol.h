/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __clreol_h__
#define __clreol_h__

#include "fileno.hpp"
#include <ios>

#define clear_to_end_of_line_escape "\x1B[K"
class ClearToEndOfLine
{
};
#ifndef __clreol_cpp__
extern ClearToEndOfLine clreol;
#endif

inline std::ostream& operator << (std::ostream& output_stream, const ClearToEndOfLine&)
{
    int ostream_fhandle = fileno (output_stream);
    if (isatty (ostream_fhandle))
        output_stream << clear_to_end_of_line_escape;
    return output_stream;
}

#endif // __clreol_h__
