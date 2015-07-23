/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "MDtag.h"


// parser for the MD tag string - fills instance of MD with passed tag's content

void MD::parse (const char* mdstring)
{
    size_t accum = 0;
    components_.clear ();
    bool in_number = false;
    bool past_delete = false;
    while (*mdstring)
    {
        if (isdigit (*mdstring))
        {
            past_delete = false;
            if (!in_number)
                in_number = true;
            accum *= 10;
            accum += *mdstring-'0';
        }
        else
        {
            if (in_number && accum)
                components_.push_back (Component (accum)), accum = 0;
            in_number = false;
            if (*mdstring == '^')
                past_delete = true;
            else if (isalpha (*mdstring))
                components_.push_back (Component (*mdstring, past_delete));
        }
        ++ mdstring;
    }
    if (in_number && accum)
        components_.push_back (Component (accum));
}

std::ostream& operator << (std::ostream& o, const MD::Component& c)
{
    o << ((c.type_ == MD::match)?'M':'R') << '.' << ((c.pastd_)?'^':'-') << '.' << (char) (c.chr_? (c.chr_+('A' - 1)):'-') << '.' << c.count_;
    return o;
}
std::ostream& operator << (std::ostream& o, const MD& md)
{
    o << "MD (" << md.size () << " el) [";
    for (unsigned cidx = 0; cidx != md.size (); ++cidx)
    {
        if (cidx) 
            o << " ";
        o << md [cidx];
    }
    o << "]";
    return o;
}