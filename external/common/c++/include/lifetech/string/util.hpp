/*
 *  Created on: 12-22-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49961 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:31:15 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef STRING_UTIL_HPP
#define STRING_UTIL_HPP

#include <vector>
#include <cstring>
#include <string>

namespace lifetechnologies
{
namespace string_util
{

inline std::string remove_char(std::string const& in, char ch) 
{
    std::string ret(in);
    std::size_t pos = ret.find(ch);
    while (pos != std::string::npos)
    {
        ret.replace(pos, 1, std::string());
        pos = ret.find(ch, pos);
    }
    return ret;
}
inline std::string remove_space(std::string const& in) 
{
    std::string ret(in);
    std::size_t pos = ret.find(' ');
    while (pos != std::string::npos)
    {
        ret.replace(pos, 1, std::string());
        pos = ret.find(' ', pos);
    }
    return ret;
}

inline std::string trim(std::string const& in) // from the begin and end
{
    std::string ret(in);
    std::size_t pos = ret.find_first_not_of(' ');
    if (pos != std::string::npos)
    {
        ret = ret.substr(pos);
    }

    pos = ret.find_last_not_of(' ');
    if (pos != std::string::npos)
    {
        ret = ret.substr(0, pos+1);
    }
    return ret;
}

inline void tokenize(const std::string &str, const std::string &delim, std::vector<std::string> &tokens)
{
    // skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delim, 0);
    // find first "non-delimiter".
    std::string::size_type pos = str.find_first_of(delim, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delim, pos);
        // find next "non-delimiter"
        pos = str.find_first_of(delim, lastPos);
    }
}

// make  the string lower
inline void make_lower_inplace(std::string& str)
{
    for (std::size_t i=0; i< str.size(); ++i)
    {
        if (str[i] >= 'A' && str[i] <= 'Z')
        {
            str[i] += 0x20;
        }
    }
}
inline std::string make_lower(std::string const& str)
{
    std::string ret(str);
    make_lower_inplace(ret);
    return ret;
}

// make  the string upper
inline void make_upper_inplace(std::string& str)
{
    for (std::size_t i=0; i< str.size(); ++i)
    {
        if (str[i] >= 'a' && str[i] <= 'z')
        {
            str[i] -= 0x20;
        }
    }
}
inline std::string make_upper(std::string const& str)
{
    std::string ret(str);
    make_upper_inplace(ret);
    return ret;
}

inline bool ends_with(std::string const& str, std::string const& ending)
{
    size_t pos = str.rfind(ending);
    return ((pos != std::string::npos) && ((str.length() - pos) == ending.length()));
}

} //namespace string_util
} //namespace lifetechnologies

#endif // STRING_UTIL_HPP
