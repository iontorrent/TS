/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __comma_locale_h__
#define __comma_locale_h__

#include <locale>

class hex_comma : public std::numpunct<char>
{
  protected:
    virtual char do_thousands_sep() const
    {
        return ',';
    }
    virtual std::string do_grouping() const
    {
        return "\04";
    }
public:
    hex_comma ()
    :
    std::numpunct<char> (1)
    {
    }
};

class dec_comma : public std::numpunct<char>
{
  protected:
    virtual char do_thousands_sep() const
    {
        return ',';
    }
    virtual std::string do_grouping() const
    {
        return "\03";
    }
public:
    dec_comma ()
    :
    std::numpunct<char> (1)
    {
    }
};

#ifndef __comma_locale_cpp__

//extern comma_numpunct numpunct;
extern std::locale deccomma_locale;
extern std::locale hexcomma_locale;
extern std::locale system_locale;

#endif

#endif
