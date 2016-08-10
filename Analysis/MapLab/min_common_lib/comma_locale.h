/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __comma_locale_h__
#define __comma_locale_h__

#include <locale>

class HexadecimalWithCommas : public std::numpunct<char>
{
public:
    HexadecimalWithCommas ()
    :
    std::numpunct<char> (1)
    {
    }
protected:
    virtual std::string do_grouping () const
    {
        return "\04";
    }
    virtual char do_thousands_sep () const
    {
        return ',';
    }
};

class DecimalWithCommas : public std::numpunct<char>
{
public:
    DecimalWithCommas ()
    :
    std::numpunct<char> (1)
    {
    }
protected:
    virtual std::string do_grouping () const
    {
        return "\03";
    }
    virtual char do_thousands_sep () const
    {
        return ',';
    }
};

#ifndef __comma_locale_cpp__

//extern comma_numpunct numpunct;
extern std::locale deccomma_locale;
extern std::locale hexcomma_locale;
extern std::locale system_locale;

#endif

#endif
