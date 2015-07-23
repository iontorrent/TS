/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define __parameters_cpp__
#include "parameters.h"
#include <ctype.h>
#include <iterator>
#include <fstream>
#include <ios>
#include <cstring>
#include <cstdlib>
#include "fileutils.h"
#include "rerror.h"
#include "portability.h"
#include "common_str.h"

static const int bufsz = 1024;
static const char whitespace [] = "\t ";
static const char delimeter  [] = "\t =:";
static const char default_params_header [] = "  Configuration file format:";

const char *volatile_section_name = "ARGUMENTS";


void Parameters::addSection (const char* sectname, const char* sectdescr, Parameter_descr* pars, int par_no)
{
    ParametersSection& section = sections_ [sectname]; // get or create
    section.name_ = sectname; // unneccessary if section existed. Does not inflict too much overhead since operaton is not iterative :)
    if (sectdescr)
        section.description_ = sectdescr; // silently replace by new value if given
    for (int i = 0; i < par_no; i ++)
    {
        Parameter& cur_par = section.parameters_ [pars [i].name_]; // silently replace by new value
        cur_par.name_ = pars [i].name_;
        cur_par.type_ = pars [i].type_;
        cur_par.def_value_ = pars [i].def_value_;
        cur_par.description_ = pars [i].description_;
    }
}

void Parameters::addParameter (const char* sectname, Parameter_descr& descr)
{
    ParametersSection& section = sections_ [sectname]; // get or create
    section.name_ = sectname; // unneccessary if section existed. Does not inflict too much overhead since operaton is not iterative :)
    Parameter& cur_par = section.parameters_ [descr.name_];
    cur_par.name_ = descr.name_;
    cur_par.type_ = descr.type_;
    cur_par.def_value_ = descr.def_value_;
    cur_par.description_ = descr.description_;
}


bool Parameters::readFile (const char* filename)
{
    if (!file_exists (filename)) return false;

    std::ifstream istr (filename, std::ios::in);
    if (!istr.is_open ()) return false;

    return read (istr);
}

bool Parameters::writeSection (const char* sectname, std::ostream& o)
{
    if (sections_.find (sectname) == sections_.end ()) return false;
    ParametersSection& section = sections_ [sectname];
    o << "[" << sectname << "]" << std::endl;
    for (ParametersSection::parmap::iterator itr = section.parameters_.begin (); itr != section.parameters_.end (); itr ++)
    {
        Parameter& p = itr->second;
        o << p.name_.c_str () << " = ";
        if (p.value_.length ())
            o << p.value_.c_str ();
        else
            o << p.def_value_.c_str ();
        o << std::endl;
    }
    return true;
}

void Parameters::writeHelp (std::ostream& o, const char* header)
{
    if (!header) header = default_params_header;
    o << header << std::endl;
    for (sectmap::iterator itr = sections_.begin (); itr != sections_.end (); itr ++)
    {
        if (itr->first != volatile_section_name)
            itr->second.writeHelp (o);
    }
}

bool Parameters::read (std::istream& istr)
{
    char buf [bufsz];
    std::string cursectname = EMPTY_STR;
    ParametersSection* cursection = NULL;

    while (istr.getline (buf, bufsz))
    {
        std::string s = buf;
        // parse the line
        std::string::size_type p = s.find_first_not_of (whitespace);
        if (p == s.npos) continue;
        else if (s [p] == '[') // section
        {
            p ++;
            std::string::size_type p1 = s.find (']');
            if (p1 == s.npos) p1 = s.length ();
            cursectname = s.substr (p, p1 - p);
            if (hasSection (cursectname.c_str ()))
                cursection = &sections_ [cursectname];
            else
                cursection = NULL;
        }
        else
        {
            if (cursection)
            {
                int name_end = s.find_first_of (delimeter, p);
                std::string name (s, p, name_end - p);
                int val_start = s.find_first_not_of (delimeter, name_end);
                std::string value;
                if (val_start >= 0)
                    value.assign (s, val_start, s.length () - val_start);
                else
                    value.assign (EMPTY_STR);
                setParameter (cursectname.c_str (), name.c_str (), value.c_str ());
            }
        }
    }
    return true;
}

bool Parameters:: write (std::ostream& ostr)
{
    for (sectmap::iterator itr = sections_.begin (); itr != sections_.end (); itr ++)
        // if (itr->first != volatile_section_name)
            if (!writeSection (itr->first.c_str (), ostr)) return false;
    return true;
}

bool Parameters:: log (Logger& log)
{
    return log.enabled () ? write (log.o_) : false;
}

bool Parameters::hasSection (const char* sectname)
{
    sectmap::iterator sitr = sections_.find (sectname);
    if (sitr == sections_.end ()) return false;
    else return true;
}

// parameters access
bool Parameters::hasParameter (const char* sectname, const char* parname)
{
    sectmap::iterator sitr = sections_.find (sectname);
    if (sitr == sections_.end ()) return false;
    ParametersSection::parmap::iterator pitr = sitr->second.parameters_.find (parname);
    if (pitr == sitr->second.parameters_.end ()) return false;
    return true;
}
bool Parameters::hasValue (const char* sectname, const char* parname)
{
    sectmap::iterator sitr = sections_.find (sectname);
    if (sitr == sections_.end ()) return false;
    ParametersSection::parmap::iterator pitr = sitr->second.parameters_.find (parname);
    if (pitr == sitr->second.parameters_.end ()) return false;
    if (!pitr->second.value_.length ()) return false;
    return true;
}

bool Parameters::setParameter (const char* sectname, const char* parname, const char* value)
{
    // only the parameters which have the types/descriptions could be set
    sectmap::iterator sitr = sections_.find (sectname);
    if (sitr == sections_.end ()) return false;
    ParametersSection::parmap::iterator pitr = sitr->second.parameters_.find (parname);
    if (pitr == sitr->second.parameters_.end ()) return false;
    pitr->second.value_ = value;
    return true;
}

const char* Parameters::getParameter (const char* sectname, const char* parname)
{
    if (!hasValue (sectname, parname))
        return getDefault (sectname, parname);
    else
        return sections_ [sectname].parameters_ [parname].value_.c_str();
}

const char* Parameters::getDefault (const char* sectname, const char* parname)
{
    sectmap::iterator sitr = sections_.find (sectname);
    if (sitr == sections_.end ()) return EMPTY_STR;
    ParametersSection::parmap::iterator pitr = sitr->second.parameters_.find (parname);
    if (pitr == sitr->second.parameters_.end ()) return EMPTY_STR;
    return pitr->second.def_value_.c_str ();
}

bool Parameters::removeParameter (const char* sectname, const char* parname)
{
    if (!hasParameter (sectname, parname)) return false;
    sections_ [sectname].parameters_.erase (parname);
    return true;
}

// utility

longlong  Parameters::getInteger (const char* sectname, const char* parname)
{
    const char* sv = getParameter (sectname, parname);
    if (!sv) return 0;
    return atoll (sv);
}

double  Parameters::getFloat (const char* sectname, const char* parname)
{
    const char* sv = getParameter (sectname, parname);
    if (!sv) return 0;
    return atof (sv);
}

static const char* truevals [] =
{
    "1",
    "YES",
    "TRUE",
    "Y"
};

bool Parameters::getBoolean (const char* sectname, const char* parname)
{
    const char* sv = getParameter (sectname, parname);
    if (!sv) return false;
    else
        for (size_t tvno = 0; tvno < sizeof (truevals) / sizeof (const char*) ; tvno ++)
            if (strcasecmp (truevals [tvno], sv) == 0)
                return true;
    return false;
}

void Parameters::setInteger (const char* sectname, const char* parname, longlong value)
{
    char buf [128];
    setParameter (sectname, parname, lltoa (value, buf, 10));
}

void Parameters::setFloat  (const char* sectname, const char* parname, double value)
{
    char buf [128];
    sprintf (buf, "%f", value);
    setParameter (sectname, parname, buf);
}

void Parameters::setBoolean (const char* sectname, const char* parname, bool value)
{
    const char* bv = ((value) ? (TRUE_STR) : (FALSE_STR));
    setParameter (sectname, parname, bv);
}

std::ostream& operator << (std::ostream& oo, Parameters& pp)
{
    pp.write (oo);
    return oo;
}
