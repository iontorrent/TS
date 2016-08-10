/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __parameters_section_h__
#define __parameters_section_h__
#ifdef _MSC_VER
#pragma warning (disable:4786)
#endif

#include <string>
#include <map>
#include <ostream>

struct Parameter_descr
{
    const char* name_;
    const char* type_;
    const char* def_value_;
    const char* description_;
};


struct Parameter
{
    Parameter () {}
    Parameter (Parameter_descr& descr)
    :
    name_ (descr.name_),
    type_ (descr.type_),
    def_value_ (descr.def_value_),
    description_ (descr.description_)
    {
    }
    std::string name_;
    std::string type_;
    std::string def_value_;
    std::string description_;
    std::string value_;
};


struct ParametersSection
{
    typedef std::map<std::string, Parameter> parmap;

    std::string name_;
    std::string description_;
    parmap parameters_;
    bool writeHelp (std::ostream& ostr);
    bool hasParameter (const char* name);
};


#endif //__parameters_section_h__
