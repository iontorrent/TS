/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
// Derived from QSimScan project, released under MIT license
// (https://github.com/abadona/qsimscan)
//////////////////////////////////////////////////////////////////////////////

#ifndef __parameters_h__
#define __parameters_h__

#ifdef _MSC_VER
#pragma warning (disable: 4786)
#endif

#include <map>
#include <ostream>
#include "tracer.h"
#include "parameters_section.h"

#ifndef __parameters_cpp__
extern const char* volatile_section_name;
#endif


class Parameters
{
public:
    typedef std::map<std::string, ParametersSection> sectmap;
    sectmap sections_;

    void addSection (const char* sectname, const char* sectdescr, Parameter_descr* pars, int par_no);
    void addParameter (const char* sectname, Parameter_descr& descr);
    void writeHelp (std::ostream& o, const char* header = NULL);

    bool readFile (const char* filename);
    bool writeSection (const char* section, std::ostream& o);
    bool read (std::istream& istr);
    bool write (std::ostream& ostr);
    bool log (Logger& log);

    // parameters access
    bool hasSection (const char* sectname);
    bool hasParameter (const char* section, const char* name);
    bool hasValue (const char* section, const char* name);
    bool setParameter (const char* section, const char* name, const char* value);
    const char* getParameter (const char* section, const char* name);
    bool removeParameter (const char* section, const char* name);
    const char* getDefault (const char* section, const char* name);

    // utility
    long long getInteger (const char* section, const char* name);
    double  getFloat (const char* section, const char* name);
    bool getBoolean (const char* section, const char* name);
    void setInteger (const char* section, const char* name, long long value);
    void setFloat  (const char* section, const char* name, double value);
    void setBoolean (const char* section, const char* name, bool value);
};

std::ostream& operator << (std::ostream&, Parameters&);
Logger& operator << (Logger&, Parameters&);

#endif // __parameters_h__
