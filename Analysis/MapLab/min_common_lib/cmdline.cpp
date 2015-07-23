/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <cstring>
#include "cmdline.h"
#include "resource.h"
#include "cmdline_s.h"
#include "common_str.h"
#include "rerror.h"


KeyFormat::KeyFormat (const char* shortopts, const char** longopts, const char* name, const char* section, const char* parameter, bool optional, bool has_arg, const char* arg_type, const char* def_value, const char* description)
:
    name_ (name),
    section_ (section),
    parameter_ (parameter),
    optional_ (optional),
    has_arg_ (has_arg),
    arg_type_ (arg_type),
    def_value_ (def_value),
    description_ (description)
{
    const char* key = shortopts;
    char bb [2];
    bb [1] = 0;
    for (; *key; key ++)
    {
        bb [0] = *key;
        shortopts_.push_back (std::string (bb));
    }

    const char** lop = longopts;
    if (lop)
        for (; *lop; lop ++)
            longopts_.push_back (std::string (*lop));
}

CmdLine :: CmdLine (KeysFormat& keys, ArgsFormat& args, int argc, char* argv [], bool strict)
:
keys_format_ (keys),
args_format_ (args),
first_optional_pos_ (-1),
last_optional_pos_ (-1),
repeatable_pos_ (-1),
ok_ (true)
{
    validate_args_format (strict);
    parse (argc, argv, strict);
}

bool CmdLine :: isOk ()
{
    return ok_;
}

void CmdLine :: validate_args_format (bool strict)
{
    if (strict)
    {
        // check valididty of argument list spec
        // the rules are:
        //   not more then 1 repeatable arg;
        //   optionals form compact group
        //   optionals preceed repeatables
        //
        // non-optional non-repeatable takes 1 position
        // optional non-repeatable takes 1 position if exists
        // repeatable takes 1 or more positions
        // optional repeatable takes 0 or more positions
        int argno = 0;
        ArgsFormat::iterator itr = args_format_.begin ();
        for (; itr != args_format_.end () && ok_; itr ++, argno ++)
        {
            ArgFormat& arg = *itr;
            if (arg.repeatable_)
            {
                if (repeatable_pos_ != -1)
                    ERR ("Internal: More then one repeatable arguments"); // second repeatable
                else
                    repeatable_pos_ = argno;
            }
            if (arg.optional_)
            {
                if (repeatable_pos_ != -1 && repeatable_pos_ != argno)
                    ERR ("Internal: Optional argument follows repeatable"); // optional after repeatable
                if (last_optional_pos_ != -1)
                    ERR ("Internal: Non-compact optional group"); // optional after the end of optional group
                if (first_optional_pos_ == -1)
                    first_optional_pos_ = argno;
            }
            else
            {
                if (first_optional_pos_ != -1 && last_optional_pos_ == -1)
                    last_optional_pos_ = argno;
            }
        }
        if (first_optional_pos_ != -1 && last_optional_pos_ == -1)
        {
            last_optional_pos_ = argno;
        }
    }
}

void CmdLine :: reportErrors (std::ostream& o)
{
    o << error_report_.c_str () << std::endl;
}

void CmdLine :: printHelp (const char* progname, std::ostream& o, bool longform)
{
    std::string arg0 = progname;
    size_t last_separator = arg0.find_last_of ("\\/:");
    if (last_separator != std::string::npos)
        arg0 = arg0.substr (last_separator+1, arg0.length ());
    size_t last_period = arg0.find_last_of (".");
    if (last_period != std::string::npos)
        arg0 = arg0.substr (0, last_period);

    o << "Usage : " << arg0.c_str ();
    if (keys_format_.size ()) o << " KEYS";
    for (ArgsFormat::iterator itr = args_format_.begin (); itr != args_format_.end (); itr ++)
        o << " " << itr->name_.c_str ();

    if (keys_format_.size () || args_format_.size ())
        o << ", where";
    if (keys_format_.size ())
    {
        o << std::endl << " KEYS are:";
        for (KeysFormat::iterator itr = keys_format_.begin (); itr != keys_format_.end (); itr ++)
        {
            o << std::endl << "  ";
            for (svec::iterator ki = (*itr).shortopts_.begin (); ki != (*itr).shortopts_.end (); ki ++)
                o << "-" << (*ki).c_str () << " ";
            for (svec::iterator loi = (*itr).longopts_.begin (); loi != (*itr).longopts_.end (); loi ++)
                o << "--" << (*loi).c_str () << " ";
            if (!(*itr).optional_) o << "[REQUIRED]";
            else if ((*itr).has_arg_) o << "[" << (*itr).def_value_.c_str () << "]";
            o << ": " << (*itr).description_.c_str ();
            if (longform)
            {
                if ((*itr).has_arg_) o << ", type: " << (*itr).arg_type_.c_str ();
                else o << ", [no arguments]";
                if ((*itr).section_.length () || (*itr).parameter_.length ())
                     o << ", parameter: [" << (*itr).section_.c_str () << ":" << (*itr).parameter_.c_str () << "]";
                o << ", mnemonics '" << (*itr).name_.c_str () << "' ";
            }
        }
    }
    if (args_format_.size ())
    {
        for (ArgsFormat::iterator itr = args_format_.begin (); itr != args_format_.end (); itr ++)
        {
            bool sep = false;
            o << std::endl << " " << (*itr).name_.c_str () << " : " << (*itr).type_.c_str ();
            if ((*itr).optional_)
            {
                if (!sep)
                {
                    sep = true;
                    o << " ";
                }
                o << "[optional]";
            }
            if ((*itr).repeatable_)
            {
                if (!sep)
                {
                    sep = true;
                    o << " ";
                }
                o << "[repeatable]";
            }
            o << " : " << (*itr).description_.c_str ();
        }
    }
    o << std::endl << std::flush;
}

void CmdLine :: printArgs (std::ostream& o)
{
    if (keys_.size ())
    {
        o << std::endl << "Keys:";
        for (KeysFormat::iterator kitr = keys_format_.begin (); kitr != keys_format_.end (); kitr ++)
        {
            std::map<std::string, std::string>::iterator valitr = keys_.find (kitr->name_);
            if (valitr != keys_.end ())
            {
                o << std::endl << "  " << kitr->name_.c_str ();
                if (kitr->has_arg_) o <<  " = " << valitr->second.c_str ();
            }
            else if (!kitr->optional_)
            {
                o << std::endl << "  " << kitr->name_.c_str () << " - No Value";
            }

        }
    }
    if (arguments_.size ())
    {
        o << std::endl << "Arguments:";
        int argno = 0;
        for (unsigned fmtpos = 0; fmtpos < args_format_.size (); fmtpos ++)
        {
            int takes_min = args_format_ [fmtpos].optional_ ? 0 : 1;
            int takes_max = arguments_.size () - (args_format_.size () - last_optional_pos_) - fmtpos;
            if (!args_format_ [fmtpos].repeatable_) takes_max = std::min (takes_max, 1);
            o << std::endl << "  " << args_format_ [argno].name_.c_str () << " : ";
            if (takes_max <= 0)
            {
                if (takes_min == 0)
                    o << "Optional - Not given";
                else
                    o << "Value missing";
            }
            else
            {
                for (int took = 0;; took ++)
                {
                    o << arguments_ [argno ++].c_str ();
                    if (took == takes_max - 1)
                        break;
                    o << ", ";
                }
            }
        }
    }
    o << std::flush;
}

bool CmdLine :: hasKey (const char* name)
{
    std::map<std::string, std::string>::iterator valitr = keys_.find (name);
    return valitr != keys_.end ();
}

const char* CmdLine :: getValue (const char* name)
{
    std::map<std::string, std::string>::iterator valitr = keys_.find (name);
    if (valitr == keys_.end ()) return NULL;
    else return valitr->second.c_str ();
}

int CmdLine :: getFmtPos (int argno) const
{
    int scanned = 0;
    int advance;
    int tail_sz = (last_optional_pos_ == -1) ? 0 : (args_format_.size () - last_optional_pos_);
    int head_sz = (first_optional_pos_ == -1)? args_format_.size () : first_optional_pos_;
    int space = arguments_.size () - (tail_sz + head_sz);
    if (space < 0)
        ERR ("Number of non-optional arguments is smaller then argument list size");
    for (unsigned fmtpos = 0; fmtpos < args_format_.size (); fmtpos ++)
    {
        advance = 0;
        ArgFormat& cur = args_format_ [fmtpos];
        if (cur.repeatable_)
        {
            if (cur.optional_)
                advance = space;
            else
                advance = space + 1;
            space = 0;
        }
        else
        {
            if (cur.optional_)
            {
                if (space > 0)
                {
                    advance = 1;
                    space -= 1;
                }
            }
            else
                advance = 1;
        }
        if (scanned + advance > argno)
            return fmtpos;
        scanned += advance;
    }
    // ERR ("Argument number too big for format line")
    return -1;
}

int CmdLine :: getArgno ()
{
    return arguments_.size ();
}

const char* CmdLine :: getArg (int no)
{
    return arguments_ [no].c_str ();
}

const unsigned MAX_LONGOPTS = 256;
const unsigned MAX_LONGOPT_LEN = 512;

void CmdLine :: parse (int argc, char** argv, bool strict)
{
    // prepare the parameters and call cmdline_s get_opt

    Arglist arglist;
    Optdict optdict;

    // allocate storage for long options spec
    unsigned longopts_no = 0;
    KeysFormat::iterator kitr;
    char longopts_spec_buf [MAX_LONGOPTS][MAX_LONGOPT_LEN];
    char* longopts_spec [MAX_LONGOPTS];
    for (kitr = keys_format_.begin (); kitr != keys_format_.end (); kitr ++) 
        longopts_no += (*kitr).longopts_.size ();
    if (longopts_no + 1 >= MAX_LONGOPTS)
        ers << "Too many command line options in command line format definition" << ThrowEx(InternalRerror);

    // fill in options specs and make option->spec map
    std::map <std::string, KeyFormat*> opt2spec;
    std::string shortopts_spec (EMPTY_STR);

    int lidx = 0;
    for (kitr = keys_format_.begin (); kitr != keys_format_.end (); kitr ++)
    {
        // short
        svec::iterator soi = (*kitr).shortopts_.begin ();
        for (; soi != (*kitr).shortopts_.end (); soi ++)
        {
            const char* optstr = (*soi).c_str ();
            shortopts_spec+= optstr;
            if ((*kitr).has_arg_) shortopts_spec += ":";

            std::string keystr ("-");
            keystr.append (optstr);
            opt2spec [keystr] = &(*kitr);
        }
        // long
        svec::iterator loi = (*kitr).longopts_.begin ();
        for (; loi != (*kitr).longopts_.end (); loi ++, lidx ++)
        {
            unsigned lolen = (*loi).length () + 1;
            if ((*kitr).has_arg_) lolen ++;
            if (lolen + 1 >= MAX_LONGOPT_LEN)
                ers << "Command line option " << *loi << " is too long" << Throw;
            // longopts_spec [lidx] = new char [lolen];
            const char* optstr = (*loi).c_str ();
            longopts_spec [lidx] = longopts_spec_buf [lidx];
            strcpy (longopts_spec [lidx], optstr);
            if ((*kitr).has_arg_) strcat (longopts_spec [lidx], "=");

            std::string keystr ("--");
            keystr.append (optstr);
            opt2spec [keystr] = &(*kitr);
        }
    }
    longopts_spec [lidx] = NULL;

    // actually parse command line
    get_opt (argc, argv, shortopts_spec.c_str (), arglist, optdict, longopts_spec);

    // fill in the keys_
    Optdict::iterator oi = optdict.begin ();
    for (;oi != optdict.end (); oi ++)
    {
        const char* key = (*oi).first.c_str ();
        // find what parameter this name corresponds to
        std::map <std::string, KeyFormat*>::iterator k = opt2spec.find (key);
        if (k != opt2spec.end ())
        {
            svec values = (*oi).second;
            if ((*k).second->has_arg_)
            {
                if (values.size () == 0)
                {
                    error_report_ += "\nParameter ";
                    error_report_ += key;
                    error_report_ += "(";
                    error_report_ += (*k).second->name_;
                    error_report_ += ") requires argument";
                    ok_ = false;
                    break;
                }
                const char* val = values [values.size () - 1].c_str ();
                keys_ [(*k).second->name_] = val;
            }
            else
            {
                keys_ [(*k).second->name_] = EMPTY_STR;
            }
            // DEBUG
            // printf ("CMDL PAR '%s' : '%s'", (*k).second->name_.c_str (), val);
        }
    }

    // check weather all non-optional keys are found
    if (ok_ && strict)
    {
        for (kitr = keys_format_.begin (); kitr != keys_format_.end (); kitr ++)
        {
            if (!kitr->optional_)
            {
                std::map<std::string, std::string>::iterator valitr = keys_.find (kitr->name_);
                if (valitr == keys_.end ())
                {
                    // add notes about missing ones to the error_report
                    error_report_ += "Required option missing from command line: ";
                    error_report_ += kitr->name_;
                    error_report_ += "\n";
                    ok_ = false;
                }
            }
        }
    }
    // fill in the arguments_
    if (ok_) std::copy (arglist.begin (), arglist.end (), std::back_inserter (arguments_));

    // check if the number of arguments is valid
    if (strict && ok_ && repeatable_pos_ == -1 && arglist.size () > args_format_.size ())
    {
        // add the note to error_report
        error_report_ += "Too many command-line arguments\n";
        ok_ = false;
    }
    unsigned min_arg_number = ((first_optional_pos_ != -1) ? first_optional_pos_ : 0)
                    + args_format_.size () - ((last_optional_pos_ == -1) ? 0 : last_optional_pos_);
    if (strict && ok_ && arguments_.size () < min_arg_number)
    {
        // add the note to error_report
        error_report_ += "Too few command-line arguments. Please use -h for help.\n";
        ok_ = false;
    }
}

KeyFormat* CmdLine :: keyFormat (const char* name)
{
    // find a key in keys_format
    for (unsigned kfi = 0; kfi < keys_format_.size (); kfi ++)
    {
        if (keys_format_[kfi].name_ == name)
            return &keys_format_ [kfi];
    }
    return NULL;
}

ArgFormat* CmdLine :: argFormat (unsigned argno)
{
    if (argno >= args_format_.size ())
        return NULL;
    else
        return &args_format_ [argno];
}
