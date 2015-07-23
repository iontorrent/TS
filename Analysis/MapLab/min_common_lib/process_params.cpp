/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "process_params.h"
#include "rerror.h"
#include <fstream>
#include <string.h>
#include "portability.h"
#include "common_str.h"

static const char* DEF_VERSION = "0.9.a";
static const char* DEF_AUTHORSHIP = "Scientific Data Management (SciDM.org) (c)1997-2013";

const char *empty_longopt_list [] = {NULL};
const char *Process_params::parfname_default_ = "default.cfg";

static const char* OVERWRITE_DEFAULT = FALSE_STR;
static const char* VERBOSE_DEFAULT = FALSE_STR;
static const char* DEBUG_DEFAULT = ZERO_STR;
static const char* VERBOSE_HELP   = "Produce verbose output";
static const char* OVERWRITE_HELP = "Overwrite existing output objects";
static const char* DEBUG_HELP   = "Sets the level of debug output: 0(critical), 1(errors), 2(warnings), 3(info), 4(debug), 5(trace)";

const char* ibs (const char* bs)
{
    if (bs == FALSE_STR || strcasecmp (bs, FALSE_STR) == 0)
        return TRUE_STR;
    else if (bs == TRUE_STR || strcasecmp (bs, TRUE_STR) == 0)
        return FALSE_STR;
    else
        return bs;
}

Process_params::Process_params (const char* header, const char* procname, const char* version, const char* authorship)
:
templates_prepared_ (false),
parameters_read_ (false),
verbose_ (false),
overwrite_ (false),
debug_level_ (0),
header_ (header),
procname_ (procname ? procname : (const char*) EMPTY_STR),
version_ (version ? version : DEF_VERSION),
authorship_ (authorship ? authorship : DEF_AUTHORSHIP),
help_mode_ (false),
cmdline_ (NULL),
parfname_ (parfname_default_),
parameters_ (NULL)
{
    if (procname_.length () != 0)
    {
        def_parfname_ = procname;
        def_parfname_ += ".cfg";
        parfname_ = def_parfname ();
    }
}


Process_params::~Process_params ()
{
    if (parameters_) delete parameters_;
    if (cmdline_) delete cmdline_;
}

static const char* lohlp []         = {"help", NULL};
static const char* lohlpx []        = {"helpx", NULL};
static const char* lohlppar []      = {"help_par", NULL};
static const char* loversion []     = {"version", NULL};
static const char* loverbose []     = {"verb", "verbose", NULL};
static const char* lodbgout []      = {"dbg", "debug", NULL};
static const char* loconfig []      = {"config", NULL};
static const char* loconfigw []     = {"outcfg", NULL};
static const char* loover []        = {"ov", "over", "overwrite", NULL};

static const char* lohlp_help       = "Prints help on command line usage";
static const char* lohlp_xhelp      = "Prints extended help on command line usage";
static const char* lohlppar_help    = "Prints help on parameters file format";
static const char* loversion_help   = "Prints version information and exits";
static const char* loconfig_help    = "Configuration file to use";
static const char* loconfigw_help   = "Save default parameters into the file";


void Process_params::add_cmdline_srv ()
{
    keys_format_.push_back (KeyFormat ("h",       lohlp,     "help",        EMPTY_STR,        EMPTY_STR,   true, false, EMPTY_STR,   EMPTY_STR,                lohlp_help));
    keys_format_.push_back (KeyFormat (EMPTY_STR, lohlpx,    "helpx",       EMPTY_STR,        EMPTY_STR,   true, false, EMPTY_STR,   EMPTY_STR,                lohlp_xhelp));
    keys_format_.push_back (KeyFormat (EMPTY_STR, lohlppar,  "help_par",    EMPTY_STR,        EMPTY_STR,   true, false, EMPTY_STR,   EMPTY_STR,                lohlppar_help));
    keys_format_.push_back (KeyFormat ("v",       loverbose, "verbose",    "GENERIC_OUTPUT", "VERBOSE"  ,  true, false, EMPTY_STR,   ibs (verbose_default ()), verbose_help ()));
    keys_format_.push_back (KeyFormat (EMPTY_STR, lodbgout,  "debug",      "GENERIC_OUTPUT", "DEBUG"  ,    true, true,  INTEGER_STR, debug_default (),         debug_help ()));
    keys_format_.push_back (KeyFormat (EMPTY_STR, loversion, "version",     EMPTY_STR,        EMPTY_STR  , true, false, EMPTY_STR,   EMPTY_STR,                loversion_help));
}

void Process_params::add_cmdline_app ()
{
    keys_format_.push_back (KeyFormat ("c",       loconfig,  "configfile", EMPTY_STR,        EMPTY_STR,   true, true,  "filename", EMPTY_STR,               loconfig_help));
    keys_format_.push_back (KeyFormat ("w",       loconfigw, "write_par",  EMPTY_STR,        EMPTY_STR,   true, true,  "filename", def_parfname (),         loconfigw_help));
}
void Process_params::add_cmdline_overwrite ()
{
    keys_format_.push_back (KeyFormat (EMPTY_STR, loover,    "overwrite",  "GENERIC_OUTPUT", "OVERWRITE", true, false, EMPTY_STR,  ibs (overwrite_default ()), overwrite_help ()));
}
bool Process_params::prepareCmdlineFormat ()
{
    add_cmdline_srv ();
    add_cmdline_app ();
    add_cmdline_overwrite ();
    return true;
}

bool  Process_params::createParamsObject ()
{
    parameters_ = new Parameters ();
    return true;
}

void Process_params::add_parameters_overwrite ()
{
    Parameter_descr GENERIC_OUTPUT_SECTION [] =
    {
        {"OVERWRITE", "Boolean", overwrite_default (), overwrite_help ()}
    };
    parameters_->addSection ("GENERIC_OUTPUT",  "Generic output parameters", GENERIC_OUTPUT_SECTION, sizeof (GENERIC_OUTPUT_SECTION) / sizeof (Parameter_descr));
}


bool Process_params::prepareParameters ()
{
    Parameter_descr GENERIC_OUTPUT_SECTION [] =
    {
        {"VERBOSE",   "Boolean", verbose_default (),  verbose_help ()},
        {"DEBUG",     "Integer", debug_default (), debug_help ()}
    };

    if (!createParamsObject ()) return false;
    parameters_->addSection ("GENERIC_OUTPUT",  "Generic output parameters", GENERIC_OUTPUT_SECTION, sizeof (GENERIC_OUTPUT_SECTION) / sizeof (Parameter_descr));
    add_parameters_overwrite ();
    return true;
}

bool Process_params::prepareTemplates ()
{
    if (!templates_prepared_)
    {
        prepareCmdlineFormat ();
        prepareParameters ();
        templates_prepared_ = true;
    }
    return true;
}

bool Process_params::parseCmdline (int argc, char* argv [], bool strict)
{
    // fill in procname from command line if not given explicitely

    if (procname_.length () == 0)
    {
        std::string arg0 = argv [0];
        size_t last_separator = arg0.find_last_of ("\\/:");
        if (last_separator != std::string::npos)
            arg0 = arg0.substr (last_separator+1, arg0.length ());
        size_t last_period = arg0.find (".exe");
        if (last_period != std::string::npos)
            arg0 = arg0.substr (0, last_period);

        procname_ = arg0;

        def_parfname_ = procname_;
        def_parfname_ += ".cfg";
        parfname_ = def_parfname ();
    }


    // do all necessary preparations (call overloaded initializers)
    prepareTemplates ();

    // parse command line into CmdLine object
    cmdline_ = new CmdLine (keys_format_, args_format_, argc-1, argv+1, strict);

    // if help is requested, print it and return false
    help_mode_ = false;
    if (cmdline_->hasKey ("help") || cmdline_->hasKey ("helpx"))
    {
        std::cout << procname_ << " version " << version_ << std::endl;
        if (header_)
            std::cout << header_ << std::endl;
        if (authorship_.size ())
            std::cout << "by " << authorship_ << std::endl;
            
        cmdline_->printHelp (argv [0], std::cout, cmdline_->hasKey ("helpx"));
        help_mode_ = true;
    }
    if (cmdline_->hasKey ("help_par"))
    {
        parameters_->writeHelp (std::cout, "Parameters file format:");
        help_mode_ = true;
    }
    if (cmdline_->hasKey ("version"))
    {
        std::cout << procname_ << " " << version_ << std::endl;
        help_mode_ = true;
    }
    if (help_mode_)
        return false;

    // check for command line format / syntax error; report and quit
    return cmdline_->isOk ();
}

bool Process_params::process ()
{
    if (!populateParameters ()) return false;
    if (!interpreteCmdline ()) return false;
    if (!interpreteParameters ()) return false;
    if (!postProcessParams ()) return false;
    return true;
}

bool Process_params::postProcessParams ()
{
    if (cmdline_->hasKey ("write_par"))
    {
        writeParams ();
        return false;
    }

    return true;
}

bool Process_params::populateParameters ()
{
    // read the parameters file name if given
    if (cmdline_->hasKey ("configfile")) parfname_ = cmdline_->getValue ("configfile");
    else parfname_ = def_parfname ();

    parameters_read_ = parameters_->readFile (parfname ());
    return true;
}

void Process_params::writeParams ()
{
    const char* fn = cmdline_->getValue ("write_par");
    std::ofstream par_file (fn);
    if (!par_file.is_open ())
        ers << "WARNING: unable to open file " << fn << " for writing. Parameters not saved." << Throw;
    else
    {
        parameters_->write (par_file);
        par_file.close ();
        if (verbose ()) std::clog << "Parameters saved to file :" << fn << std::endl;
    }
}

bool Process_params::interpreteCmdline ()
{
    // read the command line options and store to parameters
    for (CmdLine::ssmap::iterator itr = cmdline_->keys_.begin (); itr != cmdline_->keys_.end (); itr ++)
    {
        KeyFormat* kfmt = cmdline_->keyFormat (itr->first.c_str ());
        if (!kfmt) Error (InternalRerror);
        const char* parsec = kfmt->section_.c_str ();
        const char* parpar = kfmt->parameter_.c_str ();
        if (*parpar != 0)
        {
            if (!kfmt->has_arg_)
                parameters_->setParameter (parsec, parpar, kfmt->def_value_.c_str ());
            else
                parameters_->setParameter (parsec, parpar, itr->second.c_str ());
        }
    }

    std::string repv;
    std::string reppname;
    for (unsigned argno = 0; argno < cmdline_->arguments_.size (); argno ++)
    {
        int fmtpos = cmdline_->getFmtPos (argno);
        ArgFormat* afmt = cmdline_->argFormat (fmtpos);
        if (!afmt)
            ers << "Format spec not found for argument " << argno << ", format_pos " << fmtpos <<Throw;
        const char* name = afmt->name_.c_str ();

        if (afmt->repeatable_)
        {
            if (repv.length ()) repv.append (" ");
            repv.append (cmdline_->arguments_ [argno]);
            if (!reppname.length ()) reppname = name;
        }
        else
        {
            if (repv.length ())
            {
                parameters_->setParameter (volatile_section_name, reppname.c_str (), repv.c_str ());
                repv = EMPTY_STR;
            }
            parameters_->setParameter (volatile_section_name, name, cmdline_->arguments_ [argno].c_str ());
        }
    }
    if (repv.length ())
    {
        parameters_->setParameter (volatile_section_name, reppname.c_str (), repv.c_str ());
    }

    return true;
}

bool Process_params::interpreteParameters ()
{
    if (parameters_-> hasParameter ("GENERIC_OUTPUT", "VERBOSE"))
        this->verbose   (parameters_->getBoolean ("GENERIC_OUTPUT", "VERBOSE"));
    if (parameters_-> hasParameter ("GENERIC_OUTPUT", "OVERWRITE"))
        this->overwrite (parameters_->getBoolean ("GENERIC_OUTPUT", "OVERWRITE"));
    if (parameters_-> hasParameter ("GENERIC_OUTPUT", "DEBUG"))
        this->debug (parameters_->getInteger ("GENERIC_OUTPUT", "DEBUG"));
    return true;
}

CmdLine* Process_params::cmdline ()
{
    return cmdline_;
}

bool Process_params::help_mode () const
{
    return help_mode_;
}

bool Process_params::validate ()
{
    return true;
}

const char* Process_params::verbose_default () const
{
    return VERBOSE_DEFAULT;
}

const char* Process_params::overwrite_default () const
{
    return OVERWRITE_DEFAULT;
}

const char* Process_params::debug_default () const
{
    return DEBUG_DEFAULT;
}

const char* Process_params::verbose_help () const
{
    return VERBOSE_HELP;
}

const char* Process_params::overwrite_help () const
{
    return OVERWRITE_HELP;
}

const char* Process_params::debug_help () const
{
    return DEBUG_HELP;
}

std::ostream& operator << (std::ostream& o, const Process_params& p)
{
    o << "Process parameters:" << std::endl;
    o << SP2_STR << "Verbose:" << bool_str (p.verbose ()) << std::endl;
    o << SP2_STR << "Overwrite:" << bool_str (p.overwrite ()) << std::endl;
    o << SP2_STR << "Debug level:" << (int) p.debug () << std::endl;
    return o;
}
