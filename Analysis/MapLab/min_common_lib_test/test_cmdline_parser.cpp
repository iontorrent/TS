/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_cmdline_parser.h"

const char* TESTPROCNAME = "TestClib";
const char* TESTHDR = "Internal test for command-line parsing modules";
const char* KSTR_DEFAULT = "DefaultStringParam";
const char* KBOOL_DEFAULT = "False";
const char* KINT_DEFAULT = "54321";
const char* KSTR_HELP = "String parameter";
const char* KBOOL_HELP = "Boolean parameter";
const char* KINT_HELP = "Integer parameter";
const char* FARG_HELP = "First argument";
const char* OARG_HELP = "Intermediate optional argument";
const char* RARG_HELP = "Intermediate repeatable argument";
const char* LARG_HELP = "Last argument";
const char* kstr_lo [] =  {"kstr", NULL};
const char* kbool_lo [] = {"kbool", NULL};
const char* kint_lo [] =  {"kint", NULL};


void TestCmdlineParser::TestParams::rargs (const char* opt)
{
    std::string tt = opt;
    const char* delim = " ";
    std::string::size_type begIdx = 0, endIdx = 0;
    std::string::size_type ttlen = tt.length ();

    rargs_.clear ();

    while (endIdx != ttlen )
    {
        begIdx = tt.find_first_not_of (delim, endIdx);
        if (begIdx == std::string::npos)
            break;
        endIdx = tt.find_first_of (delim, begIdx);
        if (endIdx == std::string::npos)
            endIdx = ttlen;

        rargs_.push_back (tt.substr (begIdx, endIdx - begIdx));
    }
}

bool TestCmdlineParser::TestParams::prepareCmdlineFormat ()
{
    bool toR = Process_params::prepareCmdlineFormat ();
    if (toR)
    {
        keys_format_.push_back (KeyFormat ("s", kstr_lo,    "kstrpar",   "TESTSECT",  "KSTR",   true,   true,  "String",   KSTR_DEFAULT,  KSTR_HELP));
        keys_format_.push_back (KeyFormat ("b", kbool_lo,   "kboolpar",  "TESTSECT",  "KBOOL",  true,   false, "Boolean",  "True",        KBOOL_HELP));
        keys_format_.push_back (KeyFormat ("i", kint_lo,    "kintpar",   "TESTSECT",  "KINT",   true,   true,  "Integer",  KINT_DEFAULT,  KINT_HELP));

        args_format_.push_back (ArgFormat ("farg", "word", FARG_HELP, false, false));
        args_format_.push_back (ArgFormat ("oarg", "word", OARG_HELP, false, true));
        args_format_.push_back (ArgFormat ("rarg", "word", RARG_HELP, true,  false));
        args_format_.push_back (ArgFormat ("larg", "word", LARG_HELP, false, false));
    }
    return toR;
}

bool TestCmdlineParser::TestParams::prepareParameters ()
{
    static Parameter_descr TEST_SECTION [] =
    {
        {"KSTR",  "String",  KSTR_DEFAULT,  KSTR_HELP},
        {"KBOOL", "Boolean", KBOOL_DEFAULT, KBOOL_HELP},
        {"KINT",  "Integer", KINT_DEFAULT,  KINT_HELP}
    };
    static Parameter_descr ARGUMENTS_SECTION [] =
    {
        {"farg",  "String",     "", FARG_HELP},
        {"oarg",  "String",     "", OARG_HELP},
        {"rarg",  "StringList", "", RARG_HELP},
        {"larg",  "String",     "", LARG_HELP}
    };

    bool toR = Process_params::prepareParameters ();
    if (toR)
    {
        parameters_->addSection ("TESTSECT",             "Test section",       TEST_SECTION,        sizeof (TEST_SECTION) / sizeof (Parameter_descr));
        parameters_->addSection (volatile_section_name,  "Program arguments",  ARGUMENTS_SECTION,   sizeof (ARGUMENTS_SECTION) / sizeof (Parameter_descr));
    }
    return toR;
}

bool TestCmdlineParser::TestParams::interpreteParameters ()
{
    bool toR = Process_params::interpreteParameters ();
    if (toR)
    {
        kstr  (parameters_->getParameter ("TESTSECT", "KSTR"));
        kbool (parameters_->getBoolean ("TESTSECT", "KBOOL"));
        kint  (parameters_->getInteger ("TESTSECT", "KINT"));

        farg  (parameters_->getParameter (volatile_section_name, "farg"));
        oarg  (parameters_->getParameter (volatile_section_name, "oarg"));
        rargs (parameters_->getParameter (volatile_section_name, "rarg"));
        larg  (parameters_->getParameter (volatile_section_name, "larg"));
    }
    return toR;
}

bool TestCmdlineParser :: process ()
{
    try
    {
        char* argv [] = {(char*) "ARG0", (char*) "-s", (char*) "string_option", (char*) "-i", (char*) "42", (char*) "-b", (char*) "FARG", (char*) "OARG", (char*) "RARG1", (char*) "RARG2", (char*) "LARG"};
        int argc = sizeof (argv) / sizeof (*argv);

        TestParams p;
        if (!p.parseCmdline (argc, argv))
        {
            o_ << "Error processing parameters ";
            p.cmdline ()->reportErrors (o_);
            o_ << std::endl;
            return false;
        }
        o_ << "Command line parsed" << std::endl;
        if (!p.process ())
        {
            o_ << "Process methiod returned False" << std::endl;
            return false;
        }
        o_ << "Parameters processed" << std::endl;
        p.parameters_->write (o_);
        o_ << std::endl;
    }
    catch (RunTimeError& e)
    {
        o_ << "Run time error: " << (const char*) e << std::endl;
        return false;
    }
    return true;
}
