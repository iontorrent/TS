/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "context-align-params.h"
#include <common_str.h>
#include <cstring>
#include <runtime_error.h>

const char* CONTAL_HDR = "Context-specific BAM realigner";
const char* CONTAL_PROCNAME = "ContextAlign";

static const char* GIP_DEFAULT = FIVE_STR;
static const char* GEP_DEFAULT = TWO_STR;
static const char* MAT_DEFAULT = ONE_STR;
static const char* MIS_DEFAULT = THREE_STR;
static const char* BWID_DEFAULT = FIVE_STR;
static const char* ORD_DEFAULT = "ATCG";

static const char* BAMIDX_DEFAULT = EMPTY_STR;
static const char* REFNAME_DEFAULT = EMPTY_STR;
static const char* REFNO_DEFAULT = MINUS_ONE_STR;
static const char* BEGPOS_DEFAULT = ZERO_STR;
static const char* ENDPOS_DEFAULT = ZERO_STR;
static const char* SKIP_DEFAULT = ZERO_STR;
static const char* LIMIT_DEFAULT = ZERO_STR;

static const char* LOGFNAME_DEFAULT = EMPTY_STR;
static const char* LOGOPTS_DEFAULT = EMPTY_STR;


static const char* GIP_HELP = "Gap opening penalty";
static const char* GEP_HELP = "Gap extension penalty";
static const char* MAT_HELP = "Match bonus";
static const char* MIS_HELP = "Mismatch penalty";
static const char* BWID_HELP = "Extra alignment band width, over determined from existing alignment";
static const char* ORD_HELP = "Flow order";
static const char* INBAM_HELP = "BAM file to process";
static const char* OUTBAM_HELP = "BAM file to write results into";
static const char* BAMIDX_HELP = "BAM index, if different from BAM filename-based";
static const char* REFNAME_HELP = "Refernece name to limit processing";
static const char* REFNO_HELP = "Refernece index to limit processing to, gets precedence over name, -1 ignored";
static const char* BEGPOS_HELP = "Start position of the zone to process";
static const char* ENDPOS_HELP = "End position of the zone to process, 0 for no limit";
static const char* SKIP_HELP = "Number of records to skip before starting processing";
static const char* LIMIT_HELP = "Number of records to process, 0 for no limit";

static const char* LOGFNAME_HELP = "Name for optional log file";

static const char* VALID_LOG_OPTS [] = {"base", "diff" , "matr"};
static std::string LOGOPTS_HELP_STR = "Logging options, as comma separated list of: ";
static const char*  populate_longopts_help ()
{
    for (unsigned idx = 0, sent = sizeof (VALID_LOG_OPTS) / sizeof (*VALID_LOG_OPTS); idx != sent; ++idx)
    {
        if (idx)
            LOGOPTS_HELP_STR.append (",");
        LOGOPTS_HELP_STR.append (VALID_LOG_OPTS [idx]);
    }
    return LOGOPTS_HELP_STR.c_str ();
}
static const char* LOGOPTS_HELP = populate_longopts_help ();

static const char* logip  []   = {"gip", NULL};
static const char* logep  []   = {"gep", NULL};
static const char* lomat  []   = {"mat", NULL};
static const char* lomis  []   = {"mis", NULL};
static const char* lobwid []   = {"bwid", NULL};
static const char* loord  []   = {"ord", NULL};
static const char* loidx  []   = {"idx", NULL};
static const char* lorefn []   = {"ref", NULL};
static const char* lorefi []   = {"refidx", NULL};
static const char* lobegp []   = {"beg", NULL};
static const char* loendp []   = {"end", NULL};
static const char* loskip []   = {"skip", NULL};
static const char* lolimit[]   = {"lim", "limit", NULL};
static const char* lologop[]   = {"log", NULL};
static const char* lologf []   = {"logf", NULL};

static const char* CALIGN_SECTNAME = "CONTEXT_ALIGN";

bool ContalignParams::prepareCmdlineFormat ()
{
    bool res = Process_params::prepareCmdlineFormat ();
    if (res)
    {
        keys_format_.push_back (KeyFormat (EMPTY_STR,        logip,  "gip",    CALIGN_SECTNAME,  "GIP",  true, true, INTEGER_STR, gip_default (), gip_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        logep,  "gep",    CALIGN_SECTNAME,  "GEP",  true, true, INTEGER_STR, gep_default (), gep_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lomat,  "mat",    CALIGN_SECTNAME,  "MAT",  true, true, INTEGER_STR, mat_default (), mat_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lomis,  "mis",    CALIGN_SECTNAME,  "MIS",  true, true, INTEGER_STR, mis_default (), mis_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lobwid, "bwid",   CALIGN_SECTNAME,  "BWID", true, true, INTEGER_STR, bwid_default (), bwid_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        loord,  "ord",    CALIGN_SECTNAME,  "ORD",  true, true, STRING_STR,  ord_default (), ord_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        loidx,  "idx",    CALIGN_SECTNAME,  "IDX",  true, true, STRING_STR,  bamidx_default (), bamidx_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lorefn, "ref",    CALIGN_SECTNAME,  "REFN", true, true, STRING_STR,  refname_default (), refname_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lorefi, "refidx", CALIGN_SECTNAME,  "REFI", true, true, INTEGER_STR, refno_default (), refno_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lobegp, "beg",    CALIGN_SECTNAME,  "BEG",  true, true, INTEGER_STR, begpos_default (), begpos_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        loendp, "end",    CALIGN_SECTNAME,  "END",  true, true, INTEGER_STR, endpos_default (), endpos_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        loskip, "skip",   CALIGN_SECTNAME,  "SKIP", true, true, INTEGER_STR, skip_default (), skip_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lolimit,"lim",    CALIGN_SECTNAME,  "LIM",  true, true, INTEGER_STR, limit_default (), limit_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lologf, "logf",   CALIGN_SECTNAME,  "LOGF", true, true, STRING_STR,  logfname_default (), logfname_help ()));
        keys_format_.push_back (KeyFormat (EMPTY_STR,        lologop,"logop",  CALIGN_SECTNAME,  "LOGOP",true, true, STRING_STR,  logopts_default (), logopts_help ()));
        args_format_.push_back (ArgFormat ("INBAM",  FILENAME_STR, inbam_help (), false));
        args_format_.push_back (ArgFormat ("OUTBAM", FILENAME_STR, outbam_help (), false));
    }
    return res;
}

bool ContalignParams::prepareParameters ()
{
    bool toRet = Process_params::prepareParameters ();
    if (toRet)
    {
        Parameter_descr CALIGN_SECTION [] =
        {
            {"GIP", INTEGER_STR,  gip_default (), gip_help ()},
            {"GEP", INTEGER_STR,  gep_default (), gep_help ()},
            {"MAT", INTEGER_STR,  mat_default (), mat_help ()},
            {"MIS", INTEGER_STR,  mis_default (), mis_help ()},
            {"BWID",INTEGER_STR,  bwid_default (), bwid_help ()},
            {"ORD", STRING_STR,   ord_default (), ord_help ()},
            {"IDX", STRING_STR,   bamidx_default (), bamidx_help ()},
            {"REFN",STRING_STR,   refname_default (), refname_help ()},
            {"REFI",INTEGER_STR,  refno_default (), refno_help ()},
            {"BEG", INTEGER_STR,  begpos_default (), begpos_help ()},
            {"END", INTEGER_STR,  endpos_default (), endpos_help ()},
            {"SKIP",STRING_STR,   skip_default (), skip_help ()},
            {"LIM", STRING_STR,   limit_default (), limit_help ()},
            {"LOGF",STRING_STR,   logfname_default (), logfname_help ()},
            {"LOGOP",STRING_STR,  logopts_default (), logopts_help ()},
        };
        parameters_->addSection (CALIGN_SECTNAME, "Context-sensitive realignment parameters", CALIGN_SECTION, sizeof (CALIGN_SECTION) / sizeof (Parameter_descr));

        Parameter_descr ARGUMENTS_SECTION [] =
        {
            {"INBAM",         STRING_STR, EMPTY_STR,  inbam_help ()},
            {"OUTBAM",        STRING_STR, EMPTY_STR,  outbam_help ()},
        };
        parameters_->addSection (volatile_section_name,  "Program arguments", ARGUMENTS_SECTION, sizeof (ARGUMENTS_SECTION) / sizeof (Parameter_descr));
    }
    return toRet;
}

bool ContalignParams::interpreteParameters ()
{
    bool result = Process_params::interpreteParameters ();

    if (result)
    {
        gip (parameters_->getInteger   (CALIGN_SECTNAME, "GIP"));
        gep (parameters_->getInteger   (CALIGN_SECTNAME, "GEP"));
        mat (parameters_->getInteger   (CALIGN_SECTNAME, "MAT"));
        mis (parameters_->getInteger   (CALIGN_SECTNAME, "MIS"));
        bwid (parameters_->getInteger   (CALIGN_SECTNAME, "BWID"));
        ord (parameters_->getParameter (CALIGN_SECTNAME, "ORD"));
        bamidx (parameters_->getParameter (CALIGN_SECTNAME, "IDX"));
        refname (parameters_->getParameter (CALIGN_SECTNAME, "REFN"));
        refno (parameters_->getInteger (CALIGN_SECTNAME, "REFI"));
        begpos (parameters_->getInteger (CALIGN_SECTNAME, "BEG"));
        endpos (parameters_->getInteger (CALIGN_SECTNAME, "END"));
        skip (parameters_->getInteger (CALIGN_SECTNAME, "SKIP"));
        limit (parameters_->getInteger (CALIGN_SECTNAME, "LIM"));
        logfname (parameters_->getParameter (CALIGN_SECTNAME, "LOGF"));
        logopts (parameters_->getParameter (CALIGN_SECTNAME, "LOGOP"));
        inbam (parameters_->getParameter (volatile_section_name, "INBAM"));
        outbam (parameters_->getParameter (volatile_section_name, "OUTBAM"));
    }
    if (result)
        result = interprete_logopts ();
    return result;
}



bool ContalignParams::interprete_logopts ()
{
    // trc << "logopts is " << logopts_ << ", parsed to: ";

    size_t pos = 0, ppos = 0;
    do
    {
        pos = logopts_.find (',', ppos);
        if (pos != ppos)
        {
            std::string comp = logopts_.substr (ppos, (pos == std::string::npos)? std::string::npos : (pos - ppos));
            // very naive validation, ineficient for more then 3-4 standard options
            bool valid = false;
            for (unsigned vi = 0; vi != sizeof (VALID_LOG_OPTS) / sizeof (*VALID_LOG_OPTS) && ! valid; ++vi)
                valid = (comp == VALID_LOG_OPTS [vi]);
            if (!valid)
                errlog << "Ignoring unknown logging option: " << comp << std::endl;
            else
                logspec_.insert (comp);
        }
        if (pos != std::string::npos)
            ppos = pos + 1;
    }
    while (pos != std::string::npos);

    //if (trclog.enabled ())
    //{
    //    for (std::set <std::string> :: const_iterator itr = logspec_.begin (), sent = logspec_.end (); itr != sent; ++itr)
    //        trclog << *itr << ((itr == logspec_.begin ())? "": ", ");
    //    trclog << std::endl;
    //}
    return true;
}

const char* ContalignParams::gip_default () const
{
    return GIP_DEFAULT;
}

const char* ContalignParams::gep_default () const
{
    return GEP_DEFAULT;
}

const char* ContalignParams::mat_default () const
{
    return MAT_DEFAULT;
}

const char* ContalignParams::mis_default () const
{
    return MIS_DEFAULT;
}

const char* ContalignParams::bwid_default () const
{
    return BWID_DEFAULT;
}

const char* ContalignParams::ord_default () const
{
    return ORD_DEFAULT;
}

const char* ContalignParams::bamidx_default () const
{
    return BAMIDX_DEFAULT;
}

const char* ContalignParams::refname_default () const
{
    return REFNAME_DEFAULT;
}

const char* ContalignParams::refno_default () const
{
    return REFNO_DEFAULT;
}

const char* ContalignParams::begpos_default () const
{
    return BEGPOS_DEFAULT;
}

const char* ContalignParams::endpos_default () const
{
    return ENDPOS_DEFAULT;
}

const char* ContalignParams::skip_default () const
{
    return SKIP_DEFAULT;
}

const char* ContalignParams::limit_default () const
{
    return LIMIT_DEFAULT;
}

const char* ContalignParams::logfname_default () const
{
    return LOGFNAME_DEFAULT;
}

const char* ContalignParams::logopts_default () const
{
    return LOGOPTS_DEFAULT;
}


const char* ContalignParams::gip_help () const
{
    return GIP_HELP;
}

const char* ContalignParams::gep_help () const
{
    return GEP_HELP;
}

const char* ContalignParams::mat_help () const
{
    return MAT_HELP;
}

const char* ContalignParams::mis_help () const
{
    return MIS_HELP;
}

const char* ContalignParams::bwid_help () const
{
    return BWID_HELP;
}

const char* ContalignParams::ord_help () const
{
    return ORD_HELP;
}

const char* ContalignParams::inbam_help () const
{
    return INBAM_HELP;
}

const char* ContalignParams::outbam_help () const
{
    return OUTBAM_HELP;
}

const char* ContalignParams::bamidx_help () const
{
    return BAMIDX_HELP;
}

const char* ContalignParams::refname_help () const
{
    return REFNAME_HELP;
}

const char* ContalignParams::refno_help () const
{
    return REFNO_HELP;
}

const char* ContalignParams::begpos_help () const
{
    return BEGPOS_HELP;
}

const char* ContalignParams::endpos_help () const
{
    return ENDPOS_HELP;
}

const char* ContalignParams::skip_help () const
{
    return SKIP_HELP;
}

const char* ContalignParams::limit_help () const
{
    return LIMIT_HELP;
}

const char* ContalignParams::logfname_help () const
{
    return LOGFNAME_HELP;
}

const char* ContalignParams::logopts_help () const
{
    return LOGOPTS_HELP;
}

bool ContalignParams::logging (const char* option) const
{
    return (bool) logspec_.count (option);
}
