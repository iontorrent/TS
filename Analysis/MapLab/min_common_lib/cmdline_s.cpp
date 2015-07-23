/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "cmdline_s.h"
#include <stdlib.h>
#include <string.h>

#include <stdio.h>


bool in_longopts (const char* par, const char* const* longopts)
{
    //printf ("in_longopts (%s)? ", par);
    int lo = 0;
    if (!longopts) return false;
    if (strlen (par) < 2) return false;
    while (longopts [lo])
    {
        if (strcmp (longopts [lo], par + 2) == 0)
        {
            //printf ("Yes\n");
            return true;
        }
        lo ++;
    }
    //printf ("No\n");
    return false;
}

#define BLEN 128

void par_check (const char* par, const char* optdef, bool& reg, bool& hasarg, const char* const* longopts)
{
    reg = false, hasarg = false;
    //printf ("Check par %s: ", par);
    if ((strlen (par) == 2) && (par [0] == '-') && (par [1] != ':'))
    {
        char* fc = (char*) strchr (optdef, par [1]);
        if (fc
            && ((unsigned)(fc - optdef) + 1 < strlen (optdef))
            && (optdef [(fc - optdef) + 1] == ':'))
        {
            //printf ("short, registered, hasargs\n");
            reg = true, hasarg = true;
        }
        else if (fc)
        {
            //printf ("short, registered, no args\n");
            reg = true, hasarg = false;
        }
        //else
            //printf ("short, unregistered\n");
    }
    else if ((strlen (par) > 2) && (par [0] == '-') && (par [1] == '-'))
    {
        int pl = strlen (par);
        if (in_longopts (par, longopts))
        {
            //printf ("long, registered, no args\n");
            reg = true, hasarg = false;
        }
        else
        {
            char buf [BLEN];
            if (pl > BLEN - 2)
            {
                //printf ("Too long!");
                reg = false, hasarg = false;
            }
            else
            {
                strcpy (buf, par);
                strcat (buf, "=");
                if (in_longopts (buf, longopts))
                {
                    //printf ("long, registered, hasargs\n");
                    reg = true, hasarg = true;
                }
                //else
                    //printf ("long, unregistered\n");
            }
        }
    }
}

void get_opt (int argc, const char* const* argv, const char* optdef, Arglist& arglist, Optdict& optdict, const char* const* longopts)
{
    enum STATE { SCANOPT_S = 0, READPAR_S = 1 } state = SCANOPT_S;

    arglist.clear ();
    optdict.erase (optdict.begin (), optdict.end ());

    const char* curopt = NULL;
    int argstart = -1;

    for (int argpos = 0; argpos < argc; argpos ++)
    {
        const char* arg = argv [argpos];
        bool dashed = (arg [0] == '-');
        // printf ("Pos: %d, Arg: %s, dashed: %d, state: %d\n", argpos, arg, dashed, state);
        if (state == SCANOPT_S)
        {
            if (dashed)
            {
                bool registered, parametrized;
                par_check (arg, optdef, registered, parametrized, longopts);
                // printf ("On return from par_check: registered = %d, parametrized = %d\n", registered, parametrized);
                if (parametrized)
                    curopt = arg, state = READPAR_S;
                else if (registered)
                {
                    // printf ("Search in optdict\n");
                    if (optdict.find (arg) == optdict.end ())
                    {
                        // printf ("Adding to optdict\n");
                        optdict [arg] = Arglist ();
                    }
                    //else
                    //    printf ("Found in optdict\n");
                }
            }
            else
            {
                if (argpos - argstart == 1)
                {
                    if (argstart == -1) argstart = 0;
                    break;
                }
                argstart = argpos;
            }
        }
        else
        {
            if (dashed)
            {
                if (optdict.find (curopt) == optdict.end ()) optdict [curopt] = Arglist ();
                bool registered, parametrized;
                par_check (arg, optdef, registered, parametrized, longopts);
                if (parametrized)
                    curopt = arg;
                else if (registered)
                {
                    if (optdict.find (arg) == optdict.end ()) optdict [arg] = Arglist ();
                    state = SCANOPT_S;
                }
                else
                    state = SCANOPT_S;
            }
            else
            {
                if (optdict.find (curopt) == optdict.end ()) optdict [curopt] = Arglist ();
                optdict [curopt].push_back (arg);
                state = SCANOPT_S;
            }
        }
    }
    if (argstart == -1)
        argstart = argc;

    for (int ap = argstart; ap < argc; ap ++)
        arglist.push_back (argv [ap]);
}

void parse_options (int argc, const char* const* argv, std::string& progname, Arglist& arglist, Optdict& optdict, const char* optdef, const char* const* longopts)
{
#ifdef _MSC_VER
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    _splitpath (argv [0], drive, dir, fname, ext);
    progname = fname;
#else
    progname = argv [0];
#endif

    get_opt (argc - 1, argv + 1, optdef, arglist, optdict, longopts);
}




#ifdef CMDLINE_TEST
#include <stdio.h>


int main (int argc, char* argv [])
{
    if (argc == 1)
    {
        printf ("cmdline module. Contains parse_options and get_opt functions\n");
        printf ("To test, supply command line\n");
        printf ("Known options: a-z, --opt1, --opt2 unparametrized; A-Z, --opt3, --opt4: parametrized\n");
    }
    else
    {
        Optdict od;
        Arglist al;
        std::string pn;

        const char* shortopts = "abcdefghijklmnopqrstuvwzyzA:B:C:D:E:F:G:H:I:J:K:L:M:N:O:P:Q:R:S:T:U:V:W:X:Y:Z:";
        const char* longopts [] = {"opt1", "opt2", "opt3=", "opt4=", NULL};

        parse_options (argc, argv, pn, al, od, shortopts, longopts);

        printf ("Program name = %s\n", pn.c_str ());
        printf ("options:\n");
        for (Optdict::iterator oitr = od.begin (); oitr != od.end (); oitr ++)
        {
            printf ("  %s : ", (*oitr).first.c_str ());
            for (Arglist::iterator itr = (*oitr).second.begin (); itr != (*oitr).second.end (); itr ++)
                printf ("%s, ", (*itr).c_str ());
            printf ("\n");
        }
        printf( "Arguments:\n");
        for (int a = 0; a < al.size (); a ++)
            printf( "%d : %s\n", a, al [a].c_str ());
    }
    return 0;
}
#endif


