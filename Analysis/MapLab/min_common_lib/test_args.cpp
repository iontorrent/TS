/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_args.h"
#include <cstring>


TestEnv testEnv; // testEnv singleton
static const char zs [] = "";

const char* find_test_arg (const char* key, TestEnv* test_env) // returns value from passed test arguments if key is present or NULL
{
    if (!test_env) 
    test_env = &testEnv;
    for (int i = 1; i < test_env->argc_; ++i)
    {
        if (0 == strcmp (test_env->argv_ [i], key))
        {
            if (i + 1 == test_env->argc_)
                return zs;
            if (test_env->argv_ [i + 1][0] == '-')
                return zs;
            return test_env->argv_ [i + 1];
        }
    }
    return NULL;
}
bool has_test_key (const char *key, TestEnv* test_env) // checks if the given key is in test arguments
{
    for (int i = 1; i < test_env->argc_; ++i)
    {
        if (0 == strcmp (test_env->argv_ [i], key))
            return true;
    }
    return false;
}
const char* find_test_env (const char* name, TestEnv* test_env) // finds value under given name in environment. Returns NULL if name not there; zero string if name is present but empty
{
    if (!test_env->envp_)
        return NULL;
    unsigned nlen = strlen (name);
    for (char** ep = test_env->envp_; *ep; ++ep)
    {
        unsigned elen = strlen (*ep);
        if (elen <= nlen)
            continue;
        if ((*ep)[nlen] != '=')
            continue;
        if (strncmp (*ep, name, nlen))
            continue;
        return (*ep)+nlen+1;
    }
    return NULL;
}
