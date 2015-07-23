/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_args_h__
#define __test_args_h__

#include <iostream>

/// Test environment
/// Just keeps the pointers to command line arguments and environment strings
class TestEnv
{
public:
    /// indicates if the instance was initialized
    bool init_;
    /// holds number of command line arguments
    int argc_;
    /// holds array of pointers to command line arguments
    char** argv_;
    /// holds array of pointers to environment strings, NULL terminated
    char** envp_;
    /// Constructor
    TestEnv ()
    :
    init_ (false),
    argc_ (0),
    argv_ (NULL),
    envp_ (NULL)
    {
    }
    /// sets instance to hold the passed arguments
    void init (int argc = 0, char** argv = NULL, char** envp = NULL)
    {
        if (init_)
        {
            std::cerr << "WARNING: attempt to re-initialize TestEnv singleton: " << __FILE__ << ":" << __LINE__ << std::endl;
            return;
        }
        init_ = true;
        if (argc)
            argc_ = argc;
        if (argv)
            argv_ = argv;
        if (envp)
            envp_ = envp;
    }
};

/// singleton holding the instance of TestEnv obtained from the test executable invocation
extern TestEnv testEnv;

/// retrieves keyword argument from test environment
/// \param  key       name for key-value style argument
/// \param  test_env  pointer to the TextEnv instance; if not given, global TestEnv singleton is used
/// \return           value from passed test arguments if key is present, NULL otherwise
const char* find_test_arg (const char* key, TestEnv* test_env = NULL);
/// checks if the given key is in test arguments
/// \param  key       name for key-value style argument
/// \param  test_env  pointer to the TextEnv instance; if not given, global TestEnv singleton is used
/// \return           true if given key is present, false otherwise
bool has_test_key (const char *key, TestEnv* test_env = NULL); 
/// finds value for the given environment variable. 
/// \param  name       name for environment variable
/// \param  test_env  pointer to the TextEnv instance; if not given, global TestEnv singleton is used
/// \return           value for the existing envirpnment name; NULL if name not there; zero string if name is present but empty
const char* find_test_env (const char* name, TestEnv* test_env = NULL); 


#endif // __test_args_h__
