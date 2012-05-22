/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>

#include "Logger.h"
#include "OptionArgsBase.h"

using namespace std;
using namespace ION;

/**
 * Process the command line parameters.  Does not throw exceptions.
 * @param argc
 * @param argv
 * @return 
 */
bool OptionArgsBase::ProcessNX( int argc, char** argv )
{
    try
    {
        // Call Process and catch and handle any exception here
        // to prevent exceptions from propagating to the caller.
        // For errors, false is returned.
        Process( argc, argv );
    }
    catch( runtime_error& e )
    {
        // Catch and log runtime_error exceptions.
        LOG( e.what() )
        return false;
    }
    catch( ... )
    {
        // Catch and log an unknown type of exception.
        LOG( "Exception: Unknown exception has occurred..." )
        return false;
    }
    
    return true;
}

/**
 * Base class to process the command line parameters.  Throws exceptions.
 * This base class version does not do much processing, just some bookkeeping.
 * All the substantial processing occurs in the derived class.
 * 
 * @param argc is the argument count.
 * @param argv is the array of arguments.
 */
void OptionArgsBase::Process( int argc, char* argv[] )
{
    // Capture the raw command line arguments.
    _strRawCmdLine = "";
    // Iterate the command line arguments.
    for( int i = 0; i < argc; i++ )
    {
        // Assemble each of the orignal command line arguments into a single string.
        _strRawCmdLine += std::string( argv[i] ) + " ";
    }
}

/**
 * Tokenize the string into a vector separated by a delimiter character.
 * @param str is the string to tokenize.
 * @param delim is the delimited used to tokenize the string.
 * @param vecSplitItems is the vector of strings holding the tokenized strings.
 */
void OptionArgsBase::SplitString( const std::string &str,
                                char delim,
                                STRINGVEC& vecSplitItems )
{
    std::stringstream ss( str );
    std::string item;
    while( std::getline( ss, item, delim ) )
    {
        vecSplitItems.push_back( item );
    }
}

/**
 * Get the non-options in a vector of strings.
 * @param vecNonOptionList the vector of strings to retrieve.
 */
void OptionArgsBase::GetNonOptionList( STRINGVEC& vecNonOptionList )
{
    // Clear the return vector prior to use.
    vecNonOptionList.clear();    

    // Iterate the vector of non-options and copy each item into the return vector.
    STRINGVEC::iterator iter = _vecNonOptions.begin();    
    for( ; iter != _vecNonOptions.end(); iter++ )
    {
        vecNonOptionList.push_back( *iter );
    }
}
