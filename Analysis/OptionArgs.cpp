/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "Logger.h"
#include "json/json.h"

#include "OptionArgs.h"

using namespace std;
using namespace ION;

// String and character constants.
const std::string strArgDelim( "," );
const std::string strCategoryOptions("Options");
const std::string strCategoryNonOptions("NonOptions");

/**
 * Process the command line arguments.
 * @param argc is the count of command line arguments.
 * @param argv is the array of command line arguments.
 */
void OptionArgs::Process( int argc, char **argv )
{
    // Call the base class Process.
    OptionArgsBase::Process( argc, argv );
    // Do the processing.
    DoProcessing( argc, argv );    
}

/**
 * Non definition table based command line processing
 * @param argc is the count of command line arguments.
 * @param argv is the array of command line arguments.
 */
void OptionArgs::DoProcessing( int argc, char **argv )
{
    // Clear the collections prior to use.
    _vecNonOptions.clear();
    _mapRawOptions.clear();
    
    // The argument string.
    std::string strArg;
    // Pair instance for map insertion.
    MAPPAIROPTION mapPair; 
    // Option flag.
    bool bIsOption = false;
        
    // Iterate each command line argument.
    for( int i = 1; i < argc; i++ )
    {
        // Save a copy of the argument at index i.
        strArg = argv[i];
        
        // If this option is merely two dashes, e.g. "--"
        // then treat all the remaining command line 
        // arguments as non-options.
        if( "--" == strArg )
        {
            // Copy the remaining parameters following the "--"
            // into the non-options collection.
            for( int j = i; j < argc; j++ )
            {
                _vecNonOptions.push_back( argv[j] );
            }
            // Return out of the method.
            return;
        }

        // An option must be at least one character in length.
        if( strArg.size() > 1 )
        {
            // If this is a long option.
            if( '-' == strArg[0] && '-' == strArg[1] )
            {
                // Strip away the "--" long option prefix.
                strArg = strArg.substr( 2, strArg.size() - 1 );
                bIsOption = true;
            }
            else
            {
                // If this is a short option.
                if( '-' == strArg[0] )
                {
                    // Strip away the "-" short option prefix.
                    strArg = strArg.substr( 1, strArg.size() - 1 );
                    bIsOption = true;
                }
            }
        }
        
        // If this is a valid option format.
        if( bIsOption )
        {
            // The map key is the option string.
            mapPair.first = strArg;
            
            // Get the next index value and validate it.
            const int iNext = i + 1;
            if( iNext < argc )
            {
                // Get the next command line argument and
                // store the next argument into the value of this map item.
                mapPair.second = argv[iNext];
            }
            else
            {
                // There is no subsequent argument for this option.
                mapPair.second = "";
            }

            // This is the result pair for the map insertion.
            std::pair<MAPOPTION::iterator,bool> mapRet;
            
            // Attempted to insert the item into the map.
            mapRet = _mapRawOptions.insert( mapPair );
            // If the second value is false, then the map insertion failed.
            if( !mapRet.second )
            {
                ION_THROW( "Duplicated Option." )
            }
            // Reset the options flag.
            bIsOption = false;
        }
        else
        {
            // This parameter is not an option, so it is a non-option.
            _vecNonOptions.push_back( strArg );
        }
    } // END FOR loop
}

/**
 * Find the option name in the _mapRawOptions map.
 * @param strOptionName is the option name.
 * @return true if the option exists, false otherwise.
 */
bool OptionArgs::FindOption( const std::string& strOptionName )
{
    // Iterator into the map for the search.
    MAPOPTION::iterator iter;
    // Find the option name in the map.
    bool bRet = FindOption( iter, strOptionName );
    return bRet;
}

/**
 * Find the option name and return the iterator.
 * @param iter is the iterator pointing into the option map.
 * @param strOptionName is the option name.
 * @return return true if the option exists, false otherwise.
 */
bool OptionArgs::FindOption( MAPOPTION::iterator& iterFound,
                              const std::string& strOptionName )
{
    bool bRet = false;

    // Find the option name in the map.
    iterFound = _mapRawOptions.find( strOptionName );
    
    // If the item exists in the map, then return true.
    if( iterFound != _mapRawOptions.end() )
    {
        bRet = true;
    }
    
    return bRet;
}

/**
 * Get the option.
 * @param strOptionName is the option name.
 * @return true if the option exists, false otherwise. 
 */
bool OptionArgs::GetOption( const std::string& strOptionName )
{
    return FindOption( strOptionName );
}

/**
 * Get the option.
 * @param strOptionName is the option name.
 * @param strDefaultValue is the default argument value of the option.
 * @param vecArgValues is the vector of parsed argument values.
 * @return true if the option exists, false otherwise. 
 */
bool OptionArgs::GetOption( const std::string& strOptionName,
                             const std::string& strDefaultValue,
                             STRINGVEC& vecArgValues )
{
    bool bRet = true;
    // Clear the returned output collection prior to use.
    vecArgValues.clear();
    // Declare an iterator into the the map for the return value.
    MAPOPTION::iterator iter;
    // Find the option name.
    bool bOptionExists = FindOption( iter, strOptionName );
    if( bOptionExists )
    {
        // The option exists in the map, so save the argument value.
        const std::string strArgVal = iter->second;
        
        // Declare an iterator into the non-options collection.
        STRINGVEC::iterator it; 
        // Remove this argument for this option out of the non-option collection.
        //
        // NOTE:
        // The reason for this is because, when the command line is originally scanned,
        // the parser treats an option's argument also as a non-option item.
        // So, this argument is duplicated both as an option argument and as a non-option
        // argument.  Therefore, we need to remove it here.
        //
        // A priori, we do not have any information defining the characteristics of
        // each option before parsing the command line, thus we cannot determine whether
        // the item following an option is that option's argument or merely a non-option
        // following that argument.  Hence, we enter the item both as an option argument
        // and as a non-option argument.  This ambiguity is resolved the moment we call
        // GetOption() expecting return argument on that option name.  If GetOption() with
        // a return argument is never called on the option name, then that argument value
        // is presumed to be a non-option, and we leave it in the non-options collection.
        it = std::find( _vecNonOptions.begin(), _vecNonOptions.end(), strArgVal );
        if( it != _vecNonOptions.end() )
        {
            // Remove this argument value from the non-options vector
            _vecNonOptions.erase( it );
        }
        
        // Tokenize this argument into components and put the results in the
        // returned output collection.
        SplitString( strArgVal, strArgDelim[0], vecArgValues );
    }
    else
    {
        // This argument is not in the map as an option.
        // So save it as a non-option and return failure.
        vecArgValues.push_back( strDefaultValue );
        bRet = false;
    }
    
    return bRet;
}

/**
 * Get the option.
 * @param strOptionName is the option name.
 * @param strShortOptionName is the short option name.
 * @param strDefaultValue is the default argument value of the option.
 * @param vecArgValues is the vector of parsed argument values.
 * @return true if the option exists, false otherwise.  
 */
bool OptionArgs::GetOption( const std::string& strOptionName,
                             const std::string& strShortOptionName,
                             const std::string& strDefaultValue,
                             STRINGVEC& vecArgValues )
{
    bool bRet = true;
    // Clear the returned output collection prior to use.
    vecArgValues.clear();

    // False, if the long option name is empty.
    const bool bOptionName = strOptionName.empty() ? false : true;
    // False, if the short option name is empty OR if it is a dash '-' character.
    const bool bShortOptionName = ( strShortOptionName.empty() || ( strShortOptionName == "-" ) ? false : true );
    // If both the long and short option names are false,
    // then return the default value and return failure.
    if( !bOptionName && !bShortOptionName )
    {
        vecArgValues.push_back( strDefaultValue );
        bRet = false;
    }
    else
    {
        // One or both option names are valid.
        //
        // Thus, if both are present, the long option name has
        // higher precedence than the short option name.
        //
        // First test the long option name.
        if( bOptionName )
        {
            bRet = GetOption( strOptionName, strDefaultValue, vecArgValues );
        }
        else
        {
            // Otherwise, we only have a short option name present.
            bRet = GetOption( strShortOptionName, strDefaultValue, vecArgValues );
        }
    }
    
    return bRet;
}

/**
 * Returns the list of options from the options map.
 * @param vecOptionList
 */
void OptionArgs::GetOptionList( STRINGVEC& vecOptionList )
{
    // Clear the returned output collection prior to use.
    vecOptionList.clear();

    // Declare an iterator into the the map.
    MAPOPTION::iterator iterOpts;
    iterOpts = _mapRawOptions.begin();

    // Copy each of the options map keys into the return vector.
    for( ; iterOpts != _mapRawOptions.end(); iterOpts++ )
    {
        // Copy the item into the vector.
        vecOptionList.push_back( iterOpts->first );        
    }
}

/**
 * Read the options from the specified JSON file.
 * @param strOptionsFileName is the JSON file to read.
 */
void OptionArgs::ReadOptions( const std::string& strOptionsFileName )
{
    // Create a character buffer for the input stream.
    const int MAXBUF = 512;
    char buf[MAXBUF];
    memset( buf, 0, sizeof(char) * MAXBUF );

    // Declare a string for the JSON input.
    std::string strJson;

    // Open the input stream with the file name.
    ifstream in( strOptionsFileName.c_str() );

    // Read in each line while the stream is good.
    // The input stream returns not good at EOF.
    while( in.good() )
    {
        // Get a line from the input stream.
        in.getline( buf, MAXBUF );
        // Concatenate the new line into the JSON string.
        strJson += buf;
    }

    // We should have the JSON file read this point.
    // Create JSON objects.
    Json::Value root;
    Json::Reader reader;

    // Parse the JSON string into the root value object.
    bool bParsed = reader.parse( strJson, root, false );
    if( !bParsed )        
    {
        // Failure, so throw exception.
        ION_THROW( std::string( "Failed to parse JSON " ) + reader.getFormatedErrorMessages() );
    }
    else
    {
        // Read the Non-Options block from the JSON file.
        Json::Value nonOptions = root[strCategoryNonOptions];
        // This block should be an array type.
        if( nonOptions.isArray() )
        {
            // Clear the non-options vector.
            _vecNonOptions.clear();
            // Copy the non-option elements as strings into the vector.
            for( size_t i = 0; i < nonOptions.size(); i++ )
                _vecNonOptions.push_back( nonOptions[(int)i].asString() );
        }
        else
        {
            // Error, so throw an exception.
            ION_THROW( strCategoryNonOptions + " is not an array." )
        }

        // Clear the options map.
        _mapRawOptions.clear();
        // Declare a pair for the map insertion operation.
        MAPPAIROPTION mapPair;
        
        // Read the Options block from the JSON file
        // Declare a JSON value iterator to the option block.
        Json::Value::iterator iter = root[strCategoryOptions].begin();
        
        // Iterate each option item in the Option block.
        for( ; iter != root[strCategoryOptions].end(); iter++ )
        {
            // Read the Options item key as a string.
            std::string strOption( iter.key().asString() );
            // Set this as string as the key item for the map insertion.
            mapPair.first = strOption;
            // Get the option item's value.
            Json::Value optionValue = root[strCategoryOptions][strOption];
            
            // If this item is a string type.
            if( optionValue.isString() )
            {
                // Set the value as a string into the map item's value field.
                mapPair.second = optionValue.asString();                
            }
            // Else, this item is an array of elements.
            else if( optionValue.isArray() )
            {
                // Declare a string in which to concatenate together the array items.
                std::string strOptArgs;

                // Iterate each element in this array.
                for( size_t i = 0; i < optionValue.size(); i++ )
                {
                    // Except for the first item,
                    // separate each element by a delimiter.
                    if( i != 0 )
                    {
                        strOptArgs += strArgDelim + " ";
                    }
                    // Concatenate the elements together into the option argument string.
                    strOptArgs += optionValue[(int)i].asString();
                }
                // Set the option argument string into the map item value.
                mapPair.second = strOptArgs;
            }
            // Otherwise, the type is unrecognized.
            else
            {
                mapPair.second = "N/A";
            }

            // Declare a pair for the map insertion return status.
            std::pair<MAPOPTION::iterator,bool> mapRet;
            // Attempt to insert the pair into the map.
            mapRet = _mapRawOptions.insert( mapPair );
            // Successful, if the second element in the pair is true.
            // Otherwise, throw an exception.
            if( !mapRet.second )
            {
                ION_THROW( "Unable to insert option into the map: Duplicated Option." )
            }
        }
    }
}

/**
 * Write the contents of the options map to a JSON file.
 * @param strOptionsFileName is the name of the JSON file to be written.
 */
void OptionArgs::WriteOptions( const std::string& strOptionsFileName )
{
    // Vector of strings for option names.
    STRINGVEC vecOptionList;
    // Get the list of option names in the map.
    GetOptionList( vecOptionList );

    // Declare a JSON Value object as root.
    Json::Value root;

    // Iterate the list of map options.
    STRINGVEC::iterator iterOpts = vecOptionList.begin();
    for( ; iterOpts != vecOptionList.end(); iterOpts++ )
    {
        // Declare a default string value.  This is required but not used elsewhere.
        const std::string strDefaultValue;
        // Declare a vector of argument value strings.
        STRINGVEC vecArgValues;

        // Get the option from the map.
        bool bResult = GetOption( *iterOpts, strDefaultValue, vecArgValues );
        if( bResult )
        {
            // Get the option name string.
            const std::string strOption( *iterOpts );

            // If the retrieved argument vector has no elements, store an empty string.
            if( 0 == vecArgValues.size() )
            {
                root[strCategoryOptions][strOption] = "";
            }
            // If there is only one element in the retrieved argument vector.
            else if( 1 == vecArgValues.size() )
            {
                // Set the single value into the string.
                root[strCategoryOptions][strOption] = vecArgValues[0];
            }
            else
            {
                // There is more than one element in the retrieved argument vector,
                // so this JSON node represents an array of elements.
                
                // Iterate each of the elements in the vector and append them into
                // and array at this JSON node.
                STRINGVEC::iterator iter = vecArgValues.begin();
                for( ; iter != vecArgValues.end(); iter++ )
                {
                    // Append this item into the array of this JSON node.
                    root[strCategoryOptions][strOption].append( Json::Value(*iter) );
                }
            }
        }
        else
        {
            // This is a fatal error and should not happen.
            assert( false );
        }
    } // END FOR() of option names.

    // Declare a vector of strings for the non-options.
    STRINGVEC vecNonOptionList;
    // Get the non-option items.
    GetNonOptionList( vecNonOptionList );

    // Iterate the vector of non-options.
    STRINGVEC::iterator iterNonOpts = vecNonOptionList.begin();
    for( ; iterNonOpts != vecNonOptionList.end(); iterNonOpts++ )
    {
        // Append this non-option item into the non-option array of this JSON node.
        root[strCategoryNonOptions].append( Json::Value(*iterNonOpts) );    
    }

    // Get the JSON information as a string.
    const std::string strJSON( root.toStyledString() );
    
    // Open an output stream to write the JSON file.
    ofstream out( strOptionsFileName.c_str(), ios::out );
    if( out.good() )
    {
        // Write the JSON information into the output stream.
        out << strJSON;
    }
    else
    {
        ION_THROW( "Unable to write JSON file " + strOptionsFileName )
    }
}

/**
 * DefineOption information
 * 
 * @param strOptionName
 * @param strArgument
 */
void OptionArgs::DefineOption( const std::string& strOptionName,
                                const std::string& strArgument )
{
    // Declare an iterator into the the map for the return value.
    MAPOPTION::iterator iter;
    
    // Find the option name.
    bool bOptionExists = FindOption( iter, strOptionName );
    if( bOptionExists )
    {
        // Option exists, so replace the argument value.
        _mapRawOptions[ strOptionName ] = strArgument;
    }
    else
    {
        // Option does NOT exist, so add it to the map.
        // Create and fill in a Pair instance for map insertion.
        MAPPAIROPTION mapPair;
        mapPair.first = strOptionName;
        mapPair.second = strArgument;
        
        // This is the result pair for the map insertion.
        // Attempted to insert the item into the map.        
        std::pair<MAPOPTION::iterator,bool> mapRet = _mapRawOptions.insert( mapPair );
        
        // If the second value is false, then the map insertion failed.
        if( !mapRet.second )
        {
            ION_THROW( "Duplicated Option." )
        }
    }
}

