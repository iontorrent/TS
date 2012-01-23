/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef OPTIONARGS_H
#define OPTIONARGS_H

#include "OptionArgsBase.h"

namespace ION
{
/**
 * class OptionArgs
 */
class OptionArgs : public OptionArgsBase
{
public:
    
    typedef std::map<std::string, std::string> MAPOPTION;
    typedef std::pair<std::string,std::string> MAPPAIROPTION;
    
    OptionArgs() {}
    virtual ~OptionArgs() {}
    
    virtual void Process( int argc, char** argv );

    MAPOPTION& GetRawOptions() { return _mapRawOptions; }

    virtual bool GetOption( const std::string& strOptionName );

    virtual bool GetOption( const std::string& strOptionName,
                             const std::string& strDefaultValue,
                             STRINGVEC& vecArgValues );

    virtual bool GetOption( const std::string& strOptionName,
                             const std::string& strShortOptionName,
                             const std::string& strDefaultValue,
                             STRINGVEC& vecArgValues );
    
    void GetOptionList( STRINGVEC& vecOptionList );
    
    void ReadOptions( const std::string& strOptionsFileName );
    void WriteOptions( const std::string& strOptionsFileName );
    
    void DefineOption( const std::string& strOptionName,
                         const std::string& strArgument );
    
protected:
    
    virtual void DoProcessing( int argc, char **argv );
    bool FindOption( const std::string& strOptionName );
    bool FindOption( MAPOPTION::iterator& iterFound,
                      const std::string& strOptionName );
    
    MAPOPTION _mapRawOptions;
};
// END class OptionArgs


/**
 * Template function to get the options from the map.
 * 
 * @param options is the OptionArgs object instance.
 * @param strOptionName is the option name.
 * @param vecArgValues is the return vector of argument values.
 * @param strDefault is the default value to use upon failure.
 * @return True if successful, false otherwise.
 */
template< class T >
bool GetOption( OptionArgs& options,
                 const std::string& strOptionName,
                 std::vector<T>& vecArgValues,
                 const std::string strDefault = "" )
{
    // Clear the return vector of type T prior to use.
    vecArgValues.clear();

    // Declare a vector of strings.
    OptionArgsBase::STRINGVEC vecArgs;
    // Get the arguments for the option.
    bool bRet = options.GetOption( strOptionName, strDefault, vecArgs );

    // Iterate the vector of arguments returned from GetOption(),
    // converting each element to type T.
    OptionArgsBase::STRINGVEC::iterator iter = vecArgs.begin();
    for( ; iter != vecArgs.end(); iter++ )
    {
        // Create a string stream and do the conversion.
        std::stringstream sstr( *iter );        
        T value;
        sstr >> value;
        // Push this converted value to type T into the return vector.
        vecArgValues.push_back( value );
    }

    return bRet;
}

/**
 * Template function to get the options from the map.
 * 
 * @param options is the OptionArgs object instance.
 * @param strLongOptionName is the long option name.
 * @param strShortOptionName is the short option name.
 * @param vecArgValues is the return vector of argument values.
 * @param strDefault is the default value to use upon failure.
 * @return True if successful, false otherwise. 
 */
template< class T >
bool GetOption( OptionArgs& options,
                 const std::string& strLongOptionName,
                 const std::string& strShortOptionName,
                 std::vector<T>& vecArgValues,
                 const std::string strDefault = "" )
{
    // Clear the return vector prior to use.
    vecArgValues.clear();

    // Declare a vector of strings.
    OptionArgsBase::STRINGVEC vecArgs;
    // Get the arguments for the option.
    bool bRet = options.GetOption( strLongOptionName, strShortOptionName, strDefault, vecArgs );

    // Iterate the vector of arguments returned from GetOption(),
    // converting each element to type T.
    OptionArgsBase::STRINGVEC::iterator iter = vecArgs.begin();
    for( ; iter != vecArgs.end(); iter++ )
    {
        // Create a string stream and do the conversion.
        std::stringstream sstr( *iter );        
        T value;
        sstr >> value;
        // Push this converted value to type T into the return vector.
        vecArgValues.push_back( value );
    }

    return bRet;
}

}
// END namespace ION

#endif // OPTIONARGS_H

