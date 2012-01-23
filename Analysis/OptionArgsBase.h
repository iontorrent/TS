/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef OPTIONARGSBASE_H
#define OPTIONARGSBASE_H

#include <map>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>

#include <string.h>
#include "IonUtils.h"

namespace ION
{
/**
 * Abstract base class for command line option processing.
 */
class OptionArgsBase
{
public:
    
    typedef std::vector<std::string> STRINGVEC;

    OptionArgsBase() {}
    virtual ~OptionArgsBase() {}
    
    virtual void Initialize() {}
    
    virtual void Process( int argc, char** argv );    
    virtual bool ProcessNX( int argc, char** argv );
    
    STRINGVEC& GetNonOptions() { return _vecNonOptions; }    
    std::string GetRawCmdLine() { return _strRawCmdLine; }
    
    void GetNonOptionList( STRINGVEC& vecNonOptionList );

    void SplitString( const std::string &str,
                       char delim,
                       STRINGVEC& vecSplitItems );
    
protected:
  
    virtual void DoProcessing( int argc, char **argv ) = 0;
    
    STRINGVEC _vecNonOptions;
    std::string _strRawCmdLine;
};
// END class OptArgsBase

/**
 * Converts a string to type T
 * @param str to be converted.
 * @return the value of the converted type.
 */
template< class T >
T FromString( std::string& str )
{
    T retValue;
    std::stringstream ss( str );
    ss >> retValue;
    return retValue;
}

/**
 * Convert value to a string.
 * @param value
 * @return 
 */
template< class T >
std::string ToString( T value )
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

}
// END namespace ION

#endif // OPTIONARGSBASE_H 
