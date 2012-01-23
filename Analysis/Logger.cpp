/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <sstream>

#include "Logger.h"

// **NOTE**
// To use multi-threaded logger, build with LOGGER_MULTI_THREADED defined.
// Otherwise, a single-threaded logger will be used.
//#define LOGGER_MULTI_THREADED

// Namespaces used
using namespace std;
using namespace ION;

/**
 * Static class initialization
 */
Logger* Logger::_spThisObj = 0;

/**
 * Instance method.
 * @return a pointer instance of the class.
 */
Logger* Logger::Instance()
{
    // Lock this method
    static pthread_mutex_t mutexInst = PTHREAD_MUTEX_INITIALIZER;
    ScopedMutex localMtx( &mutexInst );
    
    // If the instance pointer is null, then allocated a new instance.
    // Otherwise, just return the pointer to the existing object instance.
    if( 0 == _spThisObj )
    {
        // Allocate a new instance of the class.
        _spThisObj = new Logger();
    }
    // Return a pointer to the instance.
    return _spThisObj;
}

/**
 * Destroy this singleton instance.
 */
void Logger::Destroy()
{
    delete _spThisObj;
    _spThisObj = 0;
}


/**
 * Ctor
 */
Logger::Logger() : _pLogger(0)
{
#ifdef LOGGER_MULTI_THREADED
    // Create a multi-threaded logger object.
    _pLogger = new MTLogger();
    // Initialize and start the threads here.
    _pLogger->Init();
#else
    // Create a single-thread logger.
    _pLogger = new STLogger();
#endif // END LOGGER_MULTI_THREADED
}

/**
 * Dtor
 */
Logger::~Logger()
{    
    // Flush any remaining logger messages before destruction.
    // This method blocks but will time out.
    _pLogger->FlushMessages();
    
    // Destroy the logger object.
    delete _pLogger;
    _pLogger = 0;
}

/**
 * Log method.
 * @param strLogMsg is the message to be logged.
 */
void Logger::Log( const std::string& strLogMsg )
{
    assert( _pLogger );
    
    // Prefix the message with a timestamp.
    std::string strTSMsg( LoggerBase::GenerateTimeStamp() );
    strTSMsg += ": ";
    strTSMsg += strLogMsg;

    // Write the message to the logger.
    _pLogger->Write( strTSMsg );
}

/**
 * Log method.
 * @param pLogMsg is the message to be logged.
 */
void Logger::Log( const char* pLogMsg )
{
    std::string strMsg( pLogMsg );
    Log( strMsg );
}

/**
 * @param strLogMsg is the string to send to cout.
 */
void Logger::LogConsole( const std::string& strLogMsg )
{
    assert( _pLogger );
    _pLogger->WriteConsole( strLogMsg );
}

/**
 * @param pLogMsg is the string to send to cout.
 */
void Logger::LogConsole( const char* pLogMsg )
{    
    assert( _pLogger );
    std::string strMsg( pLogMsg );
    LogConsole( strMsg );
}

/**
 * @return is a copy of the log file name string.
 */
std::string Logger::LogFileName() const
{
    assert( _pLogger );    
    return _pLogger->LogFileName();
}

/** 
 * @return is a copy of the log file name path.
 */
std::string Logger::LogFilePath() const
{
    assert( _pLogger );
    return _pLogger->LogFilePath();
}

std::string Logger::LogFileExt() const
{
    assert( _pLogger );
    return _pLogger->LogFileExt();
}

/**
 * Enable output to console stream.
 * @param bState to console stream if true.
 */
void Logger::EnableConsole()
{
    assert( _pLogger );
    _pLogger->Console() = true;
}

/**
 * Disable output to console stream.
 * @param bState to console stream if true.
 */
void Logger::DisableConsole()
{
    assert( _pLogger );
    _pLogger->Console() = false;
}

/**
 * Determines if console stream output enabled.
 * @return true if enabled, false otherwise.
 */
bool Logger::IsConsole() const
{
    assert( _pLogger );
    return _pLogger->Console();
}

/**
 * Enable the logger output.
 */
void Logger::Enable()
{
    assert( _pLogger );
    _pLogger->Enabled() = true;
}

/**
 * Disable the logger output.
 */
void Logger::Disable()
{    
    assert( _pLogger );
    _pLogger->Enabled() = false;
}

/**
 * @return true if the logger output is enabled, else false.
 */
bool Logger::IsEnabled() const
{   
    assert( _pLogger );
    bool bRet = false;
    if( _pLogger->Enabled() )
        bRet = true;
    return bRet;
}

/**
 * Set the log file name.
 * @param strFile
 */
void Logger::SetLogFileName( const std::string& strFile )
{
    assert( !strFile.empty() );
    assert( _pLogger );
    _pLogger->LogFileName() = strFile;
}

/**
 * Set the log file path.
 * @param strPath
 */
void Logger::SetLogPath( const std::string& strPath )
{
    assert( !strPath.empty() );
    assert( _pLogger );
    _pLogger->LogFilePath() = strPath;
}

/**
 * Set the file extension.
 * @param strExt
 */
void Logger::SetLogFileExt( const std::string& strExt )
{
    assert( !strExt.empty() );
    assert( _pLogger );
    _pLogger->LogFileExt() = strExt;
}

/**
 * Set the log file name.
 * @param strFile
 */
void Logger::SetLogFileName( const char* strFile )
{
    assert( 0 != strFile );
    assert( _pLogger );
    _pLogger->LogFileName() = std::string( strFile );
}

/**
 * Set the log file path.
 * @param strPath
 */
void Logger::SetLogPath( const char* strPath )
{
    assert( 0 != strPath );
    assert( _pLogger );
    _pLogger->LogFilePath() = std::string( strPath );
}

/**
 * Set the log file extension.
 * @param strExt the file extension.
 */
void Logger::SetLogFileExt( const char* strExt )
{
    assert( 0 != strExt );
    assert( _pLogger );
    _pLogger->LogFileExt() = std::string( strExt );
}

/**
 * Enable the rotation of log files exceeding maximum size.
 */
void Logger::EnableLogRotation()
{
    assert( _pLogger );
    _pLogger->LogRotation() = true;
}

/**
 * Disable log file rotation.
 */
void Logger::DisableLogRotation()
{
    assert( _pLogger );
    _pLogger->LogRotation() = false;
}

/**
 * Return the current maximum log file size in bytes.
 * @return the maximum log file size in bytes.
 */
long Logger::GetMaxFileSize()
{
    assert( _pLogger );
    return _pLogger->MaxFileSize();
}

/**
 * Set the maximum log file size in bytes.
 * @param lMaxFileSizeBytes is the log file maximum size in bytes.
 *      This value has no effect if it has a value less than zero.
 */
void Logger::SetMaxFileSizeBytes( const long lMaxFileSizeBytes )
{
    assert( _pLogger );
    if( lMaxFileSizeBytes <= 0 )
        _pLogger->MaxFileSize() = lMaxFileSizeBytes;
}

