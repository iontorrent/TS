/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>

#include "IonUtils.h"
#include "MutexUtils.h"
#include "LoggerBase.h"

// Uncomment this compile flag to emit the
// file and line number of logging location.
#define DBG_LOGGER

namespace ION
{
/**
 * Class Logger
 */
class Logger
{
public:
    
    static Logger* Instance();
    ~Logger();

    static void Destroy();
    
    void Log( const std::string& strLogMsg );
    void Log( const char* pLogMsg );    
    
    void LogConsole( const std::string& strLogMsg );
    void LogConsole( const char* pLogMsg );
    
    void EnableConsole();
    void DisableConsole();
    bool IsConsole() const;
    
    void Enable();
    void Disable();
    bool IsEnabled() const;
    
    void EnableLogRotation();
    void DisableLogRotation();
    
    long GetMaxFileSize();
    void SetMaxFileSizeBytes( const long lMaxFileSizeBytes );
    
    std::string LogFileName() const;
    std::string LogFilePath() const;
    std::string LogFileExt() const;
    
    void SetLogFileName( const std::string& strFile );
    void SetLogPath( const std::string& strPath );
    void SetLogFileExt( const std::string& strExt );
    void SetLogFileName( const char* strFile );
    void SetLogPath( const char* strPath );
    void SetLogFileExt( const char* strExt );
    
protected:
    
    static Logger* _spThisObj;
    LoggerBase *_pLogger;

    Logger();
};
// END class Logger

/**
 * LoggerLifeTimeManager is a structure to implement RAII.
 * Create an instance of this at the top of main() in your app.
 * At the end of main(), this object leaves scope and the
 * Logger::Destroy() method is automatically called.
 * This will properly shutdown the Logger system and will wait
 * to flush any remaining logger messages to file before
 * destruction completes.
 */
struct LoggerLifeTimeManager
{
    LoggerLifeTimeManager() { ION::Logger::Instance(); }
    ~LoggerLifeTimeManager() { ION::Logger::Instance()->Destroy(); }
};

/**
 * Logging Macros
 */
#define LOG( strMsg ) ION::Logger::Instance()->Log( std::string( strMsg ) );
}
// END namespace ION

#endif // LOGGER_H

