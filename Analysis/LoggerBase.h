/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef LOGGERBASE_H
#define LOGGERBASE_H

// STL headers
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
// OS headers
#include <pthread.h>
// App headers
#include "MutexUtils.h"

namespace ION
{
/**
 * Basic logger behaviors.
 * An abstract base class.
 */
struct LoggerBase
{
    LoggerBase();
    virtual ~LoggerBase() {}

    virtual void Init() {}    
    
    void WriteConsole( const std::string& strLogMsg );    
    virtual void Write( const std::string& strLogMsg ) = 0;

    static std::string GenerateTimeStamp();

    std::string& LogFileName() { return _strFileNameNoExt; }
    std::string& LogFilePath() { return _strFilePath; }
    std::string& LogFileExt() { return _strFileExt; }
    
    bool& Console() { return _bConsole; }
    bool& Enabled() { return _bEnabled; }
    
    bool& LogRotation() { return _bEnableLogRotation; }
    long& MaxFileSize() { return _lMaxLogFileSize; }
    
    static bool FileExists( const std::string& strPathFileName );
    
    void FlushMessages();
    bool HasLogMessages() { return _sLoggerQueue.size() > 0; }
    
protected:

    static std::string GenerateUniqueNumericString();
    static std::string GenerateFileDateString();    
    static std::string GenerateUniqueFileName( const std::string& strPathFileName, const std::string& strFileExt );
    static std::string StripExtension( const std::string& str );
    static void ValidateLogFileName( std::string& strPathFileName, const std::string& strFileExt );
    static void WriteMessageToFile( const std::string& strLogFileName, const std::string& strLogMsg );
    
    static std::queue<std::string> _sLoggerQueue;
    static long _lMaxLogFileSize;
    static bool _bEnableLogRotation;
    
    void AppendExtension( std::string& str );
    
    // TODO: Make these static variables to optimize performance.
    std::string _strFileNameNoExt;
    std::string _strFilePath;
    std::string _strFileExt;
    
    bool _bConsole;
    bool _bEnabled;
};
// END struct LoggerBase

/**
 * Single threaded logger class.
 */
struct STLogger : public LoggerBase
{
    STLogger() {}
    virtual void Write( const std::string& strLogMsg );
};

/**
 * Multi threaded logger class.
 */
struct MTLogger : public LoggerBase
{
    MTLogger();
    virtual ~MTLogger();
    void Init();
    virtual void Write( const std::string& strLogMsg );
protected:
    static void *ThreadFunc( void *pThreadParams );
private:
    static std::string _sStrLogMsg;
    static bool _bThreadTerminate;
    
    pthread_t _pThreadLog;
    bool _bFirstTime;
};

}
// END namespace ION

#endif // LOGGERBASE_H

