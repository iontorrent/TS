/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdio.h>
#include <limits>
#include <time.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "Logger.h"

using namespace std;
using namespace ION;

// Max file size limit for log file rotation.
const long INITIAL_MAXFILESIZE_BYTES = 100000000;

// String Constants
const std::string& cstrPath = ".";
const std::string& cstrFile = "Log";
const std::string& cstrExt = "txt";
const std::string& cstrUnderscore = "_";
const std::string& cstrPeriod = ".";

// Static variable initialization - Module Scope
std::string _sStrFileNameNoExt;    
std::string _sStrFilePath;    
std::string _sStrFileExt;

// Mutexes - Module Scope
pthread_mutex_t mutexLoadLogData = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutexLoggerThread = PTHREAD_MUTEX_INITIALIZER;

// Static variable initialization - Class scope
long LoggerBase::_lMaxLogFileSize = INITIAL_MAXFILESIZE_BYTES;
bool LoggerBase::_bEnableLogRotation = true;
std::queue<std::string> LoggerBase::_sLoggerQueue;
std::string MTLogger::_sStrLogMsg;
bool MTLogger::_bThreadTerminate = false;


/**
 * BaseLogger Ctor
 */
LoggerBase::LoggerBase()
    : _strFileNameNoExt( cstrFile )
    , _strFilePath( cstrPath )
    , _strFileExt( cstrExt )
    , _bConsole( true )
    , _bEnabled( true )
{
    // Initialize the random number generator that
    // is used to generate a unique file name if needed.
    srand( time(0) );
}

/**
 * Base class call that writes the log message to console.
 * @param strLogMsg
 */
void LoggerBase::WriteConsole( const std::string& strLogMsg )
{
    if( _bConsole )
    {
        std::cout << strLogMsg << endl;
    }
}

/**
 * Base class Write to console.
 * @param strLogMsg is the message to be sent to console.
 */
void LoggerBase::Write( const std::string& strLogMsg )
{
    WriteConsole( strLogMsg );
}

/**
 * Generate a timestamp string.
 * @return the timestamp string.
 */
std::string LoggerBase::GenerateTimeStamp()
{
    struct timeval tv;
    gettimeofday( &tv, 0 );
    
    struct tm *pTime = localtime( &tv.tv_sec );

    stringstream ss;
    ss << pTime->tm_year + 1900
       << "/"
       << ((pTime->tm_mon)+1)
       << "/"
       << pTime->tm_mday
       << " - "
       << pTime->tm_hour
       << ":"
       << pTime->tm_min
       << ":"
       << pTime->tm_sec + tv.tv_usec / 1000000.0;
            
    return ss.str();
}

/**
 * Determine if the file exists.
 * @param strPathFileName is the file name to test for existence.
 * @return true of false
 */
bool LoggerBase::FileExists( const std::string& strPathFileName )
{
    // Attempt to open the file, read the condition, then close it.
    ifstream file( strPathFileName.c_str() );
    return (file.is_open() ? true : false);
}

/**
 * Validate the log file name and modify as necessary.
 * @param strPathFileName is the log path and file name to be validated.
 */
void LoggerBase::ValidateLogFileName( std::string& strPathFileName, const std::string& strFileExt )
{
    // Return if log file rotation is disabled.
    if( !_bEnableLogRotation )
    {
        return;
    }
    
    // Save the original file name.
    const std::string strOriginalFileName( strPathFileName );
    
    // Check if the file exists in the current directory.
    bool bLogFileExists = FileExists( strPathFileName );
    
    // If it does exist, then check the file size.
    // Else it does not exist, so do nothing.
    if( bLogFileExists )
    {
        struct stat statbuf;
        stat( strPathFileName.c_str(), &statbuf );
        
        const long lLogFileSizeBytes = statbuf.st_size;
        
        // If the file size exceeds the _lMaxLogFileSize.
        // Otherwise, the file is not big enough to take any action,
        // so just continue and do nothing to the name.
        if( lLogFileSizeBytes > _lMaxLogFileSize )
        {
            // Generate a unique file and if a file exists,
            // then generate a new unique file name.
            while( FileExists( strPathFileName ) )
            {
                // The new candidate file name exists!  Impossible but true!
                // So restore the original file name and generate a new unique
                // file name.
                strPathFileName = strOriginalFileName;
                strPathFileName = GenerateUniqueFileName( strPathFileName, strFileExt );
            }
            
            // Then rename the file to the new name and
            // continue with the original file name unmodified.
            const int iRenameResult = rename( strOriginalFileName.c_str(), strPathFileName.c_str() );
            if( 0 != iRenameResult )
            {
                ION_THROW( "I/O error.  Unable to rename log file: " + strPathFileName )
            }           
        }
    }
}

/**
 * Generate a unique file name for log file rotation.
 * @param strPathFileName path and file name.
 * @param strFileExt file extension.
 * @return a uniquely file name.
 */
std::string LoggerBase::GenerateUniqueFileName( const std::string& strPathFileName,
                                                   const std::string& strFileExt )
{
    // adding a random number suffix to make it unique.
    std::string strNewLogFileName;
    strNewLogFileName += StripExtension( strPathFileName ) + cstrUnderscore;
    strNewLogFileName += GenerateFileDateString() + cstrUnderscore;
    strNewLogFileName += GenerateUniqueNumericString() + ".";
    strNewLogFileName += strFileExt;
    
    return strNewLogFileName;
}

/**
 * Append the current file extension to the string.
 * @param str is the string to which the file extension is to be appended.
 */
void LoggerBase::AppendExtension( std::string& str )
{
    str += _strFileExt;
}

/**
 * Return a string with the current file extension removed.
 * @param str string from which the file extension is to be removed.
 * @return the string without the file extension.
 */
std::string LoggerBase::StripExtension( const std::string& str )
{
    std::string strRet = str;
    
    const size_t iIndexExtStart = str.find_last_of( cstrPeriod );
    if( iIndexExtStart < str.length() )
    {
        strRet = str.substr( 0, iIndexExtStart );
    }
    
    return strRet;
}

/**
 * Return a string of 5 digits for use with naming the log file.
 * @return 
 */
std::string LoggerBase::GenerateUniqueNumericString()
{
    const long lRand = rand() % std::numeric_limits<short>().max();
    stringstream ss;
    ss << setw(5) << setfill('0') << lRand;
    
    return ss.str();
}

/**
 * Generate the file date string.
 * @return the file date string.
 */
std::string LoggerBase::GenerateFileDateString()
{
    struct timeval tv;
    gettimeofday( &tv, 0 );
    
    struct tm *pTime = localtime( &tv.tv_sec );

    stringstream ss;
    ss << setw(4) << setfill('0') << pTime->tm_year + 1900
       << setw(2) << ((pTime->tm_mon)+1)
       << pTime->tm_mday;
            
    return ss.str();
}

/**
 * Open a file and write the log message out.
 * Append if it exists, and create a new file otherwise.
 * @param strLogFileName
 * @param strLogMsg
 */
void LoggerBase::WriteMessageToFile( const std::string& strLogFileName, const std::string& strLogMsg )
{
    // Write the message to the log file.
    ofstream out( strLogFileName.c_str(), ios::app | ios::out );
    if( out.good() )
    {
        out << strLogMsg << "\r\n";
    }

    // Close the file stream.
    out.close();
}

/**
 * Enter a loop until the logger buffer is empty and all the messages have been written to file.
 */
void LoggerBase::FlushMessages()
{
    const double cdMaxTimeOutSeconds = 10.0;
    // Setup a timeout timer and get the current time.
    timeval start_time;
    timeval end_time;
    gettimeofday( &start_time, 0 );
    
    // Enter an infinite loop until all log message have been written out.
    while( HasLogMessages() )
    {
        // Sleep the thread for a moment.
        usleep( 10 );
        
        // Get the current time and calculate the difference.
        gettimeofday( &end_time, 0 );
        const double dTimeDiff = end_time.tv_sec - start_time.tv_sec
                                + static_cast<double> (end_time.tv_usec - start_time.tv_usec)
                                / (1000000.0);
        // If the time difference exceeds the MaxLimit, then break out of the loop
        // to prevent permanently freezing the system at shutdown.
        if( dTimeDiff > cdMaxTimeOutSeconds )
            break;
    }
}

/**
 * Single threaded write method.
 * @param strLogMsg is the message to be logged.
 */
void STLogger::Write( const std::string& strLogMsg )
{
    try
    {
        // Call the base class method.
        LoggerBase::Write( strLogMsg );

        // Emit output if the logging system is enabled.
        if( _bEnabled )
        {
            // Create a temporary log file name.
            std::string strLogFileName( _strFilePath + "//" + _strFileNameNoExt + "." + _strFileExt );

            // Validate the log file name.
            ValidateLogFileName( strLogFileName, _strFileExt );

            // Create the log file and write the message.
            WriteMessageToFile( strLogFileName, strLogMsg );
        }
    }
    catch( ... )
    {
        std::cout << "I/O Error:  Cannot write to log file." << endl;
    }
}

/**
 * Ctor for multi-threaded logger
 * 
 * @param strPath
 * @param strFile
 */
MTLogger::MTLogger() : _bFirstTime(true)
{
}

/**
 * Dtor for multi-threaded logger
 */
MTLogger::~MTLogger()
{
    _bThreadTerminate = true;
}

/**
 * Initialize the class
 */
void MTLogger::Init()
{
    // Do only once at startup.
    if( _bFirstTime )
    {
        // Disable any successive passes
        _bFirstTime = false;
        
        // Clear the static queue.
        while( !_sLoggerQueue.empty() )
            _sLoggerQueue.pop();
        
        // Start the separate logger thread here...
        const int iResult = pthread_create( &_pThreadLog, 0, MTLogger::ThreadFunc, (void*)0 );
        assert( iResult == 0 );
    }
}

/**
 * Multi threaded write method.
 * @param strLogMsg is the message to be logged.
 */
void MTLogger::Write( const std::string& strLogMsg )
{
    // Call the base class method.
    LoggerBase::Write( strLogMsg );

    // Emit output if the logging system is enabled.
    if( _bEnabled )
    {
        // Lock
        ScopedMutex localMtx( &mutexLoadLogData );
        
        // Copy the message to the separate thread to write.
        _sStrFileNameNoExt = _strFileNameNoExt;
        _sStrFilePath = _strFilePath;
        _sStrFileExt = _strFileExt;
        
        // Push the log message into the static queue.
        _sLoggerQueue.push( strLogMsg );
    }
}

/**
 * The logger thread function.
 * @param 
 * @return 
 */
void* MTLogger::ThreadFunc( void* /*pThreadParams*/ )
{
    // Enter an infinite loop.
    for( ;; )
    {
        // Do only if there is a message to process in the logger queue.
        if( !_sLoggerQueue.empty() )
        {
            // Lock
            ScopedMutex localMtx( &mutexLoggerThread );

            // Get the next message from the queue.
            std::string strLogMsg = _sLoggerQueue.front();

            // Create a temporary log file name.
            std::string strLog( _sStrFilePath + "//" + _sStrFileNameNoExt + "." + _sStrFileExt );

            // Validate the log file name.
            ValidateLogFileName( strLog, _sStrFileExt );

            // Create the log file and write the message.
            WriteMessageToFile( strLog, strLogMsg );
            
            // Remove the message just processed out of the queue.
            _sLoggerQueue.pop();
        }

        // Sleep for a very brief moment.
        // More sensitivity to new logger messages for smaller sleep times values.
        usleep( 10 );

        // Break out of the infinite loop if the thread terminate flag
        // is set AND there are no more message in the queue.
        // IF there are remaining messages in the queue, then process
        // them before ending this thread.
        if( _bThreadTerminate && _sLoggerQueue.empty() )
            break;
     }
    
    // Thread terminating...
    return 0;
}
