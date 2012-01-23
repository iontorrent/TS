/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <memory.h>
#include <dirent.h>
#include <sys/stat.h>

#include "ElapsedTimer.h"
#include "PinnedWellReporter.h"
#include "Image.h"

using namespace std;
using namespace PWR;

// String constants
const std::string cstrTXT = "txt";
const std::string cstrDAT = "dat";

/**
 * struct PWCommandLine holds the methods to process the command line.
 */
struct PWCommandLine
{
    /**
     * Display the usage for the application.
     */
    void DisplayUsage()
    {
        cout << "Usage: PinnedWells [DAT file directory]" << endl;
        cout << "    or PinnedWells <file 1> <file 2> ... <file N>" << endl;
    }

    /**
     * Determines whether the file has the provided extension.
     * @param strExt is the extension to for which to test.
     * @param strFileName is the file name to test.
     * @return true if the file has the matching extension, false otherwise.
     */
    bool HasFileNameExtension( const string& strExt, const string& strFileName )
    {
        bool bRet = false;
        
        // Make a lower string copy of the parameter.
        string str( strFileName );    
        std::transform( str.begin(), str.end(), str.begin(), ::tolower );
        
        // Search for the file extension delimiter.
        size_t foundAt = str.find_last_of( "." );

        // Extract the extension substring.
        const string strFExt = str.substr( foundAt + 1 );

        // Return true if there is a match.
        if( strFExt == strExt )
            bRet = true;

        return bRet;
    }

    /**
     * Determines if the file name is a directory.
     * @param strFileName is the file name to be tested.
     * @return true if this file is a directory, false otherwise.
     */
    bool IsDirectory( const std::string& strFileName )
    {
        bool bRet = false;

        // Call the stat() file status C++ API function.
        struct stat filestat;
        stat( strFileName.c_str(), &filestat );
        
        // Return true if the status of the file is a directory.
        if (S_ISDIR( filestat.st_mode ))
            bRet = true;

        return bRet;
    }

    /**
     * Process the file names in the directory.
     * @param strDirName is the directory name.
     * @param vecDATFiles is a vector of the files in the directory.
     * @return true if successful, otherwise false.
     */
    bool HandleAsDirectory( const std::string& strDirName, vector<string>& vecDATFiles )
    {
        bool bRet = true;

        // Open the directory.
        DIR* pDir = opendir( strDirName.c_str() );
        struct dirent *pDirEntry = 0;
        
        // Iterate each file in the directory.
        while( 0 != ( pDirEntry = readdir( pDir ) ) )
        {
            // Assemble the full path and file name string.
            string strFName( strDirName );
            strFName += "/";
            strFName += pDirEntry->d_name;
            
            // Test if this new file name is a directory.
            // Continue if it is not a directory.
            // We do not recursively continue into a subdirectory.
            if( !IsDirectory( strFName ) )
            {
                // Test if this file has the desired extension.
                const bool bIsDAT = HasFileNameExtension( cstrDAT, strFName );
                
                // If it does, then add it to the file name collection.
                if( bIsDAT )
                    vecDATFiles.push_back( strFName );
            }
        }
        
        // Close the directory from opendir().
        closedir( pDir );
        
        return bRet;
    }
    
    /**
     * Validates the command line arguments.
     * @param argc is the command line argument count.
     * @param argv is the array of command line items.
     * @param vecDATFiles is the vector of command line items.
     * @return true if successful, false otherwise.
     */
    bool ValidateArgs( const int argc, char** argv, vector<string>& vecDATFiles )
    {
        bool bRet = false;

        // Clear the vector of file names.
        vecDATFiles.clear();

        try
        {
            // Throw if there are not enough command line args.
            if( argc < 2 )
            {
                throw runtime_error( "Invalid argument count." );
            }    
            else
            {
                // Otherwise, we have enough args, so we process each of them.
                for( int i = 1; i < argc; i++ )
                {
                    // Create a temporary string for this arg.
                    string str( argv[i] );
                    
                    // If the name is a directory, then process the directory.
                    if( IsDirectory( str ) )
                    {
                        // Process the file in the directory.
                        HandleAsDirectory( str, vecDATFiles );
                    }
                    else
                    {
                        // Process the argument as a file name.
                        const bool bIsDAT = HasFileNameExtension( cstrDAT, str );
                        
                        // Add this file name to the vector of file names.
                        if( bIsDAT )
                            vecDATFiles.push_back( str );
                    }
                } // END for() loop
                
                // Sort the collection of file names to be processed.
                std::sort( vecDATFiles.begin(), vecDATFiles.end() );

                bRet = true;
            }
        }
        catch(  std::runtime_error& e )
        {
            cout << e.what() << endl;
            DisplayUsage();
        }
        catch( ... )
        {
            DisplayUsage();
        }

        return bRet;
    }
};
// END PWCommandLine

/*
 * Main
 */
int main( int argc, char** argv )
{
    ION::ElapsedTimer timer;
    timer.Start();
    
    // The number of files processed.
    int nFiles = 0;
        
    try
    {
        // The command line object.
        PWCommandLine pwCmdLine;
        // The vector of files to be processed.
        vector<string> vecDATFiles;

        // Process and validate the command line entries.
        const bool bValidArgs = pwCmdLine.ValidateArgs( argc, argv, vecDATFiles );
        if( !bValidArgs )
            return 1;
        
        // Continue if there are files to process, otherwise, return a message.
        nFiles = vecDATFiles.size();
        if( nFiles < 1 )
        {
            cout << "Nothing to process... Done." << endl;
            return 0;
        }

        // Create an image object.
        Image iImage;

        // Iterate each file supplied from the command line.
        vector<string>::iterator fileIter = vecDATFiles.begin();
        for( ; fileIter != vecDATFiles.end(); fileIter++ )
        {
            // Load the file name and process the file.
            iImage.LoadRaw( fileIter->c_str(), 0, true, false );
        }
    }
    catch( std::runtime_error& e )
    {
        // There was some kind of runtime error.
        std::cout << "Exception: " << e.what() << std::endl;
        return 0;
    }
    catch( ... )
    {
        // There was some kind of unknown error.
        std::cout << "Exception caught in main(). " << std::endl;
        return 0;
    }
    
    timer.Stop();
    
    cout << endl << nFiles << " files processed." << endl;
    cout << "Total elapsed time: " << timer.GetActualElapsedSeconds() << " seconds." << endl;
    cout << "Done..." << endl;
    
    return 0;
}
