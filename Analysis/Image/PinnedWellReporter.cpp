/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <exception>
#include <iostream>
#include "PinnedWellReporter.h"

using namespace std;
using namespace PWR;

/**
 * Constants
 */
const string strTab( "\t" );

/**
 * Static class initialization
 */
PinnedWellReporter* PinnedWellReporter::_spThisObj = 0;
bool PinnedWellReporter::_sbDisabled = false;
pthread_mutex_t PinnedWellReporter::_sMutexInst = PTHREAD_MUTEX_INITIALIZER;

/**
 * Static variable initialization
 */
static pthread_mutex_t sMutexAdd = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t sMutexWriteDatData = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t sMutexWriteDatTable = PTHREAD_MUTEX_INITIALIZER;

/**
 * Convert value of type T to an STL string.
 * @param value
 * @return 
 */
template< class T >
string ConvertT( T value )
{
   stringstream ss;
   ss << value;
   return ss.str();
}

/**
 * Copy constructor
 * @param rhs
 */
PinnedWell::PinnedWell( const PinnedWell& rhs )
{
    _vecPWData.clear();

    std::vector<PinnedWell::PWData>::const_iterator iter = rhs._vecPWData.begin();
    for( ; iter != rhs._vecPWData.end(); iter++ )
    {
        _vecPWData.push_back( *iter );
    }
}

/**
 * Add the well information to this data object.
 * @param x well coordinate.
 * @param y well coordinate.
 * @param valueInFrame well value.
 * @param pinnedState of the well.
 */
void PinnedWell::Add( int x, int y,
                      unsigned int valueInFrame,
                      PinnedWellReporter::PinnedStateEnum pinnedState )
{
    try
    {
        // Create a data structure.
        PWData pwd;
        pwd.X() = x;
        pwd.Y() = y;
        pwd.Value() = valueInFrame;
        pwd.PinnedState() = pinnedState;
                
        // Insert the data structure into the vector.
        _vecPWData.push_back( pwd );
    }
    catch( ... )
    {
        cout << "ERROR in PinnedWell::Add()" << endl;
    }
}

/**
 * Sets the id for each frame of the PinnedWell.
 * @param id is the item Id.
 */
void PinnedWell::SetId( const int id )
{
    std::vector<PinnedWell::PWData>::iterator iter = _vecPWData.begin();
    for( ; iter != _vecPWData.end(); iter++ )
    {
        iter->Id() = id;
    }
}

/**
 * Sets the total number of pinned frames in the well.
 * @param iTotalPinnedFramesInWell in the well.
 */
void PinnedWell::SetTotalPinnedFramesInWell( const int iTotalPinnedFramesInWell )
{
    std::vector<PinnedWell::PWData>::iterator iter = _vecPWData.begin();
    for( ; iter != _vecPWData.end(); iter++ )
    {
        iter->NumPinnedFrames() = iTotalPinnedFramesInWell;
    }
}

/**
 * Convert PinnedWell data to a string.
 * @param str is an out parameter reference to a string.
 */
void PinnedWell::PWData::ToString( std::string& str )
{
    str =  ConvertT<int>( _id ) + strTab;
    str += _xy.ToStringX() + strTab;
    str += _xy.ToStringY() + strTab;
    str += ConvertT<unsigned int>( _dValue ) + strTab;
    str += ConvertT<int>( _ePinnedState ) + strTab;
    str += ConvertT<int>( _iNumPinnedFrames );
}

/**
 * Convert PinnedWell data to a string.
 * @return The converted string.
 */
std::string PinnedWell::PWData::ToString()
{
    std::string str;
    ToString( str );
    return str;
}

/**
 * Default constructor
 */
PinnedWellReporter::PinnedWellReporter()
        : _strOutputFileName("./PWR")
        , _bConsoleOut( false )
{
    if( IsEnabled() )
        OpenFilePersistentStreams();
}

/**
 * Destructor
 */
PinnedWellReporter::~PinnedWellReporter()
{
    //
    // Close the data file stream, if needed.
    //
    if( IsEnabled() )
        _fDataPersisentStream.close();
    //
    // Clear the map.
    //
    ClearMapData();
}

/**
 * Static instance method to implement singleton pattern.
 * @param bEnabled flag, object disabled if false, enabled otherwise.
 * @return a pointer to an allocated instance of the object.
 */
PinnedWellReporter* PinnedWellReporter::Instance( const bool bEnabled /*= true*/ )
{
    ScopedMutex localMtx( &_sMutexInst );
    
    // If the instance pointer is null, then allocated a new instance.
    // Otherwise, just return the pointer to the existing object instance.
    if( 0 == _spThisObj )
    {
    	// Save the flag in the static variable.
    	_sbDisabled = !bEnabled;

        // If disabled is requested, then allocated a
        // disabled instance object.
        // Otherwise, allocate a new instance of the class.
        if( _sbDisabled )
            _spThisObj = new DisabledPinnedWellReporter();
        else
            _spThisObj = new PinnedWellReporter();
    }
    // Return a pointer to the instance.
    return _spThisObj;
}

/**
 * Destroy the singleton instance.
 */
void PinnedWellReporter::Destroy()
{
    // Delete the singleton object instance
    delete _spThisObj;
    _spThisObj = 0;
    // Destroy the singleton mutex.
    pthread_mutex_destroy( &_sMutexInst );
}

/**
 * Clear the map.
 */
void PinnedWellReporter::ClearMapData()
{
    static pthread_mutex_t sMutexClear = PTHREAD_MUTEX_INITIALIZER;
    ScopedMutex localMtx( &sMutexClear );
    _mapFileID.clear();
}

/**
 * Return the count of items in the map.
 * @return the count of the number of items in the map.
 */
int PinnedWellReporter::Count() const
{
    static pthread_mutex_t sMutexCount = PTHREAD_MUTEX_INITIALIZER;
    ScopedMutex localMtx( &sMutexCount );
    return _mapFileID.size();            
}

/**
 * Return the ID for the given DAT file.
 * @param strFileName of the DAT file.
 * @return the unique ID assigned to the DAT file.  Returns -1 on failure.
 */
int PinnedWellReporter::GetIdByFileName( const std::string& strFileName )
{
    static pthread_mutex_t sMutexGetIdByFileName = PTHREAD_MUTEX_INITIALIZER;
    ScopedMutex localMtx( &sMutexGetIdByFileName );
    
    int iRetId = -1;
    // Iterate the map and search for a matching ID value.
    std::map<int,std::string>::iterator iterMap;
    for( iterMap = _mapFileID.begin(); iterMap != _mapFileID.end(); iterMap++ )
    {
        // If there is a match then copy the string value.
        if( strFileName == iterMap->second )
        {
            iRetId = iterMap->first;
            break;
        }
    }    
    // Return the ID.
    return iRetId;
}

/**
 * Return the DAT file name string for the given ID value.
 * @param Id is the ID value.
 * @return the string for the given ID value.  Empty string on failure.
 */
std::string PinnedWellReporter::GetById( const int Id )
{
    static pthread_mutex_t sMutexGetById = PTHREAD_MUTEX_INITIALIZER;
    ScopedMutex localMtx( &sMutexGetById );
    
    std::string strRet;
    
    // Search for the string in the map.
    std::map<int,std::string>::iterator iter;
    iter = _mapFileID.find( Id );
    // If the item exists in the map then retrieve the ID value.
    if( iter != _mapFileID.end() )
    {
        strRet = iter->second;
    }
    // Return the string.
    return strRet;
}

/**
 * Write the output files.
 * @param strDatFileName of the base file name.
 * @param pinnedWellData
 * @param nTotalPinnedFramesInWell
 * @return the unique ID in the map for the DAT file. Returns -1 on failure.
 */
int PinnedWellReporter::Write( const std::string& strDatFileName,
                               PinnedWell& pinnedWellData,
                               const int nTotalPinnedFramesInWell )
{
    ScopedMutex localMtx( &sMutexAdd );

    // A running count of the items in the map.
    int iUniqueDATFileId = -1;
    
    try
    {
        // Search the map for the DAT file name.
        const int iRetId = GetIdByFileName( strDatFileName );
        
        // If the DAT file name does NOT exist then add it to the map.
        if( iRetId < 0 )
        {
            // Get the total item count of the map.
            iUniqueDATFileId = Count() + 1;
            
            // Add the PWRData object into the map by the DAT file name.
            pair<int,std::string> pairData( iUniqueDATFileId, strDatFileName );
            _mapFileID.insert( pairData );
            
            // We have now entries so write the DAT table to file in recreate mode.
            WriteDatTable();
            
            // Set the unique ID value for this new DAT file entry
            // into this pinned well.
            pinnedWellData.SetId( iUniqueDATFileId );
        }
        else // Then the file name does exist.
        {
            // Else, the DAT files exists in the map table,
            // so set the unique ID value for this existing
            // DAT file into this pinned well data object.
            pinnedWellData.SetId( iRetId );
        }
        
        // Set the total count of pinned frames in this well.
        pinnedWellData.SetTotalPinnedFramesInWell( nTotalPinnedFramesInWell );
        
        // Write the PinnedWell information to file in append mode.
        WriteDatData( pinnedWellData );
    }
    catch( ... )
    {
        // Error has occurred so return bad value.
        iUniqueDATFileId = -1;
    }
    // The assigned ID for the entry.
    return iUniqueDATFileId;
}

/**
 * Open the output file stream for data.
 */
void PinnedWellReporter::OpenFilePersistentStreams()
{
    try
    {
        // Assemble the PinnedWells file name.
        std::string strPinnedWellsFileName( _strOutputFileName + "_PinnedWells" );
        // Create an output file stream in append mode to accumulate records.
        _fDataPersisentStream.open( strPinnedWellsFileName.c_str(),
                                       ios_base::out | ios_base::trunc );
        
        std::string strHeaders( "ID" + strTab + "X" + strTab + "Y" + strTab
                              + "Val" + strTab + "Pin" + strTab + "#Frms" + strTab + "\n"
                              + "--" + strTab + "-" + strTab + "-" + strTab
                              + "---" + strTab + "---" + strTab + "-----" );

        // Write the string to the file stream.
        _fDataPersisentStream << strHeaders << endl;

        // Dump to console as desired.
        if( _bConsoleOut )
            cout << strHeaders << endl;                
    }
    catch( ... )
    {
        cout << "I/O ERROR: Unable to open file streams in PinnedWellReporter::OpenFileStreams()"
             << endl;
    }
}

/**
 * Write the mapping table to file.
 */
void PinnedWellReporter::WriteDatTable()
{
    ScopedMutex localMtx( &sMutexWriteDatTable );
    
    bool bRet = true;
    
    try
    {
        // Assemble the DAT table file name.
        std::string strDatTableFileName( _strOutputFileName + "_MapTable" );

        // Create an output file stream in truncate mode so a new file is created.
        _fMapTableScopedStream.open( strDatTableFileName.c_str(),
                                        ios_base::out | ios_base::trunc );

        // Do if the stream is in good condition.
        if( _fMapTableScopedStream.good() )
        {
            // Assemble a header string.
            std::string strHeaders( "ID" + strTab + "DAT File Name\n"
                                  + "--" + strTab + "-------------" );

            // Write the string to the file stream.
            _fMapTableScopedStream << strHeaders << endl;

            // Dump to console as desired.
            if( _bConsoleOut )
                cout << strHeaders << endl;
            
            // Iterate the map extracting the id
            // and storing it in a vector of keys.
            std::vector<int> vecId;
            for( std::map<int,std::string>::const_iterator itId = _mapFileID.begin();
                 itId != _mapFileID.end();
                 ++itId )
            {
                vecId.push_back( itId->first );
            }
            
            // Sort the vector of keys in ascending order.
            std::sort( vecId.begin(), vecId.end() );
            
            // Now we iterate the vector of sorted keys.
            for( std::vector<int>::const_iterator iterVec = vecId.begin();
                 iterVec != vecId.end();
                 ++iterVec )
            {
                // Get the file name for the given key from the map.
                const std::string strItem( _mapFileID[*iterVec] );

                // Assemble a string by ID and file name.
                std::string strRec;
                strRec = ConvertT<int>( *iterVec );
                strRec += strTab + strItem;

                // Write the string to the file stream.
                _fMapTableScopedStream << strRec << endl;

                // Dump to console as desired.
                if( _bConsoleOut )
                    cout << strRec << endl;
            }
        }
        else
        {
            // Status not good...
            throw std::exception();
        }
        
        // Close the file stream.
        _fMapTableScopedStream.close();
    }
    catch( ... )
    {
        bRet = false;
        cout << "I/O ERROR in PinnedWellReporter::WriteDatTable()" << endl;
    }
}

/**
 * Write the pinned well data to file.
 * @param pinnedWell is the data to write.
 */
void PinnedWellReporter::WriteDatData( PinnedWell& pinnedWell )
{
    ScopedMutex localMtx( &sMutexWriteDatData );
    
    bool bRet = true;
    
    try
    {
        // Do if the stream is in good condition.
        if( _fDataPersisentStream.good() )
        {            
            // Iterate the frames in PinnedWell.
            std::vector<PinnedWell::PWData>::iterator iter = pinnedWell.GetData().begin();
            for( ; iter != pinnedWell.GetData().end(); iter++ )
            {
                
                // Assemble the string record for the file.
                std::string strPW;
                iter->ToString( strPW );

                // Write the string to the file stream.
                _fDataPersisentStream << strPW << endl;

                // Dump to console as desired.
                if( _bConsoleOut )
                    cout << strPW << endl;
            }
        }
        else
        {
            // Status not good...
            throw std::exception();
        }
    }
    catch( ... )
    {
        bRet = false;
        cout << "I/O ERROR in PinnedWellReporter::WriteDatData()" << endl;
    }
}
