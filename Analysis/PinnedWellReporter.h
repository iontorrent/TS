/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef PINNEDWELLREPORTER_H
#define PINNEDWELLREPORTER_H

#include <assert.h>
#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

#include "MutexUtils.h"

using namespace ION;

namespace PWR
{
struct PinnedWell;

/**
 * Double template class
 */
template< class T >
struct Doublet
{
    Doublet( T x = 0, T y = 0 ) : _x(x), _y(y) {}
    Doublet( const Doublet& rhs ) : _x(rhs._x), _y(rhs._y) {}
    T& X() { return _x; }
    T& Y() { return _y; }
    void Set( T x = 0, T y = 0 ) { _x = x; _y = y; }
    std::string ToStringX() { return ToString( _x ); }
    std::string ToStringY() { return ToString( _y ); }
    const Doublet& operator==( const Doublet& rhs )
        { _x = rhs._x; _y = rhs._y; }
    bool operator=( const Doublet& rhs )
        { return ( _x == rhs._x && _y == rhs._y ) ? true : false; }
protected:    
    T _x;
    T _y;    
    std::string ToString( T value )
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
};
    
/**
 * class PinnedWellReporter
 */
class PinnedWellReporter
{
public:

    enum PinnedStateEnum { LOW = 0, HIGH = 1 };
    
    virtual ~PinnedWellReporter();
    
    static PinnedWellReporter* Instance( const bool bEnabled = true );
    static void Destroy();
    static bool IsEnabled() { return !_sbDisabled; }
    
    virtual int Write( const std::string& strDatFileName,
                       PinnedWell& pinnedWellData,
                       const int nTotalPinnedFramesInWell );
    virtual int Count() const;
    virtual int GetIdByFileName( const std::string& strFileName );
    virtual std::string GetById( const int Id );
    virtual void ConsoleOut( bool bConOut = false ) { _bConsoleOut = bConOut; }

protected:

    PinnedWellReporter();
    
    void ClearMapData();    
    void OpenFilePersistentStreams();
    void WriteDatTable();
    void WriteDatData( PinnedWell& pinnedWell );

    std::map<int,std::string> _mapFileID;
    std::string _strOutputFileName;
    bool _bConsoleOut;
    
    std::ofstream _fMapTableScopedStream;
    std::ofstream _fDataPersisentStream;
    
    static PinnedWellReporter* _spThisObj;
    static bool _sbDisabled;
    static pthread_mutex_t _sMutexInst;
    
};
// END class PinnedWellReporter

/**
 * class DisabledPinnedWellReporter
 */
class DisabledPinnedWellReporter : public PinnedWellReporter
{
public:
    int Write( const std::string& strDatFileName,
               PinnedWell& pinnedWellData,
               const int nTotalPinnedFramesInWell ) { return -1; }
    int Count() const { return 0; }
    int GetIdByFileName( const std::string& strFileName ) { return 0; }
    std::string GetById( const int Id ) { return 0; }
   
};
// END class DisabledPinnedWellReporter

/**
 * PinnedWell structure holds information for each pinned well.
 */
struct PinnedWell
{
    struct PWData
    {
        PWData()
            : _id(-1)
            , _dValue(0)
            , _ePinnedState( PinnedWellReporter::LOW )
            , _iNumPinnedFrames(0)
        {
        }

        PWData( const PWData& rhs )
            : _id( rhs._id )
            , _dValue( rhs._dValue )
            , _ePinnedState( rhs._ePinnedState )
            , _iNumPinnedFrames( rhs._iNumPinnedFrames )
            , _xy( rhs._xy )
        {   
        }
        
        int& Id() { return _id; }
        int& X() { return _xy.X(); }
        int& Y() { return _xy.Y(); }
        unsigned int& Value() { return _dValue; }
        PinnedWellReporter::PinnedStateEnum& PinnedState() { return _ePinnedState; }
        int& NumPinnedFrames() { return _iNumPinnedFrames; }

        Doublet<int>& XY() { return _xy; }

        std::string ToString();
        void ToString( std::string& str );

    private:

        int _id;
        unsigned int _dValue;
        PinnedWellReporter::PinnedStateEnum _ePinnedState;
        int _iNumPinnedFrames;

        Doublet<int> _xy;
    };
    
    PinnedWell() {}
    PinnedWell( const PinnedWell& rhs );
    
    void Add( int x, int y,
              unsigned int valueInFrame,
              PinnedWellReporter::PinnedStateEnum pinnedState );
    
    void SetId( const int id );
    void SetTotalPinnedFramesInWell( const int iFramesInWell );
    
    int Count() const { return _vecPWData.size(); }
    std::vector<PWData>& GetData() { return _vecPWData; }

private:
    
    std::vector<PWData> _vecPWData;
};

}
// END PWR namespace

#endif // PINNEDWELLREPORTER_H 
