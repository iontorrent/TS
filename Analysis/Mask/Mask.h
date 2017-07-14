/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MASK_H
#define MASK_H

#include <cinttypes>
#include "Region.h"
#include <vector>
#include "Serialization.h"

#define MATCH_ANY 1
#define MATCH_ALL 2
#define MATCH_ONLY 3

// All wells start out with MaskEmpty
//      MaskExclude marks wells outside the area of interest on the chip
//      MaskEmpty marks wells estimated as empty of any bead
//      MaskBead marks wells estimated as having a bead
// MaskEmpty, MaskExclude, MaskBead are mutually exclusive
// MaskLive, MaskDud are mutually exclusive and a subset of MaskBead
// MaskTF, MaskLib are mutually exclusive and a subset of MaskLive
//      MaskReference marks wells used in reference trace removed prior to fitting
//      MaskPinned is mutually exclusive of other flags (why?)
enum MaskType {
  MaskNone                = 0,
  MaskEmpty               = ( 1<<0 ),
  MaskBead                = ( 1<<1 ),
  MaskLive                = ( 1<<2 ),
  MaskDud                 = ( 1<<3 ),
  MaskReference           = ( 1<<4 ),
  MaskTF                  = ( 1<<5 ),
  MaskLib                 = ( 1<<6 ),
  MaskPinned              = ( 1<<7 ),
  MaskIgnore              = ( 1<<8 ),
  MaskWashout             = ( 1<<9 ),
  MaskExclude             = ( 1<<10 ),
  MaskKeypass             = ( 1<<11 ),
  MaskFilteredBadKey      = ( 1<<12 ),
  MaskFilteredShort       = ( 1<<13 ),
  MaskFilteredBadPPF      = ( 1<<14 ),
  MaskFilteredBadResidual = ( 1<<15 ),
  MaskAll                 = 0xffff
  // No more room at the inn - we've used up all the mask patterns that can be stored in the current bfmask.bin format, which uses 16-bit ints.
};



class Mask
{
  public:
    Mask ( int w, int h );
    Mask ( Mask *origmask );
    Mask ( const char *fileName, bool fillMask=true );
    virtual ~Mask(){}
    Mask() {
      mask.clear();
    }

    void Init ( int w, int h );
    void Init ( int w, int h, MaskType these );
    void Init ( Mask *origmask );
    void Copy ( Mask *origmask );
    int W() const {
      return w;
    }
    int H() const {
      return h;
    }
    int Xoffset() const {
      return xOffset;
    }
    int Yoffset() const {
      return yOffset;
    }
    int ToIndex ( int row, int col ) const {
      return row * w + col;
    }
    void IndexToRowCol ( int idx, int &row, int &col ) const {
      row = idx / w;
      col = idx % w;
    }
    const unsigned short *GetMask() const {
      return &mask[0];
    }
    uint16_t GetBarcodeId ( int x, int y ) const;
    bool Match ( int x, int y, MaskType type, int method ) const;
    bool Match ( int n, MaskType type, int method ) const;

    inline bool Match(int x, int y, MaskType type) const
    {
    	int n = y*w+x;

		if ( n < 0 || n >= ( w*h ) )
			return false;

		return ( ( mask[n] & type ? true : false ) );
    }

    inline bool Match(int n, MaskType type) const
    {
		if ( n < 0 || n >= ( w*h ) )
			return false;

		return ( ( mask[n] & type ? true : false ) );
    }


    void    Set ( int x, int y, MaskType type );
    void    SetAll(MaskType type );
    void    SetBarcodeId ( int x, int y, uint16_t barcodeId );
    void AddThese ( Mask *fromMask, MaskType these );
    void SetThese ( Mask *fromMask, MaskType these );
    int GetCount ( MaskType these ) const;
    int GetCount ( MaskType these, Region region ) const;
    int GetCountExact ( MaskType val ) const;
    int Export ( char *fileName, MaskType these ) const;
    int Export ( char *fileName, MaskType these, Region region ) const;
    int MaskList ( char *fileName, MaskType these ) const;
    int MaskList ( char *fileName, MaskType these, Region region ) const;
    int WriteRaw ( const char *fileName ) const;
    int SetMask ( const char *fileName );
    int SetMaskFullChipText(const char *fileName, int offset_x, int offset_y, int size_x, int size_y);

    void LoadMaskAndAbortOnFailure(const char *maskFileName);
    void UpdateBeadFindOutcomes( Region &wholeChip, char const *maskFileName, bool not_single_beadfind, int update_stats, char const *maskStatsName);
    int DumpStats ( Region region, char *fileName, bool showWashouts = true ) const;
    unsigned short & operator [] ( int n ) {
      return mask[n];
    }
    unsigned short lookup ( int n ) {
      return ( *this ) [n];
    }
    int32_t Crop ( int32_t width, int32_t height, int32_t top,
                   int32_t left );
    void validateMask () const;
    void MarkCrop ( Region region, MaskType );
    void MarkRegion ( Region region, MaskType );
    void CropRegions ( Region *regions, int numRegions, MaskType these );

//    void OnlySomeWells(std::vector<int> mWellIdx);

    std::vector<uint16_t> mask;

protected:
    int32_t w, h;
    int32_t xOffset, yOffset;
    //uint16_t *mask;
    std::vector<char> numLiveNeighbors;

  private:
    //Mask(); // not implemented, don't call!

    friend class boost::serialization::access;
    template<typename Archive>
      void serialize(Archive& ar, const unsigned version) {
      // fprintf(stdout, "Serialize Mask ... ");
      ar & 
	w & h &
	xOffset & yOffset &
	mask &
	numLiveNeighbors;
      // fprintf(stdout, "done Mask\n");
    }
    
};

#endif // MASK_H

