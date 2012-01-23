/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MASK_H
#define MASK_H

//#include <stdlib.h>
#include <inttypes.h>
#include "Region.h"
#include <vector>
//#include <vector>
//#include <algorithm>


#define MATCH_ANY	1
#define MATCH_ALL	2
#define MATCH_ONLY	3

//	All wells start out with MaskEmpty
//	MaskEmpty, MaskPinned, MaskBead are mutually exclusive
//	MaskLive, MaskDud, MaskAmbiguous are mutually exclusive
//	MaskTF, MaskLib are mututally exclusive
enum MaskType {
	MaskNone                = 0,
	MaskEmpty               = (1<<0),
	MaskBead                = (1<<1),
	MaskLive                = (1<<2),
	MaskDud                 = (1<<3),
	MaskAmbiguous           = (1<<4),
	MaskTF                  = (1<<5),
	MaskLib                 = (1<<6),
	MaskPinned              = (1<<7),
	MaskIgnore              = (1<<8),
	MaskWashout             = (1<<9),
	MaskExclude             = (1<<10),
	MaskKeypass             = (1<<11),
	MaskFilteredBadKey      = (1<<12),
	MaskFilteredShort       = (1<<13),
	MaskFilteredBadPPF      = (1<<14),
	MaskFilteredBadResidual = (1<<15),
	MaskAll                 = 0xffff
	// No more room at the inn - we've used up all the mask patterns that can be stored in the current bfmask.bin format, which uses 16-bit ints.
};

class Mask {
	public:
		Mask(int w, int h);
		Mask(Mask *origmask);
		Mask (const char *fileName, bool fillMask=true);
		virtual ~Mask();
		Mask() {mask = '\0';  isHex = false;}
		void SetHex(bool hex) { isHex = hex; }
		void Init(int w, int h);
		void Init(int w, int h, MaskType these);
		void Init(Mask *origmask);
		void Copy(Mask *origmask);
		int	W() const {return w;}
		int	H() const {return h;}
		int	Xoffset() const {return xOffset;}
		int	Yoffset() const {return yOffset;}
		int ToIndex(int row, int col) const { return row * w + col; }
		void IndexToRowCol(int idx, int &row, int &col) const { row = idx / w; col = idx % w; }
		const unsigned short *GetMask() {return mask;}
		uint16_t GetBarcodeId(int x, int y);
		bool	Match(int x, int y, MaskType type, int method = MATCH_ANY);
		bool	Match(int n, MaskType type, int method = MATCH_ANY);
        void    Set(int x, int y, MaskType type);
        void    SetBarcodeId(int x, int y, uint16_t barcodeId);
        void	AddThese(Mask *fromMask, MaskType these);
		void	SetThese(Mask *fromMask, MaskType these);
		int	GetCount(MaskType these);
		int	GetCount(MaskType these, Region region);
		int GetCountExact(MaskType val);
		int	Export(char *fileName, MaskType these);
		int	Export(char *fileName, MaskType these, Region region);
		int MaskList(char *fileName, MaskType these);
		int MaskList(char *fileName, MaskType these, Region region);
		int WriteRaw (const char *fileName);
		int SetMask (const char *fileName);
		int DumpStats (Region region, char *fileName, bool showWashouts = true);
		unsigned short & operator [](int n) {return mask[n];}
		unsigned short lookup(int n){return (*this)[n];}
		int32_t Crop(int32_t width, int32_t height, int32_t top,
			     int32_t left);
		void validateMask ();
		void MarkCrop (Region region, MaskType);
		void MarkRegion (Region region, MaskType);
		void CropRegions (Region *regions, int numRegions, MaskType these);
		// For now just check for 318 width.

		bool isHexPack() { return isHex || h == 3792; }
		static Region chipSubRegion;

		/**
		 * Fill in vector with indices for all of the
		 * neighbors. Start in the lower left neighbor well and
		 * continue clockwise around center well. Wells out of bounds (e.g. neighbors
		 * for edge wells) will have index -1
		 * 
		 * square packed - just square around center well.
		 *    indices are (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), 
		 * hex packed - Little more complicated as depends on if row is even or odd.
		 *    row odd   (-1, 0), *(0,-1), (1, 0), (1,1), *(0,1), (-1,1)
		 *    row even  (-1,-1), *(0,-1), (1,-1), (1,0), *(0,1), (-1,0),
		 * 
		 */
		void GetNeighbors (int row, int col, std::vector<int> &wells);

		/* Append the neighbor index or -1 for off grid as appropriate. */
		void AddNeighbor(int row, int col, int rOff, int cOff, std::vector<int> &wells);

		/* Fill in the neighbors for a hex grid starting in lower left neighbor. Note that
		   odd and even rows have different behavior */
		void GetHexNeighbors(int row, int col, std::vector<int> &wells);

		/* Fill in neigbors for a square grid starting in lower left neighbor. */
		void GetSquareNeigbors(int row, int col, std::vector<int> &wells);

		void CalculateLiveNeighbors();

		int GetNumLiveNeighbors(int row, int col);		
		
//    void OnlySomeWells(std::vector<int> mWellIdx);

	protected:
		int32_t w, h;
		int32_t xOffset, yOffset;
		uint16_t *mask;
		bool isHex;
		std::vector<char> numLiveNeighbors;
	private:
		//Mask(); // not implemented, don't call!
};

#endif // MASK_H

