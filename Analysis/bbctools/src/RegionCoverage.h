// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * RegionCoverage.h
 *
 *  Created on: Aug 27, 2015
 *      Author: Guy Del Mistro
 */

#ifndef REGIONCOVERAGE_H_
#define REGIONCOVERAGE_H_

#include "api/BamAlignment.h"

#include <string>
#include <vector>
#include <map>
#include <stdint.h>
using namespace std;

class RegionCoverage
{
    public:
		RegionCoverage( const BamTools::RefVector& references );
		virtual ~RegionCoverage(void);

    protected:
        struct TargetRegion {
        	uint32_t       trgIdx;
        	uint32_t       trgSrt;
            uint32_t       trgEnd;
            vector<string> auxData;
            void          *extData;
           	uint32_t      *covAtReads;
            TargetRegion  *next;

            TargetRegion( uint32_t srt, uint32_t end, const vector<string>& aux )
                : trgIdx(0)
                , trgSrt(srt)
                , trgEnd(end)
            	, auxData(aux)
            	, extData(NULL)
                , covAtReads(NULL)
            	, next(NULL) {}

			~TargetRegion() {
				delete [] covAtReads;
			}
        };

        struct TargetContig {
			TargetContig( string nID = "", int32_t len = 0 )
				: id(nID)
				, length(len)
				, numRegions(0)
				, fwdReads(0)
				, revReads(0)
				, fwdTrgReads(0)
				, revTrgReads(0)
				, targetRegionHead(NULL) {}

			~TargetContig(void) {
				TargetRegion *nrg = targetRegionHead;
				while(nrg) {
					TargetRegion *next = nrg->next;
					delete nrg;
					nrg = next;
				}
			}

			// Resulting link list starting at targetRegionHead will be in reverse order of target regions.
			// Assumes that most calls will add (this)
			// reverse order f to top of the list given pre-ordered target regions.
			void AddRegion( TargetRegion *nrg ) {
				if( !nrg ) return;
				nrg->trgIdx = ++numRegions;	// initially set to order loaded
				if( !targetRegionHead ) {
					targetRegionHead = nrg;
					return;
				}
				TargetRegion *prev = NULL;
				TargetRegion *cur  = targetRegionHead;
				size_t srt = nrg->trgSrt;
				while( cur && srt < cur->trgSrt ) {
					prev = cur;
					cur = cur->next;
				}
				// find previous region behind this based on equal trgSrt but behind trgEnd
				while( cur && srt == cur->trgSrt && nrg->trgEnd < cur->trgEnd ) {
					prev = cur;
					cur = cur->next;
				}
				// insert after previous
				if( prev ) prev->next = nrg;
				else targetRegionHead = nrg;
				nrg->next = cur;
			}

			// convert reverse ordered linked list to be forward ordered and reset region index
			void ReverseSort( uint32_t idxOffset = 0 ) {
				TargetRegion *prv = NULL, *nxt;
				idxOffset += numRegions;
				for( TargetRegion *cur = targetRegionHead; cur; cur = nxt ) {
					cur->trgIdx = idxOffset--;
					nxt = cur->next;
					cur->next = prv;
					prv = cur;
				}
				targetRegionHead = prv;
			}

			string id;
			int32_t length;
			uint32_t numRegions;
			uint64_t fwdReads;
			uint64_t revReads;
			uint64_t fwdTrgReads;
			uint64_t revTrgReads;
			TargetRegion *targetRegionHead;
        };

        size_t               m_numRefContigs;
        TargetContig       **m_contigList;
        map<string,size_t>   m_contigIdx;
        string               m_headerLine;


        // on-target iteration tracking
        uint32_t             m_bcovRegionPos;
        uint32_t             m_bcovContigIdx;
        TargetRegion        *m_bcovRegion;
        uint32_t             m_rcovContigIdx;
        TargetRegion        *m_rcovRegion;
        TargetRegion        *m_lastRegionAssigned;

        // common optional depth at coverage stats
        size_t      m_numAuxFields;
        size_t      m_ncovDepths;
        vector<int> m_covAtDepth;

        void     AddCovAtDepth( TargetRegion *tr, int totReads );
        uint32_t BaseOnRegion( uint32_t contigIdx, uint32_t position );
        uint32_t ReadOnRegion( uint32_t contigIdx, uint32_t readSrt, uint32_t readEnd );

    public:

        // Clear out all contigs in memory: Allows Load() to be re-enterant
        void Clear(void);

    	// Returns an array of size numBins (or more) containing the contiguous number of bases in target
        // regions within the given range window, or NULL if an error occurs (e.g. no regions in window).
        // If binSize > 0 these bins will be this size or less, the regions being divided in seveal bins
        // of this size if greater than an individual region size.
        // If binSize == 0 all the regions will be divided into N bins such that no two regions are in the same bin.
    	// numBins is automatically reset to the number of target regions in the window if initially smaller.
        // The individual regions are pulled using SetCursorOnRegion()/PullSubRegion().
        uint32_t *CreateTargetBinSizes( uint32_t &numBins, uint32_t binSize,
        	uint32_t srtContig = 0, uint32_t srtPosition = 1, uint32_t endPosition = 0, uint32_t endContig = 0 );

        // Return a comma separated list for up to maxValues of (first) loaded auxiliary fields spanned by
        // the given locus. If more than maxValues then return <first>,...(N)...,<last> (N>1).
        string FieldsOnRegion( uint32_t contigIdx, uint32_t readSrt, uint32_t readEnd, uint32_t maxValues = 3 );

        // Return the target index for the region the last call to ReadOnRegion() mapped to, or 0 if none
        uint32_t GetLastReadOnRegionIdx(void) const;

        // An iterator of all target regions returning false for a call after the last region is returned.
        // The optional start argument may be used to reset this iterator to the first region.
        // Note that srtPositon is returned as a 0-based integer (etc.) for intended usage with the BamMultiReader::SetRegion().
        bool GetNextRegion( int &contigIdx, int &srtPosition, int &endPosition, bool start = false );

        // Return the total length of targets within the given region boundaries
        uint32_t GetTargetedWindowSize(
        	uint32_t srtContig = 0, uint32_t srtPosition = 1, uint32_t endPosition = 0, uint32_t endContig = 0 );

        // Load a complete targets file into memory.
        // Return error message on failure or "" for standard failure.
        // Or throw runtime_error() for arguments error, etc.
        string Load(
        	const string& fileName,
        	const string& fileType = "BED",
        	const string& auxFields = "3,-1",
        	const string& trgFields = "0,1,2" );

        // Return the sub-region for the next pullSize bases (or less) from the current base coverage iterator.
        // Used after a call to SetCovAtDepths(). The return value is the size of the region pulled (<= pullSize).
        // (The size pulled depends on how long the current region is and where the last pull left off.)
        // Special case for pullSize == 0: Return size for the current (partial) region and move to next.
        // firstContigRegion returns true if this is the first region of a new contig
        uint32_t PullSubRegion(
        	uint32_t pullSize, uint32_t &contig, uint32_t &srtPos, uint32_t &endPos, bool &lastContigRegion );

        // Sets the specific depths at which numbers of bases at those depths are covered
        // Should be comma-separated list of integers depths or "".
        void SetCovAtDepths( const string &depths );

        // Set the base coverage iterator cursor to the region at or the first beyond the given locus.
        // The position within or at the start of first region beyond is also saved (=> subregion start)
        // Returns false if locus is out of bounds or there is no region beyond the locus passed.
        bool SetCursorOnRegion( uint32_t contigIdx, uint32_t position );

        // Set up targets as whole reference contigs.
        // Alternative to Load() method (when no targets file)
        void SetWholeContigTargets(void);

        // Print all target regions and all statistics attached to them
        void Write(
            const string &filename = "STDOUT",
        	const string &columnTitles = "contig_id,contig_srt,contig_end,region_id,attributes" );

        // Print all target contigs with total/on-target read counts
        void WriteSummary(
            const string &filename = "STDOUT",
        	bool invertReadsOnTarget = false,
        	const string &columnTitles = "contig_id,fwd_reads,rev_reads,fwd_trg_reads,rev_trg_reads" );

        virtual uint32_t BaseDepthOnRegion( uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads );

        virtual void TrackReadsOnRegion( const BamTools::BamAlignment &aread, uint32_t endPos = 0 );

        virtual string ReportRegionStatistics( TargetRegion *region );
};

#endif /* REGIONCOVERAGE_H_ */
