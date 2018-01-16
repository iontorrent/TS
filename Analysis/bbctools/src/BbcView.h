// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcView.h
 *
 *  Created on: Sep 21, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BBCVIEW_H_
#define BBCVIEW_H_

#include "BbcCreate.h"
#include "BbcIndex.h"
#include "BbcCoarse.h"
#include "RegionCoverage.h"

#include "api/BamAux.h"

#include <string>
#include <vector>
#include <stdint.h>
#include <cstdio>
using namespace std;

#define MEMBER_FUNCTION(obj,func) ((obj).*(func))

class BbcView
{
    public:
		BbcView(void);
		virtual ~BbcView(void);		// not really needed unless derived class deletes from a BbcView pointer

		// Close an the open BBC file - also called on destruction
		void Close(void);

		// Create a BBC coarse coverage (CBC) file
		bool CreateCbc( BbcCoarse &bbcCoarse );

		// Create a BBC index file
		bool CreateIndex( BbcIndex &indexer );

		// Return reference contigs and sizes. Empty if file not open.
		const BamTools::RefVector GetReferenceData(void) const;

		// Return totals for all reads (e.g. for report summary)
		bool GetTotalBaseReads( uint64_t &fwdReads, uint64_t &revReads, uint64_t &fwdTrgReads, uint64_t &revTrgReads );

		// Return the total number of bases considered in the last call to GetWindowsSize(), e.g. from ReadRange()
		uint64_t GetWindowSize(void) const;

		// Return the total number of bases spanned by the range, or target regions within the range if target
		// regions were set using SetRegionCoverage(). Size is recorded for subsequent calls to GetWindowsSize(void).
		uint64_t GetWindowSize( uint32_t srtContig, uint32_t srtPosition, uint32_t endContig, uint32_t endPosition );

		// Open a BBC file, check format, load header
		bool Open( const string &filename, bool test = false );

		// Parses a range string in to an explicit locus range.
		// Returns an error message string or "" for success.
		// Valid range examples are "chr1", "chr1:10001", "chr1:10001-20000", "chr1:10001-chr2:10001", "chr1-chr2", etc.
		string ParseRegionRange( const string &range,
			uint32_t &srtContig, uint32_t &srtPosition, uint32_t &endContig, uint32_t &endPosition );

		// Print formatted base coverage at given coordinates
		void PrintBaseCoverage(
			const char* contig, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType );

		// Read & stream entire contents of BBC file. Rewinds to OpenBbcFile() state.
		bool ReadAll(void);

		// Generalized reading summed coverage over region of the reference to multiple sub-regions (bins).
		// Valid range strings are those parsed by ParseRegionRange() (see above).
		// Valid bintype values are: "WIDTH", "COUNT" and "REGION".
		// Valid bins values are a fixed bin size for "WIDTH" or fixed number of bins (variable size) otherwise.
		// srtBin and endBin define a range of bins to output (if not 0). Values outside of the (calculated) number
		// of bins are assumed to be the boundaries and negative values are relative to last bin rather than first.
		// For bintype=="WIDTH", bins==0 is the same as bins==1 except output only shows single base locus
		// (rather than 1 base range) and some specific formating options either do or do not apply.
		// For "COUNT and "REGION" bins are number of bins and using bins=0 is identical to using bins=1.
		// For "REGION" the bins are such that every contig (or target region) are output to a number
		// of distinct bins (at least one even if the contig is relatively tiny such as chrM).
		// Otherwise considered the (targeted) is reference is treated as contiguous so bins may span contig boundaries
		// Returns error message or "" for success.
		string ReadRange( const string &range, bool isolateRegions = false,
			uint32_t numBins = 0, uint32_t binSize = 0, int32_t srtBin = 0, int32_t endBin = 0 );

		// Print a continuous region of base coverage from srtContig:srtPosition to endContig:endPosition-1,
		// for one base at a time using BBC View or other single base output formats.
		// Default args endPosition = 0 => to end of srtContig, endContig = 0 => endContig == srtContig.
		bool ReadRegion( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition = 0, uint32_t endContig = 0 );

		// Read & stream ranges of base coverage specified by the target regions set by SetRegionCoverage().
		// These targets are further bounded by given window locus arguments, that all have default values.
		bool ReadRegions(
			uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition = 0, uint32_t endContig = 0 );

		// Rewind the BBC file to the start of data - after header.
		void Rewind(void);

		// Set the text output stream for BBC coverage.
		void SelectPrintStream( const string &streamID );

		// Set the flag to use the regions flag for annotations only, for loaded auxiliary fields.
		void SetUseRegionAnnotation( bool use = true );

		// Set BbcCreate object for creating a new BBC file from current BBC file.
		void SetBbcCreate( BbcCreate *bbcCreate );

		// Set BbcCoarse object to assist in fast viewing of large random regions.
		void SetBbcCoarse( BbcCoarse *bbcCoarse );

		// Set BbcIndex object to assist in fast viewing of random regions.
		void SetBbcIndex( BbcIndex *bbcIndex );

		// Set RegionCoverage object for creating or viewing with respect to targeted regions.
		void SetRegionCoverage( RegionCoverage *regionCoverage );

		// Set the output header line. Commas may be translated to tabs. No whitespace removal.
		void SetHeaderLine( const string &headerLine, bool commasToTabs = true );

		// Configure printing formats to not include the on-target data field(s).
		void SetHideOnTargetCoverage( bool hide = true );

		// Configure printing formats to not include the individual region contig names.
		void SetHideContigNames( bool hide = true );

		// Configure printing formats to not include the individual region positions.
		void SetHideRegionCoordinates( bool hide = true );

		// Set flag to invert on/off target base coverage status.
		// NOTE: This is a streaming operation so affects both text output and BbcCreate
		void SetInvertOnTarget( bool invert = true );

		// Configure printing formats to not include bases that are not on-target.
		// Does not affect output for binned region coverage.
		void SetNoOffTargetPositions( bool hide = true );

		// Configure printing formats to use BED coordinates (0-base start position).
		void SetOutputBedCoordinates( bool bed = true );

		// Configure printing to ONLY output region coordinates (loci).
		// This option overrides all other print coverage formating options.
		void SetShowLociOnly( bool show = true );

		// Configure printing formats to include bases with 0 read coverage.
		// Does not affect output for binned region coverage.
		void SetShowZeroCoverage( bool show = true );

		// Configure printing formats to sum forward and reverse coverage or to display separately.
		void SetSumFwdRevCoverage( bool sum = true );

		// Return version number/string of open BBC file, or maximum version readable if none open.
		string VersionString(void);

    private:
		BamTools::RefVector m_references;
		FILE       *m_bbcfile;
		bool        m_noOffTargetPositions;
		bool        m_showOnTargetCoverage;
		bool        m_showContigNames;
		bool        m_showRegionCoordinates;
		bool        m_outputBedCoordinates;
		bool        m_showLociOnly;
		bool        m_showZeroCoverage;
		bool        m_sumFwdRevCoverage;
		bool     	m_useRegionAnnotation;
		bool        m_cbcLoaded;
		uint16_t    m_versionNumber;
		uint32_t    m_numContigs;
		uint32_t    m_contigIdx;
		uint32_t    m_firstContigIdx;
		uint32_t    m_position;
		uint32_t    m_wordsize;
		uint32_t    m_readlen;
		uint32_t    m_fcov;
		uint32_t    m_rcov;
		uint32_t    m_ontarg;
		uint32_t    m_ontargInvert;
		uint32_t    m_lastSeekContig;
		uint32_t    m_lastSeekPos;
		uint32_t    m_cbcMinorWidth;
		uint32_t    m_cbcMajorWidth;
		uint64_t    m_fcovSum;
		uint64_t    m_rcovSum;
		uint64_t    m_fcovSumTrg;
		uint64_t    m_rcovSumTrg;
		uint64_t    m_windowSize;
		long        m_bbcfileRewindPos;
		const char *m_contigStr;

	    typedef void (BbcView::*BaseCoverageStream)( const char*, uint32_t, uint32_t, uint32_t, uint32_t );

		RegionCoverage *m_regionCoverage;
		BbcCreate      *m_bbcCreate;
		BbcIndex       *m_bbcIndex;
		BbcCoarse      *m_bbcCoarse;
		BaseCoverageStream m_bcStream;

		string m_headerLine;

		// ---- BBC file version readers ----
		// If adding more probably best to derive from a base class or interface

		bool CreateIndexVersion0( BbcIndex &indexer );
		bool CreateIndexVersion1000( BbcIndex &indexer );

		bool ReadBaseCov0( uint32_t skipToPosition = 0 );
		bool ReadBaseCov1000( uint32_t skipToPosition = 0 );

		// ---- Private methods ----

		void BaseCoveragePrint( uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType );

		uint32_t *CreateContigBinSizes( uint32_t &numBins, uint32_t binSize,
			uint32_t srtContig, uint32_t srtPosition, uint32_t endContig, uint32_t endPosition );

		bool ReadFileHeader(void);

		bool ReadSum(
			uint32_t srtContig = 0, uint32_t srtPosition = 1, uint32_t endPosition = 0, uint32_t endContig = 0 );

		bool ReadSumFragment( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition,
				uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg );

		bool ReadSumNoCbc( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition = 0, uint32_t endContig = 0 );

		void PrintRegionCoverage(
			uint32_t srtContig, uint32_t srtPosition, uint32_t endContig, uint32_t endPosition,
			uint64_t fcovSum, uint64_t rcovSum, uint64_t fcovSumTrg, uint64_t rcovSumTrg, const string annotation = "" );

		bool SeekStart( uint32_t contigIdx, uint32_t position, bool softRewind = true );

		// ---- Print stream formats as member functions ----

		void BbcViewPrint(
			const char* contig, uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads );
		void CompactPrint(
			const char* contig, uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads );
		void SamDepthPrint(
			const char* contig, uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads );

		// ---- Private virtual methods allowing derived methods to process base coverage -----

		virtual void StreamCoverage( uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType );

};

#endif /* BBCVIEW_H_ */
