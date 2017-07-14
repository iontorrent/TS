// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
//============================================================================
// Name        : BbcMain.cpp
// Author      : Guy Del Mistro
// Version     :
// Description : bbctools
//============================================================================

#include <api/BamMultiReader.h>
#include <api/BamIndex.h>
using namespace BamTools;

#include "BbcUsage.h"
#include "BbcUtils.h"

#include "RegionStatistics.h"
#include "AmpliconRegionStatistics.h"
#include "BaseCoverage.h"
#include "BbcCreate.h"
#include "BbcView.h"
#include "BbcIndex.h"
#include "BbcDepth.h"
#include "TrackReads.h"

// Number alignments returned using SetRegion() appears incorrect when using closed regions!
// It appears to work when using right-open regions but has MASSIVE performance issue.
// Direct index Jump() works! (With care t avoid reviewing reads more than once.)
const bool s_useBamReaderJump = true;

// A large initial value is used since making lots of Jump() calls is itself inefficient
const int s_initialMinJumpLen = 1000000;

// Read length is tracked as an offset to target for Jump() - this does not appear to be necessary?
const int s_initialMaxReadLen = 1000;

// Current version number string for bbctools executable
const uint16_t s_versionNumber = 1103;

// Forward declarations of functions used by main()
int bbctools_create( BbcUtils::OptParser &optParser );
int bbctools_report( BbcUtils::OptParser &optParser );
int bbctools_view( BbcUtils::OptParser &optParser );
int bbctools_version( BbcUtils::OptParser &optParser );

int main( int argc, char* argv[] ) {
	//
	// general command line argument validation
	//
 	if( argc <= 1 ) {
		Usage("");
		return 1;
	}
	int (*subcmdFunc)(BbcUtils::OptParser &optParser);
	string subcmd = argv[1];
	string parseString;
	int minArgs = 1, maxArgs = 1;
	if( subcmd == "create" ) {
		subcmdFunc = &bbctools_create;
		parseString =  "A=annotationFields:B=bbc:C=covStats:D=covDepths:E=e2eGap,L=minAlignLength,M=minPcCov;";
		parseString += "O=readOrigin:P=primerLength,Q=minMAPQ,R=regions:S=sumStats:T=readType:a=autoCreateBamIndex ";
		parseString += "b=onTargetBases c=coarse i=index d=noDups r=onTargetReads s=samdepth u=unique";
		maxArgs = 0;
	} else if( subcmd == "report" ) {
		subcmdFunc = &bbctools_report;
		parseString =  "B=strandBiasThreshold;C=coverage:D=covDepths:M=statsMinReads,";
		parseString += "R=regions:U=uniformityThreshold;g=genomeReport t=reportOnTarget";
		maxArgs = 2;
	} else if( subcmd == "view" ) {
		subcmdFunc = &bbctools_view;
		parseString =  "A=annotationFields:B=binsize,E=endBin,H=headerLine:N=numbins,R=regions:S=startBin,";
		parseString += "b=bedCoordinates c=compactCoverage i=isolateRegions l=lociOnly n=omitContigNames ";
		parseString += "p=omitContigPositions s=sumStrandCoverage t=omitTargetCoverage z=includeZeroCoverage";
		maxArgs = 2;
	} else if( subcmd == "version" ) {
		subcmdFunc = &bbctools_version;
		parseString = "b=brief";
		minArgs = 0;
	} else {
		Usage(subcmd);
		return 1;
	}
	BbcUtils::OptParser optParser;
	string helpOpt = optParser.PreParseHelpOption( argc-2, argv+2 );
    if( !helpOpt.empty() ) {
    	Usage( subcmd, helpOpt.substr(0,2)=="--" );
    	return 0;
    }
	const string &errMsg = optParser.Parse( argc-2, argv+2, parseString, minArgs, maxArgs );
    if( !errMsg.empty() ) {
    	cerr << "Error parsing bbctools " << subcmd << " command arguments: " << endl << "  " << errMsg << endl << endl;
    	Usage(subcmd);
    	return 1;
    }
    // Run and check for further user input error if unsuccessful
    int status = subcmdFunc(optParser);
    if( status < 0 ) {
    	Usage(subcmd);
    	status = 1;
    }
    return status;
}

//
// Functions to handle bbctools commands
// All return 0 (success), 1 (run time failure), or -1 (input/parsing error).
//

int bbctools_create( BbcUtils::OptParser &optParser ) {
    const vector<string> cmdArgs = optParser.getArgs();

	// remove .bbc extension from bbc file root, if present
	string bbcfileRoot = optParser.getOptValue( "bbc" );
	int i = bbcfileRoot.size() - 4;
	if( i > 0 && bbcfileRoot.substr(i,4) == ".bbc" ) {
		bbcfileRoot = bbcfileRoot.substr(0,i);
	}
	bool f_bci = optParser.getOptBoolean("index");
	bool f_cbc = optParser.getOptBoolean("coarse");

	string targetRegions   = optParser.getOptValue("regions");
	string annotationFields = optParser.getOptValue( "annotationFields");
	vector<string> auxRegionSplit = BbcUtils::mapToPairList(annotationFields);

	string  sumstatsFile = optParser.getOptValue("sumStats");
	string  covstatsFile = optParser.getOptValue("covStats");
	string  readOrigFile = optParser.getOptValue("readOrigin");
	string  readType     = optParser.getOptValue("readType");
	string  covDepths    = optParser.getOptValue("covDepths","-");
	double  minPcCov     = optParser.getOptNumber("minPcCov");
	int32_t primerLength = optParser.getOptInteger( "primerLength", (readType == "AmpliSeq" ? 30 : 0) );
	int32_t maxE2eEndGap = optParser.getOptInteger( "e2eGap", (readType == "AmpliSeq" ? 2 : 0) );

	bool   autoCreateBamIndex = optParser.getOptBoolean("autoCreateBamIndex");
	bool     samdepth         = optParser.getOptBoolean("samdepth");
	int32_t  filterQuality    = optParser.getOptInteger("minMAPQ");
	int32_t  minAlignLength   = optParser.getOptInteger("minAlignLength");
	bool     filterDuplicates = optParser.getOptBoolean("noDups");
	bool     filterUnique     = optParser.getOptBoolean("unique");
	uint32_t skipFlag         = filterDuplicates ? 0x704 : 0x304;
	uint16_t minMapQuality    = filterUnique ? 1 : filterQuality;

	bool onlyOnTargetReads = optParser.getOptBoolean("onTargetReads");
	bool onlyOnTargetBases = optParser.getOptBoolean("onTargetBases");

	// possible future options
	bool invertOnTarget = false;

	// check basic valid argument values and combinations
	int numOuts  = !bbcfileRoot.empty() + !covstatsFile.empty() + !sumstatsFile.empty() + !readOrigFile.empty();
	int numPipes = (bbcfileRoot == "-") + (covstatsFile == "-") + (sumstatsFile == "-") + (readOrigFile == "-");
	if( numOuts == 0 && !f_bci && !f_cbc ) {
		bbcfileRoot = "-";	// default if no other output specified
	} else if( numPipes > 1 ) {
		cerr << "Error: bbctools create: Only one file output (--covStats, --sumStats, --readOrigin or --bbc) may be piped to STDOUT." << endl;
		return -1;
	} else if( samdepth && numOuts ) {
		cerr << "Error: bbctools create: --samdepth (-s) option may only be used without other output options." << endl;
		return -1;
	}
	// check if single argument is a BBC file and leave open for reading if so
	BbcView bbcView;
	bool haveBbcFile = cmdArgs.size() == 1 && bbcView.Open( cmdArgs[0], true );
	bbcView.SelectPrintStream( samdepth ? "SAMDEPTH" : "BBCVIEW" );

	// check distinction between default and explicit no target regions - only for BBC input
	bool explicitNoTargetRegions = false;
	if( targetRegions == "-" ) {
		explicitNoTargetRegions = haveBbcFile;
		targetRegions = "";
	}
	if( targetRegions.empty() ) {
		if( onlyOnTargetBases && explicitNoTargetRegions && !invertOnTarget ) {
			cerr << "Warning: bbctools create --onTargetBases (-b) option with --regions '-' produces no coverage." << endl;
		} else if( onlyOnTargetReads ) {
			cerr << "Error: bbctools create --onTargetReads (-r) option requires a --regions file." << endl;
			return -1;
		}
	}
	// check for legal BBC create options
	if( f_bci || f_cbc ) {
		if( (bbcfileRoot.empty() || bbcfileRoot == "-") && !haveBbcFile ) {
			string opt = f_bci ? "--index (-i)" : "--coarse (-c)";
			cerr << "Error: bbctools create "+opt+" option requires the --bbc (-B) option or a BBC source file." << endl;
			return -1;
		}
	}
	BamMultiReader bamReader;
	if( haveBbcFile ) {
		// warn for options that do not work with BBC input
		if( filterQuality > 0 || filterDuplicates || filterUnique || minAlignLength ) {
			cerr << "Warning: SAM flag, alignment length and MAPQ filters ignored for BBC source file." << endl;
		}
		if( samdepth ) {
			cerr << "Error: --samdepth option is not supported for BBC source files." << endl;
			return -1;
		}
		if( !readOrigFile.empty() ) {
			cerr << "Error: --readOrigin option is not supported for BBC source files." << endl;
			return -1;
		}
	} else {
		// check / open for multiple BAM file inputs
		if ( !bamReader.Open(cmdArgs) ) {
			if( cmdArgs.size() == 1 ) cerr << "ERROR: Could not read input BAM file:";
			else cerr << "ERROR: Could not read all input BAM files:";
			// get and clean up bamtools error msg
			string errMsg = bamReader.GetErrorString();
			size_t i = errMsg.find_first_of('\n');
			if( i != string::npos ) errMsg = errMsg.substr(i+1);
			i = errMsg.find("::");
			if( i != string::npos ) {
				i = errMsg.find(": ");
				if( i != string::npos ) errMsg = errMsg.substr(i+1);
			}
			errMsg = BbcUtils::stringTrim(errMsg);
			errMsg[0] = toupper(errMsg[0]);
			cerr << endl << errMsg << "." << endl;
			return 1;
		}
	}
	// grab reference list from either input source
	const RefVector &references = haveBbcFile ? bbcView.GetReferenceData() : bamReader.GetReferenceData();
	if( !references.size() ) {
		// Issue would already been detected if input was BBC file
		cerr << "ERROR: " << (cmdArgs.size() > 1 ? "One or more " : "");
		cerr << "BAM file contains unaligned reads (no references).\n";
		return 1;
	}
	// check/set up target regions input regions/region statistics output
	RegionCoverage *regions = NULL;
	string covstatsStaticFields;
	bool trackRegionBaseCov = !covDepths.empty();
	if( covstatsFile.empty() ) {
		trackRegionBaseCov = false;
		if( !annotationFields.empty() ) {
			cerr << "Warning: --annotationFields (A) option ignored without --covStats (-C) option." << endl;
		}
		if( !covDepths.empty() && covDepths != "-" ) {
			cerr << "Warning: --covDepths (-D) option ignored without --covStats (-C) option." << endl;
		}
		if( !readType.empty() ) {
			cerr << "Warning: --readType (-T) option ignored without --covStats (-C) option." << endl;
		}
		// read regions for input only and/or creating sumStats
		if( !targetRegions.empty() || explicitNoTargetRegions || !sumstatsFile.empty() ) {
			regions = new RegionCoverage(references);
		}
	} else if( readType == "trgreads" || readType == "amplicon" || readType == "AmpliSeq" ) {
		if( haveBbcFile ) {
			cerr << "Creation of read coverage requires BAM file input." << endl;
			return -1;
		}
		AmpliconRegionStatistics *ampRegionStats = new AmpliconRegionStatistics(references);
		ampRegionStats->SetGenericReads( readType == "trgreads" );
		ampRegionStats->SetSigFacCoverage( minPcCov/100 );
		ampRegionStats->SetMaxUpstreamPrimerStart( primerLength );
		ampRegionStats->SetMaxE2eEndDistance( maxE2eEndGap );
		covstatsStaticFields = "overlaps,";
		covstatsStaticFields += (minPcCov > 0) ? "fwd_cov,rev_cov" : "fwd_e2e,rev_e2e";
		covstatsStaticFields += ",total_reads,fwd_reads,rev_reads";
		regions = ampRegionStats;
	} else if( readType == "trgbases" ) {
		if( haveBbcFile && targetRegions.empty() && !explicitNoTargetRegions ) {
			cerr << "Warning: Assuming reference contigs for base coverage targets (=> option --regions -)" << endl;
		}
		RegionStatistics *regionStats = new RegionStatistics(references);
		covstatsStaticFields = "covered,uncov_5p,uncov_3p,ave_basereads,fwd_basereads,rev_basereads";
		trackRegionBaseCov = true;
		regions = regionStats;
	} else if( readType == "covdepth" || readType.empty() ) {
		// output (sorted) targets file with only covDepth stats (if any)
		regions = new RegionCoverage(references);
	} else {
		cerr << "Unknown read type '" << readType << "'" << endl;
		return -1;
	}
	// Load the input regions or default to whole reference contig targets
	if( regions ) {
		regions->SetCovAtDepths( covDepths == "-" ? "20,100,500" : covDepths );
		if( targetRegions.empty() ) {
			regions->SetWholeContigTargets();
			// set contigs as explicit regions means all reads will seen as on-target
			// for consistency these are inverted (for input from BBC)
			invertOnTarget = true;
		} else {
			string auxFieldIdx = auxRegionSplit.size() ? auxRegionSplit[0] : "";
			string errMsg = regions->Load( targetRegions, "BED", auxFieldIdx );
			if( !errMsg.empty() ) {
				cerr << "ERROR: " + errMsg + "\n";
				return 1;
			}
		}
		if( onlyOnTargetReads && haveBbcFile ) {
			cerr << "Error: bbctools create --onTargetReads option is not supported for BBC source file." << endl;
			return -1;
		}
	}
	//
	// Perform all bbctools create utilities
	//
	BbcCreate *bbcCreate = NULL;
	if( !bbcfileRoot.empty() && (bbcfileRoot != "-" || !haveBbcFile) ) {
		bbcCreate = new BbcCreate(references);
		if( bbcfileRoot != "-" && !bbcCreate->Open(bbcfileRoot+".bbc") ) {
			return 1;
		}
		bbcCreate->SetNoOffTargetPositions(onlyOnTargetBases);
	}
	bbcView.SetNoOffTargetPositions(onlyOnTargetBases);
	// Stream input to output creators
	if( haveBbcFile ) {
		// BBC reader and driver via BbcView object
		if( bbcfileRoot != "-" || !covstatsFile.empty() ) {
			// disable BbcView text stream if using for file creation
			bbcView.SelectPrintStream("NONE");
		}
		// process input BBC for just new BBC and target coverage (defer BCI/CBC)
		bbcView.SetBbcCreate(bbcCreate);
		bbcView.SetRegionCoverage(regions);
		// explicitNoTargetRegions intended for explicitly removing on-target coverage
		bbcView.SetInvertOnTarget(explicitNoTargetRegions ^ invertOnTarget);
		if( bbcCreate || regions || bbcfileRoot == "-" ) {
			bbcView.ReadAll();
		}
	} else {
		// Test read tracking option for file write
		TrackReads *readTracker = NULL;
		try {
			if( !readOrigFile.empty() )
				readTracker = new TrackReads( readOrigFile, regions );
		} catch( std::runtime_error & ) {
			cerr << "ERROR: Unable to write to read tracking file " << readOrigFile << endl;
			return 1;
		}
		// BAM reader, BaseCoverage driver, dispatching to BbcCreate and BbcView objects
		BaseCoverage baseCov(references);
		baseCov.SetRegionCoverage(regions);
		baseCov.SetBbcCreate(bbcCreate);
		baseCov.SetInvertOnTarget(invertOnTarget);
		if( bbcfileRoot == "-" ) {
			baseCov.SetBbcView(&bbcView);
		}
		// Certain options require that all reads are processed, invalidating other performance options
		bool trackAllReads = !sumstatsFile.empty() || readTracker;
		// Implicit set of onlyOnTargetReads for performance when only these reads are required
		bool useBaseCov = (bbcfileRoot == "-" || bbcCreate);
		if( !targetRegions.empty() && !trackAllReads ) {
			onlyOnTargetReads |= onlyOnTargetBases;
			if( samdepth || !useBaseCov ) onlyOnTargetReads = true;
		}
		useBaseCov |= trackRegionBaseCov;
		// do not allow jumping if sumStats option is used - need to count all reads
		bool bamReaderSetRegions = (s_useBamReaderJump && !trackAllReads);
		int trgContig = 0, trgSrtPos = 0, trgEndPos = 0;
		int minJumpLen = s_initialMinJumpLen;
		int maxReadLen = s_initialMaxReadLen;
		if( onlyOnTargetReads ) {
			// load/create BAM index files for targeted reading
			// Note: BamIndex::BAMTOOLS format performed very badly and cannot use mixed with BTI/BAI files
			if( bamReaderSetRegions && !bamReader.LocateIndexes() ) {
				string plural( cmdArgs.size() > 1 ? "s" : "" );
				if( autoCreateBamIndex ) {
					cerr << "Warning: Did not locate BAM index (BAI) file" << plural << ", creating bamtools version..." << endl;
					// to avoid bug use new instance of BamMultiReader
					BamMultiReader bamReader2;
					if( !bamReader2.Open(cmdArgs) || !bamReader2.CreateIndexes() ) {
						cerr << "WARNING: Failed to create BAM index file" << plural << "." << endl;
						bamReaderSetRegions = false;
					} else {
						if( cmdArgs.size() == 1 ) {
							cerr << "Successfully created BAM index file: " << BbcUtils::fileName(cmdArgs[0]) << ".bai" << endl;
						} else {
							cerr << "Successfully created BAM index files." << endl;
						}
						// re-locate indexes with first reader - could not seem to locate BTI files created!
						if( !bamReader.LocateIndexes() ) {
							cerr << "WARNING: Failed to locate BAM index file" << plural << " just created!" << endl;
							bamReaderSetRegions = false;
						}
					}
				} else {
					cerr << "Warning: BAM index file" << plural << " not located for targeted BAM access." << endl;
					bamReaderSetRegions = false;
				}
			}
			// cancel region filtering if there are no regions to iterate (unexpected)
			if( !regions->GetNextRegion( trgContig, trgSrtPos, trgEndPos ) ) {
				onlyOnTargetReads = bamReaderSetRegions = false;
			}
			if( bamReaderSetRegions ) {
				bamReader.Jump( trgContig, trgSrtPos-maxReadLen );
			}
		}
		BamAlignment aln;
		while( bamReader.GetNextAlignmentCore(aln) ) {
			// appears to be an undocumented behavior here
			if( aln.RefID < 0 ) continue;
			// skip filtered reads by flag, length or mapping quality
			if( aln.AlignmentFlag & skipFlag ) continue;
			if( aln.MapQuality < minMapQuality ) continue;
			int32_t endPos = aln.GetEndPosition();
			if( minAlignLength > 0 ) {
				if( endPos - aln.Position < minAlignLength ) continue;
			}
			// screen for on-target reads
			if( onlyOnTargetReads ) {
				// find next region overlapping or beyond of current read
				bool moreRegions = true;
				bool setRegion = false;
				while( aln.RefID > trgContig || (aln.RefID == trgContig && aln.Position > trgEndPos) ) {
					if( !regions->GetNextRegion( trgContig, trgSrtPos, trgEndPos ) ) {
						moreRegions = false;
						break;
					}
					setRegion = bamReaderSetRegions;
				}
				if( !moreRegions ) {
					// prevent further on-target checks and exit early if not using sumStats
					onlyOnTargetReads = false;
					if( trackAllReads ) {
						// force tracking of off-target reads
						regions->TrackReadsOnRegion(aln,endPos);
						if( readTracker ) readTracker->Write(aln,endPos);
						continue;
					}
					break;
				}
				if( setRegion ) {
					// track max read length for future index jumps - just in case long reads ever used
					if( endPos - aln.Position > maxReadLen ) {
						maxReadLen = endPos - aln.Position;
						if( maxReadLen > minJumpLen ) minJumpLen = maxReadLen;
					}
					if( aln.RefID != trgContig || trgSrtPos - aln.Position > minJumpLen ) {
						bamReader.Jump( trgContig, trgSrtPos-maxReadLen );
					}
				}
				if( aln.RefID < trgContig || endPos < trgSrtPos ) {
					// force tracking of off-target reads
					if( trackAllReads ) {
						regions->TrackReadsOnRegion(aln,endPos);
						if( readTracker ) readTracker->Write(aln,endPos);
					}
					continue;	// current is before next target region - fetch the next within bounds
				}
			}
			// record base coverage and region coverage statistics
			if( useBaseCov ) {
				endPos = baseCov.AddAlignment(aln,endPos);
				if( endPos <= 0 ) {
					if( endPos == 0 ) continue;	// read was silently ignored
					cerr << "ERROR: BAM file is not correctly sorted vs. reference." << endl;
					return 1;
				}
			}
			// record read coverage and region coverage statistics
			if( regions ) {
				regions->TrackReadsOnRegion(aln,endPos);
			}
			if( readTracker ) {
				readTracker->Write(aln,endPos);
			}
		}
		// flush and close objects associated with output
		baseCov.Flush();
	}
	// Output in-memory region stats file and ensure BBC file is closed
	if( regions ) {
		// build output fields title string
		string outFields = "contig_id,contig_srt,contig_end";
		if( !auxRegionSplit.empty() ) outFields += "," + auxRegionSplit[1];
		if( !covstatsStaticFields.empty() ) outFields += "," + covstatsStaticFields;
		regions->Write( covstatsFile, outFields );
		if( !sumstatsFile.empty() ) {
			regions->WriteSummary( sumstatsFile, invertOnTarget );
		}
		delete regions;
	}
	delete bbcCreate;

	// Complete remaining file creation options using a BBC file input
	// NOTE: Using BbbCreate for this would require code duplication and concurrent file output streaming
	if( f_bci || f_cbc ) {
		// Check BBC file source
		if( haveBbcFile ) {
			bbcfileRoot = cmdArgs[0];
	    	int i = bbcfileRoot.size() - 4;
	    	if( i > 0 && bbcfileRoot.substr(i,4) == ".bbc" ) {
	    		bbcfileRoot = bbcfileRoot.substr(0,i);
	    	}
		} else if( !bbcView.Open( bbcfileRoot+".bbc", true ) ) {
			cerr << "ERROR: Unexpected failure to read new BBC file '"+bbcfileRoot+".bam'" << endl;
			return 1;
		}
		if( f_bci ) {
			BbcIndex indexer( bbcfileRoot+".bci" );
			if( !bbcView.CreateIndex(indexer) ) {
				cerr << "ERROR: Failed to create index file '" << bbcfileRoot << ".bci'" << endl;
				return 1;
			}
		}
		if( f_cbc ) {
			// CBC generation can use BCI file but is no faster since whole BBC file is read
			BbcCoarse cbcWriter( bbcfileRoot+".cbc" );
			if( !bbcView.CreateCbc(cbcWriter) ) {
				cerr << "ERROR: Failed to create coarse base coverage file '" << bbcfileRoot << ".cbc'" << endl;
				return 1;
			}
		}
	}
	return 0;
}

int bbctools_report( BbcUtils::OptParser &optParser )
{
    const vector<string> cmdArgs = optParser.getArgs();
	BbcDepth bbcDepth;
	if( !bbcDepth.Open( cmdArgs[0], true ) ) {
		cerr << "Error: BBC file '" << cmdArgs[0] << "' was not found or does not have BBC format." << endl;
		return -1;
	}
	string docFile = optParser.getOptValue("coverage");
	string covDepths = optParser.getOptValue("covDepths","-");
	bbcDepth.SetCovAtDepths( covDepths == "-" ? "1,20,100,500" : covDepths );
    bbcDepth.SetIgnoreOnTarget( !optParser.getOptBoolean("reportOnTarget") );
    bool genomeReport = optParser.getOptBoolean("genomeReport");
    bbcDepth.SetGenomeReport(genomeReport);

	double unithresh = optParser.getOptNumber("uniformityThreshold",20);
	if( unithresh <= 0 ) {
		cerr << "Error: --uniformityThreshold (-U) must be a positive (%) value." << endl;
		return -1;
	}
	bbcDepth.SetUniformityMeanThreshold(unithresh/100);
	double strandbias = optParser.getOptNumber("strandBiasThreshold",70);
	if( strandbias < 50 ) {
		cerr << "Error: --strandBiasThreshold (-B) must be a value >= 50 (%)." << endl;
		return -1;
	}
	bbcDepth.SetStrandBiasThreshold(strandbias/100);
	int minStrandReads = optParser.getOptNumber("statsMinReads",10);
	if( minStrandReads <= 0 ) {
		cerr << "Error: --statsMinReads (-M) must be a positive integer." << endl;
		return -1;
	}
	bbcDepth.SetStrandBiasMinCount(minStrandReads);

	// restricting regions
	string targetRegions  = optParser.getOptValue("regions");
	RegionCoverage *regions = NULL;
	if( !targetRegions.empty() && targetRegions != "-" ) {
		regions = new RegionCoverage( bbcDepth.GetReferenceData() );
		string errMsg = regions->Load( targetRegions, "BED", "" );
		if( !errMsg.empty() ) {
			cerr << "ERROR: " + errMsg + "\n";
			return 1;
		}
		bbcDepth.SetRegionCoverage(regions);
	}
	string range = cmdArgs.size() > 1 ? cmdArgs[1] : "";
	//
	// Perform all bbctools depth utilities
	//
	// load index file, if present (CBC only useful with binning)
	string bbcfileRoot = cmdArgs[0];
	int i = bbcfileRoot.size() - 4;
	if( i > 0 && bbcfileRoot.substr(i,4) == ".bbc" ) {
		bbcfileRoot = bbcfileRoot.substr(0,i);
	}
	BbcIndex indexer( bbcfileRoot+".bci" );
	if( indexer.Open() ) bbcDepth.SetBbcIndex(&indexer);
	// output the ranged/binned view to STDOUT
    string errMsg = bbcDepth.ReadRange( range );
    if( !errMsg.empty() ) {
    	cerr << errMsg << endl;
    	return -1;
    }
    bbcDepth.Report( regions || !range.empty() );
    if( !docFile.empty() ) {
      bbcDepth.Write(docFile);
    }
    delete regions;
	return 0;
}

int bbctools_view( BbcUtils::OptParser &optParser )
{
    const vector<string> cmdArgs = optParser.getArgs();
	BbcView bbcView;
	if( !bbcView.Open( cmdArgs[0], true ) ) {
		cerr << "Error: BBC file '" << cmdArgs[0] << "' was not found or does not have BBC format." << endl;
		return -1;
	}
	// binning type options and window size
	bool isolateRegions  = optParser.getOptBoolean("isolateRegions");
	int32_t numBins = optParser.getOptInteger("numbins");
	int32_t binSize = optParser.getOptInteger("binsize");
	int32_t srtBin = optParser.getOptInteger("startBin");
	int32_t endBin = optParser.getOptInteger("endBin");

	// configure output formating options
	bbcView.SetHeaderLine( optParser.getOptValue("headerLine") );
	bbcView.SetOutputBedCoordinates( optParser.getOptBoolean("bedCoordinates") );
	bbcView.SetHideOnTargetCoverage( optParser.getOptBoolean("omitTargetCoverage") );
	bbcView.SetSumFwdRevCoverage( optParser.getOptBoolean("sumStrandCoverage") );

	string annotationFields = optParser.getOptValue("annotationFields");
	bool annotate = !annotationFields.empty();
	bool compactCov   = optParser.getOptBoolean("compactCoverage");
	bool showZeroCov  = optParser.getOptBoolean("includeZeroCoverage");
	bool lociOnly     = optParser.getOptBoolean("lociOnly");
	bool omitContigNames     = optParser.getOptBoolean("omitContigNames");
	bool omitContigPositions = optParser.getOptBoolean("omitContigPositions");

	bbcView.SetUseRegionAnnotation(annotate);
	bbcView.SelectPrintStream(compactCov ? "COMPACT" : "BBCVIEW");
	bbcView.SetShowZeroCoverage(showZeroCov);
	bbcView.SetShowLociOnly(lociOnly);
	bbcView.SetHideContigNames(omitContigNames);
	bbcView.SetHideRegionCoordinates(omitContigPositions);

	// check for illegal/ignored option combinations
	if( numBins == 0 && binSize == 0 && isolateRegions == false ) {
		// default view => disallow options dependent on binned-only options
		if( annotate ) {
			cerr << "Error: The --Fields (-A) option is only valid for binned coverage (with -B, -N or -i options)." << endl;
			return -1;
		}
		if( srtBin ) {
			cerr << "Error: The --startBin (-S) option is only valid for binned coverage (with -B, -N or -i options)." << endl;
			return -1;
		}
		if( endBin ) {
			cerr << "Error: The --endBin (-E) option is only valid for binned coverage (with -B, -N or -i options)." << endl;
			return -1;
		}
	} else {
		// binned region view => disallow options dependent on 'streamed' base coverage
		if( compactCov ) {
			cerr << "Error: The --compactCoverage (-c) option is only valid for base coverage (no -B, -N or -i options)." << endl;
			return -1;
		}
		if( showZeroCov ) {
			cerr << "Error: The --includeZeroCoverage (-z) option is only valid for the base coverage (no -B, -N or -i options)." << endl;
			return -1;
		}
	}
	if( numBins && binSize ) {
		cerr << "Error: --binsize (-B) option (" << binSize << ") and --numbins (-N) option (" << numBins << ") ";
		cerr << "may not be both specified values > 0." << endl;
		return -1;
	}
	if( lociOnly & omitContigNames & omitContigPositions ) {
		cerr << "Warning: Using --lociOnly, --omitContigNames and --omitContigPositions together produces no output."<< endl;
	}
	// restricting/annotating regions
	string targetRegions  = optParser.getOptValue("regions");
	RegionCoverage *regions = NULL;
	if( !targetRegions.empty() && targetRegions != "-" ) {
		regions = new RegionCoverage( bbcView.GetReferenceData() );
		string errMsg = regions->Load( targetRegions, "BED", annotationFields );
		if( !errMsg.empty() ) {
			cerr << "ERROR: " + errMsg + "\n";
			return 1;
		}
		bbcView.SetRegionCoverage(regions);
	} else if( annotate ) {
		cerr << "Error: --annotationFields (-A) option is only valid with the --regions option." << endl;
		return -1;
	}
	string range = cmdArgs.size() > 1 ? cmdArgs[1] : "";
	//
	// Perform all bbctools view utilities
	//
	// load index and open cbc files, if present
	string bbcfileRoot = cmdArgs[0];
	int i = bbcfileRoot.size() - 4;
	if( i > 0 && bbcfileRoot.substr(i,4) == ".bbc" ) {
		bbcfileRoot = bbcfileRoot.substr(0,i);
	}
	BbcIndex indexer( bbcfileRoot+".bci" );
	if( indexer.Open() ) bbcView.SetBbcIndex(&indexer);
	BbcCoarse cbc( bbcfileRoot+".cbc" );
	if( cbc.Open() ) bbcView.SetBbcCoarse(&cbc);
	// output the ranged/binned view to STDOUT
    string errMsg = bbcView.ReadRange( range, isolateRegions, numBins, binSize, srtBin, endBin );
    if( !errMsg.empty() ) {
    	cerr << errMsg << endl;
    	return 1;
    }
    delete regions;
	return 0;
}

int bbctools_version( BbcUtils::OptParser &optParser ) {
    const vector<string> cmdArgs = optParser.getArgs();
    bool brief = optParser.getOptBoolean("brief");

    if( cmdArgs.size() == 0 ) {
    	if( !brief ) cout << "bbctools version: ";
    	cout << BbcUtils::numberToString( (double)s_versionNumber/1000, 3 ) << endl;
    	return 0;
    }
    // since binary files do not have a GUID (magic) number, rely on file extension
	string file = cmdArgs[0];
	string fileName = BbcUtils::fileName(file);
	string fileExt = BbcUtils::fileNameExt(file);
	string fileVersion, fileType;
	if( fileExt == "bbc" ) {
		fileExt = "BBC";
		fileType = "Binary Base Coverage";
		BbcView bbcView;
		if( bbcView.Open( file, true ) ) fileVersion = bbcView.VersionString();
	} else if( fileExt == "bci" ) {
		fileExt = "BCI";
		fileType = "Base Coverage Index";
		BbcIndex bbcIndex(file);
		if( bbcIndex.Open( false, true ) ) fileVersion = bbcIndex.VersionString();
	} else if( fileExt == "cbc" ) {
		fileExt = "CBC";
		fileType = "Coarse Base Coverage";
		BbcCoarse bbcCoarse(file);
		if( bbcCoarse.Open() ) fileVersion = bbcCoarse.VersionString();
	} else {
		cerr << "Error: File format checking requires file has a bbc, bci or cbc file name extension." << endl;
		return -1;
	}
	if( fileVersion.empty() ) {
		cerr << "Error: File '" << fileName << "' does not have expected (" << fileExt << ") format." << endl;
		return -1;
	}
	if( brief ) {
		cout << fileVersion << endl;
	} else {
		cout << "File name: " << fileName << endl;
		cout << "File type: " << fileType << " (" << fileExt << ")" << endl;
		cout << "Version:   " << fileVersion << endl;
	}
	return 0;
}
