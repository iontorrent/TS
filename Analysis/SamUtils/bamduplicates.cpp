/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <string>
#include <sstream>
#include <map>
#include <limits>
#include <list>
#include <ostream>
#include "IonVersion.h"
#include "api/BamReader.h"
#include "api/BamWriter.h"
#include "api/SamConstants.h"
#include "Util/OptArgs.h"
#include "json/json.h"
#include "ion_util.h"

using namespace BamTools;
const int NUM_WRITER_THREADS = 6; //reasonable number of threads allocated to writing data.

void DBG_PRINTF(...){}
//#define DBG_PRINTF printf

bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value);

//Algorithm:
//Handle forward and reverse strands separately
//________________________________________________________________________________________________________
//
//FORWARD STRAND:
// 1. Collect all strands that start at the given position
// 2. Sort by 3' position. The longest strand is not a dup.
// Mark all shorter strands without B adapter as dups.Mark strands (except 1) with the same 3' length as duplicates.
//
//        ------------------------------------------------>3'
//        -------------------------------------------->B3'
//        ---------------------------------------->B3'
//        ---------------------------------------->B3'                      Dup
//        ------------------------------------>3'                             Dup
//        ---------------------------------->3'                               Dup
//
//
// 3. Next algorithm refinement - make flow space comparisons between strands in the same position
//________________________________________________________________________________________________________
//
//REVERSE STRAND:
// Note: in the current algorithm among all strands for a given position, longer one "wins".
// So we can make decisions for reverse strands as they come in.
//
// Keep a map
//  3'<-----------------------------------------
//     B3'<-------------------------------------
//     B3'<-------------------------------------
//       3'<-------------------------------------
//


class AlignCache{
    BamWriter* writer;
    typedef std::multimap<int, BamAlignment, std::greater<int> > StrandMapType;

    struct AlignStore{
        StrandMapType fwdStrand; //"key" for both strands is flow where adaptor is found or last flow (if adaptor is not found).
        StrandMapType revStrand;
    };

    //convention: for both forward and reverse strands "key" is the "start position" of the read  (forward read "start position" ----- "end position")
    //Note that for reverse strand "start position" is the larger position                                  ( reverse read "end position" ------ "start position")
    typedef std::map< int, AlignStore > AlignStoreType;
    AlignStoreType alignStore;
    struct ProcessedReadStore{
        int numUnprocessedRevReads;
        std::list<BamAlignment> readList;
    };

    typedef std::map< int, ProcessedReadStore > ProcessedStoreType;
    ProcessedStoreType processedReadStore; //store processed reads as they are done. "key" is the aligned position.

    int32_t curRefId, curPos;

    void MarkDuplicates(int32_t refId, int32_t pos );
    void MarkDuplicatesStrand(int32_t pos, StrandMapType& alignStore, bool isReverse, char *str);
    //void MarkDuplicatesRev(int32_t pos);
    void WriteData(int32_t refId, int32_t pos );


    inline int adaptorFlow( BamAlignment& al ){
        std::vector<int> pcr_duplicate_signature(3,0);
        if( al.HasTag("ZC") )
            al.GetTag("ZC",pcr_duplicate_signature);
        return pcr_duplicate_signature[1]; //entry 2 - last insert flow.

//        int64_t adapterFlow=0;
//        if( al.HasTag("ZG") )
//            getTagParanoid(al,"ZG", adapterFlow);
//        return adapterFlow;
    }

    inline int lastFlow( BamAlignment& al ){
        int adFlow = adaptorFlow( al );
        if( adFlow ) return adFlow;

        std::vector<int16_t> zm_flowData;
        if( al.GetTag("ZM", zm_flowData) && zm_flowData.size()>0 )
            return zm_flowData.size();

        std::vector<uint16_t> fz_flowData;
        if( al.GetTag("FZ", fz_flowData) && fz_flowData.size()>0 )
            return fz_flowData.size();
        return 0;
    }

    inline int GetEndPosition( const BamAlignment& al ){
        return al.GetEndPosition(false, true);
    }

    void FlushData(int32_t refId, int32_t pos, bool reset);
    bool saveDups;
    std::ofstream dupFile;

public:
    int nTotalReads, nTotalMappedReads, nDuplicates, nWithAdaptor;
    AlignCache( BamWriter& _writer ) : writer(&_writer), curRefId(-1), curPos(-1), saveDups(false), nTotalReads(0), nTotalMappedReads(0), nDuplicates(0), nWithAdaptor(0) {}
    void NewAlignment( BamAlignment& al );
    void FlushData(void) { FlushData(0, std::numeric_limits<int32_t>::max(), true); }
    void OutputDuplicates( const std::string& dupsFileName ){ if((saveDups = (dupsFileName.size()>0))) dupFile.open( dupsFileName.c_str() ); }
};

inline void outputDupString( std::stringstream& duplicateStream, const char* al )
{
    int thisCol = 0;
    int thisRow = 0;
    ion_readname_to_rowcol(al, &thisRow, &thisCol);
    duplicateStream << thisRow << " " << thisCol << " ";
}

void AlignCache::MarkDuplicatesStrand( int32_t pos, StrandMapType& strand, bool isReverse, char* str )
{

    if( strand.size()==0) return;

    std::stringstream duplicateStream;
    bool hasDuplicate = false;

    //handle the forward strand
    StrandMapType::iterator it = strand.begin();
    BamAlignment& al_first = it->second; //longest/best alignment won't be marked duplicate

    bool hasAdaptor = adaptorFlow( al_first );
    int maxFlow = lastFlow( al_first );

    if( isReverse ){
        --(processedReadStore[al_first.Position].numUnprocessedRevReads);
        DBG_PRINTF("Pos: %d, Start: %d, Unp: %d\n", pos, al_first.Position, processedReadStore[al_first.Position].numUnprocessedRevReads);
    }

    processedReadStore[al_first.Position].readList.push_back(al_first);
    nWithAdaptor += hasAdaptor? 1 : 0;
    nTotalMappedReads += (al_first.IsMapped())?1:0;

    if( saveDups )
        outputDupString( duplicateStream, al_first.Name.c_str() );

    DBG_PRINTF("%s: %d %d %c %d\tOk First\t\t%s\n", str, al_first.Position, GetEndPosition(al_first), adaptorFlow(al_first)?'T':'F', lastFlow(al_first), al_first.Name.c_str());

    ++it;  //longest (in flow space) read is not a duplicate

    while( it != strand.end() ){

        StrandMapType::iterator lastReadOfSameLen_iter = strand.upper_bound(it->first);

        while( it != lastReadOfSameLen_iter ){
            BamAlignment& currAlignment = it->second;
            hasAdaptor = adaptorFlow( currAlignment );

            if( isReverse ){
                --(processedReadStore[currAlignment.Position].numUnprocessedRevReads);
                DBG_PRINTF("Pos: %d, Start: %d, Unp: %d\n", pos, currAlignment.Position, processedReadStore[currAlignment.Position].numUnprocessedRevReads);
            }

            if( not ( hasAdaptor && (lastFlow(currAlignment)<maxFlow) ) && currAlignment.IsMapped() ){
                currAlignment.SetIsDuplicate(true);
                ++nDuplicates;
                DBG_PRINTF("%s: %d %d %c %d\tDup Shorter/No adaptr\t\t%s\n", str, currAlignment.Position, GetEndPosition(currAlignment), hasAdaptor?'T':'F',lastFlow(currAlignment), currAlignment.Name.c_str());
                if( saveDups ){
                    outputDupString( duplicateStream, currAlignment.Name.c_str() );
                    hasDuplicate = true;
                }
            }
            else{
                DBG_PRINTF("%s: %d %d %c %d\tOk Adaptr\t\t%s\n", str, currAlignment.Position, GetEndPosition(currAlignment), hasAdaptor?'T':'F',lastFlow(currAlignment), currAlignment.Name.c_str());
            }
            if( hasAdaptor )
                maxFlow = lastFlow( currAlignment );

            processedReadStore[currAlignment.Position].readList.push_back( currAlignment );
            nWithAdaptor += hasAdaptor? 1 : 0;
            nTotalMappedReads += (currAlignment.IsMapped())?1:0;
            ++it;
        }
    }
    strand.clear();
    if( saveDups && hasDuplicate && dupFile.is_open() ){
        dupFile << duplicateStream.str() << "\n";
    }
}


void AlignCache::WriteData( int32_t refId, int32_t pos  )
{
    //at this point we know that all reads that start at "pos" have been read.
    ProcessedStoreType::iterator it;
    for( it = processedReadStore.begin(); it != processedReadStore.lower_bound( pos ); ++it ){
        std::list<BamAlignment>& readList = it->second.readList;
        int& numUnprocessedRevReads = it->second.numUnprocessedRevReads;
        if( numUnprocessedRevReads>0 ){
            DBG_PRINTF("\t\tGated by unprocessed: %d Pos: %d\n", numUnprocessedRevReads, it->first);
            break;
        }
        else{
            DBG_PRINTF("\t\tSaving Pos: %d\n", it->first);
        }

        for(std::list<BamAlignment>::iterator read_it = readList.begin(); read_it!= readList.end(); ++read_it ){
            DBG_PRINTF("\t\t\tSaving %s\n", read_it->Name.c_str());
            writer->SaveAlignment( *read_it );
        }
    }
    processedReadStore.erase(processedReadStore.begin(), it);
    alignStore.erase(alignStore.begin(), alignStore.lower_bound( it->first ));
}

void AlignCache::MarkDuplicates(int32_t refId, int32_t pos  )
{
    if( refId == -1 ) //unaligned content
        return;

    MarkDuplicatesStrand(pos, alignStore[pos].fwdStrand, false, "Fwd");
    MarkDuplicatesStrand(pos, alignStore[pos].revStrand, true, "Rev");
}

void AlignCache::FlushData( int32_t refId, int32_t pos, bool reset )
{
    if( alignStore.empty() )
        return;

    AlignStoreType::iterator it = alignStore.begin();
    AlignStoreType::iterator end = alignStore.upper_bound( pos );

    for( ; it != end; it = alignStore.upper_bound( it->first ) ){
        DBG_PRINTF("Processing pos: %d\n", it->first);
        MarkDuplicates( refId, it->first);
    }

    WriteData( refId, pos );

    if( reset ){
        processedReadStore.clear();
    }
}

void AlignCache::NewAlignment( BamAlignment& al )
{
    if( curRefId != al.RefID ){
        FlushData( curRefId, std::numeric_limits<int32_t>::max(), true );
    }

    al.BuildCharData(); //needed for reading tags

    if( al.IsDuplicate() )
        DBG_PRINTF("Reads already marked as duplicates.");

    if( al.IsReverseStrand() ){
        alignStore[GetEndPosition(al)].revStrand.insert(std::pair<int, BamAlignment>(lastFlow(al), al ));
        DBG_PRINTF("\tAdding REV Pos: %d, End: %d, Flow: %d\n", al.Position, GetEndPosition(al), lastFlow(al));
        processedReadStore[al.Position].numUnprocessedRevReads += 1;
    }
    else{
        alignStore[al.Position].fwdStrand.insert(std::pair<int, BamAlignment>(lastFlow(al), al ));
        DBG_PRINTF("\tAdding FWD Pos: %d, End: %d, Flow: %d\n", al.Position, GetEndPosition(al), lastFlow(al));
    }

    if( curRefId != al.RefID || curPos != al.Position ){
        //We got all alignments for the previous position. We can mark duplicates and write to disk.
        FlushData( curRefId, curPos, false );
    }

    curRefId = al.RefID;
    curPos = al.Position;

}

void PrintHelp()
{
  printf ("\n");
  printf ("Usage: BamDuplicates [options] -i Input.bam -o Output.bam\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -h,--help                             print this help message and exit\n");
  printf ("  -v,--version                          print version and exit\n");
  printf ("  -i,--input             FILE       input bam file or \"stdin\" [required option]\n");
  printf ("  -o,--output         FILE        results file or \"stdout\" [required option]\n");
  printf ("  -d,--dir               DIR        output directory\n");
  printf ("  -j,--json               FILE       output statistics file\n");
  printf ("  -x,--save-duplicates    FILE     save duplicate reads\n");
  printf ("\n");

  exit (EXIT_SUCCESS);
}

int main( int argc, const char* argv[] )
{
    std::string inputFile, outputFile, outputDir, outputJSON, saveDuplicates;

    OptArgs opts;
    opts.ParseCmdLine(argc, argv);

    if (opts.GetFirstBoolean('h', "help", false) or argc == 1)
      PrintHelp();

    if (opts.GetFirstBoolean('v', "version", false)) {
      fprintf (stdout, "%s", IonVersion::GetFullVersion ("BamDuplicates").c_str());
      exit (EXIT_SUCCESS);
    }

    inputFile = opts.GetFirstString('i',"input","stdin");
    outputFile = opts.GetFirstString('o',"output","stdout");
    outputDir = opts.GetFirstString('d',"dir",".");
    outputJSON = opts.GetFirstString('j',"json","BamDuplicates.json");
    saveDuplicates = opts.GetFirstString('x',"save-duplicates","");
    opts.CheckNoLeftovers();

    BamReader reader;
    if( !reader.Open( inputFile ) ){
        DBG_PRINTF("Cannot open bam file: %s\n", inputFile.c_str());
        return 1;
    }

    const SamHeader header =  reader.GetHeader();
    const RefVector references = reader.GetReferenceData();

    BamWriter writer;
    writer.SetNumThreads( NUM_WRITER_THREADS );

    if( header.HasSortOrder() && (header.SortOrder != Constants::SAM_HD_SORTORDER_COORDINATE ) ){
        DBG_PRINTF("Bam file has to be sorted by coordinate.");
        return 1;
    }

    if( !writer.Open(outputFile, header, references) ){
        DBG_PRINTF("Cannot open output bam file: %s\n", outputFile.c_str());
    }
    
   
    writer.SetNumThreads( 6 );

    AlignCache cache(writer);
    cache.OutputDuplicates( saveDuplicates );
    BamAlignment al;
    while( reader.GetNextAlignmentCore(al) ){
        ++cache.nTotalReads;
        if( al.IsMapped() )
            cache.NewAlignment( al );
        else{
            cache.FlushData();
            writer.SaveAlignment( al );
        }
    }
    cache.FlushData();

    reader.Close();
    writer.Close();

    Json::Value BamDuplicates_json(Json::objectValue);
    BamDuplicates_json["total_reads"] = cache.nTotalReads;
    BamDuplicates_json["total_mapped_reads"] = cache.nTotalMappedReads;
    BamDuplicates_json["duplicate_reads"] = cache.nDuplicates;
    BamDuplicates_json["reads_with_adaptor"] = cache.nWithAdaptor;
    BamDuplicates_json["fraction_duplicates"] = (float)cache.nDuplicates/(cache.nTotalMappedReads+1.);
    BamDuplicates_json["fraction_with_adaptor"] = (float)cache.nWithAdaptor/(cache.nTotalMappedReads+1.);

    std::ofstream out((outputDir+"/"+outputJSON).c_str(), std::ios::out);
    if( out.good() )
        out<<BamDuplicates_json.toStyledString();

    return 0;
}

//        std::vector<int> clipSizes, readPositions, genomePositions;
//        al.GetSoftClips(clipSizes, readPositions, genomePositions);
//        for( int i=0; i<clipSizes.size(); ++i )
//            DBG_PRINTF("%d %d %d; ", clipSizes[i], readPositions[i], genomePositions[i] );
//        DBG_PRINTF("\n");


bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value) {
    char tagType = ' ';
    if(alignment.GetTagType(tag, tagType)) {
        switch(tagType) {
            case BamTools::Constants::BAM_TAG_TYPE_INT8: {
                int8_t value_int8 = 0;
                alignment.GetTag(tag, value_int8);
                value = value_int8;
            } break;
            case BamTools::Constants::BAM_TAG_TYPE_UINT8: {
                uint8_t value_uint8 = 0;
                alignment.GetTag(tag, value_uint8);
                value = value_uint8;
            } break;
            case BamTools::Constants::BAM_TAG_TYPE_INT16: {
                int16_t value_int16 = 0;
                alignment.GetTag(tag, value_int16);
                value = value_int16;
            } break;
            case BamTools::Constants::BAM_TAG_TYPE_UINT16: {
                uint16_t value_uint16 = 0;
                alignment.GetTag(tag, value_uint16);
                value = value_uint16;
            } break;
            case BamTools::Constants::BAM_TAG_TYPE_INT32: {
                int32_t value_int32 = 0;
                alignment.GetTag(tag, value_int32);
                value = value_int32;
            } break;
            case BamTools::Constants::BAM_TAG_TYPE_UINT32: {
                uint32_t value_uint32 = 0;
                alignment.GetTag(tag, value_uint32);
                value = value_uint32;
            } break;
            default: {
                alignment.GetTag(tag, value);
            } break;
        }
        return(true);
    } else {
        return(false);
    }
}
