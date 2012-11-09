/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
/* I N C L U D E S ***********************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "samita/align/align_reader.hpp"
#include "samita/common/types.hpp"
#include <samita/common/interval.hpp>
#include <sam.h>
#include <bam.h>
#include "samita/sam/bam_metadata.hpp"
#include "samita/sam/bam.hpp"
#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "semaphore.h"


#define NUMALLELE 6
#define FILENAME_MAX_LENGTH 1024
#define VERSION "0.1.0"
#define PRG_NAME "StrandBiasIdentifier"

using namespace std;
using namespace lifetechnologies;
using namespace ion;

class FlowDist {
	uint32_t refPosition;
	char refAllele;
	uint16_t * plusAlleleCounts;
	uint16_t * negAlleleCounts;
	vector<string> * nameVector;
	vector<uint16_t> * flowPosVector;
        bool isStrandBiasChecked;
 	bool isStrandBiased;	
	public:
	FlowDist():refPosition(0),refAllele('A'),plusAlleleCounts(NULL),negAlleleCounts(NULL), nameVector(NULL), flowPosVector(NULL),isStrandBiasChecked(false),isStrandBiased(false) {
	};
	
	FlowDist(uint32_t refPos, char ref) {
		refPosition = refPos;
		refAllele = ref;
		plusAlleleCounts = new uint16_t[NUMALLELE];
		negAlleleCounts = new uint16_t[NUMALLELE];
		nameVector = new vector<string> ();
		flowPosVector = new vector<uint16_t> ();
		for (int counter = 0; counter < NUMALLELE; counter++) {
			plusAlleleCounts[counter] = 0;
			negAlleleCounts[counter] = 0;
		}
		isStrandBiasChecked = false;
		isStrandBiased = false;
	};
	
	~FlowDist() {
		if (plusAlleleCounts != NULL)
			delete [] plusAlleleCounts;
		if (negAlleleCounts != NULL)
			delete [] negAlleleCounts;
		if (nameVector != NULL)
			delete nameVector;
		if (flowPosVector != NULL)
			delete flowPosVector;
		
	};
	
	
		
	uint32_t getRefPosition() {
		return refPosition;
	};
	
	char getRefAllele() {
		return refAllele;
	};

	bool wasStrandBiasChecked() {
		return isStrandBiasChecked;
	};
	
	vector<string> * getRefNames() {
		return nameVector;
	};
	
	vector<uint16_t> * getFlowPositions () {
		return flowPosVector;
	};
	
	void addRefName(string name) {
		if (nameVector != NULL)
			nameVector->push_back(name);
		else {
			nameVector = new vector<string> ();
			nameVector->push_back(name);
		}
	};
	
	void addFlowPosition(uint16_t flowpos) {
		if (flowPosVector != NULL)
			flowPosVector->push_back(flowpos);
		else {
			flowPosVector = new vector<uint16_t> ();
			flowPosVector->push_back(flowpos);
		}
	};
	
	
	void incrementPlusAlleleCount(char allele) {
		if (tolower(allele) == 'a') 
			plusAlleleCounts[0]++;
		else if (tolower(allele) == 'c')
			plusAlleleCounts[1]++;
		else if (tolower(allele) == 'g')
			plusAlleleCounts[2]++;
		else if (tolower(allele) == 't')
			plusAlleleCounts[3]++;
		else if (tolower(allele) == '-')
			plusAlleleCounts[4]++;
	
	};
	
	void incrementNegAlleleCount(char allele) {
		if (tolower(allele) == 'a') 
			negAlleleCounts[0]++;
		else if (tolower(allele) == 'c')
			negAlleleCounts[1]++;
		else if (tolower(allele) == 'g')
			negAlleleCounts[2]++;
		else if (tolower(allele) == 't')
			negAlleleCounts[3]++;
		else if (tolower(allele) == '-')
			negAlleleCounts[4]++;
	
	};

	bool isStrandBiasPosition(double variantFreqThreshold, double minVariantFreqThreshold, int minCoverageThreshold) {
		if (isStrandBiasChecked) 
			return isStrandBiased;
		else
			return checkStrandBiasPosition(variantFreqThreshold, minVariantFreqThreshold, minCoverageThreshold);
	};
	
	bool checkStrandBiasPosition(double variantFreqThreshold, double minVariantFreqThreshold, int minCoverageThreshold) {
		int totPosCount = 0;
		int totNegCount = 0;
		int plusVarCount = 0;
		int negVarCount = 0;
		int varAlleleIndex = 0;
		int refAlleleIndex = 0;
		double strandBias  = 0;
	        isStrandBiasChecked = true;
	
		for (int i = 0; i < 5; i++) {
			totPosCount += plusAlleleCounts[i];
			totNegCount += negAlleleCounts[i];
		}
		
		if (tolower(refAllele) == 'a')
			refAlleleIndex = 0;
		else if (tolower(refAllele) == 'c')
			refAlleleIndex = 1;
		else if (tolower(refAllele) == 'g')
			refAlleleIndex = 2;
		else if	(tolower(refAllele) == 't')
			refAlleleIndex = 3;
			
		int maxCount = 0;

		for (int i = 0; i < 5; i++) {
				if (plusAlleleCounts[i] > maxCount && i != refAlleleIndex) {
					varAlleleIndex = i;
					maxCount = plusAlleleCounts[i];
				}
				if (negAlleleCounts[i] > maxCount && i != refAlleleIndex) {
					varAlleleIndex = i;
					maxCount = negAlleleCounts[i];
				}
		}
		
		plusVarCount = plusAlleleCounts[varAlleleIndex];
		negVarCount = negAlleleCounts[varAlleleIndex];
		
		long int num = max((long int)plusVarCount*totNegCount, (long int)negVarCount*totPosCount);
		long int denum = (long int)plusVarCount*totNegCount + (long int)negVarCount*totPosCount;
		
		if ((plusVarCount*totNegCount + negVarCount*totPosCount) == 0 || (plusVarCount < minCoverageThreshold && negVarCount < minCoverageThreshold) || ((float)totPosCount)/(totPosCount+totNegCount) > 0.90 || ((float)totNegCount)/(totPosCount+totNegCount) > 0.90)
			return 0;
			
		strandBias = (double)num/denum;
		
				
		if (strandBias > 1.0)
			cout << "strand = " << strandBias << "Max = " << num << " denominator = " << denum << endl;
			
		if (strandBias > variantFreqThreshold && (((double)(plusVarCount)/(totPosCount) > minVariantFreqThreshold ) || ((double)(negVarCount)/(totNegCount) > minVariantFreqThreshold) ) ) {
			isStrandBiased = true;
			return 1;
		
		}
		
		return 0;
		
	}
};

void help_message(void);
void version_info(void);
void read_fasta(string, map<string, string> &);
void* read_sam(void *);
void get_alignments(string base_seq, string *chrseq, int start_pos, string cigar, string & ref_aln, string & seq_aln);
int get_next_event_pos (string cigar, int pos);
void getHardClipPos(string cigar, int & startHC, int & startSC, int & endHC, int & endSC) ;
bool getStrand (int mates_flag);
int get_seq_length(string);
void parse_cigar(string cigar, bool strand, int &, int, int &, int &, int &, int &, int &, int &);
void deleteArray (double *** array, int ht, int width);
void allocateArray (double *** array, int ht, int width);
void parse_flag(int mates_flag, int & isFirst, int & isSecond, int & isUnMapped, int & isMateUnMapped, int & isSecondary, int &isProperPair, int &, int &);
void dec2bin(int number, int** bin, int index);
void update_statistics (string seq_name, string *chrseq, bool strand, string ref_aln, string seq_aln, int start_pos, 
			int startHC, int startSC, int endHC, int endSC,  vector<int> *flowIndex,  map<uint32_t, FlowDist*> *vec_positions, long int * maxFilledPosition );
string intToString(int x, int width) ;

//Input Structures
static map<string,string> chr;
static vector<string> sequences;
static vector<string> refSequences;

//default threshold variables
double variantFreqThreshold = 0.95;
double minVariantFreqThreshold = 0.10;
int minCoverageThreshold = 3;

//string chrseq;
string chrLenseq;
string chr_id = "";
int chrNumInput = 0;
int numberOfContigs = 0;
int numSeqInBam = 0;
long int totalCheckedPositions = 0;
int totalStrandBiasPositions = 0;

//Input files
string inputName = "";
string indexName = "";
string chrFile = "";
string chrFileIub = "";

string selectRegion = "";
string selectChr = "";
string outputDir = ".";
string outputFitResults = "results";
struct stat out_stat;
bool flowSigPresent = false;
string flowOrder = "";
string flowKey = "TCAG";
int totalSC = 0;

ofstream tempDataFile;
char *tempDataFileName;

int max_threads = 8;
semaphore_t max_threads_sem;
semaphore_t total_threads_sem;

void help_message() {
    fprintf(stdout, "Usage: %s %s \n", PRG_NAME, "[options]");
    fprintf(stdout, "\t%-20s%-20s\n", "-h", "This help message");
    fprintf(stdout, "\t%-20s%-20s\n", "-v", "Program and version information");
    fprintf(stdout, "\t%-20s%-20s %-20s %-20s %-20s %s %s\n",
                  	" -r <reference file name>",
					" -n <number of threads default=8>",
					" -t <strand-bias threshold default=0.95>",
					" -c <variant freq threshold default=0.10>",
					" -m <Min number of reads with variant allele default = 3>",
					" -b <BAM file name> "
					" -i <BAM Index bai, file name>"
                    " -o <Output Directory name>",
                    " -f <Output file name>" );
}

void version_info() {
     // fprintf (stdout, "%s", IonVersion::GetFullVersion ("StrandBiasIdentifier").c_str());

}

void Tokenize(const string& str,
                      vector<string>& tokens,
                      const string& delimiters = " ")
{
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     =  str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}




void initialize(string bamFileName1) 
{
   
   vector<BamHeader> bamHeaders;
   AlignReader bam(inputName.c_str());
   BamHeader &origheader = bam.getHeader();
   bamHeaders.push_back(origheader);
   BamMetadata header(origheader);
   SequenceIntervalArray sequenceIntervals = header.getIntervals();
   vector<SequenceInterval>::const_iterator siItr = sequenceIntervals.begin();
   string bamseq;  
   for (; siItr != sequenceIntervals.end(); ++siItr) {
		bamseq = (*siItr).getSequence();
		cout << "Contig Name in BAM file : " << bamseq << endl;
		sequences.push_back(bamseq);
		numSeqInBam++;
		bool reffound = false;
		//check if reference fasta file provoided has the sequence name , if not exit
		for (unsigned int i = 0; i < refSequences.size(); i++)
		{
			if (refSequences[i].compare(bamseq) == 0)
			{
				reffound = true;
				break;
			
			}
		}
		if (!reffound ) {
			cerr << "FATAL: Sequence Name: " << bamseq << " in BAM file is not found in Reference csfasta file provoided " << endl;
			exit(-1);
		}
		
   }
   vector<RG> allrg = header.getReadGroups();
   vector<RG>::const_iterator rgItr = allrg.begin();
   for(; rgItr != allrg.end(); ++rgItr) {
       flowOrder = (*rgItr).FO;
		flowKey = (*rgItr).KS;
      
    }

    if (flowOrder.length() > 0) flowSigPresent = true;

	cout << "Finished Initialization - flowPresent = " << flowSigPresent << " flowOrder  = " << flowOrder << endl;
    
	
}

void read_fasta(string chr_file, map<string, string> & chr_seq)
{
	//chr_seq.clear();
	map<string,string>::iterator it;
	
	string temp_chr_seq = "";
	string line = "";
	ifstream inFile (chr_file.c_str());
	string chr_num = "";
	int chrNumber = 0;
	string chrName;
	if (!inFile.is_open())
	{
		cout << "Error opening file: " << chr_file << endl;
		return;
	}
	getline (inFile,line);
	
	it = chr_seq.begin();
	
	// Read and process records
	while (!inFile.eof())
	{
           	if (line == "" || line.at(0) == '#') 
        	{
           	  getline(inFile,line);
        	}
        	else if (line.at(0) == '>' || line.at(0) == '<')
        	{
					
             		if (chrNumber > 0 )
             		{
                 		//chr_seq.push_back(temp_chr_seq);
						chr_seq.insert (it, pair<string,string>(chrName,temp_chr_seq));
						
						cout << "Chromosome Name = " << chrName << endl;
						
						if (temp_chr_seq.length() > 0 )
						{
							cout << temp_chr_seq.length() << endl;
							temp_chr_seq.clear();
						}
						
						//chrNumber++;
             		}
					
					chrNumber++;
					//find if there are more than contig name in the line
					int firstspace = line.find(" ");	
					chrName = line.substr(1, firstspace-1);
					if (chrName.length() == 0)
					{
						cerr << " Reference csfasta file provoided has no contig name : " << line << endl;
						exit(-1);
					}	
					cout << " Chr Name found in Reference csfasta : " << chrName << endl;
					refSequences.push_back(chrName);

             		getline(inFile, line);
             		
        	}
			else  
        	{
             	temp_chr_seq.append(line);
	     		getline (inFile,line);
        	}
       
	}
	
	if (temp_chr_seq.length() > 0)
	{
		//cout << temp_chr_seq.length() << endl;cout << "Dima's seq : " << temp_chr_seq.substr(279750,100) << endl;
		chr_seq.insert (it, pair<string,string>(chrName,temp_chr_seq));
	}
	
	
  
	numberOfContigs = chrNumber;
	inFile.close();
}

// Returns true for + (forward) and false for - (reverse) strand
bool getStrand (int mates_flag)
{
	// 2^5 's re//mainder = 0 for forward strand and = 1 for reverse strand

	mates_flag /= 2;
	mates_flag /= 2;
	mates_flag /= 2;
	mates_flag /= 2;
	if (mates_flag % 2 == 0)
		return true;
	else
		return false;
}

void getHardClipPos(string cigar, int & startHC, int & startSC, int & endHC, int & endSC) {	
	bool matchFound = false;
	unsigned int cigar_pos = 0;
	while(cigar_pos < cigar.length())
	{
		unsigned int event_pos = get_next_event_pos(cigar,cigar_pos);
		if (event_pos >= cigar.length())
           	{
                         break;
           	}
           	char event = cigar.at(event_pos);
           	int l = atoi(cigar.substr(cigar_pos,event_pos-cigar_pos).c_str());
		if (event == 'M') matchFound = true;
		else if (event == 'H') {
		   if (matchFound) endHC = l;
		   else startHC = l;
		}
		else if (event == 'S') {
		   if (matchFound) endSC = l;
		   else startSC = l;
		}
		cigar_pos = event_pos + 1;

	}
}


void parse_cigar(string cigar, bool strand, int & offset,  int seqLen, int & softCPos, int & totIns, int & totDel, int & totMatch, int & maxMatch, int & totSC)
{
    
     unsigned int cigar_pos = 0;
     int rightOffset = 0;
     int matchLength = 0;
     int totLength = 0;
     int numInsertions = 0;
     int numDeletions = 0;
     int maxMatchLen = 0;
     int totMatchLen = 0;
   
     while (cigar_pos < cigar.length())
     {
           unsigned int event_pos = get_next_event_pos(cigar, cigar_pos);
           if (event_pos >= cigar.length())
           {
                         break;
           }
           char event = cigar.at(event_pos);
           int l = atoi(cigar.substr(cigar_pos,event_pos-cigar_pos).c_str());
           
           if ((event == 'H' || event == 'S') && cigar_pos == 0)
           {
                     offset = l;
					 if (event == 'S') {
						softCPos = l;
						totSC += l;
						totLength += l;
					}
           }
           else if (event == 'M')
           {
                matchLength = l;
				if (matchLength > maxMatchLen) 
					maxMatchLen = matchLength;
				totMatchLen += l;
				totLength += l;
           }
           else if (event == 'H' || event == 'S')
           {
                rightOffset = l;
				if (event == 'S') {
					softCPos = l;
					totSC += l;
					totLength += l;
				}
           }
		   else if (event == 'I') {
				numInsertions++;
						
				totLength += l;
		    }
			else if (event == 'D') {
				numDeletions++;
				
			}
			else if (event == 'N') {
			
			}

           cigar_pos = event_pos + 1;

     } //end while loop
     
     if (totLength != seqLen)
     {
                //cout << "Warning total CIGAR length is not matching SEQ length " << cigar << endl;
     }

     if (!strand)
	offset = rightOffset;

     
	
}


void deleteArray(double *** darray, int HT, int WD) {

    for (int i = 0; i < HT; i++) {
	delete [] (*darray)[i];
//	cout << "Deleting row " << i << endl;
    }
    delete [] *darray;
     //cout << "Deleted array" << endl;
}

void allocateArray(double *** array, int HT, int WD) {
	*array = new double*[HT];
	for (int i = 0; i < HT; i++ )
	   (*array)[i] = new double[WD];

	//initialize the array once allocated
	for (int i = 0; i < HT; i++) 
	   for (int j = 0; j < WD; j++)
		(*array)[i][j] = 0;
}

void * read_sam(void * selectRegion)
{
	
    string region = "";
   	region = (*(string *)selectRegion);  string *chrseq = new string(); 
	
    AlignReader sam;
    sam.open(inputName.c_str());
	if (region.length() > 0) {
		cout << "Processing BAM for region " << region << endl;
		sam.select(region.c_str());
	}
	
    AlignReader::iterator iter = sam.begin();
    AlignReader::iterator end = sam.end();
	
	map<string,string>::iterator chrit = chr.begin();
	map<string,string>::iterator chritend = chr.end();
	map<uint32_t, FlowDist*> *vec_positions = new map<uint32_t, FlowDist*>();
	
    string seq_name;
    string chr_i = "";
	int start_pos = 0;
	
    string base_seq;
  	string cigar;
    string ref_aln;
    string seq_aln;
	int n_reads = 0;
	string flowsigStr;
	int startHC =0;
	int endHC = 0;
	int startSC = 0;
	int endSC = 0;

	int contigIndex = 0;
	long int maxPosition = 0;
	long int * maxFilledPosition = &maxPosition;		
	
	if (region.length() > 0) {
		chrit = chr.find(region);
		if (chrit != chritend) {
			*chrseq = (string)chrit->second;
		}
		else
		{
			cerr << "FATAL: Reference sequence for Contig " << region << " specified using -region option, not found in reference fasta file " << endl;
			pthread_exit((void *)-1);
		}
		
		
	}
	
	for (unsigned int i = 0; i < refSequences.size(); i++)
	{
		if (refSequences[i].compare(region) == 0)
		{
			contigIndex = i;
			break;
		}
	}
	
    vector<int> *flowIndex = NULL;
           // Read and process records
    while (iter != end)
    {
        Align const& a = *iter;
       
		flowIndex = new vector<int>();
		
		seq_name = a.getName();
		bool strand = a.getStrand();
				
		chr_i = intToString(a.getRefId(),0);
		
		if (chr_i.length() == 0 || chr_i.compare("-1") == 0) {
				break; // stop once you have reached unmapped beads
		}
		
					
			start_pos = a.getStart()-1;
			cigar = a.getCigar().toString();
			base_seq = a.getSeq();
						
			//parse_cigar(cigar, strand, offset, base_seq_length, softclipPos, numInsertions, numDeletions, totalMatch, maxMatch, totalSC);
			       		
			flowsigStr = a.getFlowSignals();
			if (flowsigStr.length() > 0) {
				flowSigPresent = true;
				vector<string> flowsigTokens;
				Tokenize(flowsigStr, flowsigTokens, ",");
				vector<string>::iterator it = flowsigTokens.begin();
				it++; //skip the first token FZ:B:S,
				int flowsig = 0;
				int nbases = 0;
				int basecounter = 0;
				int flowpos = 0;
				for (; it < flowsigTokens.end(); it++) {
					flowsig = atoi((*it).c_str());
					flowpos++;
					nbases = (int) round((float)flowsig/100);
					basecounter = 0;
					while(basecounter < nbases) {
						basecounter++;
						flowIndex->push_back(flowpos);
					}
				}
			}

			n_reads++;
			
			get_alignments(base_seq, chrseq, start_pos, cigar,ref_aln,seq_aln);

			startHC = endHC = startSC = endSC = 0;
			getHardClipPos(cigar, startHC, startSC, endHC, endSC);
			
			update_statistics(seq_name, chrseq, strand, ref_aln, seq_aln, start_pos, startHC, startSC, endHC, endSC, flowIndex, vec_positions, maxFilledPosition);
			
			if (n_reads % 100000 == 0) {
					
					map<uint32_t,FlowDist*>::iterator iterVF = vec_positions->begin();
					FlowDist *tempDist;
					while (iterVF != vec_positions->end() ) {
						tempDist= iterVF->second;
						if (tempDist->getRefPosition() >= (uint32_t)start_pos ) {
							break;
						}
						else {
							if (tempDist->isStrandBiasPosition( variantFreqThreshold , minVariantFreqThreshold, minCoverageThreshold)) {
								//totalStrandBiasPositions++;
								//refnames = tempDist->getRefNames();
								//flowpositions = tempDist->getFlowPositions();
								//for (int count = 0; count < refnames->size(); count++) {
								//	tempDataFile << tempDist->getRefPosition() << "," << (*refnames)[count] << "," << (*flowpositions)[count] << std::endl;
						
								//}
								iterVF++;
							}
							else {
								delete tempDist;
								vec_positions->erase(iterVF++);
							}
						}
					}
					cout << "Finished processing reads = " << n_reads << endl;
			}
		
			++iter;
		
		delete flowIndex;
		

	}
	
	while (total_threads_sem.count < contigIndex) {
			pthread_cond_wait(&(total_threads_sem.cond), &(total_threads_sem.mutex));
	}
	
	//if (total_threads_sem.count == contigIndex) {
		map<uint32_t,FlowDist*>::iterator iterVF;
		vector<string> * refnames;
		vector<uint16_t> * flowpositions;
		FlowDist *tempDist;
		for (iterVF=vec_positions->begin(); iterVF != vec_positions->end(); iterVF++) {
			 tempDist= iterVF->second;
			if (tempDist->isStrandBiasPosition(variantFreqThreshold , minVariantFreqThreshold, minCoverageThreshold)) {
				totalStrandBiasPositions++;
				refnames = tempDist->getRefNames();
				flowpositions = tempDist->getFlowPositions();
				for (unsigned int count = 0; count < refnames->size(); count++) {
					tempDataFile << region << "\t" << tempDist->getRefPosition()+1 << "\t" << (*refnames)[count] << "\t" << (*flowpositions)[count] << std::endl;
					
				}
			}
			delete tempDist;
						
		}	
			 //iterVF = vec_homPoly_dist_nmer.erase(iterVF);
		
		
		vec_positions->clear();
	//}
		delete chrseq;	
		up(&total_threads_sem);
		down(&max_threads_sem);
		delete vec_positions;
		pthread_exit((void *)0);
		
}

void update_statistics (string seq_name, string *chrseq, bool strand, string ref_aln, string seq_aln, int start_pos, 
			int startHC, int startSC, int endHC, int endSC,  vector<int> *flowIndex, map<uint32_t, FlowDist*> *vec_positions, long int * maxFilledPosition ) 
{
	
	bool flowSigPresent = false;
	
	vector<int> *flowIndexRev = NULL;
	
	
	if ( flowIndex != NULL ) flowSigPresent = true;

	//if flow present and mapped to negative strand reverse the flows
	if (!strand) {
		flowIndexRev = new vector<int>();
		vector<int>::reverse_iterator flowIndexRevItr;
		for(flowIndexRevItr = flowIndex->rbegin(); flowIndexRevItr < flowIndex->rend(); flowIndexRevItr++) {
			flowIndexRev->push_back(*flowIndexRevItr);
			//cout << "Reversing index " << *flowIndexRevItr << " ";
		}
		
	}
	
	int pos_ref = start_pos;
	int pos_seq = 0;
	int flowPos = 0;
	int flowIndexPos = 0;
	if (strand) flowPos = 5 + startHC + startSC;//1-based flow positions
	else flowPos = 1 + startHC + startSC; //1-based flow positions

	for(unsigned int i=0; i< ref_aln.length() && i<seq_aln.length();i++) 
	{
		if((uint32_t)pos_ref>=(*chrseq).length() ) 
		{
			cout << "WARN: Mapping found outside the chromosome: " << ". Start position: " << start_pos << " Chromosome Length = " << (*chrseq).length() << " " << chrLenseq.length() << endl;
			break;
		}
		char base_ref = tolower(ref_aln.at(i));
		char base_seq = tolower(seq_aln.at(i));
						
		if (flowSigPresent && strand) { 
			if (flowPos < 1 ) {
			  cout << "ERR: Position out of range of vector FlowSigValue: flowPos = " << flowPos << " vector size = " << flowIndex->size() << endl;
			  exit(-1);
			}
			
			flowIndexPos = flowIndex->at(flowPos-1);
			
			
		}
		else if (flowSigPresent && !strand) {
			if (flowPos < 1 ) {
			  cout << "ERR: Position out of range of vector FlowSigRev: flowPos = " << flowPos << " vector size = " << endl;
			  exit(-1);
			}
			
			flowIndexPos = flowIndexRev->at(flowPos-1);
			
		}
	
		
			if(base_ref != '-' && base_seq !='-') {
				FlowDist *tempDist;				
				if (pos_ref <= *maxFilledPosition) {
					map<uint32_t,FlowDist*>::iterator iterVF;
					iterVF = vec_positions->find(pos_ref);
					if (iterVF != vec_positions->end()) {
						tempDist = iterVF->second;
						tempDist->addRefName(seq_name);
						tempDist->addFlowPosition(flowIndexPos);
						if (strand)
							tempDist->incrementPlusAlleleCount(base_seq);
						else
							tempDist->incrementNegAlleleCount(base_seq);
					}
				}
				else {
					tempDist = new FlowDist(pos_ref, base_ref);
					vec_positions->insert(pair<uint32_t, FlowDist*>(pos_ref, tempDist));
					tempDist->addRefName(seq_name);
					tempDist->addFlowPosition(flowIndexPos);
					if (strand)
						tempDist->incrementPlusAlleleCount(base_seq);
					else
						tempDist->incrementNegAlleleCount(base_seq);
					
					//vec_positions->insert(pair<uint32_t, FlowDist*>(pos_ref, tempDist));
					*maxFilledPosition = pos_ref;
					totalCheckedPositions++;
				}
				flowPos++;
				pos_ref++;
				pos_seq++;
			}
			else if (base_ref == '-' && base_seq != '-' ) { //Insertion
				pos_seq++;
				flowPos++;
			}
			else if (base_ref != '-' && base_seq == '-') {
				FlowDist *tempDist;
                                if (pos_ref <= *maxFilledPosition) {
                                        map<uint32_t,FlowDist*>::iterator iterVF;
                                        iterVF = vec_positions->find(pos_ref);
                                        if (iterVF != vec_positions->end()) {
                                                tempDist = iterVF->second;
                                                tempDist->addRefName(seq_name);
                                                tempDist->addFlowPosition(flowIndexPos);
                                                if (strand)
                                                        tempDist->incrementPlusAlleleCount(base_seq);
                                                else
                                                        tempDist->incrementNegAlleleCount(base_seq);
                                        }
                                }
                                else {
                                        tempDist = new FlowDist(pos_ref, base_ref);
                                        tempDist->addRefName(seq_name);
                                        tempDist->addFlowPosition(flowIndexPos);
                                        if (strand)
                                                tempDist->incrementPlusAlleleCount(base_seq);
                                        else
                                                tempDist->incrementNegAlleleCount(base_seq);

                                        vec_positions->insert(pair<uint32_t, FlowDist*>(pos_ref, tempDist));
                                        *maxFilledPosition = pos_ref;
                                        totalCheckedPositions++;
                                }

				pos_ref++;
			}
		
		
	}
}

void get_alignments(string base_seq, string *chrseq, int start_pos, string cigar, string & ref_aln, string & seq_aln) 
{
	unsigned int ref_pos = (uint32_t)start_pos;
	unsigned int seq_pos = 0;
	unsigned int cigar_pos = 0;
	ref_aln.assign("");
	seq_aln.assign("");
	
	while (seq_pos < base_seq.length() && ref_pos < (*chrseq).length() && cigar_pos < cigar.length()) 
	{
		//cout << "cigar_pos " << cigar_pos << endl;
		int event_pos = get_next_event_pos(cigar,cigar_pos);
		
		if(event_pos >= (int)cigar.length())
		{
			//cout << "break reached enventpos = " << event_pos << "cigar length = " << cigar.length() << endl;
			break;
		}
		
		char event = cigar.at(event_pos);
		
		int l = atoi(cigar.substr(cigar_pos,event_pos-cigar_pos).c_str());
		
		if (event == 'M')
		{
			seq_aln.append(base_seq,seq_pos,l);
			ref_aln.append(*chrseq,ref_pos,l);
			seq_pos +=l;
			ref_pos += l;
		}
		else if (event == 'D' || event == 'N') 
		{
			seq_aln.append(l,'-');
			ref_aln.append(*chrseq,ref_pos,l);
			ref_pos += l;
		}
		else if (event == 'I') 
		{
			ref_aln.append(l,'-');
			seq_aln.append(base_seq,seq_pos,l);
			seq_pos +=l;
		}
		else if (event == 'S') 
		{
			seq_pos += l;
		}
		
		cigar_pos = event_pos + 1;
		
		
	}

	
}

int get_next_event_pos (string cigar, int pos)
{
	for(unsigned int i=pos; i<cigar.length();i++)
	{
		
		if(!isdigit(cigar.at(i)))
		{
			return i;
		}	
	}
	return cigar.length();
}

std::string intToString(int x, int width) { 
      std::string r;
      std::stringstream s;
     s << x;
     r = s.str();  
     for (int i = r.length(); i <= width; i++) {
			r += " ";
     }
 
      return r;
} 

int main (int argc, char* argv[])
{
	int c;
   
	char *opt_ref_value = NULL;
	char *opt_bam_value = NULL;
	char *opt_bam_index_value = NULL;
	char *opt_output_value = NULL;
	char *opt_output_file_value = NULL;
	char *opt_num_threads = NULL;
	char *opt_variant_threshold = NULL;
	char *opt_variant_freq_threshold = NULL;
	char *opt_coverage_threshold = NULL;
    while( (c = getopt(argc, argv, "hvr:n:b:i:o:f:t:c:m:")) != -1 ) {
		
        switch(c) {
            case 'h':
                help_message();
                exit(0);
                break;
            case 'v':
                version_info();
                exit(0);
                break;
         	case 'r':
                opt_ref_value = optarg;
                break;
			case 'b':
				opt_bam_value = optarg;
				break;
			case 'i':
				opt_bam_index_value = optarg;
				break;
			case 'o':
				opt_output_value = optarg;
				break;
			case 'f':
				opt_output_file_value = optarg;
				break;
            case 'n':
                opt_num_threads = optarg;   
                break;
			case 't':
                opt_variant_threshold = optarg;   
                break;
			case 'c':
                opt_variant_freq_threshold = optarg;   
                break;
		case 'm':
			opt_coverage_threshold = optarg;
			break;
            case '?':
                exit(1);
            default:
                abort();
        }
    }
    
	if ( opt_ref_value != NULL ) {
        //strncpy(chrFile, opt_ref_value, FILENAME_MAX_LENGTH-1);
		chrFile = string((const char *) opt_ref_value);
        //chrFile[FILENAME_MAX_LENGTH-1]='\0';
		
    }
	else {
		cout << "ERROR: Reference file not specified  "  << endl;
		help_message();
		exit(-1);
	}
	if ( opt_bam_value != NULL ) {
        //strncpy(inputName, opt_bam_value, FILENAME_MAX_LENGTH-1);
        //inputName[FILENAME_MAX_LENGTH-1]='\0';
		inputName = string((const char*) opt_bam_value);
    }
	else {
		cout << "ERROR: Input BAM file not specified  "  << endl;
		help_message();
		exit(-1);
	}
	if ( opt_bam_index_value != NULL ) {
       // strncpy(indexName, opt_bam_index_value, FILENAME_MAX_LENGTH-1);
        //indexName[FILENAME_MAX_LENGTH-1]='\0';
		indexName = string((const char *) opt_bam_index_value);
    }
	else {
		cout << "ERROR: Input BAM Index (bai) file not specified  "  << endl;
		help_message();
		exit(-1);
	}
	if ( opt_output_value != NULL ) {
        //strncpy(outputDir, opt_output_value, FILENAME_MAX_LENGTH-1);
        //outputDir[FILENAME_MAX_LENGTH-1]='\0';
		outputDir = string((const char *) opt_output_value);
		
    }
	else {
		cout << "ERROR: Output Directory not specified  "  << endl;
		help_message();
		exit(-1);
	}
	if ( opt_output_file_value != NULL ) {
        //strncpy(outputFitResults, opt_output_file_value, FILENAME_MAX_LENGTH-1);
        //outputFitResults[FILENAME_MAX_LENGTH-1]='\0';
		outputFitResults = string((const char *) opt_output_file_value);
		
    }
	else {
		cout << "ERROR: Output Filename not specified  "  << endl;
		help_message();
		exit(-1);
	}
	if (opt_num_threads != NULL) {
		max_threads = atoi(opt_num_threads);
	}
	if (opt_variant_threshold != NULL) {
		variantFreqThreshold = atof(opt_variant_threshold);
	}
	if (opt_variant_freq_threshold != NULL) {
		minVariantFreqThreshold = atof(opt_variant_freq_threshold);
	}
	if (opt_coverage_threshold != NULL) {
                minCoverageThreshold = atoi(opt_coverage_threshold);
        }

	
	
	
	if (stat(chrFile.c_str(), &out_stat) != 0)	
	{
		cout << "ERROR: Reference file specified does not exist - " << chrFile << endl;
		help_message();
		exit(-1);
	}
	if (stat(outputDir.c_str(), &out_stat) != 0)	
	{
		cout << "ERROR Invalid Output directory specified " << outputDir << endl;
		help_message();
		exit(-1);
	}
	

	string temp = outputDir + "/" + outputFitResults;
	tempDataFileName = (char*)temp.c_str();
	tempDataFile.open(tempDataFileName);
	
	cout << "Loading reference." << endl;
	read_fasta(chrFile,chr);
	cout << "Loaded reference. Ref length: " << chr.size() << endl;
	initialize(inputName);
	//thread pointer
	pthread_t *thread;
	std::vector<pthread_t*> thread_vector;
	
	init_semaphore(&max_threads_sem, 0);
	init_semaphore(&total_threads_sem, 0);
	int thread_ret_value = 0;
	
	for (unsigned int i = 0; i < refSequences.size(); i++)
	{
		while (max_threads_sem.count >= max_threads) {
				pthread_cond_wait(&(max_threads_sem.cond), &(max_threads_sem.mutex));
		}
			
		if (max_threads_sem.count < max_threads) {
			thread = new pthread_t();
			thread_ret_value = pthread_create(thread, NULL, &read_sam, (void *)&refSequences[i]);
			if (thread_ret_value) {
					fprintf(stderr, "Error: Unable to create thread - Return Value = %d \n", thread_ret_value);
					exit(-1);
			}
			thread_vector.push_back(thread);
			up(&max_threads_sem);
		}
	}
	
	while (max_threads_sem.count > 0) {
		pthread_cond_wait(&(max_threads_sem.cond), &(max_threads_sem.mutex));
	}
	
	cout << "Total strand Bias positions = " << totalStrandBiasPositions << endl;
	cout << "Total checked positions = " << totalCheckedPositions << endl;
	tempDataFile.close();
	
	

	//delete all the thread objects created
	std::vector<pthread_t*>::iterator thread_vector_iterator;
	for (thread_vector_iterator=thread_vector.begin(); thread_vector_iterator < thread_vector.end(); thread_vector_iterator++) {
		delete *thread_vector_iterator;
	
	}
	
}
