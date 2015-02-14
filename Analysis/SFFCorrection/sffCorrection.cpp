/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

/* I N C L U D E S ***********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sstream>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <tr1/unordered_map>
#include <set>
#include <fstream>
#include "IonVersion.h"

#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/BamWriter.h"

extern "C" {
	#include "../file-io/sff_definitions.h"
	#include "../file-io/sff_header.h"
	#include "../file-io/sff_read_header.h"
	#include "../file-io/sff_read.h"
    #include "../file-io/ion_alloc.h"
}
#include "SmithWaterman.h"
#include "semaphore.h"

#define DEAFAUL_QUALITY 62

using namespace ion;
using namespace std;
using namespace BamTools;

/* D E F I N E S *************************************************************/
#define VERSION "0.1.0"
#define PRG_NAME "PairedEndSFFCorrection"
#define FASTQ_FILENAME_MAX_LENGTH 1024
#define SFF_FILENAME_MAX_LENGTH 1024
#define FLOW_ORDER  "TACGTACGTCTGAGCATCGATCGATGTACAGC"
#define FLOW_CYCLES 64
#define BLANK " ";
#define MAX_NUM_THREADS 10
#define MAX_PAIRS_PER_THREAD 10000

const int DEBUG = 0;

/* P R O T O T Y P E S *******************************************************/
void help_message(void);
void version_info(void);
void process_options(int argc, char *argv[]);
void process_sff(char *sff_file, char *sff_file_rev,  char *fastq_file, char *out_sff_file, int trim_flag);
char* getBasename(char *path);
void tokenizeName(char *name_fwd, char *name_rev, int fwdlen, int revlen, int * fwd_x, int *fwd_y, int *rev_x, int *rev_y);
//int getNextSFFPair( int numReads, bool *fwd, bool *rev, FILE *sff_fp, FILE *sff_fp_rev, sff_read_header_t **rh, sff_read_header_t **rh_rev, sff_read_t **rd, sff_read_t **rd_rev, sff_header_t *h, sff_header_t *h_rev);
int getNextSFFPair( int numReads, bool *fwd, bool *rev, BamReader *pBamReader, BamReader *pBamReader_rev, sff_read_header_t **rh, sff_read_header_t **rh_rev, sff_read_t **rd, sff_read_t **rd_rev, sff_header_t *h, sff_header_t *h_rev);

std::string intToString(int x, int width);
std::string reverseComplement(std::string a);
int* reverseFlowIndex(uint8_t *flow_index, int nbases, int left_clip);
int* sumFlowIndex(uint8_t *flow_index, int nbases, int left_clip);
char complimentNuc(char);
void construct_fastq_entry(FILE *fp,
                           char *name, char *name_rev,
                           char *bases, char *bases_rev,
                           uint8_t *quality, uint8_t *quality_rev,
                           int nbases, int nbases_rev,
			   uint16_t *flows, uint16_t *flows_rev,
			   int nflows,
			   uint8_t *flow_index, uint8_t *flow_index_rev);

void output_fastq_entry(FILE *fp, char *name, std::string bases, int left_trim);

int roundsff(double x);
int errorCorrect(Alignment *alignment,  int*, int*, uint16_t *, uint16_t *,  char *, uint16_t, char *, uint16_t, ion_string_t *, ion_string_t *,
				uint16_t **, uint8_t **, uint8_t **, char **, char **, int *, int *, bool *, int nbasesRev, 
				int *mergedRegionStartBase, int *mergedRegionStopBase,  int *mergedRegionStartBaseFwd, int *mergedRegionStopBaseFwd,
				int *mergedRegionStartBaseRev, int *mergedRegionStopBaseRev, bool regionInfo);

void refineFlowAlignment(int *flowIndexFwd, int *flowIndexRev,std::string &fwdFlowAligne, std::string &revFlowAligned, char *flow_seq_fwd, char *flow_seq_rev, uint16_t *flow_sig_fwd, uint16_t *flow_sig_rev);
std::string refineGapRegion(int *flowIndexFwd, int *flowIndexRev,std::string &flowFwdAligned, std::string &flowRevAligned, char *flow_seq_fwd, char *flow_seq_rev, uint16_t *flow_sig_fwd, uint16_t *flow_sig_rev, int reg_start, int reg_stop, bool gaps_on_fwd);


/* G L O B A L S *************************************************************/
char output_fastq_file[FASTQ_FILENAME_MAX_LENGTH] = { '\0' };
char output_sff_file[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_sff_file_corrected[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_sff_file_singleton_fwd[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_sff_file_singleton_rev[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_sff_file_paired_fwd[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_sff_file_paired_rev[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_merged_region_file[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_merged_region_file_fwd[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_merged_region_file_rev[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char output_statistics_file[SFF_FILENAME_MAX_LENGTH] = { '\0' };

FILE *fastq_fp; //*sff_fp, *sff_fp_rev, *out_sff_fp, *out_sff_fwd_fp, *out_sff_rev_fp, *out_sff_paired_fwd_fp, *out_sff_paired_rev_fp;
FILE *region_merged_fp, *region_merged_fwd_fp, *region_merged_rev_fp;  
FILE *output_stats_fp;
 
char sff_file_fwd[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char sff_file_rev[SFF_FILENAME_MAX_LENGTH] = { '\0' };

//variables and functions for the handling of strand bias
char input_strand_bias_file_fwd[SFF_FILENAME_MAX_LENGTH] = { '\0' };
char input_strand_bias_file_rev[SFF_FILENAME_MAX_LENGTH] = { '\0' };
bool has_strand_bias_info_fwd = false;
bool has_strand_bias_info_rev = false;
std::tr1::unordered_map<std::string, std::set<int> > strand_bias_flow_index_map_fwd;
std::tr1::unordered_map<std::string, std::set<int> > strand_bias_flow_index_map_rev;
bool build_strand_bias_map(std::string strand_bias_file, std::tr1::unordered_map<std::string, std::set<int> > *bias_map);
bool find_read_in_bias_map(std::string read_name, std::tr1::unordered_map<std::string, std::set<int> > *bias_map, std::set<int> *bias_flow_index_set);
bool find_flow_index_with_bias(std::set<int> bias_flow_index_set, int flow_index);

char flow_order[64] = FLOW_ORDER;
int skipFWD = 0;
int skipREV = 0;
int numCorrected = 0;
int numIdentical = 0;
int numUnCorrected = 0;
const int BLOCKN = 50;
const int BLOCKM = 50;
const int MAXM = 100;
int fseekOffset = 0;
const float ALIGN_PRIOR =0.01;
const float NEUTRAL_PRIOR =1.0;
const float MISMATCH_PRIOR =1.0;
int  trim_flag  = 1;
int max_threads = MAX_NUM_THREADS;
int max_pairs_per_thread = MAX_PAIRS_PER_THREAD;
int threadCounter = 0;
int totalFlowLengthFwd = 260; //default values, changed later from sff read header
int totalFlowLengthRev = 260;
bool outputIdenticalOnly = false;

BamWriter bamWriter_c, bamWriter_f, bamWriter_r, bamWriter_fo, bamWriter_ro;
string rname, rname_rev;

//create two couting semaphores to keep track of max number of threads currently running and max number of total threads opened
semaphore_t max_threads_sem;
semaphore_t total_threads_sem;
	
struct sff_pair {
	bool fwd_present;
	bool rev_present;
	bool isCorrected;
	sff_read_header_t *rh;
	sff_read_header_t *rh_rev;
	sff_read_t *rd;
	sff_read_t *rd_rev;
	string* fastqString;
	int merged_region_start_base;
	int merged_region_stop_base;
	int merged_region_start_base_fwd;
	int merged_region_stop_base_fwd;
	int merged_region_start_base_rev;
	int merged_region_stop_base_rev;
	
	sff_pair():fwd_present(0),rev_present(0),isCorrected(0), rh(NULL),rh_rev(NULL),rd(NULL),rd_rev(NULL), fastqString(NULL),
			merged_region_start_base(0), merged_region_stop_base(0),
			merged_region_start_base_fwd(0), merged_region_stop_base_fwd(0),
			merged_region_start_base_rev(0), merged_region_stop_base_rev(0) {
	}
};

struct sff_pairs_thread {
	int threadID;
	int maxRecords;
	sff_pair** sff_pair_array;
	sff_header_t* h;
    sff_header_t* h_rev;
	sff_pairs_thread():threadID(0),maxRecords(0),sff_pair_array(NULL),h(NULL),h_rev(NULL) {
	}
};

bool dump_region_info = false;
	

/* M A I N *******************************************************************/
int main(int argc, char *argv[]) {
    std::cout << "PROGRAM : PairedEnd ErrorCorrection BAM " << std::endl;
	std::cout << "Version : " << IonVersion::GetVersion() << " (r" << IonVersion::GetGitHash() << ")" << std::endl;
	std::cout << "Author  : Sowmi Utiramerur " << std::endl;
    process_options(argc, argv);
    if(has_strand_bias_info_fwd)
        has_strand_bias_info_fwd = build_strand_bias_map(input_strand_bias_file_fwd, &strand_bias_flow_index_map_fwd);
    if(has_strand_bias_info_rev)
        has_strand_bias_info_rev = build_strand_bias_map(input_strand_bias_file_rev, &strand_bias_flow_index_map_rev);
    process_sff(sff_file_fwd,sff_file_rev, output_fastq_file, output_sff_file_corrected, trim_flag);

    return 0;
}

/* F U N C T I O N S *********************************************************/
void
help_message() {
    fprintf(stdout, "Usage: %s %s %s %s\n", PRG_NAME, "[options]", "<Fwd bam_file>", "<Reverse bam_file>");
    fprintf(stdout, "\t%-20s%-20s\n", "-h", "This help message");
    fprintf(stdout, "\t%-20s%-20s\n", "-v", "Program and version information");
    fprintf(stdout, "\t%-20s%-20s %s %s\n",
                    " -s <corrected bam file name>",
					" -n <number of threads>",
					" -r 'dump merge region information' "
					" -a 'output only reads that are identical in Fwd and Reverse tags - Increased accuracy, lower throughput' "
                    " -b <system bias file name from forward run> "
                    " -c <system bias file name from reverse run> "
                    " Input bam file from Fwd sequenceing",
                    " Input bam file from Reverse sequencing");
}

void
version_info() {
    fprintf (stdout, "%s", IonVersion::GetFullVersion ("PairedEndErrorCorrection").c_str());
}


void getBaseName(char *path, char *basename) 
{
	std::string strPath((const char*)path);
	 int pos = strPath.rfind(".");
	  if (pos != (int)std::string::npos) {
		strcpy(basename, (strPath.substr(0, pos)).c_str());
	  }
	  else {
		strcpy(basename, path);
		
	  }
	
}

void
process_options(int argc, char *argv[]) {
    int c;
    int index;
   
	char *opt_so_value = NULL;    
	char *opt_num_threads = NULL;
    char *opt_bias_file_fwd = NULL;
    char *opt_bias_file_rev= NULL;

    while( (c = getopt(argc, argv, "hvarn:s:b:c:")) != -1 ) {
		
        switch(c) {
            case 'h':
                help_message();
                exit(0);
                break;
            case 'v':
                version_info();
                exit(0);
                break;
         	case 's':
                opt_so_value = optarg;
                break;
            case 'n':
                opt_num_threads = optarg;   
                break;
            case '?':
                exit(1);
            case 'r':
				dump_region_info = true;
				break;   
			case 'a':
				outputIdenticalOnly = true;
				break;
            case 'b':
                opt_bias_file_fwd = optarg;
                break;
            case 'c':
                opt_bias_file_rev = optarg;
                break;
            default:
                abort();
        }
    }
    
	if ( opt_so_value != NULL ) {
        strncpy(output_sff_file, opt_so_value, SFF_FILENAME_MAX_LENGTH-1);
        output_sff_file[SFF_FILENAME_MAX_LENGTH-1]='\0';		
    }
	
    if ( opt_bias_file_fwd != NULL ) {
        strncpy(input_strand_bias_file_fwd, opt_bias_file_fwd, SFF_FILENAME_MAX_LENGTH-1);
        input_strand_bias_file_fwd[SFF_FILENAME_MAX_LENGTH-1]='\0';
        has_strand_bias_info_fwd = true;
    }

    if ( opt_bias_file_rev != NULL ) {
        strncpy(input_strand_bias_file_rev, opt_bias_file_rev, SFF_FILENAME_MAX_LENGTH-1);
        input_strand_bias_file_rev[SFF_FILENAME_MAX_LENGTH-1]='\0';
        has_strand_bias_info_rev = true;
    }

	if (opt_num_threads != NULL) {
		max_threads = atoi(opt_num_threads);
	}
	
	//get the basename of output file and concatenate the names of singleton files
	
	char basename[SFF_FILENAME_MAX_LENGTH] = { '\0' };
  	getBaseName(output_sff_file, basename);
	strcpy(output_fastq_file,basename);
	strcat(output_fastq_file, "_corrected.fastq");
	strcpy(output_sff_file_corrected,basename);
	strcat(output_sff_file_corrected, "_corrected.bam");
	strcpy(output_sff_file_singleton_fwd, basename);
	strcat(output_sff_file_singleton_fwd, "_Singleton_Fwd.bam");
	strcpy(output_sff_file_singleton_rev, basename);
	strcat(output_sff_file_singleton_rev, "_Singleton_Rev.bam");
	strcpy(output_sff_file_paired_fwd, basename);
	strcat(output_sff_file_paired_fwd, "_Paired_Fwd.bam");
	strcpy(output_sff_file_paired_rev, basename);
	strcat(output_sff_file_paired_rev, "_Paired_Rev.bam");
	strcpy(output_statistics_file, basename);
	strcat(output_statistics_file, "_statistics_info.txt");
	
	if( dump_region_info){
		strcpy(output_merged_region_file, basename);
		strcat(output_merged_region_file, "_region_info.txt");
		strcpy(output_merged_region_file_fwd, basename);
		strcat(output_merged_region_file_fwd, "_region_info_fwd.txt");	
		strcpy(output_merged_region_file_rev, basename);
		strcat(output_merged_region_file_rev, "_region_info_rev.txt");			
	}
	
	//std::cout << " Opt o value " << opt_o_value << std::endl;
	//std::cout << " Opt so value " << opt_so_value << std::endl;
	//std::cout << " output_sff_file " << output_sff_file << std::endl;
    /* process the remaining command line arguments */
    index = optind;
	
	
    if (argc - index < 2) {
	
        fprintf(stderr, "[ERROR] Need to specify at least two BAM files as input (Fwd and Reverse runs) \n");
		help_message();
		exit(1);
    }
    
    strncpy(sff_file_fwd, argv[index++], SFF_FILENAME_MAX_LENGTH);
    strncpy(sff_file_rev, argv[index++], SFF_FILENAME_MAX_LENGTH);
    

//    /* just take the first passed in non-getopt argument as the sff file */
//    strncpy(sff_file, argv[optind], SFF_FILENAME_MAX_LENGTH);

    /* ensure that a sff file was at least passed in! */
    if ( !strlen(sff_file_fwd) || !strlen(sff_file_rev) ) {
        fprintf(stderr, "%s %s '%s %s' %s\n",
                "[err] Need to specify two BAM files!",
                "See", PRG_NAME, "-h", "for usage!");
        exit(1);
    }
}

void *process_sff_pairs(void *ptr)
{
	int counter = 0;
	int localNumCorrected = 0;
	int localNumUnCorrected = 0;
	int currentThreadID = 0;
	int maxRecordsInThread = 0;
	
	sff_pairs_thread *sff_pairs_thread_ptr = (sff_pairs_thread *) ptr;
	sff_pair **sff_pairs = sff_pairs_thread_ptr->sff_pair_array;
	currentThreadID = sff_pairs_thread_ptr->threadID;
	maxRecordsInThread = sff_pairs_thread_ptr->maxRecords;
	
	sff_pair* current_pair;
	sff_read_header_t *rh = NULL;
	sff_read_header_t *rh_rev = NULL;
	sff_read_t *rd = NULL;
	sff_read_t *rd_rev = NULL;
	sff_header_t *h = sff_pairs_thread_ptr->h;
	sff_header_t *h_rev = sff_pairs_thread_ptr->h_rev;
	
	int left_clip = 0, right_clip = 0, left_clip_orig = 0, right_clip_orig = 0, nbases = 0;
    int left_clip_rev = 0, right_clip_rev = 0, nbases_rev = 0;
    char *name = '\0', *name_rev = '\0';
    ion_string_t *bases = '\0', *bases_rev = '\0';
    ion_string_t *quality = '\0', *quality_rev = '\0';
    uint8_t *flow_index = '\0', *flow_index_rev ='\0';
    uint16_t *flow = '\0', *flow_rev = '\0';
    int *flow_index_sum = '\0', *flow_index_rev_sum = '\0';
	
	Alignment *alignment;
	SmithWaterman *aligner = new SmithWaterman();
	
	if (DEBUG)
		fprintf(stdout, "Processing ThreadID = %d maxRecords in Thread = %d \n", currentThreadID, maxRecordsInThread);
	
	while(counter < maxRecordsInThread) {
		current_pair = sff_pairs[counter++];
		if (current_pair == NULL)
		{
			fprintf(stdout, "Sff_Pairs returned null for Counter = %d \n", counter-1);
			exit(-1);
		}
		if (current_pair->fwd_present) {
			rh = current_pair->rh;
			rd = current_pair->rd;
		}
		if (current_pair->rev_present) {
			rh_rev = current_pair->rh_rev;
			rd_rev = current_pair->rd_rev;
		}
		
		//error correct only if both fwd and reverse bead present
		if (current_pair->fwd_present && current_pair->rev_present) {
		
			sff_read_header_get_clip_values(rh, trim_flag, &left_clip, &right_clip);
			left_clip_orig = left_clip;
			right_clip_orig = right_clip;
			//left_clip = 0;
			right_clip = rh->n_bases;
			nbases = right_clip - left_clip;
			
			/* create bases string */
			bases = (sff_read_get_read_bases(rd, left_clip, right_clip));
			/* create quality array */
			quality = sff_read_get_read_quality_values(rd, left_clip, right_clip);
			
			/* get flowspace info */
			flow = rd->flowgram;
			flow_index = rd->flow_index;
		
			/* create read name string */
			int name_length = (int) rh->name_length + 1; // account for NULL termination
			name = (char *) malloc( name_length * sizeof(char) );
			if (!name) {
					fprintf(stderr, "Out of memory! For read name string!\n");
					pthread_exit((void *)-1);
			}

			memset(name, '\0', (size_t) name_length);
			
			strncpy(name, rh->name->s, (size_t) rh->name_length);
			
			// now get the revese read bases and quality values
			sff_read_header_get_clip_values(rh_rev, trim_flag, &left_clip_rev, &right_clip_rev);
			//left_clip_rev = 0;
			right_clip_rev = rh_rev->n_bases;
			nbases_rev = right_clip_rev - left_clip_rev;
			bases_rev = (sff_read_get_read_bases(rd_rev, left_clip_rev, right_clip_rev));
			quality_rev = sff_read_get_read_quality_values(rd_rev, left_clip_rev, right_clip_rev);
			flow_rev = rd_rev->flowgram;
			flow_index_rev = rd_rev->flow_index;

			name_length = (int) rh_rev->name_length + 1; // account for NULL termination
			name_rev = (char*) malloc( name_length * sizeof(char) );
			if (!name_rev) {
				fprintf(stderr, "Out of memory! For read name string!\n");
				pthread_exit((void *)-1);
			}
			memset(name_rev, '\0', (size_t) name_length);
		
			strncpy(name_rev, rh_rev->name->s, (size_t) rh_rev->name_length);
			
			std::string str1 ((char*)bases->s); 
			std::string str2 ((char*)bases_rev->s); 
			std::string revComplement_str2 = reverseComplement(str2); //reverse complement the reverse read before aligning to fwd read			
			
			flow_index_sum = sumFlowIndex(flow_index, nbases, left_clip); 
			flow_index_rev_sum = reverseFlowIndex(flow_index_rev, nbases_rev, left_clip_rev);			
			
			alignment = aligner->align(str1, revComplement_str2, 2, 0.5, -1); //smith waterman alignment of fwd and rev bases
			alignment->setName1(name);
			alignment->setName2(name_rev);
			
			if ( alignment->getSequence1().length() >= 50 && ((float)alignment->getIdentity())/alignment->getSequence1().length() >= 0.85) {
				uint16_t *corr_flow;
				uint8_t *corr_flow_index ;
				uint8_t *corr_quality ;
				char *corr_bases;
				char *corr_fastq_bases;
				int totalBases = 0;
				int mergedRegionStartBase = 0;
				int mergedRegionStopBase = 0;
				int mergedRegionStartBaseFwd = 0;
				int mergedRegionStopBaseFwd = 0;				
				int mergedRegionStartBaseRev = 0;
				int mergedRegionStopBaseRev = 0;
				int newLeftClip = 0;
				int newRightClip = 0;
				bool isIdentical = false;
				
				totalBases = errorCorrect(alignment, flow_index_sum, flow_index_rev_sum, flow, flow_rev, h->flow->s, h->flow_length, h_rev->flow->s, h_rev->flow_length,
												quality, quality_rev, &corr_flow, &corr_flow_index, &corr_quality, &corr_bases, &corr_fastq_bases, &newLeftClip, &newRightClip, &isIdentical, nbases_rev,
												&mergedRegionStartBase, &mergedRegionStopBase, &mergedRegionStartBaseFwd, &mergedRegionStopBaseFwd, &mergedRegionStartBaseRev, &mergedRegionStopBaseRev, dump_region_info);	
			
				if (isIdentical && outputIdenticalOnly) {
					localNumCorrected++;
					current_pair->isCorrected = 1;	
					rh->clip_qual_right = newRightClip;
					if (newLeftClip > left_clip_orig) {
						rh->clip_qual_left = newLeftClip;
					}
					
				}
				else if (totalBases)  // if the totalBases returned by errorCorrect function is 0 then it means we were unable to correct to misaglignment of flows and hence the record should go to singleton file
				{	
					//Comment: originally there was no rd_coorected
					sff_read_t *rd_corrected = sff_read_init(); 
					rd_corrected->flowgram = corr_flow;
					rd_corrected->flow_index = corr_flow_index;
					rd_corrected->bases = ion_string_init(totalBases+1);
					ion_string_copy1(rd_corrected->bases, corr_bases);			
					rd_corrected->quality = ion_string_init(totalBases+1);
					for(int i=0; i<totalBases; i++){
						rd_corrected->quality->s[i] = corr_quality[i];			
					}
                    //rh->n_bases = totalBases;
					

					if (newRightClip > 0)
					{
						rh->clip_qual_right = newRightClip;
                        rh->clip_adapter_right = newRightClip;
						
					}
					else if (right_clip_orig > totalBases) {
                        rh->clip_qual_right = totalBases;
                        rh->clip_adapter_right = totalBases;
                    }
                    rh->n_bases = rh->clip_qual_right + 1;

/*
                    rh->clip_adapter_right = totalBases + 1;
                    rh->clip_qual_right = totalBases + 1;
*/

					localNumCorrected++;
					//release original read data rd as its no longer needed
					sff_read_destroy(rd);
					current_pair->rd = rd_corrected;
					current_pair->isCorrected = 1;		
					
					
					if(dump_region_info){					
						
						current_pair->merged_region_start_base = mergedRegionStartBase - left_clip;
						current_pair->merged_region_stop_base = mergedRegionStopBase - left_clip;
						current_pair->merged_region_start_base_fwd = mergedRegionStartBaseFwd - left_clip;
						current_pair->merged_region_stop_base_fwd = mergedRegionStopBaseFwd - left_clip;		
						//revert positions for reverse read																					
						current_pair->merged_region_start_base_rev = mergedRegionStartBaseRev - left_clip_rev;
						current_pair->merged_region_stop_base_rev = mergedRegionStopBaseRev - left_clip_rev;						
						
					}					
					
					//fprintf(stdout,"Name = %s, Iscorrected = %d \n", current_pair->rh->name->s, current_pair->isCorrected);
					//construct bases and qv for merged Fwd+Rev read to be output only to fastq file
					/*
					if (alignment->getStart2() + alignment->getLengthOfSequence2() < (int)revComplement_str2.length()) {
						string revExtnString = revComplement_str2.substr(alignment->getStart2() + alignment->getLengthOfSequence2());
						string* mergedFastqBaseString = new string((const char *)corr_bases, rh->clip_qual_right);
						std::cout << "ERROR: right Clip = " << right_clip << " New right clip = " << newRightClip << " rh clip = " << rh->clip_qual_right << std::endl;
						std::cout << *mergedFastqBaseString << std::endl;
						*mergedFastqBaseString = *mergedFastqBaseString + revExtnString;
						current_pair->fastqString = mergedFastqBaseString;
						
							
							std::cout << revExtnString << std::endl;
							std::cout << *mergedFastqBaseString << std::endl;
							std::cout << revComplement_str2 << std::endl;
						
					}
					else 
						current_pair->fastqString = new string((const char *)corr_bases, rh->clip_qual_right);
					*/
					current_pair->fastqString = new string((const char *)corr_fastq_bases);
					if (DEBUG) {
					std::cout << "ERROR: right Clip = " << right_clip << " New right clip = " << newRightClip << " rh clip = " << rh->clip_qual_right << std::endl;
					std::cout << corr_bases << std::endl;
					std::cout << corr_fastq_bases << std::endl;
					std::cout << revComplement_str2 << std::endl;
					}
					
					free(corr_fastq_bases);
					free(corr_bases);
					free(corr_quality);
				}
				else {
					localNumUnCorrected++;
					current_pair->isCorrected = 0;	
				}
			
			}
			else {
				//output misaligned reads to singleton file
				localNumUnCorrected++;
				current_pair->isCorrected = 0;	
			
			}
			
			
			delete alignment;
			delete[] flow_index_sum;
			delete[] flow_index_rev_sum;
			
			free(name);
			name = NULL;
			ion_string_destroy(bases);
			ion_string_destroy(quality);
		
			free(name_rev);
			name_rev = NULL;
        	ion_string_destroy(bases_rev);                
        	ion_string_destroy(quality_rev);
			
		
		}
		
	}//finished processsing all the record pairs
	
	//fprintf(stdout, "localNumCorrected = %d \n", localNumCorrected);
	delete aligner;
	
	//now check if this thread is ready to write to SFF files, if so write and exit, else sleep
	while (total_threads_sem.count < (currentThreadID-1)) {
			pthread_cond_wait(&(total_threads_sem.cond), &(total_threads_sem.mutex));
	}
	if (total_threads_sem.count == (currentThreadID-1)) {
		//loop thru all the sff pairs and write to appropriate SFF output file
		counter = 0;
		if (DEBUG)
			fprintf(stdout, "Starting to write to SFF file from thead ID = %d, total threads sem count = %d, max threads sem count = %d \n", currentThreadID, total_threads_sem.count, max_threads_sem.count);
		
		while(counter < maxRecordsInThread) {
			current_pair = sff_pairs[counter++];
			
			if (current_pair->isCorrected) {
				//sff_read_header_write(out_sff_fp, current_pair->rh);
				//sff_read_write(out_sff_fp, h, current_pair->rh, current_pair->rd);
				
				//JZ begins
				BamAlignment bam_alignmentc;
				bam_alignmentc.SetIsMapped(false);
				bam_alignmentc.Name = current_pair->rh->name->s;
				size_t nBases = current_pair->rh->n_bases + 1 - current_pair->rh->clip_qual_left;
				if(current_pair->rh->clip_qual_right > 0)
				{
					nBases = current_pair->rh->clip_qual_right - current_pair->rh->clip_qual_left;
				}
				if(nBases > 0)
				{
					bam_alignmentc.QueryBases.reserve(nBases);
					bam_alignmentc.Qualities.reserve(nBases);
					for (int base = current_pair->rh->clip_qual_left - 1; base < current_pair->rh->clip_qual_right - 1; ++base)
					{
						bam_alignmentc.QueryBases.push_back(current_pair->rd->bases->s[base]);
						bam_alignmentc.Qualities.push_back(current_pair->rd->quality->s[base] + 33);
					}
				}

				int clip_flow = 0;
				for (unsigned int base = 0; base < current_pair->rh->clip_qual_left && base < current_pair->rh->n_bases; ++base)
				{
					clip_flow += current_pair->rd->flow_index[base];
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}

                bam_alignmentc.AddTag("RG","Z", rname);
				bam_alignmentc.AddTag("PG","Z", string("sff2bam"));
				bam_alignmentc.AddTag("ZF","i", clip_flow); // TODO: trim flow
				vector<uint16_t> flowgram0(h->flow_length);
				copy(current_pair->rd->flowgram, current_pair->rd->flowgram + h->flow_length, flowgram0.begin());
				bam_alignmentc.AddTag("FZ", flowgram0);

				bamWriter_c.SaveAlignment(bam_alignmentc);
				//JZ ends
				//std::cout << " Fastq entry for read " << current_pair->rh->name->s << std::endl;
				//std::cout << " String = " << *current_pair->fastqString << " Clip value = " <<  current_pair->rh->clip_qual_left << std::endl;
				output_fastq_entry(fastq_fp, current_pair->rh->name->s, *(current_pair->fastqString), current_pair->rh->clip_qual_left);
				if(dump_region_info){
					/*
					std::string read_id(current_pair->rh->name->s);
					read_id = read_id.substr(6);
					fprintf(region_merged_fp, "%s\t%d\t%d\n", read_id.c_str(), current_pair->merged_region_start_base, current_pair->merged_region_stop_base);
					fprintf(region_merged_fwd_fp, "%s\t%d\t%d\n", read_id.c_str(), current_pair->merged_region_start_base_fwd, current_pair->merged_region_stop_base_fwd);
					fprintf(region_merged_rev_fp, "%s\t%d\t%d\n", read_id.c_str(), current_pair->merged_region_start_base_rev, current_pair->merged_region_stop_base_rev);
					*/
					fprintf(region_merged_fp, "%s\t%d\t%d\n", current_pair->rh->name->s, current_pair->merged_region_start_base, current_pair->merged_region_stop_base);
					fprintf(region_merged_fwd_fp, "%s\t%d\t%d\n", current_pair->rh->name->s, current_pair->merged_region_start_base_fwd, current_pair->merged_region_stop_base_fwd);
					fprintf(region_merged_rev_fp, "%s\t%d\t%d\n", current_pair->rh_rev->name->s, current_pair->merged_region_start_base_rev, current_pair->merged_region_stop_base_rev);		
					
				} 
				
				numCorrected++;	
								
			}
			else if (current_pair->fwd_present && current_pair->rev_present) {
				//sff_read_header_write(out_sff_paired_fwd_fp, current_pair->rh);
				//sff_read_write(out_sff_paired_fwd_fp, h, current_pair->rh, current_pair->rd);
				//sff_read_header_write(out_sff_paired_rev_fp, current_pair->rh_rev);
				//sff_read_write(out_sff_paired_rev_fp, h_rev, current_pair->rh_rev, current_pair->rd_rev);

				//JZ begins
				BamAlignment bam_alignmentf;
				bam_alignmentf.SetIsMapped(false);
				bam_alignmentf.Name = current_pair->rh->name->s;
				size_t nBases = current_pair->rh->n_bases + 1 - current_pair->rh->clip_qual_left;
				if(current_pair->rh->clip_qual_right > 0)
				{
					nBases = current_pair->rh->clip_qual_right - current_pair->rh->clip_qual_left;
				}
				if(nBases > 0)
				{
					bam_alignmentf.QueryBases.reserve(nBases);
					bam_alignmentf.Qualities.reserve(nBases);
					for (int base = current_pair->rh->clip_qual_left - 1; base < current_pair->rh->clip_qual_right - 1; ++base)
					{
						bam_alignmentf.QueryBases.push_back(current_pair->rd->bases->s[base]);
						bam_alignmentf.Qualities.push_back(current_pair->rd->quality->s[base] + 33);
					}
				}

				int clip_flow = 0;
				for (unsigned int base = 0; base < current_pair->rh->clip_qual_left && base < current_pair->rh->n_bases; ++base)
				{
					clip_flow += current_pair->rd->flow_index[base];
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}

                bam_alignmentf.AddTag("RG","Z", rname);
				bam_alignmentf.AddTag("PG","Z", string("sff2bam"));
				bam_alignmentf.AddTag("ZF","i", clip_flow); // TODO: trim flow
				vector<uint16_t> flowgram0(h->flow_length);
				copy(current_pair->rd->flowgram, current_pair->rd->flowgram + h->flow_length, flowgram0.begin());
				bam_alignmentf.AddTag("FZ", flowgram0);

				bamWriter_f.SaveAlignment(bam_alignmentf);

				BamAlignment bam_alignmentr;
				bam_alignmentr.SetIsMapped(false);
				bam_alignmentr.Name = current_pair->rh_rev->name->s;
                nBases = current_pair->rh_rev->n_bases + 1 - current_pair->rh_rev->clip_qual_left;
				if(current_pair->rh_rev->clip_qual_right > 0)
				{
					nBases = current_pair->rh_rev->clip_qual_right - current_pair->rh_rev->clip_qual_left;
				}
				if(nBases > 0)
				{
					bam_alignmentr.QueryBases.reserve(nBases);
					bam_alignmentr.Qualities.reserve(nBases);
					for (int base = current_pair->rh_rev->clip_qual_left - 1; base < current_pair->rh_rev->clip_qual_right - 1; ++base)
					{
						bam_alignmentr.QueryBases.push_back(current_pair->rd_rev->bases->s[base]);
						bam_alignmentr.Qualities.push_back(current_pair->rd_rev->quality->s[base] + 33);
					}
				}

                 clip_flow = 0;
				for (unsigned int base = 0; base < current_pair->rh_rev->clip_qual_left && base < current_pair->rh_rev->n_bases; ++base)
				{
					clip_flow += current_pair->rd_rev->flow_index[base];
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}

                bam_alignmentr.AddTag("RG","Z", rname_rev);
				bam_alignmentr.AddTag("PG","Z", string("sff2bam"));
				bam_alignmentr.AddTag("ZF","i", clip_flow); // TODO: trim flow
                vector<uint16_t> flowgram1(h_rev->flow_length);
                copy(current_pair->rd_rev->flowgram, current_pair->rd_rev->flowgram + h_rev->flow_length, flowgram1.begin());
                bam_alignmentr.AddTag("FZ", flowgram1);

				bamWriter_r.SaveAlignment(bam_alignmentr);
				//JZ ends

				numUnCorrected++;
			}
			else if (current_pair->fwd_present && !current_pair->rev_present) {
				//sff_read_header_write(out_sff_fwd_fp, current_pair->rh);
				//sff_read_write(out_sff_fwd_fp, h, current_pair->rh, current_pair->rd);

				//JZ begins
				BamAlignment bam_alignmentf;
				bam_alignmentf.SetIsMapped(false);
				bam_alignmentf.Name = current_pair->rh->name->s;
				size_t nBases = current_pair->rh->n_bases + 1 - current_pair->rh->clip_qual_left;
				if(current_pair->rh->clip_qual_right > 0)
				{
					nBases = current_pair->rh->clip_qual_right - current_pair->rh->clip_qual_left;
				}
				if(nBases > 0)
				{
					bam_alignmentf.QueryBases.reserve(nBases);
					bam_alignmentf.Qualities.reserve(nBases);
					for (int base = current_pair->rh->clip_qual_left - 1; base < current_pair->rh->clip_qual_right - 1; ++base)
					{
						bam_alignmentf.QueryBases.push_back(current_pair->rd->bases->s[base]);
						bam_alignmentf.Qualities.push_back(current_pair->rd->quality->s[base] + 33);
					}
				}

				int clip_flow = 0;
				for (unsigned int base = 0; base < current_pair->rh->clip_qual_left && base < current_pair->rh->n_bases; ++base)
				{
					clip_flow += current_pair->rd->flow_index[base];
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}

                bam_alignmentf.AddTag("RG","Z", rname);
				bam_alignmentf.AddTag("PG","Z", string("sff2bam"));
				bam_alignmentf.AddTag("ZF","i", clip_flow); // TODO: trim flow
				vector<uint16_t> flowgram0(h->flow_length);
				copy(current_pair->rd->flowgram, current_pair->rd->flowgram + h->flow_length, flowgram0.begin());
				bam_alignmentf.AddTag("FZ", flowgram0);

				bamWriter_fo.SaveAlignment(bam_alignmentf);
				//JZ ends
			}
			else if (current_pair->rev_present && !current_pair->fwd_present) {
				//sff_read_header_write(out_sff_rev_fp, current_pair->rh_rev);
				//sff_read_write(out_sff_rev_fp, h_rev, current_pair->rh_rev, current_pair->rd_rev);

				//JZ begins
				BamAlignment bam_alignmentr;
				bam_alignmentr.SetIsMapped(false);
				bam_alignmentr.Name = current_pair->rh_rev->name->s;
				size_t nBases = current_pair->rh_rev->n_bases + 1 - current_pair->rh_rev->clip_qual_left;
				if(current_pair->rh_rev->clip_qual_right > 0)
				{
					nBases = current_pair->rh_rev->clip_qual_right - current_pair->rh_rev->clip_qual_left;
				}
				if(nBases > 0)
				{
					bam_alignmentr.QueryBases.reserve(nBases);
					bam_alignmentr.Qualities.reserve(nBases);
					for (int base = current_pair->rh_rev->clip_qual_left - 1; base < current_pair->rh_rev->clip_qual_right - 1; ++base)
					{
						bam_alignmentr.QueryBases.push_back(current_pair->rd_rev->bases->s[base]);
						bam_alignmentr.Qualities.push_back(current_pair->rd_rev->quality->s[base] + 33);
					}
				}

				int clip_flow = 0;
				for (unsigned int base = 0; base < current_pair->rh_rev->clip_qual_left && base < current_pair->rh_rev->n_bases; ++base)
				{
					clip_flow += current_pair->rd_rev->flow_index[base];
				}
				if (clip_flow > 0)
				{
					clip_flow--;
				}

                bam_alignmentr.AddTag("RG","Z", rname_rev);
				bam_alignmentr.AddTag("PG","Z", string("sff2bam"));
				bam_alignmentr.AddTag("ZF","i", clip_flow); // TODO: trim flow
				vector<uint16_t> flowgram0(h_rev->flow_length);
				copy(current_pair->rd_rev->flowgram, current_pair->rd_rev->flowgram + h_rev->flow_length, flowgram0.begin());
				bam_alignmentr.AddTag("FZ", flowgram0);

				bamWriter_ro.SaveAlignment(bam_alignmentr);
				//JZ ends
			}
			
			//release memory 
			if (current_pair->fwd_present) {
		
				sff_read_header_destroy(current_pair->rh);
				sff_read_destroy(current_pair->rd);
			}


			if (current_pair->rev_present) {
		
				sff_read_header_destroy(current_pair->rh_rev);
				sff_read_destroy(current_pair->rd_rev);
			}
			delete current_pair->fastqString;
			//fprintf(stdout, "finished writing record number %d \n", counter);
			
		}
		
		//update the global counters now that the current thread has the lock on the mutex
		//numCorrected += localNumCorrected;
		//numUnCorrected += localNumUnCorrected;
		for (int i = 0; i < maxRecordsInThread; i++)
			delete sff_pairs[i];
			
		delete[] sff_pairs;
		delete sff_pairs_thread_ptr;
		
		//up the total_theards_semaphore to current thread id, so the next thread can start writing to SFF file
		if (DEBUG)
			fprintf(stdout, "trying up total_threads_sem \n");
		up(&total_threads_sem);
		
		if (DEBUG)
			fprintf(stdout, "finished up total_threads_sem \n");
			
		down(&max_threads_sem);
		
		if (DEBUG)
			fprintf(stdout, "Finished write to SFF file from thead ID = %d, tot threads sem count = %d , max threads sem count = %d\n", currentThreadID, total_threads_sem.count, max_threads_sem.count);
	}
	
	pthread_exit((void *)0);
	
}



void
process_sff(char *sff_file, char *sff_file_rev, char *fastq_file, char *out_sff_file, int trim_flag) {

    sff_header_t* h;
    sff_header_t* h_rev;
    sff_read_header_t* rh;
    sff_read_header_t *rh_rev ;
    //sff_read_header_t *test_rh_rev;
    sff_read_t* rd ;
    sff_read_t* rd_rev ;

    int both_fwd_rev_present = 0;
    int fwd_only_present = 0;
    int rev_only_present = 0;
   
	
	//thread pointer
	pthread_t *thread;
	std::vector<pthread_t*> thread_vector;
	
	init_semaphore(&max_threads_sem, 0);
	init_semaphore(&total_threads_sem, 0);
	
	if (DEBUG) {
		fprintf(stdout, "fwd file = %s \n", sff_file);
		fprintf(stdout, "rev file = %s \n", sff_file_rev);
	}
/*    if ( (sff_fp = fopen(sff_file, "r")) == NULL ) {
        fprintf(stderr,
                "[ERROR] Could not open file '%s' for reading.\n", sff_file);
        exit(1);
    }

    if ( (sff_fp_rev = fopen(sff_file_rev, "r")) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not open file '%s' for reading.\n", sff_file_rev);
        exit(1);
    }
	
	if ( (out_sff_fp = fopen(output_sff_file_corrected, "wb")) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not open SFF file '%s' for writing.\n", output_sff_file_corrected);
        exit(1);
    }

	if ( (out_sff_fwd_fp = fopen(output_sff_file_singleton_fwd, "wb")) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not open SFF file '%s' for writing.\n", out_sff_file);
        exit(1);
    }
	
	if ( (out_sff_rev_fp = fopen(output_sff_file_singleton_rev, "wb")) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not open SFF file '%s' for writing.\n", out_sff_file);
        exit(1);
    }
	if ( (out_sff_paired_fwd_fp = fopen(output_sff_file_paired_fwd, "wb")) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not open SFF file '%s' for writing.\n", out_sff_file);
        exit(1);
    }
	if ( (out_sff_paired_rev_fp = fopen(output_sff_file_paired_rev, "wb")) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not open SFF file '%s' for writing.\n", out_sff_file);
        exit(1);
    }
*/
	if ( (output_stats_fp = fopen(output_statistics_file, "w" )) == NULL) {
		fprintf(stderr,
                "[ERROR] Could not output Statistics file '%s' for writing.\n", output_statistics_file);
        exit(1);
    }	
	if( dump_region_info){
		if ( (region_merged_fp = fopen(output_merged_region_file, "w")) == NULL) {
			fprintf(stderr,
	                "[ERROR] Could not open merged region info file '%s' for writing.\n", out_sff_file);
	        exit(1);
	    }
		if ( (region_merged_fwd_fp = fopen(output_merged_region_file_fwd, "w")) == NULL) {
			fprintf(stderr,
	                "[ERROR] Could not open forward run merged region info file '%s' for writing.\n", out_sff_file);
	        exit(1);
	    }
		if ( (region_merged_rev_fp = fopen(output_merged_region_file_rev, "w")) == NULL) {
			fprintf(stderr,
	                "[ERROR] Could not open reverse run merged region info file '%s' for writing.\n", out_sff_file);
	        exit(1);
	    }		
	}
	
//	h = sff_header_read(sff_fp);
//	h_rev = sff_header_read(sff_fp_rev);

	//JZ begins
    BamReader bamReader;
    if (!bamReader.Open(sff_file))
    {
		fprintf(stderr,
                "[ERROR] Could not open file '%s' for reading.\n", sff_file);
        exit(1);
    }
    BamReader bamReader_rev;
    if (!bamReader_rev.Open(sff_file_rev))
    {
		bamReader.Close();
		fprintf(stderr,
                "[ERROR] Could not open file '%s' for reading.\n", sff_file_rev);
        exit(1);
    }

	SamHeader samHeader = bamReader.GetHeader();
    if(!samHeader.HasReadGroups())
    {
        bamReader.Close();
		bamReader_rev.Close();
        fprintf(stderr,
                "[ERROR] there is no read group in file '%s'.\n", sff_file);
        exit(1);
    }

    string flow_order;
    string key;
    for (SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr )
    {
        if(itr->HasFlowOrder())
        {
            flow_order = itr->FlowOrder;
        }
        if(itr->HasKeySequence())
        {
            key = itr->KeySequence;
        }
    }
	    
    uint16_t nFlows = flow_order.length();
    uint16_t nKeys = key.length();
    if(!nFlows || nKeys < 1)
    {
        bamReader.Close();
		bamReader_rev.Close();
        fprintf(stderr,
                "[ERROR] there is no flow order or key in file '%s'.\n", sff_file);
        exit(1);
    }

    h = sff_header_init1(0, nFlows, flow_order.c_str(), key.c_str());

	SamHeader samHeader_rev = bamReader_rev.GetHeader();
    if(!samHeader_rev.HasReadGroups())
    {
        bamReader.Close();
		bamReader_rev.Close();
        fprintf(stderr,
                "[ERROR] there is no read group in file '%s'.\n", sff_file_rev);
        exit(1);
    }

    string flow_order_rev;
    string key_rev;
    for (SamReadGroupIterator itr = samHeader_rev.ReadGroups.Begin(); itr != samHeader_rev.ReadGroups.End(); ++itr )
    {
        if(itr->HasFlowOrder())
        {
            flow_order_rev = itr->FlowOrder;
        }
        if(itr->HasKeySequence())
        {
            key_rev = itr->KeySequence;
        }
    }
	    
    uint16_t nFlows_rev = flow_order_rev.length();
    uint16_t nKeys_rev = key_rev.length();
    if(!nFlows_rev || nKeys_rev < 1)
    {
        bamReader.Close();
		bamReader_rev.Close();
        fprintf(stderr,
                "[ERROR] there is no flow order or key in file '%s'.\n", sff_file_rev);
        exit(1);
    }

    h_rev = sff_header_init1(0, nFlows_rev, flow_order_rev.c_str(), key_rev.c_str());

	RefVector refvec;
	bamWriter_c.SetCompressionMode(BamWriter::Compressed);
    if(!bamWriter_c.Open(output_sff_file_corrected, samHeader, refvec))
    {
        bamReader.Close();
		bamReader_rev.Close();
        fprintf(stderr,
                "[ERROR] Could not open file '%s' for writing.\n", output_sff_file_corrected);
        exit(1);
    }
	bamWriter_f.SetCompressionMode(BamWriter::Compressed);
    if(!bamWriter_f.Open(output_sff_file_paired_fwd, samHeader, refvec))
    {
        bamReader.Close();
		bamReader_rev.Close();
		bamWriter_c.Close();
        fprintf(stderr,
                "[ERROR] Could not open file '%s' for writing.\n", output_sff_file_paired_fwd);
        exit(1);
    }
	bamWriter_fo.SetCompressionMode(BamWriter::Compressed);
    if(!bamWriter_fo.Open(output_sff_file_singleton_fwd, samHeader, refvec))
    {
        bamReader.Close();
		bamReader_rev.Close();
		bamWriter_c.Close();
		bamWriter_f.Close();
        fprintf(stderr,
                "[ERROR] Could not open file '%s' for writing.\n", output_sff_file_singleton_fwd);
        exit(1);
    }
	bamWriter_r.SetCompressionMode(BamWriter::Compressed);
    if(!bamWriter_r.Open(output_sff_file_paired_rev, samHeader_rev, refvec))
    {
        bamReader.Close();
		bamReader_rev.Close();
		bamWriter_c.Close();
		bamWriter_f.Close();
		bamWriter_fo.Close();
        fprintf(stderr,
                "[ERROR] Could not open file '%s' for writing.\n", output_sff_file_paired_rev);
        exit(1);
    }
	bamWriter_ro.SetCompressionMode(BamWriter::Compressed);
    if(!bamWriter_ro.Open(output_sff_file_singleton_rev, samHeader_rev, refvec))
    {
        bamReader.Close();
		bamReader_rev.Close();
		bamWriter_c.Close();
		bamWriter_f.Close();
		bamWriter_fo.Close();
		bamWriter_r.Close();
        fprintf(stderr,
                "[ERROR] Could not open file '%s' for writing.\n", output_sff_file_singleton_rev);
        exit(1);
    }
	//JZ ends
	
    //read_sff_common_header(sff_fp, &h);
    //read_sff_common_header(sff_fp_rev, &h_rev);

    //verify_sff_common_header(PRG_NAME, VERSION, &h);	
	
	if (DEBUG) {
    printf("size of header: %d \n", (int)sizeof(sff_header_t));

    printf("\tmagic        : 0x%x\n" , h->magic);
    printf("\tindex_offset : 0x%lx\n", (long )h->index_offset);
    printf("\tindex_len    : 0x%x\n" , h->index_length);
    printf("\tnumreads     : 0x%x\n" , h->n_reads);
    printf("\theader_len   : 0x%x\n" , h->gheader_length);
    printf("\tkey_len      : 0x%x\n" , h->key_length);
    printf("\tflow_len     : 0x%x\n" , h->flow_length);
    printf("\tflowgram_fmt : 0x%x\n" , h->flowgram_format);
    printf("\tflow         : %s\n  " , h->flow->s);
    printf("\tkey          : %s\n  " , h->key->s);
    printf("\n\n");


    printf("size of reverse header: %d \n", (int)sizeof(sff_header_t));
    printf("\tmagic        : 0x%x\n" , h_rev->magic);
    printf("\tindex_offset : 0x%lx\n", (long )h_rev->index_offset);
    printf("\tindex_length    : 0x%x\n" , h_rev->index_length);
    printf("\tnumreads     : 0x%x\n" , h_rev->n_reads);
    printf("\theader_len   : 0x%x\n" , h_rev->gheader_length);
    printf("\tkey_len      : 0x%x\n" , h_rev->key_length);
    printf("\tflow_len     : 0x%x\n" , h_rev->flow_length);
    printf("\tflowgram_fmt : 0x%x\n" , h_rev->flowgram_format);
    printf("\tflow         : %s\n  " , h_rev->flow->s);
    printf("\tkey          : %s\n  " , h_rev->key->s);
    printf("\n\n");
	
	}
	
	totalFlowLengthFwd = h->flow_length;
	totalFlowLengthRev = h_rev->flow_length;
	
	//write sff common header first to output SFF file
/*	sff_header_write(out_sff_fp, h);
	sff_header_write(out_sff_fwd_fp, h);
	sff_header_write(out_sff_paired_fwd_fp, h);
	sff_header_write(out_sff_paired_rev_fp, h_rev);
	sff_header_write(out_sff_rev_fp, h_rev);
	fseekOffset = sizeof(h->magic) + sizeof(h->index_offset) + sizeof(h->index_length); //seek to this location in file to set the nreads value in header at the end of processing
*/	
    if ( !strlen(fastq_file) ) {
        fastq_fp = stdout;
    }
    else {
        if ( (fastq_fp = fopen(fastq_file, "w")) == NULL ) {
            fprintf(stderr,
                    "[err] Could not open file '%s' for writing.\n",
                    fastq_file);
            exit(1);
        }
    }

	int readsPerThreadCounter = 0;

    int totalReads = 0;
    bool fwd_present = 0;
    bool rev_present = 0;
	int thread_ret_value = 0;   
   
	sff_pairs_thread *sffPairArray = NULL;

    //while (!getNextSFFPair(totalReads, &fwd_present, &rev_present, sff_fp, sff_fp_rev, &rh, &rh_rev, &rd, &rd_rev, h, h_rev) )  {
    while (!getNextSFFPair(totalReads, &fwd_present, &rev_present, &bamReader, &bamReader_rev, &rh, &rh_rev, &rd, &rd_rev, h, h_rev) )  {
		totalReads++;
		sff_pair *readPair = new sff_pair();
		readPair->fwd_present = fwd_present;
		readPair->rev_present = rev_present;
		readPair->rh = rh;
		readPair->rh_rev = rh_rev;
		readPair->rd = rd;
		readPair->rd_rev = rd_rev;
		
		if (fwd_present && rev_present)
			both_fwd_rev_present++;
		else if (fwd_present && !rev_present)
			fwd_only_present++;
		else if (rev_present && !fwd_present)
			rev_only_present++;
		
		if (sffPairArray == NULL) {
			threadCounter++;
			sffPairArray = new sff_pairs_thread();
			sffPairArray->sff_pair_array = new sff_pair*[max_pairs_per_thread];
			sffPairArray->threadID = threadCounter;
			sffPairArray->h = h;
			sffPairArray->h_rev = h_rev;
			readsPerThreadCounter = 0;
		}
		
		if (readsPerThreadCounter < max_pairs_per_thread) {
			sffPairArray->sff_pair_array[readsPerThreadCounter++] =  readPair;
		}
		else {
			
			//check for the number of running threads and if less than MAX create new thread and pass the array of sff pairs
			//TO-DO
			//if not sleep for 5 secs or until the num threads becomes less than MAX THREADS
			while (max_threads_sem.count >= max_threads) {
				pthread_cond_wait(&(max_threads_sem.cond), &(max_threads_sem.mutex));
			}
			
			if (max_threads_sem.count < max_threads) {
				if (DEBUG)
					fprintf(stdout, "Starting Thread ID = %d, readPairs count = %d, max threads sem count = %d \n", threadCounter, readsPerThreadCounter, max_threads_sem.count);
				thread = new pthread_t();
				sffPairArray->maxRecords = readsPerThreadCounter;
				thread_ret_value = pthread_create(thread, NULL, &process_sff_pairs, (void *)sffPairArray);
				if (thread_ret_value) {
					fprintf(stderr, "Error: Unable to create thread - Return Value = %d \n", thread_ret_value);
					exit(-1);
				}
				//pthread join
				thread_vector.push_back(thread);
				if (DEBUG)
					fprintf(stdout, "Starting Thread ID = %d, trying to up max_threads_sem \n", threadCounter);
				up(&max_threads_sem);
				//pthread_join(*thread, NULL);
				if (DEBUG)
					fprintf(stdout, "Finished starting Thread ID = %d, readPairs count = %d, max threads sem count = %d \n", threadCounter, readsPerThreadCounter, max_threads_sem.count);
				
				sffPairArray = NULL;
				threadCounter++;
				sffPairArray = new sff_pairs_thread();
				sffPairArray->threadID = threadCounter;
				
				sffPairArray->sff_pair_array = new sff_pair*[max_pairs_per_thread];
				sffPairArray->h = h;
				sffPairArray->h_rev = h_rev;
				readsPerThreadCounter = 0;
				sffPairArray->sff_pair_array[readsPerThreadCounter++] =  readPair;
				
			}
			
		}
		
	}
	if (sffPairArray != NULL) {
		thread = new pthread_t();
		sffPairArray->maxRecords = readsPerThreadCounter;
		thread_ret_value = pthread_create(thread, NULL, &process_sff_pairs, (void *)sffPairArray);
		if (thread_ret_value) {
			fprintf(stderr, "Error: Unable to create thread - Return Value = %d \n", thread_ret_value);
			exit(-1);
		}
				//pthread join
		thread_vector.push_back(thread);
		if (DEBUG)
			fprintf(stdout, "Starting Thread ID = %d, trying to up max_threads_sem \n", threadCounter);
			
		up(&max_threads_sem);
		
		if (DEBUG)
			fprintf(stdout, "Finished starting Final Thread ID = %d, readPairs count = %d, max threads sem count = %d \n", threadCounter, readsPerThreadCounter, max_threads_sem.count);
	}
	//join all the threads
	while (max_threads_sem.count > 0) {
		pthread_cond_wait(&(max_threads_sem.cond), &(max_threads_sem.mutex));
	}

	//delete all the thread objects created
	std::vector<pthread_t*>::iterator thread_vector_iterator;
	for (thread_vector_iterator=thread_vector.begin(); thread_vector_iterator < thread_vector.end(); thread_vector_iterator++) {
		delete *thread_vector_iterator;
	
	}
	
	
	//once all the paired and corrected records are written to sff file, reset the number of reads in sff commonheader
/*	h->n_reads = numCorrected;
	fseek(out_sff_fp, 0, SEEK_SET);
	sff_header_write(out_sff_fp, h);
	
	//now reset the number of reads in header to fwd singletons and write the header to fwd singleton file
	h->n_reads = fwd_only_present;
	fseek(out_sff_fwd_fp, 0, SEEK_SET);
	sff_header_write(out_sff_fwd_fp, h);
	
	h_rev->n_reads = rev_only_present;
	fseek(out_sff_rev_fp, 0, SEEK_SET);
	sff_header_write(out_sff_rev_fp, h_rev);
	
	h->n_reads = numUnCorrected; //two reads fwd and rev per pair
	fseek(out_sff_paired_fwd_fp, 0, SEEK_SET);
	sff_header_write(out_sff_paired_fwd_fp, h);
	
	h_rev->n_reads = numUnCorrected;
	fseek(out_sff_paired_rev_fp, 0, SEEK_SET);
	sff_header_write(out_sff_paired_rev_fp, h_rev);
*/ 
    fprintf(output_stats_fp, "Summary Statistics	 \n");
    fprintf(output_stats_fp, "Number of beads present in both FWD and REV = %d \n", both_fwd_rev_present);
    fprintf(output_stats_fp, "Number of beads present just in FWD         = %d \n", fwd_only_present);
    fprintf(output_stats_fp, "Number of beads present just in REV	 = %d \n", rev_only_present);
	fprintf(output_stats_fp, "Number of reads corrected	 = %d \n", numCorrected);
    fprintf(output_stats_fp, "Number of reads Uncorrected	 = %d \n", numUnCorrected);
	
	sff_header_destroy(h);
	sff_header_destroy(h_rev);
   
/*    fclose(sff_fp);
    fclose(sff_fp_rev);
	fclose(out_sff_fp);
	fclose(out_sff_fwd_fp);
	fclose(out_sff_rev_fp);
	fclose(out_sff_paired_fwd_fp);
	fclose(out_sff_paired_rev_fp);
*/
	//JZ begins
    bamReader.Close();
	bamReader_rev.Close();
	bamWriter_c.Close();
	bamWriter_f.Close();
	bamWriter_fo.Close();
	bamWriter_r.Close();
	//JZ ends
	fclose(output_stats_fp);
	fclose(fastq_fp);
	
	if(dump_region_info){
		fclose(region_merged_fp);
		fclose(region_merged_fwd_fp);
		fclose(region_merged_rev_fp);		
	}
}

void tokenizeName(char *name_fwd, char *name_rev, int fwdlen, int revlen, int * fwd_x, int *fwd_y, int *rev_x, int *rev_y) {
	char *tok, *name;
	int xf = -1;
	int xr = -1;
        int yf = -1;
	int yr = -1;

	int name_length = fwdlen + 1; // account for NULL termination
        name = (char *) malloc( name_length * sizeof(char) );
        if (!name) {
            fprintf(stderr, "Out of memory! For read name string!\n");
            exit(1);
        }
        memset(name, '\0', (size_t) name_length);
        strncpy(name, name_fwd, (size_t) fwdlen);

	tok = strtok(name, ":");
        int index = 0;
        while (tok != NULL) {
             tok = strtok(NULL, ":");
             index++;
             if (index == 1) xf = atoi(tok);
             else if (index == 2) yf = atoi(tok);
        }

	free(name);
	name_length = revlen + 1; // account for NULL termination
        name = (char *) malloc( name_length * sizeof(char) );
        if (!name) {
            fprintf(stderr, "Out of memory! For read name string!\n");
            exit(1);
        }
        memset(name, '\0', (size_t) name_length);
        strncpy(name, name_rev, (size_t) revlen);

        index = 0;
        tok = strtok(name, ":");
         while (tok != NULL) {
             tok = strtok(NULL, ":");
             index++;
             if (index == 1) xr = atoi(tok);
             else if (index == 2) yr = atoi(tok);
        }

	(*fwd_x) = xf;
	(*fwd_y) = yf;
	(*rev_x) = xr;
	(*rev_y) = yr;

	free(name);


}

//int getNextSFFPair( int numReads, bool *fwd_present, bool *rev_present, FILE *sff_fp, FILE *sff_fp_rev, sff_read_header_t **rh, sff_read_header_t **rh_rev, sff_read_t **rd, sff_read_t **rd_rev, sff_header_t *h, sff_header_t *h_rev){
int getNextSFFPair( int numReads, bool *fwd_present, bool *rev_present, BamReader *pBamReader, BamReader *pBamReader_rev, sff_read_header_t **rh, sff_read_header_t **rh_rev, sff_read_t **rd, sff_read_t **rd_rev, sff_header_t *h, sff_header_t *h_rev){
   int retVal = 0;
   int EOF_FWD = 0;
   int EOF_REV = 0;
   //sff_read_header *prev_fwd_read_header;
   //sff_read_header *prev_rev_read_header;
   //sff_read_data *prev_fwd_read_data;
   //sff_read_data *prev_rev_read_data;
   char *name_fwd = '\0';
   char *name_rev = '\0';
   int fwd_x = 0;
   int fwd_y = 0;
   int rev_x = 0;
   int rev_y = 0;
   long int beadIndexFwd  = 0;
   long int beadIndexRev = 0;
   //char* tok;

   (*fwd_present) = 1;
   (*rev_present) = 1;

	//JZ begins   
	uint16_t nFlows = h->flow_length;
	uint16_t nKeys = h->key_length;    	

	uint16_t nFlows_rev = h_rev->flow_length;
	uint16_t nKeys_rev = h_rev->key_length;
	
    BamAlignment alignment;
    vector<uint16_t> flowInt(nFlows);
    vector<uint16_t> flowInt_rev(nFlows_rev);    
	//JZ ends

   if (numReads == 0) {
/*		*rh = sff_read_header_read(sff_fp);
		EOF_FWD = (*rh==NULL)?1:0;
        if (!EOF_FWD) {
			*rd = sff_read_read(sff_fp, h, *rh);
			EOF_FWD = (*rd==NULL)?1:0;
		}
		*rh_rev = sff_read_header_read(sff_fp_rev);
		EOF_REV = (*rh_rev==NULL)?1:0;
        if (!EOF_REV){
			*rd_rev = sff_read_read(sff_fp_rev, h_rev, *rh_rev);
			EOF_REV = (*rd_rev==NULL)?1:0;
		}
*/
        //JZ begins
   		*rh = NULL;
		*rd = NULL;
		EOF_FWD = 1;
        if(pBamReader->GetNextAlignment(alignment) && alignment.GetTag("FZ", flowInt))
		{	
			rname = alignment.Name;
            int index0 = rname.find(":");
            rname = rname.substr(0, index0);
			*rh = sff_read_header_init();
            (*rh)->name_length = alignment.Name.length();
            (*rh)->name = ion_string_init(alignment.Name.length()+1);
            strcpy((*rh)->name->s, (char*)alignment.Name.c_str());
            (*rh)->n_bases = nKeys + alignment.Length;
			(*rh)->clip_qual_left = nKeys + 1;
			(*rh)->clip_adapter_left = 0;	
			(*rh)->clip_qual_right = (*rh)->n_bases + 1;
			(*rh)->clip_adapter_right = (*rh)->n_bases + 1;

			*rd = sff_read_init();
            (*rd)->flowgram = (uint16_t*)ion_malloc(sizeof(uint16_t)*nFlows, __func__, "(*rd)->flowgram");
            (*rd)->flow_index = (uint8_t*)ion_malloc(sizeof(uint8_t)*((*rh)->n_bases), __func__, "(*rd)->flow_index");
            (*rd)->bases = ion_string_init((*rh)->n_bases+1);
            (*rd)->quality = ion_string_init((*rh)->n_bases+1);

            copy(flowInt.begin(), flowInt.end(), (*rd)->flowgram);

            uint32_t nBase = 0;
			int index = 1;
			vector<uint16_t>::iterator iter = flowInt.begin();
			while(nBase < (*rh)->n_bases && iter != flowInt.end())
			{
				int nHp = ((*iter) + 50) / 100;
				if(nHp > 0)
				{
					(*rd)->flow_index[nBase] = index;
					++nBase;
					--nHp;
					while(nHp > 0 && nBase < (*rh)->n_bases)
					{
						(*rd)->flow_index[nBase] = 0;
						++nBase;
						--nHp;
					}
					index = 1;
				}
				else
				{
					++index;
				}

				++iter;
			} 	
					  
			for(nBase = 0; nBase < nKeys; ++nBase)
			{
                (*rd)->bases->s[nBase] = h->key->s[nBase];
                (*rd)->quality->s[nBase] = DEAFAUL_QUALITY;
			}

            for(int base = 0; base < alignment.Length; ++base, ++nBase)
			{
                (*rd)->bases->s[nBase] = alignment.QueryBases[base];
                (*rd)->quality->s[nBase] = alignment.Qualities[base] - 33;
            }

            (*rd)->bases->l = (*rh)->n_bases;
            (*rd)->quality->l = (*rh)->n_bases;
            (*rd)->bases->s[(*rd)->bases->l]='\0';
            (*rd)->quality->s[(*rd)->quality->l]='\0';

			EOF_FWD = 0;
		}

   		*rh_rev = NULL;
		*rd_rev = NULL;
		EOF_REV = 1;
        if(pBamReader_rev->GetNextAlignment(alignment) && alignment.GetTag("FZ", flowInt_rev))
		{	
			rname_rev = alignment.Name;
			int index1 = rname_rev.find(":");
			rname_rev = rname_rev.substr(0, index1);
			*rh_rev = sff_read_header_init();
            (*rh_rev)->name_length = alignment.Name.length();
            (*rh_rev)->name = ion_string_init(alignment.Name.length()+1);
            strcpy((*rh_rev)->name->s,(char*)alignment.Name.c_str());
            (*rh_rev)->n_bases = nKeys_rev + alignment.Length;
			(*rh_rev)->clip_qual_left = nKeys_rev + 1;
			(*rh_rev)->clip_adapter_left = 0;	
			(*rh_rev)->clip_qual_right = (*rh_rev)->n_bases + 1;
			(*rh_rev)->clip_adapter_right = (*rh_rev)->n_bases + 1;

			*rd_rev = sff_read_init();
            (*rd_rev)->flowgram = (uint16_t*)ion_malloc(sizeof(uint16_t)*nFlows_rev, __func__, "(*rd_rev)->flowgram");
            (*rd_rev)->flow_index = (uint8_t*)ion_malloc(sizeof(uint8_t)*((*rh_rev)->n_bases), __func__, "(*rd_rev)->flow_index");
            (*rd_rev)->bases = ion_string_init((*rh_rev)->n_bases+1);
            (*rd_rev)->quality = ion_string_init((*rh_rev)->n_bases+1);

            copy(flowInt_rev.begin(), flowInt_rev.end(), (*rd_rev)->flowgram);

            unsigned int nBase = 0;
            int index = 1;
            vector<uint16_t>::iterator iter = flowInt_rev.begin();
			while(nBase < (*rh_rev)->n_bases && iter != flowInt_rev.end())
			{
				int nHp = ((*iter) + 50) / 100;
				if(nHp > 0)
				{
					(*rd_rev)->flow_index[nBase] = index;
					++nBase;
					--nHp;
					while(nHp > 0 && nBase < (*rh_rev)->n_bases)
					{
						(*rd_rev)->flow_index[nBase] = 0;
						++nBase;
						--nHp;
					}
					index = 1;
				}
				else
				{
					++index;
				}

				++iter;
			} 	

            for(nBase = 0; nBase < nKeys_rev; ++nBase)
            {
                (*rd_rev)->bases->s[nBase] = h_rev->key->s[nBase];
                (*rd_rev)->quality->s[nBase] = DEAFAUL_QUALITY;
            }

            for(int base = 0; base < alignment.Length; ++base, ++nBase)
            {
                (*rd_rev)->bases->s[nBase] = alignment.QueryBases[base];
                (*rd_rev)->quality->s[nBase] = alignment.Qualities[base] - 33;
            }

            (*rd_rev)->bases->l = (*rh_rev)->n_bases;
            (*rd_rev)->quality->l = (*rh_rev)->n_bases;
            (*rd_rev)->bases->s[(*rd_rev)->bases->l]='\0';
            (*rd_rev)->quality->s[(*rd_rev)->quality->l]='\0';

			EOF_REV = 0;
		}
		//JZ ends

	if (EOF_FWD && EOF_REV) {
	   retVal = 1;
	   (*fwd_present) = 0;
	   (*rev_present) = 0;
	   return retVal;
	}
	else if (EOF_FWD || EOF_REV) {
		if (EOF_FWD) {
		    (*fwd_present) = 0;
		}
		else if (EOF_REV) {
		    (*rev_present) = 0;
		}
		return retVal; 
	}
	
	name_fwd = (*rh)->name->s;
    name_rev= (*rh_rev)->name->s;
    

	//fprintf(stdout, "%s, %s \n", name_fwd, name_rev);

	tokenizeName(name_fwd, name_rev, (*rh)->name_length, (*rh_rev)->name_length, &fwd_x, &fwd_y, &rev_x, &rev_y);


	beadIndexFwd = (long int) BLOCKN * BLOCKM * (MAXM * ((int)floor((float)fwd_x/BLOCKN)) +  (int)floor((float)fwd_y/BLOCKM)) 
			+  ((fwd_x % BLOCKN) * BLOCKN + ( fwd_y % BLOCKM));
	
	beadIndexRev = (long int) BLOCKN * BLOCKM * (MAXM * ((int)floor((float)rev_x/BLOCKN)) +  (int)floor((float)rev_y/BLOCKM)) 
			+  ((rev_x % BLOCKN) * BLOCKN + ( rev_y % BLOCKM));
	
	name_fwd = (*rh)->name->s;
	name_rev = (*rh_rev)->name->s;
	
	if (beadIndexFwd == beadIndexRev) {
		skipFWD = 0;
		skipREV = 0;
	}
	else if (beadIndexFwd < beadIndexRev) {
		skipREV = 1;
		(*rev_present) = 0;
	}
	else if (beadIndexFwd > beadIndexRev) {
		skipFWD = 1;
		(*fwd_present) = 0;
	}

		

   }
   else {
	//fprintf(stdout, "skipfwd=%d skiprev=%d \n", skipFWD, skipREV);
   	if (!skipFWD) {
 /*  		*rh = sff_read_header_read(sff_fp);
		EOF_FWD = (*rh==NULL)?1:0; 
		if (!EOF_FWD){
			*rd = sff_read_read(sff_fp, h, *rh);
			EOF_FWD = (*rd==NULL)?1:0; 
		}
*/
        //JZ begins
   		*rh = NULL;
		*rd = NULL;
		EOF_FWD = 1;
        if(pBamReader->GetNextAlignment(alignment) && alignment.GetTag("FZ", flowInt))
		{	
			*rh = sff_read_header_init();
            (*rh)->name_length = alignment.Name.length();
            (*rh)->name = ion_string_init(alignment.Name.length()+1);
            strcpy((*rh)->name->s, (char*)alignment.Name.c_str());
            (*rh)->n_bases = nKeys + alignment.Length;
			(*rh)->clip_qual_left = nKeys + 1;
			(*rh)->clip_adapter_left = 0;	
			(*rh)->clip_qual_right = (*rh)->n_bases + 1;
			(*rh)->clip_adapter_right = (*rh)->n_bases + 1;

			*rd = sff_read_init();
            (*rd)->flowgram = (uint16_t*)ion_malloc(sizeof(uint16_t)*nFlows, __func__, "(*rd)->flowgram");
            (*rd)->flow_index = (uint8_t*)ion_malloc(sizeof(uint8_t)*((*rh)->n_bases), __func__, "(*rd)->flow_index");
            (*rd)->bases = ion_string_init((*rh)->n_bases+1);
            (*rd)->quality = ion_string_init((*rh)->n_bases+1);

			copy(flowInt.begin(), flowInt.end(), (*rd)->flowgram);

			uint32_t nBase = 0;
			int index = 1;
			vector<uint16_t>::iterator iter = flowInt.begin();
			while(nBase < (*rh)->n_bases && iter != flowInt.end())
			{
				int nHp = ((*iter) + 50) / 100;
				if(nHp > 0)
				{
					(*rd)->flow_index[nBase] = index;
					++nBase;
					--nHp;
					while(nHp > 0 && nBase < (*rh)->n_bases)
					{
						(*rd)->flow_index[nBase] = 0;
						++nBase;
						--nHp;
					}
					index = 1;
				}
				else
				{
					++index;
				}

				++iter;
			} 	

            for(nBase = 0; nBase < nKeys; ++nBase)
            {
                (*rd)->bases->s[nBase] = h->key->s[nBase];
                (*rd)->quality->s[nBase] = DEAFAUL_QUALITY;
            }

            for(int base = 0; base < alignment.Length; ++base, ++nBase)
            {
                (*rd)->bases->s[nBase] = alignment.QueryBases[base];
                (*rd)->quality->s[nBase] = alignment.Qualities[base] - 33;
            }

            (*rd)->bases->l = (*rh)->n_bases;
            (*rd)->quality->l = (*rh)->n_bases;
            (*rd)->bases->s[(*rd)->bases->l]='\0';
            (*rd)->quality->s[(*rd)->quality->l]='\0';

			EOF_FWD = 0;
		}
		//JZ ends
   	}
   	else {
		//*rh = prev_fwd_read_header;
		//*rd = prev_fwd_read_data;
		//fprintf(stdout,"skip fwd so prev name = %s \n", rh->name);
  	}
  
  	if (!skipREV) {
/*		*rh_rev = sff_read_header_read(sff_fp_rev);
		EOF_REV = (*rh_rev==NULL)?1:0; 
        if (!EOF_REV){
			*rd_rev = sff_read_read(sff_fp_rev, h_rev, *rh_rev);
			EOF_REV = (*rd_rev==NULL)?1:0; 
		}
*/
		//JZ begins
   		*rh_rev = NULL;
		*rd_rev = NULL;
		EOF_REV = 1;
        if(pBamReader_rev->GetNextAlignment(alignment) && alignment.GetTag("FZ", flowInt_rev))
		{	
			*rh_rev = sff_read_header_init();
            (*rh_rev)->name_length = alignment.Name.length();
            (*rh_rev)->name = ion_string_init(alignment.Name.length()+1);
            strcpy((*rh_rev)->name->s, (char*)alignment.Name.c_str());
            (*rh_rev)->n_bases = nKeys_rev + alignment.Length;
			(*rh_rev)->clip_qual_left = nKeys_rev + 1;
			(*rh_rev)->clip_adapter_left = 0;	
			(*rh_rev)->clip_qual_right = (*rh_rev)->n_bases + 1;
            (*rh_rev)->clip_adapter_right = (*rh_rev)->n_bases + 1;

			*rd_rev = sff_read_init();
            (*rd_rev)->flowgram = (uint16_t*)ion_malloc(sizeof(uint16_t)*nFlows_rev, __func__, "(*rd_rev)->flowgram");
            (*rd_rev)->flow_index = (uint8_t*)ion_malloc(sizeof(uint8_t)*((*rh_rev)->n_bases), __func__, "(*rd_rev)->flow_index");
            (*rd_rev)->bases = ion_string_init((*rh_rev)->n_bases+1);
            (*rd_rev)->quality = ion_string_init((*rh_rev)->n_bases+1);

            copy(flowInt_rev.begin(), flowInt_rev.end(), (*rd_rev)->flowgram);

            uint32_t nBase = 0;
			int index = 1;
			vector<uint16_t>::iterator iter = flowInt_rev.begin();
			while(nBase < (*rh_rev)->n_bases && iter != flowInt_rev.end())
			{
				int nHp = ((*iter) + 50) / 100;
				if(nHp > 0)
				{
					(*rd_rev)->flow_index[nBase] = index;
					++nBase;
					--nHp;
					while(nHp > 0 && nBase < (*rh_rev)->n_bases)
					{
						(*rd_rev)->flow_index[nBase] = 0;
						++nBase;
						--nHp;
					}
					index = 1;
				}
				else
				{
					++index;
				}

				++iter;
			} 	
		
            for(nBase = 0; nBase < nKeys_rev; ++nBase)
            {
                (*rd_rev)->bases->s[nBase] = h_rev->key->s[nBase];
                (*rd_rev)->quality->s[nBase] = DEAFAUL_QUALITY;
            }

            for(int base = 0; base < alignment.Length; ++base, ++nBase)
            {
                (*rd_rev)->bases->s[nBase] = alignment.QueryBases[base];
                (*rd_rev)->quality->s[nBase] = alignment.Qualities[base] - 33;
            }

            (*rd_rev)->bases->l = (*rh_rev)->n_bases;
            (*rd_rev)->quality->l = (*rh_rev)->n_bases;
            (*rd_rev)->bases->s[(*rd_rev)->bases->l]='\0';
            (*rd_rev)->quality->s[(*rd_rev)->quality->l]='\0';

			EOF_REV = 0;
		}
		//JZ ends
  	}
  	else {
		//*rh_rev = prev_rev_read_header;
		//*rd_rev = prev_rev_read_data;
		//fprintf(stdout, "skip Rev, so prev name = %s \n", rh_rev->name);
  	}

	if (EOF_FWD && EOF_REV) {
           retVal = 1;
	   (*fwd_present) = 0;
	   (*rev_present) = 0;
           return retVal;
        }
        else if (EOF_FWD || EOF_REV) {
                if (EOF_FWD) {
		    (*fwd_present) = 0;
		    skipREV = 0;
                }
                else if (EOF_REV) {
		    (*rev_present) = 0;
		    skipFWD = 0;
                }
                return retVal;
        }

	
	name_fwd = (*rh)->name->s;
	name_rev = (*rh_rev)->name->s;
	//fprintf(stdout, "%s, %s \n", name_fwd, name_rev);
	tokenizeName(name_fwd, name_rev, (*rh)->name_length, (*rh_rev)->name_length, &fwd_x, &fwd_y, &rev_x, &rev_y);

	beadIndexFwd = (long int) BLOCKN * BLOCKM * (MAXM * ((int)floor((float)fwd_x/BLOCKN)) +  (int)floor((float)fwd_y/BLOCKM)) 
			+  ((fwd_x % BLOCKN) * BLOCKN + ( fwd_y % BLOCKM));
	
	beadIndexRev = (long int) BLOCKN * BLOCKM * (MAXM * ((int)floor((float)rev_x/BLOCKN)) +  (int)floor((float)rev_y/BLOCKM)) 
			+  ((rev_x % BLOCKN) * BLOCKN + ( rev_y % BLOCKM));
			
	
	 
	if (beadIndexFwd == beadIndexRev) {
		skipFWD = 0;
		skipREV = 0;
	}
	else if (beadIndexFwd < beadIndexRev) {
		skipREV = 1;
		skipFWD = 0;
		(*rev_present) = 0;
	}
	else if (beadIndexFwd > beadIndexRev) {
		skipFWD = 1;
		skipREV = 0;
		(*fwd_present) = 0;
	}



  }

  return retVal;
}

void output_fastq_entry(FILE *fp, char * name, std::string bases,  int left_trim) {
  register int j;
  uint8_t quality_char;
  std::string out = bases.substr(left_trim);
  int nbases = out.length();
  if (nbases != 0) {
	fprintf(fp,"@%s\n%s\n+\n", name, out.c_str());
	 for (j = 0; j < nbases; j++) {
				  quality_char = 66;
               // quality_char = (quality[j] <= 93 ? quality[j] : 93) + 33;
                fprintf(fp, "%c", (char) quality_char );
        }
        fprintf(fp, "\n");
  }
}


void
construct_fastq_entry(FILE *fp,
                           char *name, char *name_rev,
                           char *bases, char *bases_rev,
                           uint8_t *quality, uint8_t *quality_rev,
                           int nbases, int nbases_rev,
			   uint16_t *flow, uint16_t *flow_rev,
			   int nflows,
			   uint8_t *flow_index, uint8_t *flow_index_rev) {
    register int j;
    uint8_t quality_char;
    uint8_t flow_char;
    if (nbases != 0) {
	char newName[80];
	strcpy(newName, name);
	if (nbases_rev != 0) strcat(newName, ":BOTH");
	else strcat(newName, ":FWD");
    	/* print out the name/sequence blocks */
    	fprintf(fp, "@%s\n%s\n+", newName, bases);

    	/* print out flow values (as integer) */
    	for (j = 0; j < nflows; j++) {
        	fprintf(fp, "%d ", (int) flow[j] );
    	}
    	fprintf(fp,"\n+");
        
    	/* print flow indexes for each base */
    	int t_flows = 0;
    	for (j =0; j < nbases; j++) {
		flow_char = flow_index[j] + 48;
		t_flows += (int)flow_char - 48;
		fprintf(fp, "%c", (char) flow_char);
    	}
    	fprintf(fp, ";%d;%d\n", nbases, (int)strlen(bases));


    	/* print out quality values (as characters)
     	* formula taken from http://maq.sourceforge.net/fastq.shtml
     	*/
    	for (j = 0; j < nbases; j++) {
			fprintf(stdout, "quality = %d \n", quality[j]);
        	quality_char = (quality[j] <= 93 ? quality[j] : 93) + 33;
        	fprintf(fp, "%c", (char) quality_char );
    	}
    	fprintf(fp, "\n");

     }

     if (nbases_rev != 0) {
	char newName[80];
	strcpy(newName, name_rev);
	if (nbases != 0) strcat(newName, ":BOTH");
        else strcat(newName, ":REV");
        /* print out the name/sequence blocks */
        fprintf(fp, "@%s\n%s\n+", newName, bases_rev);

        /* print out flow values (as integer) */
        for (j = 0; j < nflows; j++) {
                fprintf(fp, "%d ", (int) flow_rev[j] );
        }
        fprintf(fp,"\n+");

        /* print flow indexes for each base */
        int t_flows = 0;
        for (j =0; j < nbases_rev; j++) {
                flow_char = flow_index_rev[j] + 48;
                t_flows += (int)flow_char - 48;
                fprintf(fp, "%c", (char) flow_char);
        }
        fprintf(fp, ";%d;%d\n", nbases_rev, (int)strlen(bases_rev));


        /* print out quality values (as characters)
        * formula taken from http://maq.sourceforge.net/fastq.shtml
        */
        for (j = 0; j < nbases_rev; j++) {
                quality_char = (quality_rev[j] <= 93 ? quality_rev[j] : 93) + 33;
                fprintf(fp, "%c", (char) quality_char );
        }
        fprintf(fp, "\n");

     }

}

int* reverseFlowIndex(uint8_t *flow_index, int nbases, int left_clip) {
	int *revFlowIndex = new int[nbases];
	int t_flows = 0;
	for (int i = 0; i < left_clip; i++) t_flows += (int)flow_index[i];
	
	for (int i = left_clip, j = nbases-1; i < nbases+left_clip; i++, j--) {
		t_flows += (int)flow_index[i];
		revFlowIndex[j] = t_flows;
	}

	return revFlowIndex;

}

int* sumFlowIndex(uint8_t *flow_index, int nbases, int left_clip) {
	int *sumFlowIndex = new int[nbases];
	int t_flows = 0;
	for (int i = 0; i < left_clip; i++) t_flows += (int)flow_index[i];
	
	for (int i = left_clip; i < nbases+left_clip; i++) {
		t_flows += (int)flow_index[i];
		sumFlowIndex[i-left_clip] = t_flows;
	}

	return sumFlowIndex;
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

 std::string reverseComplement(std::string a) {
		int len = a.length();
                char *b = new char[len + 1];
                for (int i = len -1, j = 0; i >= 0; i--, j++) {
                        if (a.at(i) == 'A')
                                b[j] = 'T';
                        else if (a.at(i) == 'T')
                                b[j] = 'A';
                        else if (a.at(i) == 'C')
                                b[j] = 'G';
                        else if (a.at(i) == 'G')
                                b[j] = 'C';
                        else
                                b[j] = a.at(i);
                }
                b[len] = '\0';
                std::string s(b);
                delete[] b;
                return s;
 }

 int roundsff(double x) {
      return (int)(x + 0.5);
 }

int  errorCorrect(Alignment* alignment,  int *flow_index, int *flow_index_rev, uint16_t *flow_sig, uint16_t * flow_rev_sig, char *flow_fwd, uint16_t flow_len, char *flow_rev, 
				uint16_t flow_rev_len, ion_string_t *quality, ion_string_t *quality_rev, uint16_t **corr_flow, uint8_t **corr_flow_index, uint8_t **corr_quality, char **corr_bases, char **corr_fastq_bases, int* leftClip, int* rightClip,
				bool *isIdentical, int nbasesRev, int *mergedRegionStartBase, int *mergedRegionStopBase, int *mergedRegionStartBaseFwd, int *mergedRegionStopBaseFwd, int *mergedRegionStartBaseRev, int *mergedRegionStopBaseRev, bool regionInfo) {
				
		int NAME_WIDTH = 18;
		int POSITION_WIDTH = 6;
		int SEQUENCE_WIDTH = 600;
		std::string sequence1 = alignment->getSequence1();
		std::string sequence2 = alignment->getSequence2();
		std::string markup = alignment->getMarkupLine();
	    std::string name1 = alignment->getName1();
		std::string name2 = alignment->getName2();	

        //variables for the handling of strand bias
        bool fwd_flow_has_bias = false;
        bool rev_flow_has_bias = false;
        bool fwd_flow_index_has_bias = false;
        bool rev_flow_index_has_bias = false;
        std::set<int> fwd_flow_bias_index_set;
        std::set<int> rev_flow_bias_index_set;

		int length = sequence1.length() > sequence2.length() ? sequence2.length() : sequence1.length();	

		
		std::string buffer = "";
	    std::string preMarkup = "";
		std::string flowSigAlign = "";
		std::string flowAlign = "";
		std::string flowIndexAlign = "";
		std::string flowSigAlignRev = "";
		std::string flowAlignRev = "";
		std::string flowIndexAlignRev = "";
		char *subsequence1 = NULL;
		char *subsequence2 = NULL;
		char *submarkup = NULL;
		
		NAME_WIDTH = name1.length();
		
		int oldPosition1 = 1 + alignment->getStart1();
		int oldPosition2 = 1 + alignment->getStart2();
		int position1 = alignment->getStart1() + alignment->getLengthOfSequence1();
		int position2 = alignment->getStart2() + alignment->getLengthOfSequence2();
		
		int rightClipFlowIndex = 0;
		
		int mergedRegionStartFlow = 0;
		int mergedRegionStopFlow = 0;
		int mergedRegionStartFlowRev = 0;
		int mergedRegionStopFlowRev = 0;			
		int mergedRegionFlowLen = 0;
		int regionAlignCase  = 0;

		SmithWaterman *flowAligner = new SmithWaterman();
		Alignment *flowAlignment = NULL;
		
		
		if (DEBUG) {
			for (int j = 0; j < NAME_WIDTH + 1 + POSITION_WIDTH + 1; j++) {
				preMarkup += BLANK;
			}
		
			position1 = 1 + alignment->getStart1();
			position2 = 1 + alignment->getStart2();
		
			
			int line;
			char c1, c2;
		
			for (int i = 0; i * SEQUENCE_WIDTH < length; i++) {
			
				oldPosition1 = position1;
				oldPosition2 = position2;
			
				line = ((i + 1) * SEQUENCE_WIDTH) < length ? (i + 1) * SEQUENCE_WIDTH: length;

				subsequence1 = new char[line - i * SEQUENCE_WIDTH];
				subsequence2 = new char[line - i * SEQUENCE_WIDTH];
				submarkup = new char[line  - i * SEQUENCE_WIDTH];
				subsequence1[line - i * SEQUENCE_WIDTH - 1] = '\0';
				subsequence2[line - i * SEQUENCE_WIDTH - 1] = '\0';
				submarkup[line - i * SEQUENCE_WIDTH - 1] = '\0';
				int k1 = 0;	
				for (int j = i * SEQUENCE_WIDTH, k = 0; j < line; j++, k++) {
					subsequence1[k] = sequence1.at(j);
					subsequence2[k] = sequence2.at(j);
					submarkup[k] = markup[j];
					c1 = subsequence1[k];
					c2 = subsequence2[k];
					if (c1 == c2) {
						position1++;
						position2++;
					} else if (c1 == GAP) {
						position2++;
					} else if (c2 == GAP) {
						position1++;
					} else {
						position1++;
						position2++;
					}
					k1 = k;
				}

				buffer += name1;
				buffer += BLANK;
				buffer += intToString(oldPosition1, POSITION_WIDTH);
				buffer += BLANK;
				buffer += std::string(subsequence1, k1);
				buffer += BLANK;
				buffer += intToString(position1-1, POSITION_WIDTH);
				buffer += '\n';
			
				buffer += preMarkup;
				buffer += std::string(submarkup, k1);
				buffer += '\n';

				buffer += name2;
				buffer += BLANK;
				buffer += intToString(oldPosition2, POSITION_WIDTH);
				buffer += BLANK;
				buffer += std::string(subsequence2, k1);
				buffer += BLANK; 
				buffer += intToString(position2 - 1, POSITION_WIDTH);
				buffer += '\n';
			
			}

			position1--;
			position2--;
		
		} //end DEBUG
		
		
		int startPos1 = oldPosition1-1;
		int startPos2 = oldPosition2-1;
		
		int *flowIndexFwdAligned = new int[flow_len];
		int *flowIndexRevAligned = new int[flow_len];
		int *flowIndexFwdCorrected = new int[flow_len];
	    int *flowIndexRevCorrected = new int[flow_len];

		float *flowPriors = new float[flow_len];
		
		int prevFlow = 0;
		int curFlow = 0;
		int curFlowRev = 0;
		int prevFlowRev = 0;
		int fwdFlowLen = 0;
		int revFlowLen = 0;
		int totalFwdFlows = 0;
		int totalRevFlows = 0;
		int indexPos = 0;
		int endPosition1 = alignment->getStart1() + alignment->getLengthOfSequence1();
		int endPosition2 = alignment->getStart2() + alignment->getLengthOfSequence2(); 

		int lastFwdFlowIndex;
		int lastRevFlowIndex;
		
		while (startPos1 < endPosition1 ) {	
			curFlow = flow_index[startPos1];
			if (curFlow != prevFlow) {
				  flowIndexFwdAligned[indexPos++] = curFlow;
				  flowAlign += flow_fwd[curFlow-1]; //essentially constructing the homopolymer sequence for fwd read
				  if (DEBUG) {
					flowIndexAlign += intToString(curFlow,3);					
					flowSigAlign += intToString(flow_sig[curFlow-1],3);
				  }
				prevFlow = curFlow;
				fwdFlowLen++;
				totalFwdFlows++;
			}	
			startPos1++;
						
		}
		//reset counter values to zero
		curFlow = prevFlow = indexPos = 0;
		
		while (startPos2 < endPosition2 && curFlow >= 0 && curFlow <= totalFlowLengthFwd) {
			curFlowRev = flow_index_rev[startPos2];
			if (curFlowRev != prevFlowRev) {
				flowIndexRevAligned[indexPos++] = curFlowRev;
				flowAlignRev += complimentNuc(flow_rev[curFlowRev-1]); //essentially constructing the homopolymer sequence for Rev read
				if (DEBUG) {
					flowIndexAlignRev += intToString(curFlowRev,3);					
					flowSigAlignRev += intToString(flow_rev_sig[curFlowRev-1],3);
				}
				prevFlowRev = curFlowRev;
				revFlowLen++;
				totalRevFlows++;
			} 
			startPos2++;
		}

		if (DEBUG) {	
			fprintf(stdout, "%s \n", buffer.c_str()) ;
			fprintf(stdout, "%s \n", flowSigAlign.c_str());
			fprintf(stdout, "%s \n", flowSigAlignRev.c_str());

			fprintf(stdout, "%s \n", flowIndexAlign.c_str());
			fprintf(stdout, "%s \n", flowIndexAlignRev.c_str());	
			fprintf(stdout, "%s \n", flowAlign.c_str());
			fprintf(stdout, "%s \n", flowAlignRev.c_str());
		
			fprintf(stdout, "Similarity = %d, Identity = %8.2f \n", alignment->getSimilarity(), ((float)alignment->getIdentity()/length));
			fprintf(stdout, "Total fwd flows = %d, Rev Flows  = %d \n", totalFwdFlows, totalRevFlows);
		
		}
		
		uint16_t *flow_fwd_sig = new uint16_t[flow_len]; 
		memcpy(flow_fwd_sig, flow_sig, flow_len*sizeof(uint16_t)); //copy value from flow_sig to this variable

		//reset the start positions for the alignment
		startPos1 = oldPosition1-1;
		startPos2 = oldPosition2-1;
		
		std::string correctedBasesString = "";

			int homPolyFwd = 0;
			//float fwdDev = 0;
			//float avgDev = 0;
			int homPolyRev = 0;
			
			//start with setting all the flow priors to 1.0
			for (int i = 0; i < flow_len; i++) 
				flowPriors[i] = NEUTRAL_PRIOR;
				
			//if the fwd and reverse reads are identical the set the prior to be 10.0 in regions of alignment
			if (alignment->getIdentity()/alignment->getSequence1().length() == 1) {
				int flowPos = 0;
				for (int counter = 0; counter < totalFwdFlows; counter++) {
					flowPos = flowIndexFwdAligned[counter]; //get the flow position in the aligned region
					flowPriors[flowPos-1] = ALIGN_PRIOR; //set the prior to 10.0 for all flows where there is perfect alignment between fwd and rev reads.
				}
				numIdentical++;
				//*leftClip = oldPosition1;
				regionAlignCase = 1;		
				*isIdentical = true;
				if (flowIndexRevAligned[totalRevFlows-1] < 20)
					rightClipFlowIndex = flowIndexFwdAligned[totalFwdFlows-1];

				
				if (outputIdenticalOnly) {
				
				  *rightClip = endPosition1-1; //set the right clip to end of the alignment
				  
				  delete[] flowIndexFwdAligned;
		          delete[] flowIndexRevAligned;
				  delete[] flow_fwd_sig;
	              delete[] flowIndexFwdCorrected;
           		  delete[] flowIndexRevCorrected;
	              delete[] flowPriors;
				  delete flowAlignment;
				  delete flowAligner;
				  flowAligner = NULL;
					if (DEBUG) {		
						delete[] subsequence1;
						delete[] subsequence2;
						delete[] submarkup;
					}
					return 0; 
				}
				//to generate merged read for fastq file used in denovo
				lastFwdFlowIndex = flowIndexFwdAligned[totalFwdFlows-1];
				lastRevFlowIndex = flowIndexRevAligned[totalRevFlows-1];
			}
			else if (flowAlign.compare(flowAlignRev) != 0) { //base space alignment is not identical and flow/homopolymer alignment is also not identical		 
				 flowAlignment = flowAligner->align(flowAlign, flowAlignRev, 0.5, 0.5, -2); //actually align the homopolymers space or flowspace
				 std::string flowSeq1 = flowAlignment->getSequence1();
				 std::string flowSeq2 = flowAlignment->getSequence2();			 

				 int adjPosition1 = flowAlignment->getStart1();
				 int adjPosition2 = flowAlignment->getStart2();
				 int flowLen = flowSeq1.length() > flowSeq2.length() ? flowSeq2.length() : flowSeq1.length();
				 
				 regionAlignCase = 2;
				 mergedRegionFlowLen = flowLen;	 
				 //int flowLen = length;
				 char fchar1, fchar2; 
				 int k = 0;
				 int miscount = 0;
				 int giveup = 0;
				 int prevFwdGap = 0;
				 int prevRevGap = 0;
				 int prevFwdFoundIndex = 0;
				 int prevRevFoundIndex = 0;
				 
				 for (k = 0; k < flowLen; k++) {
					
					fchar1 = flowSeq1.at(k);
					fchar2 = flowSeq2.at(k);	
					if (fchar1 == fchar2 ) {
					   flowIndexFwdCorrected[k] = flowIndexFwdAligned[adjPosition1];
					   flowIndexRevCorrected[k] = flowIndexRevAligned[adjPosition2];
					   adjPosition1++;
					   adjPosition2++;
					   prevFwdGap = 0;
					   prevRevGap = 0;
					   flowPriors[flowIndexFwdCorrected[k]-1] = ALIGN_PRIOR;
					
					}
					else if (fchar1 == GAP && fchar2 != GAP) {
						flowIndexRevCorrected[k] = flowIndexRevAligned[adjPosition2];
						adjPosition2++;
						int prevIndex = (prevFwdGap == 0 ? flowIndexFwdAligned[adjPosition1-1] : prevFwdFoundIndex) + 1;
						int curIndex = flowIndexFwdAligned[adjPosition1];
						int found = 0;
						
						while (prevIndex < curIndex)	{
							if (flow_fwd[prevIndex-1] == fchar2) {
								found = 1;
								break;
							}
							else
								prevIndex++;
						}
						if (found) {
							prevFwdFoundIndex = prevIndex;
							flowIndexFwdCorrected[k] = prevIndex;
							flowPriors[flowIndexFwdCorrected[k]-1] = NEUTRAL_PRIOR;
							
						}
						else {
								prevFwdFoundIndex = flowIndexFwdAligned[adjPosition1-1];
								//std::cout << "ERROR: Unable to find flow in fwd read for Base " << fchar2 << " between flow positions " << flowIndexFwdAligned[adjPosition1-1] << " and " << flowIndexFwdAligned[adjPosition1] << std::endl;
								flowIndexFwdCorrected[k] = -10;		
								miscount++;
						}
						prevFwdGap = 1;
						prevRevGap = 0;
						
					}
					else if (fchar1 != GAP && fchar2 == GAP) {
						flowIndexFwdCorrected[k] = flowIndexFwdAligned[adjPosition1];
						adjPosition1++;
						int prevIndex = (prevRevGap == 0 ? flowIndexRevAligned[adjPosition2-1] : prevRevFoundIndex);
						int curIndex = flowIndexRevAligned[adjPosition2] + 1;
						int found = 0;
						//first complement the base before searching in the reverse flow order
						fchar1 = complimentNuc(fchar1);
						//std::cout << "rev gap from  " << curIndex << " to " << prevIndex << " base search = " << fchar1 << std::endl;
						while (curIndex < prevIndex) {
							if (flow_rev[curIndex-1] == fchar1) {
								found = 1;
								break;
							}
							else
								curIndex++;
						}
						if (found) {
							prevRevFoundIndex = curIndex;
							flowIndexRevCorrected[k] = curIndex;
							flowPriors[flowIndexFwdCorrected[k]-1] = NEUTRAL_PRIOR;
						}
						else {
							prevRevFoundIndex = flowIndexRevAligned[adjPosition2-1];
							//std::cout << "ERROR: Unable to find flow in rev read for Base " << fchar1 << " between flow positions " << flowIndexRevAligned[adjPosition2] << " and " << flowIndexRevAligned[adjPosition2-1] << std::endl;
							flowIndexRevCorrected[k] = -10;
							flowPriors[flowIndexFwdCorrected[k]-1] = MISMATCH_PRIOR;
							miscount++;
						}	
						prevRevGap = 1;
						prevFwdGap = 0;
					}
					else if (fchar1 != fchar2) {
						//std::cout << "Invalid Flow space Alignment: Mismatches found in flows " << std::endl;
						miscount++;
						giveup = 1;
						break;
					}
				}//end for loop

				if (miscount > 5) giveup = 1;

				if (giveup) { 
					//std::cout << "Unable to correct - Giving up: Mismatch count = " << miscount << std::endl;
				    delete[] flowIndexFwdAligned;
                	delete[] flowIndexRevAligned;
					delete[] flow_fwd_sig;
                	delete[] flowIndexFwdCorrected;
                	delete[] flowIndexRevCorrected;
                	delete[] flowPriors;
					delete flowAlignment;
					delete flowAligner;
					flowAligner = NULL;
					if (DEBUG) {		
						delete[] subsequence1;
						delete[] subsequence2;
						delete[] submarkup;
					}

					return 0; // unable to correct due to misaligned flows, so the record should go into Singleton file
				}
				else {
  				    //refineFlowAlignment(flowIndexFwdCorrected, flowIndexRevCorrected, flowSeq1, flowSeq2, flow_fwd, flow_rev, flow_fwd_sig, flow_rev_sig);				 
					//std::cout << "correct flow space alignment " << std::endl;		
					int cA = 0;
					std::string correctedAlignIndex = "";
					std::string correctedAlignRevIndex = "";
					std::string correctedAlign = "";
					std::string correctedAlignRev = "";
					int findex = -10;
					int findexRev = -10;
					float avgFlowSignal = 0;
					//char correctedbase = ' ';
					//int homPoly = 0;
							
                    fwd_flow_has_bias = find_read_in_bias_map(name1, &strand_bias_flow_index_map_fwd, &fwd_flow_bias_index_set);
                    rev_flow_has_bias = find_read_in_bias_map(name2, &strand_bias_flow_index_map_rev, &rev_flow_bias_index_set);

					while (cA < flowLen) {
						avgFlowSignal = 0;
						
						findex = flowIndexFwdCorrected[cA];
						findexRev = flowIndexRevCorrected[cA];

                        fwd_flow_index_has_bias = false;
                        rev_flow_index_has_bias = false;

                        if(fwd_flow_has_bias && findex != -10) fwd_flow_index_has_bias = find_flow_index_with_bias(fwd_flow_bias_index_set, findex);
                        if(rev_flow_has_bias && findexRev !=-10) rev_flow_index_has_bias = find_flow_index_with_bias(rev_flow_bias_index_set, findexRev);


                        if (DEBUG) {
                            correctedAlignIndex += intToString(findex,3);
                            correctedAlignRevIndex += intToString(findexRev,3);
                        }

                        if (findex != -10) {

                            avgFlowSignal += flow_sig[findex-1];
                            //correctedbase = flow_fwd[findex-1];
                            homPolyFwd = roundsff((float)flow_sig[findex-1]/100);
                            //fwdDev = abs(flow_fwd[findex-1] - homPolyFwd*100);
                            if (DEBUG)
                                correctedAlign += intToString(flow_sig[findex-1], 3);
                        }
                        else {
                            if (DEBUG)
                                correctedAlign += intToString(0, 3);
                        }

                        if (findexRev != -10) {
                            if(!fwd_flow_index_has_bias && !rev_flow_index_has_bias) avgFlowSignal += flow_rev_sig[findexRev-1];
                            homPolyRev = roundsff((float)flow_rev_sig[findexRev-1]/100);
                            if (DEBUG)
                                correctedAlignRev += intToString(flow_rev_sig[findexRev-1],3);
                        }
                        else {
                            if (DEBUG)
                                correctedAlignRev += intToString(0,3);
                        }

                        //update the flow signals with corrected values
                        if(!fwd_flow_index_has_bias && !rev_flow_index_has_bias){
                            if (findex != -10 && findexRev != -10) {
                                if (abs((int)(avgFlowSignal/2 - flow_sig[findex-1])) < 50)
                                flow_sig[findex-1] = (uint16_t) avgFlowSignal/2;
                            }
                            else if (findex != -10 && findexRev == -10) {
                                if (flow_sig[findex-1] < 75)
                                    flow_sig[findex-1] = 0;
                            }

                            avgFlowSignal /= 200;
                        }
                        else{
                            avgFlowSignal /= 100;
                        }


						//homPoly = roundsff(avgFlowSignal );	
						//avgDev = fabs(avgFlowSignal - homPoly)*100;
						if (findex != -10 && findexRev != -10) {
							if (homPolyFwd != homPolyRev) {
								flowPriors[findex-1] = MISMATCH_PRIOR;
								//correctedBasesString += "N"; //set the first base to N
								//baseIncCounter++;
								
							}
							else {
								flowPriors[findex-1] = ALIGN_PRIOR;
							}
							
						} 
						
						cA++;
					} 	
					  
					if (flowIndexRevCorrected[flowLen-1] < 20 )
 						  rightClipFlowIndex = flowIndexFwdCorrected[flowLen-1];
					
					if (DEBUG) {
						std::cout << correctedAlignIndex << std::endl;	
						std::cout << correctedAlignRevIndex << std::endl;
						std::cout << correctedAlign << std::endl;
						std::cout << correctedAlignRev << std::endl;
						
					}
					
					lastFwdFlowIndex = flowIndexFwdCorrected[flowLen-1];
					lastRevFlowIndex = flowIndexRevCorrected[flowLen-1];

				}
				
			
				delete flowAligner;
				flowAligner = NULL;
				

		    }
		    else {
				//flowsignal alignment is identical 
				//numCorrected++;
		    	regionAlignCase = 3;	
				//int homPoly = 0;
				//char correctedbase = ' ';
				int count = 0;
				int findex = 0, findexrev = 0;
				float avgFlowSignal = 0;

                fwd_flow_has_bias = find_read_in_bias_map(name1, &strand_bias_flow_index_map_fwd, &fwd_flow_bias_index_set);
                rev_flow_has_bias = find_read_in_bias_map(name2, &strand_bias_flow_index_map_rev, &rev_flow_bias_index_set);

				while (count < totalFwdFlows) {
				   avgFlowSignal = 0;
				   findex = flowIndexFwdAligned[count];
				   findexrev = flowIndexRevAligned[count];

                   fwd_flow_index_has_bias = false;
                   rev_flow_index_has_bias = false;

                   if(fwd_flow_has_bias && findex > 0) fwd_flow_index_has_bias = find_flow_index_with_bias(fwd_flow_bias_index_set, findex);
                   if(rev_flow_has_bias && findexrev >0) rev_flow_index_has_bias = find_flow_index_with_bias(rev_flow_bias_index_set, findexrev);


                   if (findex > 0) {
                       //correctedbase = flow_fwd[findex-1];
                       avgFlowSignal += flow_sig[findex-1];
                       homPolyFwd = roundsff((float)flow_sig[findex-1]/100);
                       //fwdDev = abs(flow_fwd[findex-1] - homPolyFwd*100);
                   }

                   if(!fwd_flow_index_has_bias && !rev_flow_index_has_bias){
                     if (findexrev > 0) {
                         avgFlowSignal += flow_rev_sig[findexrev-1];
                         homPolyRev = roundsff((float)flow_rev_sig[findexrev-1]/100);
                     }
                     if (abs((int)(avgFlowSignal/2 - flow_sig[findex-1])) < 50)
                        flow_sig[findex-1] = (uint16_t)avgFlowSignal/2;

                     avgFlowSignal /= 200;
                   }
                   else{
                       avgFlowSignal /= 100;
                   }

                   //homPoly = roundsff(avgFlowSignal);
                   //avgDev = fabs(avgFlowSignal - homPoly)*100;
                   if (findex > 0) {
					 
					 if (homPolyFwd != homPolyRev) {
						flowPriors[findex-1] = MISMATCH_PRIOR;
								//correctedBasesString += "N"; //set the first base to N
								//baseIncCounter++;
								
					}
					else {
								flowPriors[findex-1] = ALIGN_PRIOR;
					}
					/*
					else if (avgDev > fwdDev && homPoly < homPolyFwd) {
								correctedBasesString += "N"; //set the first base to N
								homPoly = homPolyFwd;
								baseIncCounter++;
								
					}
					 while(baseIncCounter < homPoly) {
					 correctedBasesString += correctedbase;
					 baseIncCounter++;
					 }
					 */

                  }
				  count++;

				}	//end while loop

				if (flowIndexRevAligned[totalRevFlows-1] < 20 )
					rightClipFlowIndex = flowIndexFwdAligned[totalFwdFlows-1];
				
				lastFwdFlowIndex = flowIndexFwdAligned[totalFwdFlows-1];
				lastRevFlowIndex = flowIndexRevAligned[totalRevFlows-1];
		    }
		
			//adjust merge flow positions to minimize the cases of clipping in region-specific error calculation 
			if(regionInfo){
				int leftSafeBand = 0;
				int rightSafeBand = 0;

				if(regionAlignCase==1 || regionAlignCase==3){
					leftSafeBand = totalFwdFlows/4;
					rightSafeBand = totalFwdFlows/8;					
					mergedRegionStartFlow = flowIndexFwdAligned[leftSafeBand]; 
					mergedRegionStopFlow = flowIndexFwdAligned[totalFwdFlows-1-rightSafeBand];
					mergedRegionStartFlowRev = flowIndexRevAligned[leftSafeBand]; 
					mergedRegionStopFlowRev = flowIndexRevAligned[totalFwdFlows-1-rightSafeBand];						
				}else if(regionAlignCase==2){
					leftSafeBand = mergedRegionFlowLen/4;
					rightSafeBand = mergedRegionFlowLen/8;
					
					mergedRegionStartFlow = flowIndexFwdCorrected[leftSafeBand];
					mergedRegionStopFlow = flowIndexFwdCorrected[mergedRegionFlowLen-1-rightSafeBand];
					mergedRegionStartFlowRev = flowIndexRevCorrected[leftSafeBand];
					mergedRegionStopFlowRev = flowIndexRevCorrected[mergedRegionFlowLen-1-rightSafeBand];
					
					while((mergedRegionStartFlow==-10 || mergedRegionStartFlowRev==-10)&& leftSafeBand>0){
						leftSafeBand--;
						mergedRegionStartFlow = flowIndexFwdCorrected[leftSafeBand];
						mergedRegionStartFlowRev = flowIndexRevCorrected[leftSafeBand];
					}
					while((mergedRegionStopFlow==-10 || mergedRegionStopFlowRev==-10)&& rightSafeBand>0){
						rightSafeBand--;
						mergedRegionStopFlow = flowIndexFwdCorrected[mergedRegionFlowLen-1-rightSafeBand];
						mergedRegionStopFlowRev = flowIndexRevCorrected[mergedRegionFlowLen-1-rightSafeBand];						
					}					
				}
				
				int temp_flow = mergedRegionStartFlowRev;
				mergedRegionStartFlowRev = mergedRegionStopFlowRev;
				mergedRegionStopFlowRev = temp_flow;		

			}
			
			//once all the flows have been corrected and priors assigned, now call bases and baseQVs bases on the corrected flowgram
			int totBases = 0;
			int prevTotBases = 0;
			int baseIncCounter = 0;
			float FlowSig = 0;
			int homPoly = 0;
			int totalFastqBases = 0;
			int totalFastqFwdBases = 0;
			for (int n = 0; n < flow_len; n++) {
					FlowSig = flow_sig[n];
					homPoly = roundsff((float)FlowSig/100);
					baseIncCounter = 0;
					while (baseIncCounter < homPoly) {
						baseIncCounter++;
						totBases++;
						if (n <= lastFwdFlowIndex) 
							totalFastqBases++;
							
					}
						
			}
			totalFastqFwdBases = totalFastqBases-1;
			
			for (int n = 0; n < lastRevFlowIndex; n++) {
				FlowSig = flow_rev_sig[n];
				homPoly = roundsff((float)FlowSig/100);
				baseIncCounter = 0;
				while (baseIncCounter < homPoly) {
						baseIncCounter++;
						totalFastqBases++;
				}
			}
			
			*corr_flow = (uint16_t *) malloc(flow_len * sizeof(uint16_t) );
			*corr_flow_index = (uint8_t *) malloc( totBases * sizeof(uint8_t)  );
			*corr_quality = (uint8_t *) malloc( totBases * sizeof(uint8_t)  );
			*corr_bases = (char * ) malloc( totBases+1 * sizeof(char) );
			*corr_fastq_bases = (char *) malloc( totalFastqBases+1 * sizeof(char));
			
				prevTotBases = totBases;
				baseIncCounter = 0;
				totBases = 0;
				homPoly = 0;
				int prevPos = 0;
				float dev = 0;
				float prior = 1.0;
				uint8_t minBaseQV = 0;
				uint8_t baseQV = 0;
				int iminBaseQV = 0, iBaseQV = 0;
				for (int n = 0; n < flow_len; n++) 
				{
					(*corr_flow)[n] = flow_sig[n];
					FlowSig = flow_sig[n];
					homPoly = roundsff(((float)FlowSig)/100);
										
					dev = fabs(FlowSig - homPoly*100);
					if (dev == 0) dev = 0.1; //dont allow for 0, as probability of error calculated based on dev can never be exactly zero.
					baseIncCounter = 0;
					prior = flowPriors[n];
					if (homPoly > 5) prior = prior * 2; // set the prior different by homopolymer length
					
					iminBaseQV = (int)(-10.0*log10((dev/1000) * prior));
					if (iminBaseQV > 255) iminBaseQV = 255;
					else if (iminBaseQV < 0) iminBaseQV = 0;
					minBaseQV = (uint8_t)(iminBaseQV & 0xff); //minBaseQV applies only to positions at the edge of the homopolymer sequence ex. TTTTT, the first and last T will get minBaseQV
					if (minBaseQV > 40) minBaseQV = 40;  
					//if (minBaseQV < 0) minBaseQV = 0;
					
					iBaseQV = (int)(-10.0*log10( 0.01 * prior));
					if (iBaseQV > 255) iBaseQV = 255;
					else if (iBaseQV < 0) iBaseQV = 0;
					baseQV = (uint8_t)(iBaseQV & 0xff); //baseQV applies to positions in the middle of homopolymer sequence ex. TTTTT, the middle three T's will get baseQV.
					if (baseQV > 40) baseQV = 40;
					
					while (baseIncCounter < homPoly) {
						if (totBases > prevTotBases) {
							std::cerr << "ERROR: Calcuting Bases from corrected Flow signals, total bases before " << prevTotBases << " and after " << totBases << " are not matching " << std::endl;
							exit(-1);	
						}
						(*corr_flow_index)[totBases] = n - prevPos;
						(*corr_bases)[totBases] = flow_fwd[n];
						
						if (baseIncCounter == 0 || baseIncCounter == (homPoly-1)) //first or last base in homopolymer sequence
						{
							(*corr_quality)[totBases] = minBaseQV ;
							//std::cout << "MinBaseqv = " << (int)minBaseQV << " prior = " << prior << " dev = " << dev <<  " baseCounter = " << baseIncCounter << " homPoly = " << homPoly << std::endl;
						}
						else
						{
							(*corr_quality)[totBases] = baseQV ;
							//std::cout << "Baseqv = " << (int)baseQV << " prior = " << prior << " dev = " << dev << " baseCounter = " << baseIncCounter << " homPoly = " << homPoly << std::endl;
						}
						
						//std::cout << "minBaseQV = " << (*corr_quality)[totBases] << " int value = " << minBaseQV << std::endl;
						totBases++;
						baseIncCounter++;
						prevPos = n;
												
					}
					
					//set the right clip to end of fwd and reverse read alignment
					if (rightClipFlowIndex != 0 && rightClipFlowIndex == n) {
						*rightClip = totBases-1;
					}
					
					
					if(mergedRegionStartFlow -1 == n) *mergedRegionStartBase = totBases - baseIncCounter; 
					if(mergedRegionStopFlow -1 == n) *mergedRegionStopBase = totBases - 1;			
	
						
				}
				//copy the fwd bases to fastq bases for denovo assembly
				strncpy(*corr_fastq_bases, (const char *)(*corr_bases), totalFastqFwdBases);
				
				(*corr_bases)[totBases] = '\0';
				
				
				//now go ahead and append to fastq string from reverse bases, remember to complement the bases
				for (int n = lastRevFlowIndex-2; n > 10; n--) {
					FlowSig = flow_rev_sig[n];
					homPoly = roundsff(((float)FlowSig)/100);
					baseIncCounter = 0;
					while (baseIncCounter < homPoly) {
						if (totalFastqFwdBases >= totalFastqBases) {
							std::cout << "ERROR: totalFastqFwdBases = " << totalFastqFwdBases << " totalFastqBases " << totalFastqBases << endl;
						}
						(*corr_fastq_bases)[totalFastqFwdBases++] = complimentNuc(flow_rev[n]);
						baseIncCounter++;
					}
				}
				if (totalFastqFwdBases > totalFastqBases)
					std::cout << "ERROR: totalFastqFwdBases = " << totalFastqFwdBases << " totalFastqBases " << totalFastqBases << endl;
					
					
				(*corr_fastq_bases)[totalFastqFwdBases] = '\0';
				
			    if(regionInfo){
					//below for fwd read			    	
					int totBases_fwd = 0;
					prevTotBases = 0;
					baseIncCounter = 0;
					FlowSig = 0;
					homPoly = 0;	
			
					for (int n = 0; n < flow_len; n++) {
							FlowSig = flow_fwd_sig[n];
							homPoly = roundsff((float)FlowSig/100);
							baseIncCounter = 0;
							while (baseIncCounter < homPoly) {
								baseIncCounter++;
								totBases_fwd++;
							}
								
					}		
				
					prevTotBases = totBases_fwd;
					baseIncCounter = 0;
					totBases_fwd = 0;
					homPoly = 0;			

					for (int n = 0; n < flow_len; n++) 
					{
						FlowSig = flow_fwd_sig[n];
						homPoly = roundsff(((float)FlowSig)/100);										
						baseIncCounter = 0;
						
						while (baseIncCounter < homPoly) {
							if (totBases_fwd > prevTotBases) {
								std::cerr << "ERROR: FWD Calcuting Bases from Flow signals, total bases before " << prevTotBases << " and after " << totBases << " are not matching " << std::endl;
								exit(-1);	
							}					
							
							totBases_fwd++;
							baseIncCounter++;												
						}
						if(mergedRegionStartFlow -1 == n) *mergedRegionStartBaseFwd = totBases_fwd - baseIncCounter;
						if(mergedRegionStopFlow -1 == n) *mergedRegionStopBaseFwd = totBases_fwd - 1;							
												
					}

					//above for fwd

					//below for rev read
					int totBases_rev = 0;
					prevTotBases = 0;
					baseIncCounter = 0;
					FlowSig = 0;
					homPoly = 0;

					for (int n = 0; n < flow_len; n++) {
							FlowSig = flow_rev_sig[n];
							homPoly = roundsff((float)FlowSig/100);
							baseIncCounter = 0;
							while (baseIncCounter < homPoly) {
								baseIncCounter++;
								totBases_rev++;
							}
								
					}		
				
					prevTotBases = totBases_rev;
					baseIncCounter = 0;
					totBases_rev = 0;
					homPoly = 0;		
					

					for (int n = 0; n < flow_len; n++) 
					{
						FlowSig = flow_rev_sig[n];
						homPoly = roundsff(((float)FlowSig)/100);										
						baseIncCounter = 0;
						
						while (baseIncCounter < homPoly) {
							if (totBases_rev > prevTotBases) {
								std::cerr << "ERROR: REV read Calcuting Bases from Flow signals, total bases before " << prevTotBases << " and after " << totBases << " are not matching " << std::endl;
								exit(-1);	
							}				
								
							totBases_rev++;
							baseIncCounter++;												
						}
						if(mergedRegionStartFlowRev -1 == n) *mergedRegionStartBaseRev = totBases_rev - baseIncCounter;
						if(mergedRegionStopFlowRev -1 == n) *mergedRegionStopBaseRev = totBases_rev - 1;				
																			
					}				
					//above for rev
			    }
				   
				
			
				
		if (DEBUG) {		
			delete[] subsequence1;
			delete[] subsequence2;
			delete[] submarkup;
		}
		delete[] flowIndexFwdAligned;
		delete[] flowIndexRevAligned;
		delete[] flow_fwd_sig;
		delete[] flowIndexFwdCorrected;
		delete[] flowIndexRevCorrected;
		delete[] flowPriors;		
		if (flowAligner != NULL)
		 	delete flowAligner;	
		if (flowAlignment != NULL)
			delete flowAlignment;
		
		return totBases;
	}

	char complimentNuc (char i) {
		char o = '-';
		if (i == 'A')
			o = 'T';
		else if (i == 'C')
			o = 'G';
		else if (i == 'G')
			o = 'C';
		else if (i == 'T')
			o = 'A';
		
		return o;
	}
	
void refineFlowAlignment(int *flowIndexFwd, int *flowIndexRev,std::string &fwdFlowAligned, std::string &revFlowAligned, char *flow_seq_fwd, char *flow_seq_rev, uint16_t *flow_sig_fwd, uint16_t *flow_sig_rev)
{	
	//0): find out GAPS (>2) on foward flow and reverse flow separately
	//1): rearrange flow alignment  
		
	int flowLen = fwdFlowAligned.length(); //equals revFlowAligned.length();
		
	bool start_defined = false;
	bool stop_defined = false;
	
	int reg_start = 0;
	int reg_stop = 0;

	for (int i = 2; i < flowLen-2; i++) {
		if(fwdFlowAligned.at(i-1)!=GAP && fwdFlowAligned.at(i)==GAP && start_defined==false ){
			reg_start = i;
			start_defined = true;
		}
		if(fwdFlowAligned.at(i)==GAP && fwdFlowAligned.at(i+1)!=GAP && stop_defined ==false){
			reg_stop = i;
			stop_defined = true;
		}			
		if(start_defined && stop_defined && (reg_stop - reg_start)>=1 
		&& fwdFlowAligned.at(reg_start-2)!=GAP && fwdFlowAligned.at(reg_stop+2)!=GAP
		&& revFlowAligned.at(reg_start-2)!=GAP && revFlowAligned.at(reg_stop+2)!=GAP		
		&& revFlowAligned.at(reg_start-1)!=GAP && revFlowAligned.at(reg_stop+1)!=GAP){
			reg_start = reg_start - 2;
			reg_stop = reg_stop + 2;
			fwdFlowAligned = refineGapRegion(flowIndexFwd, flowIndexRev,fwdFlowAligned, revFlowAligned, flow_seq_fwd, flow_seq_rev, flow_sig_fwd, flow_sig_rev, reg_start, reg_stop, true);
			start_defined = false;
			stop_defined = false;
		}		
	}		
	
	start_defined = false;
	stop_defined = false;
	
	reg_start = 0;
	reg_stop = 0;

	for (int i = 2; i < flowLen-2; i++) {
		if(revFlowAligned.at(i-1)!=GAP && revFlowAligned.at(i)==GAP && start_defined==false ){
			reg_start = i;
			start_defined = true;
		}
		if(revFlowAligned.at(i)==GAP && revFlowAligned.at(i+1)!=GAP && stop_defined ==false){
			reg_stop = i;
			stop_defined = true;
		}			
		if(start_defined && stop_defined && (reg_stop - reg_start)>=1 
		&& revFlowAligned.at(reg_start-2)!=GAP && revFlowAligned.at(reg_stop+2)!=GAP
		&& fwdFlowAligned.at(reg_start-2)!=GAP && fwdFlowAligned.at(reg_stop+2)!=GAP		
		&& fwdFlowAligned.at(reg_start-1)!=GAP && fwdFlowAligned.at(reg_stop+1)!=GAP){
			reg_start = reg_start - 2;
			reg_stop = reg_stop + 2;		
			revFlowAligned = refineGapRegion(flowIndexFwd, flowIndexRev,fwdFlowAligned, revFlowAligned, flow_seq_fwd, flow_seq_rev, flow_sig_fwd, flow_sig_rev, reg_start, reg_stop, false);							
			start_defined = false;
			stop_defined = false;
		}		
	}		

}
	

//flowIndexFwd, flowIndexRev: flowIndexFwdCorrected, flowIndexRevCorrected, -10 means no match found 
//flowFwdAligned, flowRevAligned: result of flowAlignment = flowAligner->align(flowAlign, flowAlignRev, 0.5, 0.5, -2);
std::string refineGapRegion(int *flowIndexFwd, int *flowIndexRev,std::string &flowFwdAligned, std::string &flowRevAligned, char *flow_seq_fwd, char *flow_seq_rev, uint16_t *flow_sig_fwd, uint16_t *flow_sig_rev, int reg_start, int reg_stop, bool gaps_on_fwd)
{
//need to evaluate refinement results:
//invalid: flow index gap is smaller than alignment gap
	
	int flow_len = flowFwdAligned.length();
	
	std::string corrected_gap_flows;
	
	if(gaps_on_fwd){	
		int *flowIndexFwd_orig = new int[flow_len]; 
		memcpy(flowIndexFwd_orig, flowIndexFwd, flow_len*sizeof(int)); //copy value
				
		corrected_gap_flows = flowFwdAligned;
		int fwdFlowIndexStart = flowIndexFwd[reg_start];
		int fwdFlowIndexStop = flowIndexFwd[reg_stop];
		int k = reg_start;
		for(int i=fwdFlowIndexStart; i<=fwdFlowIndexStop; i++){		
			if(k>reg_stop){
				corrected_gap_flows = flowFwdAligned;
				break;
			}
			if(flow_seq_fwd[i-1]==flowRevAligned[k]){ //a hit from fwd_flow to rev_flow sequence
				flowIndexFwd[k] = i; 
				int num_hp = roundsff((float)flow_sig_fwd[i-1]/100);
				if(num_hp>=1) {
					corrected_gap_flows.replace(k, 1, flowRevAligned, k, 1); //a hit from called fwd_flow (large signal) to rev_flow sequence				
				}
				else corrected_gap_flows.replace(k, 1, "-"); //a hit from non-called fwd_flow (no signal or small signal) to rev_flow sequence
				k++;
			}
			else{//no hit, move to next fwd flow
				continue; //k does not change
				}				
			}
		//check if refinement is ok
		
		int fwd_ind_non_gap[100];
		int fwd_ind_non_gap_pos[100];
		int num_fwd_ind_non_gap = 0;
		
		for(int i = reg_start; i<= reg_stop; i++){
			if(flowIndexFwd[i]==-10) continue;
			fwd_ind_non_gap[num_fwd_ind_non_gap] = flowIndexFwd[i];
			fwd_ind_non_gap_pos[num_fwd_ind_non_gap] = i;
			num_fwd_ind_non_gap++;
		}		
		
		for(int i = 1; i < num_fwd_ind_non_gap; i++){
			if(fwd_ind_non_gap[i]-fwd_ind_non_gap[i-1] < fwd_ind_non_gap_pos[i]-fwd_ind_non_gap_pos[i-1]){
				corrected_gap_flows = flowFwdAligned; //refinement is not successful or not possible 
				memcpy(flowIndexFwd, flowIndexFwd_orig, flow_len*sizeof(int));
				break;				    		
				}
			}
		delete[] flowIndexFwd_orig;
						
		
		}else
		{ //gaps_on_reverse		
			
			int *flowIndexRev_orig = new int[flow_len]; 
			memcpy(flowIndexRev_orig, flowIndexRev, flow_len*sizeof(int)); //copy value
							
			corrected_gap_flows = flowRevAligned;
			int revFlowIndexStart = flowIndexRev[reg_start]; 
			int revFlowIndexStop = flowIndexRev[reg_stop];	
			
			int k = reg_start;
			for(int i=revFlowIndexStart; i>=revFlowIndexStop; i--){	
				if(k>reg_stop){
					corrected_gap_flows = flowRevAligned;
					break;
				}				
				if(complimentNuc(flow_seq_rev[i-1])==flowFwdAligned[k]){ //a hit from fwd_flow to rev_flow sequence
					flowIndexRev[k] = i; 
					int num_hp = roundsff((float)flow_sig_rev[i-1]/100);
					if(num_hp>=1) {
						corrected_gap_flows.replace(k, 1, flowFwdAligned, k, 1); //a hit from called fwd_flow (large signal) to rev_flow sequence				
					}
					else corrected_gap_flows.replace(k, 1, "-"); //a hit from non-called fwd_flow (no signal or small signal) to rev_flow sequence
					k++;
				}
				else{//no hit, move to next fwd flow
					continue; //k does not change
					}
				}
			
			//check if refinement is ok
			
			int rev_ind_non_gap[100];
			int rev_ind_non_gap_pos[100];
			int num_rev_ind_non_gap = 0;
			
			for(int i = reg_start; i<= reg_stop; i++){
				if(flowIndexRev[i]==-10) continue;
				rev_ind_non_gap[num_rev_ind_non_gap] = flowIndexRev[i];
				rev_ind_non_gap_pos[num_rev_ind_non_gap] = i;
				num_rev_ind_non_gap++;
			}		
			
			for(int i = 1; i < num_rev_ind_non_gap; i++){
				if(rev_ind_non_gap[i-1]-rev_ind_non_gap[i] < rev_ind_non_gap_pos[i]-rev_ind_non_gap_pos[i-1]){
					corrected_gap_flows = flowRevAligned; //refinement is not successful or not possible 
					memcpy(flowIndexRev, flowIndexRev_orig, flow_len*sizeof(int));
					break;				    		
					}
				}
				
			delete[] flowIndexRev_orig;
		}
	
	
	return corrected_gap_flows;
	
}

bool build_strand_bias_map(std::string strand_bias_file, std::tr1::unordered_map<std::string, std::set<int> > *bias_map){

    std::string read_name;
    int bias_pos;
    std::string ref_name;
    int flow_index;
    std::tr1::unordered_map<std::string,std::set<int> >::iterator read_flow_index_with_bias;
    std::ifstream fp_strand_bias;

    fp_strand_bias.open(strand_bias_file.c_str());
    std::string line;

    if (fp_strand_bias.fail()) {
        std::cerr << "strand bias file couldn't be opened" << std::endl;
        fp_strand_bias.close();
        return false;
    } else {
        while (fp_strand_bias.good()) {
            fp_strand_bias >> ref_name;
            fp_strand_bias >> bias_pos;
            fp_strand_bias >> read_name;
            fp_strand_bias >> flow_index;

            read_flow_index_with_bias = (*bias_map).find(read_name);

            if(read_flow_index_with_bias!=(*bias_map).end()){ //insert to an existing Set
                 std::set<int> myset;
                 myset = read_flow_index_with_bias->second;
                 myset.insert(flow_index);
                 (*bias_map)[read_name] = myset;
             }else{ //insert to a new Set
                 std::set<int> myset;
                 myset.clear();
                 myset.insert(flow_index);
                 (*bias_map).insert (std::make_pair<std::string,std::set<int> >(read_name,myset));
             }
           }
        }
        fp_strand_bias.close();    

/*
    std::tr1::unordered_map<std::string,std::set<int> >::const_iterator found_set;

    for(found_set=(*bias_map).begin(); found_set!=(*bias_map).end(); found_set++){

        std::set<int>::iterator it;

        if(found_set->second.size()>1){
            std::cout << "read " << found_set->first << " contains more than 1 biased positions: ";
            for(it=found_set->second.begin(); it!=found_set->second.end(); it++){
                std::cout << " " << *it;
            }

            std::cout << std::endl;
        }

    }
*/

    return true;

}

bool find_read_in_bias_map(std::string read_name, std::tr1::unordered_map<std::string, std::set<int> > *bias_map, std::set<int> *bias_flow_index_set){

    std::tr1::unordered_map<std::string,std::set<int> >::const_iterator found_set;
    found_set = (*bias_map).find (read_name);
    if(found_set!=(*bias_map).end()){
        *bias_flow_index_set = (*found_set).second;
        return true;
    }else{
        return false;
    }

}

bool find_flow_index_with_bias(std::set<int> bias_flow_index_set, int flow_index){

    std::set<int>::iterator it;
    it = bias_flow_index_set.find(flow_index);
    if(it!=bias_flow_index_set.end()){
        return true;
    }else{
        return false;
    }

}


