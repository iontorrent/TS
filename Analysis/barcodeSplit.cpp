/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/* vi: set noexpandtab ts=4 sw=4:
 * Stand alone app for development/testing of barcode matching in reads
 * Compile with
 * g++ -Wall -g -o barcodeSplit BarCode.cpp barcodeSplit.cpp -L file-io/ -ldat
 * Also need to munge the file-io Makefile to make libdat.a and not delete it.
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "IonVersion.h"
#include "BarCode.h"
#include "file-io/sff.h"
#include "file-io/fastq.h"
#include "file-io/sff_file.h"
#include "file-io/sff_read.h"
#include "file-io/sff_header.h"
#include "file-io/fastq_file.h"
#include "file-io/sff_read_header.h"

/*
 *	Match each barcode to a file descriptor
 */
typedef struct {
	char *name;
	int numReads;
	FILE *fd_fastq;
	FILE *fd_sff;
} bcTrack;

int showHelp ()
{
	fprintf (stdout, "barcodeSplit - Sort the reads in SFF file by barcode into fastq files\n");
	fprintf (stdout, "options:\n");
	fprintf (stdout, "   -b, --barcode-file\tSpecify file containing barcodes\n");
	fprintf (stdout, "   -d, --output-directory\tSpecify output directory to write to\n");
	fprintf (stdout, "   -f, --flow-order\tOverride default (TACG) flow order\n");
	fprintf (stdout, "   -h, --help\tShow command line options and exit\n");
	fprintf (stdout, "   -i, --input\tSpecify SFF file to sort\n");
	fprintf (stdout, "   -s, --sff\tCreate sff files in addition to fastq\n");
	fprintf (stdout, "   -m, --score-mode\tSet the score mode and threshold for barcode classification match in XvY format, default is 0v0.9 (score mode 0, threshold value 0.9)\n");	
	fprintf (stdout, "   -z, --score-hist-mode\tPrint mode for score hist. file.  0- off, 1- best bc, 2- all bc.\n");
	fprintf (stdout, "   -k, --bfmask-file\tBead find mask file used to find dimensions. Default: bfmask.bin\n");
	fprintf (stdout, "   -e, --bcmmask\tMakes bcmatch_mask.bin XY mask file containing number of errors, also makes a separate mask file bcmask.bin of barcode ids.\n");
	fprintf (stdout, "   -c, --bcmask-file\tMakes this file containing barcode ids in an XY mask format. If not specified, no file will be produced.\n");
	fprintf (stdout, "   -v, --version\tShow version and exit\n");
	fprintf (stdout, "\n");
	fprintf (stdout, "usage:\n");
	fprintf (stdout, "   barcodeSplit -i sff_filename -b barcode_filename\n");
	fprintf (stdout, "\n");
	exit (EXIT_SUCCESS);
}

char *makeOutputFile (char *dir, char *id, char *runid, char *ext)
{
	char *pathname = NULL;
	char *directory = NULL;
	
	if (dir == NULL) {
		directory = strdup ("./");
	}
	else {
		//make sure there is a trailing delimiter on dir
		if (NULL == (directory = (char *) malloc (strlen(dir)+2))){
			fprintf (stderr, "%s", strerror(errno));
			exit (EXIT_FAILURE);
		}
		strcpy (directory, dir);
		if (directory[strlen(directory)] != '/') {
			strcat(directory, "/");
		}
	}
	
	pathname = (char *) malloc (strlen(directory)+strlen(id)+strlen(runid)+strlen(ext)+3);
	sprintf (pathname, "%s%s_%s.%s", directory, id, runid, ext);
	free (directory);
	return (pathname);
}
/*
 *	Main function
 */
int main (int argc, char *argv[])
{
	char *sff_filename	= NULL;
	char *bc_filename	= NULL;
	char *flowOrder		= strdup ("TACG");
	char *runId		= strdup ("thisRun");
	char *outputDir		= NULL;
	char *fastq_ext		= strdup ("fastq");
	char *sff_ext		= strdup ("sff");
	char *nomatch		= strdup ("nomatch");
	bool make_sff		= false;
	bool rtbug		= false;
	int scoreMode		= -1; // -1 = not set, 0 is percent match-based, 1 is # flows-based, 2 is # flows-based with weighting to break ties
	double	scoreCutoff	= -1.0; // not set
	std::string scoreHistFn = "";
	int scoreHistPrintMode	= 1;
	char *bfmask_filename	= strdup ("bfmask.bin"); //To get flowcell's dimensions
	char *bcmask_filename	= NULL; //strdup ("bcmask.bin"); //Output barcode id "mask" file.
	char *bcmmask_filename	= NULL;

	/*
	 *	Command line opions parsing
	 */
	int c;
	int option_index = 0;
	static struct option long_options[] =
		{
			{"barcode-file",			required_argument,	NULL,	'b'},
			{"output-directory",		required_argument,	NULL,	'd'},
			{"input",					required_argument,	NULL,	'i'},
			{"version",					no_argument,		NULL,	'v'},
			{"flow-order",				required_argument,	NULL,	'f'},
			{"sff",						no_argument,		NULL,	's'},
			{"score-mode",					required_argument,	NULL,	'm'},
			{"score-hist-mode",         required_argument,  NULL,   'z'},
			{"bfmask-file",             required_argument,  NULL,   'k'},
			{"bcmmask",             no_argument,  NULL,   'e'},
			{"bcmask-file",             required_argument,  NULL,   'c'},
			{"help",					no_argument,		NULL,	'h'},
			{"debug",					no_argument,		NULL,	'x'},
			{NULL, 0, NULL, 0}
		};
		
	while ((c = getopt_long (argc, argv, "b:d:f:m:z:k:c:hi:esvx", long_options, &option_index)) != -1)
	{
		switch (c)
		{
            case (0):
                if (long_options[option_index].flag != 0)
                    break;
				        break;			
            case 'b': // barcode file
	      			bc_filename = strdup (optarg);
	      			break;
			
            case 'd': // output directory
      				outputDir = strdup (optarg);
	      			break;            
            case 'z':{ //histogram file
            	scoreHistPrintMode = atoi(strdup (optarg));
            	scoreHistFn = "scores.txt";
            	}
            	break;
            case 'k':
            	bfmask_filename = strdup (optarg);
            	break;
            case 'c':
            	bcmask_filename = strdup (optarg);
            	break;
            case 'e':{
            	bcmmask_filename = "bcmatch_mask.bin";
            	bcmask_filename = "bcmask.bin";
            	}
            	break;	
           	case 'm': {// deprecated - use the database and set the barcode score method and cutoff values
            	char *sPtr = strchr(optarg,'v');
              if (sPtr){
                int ret = sscanf (optarg, "%dv%lf", &scoreMode, &scoreCutoff);
                if (ret < 2){
            	    scoreMode = 0; 
            	    scoreCutoff = 0.9;
            		  fprintf(stderr, "ERROR!  invalid score mode, using default\n");
            		}
              }
            	else {
            	  int ret = sscanf (optarg, "%d", &scoreMode);
            	  if (ret < 1){
            	    scoreMode = 0;
            	    scoreCutoff = 0.9; 
            		  fprintf(stderr, "ERROR!  invalid score mode, using default\n");
            	} else            	
            		  // set mode defaults
            		  if (scoreMode==0) scoreCutoff = 0.9;
            	    else if (scoreMode==1) scoreCutoff = 2;
              } 
              } break;

            case 'f': // nuke flow order string
				free (flowOrder);
				flowOrder = strdup (optarg);
				break;
			
			case 'h': // show help
				showHelp ();
				exit (EXIT_SUCCESS);
				break;
			
			case 'i':	// input file
				sff_filename = strdup (optarg);
				break;
			
			case 's':	// create sff files output
				make_sff = true;
				break;
			
			case 'v':   //version
                fprintf (stdout, "%s", IonVersion::GetFullVersion("barcodeSplit").c_str());
                //fprintf (stdout, "%s %s\n", argv[0], "0.1.0");
				exit (EXIT_SUCCESS);
                break;
			
			case 'x':	//enable debug printing
				rtbug = true;
				break;
			
			default:
                fprintf (stderr, "What have we here? (%c)\n", c);
				exit (EXIT_FAILURE);
		}
	}
	
	// test for required inputs
	if (!sff_filename || !bc_filename) {
		showHelp ();
		fprintf (stdout, "\nError: The -i and -b options are required\n\n");
		exit (EXIT_FAILURE);
	}
	
	// create barcode object from given barcode file
	barcode BC; // = barcode ();
	// set debug flag
	BC.SetRTDebug (rtbug);
	if (BC.ReadBarCodeFile (bc_filename)) {
		fprintf (stderr, "Error opening barcode file: %s\n", bc_filename);
		exit (EXIT_FAILURE);
	}
	
	// Load SFF file
	FILE *sff_file = NULL;
	if (NULL == (sff_file = fopen(sff_filename, "rb"))){
		fprintf (stderr, "Error opening sff file: %s\n", sff_filename);
		exit (EXIT_FAILURE);
	}
	
	// Setup barcode mask coordinates by reading in bead finding's mask file
	if(bcmask_filename!=NULL || bcmmask_filename!=NULL)
		BC.SetupBarcodeMask(bfmask_filename);
	
	// Create runId to use for output filenames
	char *sPtr = strdup (sff_filename);
	char *endPtr = NULL;
	if (NULL != (endPtr = strrchr (sPtr, '.'))){
		*endPtr = '\0';
		free (runId);
		runId = strdup (basename (sPtr));
	}
	if (sPtr) free (sPtr);
	
	// If output directory specified, create it first
	if (outputDir != NULL) {
		if (mkdir(outputDir, 0777)) {
			if (errno == EEXIST) {
				//already exists? well okay...
			} else {
				perror(outputDir);
				exit(EXIT_FAILURE);
			}
		}
	}
	// Setup score histogram file
	if (scoreHistFn!="") {
		std::string outputDirStr = ".";
		if(outputDir!=NULL) outputDirStr = outputDir;
		BC.SetupScoreHistStrm(outputDirStr+"/"+scoreHistFn,scoreHistPrintMode);
	}
	
	// For each barcode, store a file descriptor for SFF and FASTQ files
	FILE *defaultfd_fastq = NULL;		// fastq file for unmatched reads
	FILE *defaultfd_sff = NULL;			// sff file for unmatched reads
	int defaultsff_numReads = 0;
	int num_barcodes = BC.GetNumBarcodes();
	bcTrack *bcTracker = (bcTrack *) malloc (num_barcodes * sizeof (bcTrack));
	for (int i = 0; i < num_barcodes; i++) {
		bcTracker[i].name = strdup (BC.GetBarcode(i));
		bcTracker[i].numReads = 0;
		bcTracker[i].fd_fastq = NULL;
		bcTracker[i].fd_sff = NULL;
	}
	
	// read global header to get number of reads in file
	int num_reads = 0;
	sff_header_t *gh = sff_header_read (sff_file);
	num_reads = gh->n_reads;

	// Create new SFF with left trim value adjusted to clip barcode out
	sff_header_t *new_header = sff_header_clone(gh);
	char *outsff = makeOutputFile (outputDir, "bctrimmed", runId, sff_ext);
	sff_file_t *sff_file_out = sff_fopen(outsff, "wb", new_header, NULL );
	free (outsff);
	
	// Get key sequence length from SFF and store it in barcode object
	BC.SetKey(gh->key->s, (int)gh->key_length);

	// save the flow order
	BC.SetFlowOrder(flowOrder);

	// override default score mode and cutoff from database
	if (scoreMode != -1) {
		BC.SetScoreMode(scoreMode);
		BC.SetScoreCutoff(scoreCutoff);
	}

	// sequentially read SFF for all reads
	sff_t *sff_read = NULL;
	int cnt_match = 0;
		
	for (int i = 0; i < num_reads; i++) {
		//fprintf (stdout, "Read # %d\n", i);
        if (NULL == (sff_read = sff_read1 (sff_file, gh))) {
			fprintf (stderr, "Error reading read from SFF file\n");
			exit (EXIT_FAILURE);
		}
		// for this read, do barcode scan
		bcmatch *bcm;
		if (1) // BC.GetNumBarcodes() == 1) // MGD - always run flowSpaceTrim
			bcm = BC.flowSpaceTrim (sff_read->read->flowgram, sff_read->gheader->flow_length,
sff_read->rheader->name->s);
		else
			bcm = BC.exactMatch (sff_bases(sff_read));

		if (bcm != NULL) {

			cnt_match++;

			// edit the sff clip value to remove the barcode
			// N.B. BarCode returns zero-based offset value. SFF Uses 1-based index for clips
			// Left clip must be equal to or less than right clip; unless its zero
			unsigned int clip_limit = 0;
			unsigned int clip = 0;
			//if both right clips are 0, then clip limit is total number of bases
			if ((sff_read)->rheader->clip_adapter_right == 0 &&
				(sff_read)->rheader->clip_qual_right == 0){

				clip_limit = (sff_read)->rheader->n_bases;
			}
			//if one of the right clips is non-zero, then clip limit is the non-zero number
			else if ((sff_read)->rheader->clip_adapter_right == 0 ||
					 (sff_read)->rheader->clip_qual_right == 0){

				clip_limit = max((sff_read)->rheader->clip_adapter_right,
								 (sff_read)->rheader->clip_qual_right);
			}
			//if both of the right clips are non-zero, then clip limit is lesser number
			else if ((sff_read)->rheader->clip_adapter_right > 0 &&
					 (sff_read)->rheader->clip_qual_right > 0){

				clip_limit = min((sff_read)->rheader->clip_adapter_right,
								 (sff_read)->rheader->clip_qual_right);
			}

			//limit left clip value - *** 1-based OFFSET APPLIED HERE to bc_right ***
			// and clip_limit - as left clip can be 1bp after right clip (indicating 0-mer)
			clip = ((bcm->bc_right+1) <= (clip_limit+1) ? (bcm->bc_right+1):(clip_limit+1));

			//store left clip value in sff read
			(sff_read)->rheader->clip_adapter_left = clip;
			
			// Record the sff read with the new left trim value in the composite SFF output file
			sff_t *out_sff = sff_clone(sff_read);
			sff_write(sff_file_out, out_sff);
			sff_destroy(out_sff);
			
			for (int j = 0; j < num_barcodes; j++) {
				//Note, j is NOT the barcode id from barcodeList.txt, it's an internal value.
				//That is found in BarcodeEntry's barcodeIndex.
				if (strcmp (bcm->matching_code, bcTracker[j].name) == 0) {
					if (bcmask_filename != NULL)
						BC.SetMask(sff_read->rheader->name->s,j);
					if (bcmmask_filename != NULL)
						BC.SetMatchMask(sff_read->rheader->name->s,bcm->errors);					
					if (make_sff) {
						// SFF File output
						// open file descriptor
						if (NULL == bcTracker[j].fd_sff) {
							char *filename = makeOutputFile (outputDir, bcm->id_str, runId, sff_ext);
							bcTracker[j].fd_sff = fopen(filename, "wb");
							free (filename);
							
							// initialize global header count to 0
							sff_header_t *th = sff_header_clone(gh);
							th->n_reads = 0;
							// write global header
							sff_header_write (bcTracker[j].fd_sff, th);
							sff_header_destroy(th);
						}
						
						sff_write1(bcTracker[j].fd_sff, sff_read);
					}
					
					bcTracker[j].numReads += 1;
					
					//TODO: we could use a function that updates the n_reads in the existing open file
					//by remembering the fp position, goingto the header, making the edit, and restoring fp
					//Until then, we only update n_reads at end of processing.
					
					// FASTQ File output
					// create fastq from sff read
					fq_t *fq = sff2fq (sff_read);
					if (NULL == fq) exit (EXIT_FAILURE);
					
					if(fq->l==0) {
						fq_destroy (fq);
						break; //check 0 length read
					}
					// open file descriptor
					if (NULL == bcTracker[j].fd_fastq) {
						char *filename = makeOutputFile (outputDir, bcm->id_str, runId, fastq_ext);
						bcTracker[j].fd_fastq = fq_file_open(filename);
						free (filename);
					}
					fq_write (bcTracker[j].fd_fastq, fq);
					fq_destroy (fq);

					break;
				}
			}
		}
		else {
		  
			// This read does not match any barcode
			if(bcmask_filename!=NULL)
				BC.SetMask(sff_read->rheader->name->s,-1);  //-1 will become barcodeId 0

			// Record the read in the SFF output file
			sff_t *out_sff = sff_clone(sff_read);
			sff_write(sff_file_out, out_sff);
			sff_destroy(out_sff);
			
			if (make_sff) {
				//TODO: save this sff read in output sff file
				// SFF File output
				// open file descriptor
				if (NULL == defaultfd_sff) {
					char *filename = makeOutputFile (outputDir, nomatch, runId, sff_ext);
					defaultfd_sff = fopen(filename, "wb");
					free (filename);
					
					// initialize global header count to 0
					sff_header_t *init_hd = sff_header_clone(gh);
					init_hd->n_reads = 0;
					// write global header
					sff_header_write (defaultfd_sff, init_hd);
				}
			
				sff_write1(defaultfd_sff, sff_read);
			}
			
			defaultsff_numReads += 1;
			
			// create fastq from sff read
			fq_t *fq = sff2fq (sff_read);
			if (NULL == fq) exit (EXIT_FAILURE);

			if(fq->l!=0) { //check 0 length read
				if (!defaultfd_fastq) {
					char *filename = makeOutputFile (outputDir, nomatch, runId, fastq_ext);
					defaultfd_fastq = fq_file_open(filename);
					free (filename);
				}
				fq_write (defaultfd_fastq, fq);
				fq_destroy (fq);
			}
		}
		
		// cleanup
		sff_destroy (sff_read);
		BC.bcmatch_destroy (bcm);
	}
  
  printf("barcodeSplit: processed %d reads, matched %d, unmatched %d\n",num_reads,cnt_match,num_reads-cnt_match);	

	sff_fclose(sff_file_out);
	sff_header_destroy (new_header);
	
	// set number of reads in global header for each SFF file
	if (make_sff) {
		for (int i = 0; i < num_barcodes;i++){
			if (bcTracker[i].fd_sff) {
				fseek(bcTracker[i].fd_sff, 0, SEEK_SET);
				gh->n_reads = bcTracker[i].numReads;	// reusing the global header
				sff_header_write(bcTracker[i].fd_sff,gh);
			}
		}
		if (defaultfd_sff) {
			fseek(defaultfd_sff, 0, SEEK_SET);
			gh->n_reads = defaultsff_numReads;	// reusing the global header
			sff_header_write(defaultfd_sff,gh);
		}
	}

	if (bcmask_filename!=NULL) {
		BC.OutputRawMask(bcmask_filename);
	}
	if (bcmmask_filename!=NULL) {
		BC.OutputRawMask2(bcmmask_filename);
	}

	//
	// cleanup and exit
	sff_header_destroy (gh);
	for (int i = 0; i < num_barcodes;i++){
		free (bcTracker[i].name);
		if (bcTracker[i].fd_fastq) fclose(bcTracker[i].fd_fastq);
		if (bcTracker[i].fd_sff) fclose(bcTracker[i].fd_sff);
	}
	if (defaultfd_fastq) fclose (defaultfd_fastq);
	if (defaultfd_sff) fclose (defaultfd_sff);
	free (bcTracker);
	free (flowOrder);
	fclose(sff_file);
	free (sff_filename);
	free (bc_filename);
    free (fastq_ext);
    free (sff_ext);
    free (nomatch);
	if (outputDir) free (outputDir);
	free (runId);
	if (rtbug)
		BC.DumpResiduals();
	BC.BCclose();
	
	exit (EXIT_SUCCESS);
}
