#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <config.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include "../bfast/BLibDefinitions.h"
#include "../bfast/BError.h"
#include "../bfast/BLib.h"
#include "../bfast/aflib.h"

#define Name "solid2fastq"

typedef struct {
	int32_t to_print; // whether this entry should be printed (must be populated)
	int32_t is_pop; // whether this entry has been populated
	char name[SEQUENCE_NAME_LENGTH];
	char read[SEQUENCE_LENGTH];
	char qual[SEQUENCE_LENGTH];
} fastq_t;

AFILE *open_output_file(char*, int32_t, int32_t, int32_t);
void fastq_print(fastq_t*, AFILE*);
void fastq_read(fastq_t*, AFILE*, AFILE*, int32_t);
int32_t cmp_read_names(char*, char*);
void read_name_trim(char*);
char *strtok_mod(char*, char*, int32_t*);
int32_t read_line(AFILE *afp, char *line);

int print_usage ()
{
	fprintf(stderr, "solid2fastq %s\n", PACKAGE_VERSION);
	fprintf(stderr, "\n");
	fprintf(stderr, "Usage: solid2fastq [options] <list of .csfasta files> <list of .qual files>\n");
	fprintf(stderr, "\t-c\t\tproduce no output.\n");
	fprintf(stderr, "\t-n\tINT\tnumber of reads per file.\n");
	fprintf(stderr, "\t-o\tSTRING\toutput prefix.\n");
	fprintf(stderr, "\t-j\t\tinput files are bzip2 compressed.\n");
	fprintf(stderr, "\t-z\t\tinput files are gzip compressed.\n");
	fprintf(stderr, "\t-J\t\toutput files are bzip2 compressed.\n");
	fprintf(stderr, "\t-Z\t\toutput files are gzip compressed.\n");
	fprintf(stderr, "\t-t\tINT\ttrim INT bases from the 3' end of the reads.\n");
	fprintf(stderr, "\t-h\t\tprint this help message.\n");
	fprintf(stderr, "\n send bugs to %s\n", PACKAGE_BUGREPORT);
	return 1;
}

int main(int argc, char *argv[])
{
	char *output_prefix=NULL;
	int32_t num_reads_per_file=-1;
	int32_t no_output=0;
	int32_t number_of_ends;
	int32_t num_ends_printed = 0;
	int64_t *end_counts=NULL;
	char **csfasta_filenames=NULL;
	char **qual_filenames=NULL;
	AFILE **afps_csfasta=NULL;
	AFILE **afps_qual=NULL;
	AFILE *afp_output=NULL;
	int32_t in_comp=AFILE_NO_COMPRESSION;
	int32_t out_comp=AFILE_NO_COMPRESSION;
	int32_t output_suffix_number;
	int c;
	int32_t i, j;
	int32_t trim_end = 0;
	fastq_t *reads=NULL;
	int32_t more_afps_left=1;
	int64_t output_count=0;
	int64_t output_count_total=0;
	char *min_read_name=NULL;

	// Get Parameters
	while((c = getopt(argc, argv, "n:o:t:chjzJZ")) >= 0) {
		switch(c) {
			case 'c':
				no_output=1; break;
			case 'h':
				return print_usage(); break;
			case 'j':
				in_comp=AFILE_BZ2_COMPRESSION; break;
			case 'n':
				num_reads_per_file=atoi(optarg); break;
				break;
			case 'o':
				output_prefix=strdup(optarg); break;
				break;
			case 't':
				trim_end=atoi(optarg); break;
			case 'z':
				in_comp=AFILE_GZ_COMPRESSION; break;
			case 'J':
				out_comp=AFILE_BZ2_COMPRESSION; break;
			case 'Z':
				out_comp=AFILE_GZ_COMPRESSION; break;
			default: fprintf(stderr, "Unrecognized option: -%c\n", c); return 1;
		}
	}

	// Print Usage
	if(argc == optind ||
			0 != ((argc - optind) % 2) ||
			(argc - optind) < 2) {
		return print_usage();
	}

	// Copy over the filenames
	assert(0 == (argc - optind) % 2);
	number_of_ends = (argc - optind) / 2;
	// Allocate memory
	csfasta_filenames = malloc(sizeof(char*)*number_of_ends);
	if(NULL == csfasta_filenames) {
		PrintError(Name, "csfasta_filenames", "Could not allocate memory", Exit, MallocMemory);
	}
	qual_filenames = malloc(sizeof(char*)*number_of_ends);
	if(NULL == qual_filenames) {
		PrintError(Name, "qual_filenames", "Could not allocate memory", Exit, MallocMemory);
	}
	end_counts = malloc(sizeof(int64_t)*number_of_ends);
	if(NULL == end_counts) {
		PrintError(Name, "end_counts", "Could not allocate memory", Exit, MallocMemory);
	}
	for(i=0;i<number_of_ends;i++) {
		csfasta_filenames[i] = strdup(argv[optind+i]);
		qual_filenames[i] = strdup(argv[optind+i+number_of_ends]);
		end_counts[i] = 0;
	}

	// Allocate memory for input file pointers
	afps_csfasta = malloc(sizeof(AFILE*)*number_of_ends);
	if(NULL == afps_csfasta) {
		PrintError(Name, "afps_csfasta", "Could not allocate memory", Exit, MallocMemory);
	}
	afps_qual = malloc(sizeof(AFILE*)*number_of_ends);
	if(NULL == afps_qual) {
		PrintError(Name, "afps_qual", "Could not allocate memory", Exit, MallocMemory);
	}

	// Open input files
	for(i=0;i<number_of_ends;i++) {
		if(!(afps_csfasta[i] = AFILE_afopen(csfasta_filenames[i], "rb", in_comp))) {
			PrintError(Name, csfasta_filenames[i], "Could not open file for reading", Exit, OpenFileError);
		}
		if(!(afps_qual[i] = AFILE_afopen(qual_filenames[i], "rb", in_comp))) {
			PrintError(Name, qual_filenames[i], "Could not open file for reading", Exit, OpenFileError);
		}
	}

	reads = malloc(sizeof(fastq_t)*number_of_ends);
	if(NULL == reads) {
		PrintError(Name, "reads", "Could not allocate memory", Exit, MallocMemory);
	}

	for(i=0;i<number_of_ends;i++) {
		reads[i].to_print = 0;
		reads[i].is_pop = 0;
	}

	output_suffix_number = 1;
	more_afps_left=number_of_ends;
	output_count = output_count_total = 0;
	// Open output file
	if(NULL == output_prefix) {
		if(!(afp_output=AFILE_afdopen(fileno(stdout), "wb", out_comp))) {
			PrintError(Name, "stdout", "Could not open for writing", Exit, WriteFileError);
		}
	}
	else if(0 == no_output) {
		afp_output = open_output_file(output_prefix, output_suffix_number, num_reads_per_file, out_comp); 
	}
	fprintf(stderr, "Outputting, currently on:\n0");
	while(0 < more_afps_left) { // while an input file is still open

		if(0 == (output_count_total % 100000)) {
			fprintf(stderr, "\r%lld", (long long int)output_count_total);
		}
		/*
		   fprintf(stderr, "more_afps_left=%d\n", more_afps_left);
		   */
		// Get all with min read name
		for(i=0;i<number_of_ends;i++) {
			// populate read if necessary
			if(0 == reads[i].is_pop &&
					NULL != afps_csfasta[i] &&
					NULL != afps_qual[i]) {
				fastq_read(&reads[i], afps_csfasta[i], afps_qual[i], trim_end); // Get read name
				if(0 == reads[i].is_pop) { // was not populated
					//fprintf(stderr, "EOF\n");
					AFILE_afclose(afps_csfasta[i]);
					AFILE_afclose(afps_qual[i]);
					afps_csfasta[i] = afps_qual[i] = NULL;
					reads[i].to_print = 0;
				}
			}
			else {
				reads[i].to_print = 0;
			}
			// check if the read name is the min
			if(1 == reads[i].is_pop) {
				/*
				   fprintf(stdout, "i=%d\tmin_read_name=%s\treads[i].name=%s\n", i, min_read_name, reads[i].name);
				   */
				if(NULL == min_read_name || 
						0 == cmp_read_names(reads[i].name, min_read_name)) {
					if(NULL == min_read_name) {
						min_read_name = strdup(reads[i].name);
					}
					reads[i].to_print = 1;
				}
				else if(cmp_read_names(reads[i].name, min_read_name) < 0) {
					free(min_read_name);
					min_read_name = strdup(reads[i].name);
					// Re-initialize other fps
					for(j=0;j<i;j++) {
						reads[j].to_print = 0;
					}
					reads[i].to_print = 1;
				}
				else {
					reads[i].to_print = 0;
				}
			}
			else {
				assert(0 == reads[i].to_print);
			}
		}
		/*
		   fprintf(stdout, "min_read_name was %s\n", min_read_name);
		   */
		free(min_read_name);
		min_read_name=NULL;

		// Print all with min read name
		more_afps_left = 0;
		num_ends_printed = 0;
		for(i=0;i<number_of_ends;i++) {
			if(1 == reads[i].to_print) {
				more_afps_left++;
				num_ends_printed++;
				if(0 == no_output) {
					fastq_print(&reads[i], afp_output);
				}
				reads[i].is_pop = reads[i].to_print = 0;
			}
		}
		if(0 < num_ends_printed) {
			end_counts[num_ends_printed-1]++;
			// Update counts
			output_count++;
			output_count_total++;
		}
		// Open a new output file if necessary
		if(0 < num_reads_per_file &&
				num_reads_per_file <= output_count) {
			output_suffix_number++;
			if(0 == no_output && NULL != output_prefix) {
				AFILE_afclose(afp_output);
				afp_output = open_output_file(output_prefix, output_suffix_number, num_reads_per_file, out_comp); 
			}
			output_count=0;
		}
	}
	if(0 < output_count && 0 == no_output) {
		AFILE_afclose(afp_output);
	}

	// Remove last fastq file when total input reads % num_reads_per_file == 0
	// We don't want an empty file
	if(0 == output_count && 0 == no_output) {
		char empty_fn[4096]="\0";
		assert(0 < sprintf(empty_fn, "%s.%d.fastq", output_prefix, output_suffix_number));
		if(remove(empty_fn)) { 
			PrintError(Name, "empty_fn", "Cannot remove file", Exit, DeleteFileError);
		}
	}

	fprintf(stderr, "\r%lld\n", (long long int)output_count_total);
	fprintf(stderr, "Found\n%16s\t%16s\n", "number_of_ends", "number_of_reads");
	for(i=0;i<number_of_ends;i++) {
		fprintf(stderr, "%16d\t%16lld\n", i+1, (long long int)end_counts[i]);
	}

	// Free
	if(NULL != output_prefix) free(output_prefix);
	for(i=0;i<number_of_ends;i++) {
		free(csfasta_filenames[i]);
		free(qual_filenames[i]);
	}
	free(csfasta_filenames);
	free(qual_filenames);
	free(end_counts);
	free(afps_csfasta);
	free(afps_qual);
	free(reads);

	return 0;
}

AFILE *open_output_file(char *output_prefix, int32_t output_suffix_number, int32_t num_reads_per_file, int32_t out_comp)
{
	char *FnName="open_output_file";
	char output_filename[4096]="\0";
	AFILE *afp_out=NULL;

	// Create output file name
	if(0 < num_reads_per_file) {
		assert(0 < sprintf(output_filename, "%s.%d.fastq", output_prefix, output_suffix_number));
	}
	else {
		assert(0 < sprintf(output_filename, "%s.fastq", output_prefix));
	}

	// Only if compression is used
	switch(out_comp) {
		case AFILE_GZ_COMPRESSION:
			strcat(output_filename, ".gz"); break;
		case AFILE_BZ2_COMPRESSION:
			strcat(output_filename, ".bz2"); break;
		default: 
			break;
	}

	// Open an output file
	if(!(afp_out = AFILE_afopen(output_filename, "wb", out_comp))) {
		PrintError(FnName, output_filename, "Could not open file for writing", Exit, OpenFileError);
	}

	return afp_out;
}

void fastq_print(fastq_t *read, AFILE *afp_output)
{
	//char *FnName="fastq_print";
	int32_t i;
	char at = '@';
	char plus = '+';
	char new_line = '\n';

	// Name
	AFILE_afwrite(&at, sizeof(char), 1, afp_output);
	for(i=0;i<strlen(read->name);i++) {
		AFILE_afwrite(&read->name[i], sizeof(char), 1, afp_output);
	}
	AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);

	// Sequence
	for(i=0;i<strlen(read->read);i++) {
		AFILE_afwrite(&read->read[i], sizeof(char), 1, afp_output);
	}
	AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);

	// Comment
	AFILE_afwrite(&plus, sizeof(char), 1, afp_output);
	AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);

	// Quality
	for(i=0;i<strlen(read->qual);i++) {
		AFILE_afwrite(&read->qual[i], sizeof(char), 1, afp_output);
	}
	AFILE_afwrite(&new_line, sizeof(char), 1, afp_output);
}

void fastq_read(fastq_t *read, AFILE *afp_csfasta, AFILE *afp_qual, int32_t trim_end)
{
	char *FnName="fastq_read";
	char qual_name[SEQUENCE_NAME_LENGTH]="\0";
	char qual_line[SEQUENCE_NAME_LENGTH]="\0";
	int32_t qual[SEQUENCE_LENGTH];
	int32_t i;
	char *pch=NULL, *saveptr=NULL;

	assert(0 == read->is_pop);
	assert(0 == read->to_print);

	if(NULL == afp_csfasta &&
			NULL == afp_qual) {
		//fprintf(stderr, "return 1\n"); // HERE
		return;
	}

	// Read in
	if(read_line(afp_csfasta, read->name) < 0 || 
			read_line(afp_csfasta, read->read) < 0 ||
			read_line(afp_qual, qual_name) < 0 ||
			read_line(afp_qual, qual_line) < 0) {
		//fprintf(stderr, "return 2\n"); // HERE
		return;
	}
	StringTrimWhiteSpace(read->name);
	StringTrimWhiteSpace(read->read);
	StringTrimWhiteSpace(qual_name);
	StringTrimWhiteSpace(qual_line);

	// Parse qual line
	pch = strtok_r(qual_line, " ", &saveptr);
	for(i=0;pch!=NULL;i++) {
		qual[i] = atoi(pch);
		pch = strtok_r(NULL, " ", &saveptr);
	}

	// Check that the read name and csfasta name match
	if(0 != strcmp(read->name, qual_name)) {
		fprintf(stderr, "read->name=%s\nqual_name=%s\n", read->name, qual_name);
		PrintError(FnName, "read->name != qual_name", "Read names did not match", Exit, OutOfRange);
	}
	// Remove leading '@' from the read name
	for(i=1;i<strlen(read->name);i++) {
		read->name[i-1] = read->name[i];
	}
	read->name[i-1]='\0';

	// Convert SOLiD qualities
	for(i=0;i<strlen(read->read)-1;i++) {
		/*
		   fprintf(stderr, "%c -> %c\n%d -> %d\n", qual[i], (qual[i] <= 93 ? qual[i] : 93) + 33, qual[i], (qual[i] <= 93 ? qual[i] : 93) + 33);
		   exit(1);
		   */
		qual[i] = 33 + (qual[i] <= 0 ? 0 : (qual[i] <= 93 ? qual[i] : 93));
		read->qual[i] = (char)qual[i];
	}

	// Trim last _R3 or _F3 or _whatever
	read_name_trim(read->name);

	// Trim last few colors
	if(0 < trim_end) {
		if(strlen(read->read) + 1 <= trim_end) {
			PrintError(FnName, "-t", "Trimming all the colors", Exit, OutOfRange);
		}
		read->read[strlen(read->read)-trim_end]='\0';
		read->qual[strlen(read->qual)-trim_end]='\0';
	}

	read->is_pop = 1;
}

int32_t cmp_read_names(char *name_one, char *name_two)
{
	char *name_one_cur = NULL;
	char *name_two_cur = NULL;
	int32_t name_one_num_state = 0, name_two_num_state = 0;
	int32_t return_value = 0;
	int32_t name_one_index = 0, name_two_index = 0;
	/*
	   fprintf(stderr, "comparing %s with %s\n", name_one, name_two);
	   */

	name_one_cur = strtok_mod(name_one, "_", &name_one_index);
	name_two_cur = strtok_mod(name_two, "_", &name_two_index);

	while(NULL != name_one_cur && NULL != name_two_cur) {
		/*
		   fprintf(stderr, "name_one_cur=%s\nname_two_cur=%s\n", name_one_cur, name_two_cur);
		   */
		// assumes positive
		name_one_num_state = ( name_one_cur[0] < '0' || name_one_cur[0] > '9') ? 0 : 1;
		name_two_num_state = ( name_two_cur[0] < '0' || name_two_cur[0] > '9') ? 0 : 1;
		if(1 == name_one_num_state && 1 == name_two_num_state) {
			name_one_num_state = atoi(name_one_cur);
			name_two_num_state = atoi(name_two_cur);
			return_value = (name_one_num_state < name_two_num_state) ? -1 : ((name_one_num_state == name_two_num_state) ? 0 : 1); 
		}
		else {
			return_value = strcmp(name_one_cur, name_two_cur);
		}
		/*
		   fprintf(stderr, "return_value=%d\n", return_value);
		   */
		if(0 != return_value) {
			free(name_one_cur);
			free(name_two_cur);
			return return_value;
		}

		// Get next tokens
		free(name_one_cur);
		free(name_two_cur);
		name_one_cur = strtok_mod(name_one, "_", &name_one_index);
		name_two_cur = strtok_mod(name_two, "_", &name_two_index);
	}

	free(name_one_cur);
	free(name_two_cur);

	if(NULL != name_one_cur && NULL == name_two_cur) {
		return 1;
	}
	else if(NULL == name_one_cur && NULL != name_two_cur) {
		return -1;
	}
	else {
		return 0;
	}
}

void read_name_trim(char *name)
{
	int32_t l;

	// Trim last _R3 or _F3 

	if(NULL == name) {
		return;
	}
	l=strlen(name);
	if(3 < l &&
			name[l-3]=='_' &&
			(name[l-2]=='F' || name[l-2]=='R') &&
			name[l-1]=='3') {
		name[l-3]='\0';
	}
	assert('_' != name[0]);
}

char *strtok_mod(char *str, char *delim, int32_t *index)
{
	int32_t i, prev_index=(*index);
	char *r=strdup(str + (*index));

	assert(NULL != str);

	while((*index) < strlen(str)) {
		for(i=0;i<strlen(delim);i++) {
			if(delim[i] == str[(*index)]) {
				r[(*index)-prev_index]='\0';
				(*index)++;
				return r;
			}
		}
		(*index)++;
	}

	if(prev_index == (*index)) {
		free(r);
		return NULL;
	}
	else {
		return r;
	}
}

int32_t read_line(AFILE *afp, char *line)
{
	//char *FnName="read_line";
	char c=0;
	int32_t i=0, p=0;

	int32_t state=0;

	if(NULL == afp) return 0;

	// States:
	// 0 - no characteres in line
	// 1 - reading valid line 
	// 2 - reading comment line

	// Read a character at a time
	while(0 != AFILE_afread(&c, sizeof(char), 1, afp)) {
		if(EOF == c) {
			if(0 < i) { // characters have been read
				line[i]='\0';
				//fprintf(stderr, "return read_line 1\n");
				return  1;
			}
			else { // nothing to report
				//fprintf(stderr, "return read_line 2\n");
				return -1;
			}
		}
		else if('\n' == c) { // endline
			if(1 == state) { // end of the valid line
				line[i]='\0';
				//fprintf(stderr, "return read_line 3\n");
				return 1;
			}
			i=state=0;
		}
		else {
			if(0 == state) { // first character in the line
				if('#' == c) { // comment
					state = 2;
				}
				else { // valid line
					state = 1;
					assert(i==0);
					assert(p==0);
				}
			}
			if(1 == state) { // valid line
				// if previous was not whitespace or 
				// current is not a whitespace
				if(0 == p || 0 == IsWhiteSpace(c)) { 
					line[i]=c;
					i++;
					p=IsWhiteSpace(c);
				}
			}
		}
	}
	// must have hit eof
	if(0 < i) { // characters have been read
		line[i]='\0';
		//fprintf(stderr, "return read_line 4\n");
		return  1;
	}
	return -1;
}
