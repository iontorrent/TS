#ifndef BLIBDEFINITIONS_H_
#define BLIBDEFINITIONS_H_

#include <sys/types.h>
#include <stdint.h>

/* Program defaults */
#define PROGRAM_NAME "bfast" /* Could just use PACKAGE_NAME */
#define DEFAULT_FILENAME "Default.txt"
#define MAX_FILENAME_LENGTH 2048
#define DEFAULT_OUTPUT_ID "OutputID"
#define DEFAULT_OUTPUT_DIR "./"
#define BREAK_LINE "************************************************************\n"
#define SEQUENCE_LENGTH 2048
#define SEQUENCE_NAME_LENGTH 4028
#define MAX_MASK_LENGTH 1024
#define MAX_CONTIG_NAME_LENGTH 2048
#define MAX_CONTIG_LOG_10 7 
#define MAX_POSITION_LOG_10 10
#define MAX_HEADER_LENGTH 2048
#define MAX_FASTA_LINE_LENGTH 2048
#define MAXIMUM_MAPPING_QUALITY 255
#define ONE_GIGABYTE (int64_t)1073741824
#define MERGE_MEMORY_LIMIT 12*((int64_t)1073741824) /* In Gigabytes */
#define RGINDEXLAYOUT_MAX_HASH_WIDTH 18
#define READS_BUFFER_LENGTH 40000
#define BFAST_MATCH_THREAD_SLEEP 1

/* Program Default Command-line parameters */
#define MAX_KEY_MATCHES 8
#define MAX_NUM_MATCHES 384
#define OFFSET_LENGTH 20

/* Testing/Debug */
#define TEST_RGINDEX_SORT 0
extern int32_t VERBOSE;

/* Sorting */
#define SHELL_SORT_GAP_DIVIDE_BY 2.2
#define RGINDEX_SHELL_SORT_MAX 50
#define RGMATCH_SHELL_SORT_MAX 50
#define ALIGNEDENTRY_SHELL_SORT_MAX 50
#define RGRANGES_SHELL_SORT_MAX 50
#define RGREADS_SHELL_SORT_MAX 50

/* Get opt */
#define OPTION_ARG_OPTIONAL 0
#define OPTION_NO_USAGE 0
enum {ExecuteGetOptHelp, ExecuteProgram, ExecutePrintProgramParameters};

/* Default output */
enum {TextOutput, BinaryOutput};
enum {TextInput, BinaryInput};
enum {BRG, BIF, BMF, BAF, SAM, LastFileType};
#define BPREPROCESS_DEFAULT_OUTPUT 1 /* 0: text 1: binary */
#define BMATCHES_DEFAULT_OUTPUT 1 /* 0: text 1: binary */
#define BALIGN_DEFAULT_OUTPUT 1 /* 0: text 1: binary */

/* SAM specific */
#define BFAST_SAM_VERSION "0.1.2"
#define BFAST_SAM_MAX_QNAME 254
#define BFAST_SAM_MAX_QNAME_SEPARATOR ":"

/* File extensions */
#define BFAST_RG_FILE_EXTENSION "brg"
#define BFAST_INDEX_FILE_EXTENSION "bif"
#define BFAST_MATCHES_FILE_EXTENSION "bmf"
#define BFAST_MATCHES_READS_FILTERED_FILE_EXTENSION "fastq"
#define BFAST_ALIGNED_FILE_EXTENSION "baf"
#define BFAST_SAM_FILE_EXTENSION "sam"

#define BFAST_MATCH_THREAD_BLOCK_SIZE 1024
#define BFAST_LOCALALIGN_THREAD_BLOCK_SIZE 512
#define BFAST_POSTPROCESS_THREAD_BLOCK_SIZE 512
#define RGMATCH_MERGE_ROTATE_NUM 100000
#define READ_ROTATE_NUM 1000000
#define RGINDEX_ROTATE_NUM 1000000
#define SORT_ROTATE_INC 0.01
#define RGINDEX_SORT_ROTATE_INC 0.001
#define ALIGN_ROTATE_NUM 1000
#define ALIGN_SKIP_ROTATE_NUM 100000
#define PARTITION_MATCHES_ROTATE_NUM 100000
#define ALIGNENTRIES_READ_ROTATE_NUM 10000
#define BFAST_TMP_TEMPLATE ".bfast.tmp.XXXXXX"
#define DEFAULT_RANGE "1-1:2147483647-2147483647"

/* For printing to stderr */
#define DEBUG 10

/* Algorithm defaults */
#define DEFAULT_MATCH_LENGTH 11
#define ALPHABET_SIZE 4
#define FORWARD '+'
#define REVERSE '-'
#define GAP '-'
#define GAP_QUALITY -1
#define NULL_LETTER 'N'
#define COLOR_SPACE_START_NT 'A'
// the next define should be the int representation of the previous define
#define COLOR_SPACE_START_NT_INT 0
#define BFAST_ID 'B'+'F'+'A'+'S'+'T'
#define AVG_MISMATCH_QUALITY 10
#define MAX_STD 4
#define MAX_INVERSION_LOG10_RATIO 3

/* To calculate the paired end insert size distribution in postprocess */
#define MIN_PEDBINS_SIZE 100
#define MIN_PEDBINS_DISTANCE -20000
#define MAX_PEDBINS_DISTANCE 20000
#define MAX_PEDBINS_DISTANCES 10000

/* Scoring matrix defaults */
#define SCORING_MATRIX_GAP_OPEN -175
#define SCORING_MATRIX_GAP_EXTEND -50
#define SCORING_MATRIX_NT_MATCH 50
#define SCORING_MATRIX_NT_MISMATCH -150 
#define SCORING_MATRIX_COLOR_MATCH 0
#define SCORING_MATRIX_COLOR_MISMATCH -125

/* Macro functions */
#define FILEREQUIRED(_file) ((NULL == _file) ? "[Required! --> Not Specified]" : _file)
#define FILEUSING(_file) ((NULL == _file) ? "[Not Using]" : _file)
#define FILESTDIN(_file) ((NULL == _file) ? "[STDIN]" : _file)
#define INTUSING(_int) ((1 != _int) ? "[Not Using]" : "[Using]")
#define SPACE(_space) ((0 == _space) ? "[NT Space]" : "[Color Space]")
#define PROGRAMMODE(_mode) ((0 == _mode) ? "[ExecuteGetOptHelp]" : ((1 == _mode) ? "[ExecuteProgram]" : "[ExecutePrintProgramParameters]"))
#define WHICHSTRAND(_mode) ((0 == _mode) ? "[Both Strands]" : ((1 == _mode) ? "[Forward Strand]" : "[Reverse Strand]"))
#define MIRRORINGTYPE(_mode) ((0 == _mode) ? "[Not Using]" : ((1 == _mode) ? "[First before the Second]" : ((2 == _mode) ? "[Second before the First]" : "[Both directions]")))
#define COMPRESSION(_c) ((AFILE_NO_COMPRESSION == _c) ? "[Not Using]" : ((AFILE_GZ_COMPRESSION == _c) ? "[gzip]" : ((AFILE_BZ2_COMPRESSION == _c) ? "[bzip2]" : "[Unknown]")))
#define LOWERBOUNDSCORE(_score) (_score = (_score < NEGATIVE_INFINITY) ? NEGATIVE_INFINITY : _score)
#define GETMIN(_X, _Y)  ((_X) < (_Y) ? (_X) : (_Y))
#define GETMAX(_X, _Y)  ((_X) < (_Y) ? (_Y) : (_X))
#define CHAR2QUAL(c) ((uint8_t)c-33)
#define QUAL2CHAR(q) (char)(((q<=93)?q:93)+33)
#define SPACENAME(_space) ((NTSpace == _space) ? "nt" : "cs")
#define GETMASKNUMBYTES(_m) (((int)((_m->readLength + 7)/8)))
#define GETMASKNUMBYTESFROMLENGTH(_l) (((int)((_l + 7)/8)))
#define GETMASKBYTE(_pos) ((int)(_pos / 8))
#define ROUND(_x) ((int)((_x) + 0.5))
#define COLORFROMINT(_c) (COLORS[(int)_c])
#define COMPAREINTS(_a, _b) ((_a < _b) ? -1 : ((_a == _b) ? 0 : 1))

/* For FindMatches.c */
#define FM_ROTATE_NUM 10000
#define DEFAULT_MATCHES_QUEUE_LENGTH 250000

#define NEGATIVE_INFINITY INT_MIN/16 /* cannot make this too small, otherwise we will not have numerical stability, i.e. become positive */
#define VERY_NEGATIVE_INFINITY (INT_MIN/16)-1000 /* cannot make this too small, otherwise we will not have numerical stability, i.e. become positive */

/* Algorithm command line options:
 *  * 0: Dynamic programming 
 *   * */
#define MIN_ALGORITHM 0
#define MAX_ALGORITHM 1
#define COLOR_MATCH 0
#define COLOR_ERROR -1
#define DEFAULT_LOCALALIGN_QUEUE_LENGTH 10000
#define DEFAULT_POSTPROCESS_QUEUE_LENGTH 50000

extern char COLORS[5];

enum {KILOBYTES, MEGABYTES, GIGABYTES};
enum {Contig_8, Contig_32};
enum {NTSpace, ColorSpace, SpaceDoesNotMatter};
enum {AlignedEntrySortByAll, AlignedEntrySortByContigPos};
enum {IgnoreExons, UseExons};
enum {BothStrands, ForwardStrand, ReverseStrand};
enum {StrandSame, StrandOpposite, StrandBoth}; /* brepair.c */
enum {BFASTReferenceGenomeFile, BFASTIndexFile};
enum {RGBinaryPacked, RGBinaryUnPacked};
enum {NoMirroring, MirrorForward, MirrorReverse, MirrorBoth};
enum {IndexesMemorySerial, IndexesMemoryAll};
/* For RGIndexAccuracy */
enum {SearchForRGIndexAccuracies, EvaluateRGIndexAccuracies, ProgramParameters};
enum {NO_EVENT, MISMATCH, INSERTION, DELETION};
enum{NoIndelType, DeletionType, InsertionType};
/* For BfastMatch */
enum {EndSearch, CopyForNextSearch};
enum {MainIndexes, SecondaryIndexes};
/* For BfastLocalAlign */
enum {Gapped, Ungapped};
enum {AllAlignments, BestOnly};
enum {Constrained, Unconstrained};
/* For BfastPostProcess */
enum {NoFiltering,      /* 0 */
	AllNotFiltered,     /* 1 */
	Unique,             /* 2 */
	BestScore,          /* 3 */
	BestScoreAll,       /* 4 */
};                  
enum {First, Second};
enum {NoneFound, Found};


/************************************/
/* 		Data structures 			*/
/************************************/

/* TODO */
typedef struct {
	int32_t readLength;
	char *read;
	int32_t qualLength;
	char *qual;
	int32_t maxReached;
	int32_t numEntries;
	uint32_t *contigs;
	int32_t *positions;
	char *strands;
	char **masks;
	// these are only used when the index is split into pieces
	int32_t *numOffsets;
	int32_t **offsets; 
} RGMatch;

/* TODO */
typedef struct {
	int32_t readNameLength;
	char *readName;
	int32_t numEnds;
	RGMatch *ends;
} RGMatches;

/* TODO */
typedef struct { 
	int64_t *startIndex;
	int64_t *endIndex;
	char *strand;
	int32_t *numOffsets;
	int32_t *offset;
	int32_t numEntries;
} RGRanges;

/* TODO */
typedef struct {
	int32_t numReads;
	char **reads;
	int32_t *readLength;
	int32_t *offset;
} RGReads;

/* TODO */
typedef struct {
	/* Storage */
	int32_t contigNameLength;
	char *contigName;
	/* Metadata */
	char *sequence; 
	int32_t sequenceLength;
	uint32_t numBytes;
} RGBinaryContig;

/* TODO */
typedef struct {
	/* Storage type */
	int32_t id;
	int32_t packageVersionLength;
	char *packageVersion;
	int32_t packed;
	/* RG storage */
	RGBinaryContig *contigs;
	int32_t numContigs;
	/* Metadata */
	int32_t space;
} RGBinary;

/* TODO */
typedef struct {
	/* Storage type */
	int32_t id;
	int32_t packageVersionLength;
	char *packageVersion;
	/* Index storage */
	uint8_t *contigs_8;
	uint32_t *contigs_32;
	int32_t *positions;
	int64_t length;
	int32_t contigType; 
	/* Index range */
	int32_t startContig;
	int32_t startPos;
	int32_t endContig;
	int32_t endPos;
	/* Index layout */
	int32_t width;
	int32_t keysize;
	int32_t *mask;
	/* Index properties */
	int32_t repeatMasker;
	int32_t space;
	int32_t depth;
	int32_t binNumber; /* which index out of 4^depth possible indexes (one-based) */
	int32_t indexNumber; /* the index # out of all indexes indexing this reference (one-based) */
	/* Hash storage */
	uint32_t hashWidth; /* in bases */
	int64_t hashLength; 
	uint32_t *starts;
} RGIndex;

/* TODO */
typedef struct {
	int32_t hashWidth;
	int32_t *mask;
	int32_t width;
	int32_t keysize;
	int32_t depth;
} RGIndexLayout;

/* TODO */
typedef struct {
	uint32_t startContig;
	uint32_t startPos;
	uint32_t endContig;
	uint32_t endPos;
} RGIndexExon;

/* TODO */
typedef struct {
	int numExons;
	RGIndexExon *exons;
} RGIndexExons;

/* TODO */
typedef struct {
	uint32_t contig;
	uint32_t position;
	char strand;
	int32_t score;
	uint8_t mappingQuality;
	int32_t alnReadLength; // enhanced aligned read
	uint8_t *alnRead; 
} AlignedEntry;

/* TODO */
typedef struct {
	int32_t readLength;
	char *read; /* Original read */
	int32_t qualLength;
	char *qual; /* Original quality */
	int32_t numEntries;
	AlignedEntry *entries;
} AlignedEnd;

/* TODO */
typedef struct {
	int32_t readNameLength;
	char *readName;
	int32_t space;
	int32_t numEnds;
	AlignedEnd *ends;
} AlignedRead;

/* TODO */
typedef struct {
	RGIndex *index;
	RGBinary *rg;
	int32_t space;
	int64_t low;
	int64_t high;
	int32_t threadID;
	int32_t showPercentComplete;
	char *tmpDir;
	int64_t mergeMemoryLimit;
} ThreadRGIndexSortData;

/* TODO */
typedef struct {
	RGIndex *index;
	RGBinary *rg;
	int32_t threadID;
	int64_t low;
	int64_t mid;
	int64_t high;
	int64_t mergeMemoryLimit;
	char *tmpDir;
} ThreadRGIndexMergeData;

/* TODO */
typedef struct {
	int32_t contigStart;
	int32_t contigEnd;
	int32_t positionStart;
	int32_t positionEnd;
} Range;

/* TODO */
typedef struct {
	int32_t *counts;
	int32_t numCounts;
} QualityScoreDifference;

/* TODO */
typedef struct {
	int32_t gapOpenPenalty;
	int32_t gapExtensionPenalty;
	int32_t ntMatch;
	int32_t ntMismatch;
	int32_t colorMatch;
	int32_t colorMismatch;
} ScoringMatrix;

/* RGIndexAccuracy.c */
typedef struct {
	int32_t length;
	int32_t *profile;
} Read;

/* RGIndexAccuracy.c */
typedef struct {
	int32_t numReads;
	double *accuracy;
	int32_t length; /* length of correct */
	int32_t numSNPs;
	int32_t numColorErrors;
	int32_t numAboveThreshold; /* where to start comparisons */
	int32_t accuracyThreshold;
} AccuracyProfile;

/* RGIndexAccuracy.c */
typedef struct {
	int32_t *mask;
	int32_t keySize;
	int32_t keyWidth;
} RGIndexAccuracy;

/* RGIndexAccuracy.c */
typedef struct {
	int32_t numRGIndexAccuracies;
	RGIndexAccuracy *indexes;
} RGIndexAccuracySet;

/* RGIndexAccuracy.c */
typedef struct {
	int32_t maxReadLength;
	int32_t *maxMismatches;
} RGIndexAccuracyMismatchProfile;

#endif
