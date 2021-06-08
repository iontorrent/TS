/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_DEFINITIONS_H
#define TMAP_DEFINITIONS_H

#include <stdint.h>
#include <config.h>


#ifdef __cplusplus
extern "C" 
{
#endif

/*! 
  Generic Functions
  */

/*! d TMAP_VERSION_ID
  the magic id for tmap
  */
#define TMAP_VERSION_ID ('t' + 'm' + 'a' + 'p')

/* 
 * File extensions
 */
/*! d TMAP_ANNO_FILE_EXTENSION
  the file extension for the reference sequence annotations
  */
#define TMAP_ANNO_FILE_EXTENSION ".tmap.anno"
/*! d TMAP_PAC_FILE_EXTENSION
  the file extension for the packed forward reference sequence
  */
#define TMAP_PAC_FILE_EXTENSION ".tmap.pac"
/*! d TMAP_BWT_FILE_EXTENSION
  the file extension for the BWT structure
  */
#define TMAP_BWT_FILE_EXTENSION ".tmap.bwt"
/*! d TMAP_SA_FILE_EXTENSION
  the file extension for the SA structure
  */
#define TMAP_SA_FILE_EXTENSION ".tmap.sa"

// The default compression types for each file
// Note: the implementation relies on no compression
#define TMAP_ANNO_COMPRESSION TMAP_FILE_NO_COMPRESSION 
#define TMAP_PAC_COMPRESSION TMAP_FILE_NO_COMPRESSION 
#define TMAP_BWT_COMPRESSION TMAP_FILE_NO_COMPRESSION 
#define TMAP_SA_COMPRESSION TMAP_FILE_NO_COMPRESSION

/*
   CIGAR operations, from samtools.
   */
#ifndef BAM_BAM_H
#define BAM_CMATCH      0
#define BAM_CINS        1
#define BAM_CDEL        2
#define BAM_CREF_SKIP   3
#define BAM_CSOFT_CLIP  4
#define BAM_CHARD_CLIP  5
#define BAM_CPAD        6
#endif

/* For branch prediction */
#ifdef __GNUC__
#define TMAP_LIKELY(x) __builtin_expect((x),1)
#define TMAP_UNLIKELY(x) __builtin_expect((x),0)
#else
#define TMAP_LIKELY(x) (x)
#define TMAP_UNLIKELY(x) (x)
#endif

/*! 
  for each type of file, the integer id associated with this file
  @details  can be used with 'tmap_get_file_name' 
  */
enum {
    TMAP_ANNO_FILE     = 0, /*!< the reference sequence annotation file */
    TMAP_PAC_FILE      = 1, /*!< the packed forward reference sequence file */
    TMAP_BWT_FILE      = 2, /*!< the packed BWT file */
    TMAP_SA_FILE       = 3, /*!< the packed SA file */
};

/*! 
*/
enum {
    TMAP_READS_FORMAT_UNKNOWN  = -1, /*!< the reads format is unrecognized */
    TMAP_READS_FORMAT_FASTA    = 0, /*!< the reads are in FASTA format */
    TMAP_READS_FORMAT_FASTQ    = 1, /*!< the reads are in FASTQ format */
    TMAP_READS_FORMAT_SFF      = 2, /*!< the reads are in SFF format */
    TMAP_READS_FORMAT_SAM      = 3, /*!< the reads are in SAM format */
    TMAP_READS_FORMAT_BAM      = 4 /*!< the reads are in BAM format */
};

#ifdef TMAP_BWT_32_BIT
typedef uint32_t tmap_bwt_int_t;
#define TMAP_BWT_INT_MAX UINT32_MAX 
typedef int32_t tmap_bwt_sint_t;
#define TMAP_BWT_SINT_MAX INT32_MAX 
#else
typedef uint64_t tmap_bwt_int_t;
#define TMAP_BWT_INT_MAX UINT64_MAX 
typedef int64_t tmap_bwt_sint_t;
#define TMAP_BWT_SINT_MAX INT64_MAX 
#endif

/* For branch prediction */
#ifdef __GNUC__
#define TMAP_LIKELY(x) __builtin_expect((x),1)
#define TMAP_UNLIKELY(x) __builtin_expect((x),0)
#else
#define TMAP_LIKELY(x) (x)
#define TMAP_UNLIKELY(x) (x)
#endif

// Terminal colors
#ifndef DISABLE_COLORING
#define KNRM  "\x1B[0m"
#define KBLD  "\x1B[1m" // Bold
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define KBLDRED "\x1B[1m\x1B[31m"
#else
#define KNRM  ""
#define KBLD  ""
#define KRED  ""
#define KGRN  ""
#define KYEL  ""
#define KBLU  ""
#define KMAG  ""
#define KCYN  ""
#define KWHT  ""
#define KBLDRED ""
#endif

/*!
  One gibabyte.
 */
#define TMAP_1GB (1 << 30)

/*! 
  @param  algo_id  the algorithm identifier
  @return          algorithm name
  */
char *
tmap_algo_id_to_name(uint16_t algo_id);

/*!
  @param  name  the algorithm name
  @return       the algorithm identifier, -1 if not found
  */
int32_t
tmap_algo_name_to_id(char *name);

/*! tmap_nt_char_to_int
  @details  converts a DNA base in ASCII format to its 2-bit format [0-4]. 
  */
extern uint8_t tmap_nt_char_to_int[256];

/*! tmap_nt_char_to_rc_char
  @details  converts a DNA base in ASCII format to reverse compliment in ASCII format.
  */
extern uint8_t tmap_nt_char_to_rc_char[256];

/*! tmap_iupac_char_to_int
  @details  converts a IUPAC base in ASCII format to an integer format.
  */
extern uint8_t tmap_iupac_char_to_int[256];

/*! tmap_iupac_char_to_bit_string
  @details  converts a IUPAC base in ASCII format to a one-based 4-bit string, with each bit corresponding
  to the DNA base (1=A, 2=C, 4=G, 8=T).
  */
extern uint8_t tmap_iupac_char_to_bit_string[256];

/*! tmap_int_to_iupac_char
  @details  converts a IUPAC base in a zero-based 4-bit integer to its ASCII format
  */
extern char tmap_iupac_int_to_char[17];

enum {
    TMAP_SAM_RG_ID=0,
    TMAP_SAM_RG_CN,
    TMAP_SAM_RG_DS,
    TMAP_SAM_RG_DT,
    TMAP_SAM_RG_FO,
    TMAP_SAM_RG_KS,
    TMAP_SAM_RG_LB,
    TMAP_SAM_RG_PG,
    TMAP_SAM_RG_PI,
    TMAP_SAM_RG_PL,
    TMAP_SAM_RG_PU,
    TMAP_SAM_RG_SM,
    TMAP_SAM_RG_NUM
};

extern const char *TMAP_SAM_RG_TAGS[TMAP_SAM_RG_NUM];


/*! 
  @param  c  the quality value in ASCII format
  @return    the quality value in integer format
  */
#define CHAR2QUAL(c) ((uint8_t)c-33)

/*! 
  @param  q  the quality value in integer format
  @return    the quality value in ASCII format
  */
#define QUAL2CHAR(q) (char)(((((unsigned char)q)<=93)?q:93)+33)

#ifndef htonll
/*! 
  converts a 64-bit value to network order
  @param  x  the 64-bit value to convert
  @return    the converted 64-bit value
  */
#define htonll(x) ((((uint64_t)htonl(x)) << 32) + htonl(x >> 32))
#endif

#ifndef ntohll
/*! 
  converts a 64-bit value to host order
  @param  x  the 64-bit value to convert
  @return    the converted 64-bit value
  */
#define ntohll(x) ((((uint64_t)ntohl(x)) << 32) + ntohl(x >> 32))
#endif

#ifndef tmap_roundup32
/*! 
  rounds up to the nearest power of two integer
  @param  x  the integer to round up
  @return    the smallest integer greater than x that is a power of two 
  */
#define tmap_roundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))
#endif

// debug functions
#define tmap_print_debug_int(_name) (fprintf(stderr, #_name "=%d\n", (_name)))
#define tmap_print_debug_string(_name) (fprintf(stderr, #_name "=%s\n", (_name)))

/*!
  @param  reads_format  the reads format
  @return               the sequence format (for tmap_seq_t)
  */
int32_t 
tmap_reads_format_to_seq_type(int32_t reads_format);

/*! 
  @param  v  the value to take the log 2
  @return    log of the value, base two
  */
uint32_t
tmap_log2(uint32_t v);

/*! 
  gets the name of a specific file based on the reference sequence
  @param  prefix   the prefix of the file to be written, usually the fasta file name 
  @param  type    the type of file based on this reference sequence
  @return         a pointer to the file name string
  */
char *
tmap_get_file_name(const char *prefix, int32_t type);

/*! 
  @param  optarg  the string of the file format
  @return         the format type
  */
int 
tmap_get_reads_file_format_int(char *optarg);

/*! 
  checks the extension of the file to recognize its format     
  @param  fn            the file name 
  @param  reads_format  pointer to the reads format, if any (unknown|fastq|fq|fasta|fa|sff)
  @param  compr_type    pointer the type of compression used, if any (none|gz|bz2)
  @details              if the reads_format is unknown, it will be populated; similarly for compr_type.
  */
void
tmap_get_reads_file_format_from_fn_int(char *fn, int32_t *reads_format, int32_t *compr_type);

/*! 
  @param  format  the interger file format specifier
  @return         the format type (string)
  */
char *
tmap_get_reads_file_format_string(int format);

/*!
  reverses a given string
  @param  seq  the string to reverse
  @param  len  the length of the string
  */
void
tmap_reverse(char *seq, int32_t len);

/*!
  reverses a given integer string
  @param  seq  the string to reverse
  @param  len  the length of the string
  */
void
tmap_reverse_int(uint8_t *seq, int32_t len);

/*!
  reverse compliments a given string
  @param  seq  the character DNA sequence
  @param  len  the length of the DNA sequence
  */
void
tmap_reverse_compliment(char *seq, int32_t len); 

/*!
  reverse compliments a given string
  @param  seq  the integer DNA sequence
  @param  len  the length of the DNA sequence
  */
void
tmap_reverse_compliment_int(uint8_t *seq, int32_t len);

/*!
  compliments a given string
  @param  seq  the character DNA sequence
  @param  len  the length of the DNA sequence
  */
void
tmap_compliment(char *seq, int32_t len); 

/*!
  converts a string to an integer
  @param  seq  the character DNA sequence
  @param  len  the length of the DNA sequence
  */
void
tmap_to_int(char *seq, int32_t len); 

/*!
  converts a integer to string
  @param  seq  the character integer DNA sequence
  @param  len  the length of the DNA sequence
  */
void
tmap_to_char(char *seq, int32_t len); 

/*!
  removes trailing whitespaces from a given string
  @param  str  the string to chomp
  @return      the number of characters removed
  */
int32_t
tmap_chomp(char *str);

/*!
  checks if there is any overlap between the two regions [low1,high1] and [low2,high2]
  @param  low1   the lower interval start
  @param  high1  the lower interval end
  @param  low2   the higher interval start
  @param  high2  the higher interval end
  @return        -1 if high1 < low2, 1 if high2 < low1, 0 otherwise
  */
int32_t
tmap_interval_overlap(uint32_t low1, uint32_t high1, uint32_t low2, uint32_t high2);

/*!
  compares the two version strings
  @param  v1  the first version string
  @param  v2  the second version string
  @return     -1 if v1 < v2, 0 if v1 == v2, 1 otherwise
  */
int32_t
tmap_compare_versions(const char *v1, const char *v2);

/*!
  validates the flow order
  @param  flow_order  the flow order to validate
  @return            0 if all four bases are present in the flow order, 
  -1 if there was an unrecognized base, and -2 if not all bases are present.
  */
int32_t
tmap_validate_flow_order(const char *flow_order);

/*!
  validates the key sequence
  @param  key_seq  the flow order to validate
  @return            0 if all four bases are present in the flow order, 
  and -1 if there was an unrecognized base.
  */
int32_t
tmap_validate_key_seq(const char *key_seq);

/*!
  Prints the TMAP version and initial info.
  @param  argc  the number of arguments
  @param  argv  the argument list
  @return       1 if successful, 0 otherwise
  */
int
tmap_version(int argc, char *argv[]);

/*!
  @return  the number of cpus available for multi-threading
 */
int32_t
tmap_detect_cpus();

#ifdef __cplusplus
}
#endif

#endif // TMAP_DEFINITIONS_H
