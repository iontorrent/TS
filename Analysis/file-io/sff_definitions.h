/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_DEFINITIONS_H
#define SFF_DEFINITIONS_H

#include <stdint.h>
#include "ion_string.h"

#define SFF_MAGIC 0x2E736666
#define SFF_VERSION 1

#define SFF_INDEX_VERSION 1
#define SFF_INDEX_MAGIC 0xDEADBEEF

/*! 
  A Library for SFF data
  */

// Note: currently uses a simple indexing scheme, but an R-tree would be more
// efficient
enum {
    SFF_INDEX_ROW_ONLY = 0, /*!< only the offsets of the rows are stored */
    SFF_INDEX_ALL = 1  /*!< the offsets of all reads are stored */
};

/*! 
*/
typedef struct {
    uint32_t magic;  /*!< the magic number for this file */
    uint32_t version;  /*!< the version number */
    uint64_t index_offset;  /*!< not currently used (value is 0) */
    uint32_t index_length;  /*!< not currently used (value is 0) */
    uint32_t n_reads;  /*!< the number of reads in the file */
    uint16_t gheader_length;  /*!< the number of bytes in the global header including padding */
    uint16_t key_length;  /*!< the length of the key sequence used with these reads */
    uint16_t flow_length;  /*!< the number of nucleotide flows used in this experiment */
    uint8_t flowgram_format;  /*!< the manner in which signal values are encoded (value is 1) */
    ion_string_t *flow;  /*!< the string specifying the ith nucleotide flowed  */
    ion_string_t *key;  /*!< the string specifying the ith nucleotide of the sequence key */
} sff_header_t;

/*! 
*/
typedef struct {
    uint16_t rheader_length;  /*!< the number of bytes in the  */
    uint16_t name_length;  /*!< the number of characters in the name of the read (not including the null-terminator) */
    uint32_t n_bases;  /*!< the number of bases in the read */
    /* NOTE: clip points are 1-based, inclusive ranges */
    uint16_t clip_qual_left;  /*!< the 1-based coordinate of the first base after the (quality) left clipped region (zero if no clipping has been applied) */
    uint16_t clip_qual_right;  /*!< the 1-based coordinate of the first base *before* the (quality) right clipped region (zero if no clipping has been applied) */
    uint16_t clip_adapter_left;  /*!< the 1-based coordinate of the first base after the (adapter) left clipped region (zero if no clipping has been applied) */
    uint16_t clip_adapter_right;  /*!< the 1-based coordinate of the first base *before* the (adapter) right clipped region (zero if no clipping has been applied) */
    ion_string_t *name;  /*!< the read name  */
} sff_read_header_t;

/*! 
*/
typedef struct {
    uint16_t *flowgram;  /*!< the flowgram  */
    uint8_t *flow_index;  /*!< the 1-based flow index for each base called */
    ion_string_t *bases;  /*!< the called bases */
    ion_string_t *quality;  /*!< the quality score for each base call */
} sff_read_t;

/*! 
  Structure for storing information for a SFF entry
*/
typedef struct {
    sff_header_t *gheader;  /*!< pointer to the global header */
    sff_read_header_t *rheader;  /*!< pointer to the read header */
    sff_read_t *read;  /*!< pointer to the read */
    int32_t is_int;  /*!< 1 if the bases are integer values, 0 otherwise */
} sff_t;

typedef struct {
    uint32_t index_magic_number;  /*!< the magic number of the index */
    uint32_t index_version;  /*!< the version of the index */
    int32_t num_rows;  /*!< the number of rows */
    int32_t num_cols;  /*!< the number of cols */
    int32_t type;  /*!< the SFF index type */
    uint64_t *offset; /*!< the absolute byte offset of the read in the file, in row-major order */
} sff_index_t;

/*! 
  Structure for reading or writing SFF files
*/
typedef struct {
    FILE *fp;  /*!< the file pointer from which to read/write */
    sff_header_t *header;  /*!< pointer to the global header */
    sff_index_t *index;  /*!< ponter to the SFF index if available */
    uint32_t mode;  /*!< file access mode */
} sff_file_t;

#endif // SFF_DEFINITIONS_H
