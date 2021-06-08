/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_REFSEQ_H
#define TMAP_REFSEQ_H

#include <stdint.h>
#include "../util/tmap_string.h"
#include "../util/tmap_definitions.h"
#include "../io/tmap_file.h"
#include "../map/util/tmap_map_locopt.h"


/*! 
  DNA Reference Sequence Library
  */

// buffer size for reading in from a FASTA file
#define TMAP_REFSEQ_BUFFER_SIZE 0x10000

// the number of bases on a given line for "tmap pac2refseq"
#define TMAP_REFSEQ_FASTA_LINE_LENGTH 72

/*! 
  @param  _len  the number of bases stored 
  @return       the number of bytes allocated
  */
#define tmap_refseq_seq_memory(_len) ((size_t)((((_len)-1) >> 2) + 1))

/*! 
  @param  _i  the 0-based base position 
  @return     the 0-based byte index
  */
#define tmap_refseq_seq_byte_i(_i) ((_i) >> 2)

/*! 
  @param  _i  the 0-based base position 
  @return     the number of bits the base is shifted (returns a multiple of two)
  @details    the reference is stored in a 2-bit format
  */
#define tmap_refseq_seq_byte_shift(_i) ((0x3 - ((_i) & 0x3)) << 1)

/*! 
  @param  _refseq  pointer to the structure holding the reference sequence
  @param  _i       the 0-based base position to store
  @param  _c       the base's 2-bit integer representation
  */
#define tmap_refseq_seq_store_i(_refseq, _i, _c) (_refseq->seq[tmap_refseq_seq_byte_i(_i)] |= _c << tmap_refseq_seq_byte_shift(_i))

/*! 
  @param  _refseq  pointer to the structure holding the reference sequence
  @param  _i       the 0-based base position to retrieve
  @return          the base's 2-bit integer representation
  */
#define tmap_refseq_seq_i(_refseq, _i) ((_refseq->seq[tmap_refseq_seq_byte_i(_i)] >> tmap_refseq_seq_byte_shift(_i)) & 0x3)

/*! 
  */
typedef struct 
{
    tmap_string_t *name;  /*!< the name of the contig */
    uint64_t len;  /*!< the length of the current contig  */
    uint64_t offset;  /*!< the offset from the start of the reference (zero-based) */
    uint32_t *amb_positions_start;  /*!< start positions of ambiguous bases (one-based) */
    uint32_t *amb_positions_end;  /*!< end positions of ambiguous bases (one-based) */
    uint8_t *amb_bases;  /*!< the ambiguous bases (IUPAC code) */
    uint32_t num_amb;  /*!< the number of ambiguous bases */
} 
tmap_anno_t;

/*!
 * read end statistics record (check Jingwei's specs if these are truly medians or some other measures)
 */
#pragma pack (push, 1)
typedef struct
{
    int32_t coord; /*!< position on the chromosome */
    int32_t count; /*!< median of the abslute number of full-length reads observed ending at this position */
    double  fraction; /*< median of the fraction of full-length reads ending at this position relative to coverage */
    char    flag; /*< indicates if the entry is within primer region. Values: O - on primer region, S - falls inside primer region, L - falls outside primer region */
} 
tmap_map_endpos_t;

/*!
 * block of read end statistics records for the amplicon. READ_END records follow READ_START records
 */
typedef struct 
{
    uint32_t index;  /*!< index into the first tmap_map_endpos_t in the endposmem array in tmap_refseq_t structure. */
    uint32_t starts_count; /*!< number of records for READ_START's following the index position in the endposmem array in tmap_refseq_t structure. */
    uint32_t ends_count; /*!< number of records for READ_END's following the index+starts_count position in the endposmem array in tmap_refseq_t structure. */
}
tmap_map_endstat_t;

typedef struct 
{
    tmap_map_endpos_t* positions;  /*!< pointer to the array of tmap_map_endpos_t */
    uint32_t starts_count; /*!< number of records for READ_START's in the 'positions' array above. */
    uint32_t ends_count; /*!< number of records for READ_END's following the first starts_count records in the 'positions' array above. */
}
tmap_map_endstat_p_t;
#pragma pack (pop)

/*! 
  */
typedef struct 
{
   uint64_t version_id;  /*!< the version id of this file */
   tmap_string_t *package_version;  /*!< the package version */
   uint8_t *seq;  /*!< the packed nucleotide sequence, with contigs concatenated */
   tmap_anno_t *annos;  /*!< the annotations about the contigs */
   int32_t num_annos;  /*!< the number of contigs (and annotations) */
   uint64_t len;  /*!< the total length of the reference sequence */
   uint32_t is_shm;  /*!< 1 if loaded from shared memory, 0 otherwise */
   /* not in the file */
   void *refseq_fp;
   uint32_t bed_exist;
   uint32_t beditem;  // number of contigs (== num_annos when mapping, curent contig number when masking fasta)
   uint32_t *bednum;  // number of bed records encountered per reference contg, [recno_in_ctg0, recno_in_ctg1,...]
   uint32_t **bedstart; // region starts per contig [[reg0_start,...reg<bednum[0]>start],[reg0ctg0start,...reg<bednum[1]>ctg0start>],..]
   uint32_t **bedend; // region ends per contig [[reg0_end,...reg<bednum[0]>end],[reg0ctg0start,...reg<bednum[1]>ctg0start>],..]
   uint32_t **parovr; // override data index: [[reg0ctg0_locopt_idx, reg1ctg0_locopt_idx, ...],[reg0ctg1_locopt_idx,..],..], index is UINT32_MAX for no override
   tmap_map_locopt_t *parmem; // actual storage for override data, regXctgY_locopt_ptrs from above point here
   uint32_t parmem_used; // number of members in parmem array that are in use
   uint32_t parmem_size; // size of allocated parmem array
   tmap_map_endstat_t **read_ends; // read ends index: an array per chromosome
   tmap_map_endpos_t *endposmem; // actial storage for read ends data, read_ends[contig_idx][amplicon_idx].end_positions point here
   uint32_t endposmem_used; // occupied slots in endposmem
   uint32_t endposmem_size; // allocated size of endposmem
}
tmap_refseq_t;

/*!
  returns the index version format given a package version
  @param  v  the package version string
  @return    the index version format string
  */
const char * 
tmap_refseq_get_version_format(const char *v);

/*! 
  @param  fn_fasta     the file name of the fasta file
  @param  compression  the type of compression, if any to be used
  @param  fwd_only     1 if to pack the forward sequence only, 0 otherwise
  @param  old_v        1 if to produce a version number that is old enough so can be worked with any older version of tmap
  @return              the length of the reference sequence
  */
uint64_t
tmap_refseq_fasta2pac(const char *fn_fasta, int32_t compression, int32_t fwd_only, int32_t old_v);

/*! 
  @param  fn_fasta     the file name of the fasta file
  */
void
tmap_refseq_pac2revpac(const char *fn_fasta);

/*! 
  @param  refseq    pointer to the structure in which to store the data 
  @param  fn_fasta  the fn_fasta of the file to be written, usually the fasta file name 
  */
void
tmap_refseq_write(tmap_refseq_t *refseq, const char *fn_fasta);

/*! 
  @param  fn_fasta  the fn_fasta of the file to be read, usually the fasta file name 
  @return           a pointer to the initialized memory
  */
tmap_refseq_t *
tmap_refseq_read(const char *fn_fasta);

/*! 
  @param  len  the refseq length
  @return      the approximate number of bytes required for this refseq in shared memory
  */
size_t
tmap_refseq_approx_num_bytes(uint64_t len);

/*! 
  @param  refseq  the refseq structure 
  @return         the number of bytes required for this refseq in shared memory
  */
size_t
tmap_refseq_shm_num_bytes(tmap_refseq_t *refseq);

/*! 
  @param  fn_fasta  the fn_fasta of the file to be read, usually the fasta file name 
  @return           the number of bytes required for this refseq in shared memory
  */
size_t
tmap_refseq_shm_read_num_bytes(const char *fn_fasta);

/*! 
  @param  refseq  the refseq structure to pack 
  @param  buf     the byte array in which to pack the refseq data
  @return         a pointer to the next unused byte in memory
  */
uint8_t *
tmap_refseq_shm_pack(tmap_refseq_t *refseq, uint8_t *buf);

/*! 
  @param  buf  the byte array in which to unpack the refseq data
  @return      a pointer to the initialized refseq structure
  */
tmap_refseq_t *
tmap_refseq_shm_unpack(uint8_t *buf);

/*! 
  @param  refseq  pointer to the structure in which the data is stored
  */
void
tmap_refseq_destroy(tmap_refseq_t *refseq);

/*! 
  @param  refseq      pointer to the structure in which the data is stored
  @param  pacpos      the packed FASTA position (one-based)
  @param  aln_length  the alignment length
  @param  seqid       the zero-based sequence index to be returned
  @param  pos         the one-based position to be returned
  @param  strand      the strand (0 - forward, 1 - reverse)
  @return             the one-based position, 0 if not found (i.e. overlaps two chromosomes)
  */
tmap_bwt_int_t
tmap_refseq_pac2real(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos, uint32_t aln_length, uint32_t *seqid, uint32_t *pos, uint8_t *strand);

/*! 
  Retrieves a subsequence of the reference in 2-bit format
  @param  refseq  pointer to the structure in which the data is stored
  @param  pacpos  the packed FASTA position (one-based)
  @param  length  the subsequence length retrieve
  @param  target  the target in which to store the data (must be allocated with enough memory)
  @return         the length retrieved
  */
int32_t
tmap_refseq_subseq(const tmap_refseq_t *refseq, tmap_bwt_int_t pacpos, uint32_t length, uint8_t *target);

/*! 
  Retrieves a subsequence of the reference in 2-bit format
  @param  refseq  pointer to the structure in which the data is stored
  @param  seqid   the sequence id (one-based)
  @param  start   the start position (one-based)
  @param  end     the end position (one-based)
  @param  target  pre-allocated memory for the target
  @param  to_n    change all ambiguous bases to N, otherwise they will be returned as the correct code
  @param  conv    the number of bases converted to ambiguity bases
  @return         the target sequence if successful, NULL otherwise
  */
uint8_t*
tmap_refseq_subseq2(const tmap_refseq_t *refseq, uint32_t seqid, uint32_t start, uint32_t end, uint8_t *target, int32_t to_n, int32_t *conv);

/*! 
  Checks if the given reference range has ambiguous bases
  @param  refseq  pointer to the structure in which the data is stored
  @param  seqid   the sequence index (one-based)
  @param  start   the start position (one-based and inclusive)
  @param  end     the end position (one-based and inclusive)
  @return         zero if none were found, otherwise the one-based index into "amb_bases" array
  */
int32_t
tmap_refseq_amb_bases(const tmap_refseq_t *refseq, uint32_t seqid, uint32_t start, uint32_t end);

/*! 
  main-like function for 'tmap fasta2pac'
  @param  argc  the number of arguments
  @param  argv  the argument list
  @return       0 if executed successful
  */
int
tmap_refseq_fasta2pac_main(int argc, char *argv[]);

/*! 
  main-like function for 'tmap refinfo'
  @param  argc  the number of arguments
  @param  argv  the argument list
  @return       0 if executed successful
  */
int
tmap_refseq_refinfo_main(int argc, char *argv[]);

/*! 
  main-like function for 'tmap pac2fasta'
  @param  argc  the number of arguments
  @param  argv  the argument list
  @return       0 if executed successful
  */
int
tmap_refseq_pac2fasta_main(int argc, char *argv[]);

/*!
  main-like function for 'tmap mask'
  @param  argc  the number of arguments
  @param  argv  the argument list
  @return       0 if executed successful
  */

int 
tmap_refseq_fasta2maskedfasta_main(int argc, char *argv[]);

/*! 
  Checks if the given reference range has ambiguous bases
  @param  refseq        pointer to the structure storing reference data
  @param  bedfile       name for the BED file to use
  @param  use_par_ovr   flag indicating if amplicon-specific options should be used if present in BED
  @param  use_reads_end flag indicating if amplicon-specific read ends statistics should be used if provided in BED
  @param  local_logs    pointer to the integer holding count of amplicons with overriden log flag
  @return               one if BED is parsed, zero if error occured
  */
int
tmap_refseq_read_bed (tmap_refseq_t *refseq, char *bedfile, int32_t  use_par_ovr, int32_t use_read_ends, int32_t* local_logs);

#endif // TMAP_REFSEQ_H
