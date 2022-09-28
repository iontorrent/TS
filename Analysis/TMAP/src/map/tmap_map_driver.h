/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TMAP_MAP_DRIVER_H
#define TMAP_MAP_DRIVER_H

#include <sys/types.h>
#include "../index/tmap_index.h"
#include "../seq/tmap_seqs.h"

// DVK - realignment
#include "../realign/realign_wrapper.h"

#ifdef HAVE_LIBPTHREAD
#define TMAP_MAP_DRIVER_THREAD_BLOCK_SIZE 512
#endif

/*!
  This function will be invoked after reading in all the reference data
  to initialize any program options and print messages.
  @param  data    the program persistent data
  @param  refseq  the reference sequence
  @param  opt     the program options
  @return         0 upon success, non-zero otherwise
 */
typedef int32_t (*tmap_map_driver_func_init)(void **data, tmap_refseq_t *refseq, tmap_map_opt_t *opt);

/*!
  This function will be invoked before a thread begins process its sequences.
  @param  data  the thread persistent data
  @param  opt   the program options
  @return       0 upon success, non-zero otherwise
 */
typedef int32_t (*tmap_map_driver_func_thread_init)(void **data, 
                                                    tmap_map_opt_t *opt);
/*!
  This function will be invoked to map a sequence.
  @param  data    the thread persistent data
  @param  seqs    the sequence to map (forward, reverse compliment, reverse, compliment)
  @param  index   the reference index
  @param  stat    the driver statistics (for mapall only)
  @param  rand    the random number generator
  @param  hash    the occurrence hash
  @param  opt     the program options
  @return         the mappings upon success, NULL otherwise
 */
typedef tmap_map_sams_t* (*tmap_map_driver_func_thread_map)(void **data, tmap_seq_t **seqs, tmap_index_t *index, tmap_bwt_match_hash_t *hash, tmap_rand_t *rand, tmap_map_opt_t *opt);

/*!
  This function will be invoked to give a mapping quality to a set of mappings
  @param  sams     the mappings
  @param  seq_len  the sequence length
  @param  opt      the program options
  @return          0 upon success, non-zero otherwise
  */
typedef int32_t (*tmap_map_driver_func_mapq)(tmap_map_sams_t *sams, int32_t seq_len, tmap_map_opt_t *opt, tmap_refseq_t *refseq);

/*!
  This function will be invoked after a thread has process all of its sequences.
  @param  data  the thread persistent data
  @param  opt   the program options
  @return       0 upon success, non-zero otherwise
  */
typedef int32_t (*tmap_map_driver_func_thread_cleanup)(void **data, tmap_map_opt_t *opt);

/*!
  This function will be invoked after all threads and reads have been processed.
  @param  data  the program persistent data
  @return       0 upon success, non-zero otherwise
 */
typedef int32_t (*tmap_map_driver_func_cleanup)(void **data);

/*!
  A mapping algorithm with associate functions and workspace.
 */
typedef struct 
{
    tmap_map_driver_func_init func_init; /*!< this function will be run once per program to initialize persistent data across the program */
    tmap_map_driver_func_thread_init func_thread_init; /*!< this function will be run once per thread to initialize persistent data across that thread */
    tmap_map_driver_func_thread_map func_thread_map; /*!< this function will be run once per thread per input sequence to map the sequence */
    tmap_map_driver_func_thread_cleanup func_thread_cleanup; /*!< this function will be run once per thread to cleanup/destroy any persistent data across that thread */
    tmap_map_driver_func_cleanup func_cleanup; /*!< this function will be run once per program to cleanup/destroy any persistent data across the program */
    tmap_map_opt_t *opt; /*!< the program options specific to this algorithm */
    void *data; /*< the program persistent data for the algorithm */
    void **thread_data; /*< the thread persistent data for the algorithm */
}
tmap_map_driver_algorithm_t;

/*!
  A collection of mapping algorithms to be applied in a specific stage.
 */
typedef struct 
{
    int32_t stage; /*!< the stage for these algorithms (one-based) */
    tmap_map_driver_algorithm_t **algorithms; /*!< the algorithms to run */
    int32_t num_algorithms; /*!< the number of algorithms to run */
    tmap_map_opt_t *opt; /*!< stage specific options */
    // DK: fix for:
    //   a) race condition in access to iupac_matrix from threads at different stages of different reads
    //   b) unnecessaary repeated re-population of iupac_matrix for every read 
    tmap_sw_param_t sw_param;
    int32_t matrix [IUPAC_MATRIX_SIZE];
}
tmap_map_driver_stage_t;

/*!
  The mapping driver object, composed of multiple stages of algorithms.
  */
typedef struct 
{
    tmap_map_driver_stage_t **stages; /*!< the stages for the algorithms */
    int32_t num_stages; /*< the number of stages */
    tmap_map_driver_func_mapq func_mapq; /*!< this function will be run to calculate the mapping quality */
    tmap_map_opt_t *opt; /*!< the global mapping options */
}
tmap_map_driver_t;

/*! 
  @param  algo_id    the one-base algorithm identifier
  @param  func_mapq  the mapping quality function to apply
  @return            a new mapping driver.
  */
tmap_map_driver_t*
tmap_map_driver_init(int32_t algo_id, tmap_map_driver_func_mapq func_mapq);

/*!
  @param  driver              the mapping driver.
  @param  func_init           this function will be run once per program to initialize persistent data across the program 
  @param  func_thread_init    this function will be run once per thread to initialize persistent data across that thread 
  @param  func_thread_map     this function will be run once per thread per input sequence to map the sequence 
  @param  func_thread_cleanup this function will be run once per thread to cleanup/destroy any persistent data across that thread 
  @param  func_cleanup        this function will be run once per program to cleanup/destroy any persistent data across the program 
  @param  opt                 the program options
  @details                    the option structure should identify the algorithm identifier and algorithm stage, and all functions except func_thread_map can be NULL
 */
void
tmap_map_driver_add(tmap_map_driver_t *driver,
                    tmap_map_driver_func_init func_init,
                    tmap_map_driver_func_thread_init func_thread_init,
                    tmap_map_driver_func_thread_map func_thread_map,
                    tmap_map_driver_func_thread_cleanup func_thread_cleanup,
                    tmap_map_driver_func_cleanup func_cleanup,
                    tmap_map_opt_t *opt);

/*!
  Run the mapping algorithms 
  @param  driver  the mapping driver
 */
void
tmap_map_driver_run(tmap_map_driver_t *driver);

/*!
  Destroy the mapping driver.
  @param  driver  the mapping driver
 */
void
tmap_map_driver_destroy(tmap_map_driver_t *driver);


/*!
  the core routine for mapping data with or without threads
  @param  driver  the driver code
  */
void
tmap_map_driver_core(tmap_map_driver_t *driver);

#endif // TMAP_MAP_DRIVER_H
