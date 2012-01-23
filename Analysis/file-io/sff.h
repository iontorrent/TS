/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFF_H
#define SFF_H

#include <stdint.h>
#include "ion_string.h"
#include "sff_definitions.h"

/*!
  Get the read name
  @param  sff  pointer to the sff structure
  @return      the read name character string
  @details     this macro assumes that the sff has been initialized
 */
#define sff_name(sff) ((sff)->rheader->name->s)

/*!
  Get the read bases
  @param  sff  pointer to the sff structure
  @return      the read bases character string
  @details     this macro assumes that the sff has been initialized
 */
#define sff_bases(sff) ((sff)->read->bases->s)

/*!
  Get the read bases
  @param  sff  pointer to the sff structure
  @return      the number of read bases 
  @details     this macro assumes that the sff has been initialized
 */
#define sff_n_bases(sff) ((sff)->rheader->n_bases)

/*!
  Get the left-clipped quality index (1-based)
  @param  sff  pointer to the sff structure
  @return      the 1-based index
  @details     this macro assumes that the sff has been initialized
  */
#define sff_clip_qual_left(sff) ((sff)->rheader->clip_qual_left)

/*!
  Get the right-clipped quality index (1-based)
  @param  sff  pointer to the sff structure
  @return      the 1-based index
  @details     this macro assumes that the sff has been initialized
  */
#define sff_clip_qual_right(sff) ((sff)->rheader->clip_qual_right)

/*!
  Get the left-clipped adapter index (1-based)
  @param  sff  pointer to the sff structure
  @return      the 1-based index
  @details     this macro assumes that the sff has been initialized
  */
#define sff_clip_adapter_left(sff) ((sff)->rheader->clip_adapter_left)

/*!
  Get the right-clipped adapter index (1-based)
  @param  sff  pointer to the sff structure
  @return      the 1-based index
  @details     this macro assumes that the sff has been initialized
  */
#define sff_clip_adapter_right(sff) ((sff)->rheader->clip_adapter_right)

/*!
  Get the key length
  @param  sff  pointer to the sff structure
  @return      the key length 
  @details     this macro assumes that the sff has been initialized
 */
#define sff_key_length(sff) ((sff)->gheader->key_length)

/*!
  Get the read base qualities
  @param  sff  pointer to the sff structure
  @return      the quality character string
  @details     this macro assumes that the sff has been initialized
 */
#define sff_quality(sff) ((sff)->read->quality->s)

/*!
  Get the read flowgram values
  @param  sff  pointer to the sff structure
  @return      pointer the flowgram array 
  @details     this macro assumes that the sff has been initialized
 */
#define sff_flowgram(sff) ((sff)->read->flowgram)

/*!
  Get the read flowgram index
  @param  sff  pointer to the sff structure
  @return      the flowgram index array 
  @details     this macro assumes that the sff has been initialized
 */
#define sff_flow_index(sff) ((sff)->read->flow_index)

#ifdef __cplusplus
extern "C" {
#endif


    /*!
      @param  sff_file  a file pointer from which to read
      @return          a pointer to the read in SFF read
      */
    sff_t *
      sff_read(sff_file_t *sff_file);

    /*!
      @param  fp      a file pointer from which to read
      @param  header  a pointer to the SFF header
      @return         a pointer to the SFF read-in
      */
    sff_t *
      sff_read1(FILE *fp, sff_header_t *header);

    /*!
      @param  sff_file  a file pointer to which to write
      @param  sff      a pointer to the SFF to write
      @return          the number of bytes written, including padding
      */
    uint32_t
      sff_write(sff_file_t *sff_file, const sff_t *sff);

    /*!
      @param  fp   a file pointer to which to write
      @param  sff  a pointer to the SFF to write
      @return      the number of bytes written, including padding
      */
    uint32_t
      sff_write1(FILE *fp, const sff_t *sff);
    
    /*!
      @param  fp   a file pointer to which to print
      @param  sff  a pointer to the SFF read top print
      */
    void
      sff_print(FILE *fp, sff_t *sff);

    /*! 
      @return a pointer to the empty sff 
      */
    sff_t *
      sff_init();
    
    /*! 
      @return a pointer to the empty sff, with the read and rheader allocated 
      */
    sff_t *
      sff_init1();

    /*! 
      @param  sff  a pointer to the sff to destroy
      */
    void
      sff_destroy(sff_t *sff);

    /*! 
      @param  sff  a pointer to the sff to clone
      @return a pointer to the cloned sff
      */
    sff_t *
      sff_clone(sff_t *sff);

    /*! 
      @param  sff  a pointer to the sff 
      */
    void
      sff_reverse_compliment(sff_t *sff);

    /*! 
      @param  sff  a pointer to the sff 
      */
    void
      sff_to_int(sff_t *sff);

    /*! 
      @param  sff  a pointer to the sff 
      */
    void
      sff_to_char(sff_t *sff);
    /*!
      gets the read's bases
      @param  sff  a pointer to a sequence structure
      @details     this will include the key sequence qualities
      */
    inline ion_string_t *
      sff_get_bases(sff_t *sff);

    /*!
      gets the read's qualities
      @param  sff  a pointer to a sequence structure
      @details     this will include the key sequence qualities
      */
    inline ion_string_t *
      sff_get_qualities(sff_t *sff);

    /*! 
      removes the key sequence from the read and quality fields
      @param  sff  pointer to the structure to convert
      @details     this will only remove the key sequence from the SFF
      structure, and then only the read and quality (not the read header etc.)
      */
    inline void
      sff_remove_key_sequence(sff_t *sff);

    /*! 
      Return the read start position taking into account any
      quality or adapter trimming that may be present.
      @param  sff  pointer to the sff structure
      @return the 1-based index of the first unclipped base
      */
    uint16_t
      sff_clipped_read_left(sff_t *sff);

    /*! 
      Return the read stop position taking into account any
      quality or adapter trimming that may be present.
      @param  sff  pointer to the sff structure
      @return the 1-based index of the last unclipped base
      */
    uint16_t
      sff_clipped_read_right(sff_t *sff);

    /*! 
      main function for viewing a SFF file
      @param  argc  the number of command line arguments
      @param  argv  the command line arguments
      @return       0 if successful
      */
    int
      sff_view_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // SFF_H
