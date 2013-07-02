#ifndef __SAM_HEADER_H__
#define __SAM_HEADER_H__

#define SAM_HEADER_CURRENT_VERSION "1.4"

// NB: SAM Header merging not supported

#ifdef __cplusplus
extern "C" {
#endif
    /*! @enum
      @abstract the standard record types.
     */
    enum {
        SAM_HEADER_TYPE_NONE = -1,
        SAM_HEADER_TYPE_HD = 0,
        SAM_HEADER_TYPE_SQ = 1,
        SAM_HEADER_TYPE_RG = 2,
        SAM_HEADER_TYPE_PG = 3,
        SAM_HEADER_TYPE_CO = 4,
        SAM_HEADER_TYPE_NUM = 5
    };

    /*! @typedef
      @abstract Structure for a SAM Header records.
      @field  hash    hash of tag and value
      @field  type    the record type (tag type, if non-standard -1)
      @field  tag     the record type (character value)
      */
    typedef struct {
        void *hash;
        int32_t type;
        char tag[2];
    } sam_header_record_t;

    /*! @typedef
      @abstract Structure for a group of SAM Header records of the same type.
      @field  records the list of SAM Header records of the same type
      @field  n       the number of SAM Header records
      @field  type    the record type (tag type, if non-standard -1)
      @field  tag     the record type (character value)
      */
    typedef struct {
        sam_header_record_t **records;
        int32_t n;
        int32_t type;
        char tag[2];
    } sam_header_records_t;

    /*! @typedef
      @abstract Structure for a SAM Header record
      @field  hash    the hash between record type tags and records of the same type
      */
    typedef struct {
        void *hash;
    } sam_header_t;

    /*! 
      @abstract Creates a new structure for a given header record.
      @param  tag  the record type
      @return  the initialized empty header record
      */
    sam_header_record_t*
      sam_header_record_init(const char tag[2]);

    /*! 
      @abstract Destroys a header record.
      @param  r  the record to destroy
      */
    void
      sam_header_record_destroy(sam_header_record_t *r);

    /*! 
      @abstract Adds the given tag and value to the record.
      @param  r      the record in which to insert
      @param  key    the tag (key)
      @param  value  the value 
      @return  0 if the tag could not be added (already exist or memory allocation failure), 1 upon success
      */
    int32_t
      sam_header_record_add(sam_header_record_t *r, const char *key, const char *value);

    /*! 
      @abstract Sets the given tag and value to the record. This will overwrite the given
      value if the tag is already present.
      @param  r      the record in which to insert
      @param  key    the tag (key)
      @param  value  the value 
      @return  0 if the tag could not be added (memory allocation failure), 1 upon success
      */
    int32_t
      sam_header_record_set(sam_header_record_t *r, const char *tag, const char *value);

    /*! 
      @abstract Gets the value for the given tag within the record.
      @param  r      the record in which to insert
      @param  tag    the tag (key)
      @return  the value, NULL if the tag is not present 
      */
    char*
      sam_header_record_get(const sam_header_record_t *r, const char *tag);

    /*!
      @abstract Removes the tag from the record.
      @param  r      the record in which to delete
      @param  tag    the tag (key)
      @return  1 if the tag was found and deleted, 0 otherwise
      */
    int32_t
      sam_header_record_remove(const sam_header_record_t *r, const char *tag);

    /*! 
      @abstract Checks the SAM record for consistency for required tags in standard record types.
      @param  record  the record to check
      @return  1 if the record is consistent, 0 otherwise
      */
    int32_t
      sam_header_record_check(const sam_header_record_t *record);

    /*! 
      @abstract parses a given SAM Header line.
      @param  buf  the character buffer for a given line
      @return  the returned record representing the given line
      */
    sam_header_record_t*
      sam_header_record_parse(const char *buf);

    /*! 
      @abstract clones the SAM Header record.
      @param  src  the record to clone
      @return  the cloned SAM Header record, with a deep copy of the data
      */
    sam_header_record_t*
      sam_header_record_clone(const sam_header_record_t *src);

    /*! 
      @abstract initializes a group of records with the common record type/tag. 
      @param  tag  the common record tag 
      @return  the initialized SAM Header record group
      */
    sam_header_records_t*
      sam_header_records_init(const char tag[2]);

    /*! 
      @abstract destroys the SAM Header record group.
      @param  records  the recourd group to destroy
      */
    void
      sam_header_records_destroy(sam_header_records_t *records);

    /*! 
      @abstract adds the given record to the given group.
      @param  records  the recourd group to destroy
      @return  1 if the record was added, 0 if unsuccessful (the record and record group were of different types)
      @discussion  performs a shallow copy of the record 
      */
    int32_t
      sam_header_records_add(sam_header_records_t *records, sam_header_record_t *record);

    /*!
      @abstract initializes the SAM Header with empty record groups representing the non-standard tags.
      @return the initialized SAM Header
      */
    sam_header_t*
      sam_header_init();

    /*!
      @abstract destroys the SAM Header.
      @param  h  the SAM Header to destroy
     */
    void
      sam_header_destroy(sam_header_t *h);

    /*!
      @abstract Gets the record group associated with the record type.
      @param  h         the SAM Header 
      @param  type_tag  the record group type/tag
      @return  the record group, or NULL if none were found
     */
    sam_header_records_t*
      sam_header_get_records(const sam_header_t *h, const char type_tag[2]);

    /*!
      @abstract Gets a list of records of the given record group type that have the value for the tag.
      @param  h         the SAM Header 
      @param  type_tag  the record group type/tag
      @param  tag       the record tag (specific to the group)
      @param  value     the value to match
      @param  n         pointer to the number of records that are returned
      @return           the list of records, shallow copied
     */
    sam_header_record_t**
      sam_header_get_record(const sam_header_t *h, char type_tag[2], char tag[2], char *value, int32_t *n);

    /*!
      @abstract Adds the record to the SAM Header.
      @param  h       the SAM Header 
      @param  record  the record to add
      @return         1 if successful, 0 otherwise (for example if there are too many of that record type already)
     */
    int32_t
      sam_header_add_record(sam_header_t *h, sam_header_record_t *record);

    /*!
      @abstract Clones the given SAM Header.
      @param  h  the SAM Header to clone 
      @return    the cloned SAM Header with a deep copy 
     */
    sam_header_t*
      sam_header_clone(const sam_header_t *h);

    /*!
      @abstract Creates a list of values (shallow copied) for the given record type and tag.
      @param  h         the SAM Header 
      @param  type_tag  the record group type
      @param  key_tag   the tag
      @param  n         the number of values returned
      @return           the list of values
     */
    char **
      sam_header_list(const sam_header_t *h, const char type_tag[2], const char key_tag[2], int *n);

    /*!
      @abstract Creates a hash of the tag and value for a given record group type. 
      @param  h          the SAM Header 
      @param  type_tag   the record group type
      @param  key_tag    the tag
      @param  value_tag  the value tag
      @return            the hash (with values shallow copied)
     */
    void*
      sam_header_table(const sam_header_t *h, char type_tag[2], char key_tag[2], char value_tag[2]);

    /*!
      @abstract Gets the value in the table given a key.
      @param  h    the table hash
      @param  key  the key to retrieve
      @return      the value associated with the key, NULL if none is found
     */
    const char *
      sam_tbl_get(void *h, const char *key);

    /*!
      @abstract The number of elements in the table.
      @param  h    the table hash
      @return      the number values
     */
    int 
      sam_tbl_size(void *h);

    /*!
      @abstract Destroys the given table.
      @param  h    the table hash
     */
    void 
      sam_tbl_destroy(void *h);

    // NB: deep copy
    // NB: not implemented
    int32_t
      sam_header_merge_into(sam_header_t *dst, const sam_header_t *src);

    // NB: not implemented
    sam_header_t*
      sam_header_merge(int n, const sam_header_t **headers);

    /*!
      @abstract Parse the given text into a SAM Header structure. 
      @param  text  the textual representation of the header
      @return       the initialized SAM Header
     */
    sam_header_t*
      sam_header_parse2(const char *text);

    /*!
      @abstract Returns the textual representation of the current SAM Header 
      @param  h  the SAM Header 
      @return    the SAM Header text
     */
    char*
      sam_header_write(const sam_header_t *h);

    /*!
      @abstract Checks the SAM Header for consistency for required tags in standard record types.
      @param  h  the SAM Header
      @return    1 if the SAM Header is consistent, 0 otherwise
     */
    int32_t
      sam_header_check(sam_header_t *h);

    /*!
      @abstract Populations the BAM Header fields from the internal SAM Header
      @param  h the BAM header
    */
    struct __bam_header_t*
      sam_header_to_bam_header(struct __bam_header_t *h);

#ifdef __cplusplus
}
#endif

#endif
