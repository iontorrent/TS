/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BAMREAD_H
#define BAMREAD_H

/*
 *  BAMRead.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/21/10.
 *  
 *
 */


//TODO IDEAS
/*
 merge optional fields into 1 function with this signature
	get_optional_field(char& type, char& tag, void& return_type)
 */

//stl
#include <cassert>
#include <tr1/memory>
#include <string>

//project
#include "samutils_types.h"
#include "Cigar.h"
#include "Sequence.h"
#include "Qual.h"
#include "MD.h"

//3rd party
#include "sam.h"
				
using namespace samutils_types;
/**
 *class BAMRead
 *this class is an object representation of an entry for a sam/bam file.  It contains funcitons to retrieve each field
 *from a file following the samtools spec:  http://samtools.sourceforge.net/SAM1.pdf
 *In order to use this type, you need to first construct a BAMReader.  See BAMReader's documentation
 */
class BAMRead {
	
	public:
	/**
	 Default constructor.  all fields will be null.  attempts to call functions will throw null assertions
	 This constructor is required in order to use BAMRead's in STL containers.  
	 */
	BAMRead(): bam_record(), bam_header(NULL) {}
	/**
	 A call to this constructor makes a copy of the bam1_t*, and assumes the BAM_header 
	 reference will be valid for the life time of this object
	 
	 @param bam1_t*				BAM_read	these are c-structures internal to samtools.  detailed in
											samtools* /bam.h.  They are the c-representation of the alignment, and are essentially the
											equivalent of this BAMRead object
	 @param	const bam_header_t*	BAM_header	this is a pointer to a c-structure internal to samtools.  detailed
											in samtools* /bam.h  They contain the header portion of the sam/bam file
	 */
	BAMRead(bam1_t* BAM_read, const bam_header_t* BAM_header): bam_record(bam_dup1(BAM_read), bam_cleanup() ), bam_header(BAM_header){}
	/**
	 This constructor can be used to reduce memory if you're copying BAMRead's.  However, the operator= is overloaded
	 and it's recommended you use that.  
	 @param	std::shared_ptr<bam1_t>	shared_bam_ptr		a shared pointer to the bam1_t.
	 @param	const bam_header_t*		BAM_header			this is a pointer to a c-structure internal to samtools.  detailed
														in samtools* /bam.h  They contain the header portion of the sam/bam file		
	 */
	BAMRead(bam_ptr shared_bam_ptr, const bam_header_t* BAM_header): bam_record(shared_bam_ptr), bam_header(BAM_header) {}
	/**
	 clears internal memory allocated in constructor.  
	 */
	~BAMRead() { bam_record.reset(); }
	
	/**
	 Copy constructor
	 @param BAMRead	const&		a reference to an existing BAMRead
	 */
	BAMRead(BAMRead const& other) : bam_record(other.bam_record), bam_header(other.bam_header) {}

	
	
	/**
	 overloaded operator=
	 calls the copy constructor.  
	 */
	BAMRead& operator=(BAMRead that) {
			
		swap(*this, that);
		return *this;
		
	}
	
	/**
	 
	 */
	bam1_t* get_bam_ptr() const { 
		return bam_record.get(); 
	}
	
		

		/**
		 QNAME: Query template NAME. Reads/fragments having identical QNAME are regarded to
		 come from the same template. A QNAME `*' indicates the information is unavailable.	
		 
		 @return	const char*		a pointer to a string
		 */
		str_ptr						get_qname( ) const;
		/**
		 returns the flag fields of the read.  From spec:
		 FLAG: bitwise FLAG. Each bit is explained in the following table:
		 Bit Description
		 0x1	template having multiple fragments in sequencing
		 0x2	each fragment properly aligned according to the aligner
		 0x4	fragment unmapped
		 0x8	next fragment in the template unmapped
		 0x10	SEQ being reverse complemented
		 0x20	SEQ of the next fragment in the template being reversed
		 0x40	the first fragment in the template
		 0x80	the last fragment in the template
		 0x100	secondary alignment
		 0x200	not passing quality controls
		 0x400	PCR or optical duplicate
		 
		 @return	uint32_t		unsigned encoded 32-bit integer
		 */
		uint32_t					get_flag() const;
		/**
		 RNAME: Reference sequence NAME of the alignment. If @SQ header lines are present, RNAME
		 (if not `*') must be present in one of the SQ-SN tag. An unmapped fragment without coordinate
		 has a `*' at this field. However, an unmapped fragment may also have an ordinary coordinate
		 such that it can be placed at a desired position after sorting. If RNAME is `*', no assumptions
		 can be made about POS and CIGAR.
		 
		 @return	const char*		a pointer to a string
		 */
		str_ptr						get_rname();
		/**
		 1-based leftmost mapping POSition of the first matching base. The first base in a reference
		 sequence has coordinate 1. POS is set as 0 for an unmapped read without coordinate. If POS is
		 0, no assumptions can be made about RNAME and CIGAR.
		 
		 @return	long			1-based position in genome
		 */
		coord_t						get_pos() const;
		/**
		 MAPQ: MAPping Quality. It equals -10 x log10 Pr{mapping position is wrong}, rounded to the
		 nearest integer. A value 255 indicates that the mapping quality is not available.
		 
		 @return	int				integer representing the mapping quality
		 */
		int							get_mapq() const;
		/**
		 CIGAR: CIGAR string. The CIGAR operations are given in the following table (set `*' if un-
		 available)
		 
		 @return	Cigar			an iterable object representation of the CIGAR string
		 */
		Cigar						get_cigar();
		/**
		 (only used for paired reads)
		 RNEXT: Reference sequence name of the NEXT fragment in the template. For the last fragment,
		 the next fragment is the first fragment in the template. If @SQ header lines are present, RNEXT
		 (if not `*' or `=') must be present in one of the SQ-SN tag. This field is set as `*' when the
		 information is unavailable, and set as `=' if RNEXT is identical RNAME. If not `=' and the next
		 fragment in the template has one primary mapping (see also bit 0x100 in FLAG), this field is
		 identical to RNAME of the next fragment. If the next fragment has multiple primary mappings,
		 no assumptions can be made about RNEXT and PNEXT. If RNEXT is `*', no assumptions can
		 be made on PNEXT and bit 0x20.
		 
		 @return	const char*		a string for this read's pair reference it aligned to
									(only used for paired reads)
		 */
		str_ptr						get_rnext();
		/**
		 (only used for paired reads)
		 PNEXT: Position of the NEXT fragment in the template. Set as 0 when the information is
		 unavailable. This field equals POS of the next fragment. If PNEXT is 0, no assumptions can be
		 made on RNEXT and bit 0x20.
		 
		 @return	long			position this read's pair aligns to in genome
		 */
		coord_t						get_pnext() const;
		/**
		 TLEN: signed observed Template LENgth. If all fragments are mapped to the same reference,
		 the unsigned observed template length equals the number of bases from the leftmost mapped
		 base to the rightmost mapped base. The leftmost fragment has a plus sign and the rightmost has
		 a minus sign. The sign of fragments in the middle is undefined. It is set as 0 for single-fragment
		 template or when the information is unavailable.
		 
		 @return	long			length of alignment
		 */
		coord_t						get_tlen() const;
		/**
		 SEQ: fragment SEQuence. This field can be a `*' when the sequence is not stored. If not a `*',
		 the length of the sequence must equal the sum of lengths of M/I/S/=/X operations in CIGAR.
		 An `=' denotes the base is identical to the reference base. No assumptions can be made on the
		 letter cases.
		 
		 return		Sequence		an iterable object representation of the SEQ string
		 */
		Sequence					get_seq();
		/**
		 QUAL: ASCII of base QUALity plus 33 (same as the quality string in the Sanger FASTQ format).
		 A base quality is the phred-scaled base error probability which equals 10 log10 Prfbase is wrongg.
		 This field can be a `*' when quality is not stored. If not a `*', SEQ must not be a `*' and the
		 length of the quality string ought to equal the length of SEQ.
		 
		 return		Qual			an iterable object representation of the QUAL string
		 */
		Qual						get_qual();
		//str_ptr						get_mrnm();
		//coord_t						get_mpos() const;
		
		/**
		 returns the index of the reference this read aligned to from the header.
		 
		 @return	int				an integer index to aligned reference from header
		 */
		int							get_tid() const;
		
		//optional tags
		/**
		 TMAP specific tag.  This returns the alignment algorithm used to map this read
		 
		 @return	const char*		a string representing algorithm used to map read
		 */
		inline str_ptr						get_xa() const {
			assert(bam_record);
			uint8_t* xa = bam_aux_get(bam_record.get(), "XA");
			if (xa) {
				return bam_aux2Z(xa);
			} else {
				return NULL;
			}
			
			
		}
		
		/**
		 TMAP specific tag.  not sure what this one is yet
		 
		 @return	long			not sure yet
		 */
		
		inline coord_t						get_xs() const {
			uint8_t *xs = bam_aux_get(bam_record.get(), "XS");
			if (xs != NULL) {
				return bam_aux2i(xs);
			} else {
				return -1; 
			}
			
		}
		/**
		 TMAP specific tag.  not sure what this one is yet

		 @return	long			not sure yet
		 */
		inline coord_t						get_xt() const{
			uint8_t *xt = bam_aux_get(bam_record.get(), "XT");
			if (xt != NULL) {
				return bam_aux2i(xt);
			} else {
				return -1; 
			}
			
		}
	
	
		
		/**
		 Alignment score generated by aligner
		 
		 @return	int			score
		 */
		int							get_as();
		/**
		 Query hit index, indicating the alignment record is the i-th one stored in SAM
		 
		 @return	int			i-th sam record	
		 */
		int							get_hi();
		/**
		 Number of stored alignments in SAM that contains the query in the current record
		 
		 @return	int			num of sam records
		 */
		int							get_ih();
		/**
		 Program. Value matches the header PG-ID tag if @PG is present
		 
		 @return	char*		aligner name
		 */
		str_ptr						get_pg();
		/**
		 Template-independent mapping quality
		 
		 @return	int			mapping quality
		 */
		int							get_sm();
		/**
		 Read group. Value matches the header RG-ID tag if @RG is present in the header.
		 
		 @return	char*		read group
		 */
		str_ptr						get_rg();
		/**
		 Edit distance to the reference, including ambiguous bases but excluding clipping
		 
		 
		 @return	int			hamming distance to ref
		 */
		int						get_nm();
		/**
		 Number of reported alignments that contains the query in the current record
		 
		@return		int			num of alignments
		 */
		int						get_nh(); 
	
		/**
		 The MD field aims to achieve SNP/indel calling without looking at the reference. For example, a string `10A5^AC6'
		 means from the leftmost reference base in the alignment, there are 10 matches followed by an A on the reference which
		 is different from the aligned read base; the next 5 reference bases are matches followed by a 2bp deletion from the
		 reference; the deleted sequence is AC; the last 6 bases are matches. The MD field ought to match the CIGAR string.
		 
		 String for mismatching positions. Regex : [0-9]+(([A-Z]|\^[A-Z]+)[0-9]+)
		 
		 @return	MD			iterable object representing MD string
		 */
		MD							get_md(); 
		
	
	
	
		//utilities
		/**
		 Returns true if the read is mapped
		 @return	bool			true if read is mapped
		 */
		bool						is_mapped() const;
		/**
		 Returns true if the read maps to only 1 location in the reference(s)
		 
		 @return	bool			true if uniquely mapped
		 */
		bool						is_mapped_unique() ;
		/**
		 Returns true if the alignment has a pair in this sam/bam file
		 
		 @return	bool			true if alignment is paired
		 */
		bool						is_paired() const;
		/**
		 Returns true if alignment is a proper pair.  A proper pair means the reads are
		 aligned to the same strand, and the same orientation.  
		 
		 @return	bool			true if proper pair
		 */
		bool						proper_pair() const;
		/**
		 Returns true if alignment is mapped to the reverse strand (3')
		 
		 @return	bool			true if 3' mapped
		 */
		bool						mapped_reverse_strand() const;
		/**
		 Returns true if this alignment's pair is mapped to the reverse strand
		 
		 @return	bool			true if mate is mapped to 3'
		 */
		bool						mate_mapped_reverse_strand() const;
		/**
		 Returns true if this is the first alignment in the pair
		 
		 @return	bool			true if read 1
		 */
		bool						is_read1()	const;
		/**
		 Returns true if this is the second alignment in the pair
		 
		 @return	bool			true if read 2
		 */
		bool						is_read2()	const;
		/**
		 Returns true if this is the primary alignment for this read
		 
		 @return	bool			true if primary alignment
		 */
		bool						is_primary()	const;
		/**
		 Returns true if this is a PCR duplicate 
		 
		 @return	bool			true if duplicate
		 */
		bool						is_duplicate() const;
		/**
		 Returns length of reference that this read aligns to.  
		 
		 @return	long			reference length
		 */
		coord_t						get_ref_len();
		/**
		 This function returns the right most position of the read.  It uses the cigar string
		 to do so.  
		 
		 @return	int				right most position of read
		 */
		int							calend() const;
	
		/**
		 Returns a valid sam style format of this BAMRead
		 
		 @return	string			sam format string
		 */
		std::string						to_string();
		
	
		
	
	private:
		void swap(BAMRead& first, BAMRead& second) {
			using std::swap; //enables ADL
			swap(first.bam_record, second.bam_record);
			swap(first.bam_header, second.bam_header);
		}
		bam_ptr				bam_record;
		const bam_header_t*			bam_header;
		
	
	void get_optional_field(std::string& tag, int& opt_int) {
		
		uint8_t *opt_ptr = bam_aux_get(bam_record.get(), tag.c_str());
		if (opt_ptr) {
			opt_int = bam_aux2i(opt_ptr);
		}  else {
			opt_int = -1;
		}

		
	}
	
	str_ptr 	get_optional_field(std::string& tag){
		
		
		uint8_t *opt_ptr = bam_aux_get(bam_record.get(), tag.c_str());
		str_ptr opt_str = NULL;
		if (opt_ptr) {
			opt_str = bam_aux2Z(opt_ptr);
		} 
		return opt_str;
		
	}

	

		//utility functions
	void			_init();
	std::string		to_string_1_2();
	std::string		to_string_1_3();
	
};

inline str_ptr BAMRead::get_qname() const  {
	return bam1_qname(bam_record);
	//	return qname;
	
}
inline 
uint32_t BAMRead::get_flag() const {
	if (bam_record != NULL) {
		return bam_record->core.flag;
		
	} else {
		return -1; //failed
	}
	
	
}

inline
str_ptr  BAMRead::get_rname()  {
	assert(bam_record);
	
	int32_t tid = bam_record->core.tid;	
	if (tid == -1) {
		return 0; //null value
	} else {
		//bam_header_t* bht = itr_file->header;
		return bam_header->target_name[tid];
		
	}
	
	
}


inline
coord_t BAMRead::get_ref_len() {
	assert(bam_record);
	int32_t tid = bam_record->core.tid;
	if (tid == -1) {
		return 0;
	} else {
		return bam_header->target_len[tid];
	}
	
	
}

inline
int	BAMRead::get_tid() const {
	assert(bam_record);
	return bam_record->core.tid;
}

inline
coord_t	BAMRead::get_pos()   const {
	return bam_record->core.pos + 1; //1 based
}


inline
int	BAMRead::get_mapq()  const {
	if (bam_record == NULL) {
		return -1;	
	} else {
		int ret = bam_record->core.qual;
		if (ret > 0) {
			return ret;
		} else {
			return 0;
		}
		
	}
}

inline
coord_t BAMRead::get_pnext() const {	
	assert(bam_record);
	if (bam_record->core.mpos == -1) {
		return 0;
	} else {
		return bam_record->core.mpos;
	} 
	
}


inline
coord_t BAMRead::get_tlen()	const {
	int32_t tln = bam_record->core.l_qseq;
	
	return tln;
}

/*** optional tags ***
 -have to implement MD at very least
 */

inline
str_ptr BAMRead::get_pg()	 {
	
	assert(bam_record);
	str_ptr pg;
	std::string tag("PG");
	pg = get_optional_field(tag);
	
	/*
	uint8_t *pg_ptr = bam_aux_get(bam_record.get(), "PG");
	
	if (pg_ptr) {
		pg = bam_aux2Z(pg_ptr);
	} */
	return pg;
	
}

inline
int BAMRead::get_ih()	 {
	assert(bam_record);
	int ih = -1;
	std::string tag("IH");
	get_optional_field(tag, ih);
	return ih;
	/*
	uint8_t* ih = bam_aux_get(bam_record.get(), "IH");
	if (ih) {
		return bam_aux2i(ih);
	} else {
		return -1;
	}*/
	
	
}
inline
int BAMRead::get_hi()	 {
	assert(bam_record);
	
	int hi = -1;
	std::string tag("HI");
	get_optional_field(tag, hi);
	return hi;
	/*
	uint8_t* hi = bam_aux_get(bam_record.get(), "HI");
	if (hi) {
		return bam_aux2i(hi);
	} else {
		return -1;
	}*/
	
	
	
}


inline
int	BAMRead::get_sm()  {
	
	assert(bam_record);
	int sm = -1;
	std::string tag("SM");
	get_optional_field(tag, sm);
	return sm;
	
	/*
	uint8_t *sm = bam_aux_get(bam_record.get(), "SM");
	if (sm != NULL) {
		return bam_aux2i(sm);
	} else {
		return -1; 
	}*/
}
inline
str_ptr	BAMRead::get_rg()  {
	assert(bam_record);
	str_ptr rg;
	std::string tag("RG");
	rg = get_optional_field(tag);
	/*
	uint8_t *rg_ptr = bam_aux_get(bam_record.get(), "RG");
	str_ptr rg;
	if (rg_ptr) {
		rg = bam_aux2Z(rg_ptr);
	} */
	
	return rg;
}


inline
int BAMRead::get_nm()  {
	assert(bam_record);
	int nm = -1;
	std::string tag("NM");
	get_optional_field(tag, nm);
	return nm;
	/*
	uint8_t *nm = bam_aux_get(bam_record.get(), "NM");
	if (nm) {
		return bam_aux2i(nm);
	} else{
		return -1;
	}*/
}

inline
int BAMRead::get_nh()  {
	assert(bam_record);
	int nh = -1;
	std::string tag ="NH";
	get_optional_field(tag, nh);
	return nh;
	/*
	uint8_t* nh = bam_aux_get(bam_record.get(), "NH");
	if (nh) {
		return bam_aux2i(nh);
	} else {
		return -1;
	}*/
	
}

inline 
int	BAMRead::get_as()	{
	assert(bam_record);
	int as = -1;
	std::string tag = "AS";
	get_optional_field(tag, as);
	return as;

	/*
	uint8_t *pg = bam_aux_get(bam_record.get(), "AS");
	if (pg) {
		// caveat this is a kludge have to talk to nils
		int32_t ret = bam_aux2i(pg);
		//"-2147483648"
		if (ret == -2147483648) {
			return ret;
		} else {
			return ret;
		}
		
	} else {
		return -1;
	}
	*/
	
}

inline
MD	BAMRead::get_md()  {
	assert(bam_record);
	
	return MD(bam_record.get());
	
	
	
}

/* utils */
inline
bool BAMRead::is_mapped() const {
	
	//return (nh() > 0);
	assert(bam_record);
	return !(BAM_FUNMAP & bam_record->core.flag);
	
}
inline
bool BAMRead::is_mapped_unique()  {
	
	return(get_nh() == 1);
}

inline
bool BAMRead::is_paired() const {
	assert(bam_record);
	return BAM_FPAIRED & bam_record->core.flag;
	
}
inline
bool BAMRead::proper_pair() const {
	assert(bam_record);
	return BAM_FPROPER_PAIR & bam_record->core.flag;
	
	
}
inline
bool BAMRead::mapped_reverse_strand() const {
	assert(bam_record);
	return BAM_FREVERSE & bam_record->core.flag;
	
}
inline
bool BAMRead::mate_mapped_reverse_strand() const {
	assert(bam_record);
	return BAM_FMREVERSE & bam_record->core.flag;
	
}
inline
bool BAMRead::is_read1()	const {
	
	assert(bam_record);
	return BAM_FREAD1 & bam_record->core.flag;
	
}
inline
bool BAMRead::is_read2()	const {
	assert(bam_record);
	return BAM_FREAD2 & bam_record->core.flag;
	
	
	
}
inline
bool BAMRead::is_primary()	const {
	assert(bam_record);
	return !(BAM_FSECONDARY & bam_record->core.flag);
	
}

inline
bool BAMRead::is_duplicate()	const {
	assert(bam_record);
	return BAM_FDUP & bam_record->core.flag;
	
}

inline
int BAMRead::calend() const {
	
	assert(bam_record);
	const bam1_core_t* c_tmp = &(bam_record->core);
	const bam1_t* b_tmp = bam_record.get();
	return bam_calend(c_tmp, bam1_cigar(b_tmp));
	
	
	
}



#endif //BAMREAD_H


