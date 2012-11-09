CC=			gcc
CFLAGS=		-g -Wall -O2 #-m64 
DFLAGS=		
AOBJS=		ion_error.o ion_alloc.o \
			ion_string.o \
			dat_io.o ion_util.o \
			dat_header.o dat_frame.o \
			dat_flow.o dat_chip.o \
			wells_header.o wells_data.o \
			wells_mask.o wells_chip.o \
			wells_combine.o wells_mask_combine.o \
			sff_header.o sff_read_header.o \
			sff_read.o sff_file.o sff.o \
			sff_index.o sff_sort.o sff_check.o \
			sff_iter.o fastq.o fastq_file.o \
                        main.o
PROG=		iontools	
INCLUDES=
SUBDIRS= .	
LIBPATH= -lm

.SUFFIXES:.c .o

.c.o:
		$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) $< -o $@

all-recur lib-recur clean-recur cleanlocal-recur install-recur:
		@target=`echo $@ | sed s/-recur//`; \
		wdir=`pwd`; \
		list='$(SUBDIRS)'; for subdir in $$list; do \
			cd $$subdir; \
			$(MAKE) CC="$(CC)" DFLAGS="$(DFLAGS)" CFLAGS="$(CFLAGS)" \
				INCLUDES="$(INCLUDES)" LIBPATH="$(LIBPATH)" $$target || exit 1; \
			cd $$wdir; \
		done;

all:$(PROG)

.PHONY:all lib clean cleanlocal
.PHONY:all-recur lib-recur clean-recur cleanlocal-recur install-recur

lib:libdat.a

iontools:$(AOBJS) 
	        $(CC) $(CFLAGS) -o $@ $(AOBJS) $(LIBPATH) 

cleanlocal:
	        rm -fr gmon.out *.o a.out $(PROG) *~ *.a *.so.* *.so *.dylib

clean:cleanlocal-recur

#libdat.a:$(AOBJS)
#	ar rcs libdat.a $(AOBJS)
