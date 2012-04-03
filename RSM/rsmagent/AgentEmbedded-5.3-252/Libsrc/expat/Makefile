### Path settings
srcdir = .
prefix = $(TOP_DIR)/lib
libdir = ${prefix}/lib
includedir = ${prefix}/include

CC=gcc
# If you know what your system's byte order is, define XML_BYTE_ORDER:
# use -DXML_BYTE_ORDER=12 for little-endian byte order;
# use -DXML_BYTE_ORDER=21 for big-endian (network) byte order.
# -DXML_NS adds support for checking of lexical aspects of XML namespaces spec
# -DXML_MIN_SIZE makes a smaller but slower parser
# -DXML_DTD adds full support for parsing DTDs
CFLAGS=$(EXTRA_CFLAGS) -O2 -Ixmltok -Ixmlparse -DXML_MIN_SIZE
ifeq ($(BIG_ENDIAN),yes)
	CFLAGS += -DXML_BYTE_ORDER=21
endif

AR=ar
LIBOBJS=xmltok/xmltok.o \
  xmltok/xmlrole.o \
  xmlparse/xmlparse.o

LIB=xmlparse/libexpat.a

all: $(LIB)

$(LIB): $(LIBOBJS)
	$(AR) rc $(LIB) $(LIBOBJS)

clean:
	rm -f $(LIBOBJS) $(LIB)

distclean: clean

install: all
	mkdir -p $(libdir)
	ln -f $(LIB) $(libdir)
	mkdir -p $(includedir)
	ln -f xmlparse/xmlparse.h $(includedir)

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
