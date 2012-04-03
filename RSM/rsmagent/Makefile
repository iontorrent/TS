# $Id: Makefile,v 1.20.2.3 2010/11/05 17:59:34 dkhodos Exp $

# Specify target platform settings and set of features

ifdef WIND_PLATFORM
#PPC 8560 cross-compile settings
PLATFORM = linux-ppc
CROSS_COMPILE = ppc_85xx-
EXTRA_CFLAGS =
EXTRA_LDFLAGS =
BIG_ENDIAN = yes
HAVE_OPENSSL = yes
HAVE_PTHREADS = yes
ENABLE_LARGEFILE64 = no
else
# Linux/i368 settings
PLATFORM = linux-i386
CROSS_COMPILE =
EXTRA_CFLAGS =
EXTRA_LDFLAGS =
BIG_ENDIAN = no
HAVE_OPENSSL = yes
HAVE_PTHREADS = yes
ENABLE_LARGEFILE64 = yes
endif

# uClinux/m68k-pic-coff settings
#PLATFORM = uClinux-m68k-coff
#CROSS_COMPILE = m68k-pic-coff-
#EXTRA_CFLAGS = -fPIC -fomit-frame-pointer
#EXTRA_LDFLAGS = -L/opt/uClinux/m68k-pic-coff/lib
#BIG_ENDIAN = yes
#HAVE_OPENSSL = no
#HAVE_PTHREADS = no
#ENABLE_LARGEFILE64 = no

# uClinux/m68k-pic-elf settings
#PLATFORM = uClinux-m68k-elf
#CROSS_COMPILE = m68k-elf-
#EXTRA_CFLAGS = -fomit-frame-pointer -msep-data -m68000 -Dm68k
#EXTRA_LDFLAGS = -m68000 -Wl,-elf2flt -Wl,-move-rodata
#BIG_ENDIAN = yes
#HAVE_OPENSSL = no
#HAVE_PTHREADS = no
#ENABLE_LARGEFILE64 = no

# Solaris/SPARC settings
#PLATFORM = sun-sparc
#CROSS_COMPILE =
#EXTRA_CFLAGS =
#EXTRA_LDFLAGS =
#BIG_ENDIAN = yes
#HAVE_OPENSSL = no
#HAVE_PTHREADS = yes
#ENABLE_LARGEFILE64 = yes

# The following features are available:
#	-D_REENTRANT				- MT support
#	-DENABLE_SSL				- SSL support (requires openssl)
#	-DENABLE_REMOTE_SESSION		- remote session support 
#	-DENABLE_FILE_TRANSFER		- file transfer support

FEATURES = -D_REENTRANT -DENABLE_REMOTE_SESSION -DENABLE_FILE_TRANSFER -DENABLE_SSL
#FEATURES = -D_REENTRANT -DENABLE_REMOTE_SESSION -DENABLE_FILE_TRANSFER

CONFIG =
ifeq ($(BIG_ENDIAN),yes)
	CONFIG += -DAE_BIG_ENDIAN
endif
ifeq ($(HAVE_OPENSSL),yes)
	CONFIG += -DHAVE_OPENSSL
endif
ifeq ($(HAVE_PTHREADS),yes)
	CONFIG += -DHAVE_PTHREADS
endif
ifeq ($(ENABLE_LARGEFILE64),yes)
	CONFIG += -DENABLE_LARGEFILE64
	CONFIG += -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64
endif

TOP_DIR = $(shell pwd)
AE_DIR = $(TOP_DIR)/AgentEmbedded-5.3-252
AE_LIB = $(AE_DIR)/libAgentEmbedded.a
LIB_DIR = $(AE_DIR)/lib

EXTRA_CFLAGS +=  -I$(AE_DIR)/Sysdeps/Unix
#CFLAGS = -O2 -Wall -I. -I$(TOP_DIR)/$(AE_SRC_DIR) -I$(TOP_DIR)/Include -I$(TOP_DIR)/Compat -I$(TOP_DIR)/Sysdeps/Unix -I$(LIB_DIR)/include $(EXTRA_CFLAGS) $(CONFIG) $(FEATURES)
CFLAGS = -g -Wall -I. -I$(AE_DIR)/Include -I$(AE_DIR)/Compat -I$(AE_DIR)/Sysdeps/Unix -I$(LIB_DIR)/include $(EXTRA_CFLAGS) $(CONFIG) $(FEATURES)
LDFLAGS = -g $(EXTRA_LDFLAGS) -L$(LIB_DIR)/lib -L$(AE_DIR)

CC = $(CROSS_COMPILE)gcc
AR = $(CROSS_COMPILE)ar
RANLIB = $(CROSS_COMPILE)ranlib

MAKE_FLAGS = CC=$(CC) AR=$(AR) RANLIB=$(RANLIB) TOP_DIR=$(AE_DIR) PLATFORM=$(PLATFORM) EXTRA_CFLAGS="$(EXTRA_CFLAGS)" EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" BIG_ENDIAN=$(BIG_ENDIAN) HAVE_OPENSSL=$(HAVE_OPENSSL) HAVE_PTHREADS=$(HAVE_PTHREADS) ENABLE_LARGEFILE64=$(ENABLE_LARGEFILE64)

.PHONY: all clean distclean

AE_DIR = ./AgentEmbedded-5.3-252
INCLUDES=-I$(AE_DIR)/Include -I$(AE_DIR)/Sysdeps/Unix $(EXTRA_CFLAGS)
CFLAGS=-c -Wall -g
LDFLAGS=-g $(EXTRA_LDFLAGS) -L$(AE_DIR) -L$(AE_DIR)/lib/lib

RSMAGENT_OBJS=RSMAgent.o

OBJECTS= $(RSMAGENT_OBJS)
SOURCES=$(ALL_OBJS:.o=.c)

LIBS = -lAgentEmbedded -lexpat -lz

ifeq ($(HAVE_OPENSSL),yes)
        LIBS += -lssl -lcrypto
else
        LIBS += -ldes
endif
ifeq ($(HAVE_PTHREADS),yes)
        LIBS += -lpthread
endif
ifeq ($(ENABLE_LARGEFILE64),yes)
        CFLAGS += -DENABLE_LARGEFILE64
endif
ifeq ($(PLATFORM),sun-sparc)
        LIBS += -lsocket -lnsl
endif
LIBS += -lm -lc

EXECUTABLES = RSMAgent_TS 

all: .depend $(SOURCES) $(EXECUTABLES)
	
$(AE_LIB):
	$(MAKE) -C $(AE_DIR) all

RSMAgent_TS: $(RSMAGENT_OBJS) $(AE_LIB)
	$(CC) $(LDFLAGS) $(RSMAGENT_OBJS) -o $@ $(AE_LIB) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

.depend:
	$(CC) -MM $(SOURCES) $(INCLUDES) > .depend

-include .depend

clean:
	rm -rf $(LIB_DIR)
	$(MAKE) -C $(AE_DIR) $@
	rm -f $(OBJECTS) $(EXECUTABLES)

