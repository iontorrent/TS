# $Id: Makefile,v 1.8 2008/05/21 18:24:36 dkhodos Exp $

CFLAGS = -O2 -Wall -I. -I$(TOP_DIR)/Include -I$(TOP_DIR)/Sysdeps/Unix $(EXTRA_CFLAGS)
LDFLAGS = $(EXTRA_LDFLAGS) -L$(TOP_DIR) -L$(TOP_DIR)/lib/lib

DEMOS = AeDemo1 AeDemo2 AeDemo3 AeDemo4 AeDemo5
OBJS = AeDemoCommon.o
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

.PHONY: all dep clean distclean

all: $(DEMOS)

clean:
	rm -f *.o $(DEMOS)
ifeq ($(PLATFORM),uClinux-m68k-coff)
	rm -f *.coff
endif

distclean: clean

dep:
	rm -f .dep
	$(CC) -MM $(CFLAGS) *.c > .dep

AeDemo%: $(OBJS) AeDemo%.o $(TOP_DIR)/libAgentEmbedded.a
	$(CC) $(LDFLAGS) -o $@ $(OBJS) AeDemo$*.o $(LIBS)

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

ifneq ($(MAKECMDGOALS), clean)
-include .dep
endif
