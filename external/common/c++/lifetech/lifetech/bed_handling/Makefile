# Makefile for diBayesMerged
THIRDPARTY_ROOT ?=../../../../../../../diBayes_bsp2.0/3rdParty
SAMINC=$(THIRDPARTY_ROOT)/include/samtools
SAMDIR=../../../../../../../diBayes_bsp2.0/source/dibayes/src/diBayes/../../../sam_format/c/src
SAMITA_ROOT=../../../../../../../diBayes_bsp2.0/source/dibayes/src/diBayes/../../../samita

ifeq ($(origin THIRDPARTY_ROOT), undefined)
THIRD_PARTY=../../../../../../../diBayes_bsp2.0/source/dibayes/src/diBayes/../../3rdParty
COMMON_INC=$(SAMITA_ROOT)/$(THIRD_PARTY)/include
COMMON_LIB=$(SAMITA_ROOT)/$(THIRD_PARTY)/lib
else
COMMON_INC=$(THIRDPARTY_ROOT)/include
COMMON_LIB=$(THIRDPARTY_ROOT)/lib
endif

COMMON_CPP_ROOT=../../../../../../../diBayes_bsp2.0/source/dibayes/src/diBayes/../../../common/c++
LIFETECH_INC=$(COMMON_CPP_ROOT)/include
LIFETECH_LIB=$(COMMON_CPP_ROOT)/lib

LOGGER_INC=../../../../../../../diBayes_bsp2.0/source/common/c++/lifetech/src/logging

SAMITA_INC=$(SAMITA_ROOT)
SAMITA_LIB=$(SAMITA_ROOT)/lib

DIBAYE_INC=-I./utils -I./errorprofiles -I./input -I./basespace -I./output -I./colorspace

HEADER_FLAGS=-I$(SAMITA_INC) -I$(COMMON_INC) -I$(LIFETECH_INC) -I$(SAMDIR) -I$(SAMINC) -I$(LOGGER_INC)
LIBRARY_FLAGS=-L$(SAMITA_LIB) -L$(COMMON_LIB) -L$(LIFETECH_LIB) -L$(SAMDIR)

SAMITA_LIBS=-l samita -l samita++

# Required for samita++
HDF5_LIBS=-l hdf5 -l hdf5_hl

THIRD_PARTY_LIBS=-l z -l bam -l boost_iostreams -l boost_thread-mt -l boost_system -l boost_regex -l boost_filesystem -l log4cxx -l apr-1 -l aprutil-1 -l expat -l dl -l m -lpthread 

CC ?= g++
DEBUG =
#OPTFLAG = -O2 -D MERGED
# flag -D NO_BIN in next line will eliminate the creation of quality.bin files
# flag -D  DO_PIPES will eliminate maq files
# flag -D  USE_INDEX will use bam index file (created by command: samtools index bamfilename)
# currently  DO_PIPES and USE_INDEX should be present or absent together in next line
OPTFLAG = -Wall -g -O2  -DNDEBUG -D MERGED -D NO_BIN -D DO_PIPES -D USE_INDEX -fPIC -pthread
#OPTFLAG = -g -O2 -D MERGED -D NO_BIN -D USE_INDEX
#OPTFLAG = -O2 -D MERGED -D NO_BIN -D USE_INDEX

EXE = BedValidator

all: $(EXE)

clean:
	rm -fv *.o  $(EXE)

$(EXE) : BedValidator.o
	$(CC) $(OPTFLAG)  $(DEBUG)  BedValidator.o $(HEADER_FLAGS) $(LIBRARY_FLAGS) $(DIBAYE_INC) $(SAMITA_LIBS)  $(HDF5_LIBS) $(THIRD_PARTY_LIBS) -o BedValidator

BedValidator.o: BedValidator.cpp BedValidator.h 
	$(CC) $(OPTFLAG) $(DEBUG)  -c BedValidator.cpp  $(HEADER_FLAGS) $(DIBAYE_INC) $(LIBRARY_FLAGS)

