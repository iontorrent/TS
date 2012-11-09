GCC_VERSION := $(shell gcc --version | sed -n "s/.* \([0-9]*\.[0-9]*\.[0-9]*\) .*/\1/p")

RELEASE_FLAGS=-Wall -DNDEBUG -O2 -pipe
DEBUG_FLAGS=-g -Wall -DDEBUG -pipe

CFLAGS?=-g -O1 -Wall -DDEBUG -pipe
CXXFLAGS?=$(CFLAGS)
#override CXXFLAGS+=-pthread
SHLIB_FLAGS=-shared -fPIC
override CXXFLAGS+=$(SHLIB_FLAGS)

THIRDPARTY_ROOT?=$(LIFETECH_ROOT)/../../../../3rdParty
COMMON_INC=$(THIRDPARTY_ROOT)/include
COMMON_LIB=$(THIRDPARTY_ROOT)/lib

HEADER_FLAGS=-I"$(LIFETECH_INC)" -I"$(COMMON_INC)"
LIBRARY_FLAGS=-L"$(LIFETECH_LIB)" -L"$(COMMON_LIB)"

LIFETECH_INC=$(LIFETECH_ROOT)
LIFETECH_LIB=$(LIFETECH_ROOT)/lib
LIFETECH_LIBS=-l lifetech

THIRD_PARTY_LIBS=-l log4cxx -l apr-1 -l aprutil-1 -l expat -l dl
