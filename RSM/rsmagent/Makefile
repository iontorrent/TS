.PHONY:	all debug clean
# usage: 
# make 			builds optimized version
# make debug	builds debug version
# make clean	removes compilation products; do this to change between optimized and debug builds	

CC			= gcc
LD			= gcc
SOURCES		= RSMAgent.c AlarmMgr.c
# code from external sources:
EXTSOURCES	= parson.c
OBJECTS		= $(patsubst %.c,obj/%.o,$(SOURCES))
EXTOBJECTS	= $(patsubst %.c,obj/%.o,$(EXTSOURCES))
DEPS		= $(patsubst %.c,obj/%.d,$(SOURCES)) $(patsubst %.c,obj/%.d,$(EXTSOURCES))
AEDIR		= AgentEmbedded-5.3-252
INCLUDES	= -I$(AEDIR)/Include -I$(AEDIR)/Sysdeps/Unix
DEPLIBS		= $(AEDIR)/libAgentEmbedded.a \
			  $(AEDIR)/Libsrc/expat/xmlparse/libexpat.a \
			  $(AEDIR)/Libsrc/zlib/libz.a
LIBS		= -lssl -lcrypto -lpthread
EXECUTABLE	= RSMAgent_TS

# automatically generate dependency rules
# -MMD write dependency rules to a file (include user header files only, not system header files)
# -MP  add phony target for each dependency but main file (work around make errors for absent headers)
CMNFLAGS	= $(INCLUDES) -Wall -Wextra -ggdb -MMD -MP -DENABLE_LARGEFILE64 -std=c99

all:	CFLAGS	= $(CMNFLAGS) -O3
all:	LDFLAGS	= $(LIBS) -ggdb -O3
all:	$(EXECUTABLE)

debug:	CFLAGS	= $(CMNFLAGS) -O0
debug:  LDFLAGS	= $(LIBS) -ggdb 
debug:	$(EXECUTABLE)

$(EXECUTABLE):	$(OBJECTS) $(EXTOBJECTS) $(DEPLIBS) 
	$(LD) $^ -o $@ $(LDFLAGS)

$(OBJECTS):	| obj

# don't ask for warnings on code from external sources
$(EXTOBJECTS):	CFLAGS = $(INCLUDES) -Wno-unused-result -ggdb -MMD -MP -DENABLE_LARGEFILE64 -O3

$(DEPS):	| obj

$(AEDIR)/libAgentEmbedded.a:
	cd $(AEDIR); $(MAKE)

$(AEDIR)/Libsrc/expat/xmlparse/libexpat.a:
	cd $(AEDIR); $(MAKE)

$(AEDIR)/Libsrc/zlib/libz.a:
	cd $(AEDIR); $(MAKE)

obj:
	mkdir -p $@

obj/%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(EXECUTABLE) $(OBJECTS) $(EXTOBJECTS) $(DEPS)
	rmdir obj
	cd $(AEDIR); $(MAKE) $@

# include automatically generated makefiles that contain dependency rules, 
# so header file changes will cause source files to be recompiled.
-include $(DEPS)
