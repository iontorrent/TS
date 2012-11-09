SRCDIR=src
TESTDIR=test
EXAMPLEDIR=examples

INSTALL_HDRS_CMD=rsync -aq -C lifetech ../include
INSTALL_LIBS_CMD=cp lib/*.so ../lib

STAGE_CMD= $(INSTALL_HDRS_CMD) && $(INSTALL_LIBS_CMD)

.PHONY: all debug clean test docs clean_docs

all:
	cd $(SRCDIR) && $(MAKE)
	cd $(TESTDIR) && $(MAKE)
	cd $(EXAMPLEDIR) && $(MAKE)
	$(STAGE_CMD)

debug:
	cd $(SRCDIR) && $(MAKE) debug
	cd $(EXAMPLEDIR) && $(MAKE)
	cd $(TESTDIR) && $(MAKE) debug
	$(STAGE_CMD)

clean: clean_docs
	cd $(SRCDIR) && $(MAKE) clean
	cd $(TESTDIR) && $(MAKE) clean
	cd $(EXAMPLEDIR) && $(MAKE) clean
	rm -rf ../include/lifetech

test: all
	cd $(TESTDIR) && $(MAKE) test

docs:
	doxygen doxygen.conf

clean_docs:
	rm -rf doc/html
	rm -rf doc/latex
