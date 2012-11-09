STAGE_LIB_DIR=lib
STAGE_BIN_DIR=bin

LIFEDIR=lifetech
INSTALL=/usr/bin/install

.PHONY: all setup lifetech

all: setup lifetech

setup:
	mkdir -p $(STAGE_BIN_DIR)
	mkdir -p $(STAGE_LIB_DIR)
	
lifetech: 
	cd $(LIFEDIR) && $(MAKE)

install: all
	-mkdir -p $(DESTDIR)$(PREFIX)/bin
	-mkdir -p $(DESTDIR)$(PREFIX)/lib
	$(INSTALL) -m 755 $(STAGE_BIN_DIR)/* $(DESTDIR)$(PREFIX)/bin
	$(INSTALL) -m 755 $(STAGE_LIB_DIR)/* $(DESTDIR)$(PREFIX)/lib

clean:
	cd $(LIFEDIR) && make clean
	rm -rf include
	rm -rf $(STAGE_BIN_DIR)
	rm -rf $(STAGE_LIB_DIR)
	rm -rf share
