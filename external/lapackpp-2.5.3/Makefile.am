SUBDIRS = include blaspp matrix src testing macros contrib
ACLOCAL_AMFLAGS = -I macros
EXTRA_DIST = README.W32 README MAINTAINER \
 Doxyfile \
 mainpage.doxygen.in \
 RELEASE.NOTES \
 lapackpp.spec.in \
 lapackpp.spec \
 lapackpp.iss \
 lapackpp.sln \
 lapackpp.vcproj \
 lapackpp-onlyreal.vcproj

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = $(PACKAGE).pc

# Directory where the doxygen output is written into
DOC_DIR = api-doc

# Rule for running doxygen
srcdoc: mainpage.doxygen
	make -C include lacomplex.h
	@if test ! -d $(DOC_DIR); then \
	  mkdir $(DOC_DIR); \
	fi
	doxygen	

# Rule for rpm generation
rpm:    $(PACKAGE).spec dist
	rpmbuild="rpm" && \
	if [ `rpm --version | awk '{ print $$3 }'` > /dev/null ]; then rpmbuild="rpmbuild"; fi && \
	$$rpmbuild -ta $(PACKAGE)-$(VERSION).tar.gz

# Rule for rpm generation
srpm:    $(PACKAGE).spec dist
	rpmbuild="rpm" && \
	if [ `rpm --version | awk '{ print $$3 }'` > /dev/null ]; then rpmbuild="rpmbuild"; fi && \
	$$rpmbuild -ts $(PACKAGE)-$(VERSION).tar.gz

# Rule for using the Inno Setup compiler. Set the path to iscc.exe
# here.
ISCC = "C:\Programme\Inno Setup 5\iscc.exe"
win_buildroot=distribution
setup: lapackpp.iss
	mkdir -p $(win_buildroot)
	make prefix="`pwd`/$(win_buildroot)/win32-tmp" \
	     WIN_PATH_WINDOWS_MINGW="`pwd`/$(win_buildroot)" \
	     install \
	 && $(ISCC) lapackpp.iss \
	 && rm -rf $(win_buildroot)
