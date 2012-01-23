noinst_LTLIBRARIES = libblaspp.la

libblaspp_la_LDFLAGS = -no-undefined
# BLAS_LIBS and FLIBS are only in src/Makefile.am
libblaspp_la_LIBADD =
libblaspp_la_SOURCES = \
 blas1pp.cc blas2pp.cc blas3pp.cc blaspp.cc

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include $(all_includes) 
