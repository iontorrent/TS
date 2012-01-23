TESTS = ref_copy matrix_test
TESTS_ENVIRONMENT = LD_LIBRARY_PATH=$(top_builddir)/src/.libs:$(top_builddir)/src:${LD_LIBRARY_PATH}
dist_check_SCRIPTS = matrix_test

EXTRA_DIST = tgc.vcproj tgd.vcproj ttdd.vcproj tvd.vcproj

check_PROGRAMS = tgc tgd tld tuld tud tuud tvi ttdd tstd trvd tcvd tgfqrc tgfqrd tsybmd tvd \
 ref_copy
tgc_SOURCES = tgc.cc
tgd_SOURCES = tgd.cc
tld_SOURCES = tld.cc
tuld_SOURCES = tuld.cc
tud_SOURCES = tud.cc
tuud_SOURCES = tuud.cc
tvi_SOURCES = tvi.cc
ttdd_SOURCES = ttdd.cc
tstd_SOURCES = tstd.cc
trvd_SOURCES = trvd.cc
tcvd_SOURCES = tcvd.cc
tgfqrc_SOURCES = tgfqrc.cc
tgfqrd_SOURCES = tgfqrd.cc
tsybmd_SOURCES = tsybmd.cc
tvd_SOURCES = tvd.cc
ref_copy_SOURCES = ref_copy.cc

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include $(all_includes) 
AM_LDFLAGS = $(LDFLAGS) $(LAPACKPP_INTERNAL_LIBFILE)

dist_noinst_DATA = tgd.DATA  tld.DATA  tstd.DATA  tud.DATA  \
 tuld.DATA  tuud.DATA  tvi.DATA
CLEANFILES = tgc*.out tgd*.out tld*.out tuld*.out tud*.out \
 tuud*.out tvi*.out ttdd*.out tstd*.out trvd*.out tcvd*.out tvd*.out tsybmd*.out
