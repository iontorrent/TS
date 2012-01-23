TESTS = blas++_test check_blas1pp
TESTS_ENVIRONMENT = LD_LIBRARY_PATH=$(top_builddir)/src/.libs:$(top_builddir)/src:${LD_LIBRARY_PATH}
dist_check_SCRIPTS = blas++_test

check_PROGRAMS = tblasd++ check_blas1pp
tblasd___SOURCES = tblasd++.cc
check_blas1pp_SOURCES = check_blas1pp.cc

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include $(all_includes) 
AM_LDFLAGS = $(LDFLAGS) $(LAPACKPP_INTERNAL_LIBFILE)

CLEANFILES = tblasd++*.out
