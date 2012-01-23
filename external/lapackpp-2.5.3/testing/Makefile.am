TESTS = lapack++_test
TESTS_ENVIRONMENT = LD_LIBRARY_PATH=$(top_builddir)/src/.libs:$(top_builddir)/src:${LD_LIBRARY_PATH}
dist_check_SCRIPTS = lapack++_test

EXTRA_DIST = tGenSolve.vcproj

check_PROGRAMS = tGenSolve tSpdSolve tEigSolve tSymmSolve tCmplxSolve

tGenSolve_SOURCES= tGenSolve.cc
tSpdSolve_SOURCES= tSpdSolve.cc
tEigSolve_SOURCES= tEigSolve.cc
tCmplxSolve_SOURCES = tCmplxSolve.cc
tSymmSolve_SOURCES = tSymmSolve.cc

INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include $(all_includes) 
AM_LDFLAGS = $(LDFLAGS) $(LAPACKPP_INTERNAL_LIBFILE)

CLEANFILES = tGenSolve*.out tSpdSolve*.out tEigSolve*.out tCmplxSolve*.out tSymmSolve*.out
