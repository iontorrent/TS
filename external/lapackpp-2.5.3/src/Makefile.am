# Sources
liblapackpp_la_SOURCES = \
 dopla.c \
 dtimmg.c \
 eigslv.cc \
 genmd.cc \
 laprefs.cc \
 lasvd.cc \
 lautil.cc \
 linslv.cc \
 systime.c

# Includes
INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include $(all_includes) 

# Installed libraries
lib_LTLIBRARIES = liblapackpp.la

# Dependencies for all systems
liblapackpp_la_DEPENDENCIES = \
	$(top_builddir)/matrix/src/liblamatrixpp.la \
	$(top_builddir)/blaspp/src/libblaspp.la

# if IS_WINDOWS
if IS_WINDOWS
# Dependencies for Windows
liblapackpp_la_DEPENDENCIES += ressource.o
ressource_ldflags = -Wl,ressource.o -Wl,-export-all-symbols
endif
# IS_WINDOWS

# Linker flags
liblapackpp_la_LDFLAGS = -no-undefined \
  -version-info \
  $(LAPACKPP_SO_CURRENT):$(LAPACKPP_SO_REVISION):$(LAPACKPP_SO_AGE) \
  $(ressource_ldflags)
# Linker libraries
liblapackpp_la_LIBADD = \
  $(top_builddir)/matrix/src/liblamatrixpp.la \
  $(top_builddir)/blaspp/src/libblaspp.la \
  $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)

# ressource must be packages here because MSVC won't run configure
EXTRA_DIST = ressource.rc

ressource.o: ressource.rc
	$(RC) -i $< -o $@

