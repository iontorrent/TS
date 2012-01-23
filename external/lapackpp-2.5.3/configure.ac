dnl -*-autoconf-*-
dnl Process this file with autoconf to produce a configure script.

dnl Start autoconf processing. 
AC_INIT

dnl Checking host/target/build systems, for make, install etc.
AC_CANONICAL_TARGET
dnl Perform program name transformation
AC_ARG_PROGRAM


#
# versions
#
LAPACKPP_VERSION_MAJOR=2
LAPACKPP_VERSION_MINOR=5
LAPACKPP_VERSION_PATCHLEVEL=3
rpm_release="1"
LAPACKPP_VERSION_STRING="$LAPACKPP_VERSION_MAJOR.$LAPACKPP_VERSION_MINOR.$LAPACKPP_VERSION_PATCHLEVEL"

dnl Automake doc recommends to do this only here.
AM_INIT_AUTOMAKE(lapackpp,"$LAPACKPP_VERSION_STRING")

# SO version of all liblapackpp.so
#
LAPACKPP_SO_CURRENT=16
LAPACKPP_SO_REVISION=0
LAPACKPP_SO_AGE=2

# Substitute these vars
AC_SUBST(LAPACKPP_VERSION_MAJOR)
AC_SUBST(LAPACKPP_VERSION_MINOR)
AC_SUBST(LAPACKPP_VERSION_PATCHLEVEL)
AC_SUBST(LAPACKPP_VERSION_STRING)
AC_SUBST(rpm_release)
AC_SUBST(LAPACKPP_SO_CURRENT)
AC_SUBST(LAPACKPP_SO_REVISION)
AC_SUBST(LAPACKPP_SO_AGE)

AC_DEFINE_UNQUOTED(LAPACKPP_VERSION_MAJOR,$LAPACKPP_VERSION_MAJOR, 
                   [major version])
AC_DEFINE_UNQUOTED(LAPACKPP_VERSION_MINOR,$LAPACKPP_VERSION_MINOR, 
                   [minor version])
AC_DEFINE_UNQUOTED(LAPACKPP_VERSION_PATCHLEVEL,$LAPACKPP_VERSION_PATCHLEVEL, 
                   [patchlevel])
AC_DEFINE_UNQUOTED(LAPACKPP_VERSION_STRING,"$LAPACKPP_VERSION_STRING", 
                   [version string])

dnl set the program name for make
AC_PROG_MAKE_SET

dnl make /usr/local the default for the installation
AC_PREFIX_DEFAULT(/usr/local)
if test "x$prefix" = "xNONE"; then
  prefix=$ac_default_prefix
  ac_configure_args="$ac_configure_args --prefix $prefix"
fi
AC_SUBST(prefix)

dnl without this order in this file, automake will be confused!
AC_CONFIG_HEADER(config.h)
AM_SANITY_CHECK

### -------------------------------------------------------------

dnl checks for programs.
AC_PROG_CXX
AC_PROG_CPP
AC_PROG_INSTALL
AC_ENABLE_FAST_INSTALL
AC_PATH_PROG([SED], [sed])

dnl Turn off libtool static library generation
AC_DISABLE_STATIC
AC_LIBTOOL_WIN32_DLL
AC_LIBTOOL_RC
AC_PROG_LIBTOOL

dnl Check for structs, compiler characteristics
AC_C_CONST

dnl Check for headers
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h unistd.h])

dnl Check for C++ headers
AC_LANG_PUSH(C++)
AC_CHECK_HEADERS([bits/cpp_type_traits.h], [HAVE_BITS_CPP_TYPE_TRAITS_H=1
	], [HAVE_BITS_CPP_TYPE_TRAITS_H=0])
AC_LANG_POP(C++)
AC_CXX_NAMESPACES
AC_SUBST(HAVE_BITS_CPP_TYPE_TRAITS_H)

### -------------------------------------------------------------
### Specific packages
### -------------------------------------------------------------

case "$target" in 
    *-mingw32*) 
	ACX_WINDOWS_PATHS
    ;; 
esac


dnl Check for Lapack
ACX_LAPACK([
  dnl action-if-found: 
  AC_DEFINE(HAVE_LAPACKPP,1,[Define if you have LAPACK++ library.])
  # This defines LAPACK_LIBS, BLAS_LIBS, and FLIBS
], [
  dnl action-if-not-found: 
  AC_MSG_ERROR([Blas/Lapack was not found. 
*** This means Lapack++ and matrix support cannot be compiled. 
*** This makes this library unusable. Please get blas and lapack 
*** installed. If you do have these installed, use the options 
*** --with-blas=<libname> or --with-lapack=<libname> and/or set 
*** the env variable LDFLAGS to include the appropriate linker 
*** flags.])
])

dnl Workaround for bogus FLIBS:
FLIBS=`echo ${FLIBS} | ${SED} 's/-lgfortranbegin//' `

#dnl Modern fortran compiler flags
AC_FC_LIBRARY_LDFLAGS
AC_FC_DUMMY_MAIN

### -------------------------------------------------------------
### Check the return value behaviour of zdotc()
### -------------------------------------------------------------
save_LIBS=$LIBS
LIBS="$LIBS $BLAS_LIBS $FLIBS"

# Try the return value on stack (will segfault if wrong zdotc prototype)
AC_MSG_CHECKING(whether zdotc() returns result on stack)
AC_RUN_IFELSE(
[AC_LANG_PROGRAM([[
 #define F77NAME(x) x##_
 typedef struct { double r, i; } doublecomplex;
 doublecomplex F77NAME(zdotc)(
		   const int *n, 
		   const doublecomplex *cx, 
		   const int *incx, const doublecomplex *cy,
		   const int *incy);
    ]],
		 [[
    doublecomplex cx[1], cy[1], tmp;
    int n = 1, incx = 1, incy = 1;
    cx[0].r = 2; cx[0].i = 0;
    cy[0].r = 3; cy[0].i = 0;
    tmp = F77NAME(zdotc)(&n, cx, &incx, cy, &incy);
    /* printf("Result: (%f,%f)\n", tmp.r, tmp.i); */
    return 0; /* (tmp.r == 6) ? 0 : 1; */
    ]])],
[res_with_stack=yes],
[res_with_stack=no],
[res_with_stack=unknown])
AC_MSG_RESULT($res_with_stack)

# And now the return value as first argument (will segfault if wrong
# zdotc prototype)
AC_MSG_CHECKING(whether zdotc() returns result as first argument)
AC_RUN_IFELSE(
[AC_LANG_PROGRAM([[
 #define F77NAME(x) x##_
 typedef struct { double r, i; } doublecomplex;
 void F77NAME(zdotc)(doublecomplex *c, 
		   const int *n, 
		   const doublecomplex *cx, 
		   const int *incx, const doublecomplex *cy,
		   const int *incy);
    ]],
		 [[
    doublecomplex cx[1], cy[1], tmp;
    int n = 1, incx = 1, incy = 1;
    cx[0].r = 2; cx[0].i = 0;
    cy[0].r = 3; cy[0].i = 0;
    F77NAME(zdotc)(&tmp, &n, cx, &incx, cy, &incy);
    /* printf("Result: (%f,%f)", tmp.r, tmp.i); */
    return 0; /* (tmp.r == 6) ? 0 : 1; */
    ]])],
[res_no_stack=yes],
[res_no_stack=no],
[res_no_stack=unknown])
AC_MSG_RESULT($res_no_stack)

if test x$res_with_stack = "xyes" -a x$res_no_stack = "xno"; then
  HAVE_ZDOTC_WITH_RV=1
else 
  if test x$res_with_stack = "xno" -a x$res_no_stack = "xyes"; then
    HAVE_ZDOTC_WITH_RV=0
  else 
    case "$target" in
      *-mingw32*)
	HAVE_ZDOTC_WITH_RV=1
	;;
      *)
        if test x$res_with_stack = "xyes" -a x$res_no_stack = "xyes"; then
          HAVE_ZDOTC_WITH_RV=0
        else 
	  AC_MSG_WARN(Return value behaviour of ZDOTC is unknown. Guessing as first argument.)
          HAVE_ZDOTC_WITH_RV=1
	fi
	;;
    esac
  fi
fi
AC_SUBST(HAVE_ZDOTC_WITH_RV)

LIBS=$save_LIBS

### -------------------------------------------------------------
### system specifics
### -------------------------------------------------------------
AC_MSG_CHECKING([for system type])
LAPACK_OS_WIN32=0
LAPACKPP_INCLUDES="-I${pkgincludedir} -DLA_COMPLEX_SUPPORT"
LAPACKPP_INTERNAL_LIBFILE="\${top_builddir}/src/liblapackpp.la"
LAPACKPP_LIBS="-llapackpp"
LAPACKPP_LDFLAGS="-L${libdir}"
case "$target" in
    *-mingw32*)
      LAPACKPP_LDFLAGS="-L${WIN_PATH_WINDOWS_MINGW} -L${bindir}"
      ARCH="WIN32"
      LAPACK_OS_WIN32=1
      AC_DEFINE(OS_WIN32, 1, [Define to 1 if using windows])
      AC_DEFINE_UNQUOTED(BUILDING_LAPACK_DLL, 1, [Define to 1 if DLL is built])
      ;;
    *-linux*)
      ARCH="Linux"
      ;;
    *)
      ARCH="OS_${target_os}"
      # Workaround for bogus system names
      ARCH=` echo ${ARCH} | ${SED} 's/[^a-z0-9_]//g' `
      ;;
esac
AC_MSG_RESULT($ARCH)

AC_SUBST(LAPACK_OS_WIN32)
AM_CONDITIONAL(IS_WINDOWS, [test "$ARCH" = "WIN32"])

# These are the INCLUDES and LDFLAGS as an application should use them.
AC_SUBST(LAPACKPP_INCLUDES)
AC_SUBST(LAPACKPP_LIBS)
AC_SUBST(LAPACKPP_LDFLAGS)
AC_SUBST(LAPACKPP_INTERNAL_LIBFILE)

# These are the variables for the internal tests.
# For whatever reason, the variable top_srcdir needs to be quoted here.
more_defs="-DLA_COMPLEX_SUPPORT -D${ARCH}"
all_libraries="-L\${top_builddir}/src ${LAPACKPP_LIBS}"
all_includes="-I\${top_srcdir}/include ${more_defs}"

AC_SUBST(all_includes)
AC_SUBST(all_libraries)

### -------------------------------------------------------------
### Configuration flags
### -------------------------------------------------------------

dnl Debugging code with empty default
ACX_COMPILE_WARN([ ])

dnl Whether the MSVC-compiled DLL should be included 
dnl in the self-installing exe
AC_ARG_WITH([msvc-dll],
	AS_HELP_STRING([--with-msvc-dll],
		[if the self-installing exe should include the msvc DLL]) )
if test "x$with_msvc_dll" = "xyes"; then
  ISCC_WITH_DLL=""
  ISCC_WITHOUT_DLL=";"
else
  ISCC_WITH_DLL=";"
  ISCC_WITHOUT_DLL=""
fi
AC_SUBST(ISCC_WITH_DLL)
AC_SUBST(ISCC_WITHOUT_DLL)


### -------------------------------------------------------------

dnl All Makefiles that are to be generated.
AC_CONFIG_FILES([
 Makefile
 lapackpp.iss
 mainpage.doxygen
 lapackpp.spec
 lapackpp.pc
 src/ressource.rc
 blaspp/Makefile
 blaspp/src/Makefile
 blaspp/testing/Makefile
 contrib/Makefile
 include/Makefile
 include/laversion.h
 macros/Makefile
 matrix/Makefile
 matrix/src/Makefile
 matrix/testing/Makefile
 src/Makefile
 testing/Makefile
])
AC_OUTPUT

