=== Contents ===

1. Introduction
   1.1. Citation Details

2. Installation
   2.0. Preliminaries
   2.1. Manual Installation
   2.2. Installation on Unix-like systems
   2.3. Installation on MS Windows

3. Compiling Programs and Linking
   3.0. Examples
   3.1. Compiling & Linking on Unix-like systems
   3.2. Compiling & Linking on MS Windows

4. Caveats
   4.0. Support for ATLAS
   4.1. Support for ACML and Intel MKL

5. Documentation / Reference Manual

6. Using Armadillo with IT++

7. FAQs and Bug Reports

8. Credits

9. License




=== 1.0. Introduction ===

Armadillo is a C++ linear algebra library (matrix maths)
aiming towards a good balance between speed and ease of use.
Integer, floating point and complex numbers are supported,
as well as a subset of trigonometric and statistics functions.
Various matrix decompositions are provided through optional
integration with LAPACK or LAPACK-compatible libraries.

A delayed evaluation approach is employed (during compile time)
to combine several operations into one and reduce (or eliminate)
the need for temporaries. This is accomplished through recursive
templates and template meta-programming.

This library is useful if C++ has been decided as the language
of choice (due to speed and/or integration capabilities),
rather than another language like Matlab or Octave.
It is distributed under a license that is useful in both
open-source and proprietary contexts.

Armadillo is primarily developed at NICTA (Australia),
with contributions from around the world.
More information about NICTA can be obtained from:
  http://nicta.com.au



=== 1.1. Citation Details ===

If you use Armadillo in your research and/or software,
we would appreciate a citation to the following tech report:

  Conrad Sanderson.
  Armadillo: An Open Source C++ Linear Algebra Library for
  Fast Prototyping and Computationally Intensive Experiments.
  NICTA Technical Report, 2010.



=== 2.0. Installation: Preliminaries ===

Armadillo makes extensive use of template meta-programming,
recursive templates and template based function overloading.
As such, C++ compilers which do not fully implement the C++
standard may not work correctly.

The functionality of Armadillo is partly dependent on other
libraries -- mainly LAPACK and BLAS. Armadillo can work without
LAPACK or BLAS, but its functionality will be reduced.
In particular, basic functionality will be available
(e.g. matrix addition and multiplication), but things like
eigen decomposition or will not be. Matrix multiplication
(mainly for big matrices) may not be as fast.

For manual installation on all systems, see section 2.1.

For installation on Unix-like systems, see section 2.2.
Unix-like systems include:
  - Linux
  - Mac OS X
  - FreeBSD
  - Solaris
  - CygWin

For installation on MS Windows, see section 2.3.



=== 2.1. Manual Installation ===

The manual installation is comprised of 3 steps:

* Step 1:
  Copy the entire "include" folder to a convenient location
  and tell your compiler to use that location for header files
  (in addition to the locations it uses already).
  Alternatively, you can use the "include" folder directly.

* Step 2:
  Modify "include/armadillo_bits/config.hpp" to indicate 
  which libraries are currently available on your system.
  For example, if you have LAPACK and BLAS present, 
  uncomment the following lines:
  
  #define ARMA_USE_LAPACK
  #define ARMA_USE_BLAS

* Step 3:
  If you have LAPACK and/or BLAS present, configure your 
  compiler to link with these libraries. 
  
  You can also link with the the equivalent of LAPACK and BLAS,
  e.g. Intel's MKL or AMD's ACML. Under Mac OS X, link using 
  -framework Accelerate



=== 2.2. Installation on Unix-like systems ===

If you have installed Armadillo using an RPM or DEB package,
you don't need to do anything else. Otherwise read on.

You can use the manual installation process as described in
section 2.1, or the following CMake based automatic installation.

* Step 1:
  If CMake is not already be present on your system, download
  it from http://www.cmake.org

  On major Linux systems (such as Fedora, Ubuntu, Debian, etc),
  cmake is available as a pre-built package, though it may need
  to be explicitly installed (using a tool such as PackageKit,
  yum, rpm, apt, aptitude, etc).

* Step 2:
  If you have BLAS and/or LAPACK, install them before installing
  Armadillo. Under Mac OS X this is not necessary.
  
  On Linux systems it is recommended that the following libraries
  are present: LAPACK, BLAS, ATLAS and Boost. LAPACK and BLAS are
  the most important. If you have ATLAS and Boost, it's also necessary
  to have the corresponding header files installed.

* Step 3:
  Open a shell (command line), change into the directory that was
  created by unpacking the armadillo archive, and type the following
  commands:

  cmake .
  make 

  The full stop separated from "cmake" by a space is important.
  CMake will figure out what other libraries are currently installed
  and will modify Armadillo's configuration correspondingly.
  CMake will also generate a run-time armadillo library, which is a 
  combined alias for all the relevant libraries present on your system
  (e.g. BLAS, LAPACK and ATLAS).
  
  If you need to re-run cmake, it's a good idea to first delete the
  "CMakeCache.txt" file (not "CMakeLists.txt").

* Step 4:
  If you have access to root/administrator/superuser privileges,
  first enable the privileges (e.g. through "su" or "sudo")
  and then type the following command:
  
  make install

  If you don't have root/administrator/superuser privileges, 
  type the following command:

  make install DESTDIR=my_usr_dir

  where "my_usr_dir" is for storing C++ headers and library files.
  Make sure your C++ compiler is configured to use the sub-directories
  present within this directory.



=== 2.3. Installation on MS Windows ===

There is currently no automatic installation for Windows.
Please use the manual installation process described in
section 2.1.

Pre-compiled BLAS and LAPACK libraries for Windows are
provided in the "examples/libs_win32" folder.

Please contact the authors if you'd like to contribute
a CMake based installation solution for Windows.



=== 3.0. Compiling Programs and Linking: Examples ===

The "examples" directory contains several quick example programs
that use the Armadillo library. If Armadillo was installed manually
(i.e. according to section 2.1), you will also need to explicitly
link your programs with the libraries that were specified in
"include/armadillo_bits/config.hpp".

"example1.cpp" doesn't need any external libraries.
"example2.cpp" requires the LAPACK library or its equivalent
(e.g. the Accelerate framework on Mac OS X). You may get errors
at compile or run time if LAPACK functions are not available.

NOTE: As Armadillo is a template library, we recommended that
      optimisation is enabled during compilation. For example,
      for the GCC compiler use -O1 or -O2


=== 3.1. Compiling & Linking on Unix-like systems ===

Please see "examples/Makefile", which may may need to be configured
for your system. If Armadillo header files were installed in a
non-standard location, you will need to modify "examples/Makefile"
to tell the compiler where they are.

If Armadillo was installed manually and you specified that
LAPACK and BLAS are available, instead of using "-larmadillo",
use the following:
  - under Linux, use "-llapack -lblas"
  - under Mac OS X, use "-framework Accelerate"
  - under the Sun Studio compiler, try "-library=sunperf"

NOTE: on Ubuntu and Debian based systems you may need to add 
      "-lgfortran" to the compiler flags.



=== 3.2. Compiling & Linking on MS Windows ===

As a courtesy, we've provided pre-compiled 32 bit versions of
LAPACK and BLAS for Windows, as well as MSVC project files to
compile example1.cpp and example2.cpp. The project files are
stored in the following folders:
  examples/example1_win32
  examples/example2_win32

The LAPACK and BLAS libraries are stored in:
  examples/lib_win32

If you're not using MSVC, you will need to manually modify 
"include/armadillo_bits/config.hpp" to enable the use of
LAPACK and BLAS. Please see section 2.1 for more information.

The MSCV project files were tested on Windows XP (32 bit) with
Visual C++ 2008 (Express Edition). You may need to make adaptations
for 64 bit systems, later versions of Windows and/or the compiler.
For example, you may have to enable or disable the ARMA_BLAS_LONG
and ARMA_BLAS_UNDERSCORE macros in "armadillo_bits/config.hpp".

To preserve our sanity, we (Armadillo developers) don't use Windows
on a regular basis, and as such can't help you with the adaptations.

The compiled versions of LAPACK and BLAS were downloaded from:
  http://www.fi.muni.cz/~xsvobod2/misc/lapack/

You can find other versions of LAPACK and BLAS at these sites:
  http://www.stanford.edu/~vkl/code/libs.html
  http://icl.cs.utk.edu/lapack-for-windows/lapack/
  http://software.intel.com/en-us/intel-mkl/
  http://www.amd.com/acml

If you want to compile BLAS and LAPACK yourself, you can find
the original sources at:
  http://www.netlib.org/blas/
  http://www.netlib.org/lapack/

If you encounter issues with the MS Visual C++ compiler,
the following high-quality compilers are useful alternatives:

  - Intel's C++ compiler
    http://software.intel.com/en-us/intel-compilers/

  - GCC (part MinGW)
    http://www.mingw.org/

  - GCC (part of CygWin)
    http://www.cygwin.com/

If using Intel's C++ compiler, you'll need version 10.0 or better.
If using GCC, you'll need version 4.0 or better.



=== 4.0. Caveats: Support for ATLAS ===

Armadillo can use the ATLAS library for faster versions of
certain LAPACK and BLAS functions. Not all ATLAS functions are
currently used, and as such LAPACK should still be installed.

The minimum recommended version of ATLAS is 3.8.
Old versions (e.g. 3.6) can produce incorrect results
as well as corrupting memory, leading to random crashes.

Users of Ubuntu and Debian based systems should explicitly
check that version 3.6 is not installed. It's better to
remove the old version and use the standard LAPACK library.



=== 4.1. Caveats: Support for ACML and Intel MKL ===

Armadillo can work with AMD Core Math Library and Intel's
Math Kernel Library (MKL), however there are several caveats.

On Linux systems, ACML and MKL are typically installed in a
non-standard location, which can cause problems during linking.

Before installing Armadillo, the system should know where the ACML or MKL
libraries are located (e.g., "/opt/intel/mkl/10.2.2.025/lib/em64t/").
This can be achieved by setting the LD_LIBRARY_PATH environment variable,
or, for a more permanent solution, adding the location of the libraries
to "/etc/ld.so.conf". It may also be possible to store a text file 
with the location in the "/etc/ld.so.conf.d" directory.
In the latter two cases you will need to run "ldconfig" afterwards.

The default installations of ACML 4.4.0 and MKL 10.2.2.025 are known 
to have issues with SELinux, which is turned on by default in Fedora
(and possibly RHEL). The problem may manifest itself during run-time,
where the run-time linker reports permission problems.
It is possible to work around the problem by applying an appropriate
SELinux type to all ACML and MKL libraries.

If you have ACML or MKL installed and they are persistently giving
you problems during linking, you can disable the support for them
by editing the "CMakeLists.txt" file, deleting "CMakeCache.txt" and
re-running the CMake based installation. Specifically, comment out
the lines containing:
  INCLUDE(ARMA_FindMKL)
  INCLUDE(ARMA_FindACMLMP)
  INCLUDE(ARMA_FindACML)



=== 5. Documentation / Reference Manual ===

A reference manual (user documentation) is available at
http://arma.sourceforge.net or in the "docs" directory.
Use a web browser to open the "docs/index.html" file.

The user documentation explains how to use Armadillo's
classes and functions, with snippets of example code.




=== 6. Using Armadillo with IT++ ===

If you wish to use the IT++ library in conjunction with Armadillo,
use #include "armadillo_itpp" instead of #include "armadillo"
in your code. See also the "examples/example_itpp.cpp" file.



=== 7. FAQs and Bug Reports ===

Answers to Frequently Asked Questions (FAQs) can be found at:
  http://arma.sourceforge.net/faq.html

This library has gone through extensive testing and
has been successfully used in production environments.
However, as with almost all software, it's impossible
to guarantee 100% correct functionality.

If you find a bug in the library (or the documentation),
we are interested in hearing about it. Please make a small
self-contained program which exposes the bug and send the
program source (as well as the bug description) to the 
developers. The developers' contact details are available at:
  http://arma.sourceforge.net/contact.html

Alternatively, you can post the source and bug description
on Armadillo's discussion board:
  http://sourceforge.net/apps/phpbb/arma/



=== 8. Credits ===

Main sponsoring organisation:
- NICTA
  http://nicta.com.au

Main developers:
- Conrad Sanderson - http://itee.uq.edu.au/~conrad/
- Ian Cullinan
- Dimitrios Bouzas

Contributors:
- Eric R. Anderson
- Benoît Bayol
- Salim Bcoin
- Justin Bedo
- Darius Braziunas
- Ted Campbell
- Ryan Curtin
- Chris Davey
- Dirk Eddelbuettel
- Romain Francois
- Charles Gretton
- Edmund Highcock
- Kshitij Kulshreshtha
- Oka Kurniawan
- David Lawrence
- Carlos Mendes
- Artem Novikov
- Martin Orlob
- Ken Panici
- Adam Piątyszek
- Jayden Platell
- Vikas Reddy
- Ola Rinta-Koski
- Gerhard Schreiber
- Shane Stainsby
- Petter Strandmark
- Paul Torfs
- Simon Urbanek
- Yong Kang Wong


=== 9. License ===

Please see the "LICENSE.txt" file.


