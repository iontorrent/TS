Name:           armadillo
Version:        1.0.0
Release:        1%{?dist}
Summary:        Fast C++ matrix library with interfaces to LAPACK and ATLAS

Group:          Development/Libraries
License:        LGPLv3+
URL:            http://arma.sourceforge.net/
Source:         http://download.sourceforge.net/arma/%{name}-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
BuildRequires:  cmake, boost-devel, blas-devel, lapack-devel, atlas-devel

%description
Armadillo is a C++ linear algebra library (matrix maths)
aiming towards a good balance between speed and ease of use.
Integer, floating point and complex numbers are supported,
as well as a subset of trigonometric and statistics functions.
Various matrix decompositions are provided through optional
integration with LAPACK and ATLAS libraries.
A delayed evaluation approach is employed (during compile time)
to combine several operations into one and reduce (or eliminate) 
the need for temporaries. This is accomplished through recursive
templates and template meta-programming.
This library is useful if C++ has been decided as the language
of choice (due to speed and/or integration capabilities), rather
than another language like Matlab or Octave.


%package devel
Summary:        Development headers and documentation for the Armadillo C++ library
Group:          Development/Libraries
Requires:       %{name} = %{version}-%{release}
Requires:       boost-devel, blas-devel, lapack-devel, atlas-devel, libstdc++-devel

# The header files of Armadillo include some Boost and ATLAS header files,
# delivered within the boost-devel and atlas-devel sub-packages, respectively.
# However, since there is no explicit dependency on Boost or ATLAS libraries
# (most of Boost is delivered as header files only), the RPM building process 
# does not detect these dependencies.  These dependencies must therefore be 
# added manually.

%description devel
This package contains files necessary for development using the
Armadillo C++ library. It contains header files, example programs,
and user documentation (reference guide).


%prep
%setup -q

# convert DOS end-of-line to UNIX end-of-line

for file in README.txt; do
  sed 's/\r//' $file >$file.new && \
  touch -r $file $file.new && \
  mv $file.new $file
done

%build
%{cmake}
%{__make} VERBOSE=1 %{?_smp_mflags}


%install
rm -rf $RPM_BUILD_ROOT
%{__make} install DESTDIR=$RPM_BUILD_ROOT
rm -rf   $RPM_BUILD_ROOT/%{_docdir}/%{name}-%{version}/
mkdir -p $RPM_BUILD_ROOT/%{_docdir}/%{name}-%{version}/
rm -f examples/Makefile.cmake
rm -rf examples/example1_win32
rm -rf examples/example2_win32
rm -rf examples/lib_win32
cp -r LICENSE.txt licenses README.txt index.html examples docs $RPM_BUILD_ROOT/%{_docdir}/%{name}-%{version}/

%clean
rm -rf $RPM_BUILD_ROOT


%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig


%files
%defattr(-,root,root,-)
%{_libdir}/*.so.*
%dir %{_docdir}/%{name}-%{version}/
%doc %{_docdir}/%{name}-%{version}/LICENSE.txt
%doc %{_docdir}/%{name}-%{version}/licenses/

%files devel
%defattr(-,root,root,-)
%{_libdir}/*.so
%{_includedir}/armadillo
%{_includedir}/armadillo_bits/
%{_includedir}/armadillo_itpp
%doc %{_docdir}/%{name}-%{version}/README.txt
%doc %{_docdir}/%{name}-%{version}/index.html
%doc %{_docdir}/%{name}-%{version}/examples/
%doc %{_docdir}/%{name}-%{version}/docs/

%changelog
* Thu Oct 14 2010 Conrad Sanderson - 0.9.90-1
- spec updated for Armadillo 0.9.90

* Tue Sep 21 2010 Conrad Sanderson - 0.9.80-1
- spec updated for Armadillo 0.9.80

* Wed Sep 01 2010 Conrad Sanderson - 0.9.70-1
- spec updated for Armadillo 0.9.70

* Wed Aug 04 2010 Conrad Sanderson - 0.9.60-1
- spec updated for Armadillo 0.9.60

* Wed Jul 14 2010 Conrad Sanderson - 0.9.52-1
- spec updated for Armadillo 0.9.52

* Wed Jul 07 2010 Conrad Sanderson - 0.9.50-1
- spec updated for Armadillo 0.9.50

* Wed Jun 02 2010 Conrad Sanderson - 0.9.10-1
- spec updated for Armadillo 0.9.10

* Fri May 14 2010 Conrad Sanderson - 0.9.8-1
- spec updated for Armadillo 0.9.8

* Tue Apr 28 2010 Conrad Sanderson - 0.9.6-1
- spec updated for Armadillo 0.9.6

* Tue Mar 16 2010 Conrad Sanderson - 0.9.4-1
- spec updated for Armadillo 0.9.4

* Tue Mar 02 2010 Conrad Sanderson - 0.9.2-2
- added explicit dependencies to the devel package

* Tue Mar 02 2010 Conrad Sanderson - 0.9.2-1
- spec updated for Armadillo 0.9.2

* Fri Feb 05 2010 Conrad Sanderson - 0.9.0-1
- spec updated for Armadillo 0.9.0

* Mon Jan 27 2010 Conrad Sanderson - 0.8.2-1
- spec updated for Armadillo 0.8.2

* Mon Dec 14 2009 Conrad Sanderson - 0.8.0-1
- spec updated for Armadillo 0.8.0

* Fri Oct 23 2009 Conrad Sanderson - 0.7.2-1
- spec updated for Armadillo 0.7.2

* Mon Oct 05 2009 Conrad Sanderson - 0.7.0-1
- spec updated for Armadillo 0.7.0

* Fri Jul 24 2009 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 0.6.12-3
- Rebuilt for https://fedoraproject.org/wiki/Fedora_12_Mass_Rebuild

* Wed Jul 06 2009  Conrad Sanderson - 0.6.12-2
- added conversion of DOS end-of-line to UNIX end-of-line for README.txt

* Wed Jun 22 2009  Conrad Sanderson - 0.6.12-1
- spec updated for Armadillo 0.6.12

* Wed Jun 15 2009  Conrad Sanderson - 0.6.11-8
- cleanup of dependencies
- explanation as to why boost-devel and atlas-devel are required by armadillo-devel

* Wed Jun 09 2009  Conrad Sanderson - 0.6.11-7
- explicit declaration of doc directory in the main package
- explicitly marked doc files in both packages

* Wed Jun 09 2009  Conrad Sanderson - 0.6.11-6
- removed symlinks
- placed all documentation and license files into one directory that is shared by both packages

* Wed Jun 09 2009  Conrad Sanderson - 0.6.11-5
- added symlinks to LICENSE.txt and licenses in the devel package

* Wed Jun 08 2009  Conrad Sanderson - 0.6.11-4
- added LICENSE.txt to the main package

* Wed May 22 2009  Conrad Sanderson - 0.6.11-3
- using cmake macro instead of directly calling cmake

* Wed May 21 2009  Conrad Sanderson - 0.6.11-2
- moved all text files to devel package to retain consistency with the layout in the original .tar.gz

* Wed May 08 2009  Conrad Sanderson - 0.6.10-2
- Removed several explicit build dependencies that are provided by default in Fedora
- Simplified handling of doc files

* Wed May 02 2009  Conrad Sanderson - 0.6.10-1
- Updated spec file for Armadillo 0.6.10

* Wed Apr 02 2009  Conrad Sanderson
- Updated list of files in 0.6.7 release

* Wed Apr 02 2009  Conrad Sanderson
- Updated description

* Wed Mar 24 2009  Conrad Sanderson
- Added explicit dependence on libstdc++-devel

* Wed Mar 17 2009  Conrad Sanderson
- Simplified specification of directories
- Removed library packages specified by "Requires", as library dependencies are detected automatically

* Wed Mar 12 2009  Conrad Sanderson
- Modified to generate separate devel package (subsumes previous doc package)
- Removed redundant packages specified by "BuildRequires"
- Added CMake installation prefixes to allow for x86_64

* Wed Feb  4 2009  Conrad Sanderson
- Modified to generate separate doc package

* Thu Jan 28 2009  Conrad Sanderson
- Added argument to cmake: -DCMAKE_INSTALL_PREFIX=/usr 

* Thu Jan 22 2009  Conrad Sanderson
- Initial spec file prepared

