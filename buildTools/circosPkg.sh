#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Build a distribution package for circos for use on Torrent Server
set -e

# create temporary directory
tempDir='ion-circos'
rm -rf $tempDir
mkdir -p $tempDir/opt/circos

# copy source dir to temporary directory (assumes single dir exists)
#cp -rp $(find ../external -type d -name circos-\*) $tempDir/opt/circos
cp -rp $(find ../external -type d -name circos-0.49) $tempDir/opt/circos

# remove svn control files
find $tempDir -type d -name .svn -exec rm -rf {} \; 2>/dev/null || true

# create directory DEBIAN
circos=$(basename $(find $tempDir -type d -name circos-\*))
mkdir $tempDir/DEBIAN
version=${circos#*-}
build="1ion4"

# populate directory DEBIAN with required files
#---										---#
#---    changelog                           ---#
#---										---#
cat >> $tempDir/DEBIAN/changelog << EOF
circos (${version}-${build}) lucid; urgency=low

  * New upstream release.

 -- bpuc <bpuc@magnolia.ite>  Wed, 11 May 2011 10:37:00 -0400
EOF


#---										---#
#---    control                             ---#
#---										---#
cat >> $tempDir/DEBIAN/control << EOF
Source: circos
Section: science
Priority: optional
Maintainer: Bernard Puc <bernard.puc@lifetech.com>
Homepage: http://circos.ca
Package: ion-circos
Version: $version
Architecture: amd64
Depends: perl, perl-doc, libgd2-xpm-dev
Description: circular visualizations
 Circos is a software package for visualizing data and information.
EOF


#---										---#
#---	copyright							---#
#---										---#
cat >> $tempDir/DEBIAN/copyright << EOF
Licensed under GPL
EOF


#---										---#
#---	postinst							---#
#---										---#
cat >> $tempDir/DEBIAN/postinst << EOF
#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
set -e

case "\$1" in
    configure)    
    #---												---#
    #---	Install circos and perl dependencies		---#
    #---												---#
	circos=\$(find /opt/circos -type f -name circos)
    sed -i 's:^#!/bin/env:#!/usr/bin/env:' \${circos}
    ln -sf \${circos} /usr/local/bin
    echo "Installing circos and dependencies..."
    rm -f /tmp/circos_install_log

	#---	Installation instructions for perl modules						---#
    #---	http://circos.ca/tutorials/lessons/configuration/perl_and_modules/	
    #---																	---#
    export PERL_AUTOINSTALL=--defaultdeps
    perl -MCPAN -e 'install Module::Build,Params::Validate' < /dev/null
    for pkg in GD Config::General GD::Polyline List::MoreUtils Math::Bezier Math::Round Math::VecStat Params::Validate Readonly Regexp::Common Set::IntSpan; do
    	perl -MCPAN -e "install \$pkg" 
    done
    
#    #---	Outputs status of all installed modules			---#
#    cd \$(dirname \$circos) && ./test.modules ; cd - > /dev/null

    ;;
esac

exit 0
EOF
chmod 0775 $tempDir/DEBIAN/postinst


#---										---#
#---	prerm								---#
#---										---#
cat >> $tempDir/DEBIAN/prerm << EOF
#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
set -e

case "\$1" in
	update|remove)
	rm -f /usr/local/bin/circos
	;;
esac
exit 0
EOF
chmod 0775 $tempDir/DEBIAN/prerm

# run build command
fakeroot dpkg-deb --build ./$tempDir

# enjoy the fruits of this script's labors
mv -v ./$tempDir.deb ./ion-circos_${version}-${build}_amd64.deb
exit 0
