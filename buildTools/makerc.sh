#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Automates the building of a Release Candidate deb file for a given directory
TEST=0  # Set to 1 to go into debug and not make changes to svn or apt
pkg_name=$1

# Update the version number file
source ${pkg_name}/version
current_build_number=$RELEASE
current_version_number=$MAJOR.$MINOR.$RELEASE
current_build_number=$(grep ^RELEASE ${pkg_name}/version|awk -F"=" '{print $2}')
echo $current_build_number
next_build_number=$(($current_build_number+1))
echo $next_build_number
sed -i "s:^RELEASE=.*:RELEASE=$next_build_number:" ${pkg_name}/version
# Commit version file to svn
source ${pkg_name}/version
if [ $TEST -eq 1 ]; then
    echo "Candidate release build $MAJOR.$MINOR.$RELEASE"
else
    svn commit -m"Candidate release build $MAJOR.$MINOR.$RELEASE" ${pkg_name}/version
fi

# Do the build
rm -rf build/${pkg_name}
MODULES=$pkg_name ./buildTools/build.sh -DBUILDTAG=TSRC

if [ $? -ne 0 ]; then
	echo "Error during build.sh.  Exiting"
    exit
fi

# Upload the deb file
if [ $TEST -eq 1 ]; then
    echo "Built and published ./build/${pkg_name}/ion-*.deb"
else
    ./buildTools/publish ./build/${pkg_name}/ion-*.deb
fi

# Generate email message
cd ${pkg_name}
echo "Version:" > mailmessagebody
echo "$MAJOR.$MINOR.$next_build_number" >> mailmessagebody
echo >> mailmessagebody
echo "Change Log:" >> mailmessagebody
svn2cl --reparagraph -i -r HEAD:"{`date -d '30 days ago' '+%F %T'`}"
if grep -qn "$current_version_number" ChangeLog; then
	c_v_n=$(echo $current_version_number|sed 's/\./\\./g')
    head --lines=`grep -n "$c_v_n" ChangeLog|awk -F: '{print $1}'` ChangeLog >> mailmessagebody
	echo "email notification message body in: ${pkg_name}/mailmessagebody"
else
    echo 'Failed to generate Change Log and email notification message'
    exit 1
fi


