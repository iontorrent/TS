#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Automates the building of a Release Candidate deb file for a given directory
TEST=0  # Set to 1 to go into debug and not make changes to svn or apt

USER=aptadmin
SERVER=updates.ite
PUBDIR=${PUBDIR-"lucid-alpha"}
PUBPATH=/home/aptadmin/public_html/updates_server_root/updates/software/$PUBDIR

for arg in "$@"; do

    pkg_name=$arg

    # Get latest updates from repository
    echo "Calling svn update on ${pkg_name}..."
    svn update ${pkg_name}

    # Generate Change Log file
    cd ${pkg_name}
    svn2cl --reparagraph -i -r HEAD:"{`date -d '30 days ago' '+%F %T'`}"
    cd -

    # Generate email message header
    source ${pkg_name}/version
    current_build_number=$RELEASE
    current_version_number=$MAJOR.$MINOR.$RELEASE
    current_build_number=$(grep ^RELEASE ${pkg_name}/version|awk -F"=" '{print $2}')
    echo "Version:" > ${pkg_name}/mailmessagebody
    echo "$MAJOR.$MINOR.$RELEASE" >> ${pkg_name}/mailmessagebody
    echo >> ${pkg_name}/mailmessagebody
    echo "Change Log:" >> ${pkg_name}/mailmessagebody

    if grep -qn "$current_version_number" ${pkg_name}/ChangeLog; then
        c_v_n=$(echo $current_version_number|sed 's/\./\\./g')
        head --lines=$(grep -n "$c_v_n" ${pkg_name}/ChangeLog|awk -F: '{print $1}') ${pkg_name}/ChangeLog >> ${pkg_name}/mailmessagebody
        echo "email notification message body in: ${pkg_name}/mailmessagebody"
    else
        echo 'Failed to generate Change Log and email notification message'
    fi

    # Do the build
    rm -rf build/${pkg_name}    # Remove existing build directory
    # If its Analysis, force a rebuild of external libraries
    if [ "${pkg_name:0:8}" == "Analysis" ]; then
        rm -rf external/
        svn update external >/dev/null
        # We need to manually kick off cuda building
        MODULES="gpu Analysis" ./buildTools/build.sh -DBUILDTAG=TSRC
    else
        MODULES=$pkg_name ./buildTools/build.sh -DBUILDTAG=TSRC
    fi

    if [ $? -ne 0 ]; then
        echo "Error during build.sh."
        echo "BUILD FAILURE FOR $pkg_name"
    else

        # Upload the deb file
        if [ $TEST -eq 1 ]; then
            echo "TESTING: Published ./build/${pkg_name}/ion-*.deb to $PUBDIR"
        else
        	# If its Analysis, upload gpu package as well
    		if [ "${pkg_name:0:8}" == "Analysis" ]; then
            	./buildTools/publish --no-pkgfile --$PUBDIR./build/gpu/ion-*.deb
            fi
            ./buildTools/publish --no-pkgfile --$PUBDIR ./build/${pkg_name}/ion-*.deb
        fi

        # Update the version number file
        source ${pkg_name}/version
        current_build_number=$RELEASE
        current_version_number=$MAJOR.$MINOR.$RELEASE
        current_build_number=$(grep ^RELEASE ${pkg_name}/version|awk -F"=" '{print $2}')

        echo -n "Incrementing version number "

        if echo $current_build_number|grep -q "+" ; then
            # This is a release candidate branch build.  RELEASE is "21+3" format
            build_num=${current_build_number#*+}
            release_num=${current_build_number%+*}

            next_build_num=$(($build_num+1))
            next_build_number=${release_num}"+"$next_build_num
        else
            next_build_number=$(($current_build_number+1))
        fi

        echo "from $current_build_number to $next_build_number."

        sed -i "s:^RELEASE=.*:RELEASE=$next_build_number:" ${pkg_name}/version
        # Commit version file to svn
        source ${pkg_name}/version
        if [ $TEST -eq 1 ]; then
            echo "TESTING: Updated package version $MAJOR.$MINOR.$RELEASE"
            # revert version file to previous
            sed -i "s:^RELEASE=.*:RELEASE=$current_build_number:" ${pkg_name}/version
        else
            svn commit -m "Updated package version to $MAJOR.$MINOR.$RELEASE" ${pkg_name}/version
        fi
    fi

done

# Update the Packages.gz file
echo "Writing new Packages.gz file"
if [ $TEST -eq 1 ]; then
    echo "TESTING: file would have been uploaded and Packages.gz would have been updated"
else
    ssh $USER@$SERVER "cd $PUBPATH/.. && rm -f $PUBDIR/Packages.gz && apt-ftparchive packages $PUBDIR | gzip > $PUBDIR/Packages.gz"
    if [ $? -ne 0 ]
    then
        echo "There was an error creating the Packages.gz file"
        exit 1
    fi
fi
exit 0
