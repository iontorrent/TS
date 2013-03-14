#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#
# Copies trunk source code to a production branch.
# NOTE: This is a copy and not a merge.  Existing branch code is wiped out.
#
# Exeecute this script from the toplevel directory of the repository.
#
set -x
TESTING=0       # Set to 1 to enable test mode -> no commits made

URL='https://iontorrent.jira.com/svn'
REPOSITORY="TS"
SOURCE="trunk"
BRANCH="weekly_release"

URL_TRUNK=${URL}/${REPOSITORY}/${SOURCE}
URL_BRANCH=${URL}/${REPOSITORY}/branches/${BRANCH}

#Make sure local version files are not locally modified; restore them from trunk
find ./ -maxdepth 2 -name version -exec rm {} \;
svn update

# Delete BRANCH (its okay if it doesn't exist)
del_msg="Removing branch, prior to recreating it"
CMD="svn delete ${URL_BRANCH}"
if [ $TESTING -eq 0 ]; then
    svn delete -m"${del_msg}" ${URL_BRANCH}
else
    echo "Testing Only: ${CMD}"
fi

# Copy trunk to the new branch
REVISION=$(svn info|grep ^Revision|awk '{print $2}')
add_msg="Copying ${REPOSITORY} revision ${REVISION} to branch ${BRANCH}"
if [ $TESTING -eq 0 ]; then
    svn copy -m"$add_msg" ${URL_TRUNK} ${URL_BRANCH}
else
    echo "Testing Only: svn copy -m\"$add_msg\" ${URL_TRUNK} ${URL_BRANCH}"
fi


# Increment the RELEASE number in the trunk
for file in $(find ./ -maxdepth 2 -name version); do

    echo $file                                                          # DEBUG
    source $file
    
    current_build_number=$(grep ^RELEASE ${file}|awk -F"=" '{print $2}')
    next_build_number=$(($current_build_number+1))
    
    echo "From $current_build_number to $next_build_number"             # DEBUG
    
    if [ $TESTING -eq 0 ]; then
        sed -i "s:^RELEASE=.*:RELEASE=$next_build_number:" ${file}
        svn commit -m"Update version after branch to $MAJOR.$MINOR.$RELEASE" ${file}
    else
        echo "Testing Only: RELEASE=$next_build_number"
    fi
    
done

# Append to the RELEASE number in the branch
cd ../
if ! svn checkout ${URL_BRANCH} ${BRANCH} ; then
	echo
    echo "ERROR in svn checkout"
    echo "Script is aborting.  Version numbers have NOT been updated in the BRANCH"
    echo
    exit 1
fi

for file in $(find ./${BRANCH} -maxdepth 3 -name version); do
    echo $file                                                          # DEBUG
    source $file
    current_build_number=$(grep ^RELEASE ${file}|awk -F"=" '{print $2}')
    next_build_number=${current_build_number}"+0"
    
    echo "From $current_build_number to $next_build_number"             # DEBUG
    
    if [ $TESTING -eq 0 ]; then
        sed -i "s:^RELEASE=.*:RELEASE=$next_build_number:" ${file}
        svn commit -m"Update version after branch to $MAJOR.$MINOR.$RELEASE" ${file}
    else
        echo "Testing Only: RELEASE=$next_build_number"
    fi
done
cd -

exit
