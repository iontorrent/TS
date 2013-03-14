#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

EC2_PRIVATE_KEY=/root/pk.pem
EC2_CERT=/root/cert.pem


# $MODULES defines the packages to be built
# this is the default list, it will overriddent if Build Plan provides something different
MODULES=${MODULES-"
        gpu
	Analysis
 	alignTools
	tsconfig
        pipeline
	dbReports
	plugin
        publishers
        torrentR
	rndplugins
"}
export MODULES

show_help() {
    echo "Usage: $0"
    echo "  -t tag      Build Tag (eg 'DEV', 'TRUNKCI')"
    echo "  -n num      Build Number (eg '342')"
    echo
    echo " Options related to successful build:"
    echo "  -i          Install packages"
    echo "  -a ddclient config file name"
}


# read in command line parameters provided by the Build Plan
BUILDTAG="dev"
BUILDNUM="0"
SETDNS=
DOINSTALL=0

while getopts ":t:n:ia:" opt; do
    case $opt in
        t ) BUILDTAG=$OPTARG;;
        n ) BUILDNUM=$OPTARG;;
        i ) DOINSTALL=1;;
        a ) SETDNS=$OPTARG;;
        \? ) show_help; exit;;
    esac
done;
shift $(($OPTIND - 1))

export BUILDNUM

# remove this file if it is present, messes up dbreports install
rm /etc/apache2/sites-enabled/000-default 

echo "=================================================="
echo " clear out rabbitmq-server settings "
sudo service rabbitmq-server stop
sudo rm -rf /var/lib/rabbitmq/*
sudo service rabbitmq-server start


# if the instance is fresh, configure with prerequisites
if [ -f ~/.instance_configured ]; then
   echo "instance is already configured" 
   sudo apt-get update
else
   echo "configuring new instance" 
   touch ~/.instance_configured

   # Update dynamic DNS with the hostname defined by the Build Plan 
   if [ "$SETDNS" ]; then
      echo " Configuring Dynamic DNS for $SETDNS..."
      sudo ddclient -file buildTools/$SETDNS
   fi

   # setup custom sources.list
   sudo cp buildTools/bamboo/sources.list /etc/apt
   sudo apt-get update

   # clean up torrentR
   sudo apt-get --assume-yes --force-yes purge ion-torrentr 
   sudo apt-get --assume-yes --force-yes purge r-cran*
   sudo apt-get --assume-yes --force-yes purge r-base*
   sudo rm -rf /usr/local/lib/R
   sudo apt-get --assume-yes --force-yes install ion-torrentr

   # install the desired version of cmake
   echo "=================================================="
   echo " check if cmake packages are up-to-date..."
   dpkg -l  | grep libxmlrpc-core-c3 
   if [ $? -eq 0 ]; then
      echo "...cmake is up to date"
   else
      echo "...cmake is being updated"
      sudo apt-get install libarchive1
      sudo apt-get install libxmlrpc-core-c3
      sudo scp bamboo@ie.iontorrent.com:~/cmake* .
      sudo dpkg -i cmake-data_2.8.3-3ubuntu7+lucid0_all.deb
      sudo dpkg -i cmake_2.8.3-3ubuntu7+lucid0_amd64.deb
      sudo dpkg -i cmake-curses-gui_2.8.3-3ubuntu7+lucid0_amd64.deb
   fi

fi


# 4/3/12 - quick hack to make sure the latest python-django-south is installed 
echo "=================================================="
echo " force install of python-django-south..."
sudo apt-get --assume-yes --force-yes install python-django-south

# 4/6/12 - quick hack to force install of latest tmap in advance of dbreports
echo "=================================================="
echo "  make sure tmap package is up-to-date or dbreports install will fail..."
sudo apt-get --assume-yes --force-yes install tmap

# 4/10/12 - quick hack to force install of libboost-all-dev, new trunk dependency
echo "=================================================="
echo "  make sure libboost-all-dev is installed..."
sudo apt-get --assume-yes --force-yes install libboost-all-dev

# 5/10/12 - quick hack to force install of python-beautifulsoup, new trunk dependency
echo "=================================================="
echo "  make sure python-beautifulsoup is installed..."
sudo apt-get --assume-yes --force-yes install python-beautifulsoup 

# 5/10/12 - quick hack to force install of python-tastypie 0.9.11-1ubuntu1+lucid1, on trunk
echo "=================================================="
echo "  updated tastypie version on 7/23"
#sudo apt-get --assume-yes --force-yes install python-tastypie=0.9.11-1ubuntu1+lucid1
sudo apt-get --assume-yes --force-yes install python-tastypie=0.9.11+ion1e08b4f683e-1ubuntu1+lucid1	

# 6/7/12 - quick hack to force install of python-django=1.4-1ubuntu1+lucid1, on trunk
echo "=================================================="
dpkg -l python-django | grep 1.4-1ubuntu1+lucid1
if [ $? -eq 0 ]; then
   echo "...python-django=1.4-1ubuntu1+lucid1 is already installed"
else
   echo "...python-django=1.4-1ubuntu1+lucid1 is being installed"
   sudo apt-get --assume-yes --force-yes install python-django=1.4-1ubuntu1+lucid1
fi

# 4/6/12 - quick hack to force install of latest ion-gpu
echo "=================================================="
echo "  make sure latest ion-gpu is installed..."
sudo apt-get --assume-yes --force-yes install ion-gpu

# 6/29/12 - quick hack to force install of python-endless-pagination
echo "=================================================="
echo "  make sure latest python-endless-pagination is installed..."
sudo apt-get --assume-yes --force-yes install python-endless-pagination 

# 8/3/12 - installing liblog4cxx10-dev 
echo "=================================================="
echo "  make sure liblog4cxx10-dev installed..."
sudo apt-get --assume-yes --force-yes install liblog4cxx10-dev

# 9/4/12 - installing python-guppy 
echo "=================================================="
echo "  make sure python-guppy installed..."
sudo apt-get --assume-yes --force-yes install python-guppy

# 9/11/12 - installing putty-tools
echo "=================================================="
echo "  make sure putty-tools installed..."
sudo apt-get --assume-yes --force-yes install putty-tools

# 9/12/12 - installing ion-igv
echo "=================================================="
echo "  make sure latest ion-igv is installed..."
sudo apt-get --assume-yes --force-yes install ion-igv

# 9/15/12 - installing python-jsonpipe 
echo "=================================================="
echo "  make sure latest python-jsonpipe is installed..."
sudo apt-get --assume-yes --force-yes install python-jsonpipe 

# 10/11/12 - installing texlive-latex-base
echo "=================================================="
echo "  make sure latest texlive-latex-base is installed..."
sudo apt-get --assume-yes --force-yes install texlive-latex-base

# 11/8/12 - installing libapache2-mod-wsgi 
echo "=================================================="
echo "  make sure latest libapache2-mod-wsgi is installed..."
sudo apt-get --assume-yes --force-yes install libapache2-mod-wsgi 


echo "=================================================="
echo "  Building Modules: $MODULES"

# Build the packages as defined by $MODULES
buildTools/build.sh -DBUILDTAG=$BUILDTAG -DBUILDNUM=$BUILDNUM

if [ $? != 0 ]; then
    exit 1;
fi

echo "=================================================="
echo "  Setup package location in ./results"
# Make a folder with results (artifacts) that we can easily grab
rm -rf results
mkdir results
cp build-number.txt results/
cp `find build/ -name \*.deb` results
# Clear out previous test results!!
rm test-reports/*.xml


# Install the packages that we just built
ERR=0
if [ "$DOINSTALL" != 0 ]; then

    echo "=================================================="
    echo " Installing Modules"
    for MODULE in $MODULES; do
        echo -n " $MODULE..."
        sudo dpkg -i build/$MODULE/ion-*.deb 1>install.$MODULE.out 2>&1
        if [ "$?" != 0 ]; then
            # Fetch any missing dependencies queued by previous failure
            echo " Initial install attempt failed for $MODULE. Trying to resolve dependencies..."
            sudo apt-get --assume-yes --force-yes -f install 1>>install.$MODULE.out 2>&1
            echo -n " Trying to install $MODULE again..."
            sudo dpkg -i build/$MODULE/ion-*.deb 1>>install.$MODULE.out 2>&1
            if [ "$?" != 0 ]; then
                echo "Failed."
                cat install.$MODULE.out
                ERR=$(($ERR + 1))
            else
                echo "Done. (with deps)"
            fi
        else
            echo "Done."
        fi
    done;
    echo "=================================================="
fi

if [ $ERR != 0 ]; then
    echo "FAILURES: $ERR modules failed to install."
    exit $ERR
fi




# System is configured, start running tests
echo "=================================================="
echo " Starting unit tests..."
# check if python unittest-xml-reporting package is installed
echo "---verify unittest framework is enabled---" 
python buildTools/bamboo-envcheck.py


# These tests run with every check-in, so they need to execute and complete QUICKLY
for MODULE in $MODULES; do
    if [ "$MODULE" = "dbReports" ]; then
	echo "---$MODULE module test---------------------"
	python dbReports/tests/environment_tests.py 
	python dbReports/tests/system_tests.py 
    fi
    if [ "$MODULE" = "pipeline" ]; then
	echo "---$MODULE module test---------------------"
	python pipeline/tests/test_pipeline.py 
    fi
    if [ "$MODULE" = "Analysis" ]; then
	echo "---$MODULE module test---------------------"
        python Analysis/tests/test_analysis.py 
    fi
done



# These tests take a while to run, must run in a dedicated environment
# IONTEST environment variable is configured in the Build Plan
for TEST in $IONTEST; do
    if [ "$TEST" = "smoke" ]; then
	echo "=================================================="
	echo "-- checking smoke test environment configuration"	
	if [ -f ~/.smoketest_configured ]
	then
    	    echo "system is configured" 
	else 
   	    echo "configuring environment for smoke test"
            sudo touch /opt/ion/.ion-internal-server
    	    touch ~/.smoketest_configured
    	    sudo easy_install requests
	    python buildTools/bamboo/addpgm2.py
	     #  local apt repository:    deb file:/buildhome results/ 
	    sudo rm /buildhome
	    sudo ln -s `pwd` /buildhome

	fi
	
	echo "=================================================="	
	echo " Starting smoke tests..."
	sudo cp buildTools/bamboo/sources.list /etc/apt
	# update the local apt repository index
	apt-ftparchive packages results | gzip > results/Packages.gz
	echo "---configuring Torrent Server---"
        sudo apt-get update
        sudo apt-get --assume-yes --force-yes install tmap 
	sudo TSconfig -s
        sudo TSconfig --configure-server --skip-init
	echo "---configure plugin state-----------------" 	
	sudo python buildTools/bamboo/setplugins.py
	echo "---verify unittest framework is enabled---" 	
	python buildTools/bamboo-envcheck.py
	echo "---starting buildTools/bamboo/test_systemtest.py--------------------------------"
	python buildTools/bamboo/test_systemtest.py 
	echo "---starting buildTools/bamboo/test_datatest.py--------------------------------"
	python buildTools/bamboo/test_datatest.py
	echo "---starting buildTools/bamboo/test_verifyresults.py--------------------------------"
	python buildTools/bamboo/test_verifyresults.py

        echo "====waiting two minutes====="
	sleep 120

    fi
done

#wait
