#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
#---
#	Copies analysis results files based on rules.
#	Intention is to archive based on rules and then delete entire original
#---

#--------------------------------------------------#
#---	Initialize variables					---#
#---	Destination location: archive location	---#
#--------------------------------------------------#
destination=
PLUGINS_PLS=0
DETAILED_PLS=0
REALLY_DELETE=0

#------------------------------------------#
#---	Handle command line arguments	---#
#------------------------------------------#
while [ $# -gt 0 ]; do
	case $1 in
    	-d) # Archive Destination Directory
        	shift
            destination=$1
            shift
            ;;
        -e) # Include detailed report
        	DETAILED_PLS=1
            shift
        	;;
        -p) # Include plugin results subdirectory
        	PLUGINS_PLS=1
            shift
        	;;
        -r) # Delete the source files after archiving
        	REALLY_DELETE=1
            shift
            ;;
    	-*) echo >&2 \
        	"usage: $0"
            exit 1
        	;;
        *)	break
        	;;
    esac
done

#--------------------------------------#
#---	Test for required inputs	---#
#--------------------------------------#
if [ -z $destination ]; then
	echo "Need to specify a destination directory:"
    echo "$0 -d <location> $@"
    exit 0
fi

#------------------------------------------------------------------------------#
#---	Basing our rules on the following information:						---#
#	https://iontorrent.jira.com/wiki/display/~bpuc@iontorrent.com/DataManagement+Files
#------------------------------------------------------------------------------#
rawAnalysisFiles=(
	*.wells
    bfmask.bin
    bfmask.stats
    avgNukeTrace_*.txt
    version.txt
    )
defaultReportFiles=(
	debugReport.php
	Default_Report.php
	Bead_density_contour.png
	iontrace_Library.png
	readLenHisto.png
	*.sam.zip
	*.fastq.zip
	*.sff.zip
    )

detailedReportFiles=(
	debugReport.php
	Detailed_Report.php
	log.html
	parsefiles.php
	Bead_density_contour.png
	Filtered_Alignments_Q10.png
	Filtered_Alignments_Q17.png
	iontrace_Library.png
	iontrace_Test Fragment.png
	pre_beadfind_per_region.png
	Region_CF_heat_map_LIB.png
	Region_DR_heat_map_LIB.png
	Region_IE_heat_map_LIB.png
	*.sam
	*.fastq
	*.sff
    )
    
pluginResults=(
	plugin_out
    )
    
#--------------------------------------------------------------#
#---	Default Report and basic image processing results	---#
#--------------------------------------------------------------#
fileList=( ${rawAnalysisFiles[@]}
		   ${defaultReportFiles[@]}
           )

#------------------------------------------#
#---	Optional: Detailed_Report		---#
#------------------------------------------#
if [ $DETAILED_PLS -eq 1 ]; then
	fileList=( ${fileList[@]} ${detailedReportFiles[@]} )
fi

#----------------------------------#
#---	Optional: plugins		---#
#----------------------------------#
if [ $PLUGINS_PLS -eq 1 ]; then
	fileList=( ${fileList[@]} ${pluginResults[@]} )
fi

#----------------------------------------------------------#
#---	process list of runs to archive					---#
#----------------------------------------------------------#
runList=$@
if [ "${runList}" == "" ]; then
	echo "No run directories to backup were specified"
    exit 0
fi
linkedRunList=
for run in ${runList[@]}; do
	
    echo "=========="
    echo "Archiving: $run"
    echo "Destination: $destination/$runDir"
    
	errCode=0
    
    #---												---#
    #---	Test for links to analysis results files	---#
    #---	Typically, 1.wells and bfmask.bin have links---#
    #---	If links exist, then we cannot delete		---#
    #---												---#
    linkedRunList=( $(cd $(dirname $run);ls -l */1.wells 2>/dev/null|grep ">"|cut -f2 -d'>') )
    linkFound=0
    for linkedRun in ${linkedRunList[@]}; do
    	linkedRun=$(basename $(dirname $linkedRun))
        if [ "$linkedRun" == "$(basename $run)" ]; then
        	echo "Found a linked run: $run"
		    echo "We cannot remove this run until linking run is removed"
		    echo "Status: Incomplete"
            linkFound=1
            break
        fi
    done
    if [ $linkFound -eq 1 ]; then
    	continue
    fi
    
    #---								---#
    #--- Test for valid run directory	---#
    #---								---#
    cd ${run}
    if [ $? -ne 0 ]; then
        echo "Skipping invalid directory: $run"
        echo "Status: Invalid direcetory"
        continue
    fi
    
    runDir=$(basename $run)
    
    #---									---#
    #--- Handle each file in the filelist	---#
    #---									---#
    for file in ${fileList[@]}; do
    	if [ -r $file ]; then
        	#---	destination dir must exist	---#
            mkdir -p $destination/$runDir
            if [ $? -ne 0 ]; then
            	echo "Error making: $destination/$runDir"
                errCode=1
                continue
            fi
	    	#---	archive the file	---#
        	rsync -azr $file $destination/$runDir
            if [ $? -ne 0 ]; then
            	echo "Error with rsync: $file to $destination/$runDir"
            fi
        else
        	echo "Missing: $file"
            continue
        fi
        
        #---	verify archive copy	---#
        diff -r ${run}/${file} ${destination}/${runDir}/${file}
        if [ $? -ne 0 ]; then
        	echo "File does not match archive copy: $file"
        	errCode=1
        fi
    done
    
    #---	If destination does not exist, then nothing got copied	---#
	#---	NOTE: this needs to be fixed: if no files are backed up, then the
	#--- source directory is left in place although it contains nothing worth saving.
	#--- OR, there was an error creating the destination directory.	---#
    if [ ! -d $destination/$runDir ]; then
    	echo "Nothing was archived"
        echo "Status: Incomplete"
        continue
    else
    
        #---	diagnostics: disk space savings	---#
        echo "Original size: $(du -sk $run|awk '{print $1}') kbytes"
        echo "Archived size: $(du -sk $destination/$runDir|awk '{print $1}') kbytes"
    
    fi

    #---	Remove the run directory if no errors encountered in archive	---#
	if [ $errCode -eq 0 ]; then
    	#---	remove run directory	---#
    	if [ $REALLY_DELETE ]; then rm -rf $run; fi
        if [ $? -ne 0 ]; then
        	echo "Error removing directory: $run"
            echo "Status: Incomplete"
        fi
        echo "Original deleted: $run"
        echo "Status: Complete"
    else
    	echo "Incomplete archive: original not deleted"
        echo "Status: Incomplete"
    fi
    
done

exit 0
