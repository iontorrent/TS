#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved


# write the version
VERSION="2.2.3-31037"
export metaDataFile=${RESULTS_DIR}/torrentConfig.env.txt
export logFile=${RESULTS_DIR}/log.txt
export stdoutFile=${RESULTS_DIR}/stdout.txt
export stderrFile=${RESULTS_DIR}/stderr.txt
export timeStamp=`date +%s`

COMMANDLINE="$0 $@"
COMMANDLINEARGS="$@"
PLUGINEXITCODE=0
echo "" > ${metaDataFile}
echo "" > ${stdoutFile}
echo "" > ${stderrFile}
echo VERSION=1.0 > ${logFile}
echo "`date +%F_%H%M%S`:$PROGRAM:command $COMMANDLINE" >>${logFile}


export JSON_FILENAME=""
while getopts ":hj:" opt; do
    case $opt in
        j  )
            JSON_FILENAME=$OPTARG
            ;;
        h  )
            echo "usage: $PROGRAM ..."
            echo "arguments:"
            echo "  -h                  print this help and exit"
            echo "  -j JSON_FILENAME    specify the configuration object in the form of a json object file"
            exit 0
            ;;
    esac
done
shift $(($OPTIND - 1))


cp ${PLUGIN_PATH}/status_block.html ${RESULTS_DIR}/status_block.html

export LIFESCOPESERVERADDR=""
export LIFESCOPEUSER=""
export LIFESCOPEPASSWORD=""
export LIFESCOPESERVERDEST=""
export LIFESCOPESERVERPORT=8180
export LIFESCOPESERVERADDR=`cat ${JSON_FILENAME}|grep -A 6 pluginconfig|sed 's/^ *//;s/ *$//g'|grep "^\"server"|sed 's/^\"server\": \"//'|sed 's/,$//g'|sed 's/\"$//g'`
export LIFESCOPEUSER=`cat ${JSON_FILENAME}|grep -A 6 pluginconfig|sed 's/^ *//;s/ *$//g'|grep "^\"username"|sed 's/^\"username\": \"//'|sed 's/,$//g'|sed 's/\"$//g'`
export LIFESCOPEPASSWORD=`cat ${JSON_FILENAME}|grep -A 6 pluginconfig|sed 's/^ *//;s/ *$//g'|grep "^\"password"|sed 's/^\"password\": \"//'|sed 's/,$//g'|sed 's/\"$//g'`
export LIFESCOPESERVERDEST=`cat ${JSON_FILENAME}|grep -A 6 pluginconfig|sed 's/^ *//;s/ *$//g'|grep "^\"dest"|sed 's/^\"dest\": \"//'|sed 's/,$//g'|sed 's/\"$//g'`
export LIFESCOPESERVERPORT=`cat ${JSON_FILENAME}|grep -A 6 pluginconfig|sed 's/^ *//;s/ *$//g'|grep "^\"port"|sed 's/^\"port\": \"//'|sed 's/,$//g'|sed 's/\"$//g'`
export LIFESCOPEPGMID=`cat ${ANALYSIS_DIR}/ion_params_00.json |grep serial_number |sed 's/,/\n/g' |grep serial_number |sed 's/\\\\//g' |sed 's/\"//g'|sed 's/ //g' |cut -d ":" -f2 | head -1`;
export LIFESCOPECHIPID=`cat ${ANALYSIS_DIR}/ion_params_00.json |grep chipBarcode |sed 's/,/\n/g' |grep chipBarcode |sed 's/\\\\//g' |sed 's/\"//g'|sed 's/ //g' |cut -d ":" -f2 | head -1`;
export LIFESCOPEBARCODEID=`cat ${ANALYSIS_DIR}/ion_params_00.json |grep barcodeId |sed 's/,/\n/g' |grep barcodeId |sed 's/\\\\//g' |sed 's/\"//g'|sed 's/ //g' |cut -d ":" -f2 | head -1`;




#######
#######
## write the environment variables into the environment file
##
echo RAW_DATA_DIR=$RAW_DATA_DIR  >>${metaDataFile}
echo ANALYSIS_DIR=$ANALYSIS_DIR >>${metaDataFile}
echo LIBRARY_KEY=$LIBRARY_KEY >>${metaDataFile}
echo TESTFRAG_KEY=$TESTFRAG_KEY >>${metaDataFile}
echo RESULTS_DIR=$RESULTS_DIR >>${metaDataFile}
echo URL_ROOT=$URL_ROOT >>${metaDataFile}
echo PWD=${PWD} >>${metaDataFile}
echo PATH=${PATH} >>${metaDataFile}
echo RUNINFO__PLUGIN_DIR=$RUNINFO__PLUGIN_DIR >>${metaDataFile}
echo RUNINFO__PLUGIN_NAME=$RUNINFO__PLUGIN_NAME >>${metaDataFile}
echo RUNINFO__PK=$RUNINFO__PK >>${metaDataFile}
echo TSP_RUN_NAME=$TSP_RUN_NAME >>${metaDataFile}
echo TSP_RUN_DATE=$TSP_RUN_DATE >>${metaDataFile}
echo TSP_ANALYSIS_NAME=$TSP_ANALYSIS_NAME >>${metaDataFile}
echo TSP_ANALYSIS_DATE=$TSP_ANALYSIS_DATE >>${metaDataFile}
echo TSP_PROJECT=$TSP_PROJECT >>${metaDataFile}
echo TSP_LIBRARY=$TSP_LIBRARY >>${metaDataFile}
echo TSP_SAMPLE=$TSP_SAMPLE >>${metaDataFile}
echo TSP_PGM_NAME=$TSP_PGM_NAME >>${metaDataFile}
echo TSP_NOTES=$TSP_NOTES >>${metaDataFile}
echo TSP_CHIPTYPE=$TSP_CHIPTYPE >>${metaDataFile}
echo TSP_FILEPATH_OUTPUT_STEM=$TSP_FILEPATH_OUTPUT_STEM >>${metaDataFile}
echo TSP_URLPATH_OUTPUT_STEM=$TSP_URLPATH_OUTPUT_STEM >>${metaDataFile}
echo TSP_FILEPATH_GENOME_FASTA=$TSP_FILEPATH_GENOME_FASTA >>${metaDataFile}
echo TSP_URLPATH_GENOME_FASTA=$TSP_URLPATH_GENOME_FASTA >>${metaDataFile}
echo TSP_FILEPATH_BAM=$TSP_FILEPATH_BAM >>${metaDataFile}
echo TSP_URLPATH_BAM=$TSP_URLPATH_BAM >>${metaDataFile}
echo TSP_FLOWORDER=$TSP_FLOWORDER >>${metaDataFile}
echo TSP_RUNID=$TSP_RUNID >>${metaDataFile}
echo TSP_LIBRARY_KEY=$TSP_LIBRARY_KEY >>${metaDataFile}
echo TSP_TF_KEY=$TSP_TF_KEY >>${metaDataFile}
echo TSP_NUM_FLOWS=$TSP_NUM_FLOWS >>${metaDataFile}
echo TSP_FILEPATH_PLUGIN_DIR=$TSP_FILEPATH_PLUGIN_DIR >>${metaDataFile}
echo TSP_URLPATH_PLUGIN_DIR=$TSP_URLPATH_PLUGIN_DIR >>${metaDataFile}
echo TSP_FILEPATH_BARCODE_TXT=$TSP_FILEPATH_BARCODE_TXT >>${metaDataFile}
echo TSP_URLPATH_BARCODE_TXT=$TSP_URLPATH_BARCODE_TXT >>${metaDataFile}
echo LIFESCOPESERVERADDR=$LIFESCOPESERVERADDR >>${metaDataFile}
echo LIFESCOPESERVERPORT=$LIFESCOPESERVERPORT >>${metaDataFile}
echo LIFESCOPEUSER=$LIFESCOPEUSER >>${metaDataFile}
echo LIFESCOPEPROJECT=$TSP_PROJECT >>${metaDataFile}
echo LIFESCOPEANALYSIS=$TSP_ANALYSIS_NAME >>${metaDataFile}
echo LIFESCOPEPGMID=$LIFESCOPEPGMID >>${metaDataFile}
echo LIFESCOPECHIPID=$LIFESCOPECHIPID >>${metaDataFile}
echo LIFESCOPEBARCODEID=$LIFESCOPEBARCODEID >>${metaDataFile}
echo PLUGINEXECTIMESTAMP=$timeStamp >>${metaDataFile}







#######
#######
## download the current report.pdf 
##
#analysisHome=`dirname ${ANALYSIS_DIR}`      
#analysisHome_FolderName=`basename ${analysisHome}`  #basically, get folder name of the folder one level above the actual analysis dir.
#analysisDir_FolderName=`basename  ${ANALYSIS_DIR}`
#AnalysisRootUrl=http://localhost/output/${analysisHome_FolderName}/${analysisDir_FolderName}
#reportPdfUrl=${AnalysisRootUrl}/Default_Report.php?do_print=True
#echo ${reportPdfUrl}
#wget ${reportPdfUrl} -O report.pdf






#######
#######
## setting up the environment for the cgr client
##
#export CLASSPATH=`find /results/plugins/IonReporter/ |grep "\.jar$" |xargs |sed 's/ /:/g'`
export CLASSPATH=`find ${RUNINFO__PLUGIN_DIR} |grep "\.jar$" |xargs |sed 's/ /:/g'`
export CLASSPATH=${RUNINFO__PLUGIN_DIR}/lib/java/shared/:$CLASSPATH
export LD_LIBRARY_PATH=${RUNINFO__PLUGIN_DIR}/lib:${LD_LIBRARY_PATH}
echo "CLASSPATH=${CLASSPATH}" >>${metaDataFile}
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >>${metaDataFile}
#echo $CLASSPATH



#######
#######
##touch the report to make sure it is really there
##
net=${RUNINFO__NET_LOCATION}
url=${RUNINFO__URL_ROOT}
report="/Default_Report.php?do_print=true"

echo "http://127.0.0.1/${url}${report}"
echo "${net}${url}${report}"

wget -q -O /dev/null "http://127.0.0.1/${url}${report}" || true
wget -q -O /dev/null "${net}${url}${report}" || true


#######
#######
## Get the plugin parameters from the database objects and dump it as objects.json for the client to use.
##
# first try the standard method to get the objects
echo "${RUNINFO__API_URL}/v1/plugin/?format=json&name=IonReporterUploader_V1_0&active=true" 
wget  -q -O ${RESULTS_DIR}/objects.json "${RUNINFO__API_URL}/v1/plugin/?format=json&name=IonReporterUploader_V1_0&active=true" || true
wgetexitcode=$?
sleep 2

#if [ $wgetexitcode -ne 0 ] ; then
if [ ! -s  ${RESULTS_DIR}/objects.json ] ; then
   # if the first method fails, try a hardcoded URL to get the objects
   echo "http://127.0.0.1/rundb/api/v1/plugin/?format=json&name=IonReporterUploader_V1_0&active=true"
   wget -q -O ${RESULTS_DIR}/objects.json "http://127.0.0.1/rundb/api/v1/plugin/?format=json&name=IonReporterUploader_V1_0&active=true"  || true
   wgetexitcode=$?
   sleep 2
   if [ ! -s  ${RESULTS_DIR}/objects.json ] ; then
      # if the second method also fails, then it is a problem. no use trying anything else. fail the run.
      echo ERROR getting objects from the database.
      exit 1
   fi
fi
sleep 2
if [ ! -s  ${RESULTS_DIR}/objects.json ] ; then
   # if the file size is still empty then, something went wrong in getting the objects from the db.
   echo ERROR no objects retreived from the database. zero file size on  ${RESULTS_DIR}/objects.json
   exit 1
fi

echo ""
echo ""
echo ""
echo "       Plugin log is at :    $logFile"
echo ""
echo ""
echo ""

#######
#######
## execute the launcher java client
##
echo "`date +%F_%H%M%S`:$PROGRAM:executing the IonReporter Uploader Client ..." >>${logFile}
java -Xmx4g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.torrent.plugin.lifescope.Launcher -j ${RESULTS_DIR}/startplugin.json -c ${metaDataFile} -l ${logFile}  ||true
LAUNCHERCLIENTEXITCODE=$?
# for safety sake, just sleep one or two seconds before you write anything on to the log file just closed by another progam... its causing problems..
sleep 2
echo "`date +%F_%H%M%S`:$PROGRAM: executed the IonReporter Uploader Client  ...Exit Code = ${LAUNCHERCLIENTEXITCODE}" >>${logFile}
#PLUGINEXITCODE=`expr $PLUGINEXITCODE + $LAUNCHERCLIENTEXITCODE`
#exit $PLUGINEXITCODE











