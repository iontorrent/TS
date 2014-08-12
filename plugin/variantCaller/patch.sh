# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
#!/bin/bash
#patched tvc runs 

normal=`tput sgr0`
bold=`tput bold`

if [ -z "$1" ]; then
    echo "${bold}Ussage: ${normal} takes in 1 arg, the primary key (int) of the result with the tvc to be patched"
    exit 1
fi

echo Looking up report $1

#get the report path from the API
REPORT_PATH=$(curl -s http://localhost/rundb/api/v1/results/$1/?format=json | python -c "import json;import sys;j = json.load(sys.stdin);print j.get('filesystempath')")

#if that fails then bail
if [ -z "$REPORT_PATH" ]; then
    echo "${bold}Not Found: ${normal}are you sure the key is correct? It should be an int"
    exit 1
fi

#step 1 - create or replace the extend.py file
echo "Installing new extend.py"
curl -o /results/plugins/variantCaller/extend.py http://labs.radiantmachines.com/ion/tvc/extend.py
chown ionadmin:ionadmin /results/plugins/variantCaller/extend.py

printf "\n"

if [ -n "$REPORT_PATH" ]; then
    TVC_PATH=${REPORT_PATH}/plugin_out/variantCaller_out

    #check to see if there if it is barcoded
    BARCODE=$(cat ${TVC_PATH}/startplugin.json | jsonpipe | grep runinfo | grep barcodeId | awk '{print $2}' | sed 's/\"//g')
    
    if [ -n "$BARCODE" ]; then
        echo "barcode set ${BARCODE}"
        #get the barcode dirs
        for OUTPUT in $(ls -1 -d ${TVC_PATH}/${BARCODE}*/)
        do
            echo "Patching barcode ${OUTPUT}/lifegrid"

            curl -o ${OUTPUT}/lifegrid/allelesTable.js http://labs.radiantmachines.com/ion/tvc/allelesTable.js
            chown ionian:ionian ${OUTPUT}/lifegrid/allelesTable.js

            curl -o ${OUTPUT}/lifegrid/pager.js http://labs.radiantmachines.com/ion/tvc/pager.js
            chown ionian:ionian ${OUTPUT}/lifegrid/pager.js

        done

    fi 

    if [ -z "$BARCODE" ]; then
        echo "Patching nonbarcoded run at ${TVC_PATH}"

        curl -o ${TVC_PATH}/lifegrid/allelesTable.js http://labs.radiantmachines.com/ion/tvc/allelesTable.js
        chown ionian:ionian ${TVC_PATH}/lifegrid/allelesTable.js

        curl -o ${TVC_PATH}/lifegrid/pager.js http://labs.radiantmachines.com/ion/tvc/pager.js
        chown ionian:ionian ${TVC_PATH}/lifegrid/pager.js

    fi 

    

fi

