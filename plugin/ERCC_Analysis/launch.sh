#!/usr/bin/env ionPluginShell
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

VERSION="3.6.56201"

OUTFILE=${RESULTS_DIR}/${PLUGINNAME}.html

REFERENCE="${PLUGIN_PATH}/ERCC92/ERCC92.fasta"
if [ -e "$REFERENCE" ]; then
  echo "reference is '$REFERENCE'"
else
    echo "ERROR: NO FASTA REFERENCE AVAILABLE"
    exit 1
fi

#TODO if TSP_FILEPATH_BARCODE_TXT, then INPUT_BAM = the user input for barcode, ${PLUGINCONFIG__BARCODE}, and maybe also set a new variable for BARCODING_USED
if [ -e "$TSP_FILEPATH_BARCODE_TXT" ]; then
    echo "Barcoded run"
    INPUT_BAM=${RESULTS_DIR}/$PLUGINCONFIG__BARCODE
	echo 'INPUT_BAM'
	echo $INPUT_BAM
	BARCODING_USED="Y"
	echo 'BARCODING_USED'
	echo $BARCODING_USED
else
    echo "non barcoded run"
    INPUT_BAM=${TSP_FILEPATH_BAM}
	BARCODING_USED="N"
fi

# preprocess
python $DIRNAME/code/preproc_fastq.py "$INPUT_BAM" "$BARCODING_USED" "${RESULTS_DIR}"

#call tmap to create the sam file
if [ -e "${RESULTS_DIR}/filtered.fastq" ]; then
	tmap mapall -f "$REFERENCE" -r "${RESULTS_DIR}/filtered.fastq" -a 1 -g 0 -n 8 stage1 map1 --seed-length 18 stage2 map2 map3 --seed-length 18 > ${RESULTS_DIR}/tmap.sam
	FASTQ_EXISTS="Y"
else
	FASTQ_EXISTS="N"
fi

#create the html report
python $DIRNAME/ERCC_Analysis.py "${RESULTS_DIR}" "${ANALYSIS_DIR}" "${URL_ROOT}" "${PLUGINNAME}" "${PLUGINCONFIG__MINRSQUARED}" "$DIRNAME" "${PLUGINCONFIG__MINCOUNTS}" "${PLUGINCONFIG__ERCCPOOL}" "$FASTQ_EXISTS" "${PLUGINCONFIG__BARCODE}"> $OUTFILE

# find filtered*.bam files in $(RESULTS_DIR) and delete them
if [ -e "${RESULTS_DIR}/filtered.fastq" ]; then
    find ${RESULTS_DIR} -type f -name 'filtered*.fastq' | xargs echo
    find ${RESULTS_DIR} -type f -name 'filtered*.fastq' | xargs rm
fi
if [ -e "${RESULTS_DIR}/tmap.sam" ]; then
    echo 'now we remove tmap.sam'
    rm ${RESULTS_DIR}/tmap.sam
fi
