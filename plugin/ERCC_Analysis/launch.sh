#!/usr/bin/env ionPluginShell
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

VERSION="3.4.50010"

OUTFILE=${RESULTS_DIR}/${PLUGINNAME}.html

REFERENCE="${PLUGIN_PATH}/ERCC92/ERCC92.fasta"
if [ -e "$REFERENCE" ]; then
  echo "reference is '$REFERENCE'"
else
    echo "ERROR: NO FASTA REFERENCE AVAILABLE"
    exit 1
fi


if [ -e "$TSP_FILEPATH_BARCODE_TXT" ]; then
    echo "Barcoded run"
    INPUT_BAM=${REPORT_ROOT_DIR}/nomatch_rawlib.bam
else
    echo "non barcoded run"
    INPUT_BAM=${TSP_FILEPATH_BAM}
fi

# preprocess
python -c"import sys; sys.path.insert(0, '$DIRNAME/code'); import preproc_fastq; preproc_fastq.bam_preproc('${INPUT_BAM}', '${RESULTS_DIR}/filtered.fastq',15,20);"

#call tmap to create the sam file
tmap mapall -f "$REFERENCE" -r "${RESULTS_DIR}/filtered.fastq" -a 1 -g 0 -n 8 stage1 map1 --seed-length 18 stage2 map2 map3 --seed-length 18 > ${RESULTS_DIR}/tmap.sam

#create the html report
python $DIRNAME/ERCC_Analysis.py "${RESULTS_DIR}" "${ANALYSIS_DIR}" "${URL_ROOT}" "${PLUGINNAME}" "${PLUGINCONFIG__MINRSQUARED}" "$DIRNAME" "${PLUGINCONFIG__MINCOUNTS}" "${PLUGINCONFIG__ERCCPOOL}"> $OUTFILE

# find filtered*.bam files in $(RESULTS_DIR) and delete them
find ${RESULTS_DIR} -type f -name 'filtered*.fastq' | xargs echo
find ${RESULTS_DIR} -type f -name 'filtered*.fastq' | xargs rm
echo 'now we remove tmap.sam'
rm ${RESULTS_DIR}/tmap.sam
