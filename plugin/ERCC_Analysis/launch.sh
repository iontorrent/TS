#!/usr/bin/env ionPluginShell
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

VERSION="3.2.44320"

OUTFILE=${RESULTS_DIR}/${PLUGINNAME}.html

REFERENCE="${PLUGIN_PATH}/ERCC92/ERCC92.fasta"
if [ -e "$REFERENCE" ]; then
  echo "reference is '$REFERENCE'"
else
    echo "ERROR: NO FASTA REFERENCE AVAILABLE"
    exit 1
fi

#call tmap to create the sam file
python -c"import sys; sys.path.insert(0, '$DIRNAME/code'); import preproc_fastq; preproc_fastq.fastq_preproc('${TSP_FILEPATH_FASTQ}', '${RESULTS_DIR}/filtered.fastq',15,20);"
tmap mapall -f "$REFERENCE" -r "${RESULTS_DIR}/filtered.fastq" -a 1 -g 0 -n 8 stage1 map1 --seed-length 18 stage2 map2 map3 --seed-length 18 > ${RESULTS_DIR}/tmap.sam
#create the html report
python $DIRNAME/ERCC_Analysis.py "${RESULTS_DIR}" "${ANALYSIS_DIR}" "${URL_ROOT}" "${PLUGINNAME}" "${PLUGINCONFIG__MINRSQUARED}" "$DIRNAME"> $OUTFILE
