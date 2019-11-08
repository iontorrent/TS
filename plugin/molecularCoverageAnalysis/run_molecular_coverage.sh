#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

BIN_DIR=$1
BAM_FILE=$2
REF_GENOME=$3
SAMPLEID=$4
LIBTYPE=$5
TARGET_BED_FILE=$6
NUM_THREADS=$7
OUTPUT_DIR=$8
OUTPUT_PREFIX=$9
FILE_LINKS_FILE=${OUTPUT_DIR}/filelinks.xls
TVC_JSON=${OUTPUT_DIR}/local_parameters.json
STATS_FILE=${OUTPUT_DIR}/${OUTPUT_PREFIX}.cov.stats.json
STATS_TXT_FILE=${OUTPUT_DIR}/${OUTPUT_PREFIX}.cov.stats.txt
COV_FILE=${OUTPUT_DIR}/${OUTPUT_PREFIX}.amplicon.cov.xls
TCCINITFILE=${OUTPUT_DIR}/tmc.aux.ttc.xls

python ${BIN_DIR}/molecular_coverage_analysis.py --bam-file ${BAM_FILE} --bed-file ${TARGET_BED_FILE} --num-threads ${NUM_THREADS} --output-dir ${OUTPUT_DIR} --output-prefix ${OUTPUT_PREFIX} --tvc-json ${TVC_JSON} --ignore-zr  1 --make-plots 0
cp ${COV_FILE} ${COV_FILE}.bak
head -1 ${COV_FILE}.bak > ${COV_FILE}
tail -n +2  ${COV_FILE}.bak | sort -nk 6 >> ${COV_FILE}
rm -f ${COV_FILE}.bak
perl ${BIN_DIR}/bin/mol_stats.pl ${OUTPUT_DIR} ${OUTPUT_PREFIX}
perl ${BIN_DIR}/bin/target_coverage.pl -G ${REF_GENOME} ${COV_FILE} - - 0 100000000 100 0 100 -1 > ${TCCINITFILE}
echo -e "\n\n" >> ${STATS_FILE}
echo -e "Reference Genome: hg19" >> ${STATS_TXT_FILE}
echo -e "Target Regions: `basename ${TARGET_BED_FILE} .bed`" >> ${STATS_TXT_FILE}
echo -e "Sample Name: ${SAMPLEID}" >> ${STATS_TXT_FILE}
echo -e "Library Type: ${LIBTYPE}" >> ${STATS_TXT_FILE}
perl ${BIN_DIR}/bin/summary.pl ${COV_FILE} ${TVC_JSON} >> ${STATS_TXT_FILE}
perl ${BIN_DIR}/bin/rename.pl ${STATS_TXT_FILE} >> ${STATS_FILE}

PLUGIN_OUT_STATSFILE="${OUTPUT_PREFIX}.cov.stats.txt"
PLUGIN_OUT_AMPCOVFILE="${OUTPUT_PREFIX}.amplicon.cov.xls"
PLUGIN_OUT_BEDPAGE=`echo ${TARGET_BED_FILE} | sed -e 's/^.*\/uploads\/BED\/\([0-9][0-9]*\)\/.*/\/rundb\/uploadstatus\/\1/'`

source "${BIN_DIR}/bin/fileLinks.sh"
write_file_links "${OUTPUT_DIR}" "filelinks.xls";
