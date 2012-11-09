#!/usr/bin/env bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
set -ex


EXPECTED_ARGS=1
if [ $# -ne $EXPECTED_ARGS ]
then
   echo "please specify a raw directory e.g. /results/PGM_test/cropped_CB1-42"
   exit 1
fi

if [ -e $1/acq_0000.dat ]
then
    BLOCK=$1
else
    echo "raw directory doesn't contain acq_0000.dat"
    exit 1
fi

function drop_caches {
   sync
   echo 1 > /proc/sys/vm/drop_caches
   echo 2 > /proc/sys/vm/drop_caches
   echo 3 > /proc/sys/vm/drop_caches
}

function init_block {
   TEMPDIR=`mktemp -d`
   echo $TEMPDIR
}


JBF="justBeadFind --beadfind-minlivesnr 3 --beadfind-lagone-filt 0 --local-wells-file off \
         --region-size=216x224 --beadfind-num-threads 12 --no-subdir --librarykey=TCAG --tfkey=ATCG"



ANA="Analysis --beadfind-minlivesnr 3 --beadfind-lagone-filt 0 --clonal-filter-bkgmodel on --bkg-bfmask-update off \
         --region-size=216x224 --gpuWorkLoad 1 --numcputhreads 12 --from-beadfind --local-wells-file off --restart-next step.59 --flowlimit 260 \
         --start-flow-plus-interval 0,60 --no-subdir --librarykey=TCAG --tfkey=ATCG"


#ANA="Analysis --regional-sampling on --beadfind-minlivesnr 3 --beadfind-lagone-filt 0 --clonal-filter-bkgmodel off \
#         --region-size=216x224 --gpuWorkLoad 1 --numcputhreads 12 --from-beadfind --local-wells-file off --restart-next step.259 --flowlimit 260 \
#         --start-flow-plus-interval 0,260 --no-subdir --librarykey=TCAG --tfkey=ATCG"


BC="BaseCaller --beverly-filter 0.04,0.04,8 --keypass-filter off --phasing-residual-filter=2.0 --trim-qual-cutoff 100.0 --trim-adapter-cutoff 16 \
         --librarykey=TCAG --tfkey=ATCG --run-id=VJ0V5 --block-col-offset 6440 --block-row-offset 3335 \
         --flow-order TACGTACGTCTGAGCATCGATCGATGTACAGC --trim-qual-cutoff 9 --trim-qual-window-size 30 --trim-adapter-cutoff 16 \
         --trim-adapter ATCACCGACTGCCCATAGAGAGGCTGAGAC --barcode-filter 0.01"


AL="alignmentQC.pl --genome hg19 --max-plot-read-len 400 -p 1"

function run_block {
  time $JBF $EXTRA_JBF_OPTIONS --output-dir=$TEMPDIR $BLOCK >  test1.log
  time $ANA $EXTRA_ANA_OPTIONS --output-dir=$TEMPDIR $BLOCK >> test1.log
  time $BC                     --input-dir=$TEMPDIR --output-dir=$TEMPDIR >> test1.log
  time $AL                     --input $TEMPDIR/*basecaller.bam  --output-dir=$TEMPDIR >>  test1.log
  grep -e "Filtered Mapped Bases in Q17 Alignments" -e "Filtered Mapped Bases in Q20 Alignments" $TEMPDIR/alignment.summary
}

#1
drop_caches
EXTRA_JBF_OPTIONS=""
EXTRA_ANA_OPTIONS="--fitting-taue on --bkg-single-alternate"
init_block
#run_block

#2
drop_caches
EXTRA_JBF_OPTIONS=""
EXTRA_ANA_OPTIONS="--bkg-single-alternate"
init_block
#run_block

#3
drop_caches
EXTRA_JBF_OPTIONS=""
EXTRA_ANA_OPTIONS="--fitting-taue on"
init_block
#run_block

#4
drop_caches
EXTRA_JBF_OPTIONS=""
EXTRA_ANA_OPTIONS=""
init_block
run_block

#5
drop_caches
EXTRA_JBF_OPTIONS="--regional-sampling on"
EXTRA_ANA_OPTIONS="--fitting-taue on --bkg-single-alternate"
init_block
#run_block

#6
drop_caches
EXTRA_JBF_OPTIONS="--regional-sampling on"
EXTRA_ANA_OPTIONS="--bkg-single-alternate"
init_block
#run_block

#7
drop_caches
EXTRA_JBF_OPTIONS="--regional-sampling on"
EXTRA_ANA_OPTIONS="--fitting-taue on"
init_block
#run_block

#8
drop_caches
EXTRA_JBF_OPTIONS="--regional-sampling on"
EXTRA_ANA_OPTIONS="--regional-sampling on"
init_block
run_block
