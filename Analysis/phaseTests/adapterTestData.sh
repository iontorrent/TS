# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#!/bin/bash

N_READS=10
MEAN_INSERT_LEN=100
SNR=10

for FLOW_ORDER in TACG TACGTACGTCTGAGCATCGATCGATGTACAGC; do
  for ADAPTER_SEQ in CTGAGACTGCCAAGGCACACAGGGGATAGG ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG; do
    OUT_FILE=sim.flow-$FLOW_ORDER.adapter-$ADAPTER_SEQ.txt
    flowSim \
      --out         $OUT_FILE \
      --flow-order  $FLOW_ORDER \
      --adapter-seq $ADAPTER_SEQ \
      --reads       $N_READS \
      --len-mean    $MEAN_INSERT_LEN \
      --snr         $SNR
  done
done
