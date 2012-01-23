# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#!/bin/bash
PHASE_SOLVE_TEST=../../build/Analysis/phaseSolveTest
PHASE_PARAMS_SIM="--cf 0.010 --ie 0.008 --dr 0.0015"
PHASE_PARAMS_SOL="--cf 0.013 --ie 0.012 --dr 0.0020"
N_READS=100
SNR=8

for FLOW_ORDER in TACG TACGTACGTCTGAGCATCGATCGATGTACAGC TACGTACGTACGTACGTACGTACATACGCACGTGCGTATG; do
  for N_BASES in 100; do
    DATA_FILE=cafiedata.$FLOW_ORDER.$N_READS
    $PHASE_SOLVE_TEST $PHASE_PARAMS_SIM --output-flowvals $DATA_FILE --bases $N_BASES --reads $N_READS --snr $SNR --flow-order $FLOW_ORDER
  
    time ($PHASE_SOLVE_TEST $PHASE_PARAMS_SOL --input-flowvals $DATA_FILE --bases $N_BASES --reads $N_READS --snr $SNR --flow-order $FLOW_ORDER --solver-type CafieSolver \
      > out.$FLOW_ORDER.$N_BASES.CafieSolver) 2> timing.$FLOW_ORDER.$N_BASES.CafieSolver
    for POP_LIMIT in 00 15 20 25 30; do
      command="$PHASE_SOLVE_TEST $PHASE_PARAMS_SOL --input-flowvals $DATA_FILE --bases $N_BASES --reads $N_READS --snr $SNR --flow-order $FLOW_ORDER --solver-type PhaseSim --populations-solve $POP_LIMIT"
      echo $command
      time ( $command > out.$FLOW_ORDER.$N_BASES.PhaseSim.lim$POP_LIMIT) 2> timing.$FLOW_ORDER.$N_BASES.PhaseSim.lim$POP_LIMIT
    done
  done
done
