# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# Build and run SFFTrimTest.
# Invoke SFFTrimTest.R to evaluate results.

g++ -I../ SFFTrimTest.cpp -o SFFTrimTest
./SFFTrimTest 16 sim-xdb-p1.frame > test.out
R CMD BATCH SFFTrimTest.R
cat test.results

