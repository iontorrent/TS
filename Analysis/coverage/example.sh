#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

echo "EXAMPLE 1 - summary only:"
./seqCoverage < example.input

echo
echo "EXAMPLE 2 - union blocks:"
./seqCoverage -b < example.input
