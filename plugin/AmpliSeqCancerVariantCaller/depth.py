#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import fileinput
from collections import defaultdict

cov = 0 
bases_1x = 0
bases_10x = 0
bases_100x = 0
#total_target_bases = 22693-1635 # remove primer bases upto second U - primer cleavage
total_target_bases = 13370 # remove all primer bases 
covered_bases = 0
depth = defaultdict(int)

for p in fileinput.input():
    tmp = p.split()
    cov += int(tmp[2])
    depth[int(tmp[2])] += 1
    covered_bases += 1

for k in depth.keys():
    if k >= 10:
        bases_10x += depth[k]
    if k >=100:
        bases_100x += depth[k]

frac_uncovered = float(total_target_bases - covered_bases)*100.0/float(total_target_bases)
frac_1x = float(covered_bases)*100.0/float(total_target_bases)
frac_10x = float(bases_10x)*100.0/float(total_target_bases)
frac_100x = float(bases_100x)*100.0/float(total_target_bases)


print  str(cov)
print  str(float(cov)/float(total_target_bases))
print  str(frac_uncovered)
print  str(frac_1x)
print  str(frac_10x)
print  str(frac_100x)
