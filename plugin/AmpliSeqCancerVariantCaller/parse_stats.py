#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import json

inf = open(sys.argv[1],'r')
out = open(sys.argv[2],'w')

total_reads = int(inf.readline())
mapped_reads = int(inf.readline())
target_reads = int(inf.readline())
total_bases = int(inf.readline())
avg_cov = float(inf.readline())
uncov = float(inf.readline())
cov_1x = float(inf.readline())
cov_10x = float(inf.readline())
cov_100x = float(inf.readline())

percent_on_target = 100.0*float(target_reads)/float(total_reads)
percent_off_target = 100.0*float(mapped_reads-target_reads)/float(total_reads)
percent_unmapped = 100.0*float(total_reads-mapped_reads)/float(total_reads)


sys.stdout.write("\t\t\t\t\t\t<tr><td>%d</td><td>%d</td><td>%d</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%d</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>\n" % (total_reads,mapped_reads,target_reads,percent_on_target,percent_off_target,percent_unmapped,total_bases,avg_cov,uncov,cov_1x,cov_10x,cov_100x))
out.write(json.dumps ( {'Target Statistics' : {'total_reads' : total_reads, 'mapped_reads' : mapped_reads, 'on_target' : target_reads, '% on target': percent_on_target, '% off target': percent_off_target, '% unmapped':percent_unmapped, 'Total covered bases': total_bases, 'Avg cov': avg_cov, '% uncovered bases': uncov, '% cov 1x':cov_1x, '% cov 10x':cov_10x, '% cov 100x':cov_100x}},indent=4))

inf.close()
out.close()
