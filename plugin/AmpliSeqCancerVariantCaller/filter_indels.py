#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import math


inf = open(sys.argv[1],'r')
out = open(sys.argv[2],'w')
		
for lines in inf:
	if lines[0]=='#':
		out.write(lines)
	else:
                attr={}
		fields = lines.split('\t')
		variant = 0
		info = fields[7].split(';')
                for items in info[:-1]:
                    key,val = items.split('=')
                    attr[key]=val
                var_freq = float(attr['Variants-freqs'].split(',')[0])*100.0
                total_cov = attr['Num-spanning-reads']
		var_list = attr['Num-variant-reads'].split(',')
		var_list = [int(x) for x in var_list];
		ref_cov = int(total_cov) - sum(var_list)
                var_cov = attr['Num-variant-reads'].split(',')[0]
		(plus,minus) = attr['Plus-minus-strand-counts'].split(',')[0].split('/')
	
                if ( float(attr['Score']) < 10.0 or int(var_cov) < 10 or var_freq < 5.0 or (len(fields[3]) == 1 and len(fields[4])==1)):
			continue
		else:
			out.write(lines)
inf.close()
out.close()
