#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser
import re
import gzip

def check_option(parser, value, name):
    if None == value:
        print 'Option ' + name + ' required.\n'
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-v', '--vcf-file', help='the bgzip compressed VCF file', dest='vcf_file')
    parser.add_option('-n', '--top-x-variants', type="int", help='the number of top X variants to display', dest='top_x_variants', default=100)
    parser.add_option('-s', '--igv-session-xml-url', help='the url to igv session XML file', dest='igv_session_xml_url')
    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        exit(1)
    check_option(parser, options.vcf_file, '-v')
    check_option(parser, options.top_x_variants, '-n')
    check_option(parser, options.igv_session_xml_url, 's')

vcf = gzip.open(options.vcf_file)
queue = [];

for lines in vcf:
    if not lines[0]=='#':
        genotype = ''
        fields = lines.split('\t')
        contig = fields[0]
        pos = fields[1]
        ref = fields[3]
        alt = fields[4].split(',')
        alt_first = alt[0]
        # output alternate alleles
        if 1 < len(alt):
            cur = alt[0] + ', '+ alt[1]
            for i in range(2, len(alt)):
                cur = cur + ', ' + alt[i]
            alt = cur;
        else:
            alt = alt[0]
        qual = fields[5]
        info = fields[7]
        gt = fields[9].split(':')[0]
        if re.search('0',gt):
            genotype = "Het"
        else:
            genotype = "Hom"
        dp = re.search("DP=\d+",info)
        dp4 = re.search("DP4=\d+,\d+,\d+,\d+",info)
        if re.search('INDEL',info):
            variant_type = "indel"
            if len(ref) < len(alt_first):
                variant_type = "Insertion"
            else:
                variant_type = "Deletion"

        else:
            variant_type = "SNP"

        cov_total = dp.group(0).split('=')[1]
        (ref_cov_top,ref_cov_bot,alt_cov_top,alt_cov_bot) = dp4.group(0).split('=')[1].split(',')

        #link = 'http://www.broadinstitute.org/igv/projects/current/igv.php?sessionURL=' + options.igv_session_xml_url + '&locus=%s:%s' % (contig, pos)
        link = str(contig) + ":" + str(pos)

        data = [contig, pos, ref, alt, variant_type, genotype, float(qual), cov_total, ref_cov_top, ref_cov_bot, alt_cov_top, alt_cov_bot, link]
    
        if len(queue) < options.top_x_variants:
            queue = queue + [data]
        else:
            queue[len(queue)-1] = data
        # shift
        for i in range(len(queue)-1, 0, -1):
            if queue[i-1][6] < queue[i][6]:
                swap = queue[i-1]
                queue[i-1] = queue[i]
                queue[i] = swap
            

for i in range(len(queue)):
    sys.stdout.write("\t\t\t\t\t\t<tr><td><a class='igvTable' data-locus=\"%s\">IGV</a></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.1f</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % \
            (queue[i][12], queue[i][0], queue[i][1], queue[i][2], queue[i][3], queue[i][4], queue[i][5], \
            queue[i][6], queue[i][7], queue[i][8], queue[i][9], queue[i][10], queue[i][11]))

vcf.close()
