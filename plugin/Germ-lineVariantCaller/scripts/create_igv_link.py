#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser

# Notes: see http://www.broadinstitute.org/software/igv/ControlIGV 
# for more details

def check_option(parser, value, name):
    if None == value:
        print 'Option ' + name + ' required.\n'
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-r', '--results-dir', help='the results directory', dest='results_dir') 
    parser.add_option('-b', '--bam-file', help='the BAM file name (no path)', dest='bam_file')
    parser.add_option('-v', '--vcf-file', help='the VCF file name (no path)', dest='vcf_file')
    parser.add_option('-g', '--genome-name', help='the genome name to be loaded by IGV', dest='genome_name')
    parser.add_option('-l', '--locus', help='the locus name to be initially loaded (ex. "chr1:1-1000"', dest='locus')
    parser.add_option('-s', '--session-xml-name', help='the session XML name (no path)', dest='session_xml_name')
    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        exit(1)
    check_option(parser, options.results_dir, '-r')
    check_option(parser, options.bam_file, '-b')
    check_option(parser, options.vcf_file, '-v')
    check_option(parser, options.genome_name, '-g')
    check_option(parser, options.locus, '-l')
    
    # The paths to the resources should be accessible via http from outside the torrent srever
    bam_url = "{url_root}" + '/' + options.bam_file
    vcf_url = "{plugin_url}" + '/' + options.vcf_file
    session_url = "{plugin_url}"  + '/' + options.session_xml_name

    # Write the session XML file
    fn_xml = options.results_dir + '/' + options.session_xml_name
    fxml = open(fn_xml, "w")
    fxml.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
    fxml.write('<Global genome="' + options.genome_name + '" locus="' + options.locus + '" version="3">\n')
    fxml.write('\t<Resources>\n')
    fxml.write('\t\t<Resource name="'+ options.vcf_file + '" path="' + vcf_url + '"/>\n')
    fxml.write('\t\t<Resource name="'+ options.bam_file + '" path="' + bam_url + '"/>\n')
    fxml.write('\t</Resources>\n')
    fxml.write('\t<Panel name="DataPanel" height="100">\n')
    fxml.write('\t\t<Track displayMode="EXPANDED" id="' + vcf_url + '" name="Variants" visible="true"/>\n')
    fxml.write('\t</Panel>\n')
    fxml.write('\t<Panel height="525">\n')
    fxml.write('\t\t<Track displayMode="COLLAPSED" id="' + bam_url + '_coverage" name="Coverage" visible="true"/>\n')
    fxml.write('\t\t<Track displayMode="EXPANDED" id="' + bam_url + '" name="Alignments" visible="true"/>\n')
    fxml.write('\t</Panel>\n')
    fxml.write('\t<Panel name="FeaturePanel" height="75">\n')
    fxml.write('\t\t<Track displayMode="COLLAPSED" id="Reference sequence" name="Reference sequence" visible="true"/>')
    fxml.write('\t</Panel>\n')
    fxml.write('\t<PanelLayout dividerFractions="0.15,0.75"/>\n')
    fxml.write('</Global>\n')
    fxml.close()
