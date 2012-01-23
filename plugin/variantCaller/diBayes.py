# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys

class DiBayes():
    
    def __init__(self):
        self.DefaultParams()
    
    def Init(self, reference, bamFile, bedFile, expName, outDir='.', runDir='.' ):
        self.programName = runDir + "/diBayes"
        self.log_dir = outDir + "/log/"
        self.temp_dir = outDir + "/temp/"
        self.out_dir = outDir + "/dibayes_out/"
        self.bamFile = bamFile
        self.bedFile = bedFile
        self.reference = reference
        self.experimentName = expName
        
    def DefaultParams(self):
        self.programName = "diBayes"
        self.call_stringency = "medium"
        self.het_skip_high_coverage = 0        
        self.reads_min_mapping_qv = 2
        self.het_min_lca_start_pos = 0
        self.het_min_lca_base_qv = 14
        self.het_lca_both_strands = 0
        self.het_min_allele_ratio = 0.15
        self.het_max_coverage_bayesian = 60
        self.het_min_nonref_base_qv = 14
        self.snps_min_base_qv = 14
        self.snps_min_nonref_base_qv = 14
        self.reads_with_indel_exclude = 0   
        self.het_min_coverage = 2 
        self.het_min_start_pos = 1 
        self.hom_min_coverage = 1 
        self.hom_min_nonref_allele_count = 3
        self.snps_min_filteredreads_rawreads_ratio = 0.15
        self.het_min_validreads_totalreads_ratio = 0.65
        self.reads_min_alignlength_readlength_ratio = 0.2 
        self.hom_min_nonref_base_qv = 14 
        self.hom_min_nonref_start_pos = 0      
    
    def SetReference( self, reference ):
        self.reference = reference
    
    def SetBAMFile( self, bamFile ):
        self.bamFile = bamFile
    
    def SetBEDFile( self, bedFile ):
        self.bedFile = bedFile
    
    def CommandLine(self):
        """
            command line for plugin, if there is one
            ./diBayes --AllSeq=1 -b 1 --platform=2 -c low_frequency --het-skip-high-coverage=0 -f reference.fasta -g log_directory -w temp_dir -o output_dir
            --reads-min-mapping-qv=16 --het-min-lca-start-pos=3 --het-min-lca-base-qv=12 --het-max-diff-base-qv=8 --het-lca-both-strands=0 --het-min-allele-ratio=0.002
            --het-max-coverage-bayesian=60 --het-min-nonref-base-qv=10 --snps-min-base-qv=18 --snps-min-nonref-base-qv=18 --reads-with-indel-exclude=0 -n <Experiment Name>
            <Bam File Location>
        """
        piece1 = "%s --AllSeq=1 -b 1 --platform=2 -c %s -d 1 -C 0 -W 0  --het-skip-high-coverage=%d -f %s -g %s -w %s -o %s" % (
             self.programName, self.call_stringency, int(self.het_skip_high_coverage), self.reference, self.log_dir, self.temp_dir, self.out_dir, )
        if( self.bedFile != "" ):
            piece1 += " -R " + self.bedFile;
        piece2 = " --reads-min-mapping-qv=%d --het-min-lca-start-pos=%d --het-min-lca-base-qv=%d --het-lca-both-strands=%d --het-min-allele-ratio=%.3f" % (
            int(self.reads_min_mapping_qv), int(self.het_min_lca_start_pos), int(self.het_min_lca_base_qv),
            int(self.het_lca_both_strands), float(self.het_min_allele_ratio) )
        piece3 = " --het-max-coverage-bayesian=%d --het-min-nonref-base-qv=%d --snps-min-base-qv=%d --snps-min-nonref-base-qv=%d --reads-with-indel-exclude=%d " % (
            int(self.het_max_coverage_bayesian), int(self.het_min_nonref_base_qv), int(self.snps_min_base_qv),
            int(self.snps_min_nonref_base_qv), int(self.reads_with_indel_exclude) )
        piece4 = " --het-min-coverage=%d --het-min-start-pos=%d --hom-min-coverage=%d --hom-min-nonref-allele-count=%d --snps-min-filteredreads-rawreads-ratio=%.3f --het-min-validreads-totalreads-ratio=%.3f --reads-min-alignlength-readlength-ratio=%.3f --hom-min-nonref-base-qv=%d --hom-min-nonref-start-pos=%d -n %s %s " % (
            int(self.het_min_coverage), int(self.het_min_start_pos), int(self.hom_min_coverage), int(self.hom_min_nonref_allele_count), 
            float(self.snps_min_filteredreads_rawreads_ratio), float(self.het_min_validreads_totalreads_ratio), float(self.reads_min_alignlength_readlength_ratio),
            int(self.hom_min_nonref_base_qv), int(self.hom_min_nonref_start_pos), self.experimentName, self.bamFile )
        return "%s %s %s %s" % (piece1, piece2, piece3, piece4)
        
if __name__ == '__main__':
    diBayes = DiBayes()
    diBayes.Init( sys.argv[1] )
    print "commandline: ", diBayes.CommandLine()

