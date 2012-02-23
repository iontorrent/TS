# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

from json_utils import PluginWrapper, AnalysisDir, ReferenceLibrary
import os
import sys
from alignment_qc import AlignmentQC
import pysam
import samutils

class DiBayes(PluginWrapper):
    
    def __init__(self, pluginJSONDir):
        self.DefaultParams()
        super(DiBayes, self).__init__(pluginJSONDir)
        
    
    def Init(self, libraryName=None):
        #defaults not found in JSON
        self.programName = self.runInfo.PluginDir() + "/diBayes"
        if not libraryName:
            self.referenceLibrary = ReferenceLibrary( self.analysisDir.LibraryName() )
        else:
            self.referenceLibrary = ReferenceLibrary( libraryName )

        self.log_dir = self.OutputDir() + "log/"
        self.temp_dir = self.OutputDir() + "temp/"
        self.out_dir = self.OutputDir() + "dibayes_out/"
        self.bamFile = self.GetAnalysisDir().GetBamFile()
        
        
    def DefaultParams(self):
        self.programName = "diBayes"
        self.call_stringency = "low_frequency"
        self.het_skip_high_coverage = 0
        
        self.reads_min_mapping_qv = 16
        self.het_min_lca_start_pos = 1
        self.het_min_lca_base_qv = 12
        self.het_max_diff_base_qv = 8
        self.het_lca_both_strands = 0
        self.het_min_allele_ratio = 0.05
        self.het_max_coverage_bayesian = 60
        self.het_min_nonref_base_qv = 10
        self.snps_min_base_qv = 18
        self.snps_min_nonref_base_qv = 18
        self.reads_with_indel_exclude = 0        
    
    def SetReference( self, libraryName ):
        self.referenceLibrary = ReferenceLibrary( libraryName )
    
    def SetBAMFile( self, bamFile ):
        self.bamFile = bamFile
    
    def CommandLine(self):
        """
            command line for plugin, if there is one
            ./diBayes --AllSeq=1 -b 1 --platform=2 -c low_frequency --het-skip-high-coverage=0 -f reference.fasta -g log_directory -w temp_dir -o output_dir
            --reads-min-mapping-qv=16 --het-min-lca-start-pos=3 --het-min-lca-base-qv=12 --het-max-diff-base-qv=8 --het-lca-both-strands=0 --het-min-allele-ratio=0.002
            --het-max-coverage-bayesian=60 --het-min-nonref-base-qv=10 --snps-min-base-qv=18 --snps-min-nonref-base-qv=18 --reads-with-indel-exclude=0 -n <Experiment Name>
            <Bam File Location>
        """
        piece1 = "%s --AllSeq=1 -b 1 --platform=2 -c low_frequency -d 1 -C 0 -W 0  --het-skip-high-coverage=%d -f %s -g %s -w %s -o %s" % ( self.programName, int(self.het_skip_high_coverage),
                                                                                                     self.referenceLibrary.FastaPath(), self.log_dir,
                                                                                                     self.temp_dir, self.out_dir, )
        piece2 = "-R /results/plugins/AmpliSeqCancerVariantCaller/bedfiles/HSM_ver12_1_loci.bed --reads-min-mapping-qv=%d --het-min-lca-start-pos=%d --het-min-lca-base-qv=%d --het-max-diff-base-qv=%d --het-lca-both-strands=%d --het-min-allele-ratio=%.3f"% (int(self.reads_min_mapping_qv), int(self.het_min_lca_start_pos), int(self.het_min_lca_base_qv),
                                                                                                                                                                                   int(self.het_max_diff_base_qv), int(self.het_lca_both_strands), float(self.het_min_allele_ratio) )
        piece3 = " --het-max-coverage-bayesian=%d --het-min-nonref-base-qv=%d --snps-min-base-qv=%d --snps-min-nonref-base-qv=%d --reads-with-indel-exclude=%d -n %s %s" % (int(self.het_max_coverage_bayesian), int(self.het_min_nonref_base_qv), int(self.snps_min_base_qv),
                                                                                                                                                                            int(self.snps_min_nonref_base_qv), int(self.reads_with_indel_exclude),
                                                                                                                                                                            self.GetAnalysisDir().ExperimentLog().ExperimentName(), self.bamFile)
        return "%s %s %s" % (piece1, piece2, piece3)
        


if __name__ == '__main__':
    
    
    
    diBayes = DiBayes( sys.argv[1] )
    
    if samutils.GetLibraryFromSam( diBayes.GetAnalysisDir().GetBamFile() ) != "hg19":
        ReAlignReads(diBayes.GetAnalysisDir(), diBayes.GetAnalysisDir().GetBamFile() )
    
    diBayes.Init()
    diBayes.DefaultParams()
    print "commandline: ", diBayes.CommandLine()

    


    
