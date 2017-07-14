#!/usr/bin/env python
import json
import os
import subprocess
import sys

def fileExistsAndNonEmpty(filename):
    if not os.path.exists(filename):
        return False
    return os.stat(filename).st_size > 0

class AssemblerRunner(object):
    def __init__(self, sample_id, sample_seq, bam_file):
        with open("startplugin.json", "r") as fh:
            self.config = json.load(fh)
            self.params = self.config['pluginconfig']

        # launch.sh creates a symlink to the input BAM file in this directory
        self.output_dir = self.config['runinfo']['results_dir']
        self.sample_id = sample_id
        self.sample_seq = sample_seq
        self.sample_name = sample_id + "." + sample_seq
        self.sample_output_dir = os.path.join(self.output_dir, self.sample_name)
        self.bam_file = bam_file
        self.bam_rel_path = os.path.join(self.sample_name, self.bam_file)

        # relative path to the input bam file
        self.bam_to_assemble = os.path.join(self.output_dir, self.bam_rel_path)

        # how much to downsample (the step is skipped if it equals to 1)
        if self.params.has_key('fraction_of_reads'):
            self.fraction_of_reads = float(self.params['fraction_of_reads'])

        # all executables are located in bin/ subdirectory
        self.assembler_path = os.path.join(os.environ['DIRNAME'], 'bin')

        # where to output HTML with results
        self.url_root = self.config['runinfo']['url_root']

        # skip assembly (and run only QUAST) if contigs exist
        self.quast_only = self.params.has_key('quastOnly')

        # information will be printed to "info.json"
        self.info = { 'params' : self.params, 'executedCommands' : [] }
        if sample_id != '' and sample_seq != '':
            self.info['sampleId'] = sample_id
            self.info['sampleSeq'] = sample_seq
            self.info['sampleName'] = self.sample_name

    # Prints 'pluginconfig' section of 'startplugin.json'
    def printAssemblyParameters(self):
        print("AssemblerSPAdes run parameters:")
        print(self.params)

    def writeInfo(self, json_filename):
        with open(json_filename, 'w+') as f:
            json.dump(self.info, f, indent=4)

    def runCommand(self, command, description=None):
        if description:
            print(description)
        else:
            print(command)
        sys.stdout.flush()
        os.system(command)
        self.info['executedCommands'].append(command)

    def runDownsampling(self):
        print("\nSubsampling using Picard")
        # downsampler = os.path.join(self.assembler_path, 'DownsampleSam.jar')
        downsampler = "/opt/picard/picard-tools-current/picard.jar"
        out = os.path.join(self.sample_output_dir, self.bam_file + "_scaled")

        cmd = ("java -Xmx2g -jar {downsampler} "
               "DownsampleSam "
               "INPUT={self.bam_to_assemble} OUTPUT={out} "
               "PROBABILITY={self.fraction_of_reads}").format(**locals())
        self.runCommand(cmd)

        cmd = ("mv {out} {self.bam_to_assemble}").format(**locals())
        self.runCommand(cmd)

    def execute(self):
        self.printAssemblyParameters()
        read_count_cmd = "samtools view -c " + self.bam_rel_path
        read_count_process = subprocess.Popen(read_count_cmd, shell=True,
                                              stdout=subprocess.PIPE)
        num_reads = int(read_count_process.communicate()[0])

        def tooFewReads():
            if not self.params.has_key('min_reads'):
                return False
            self.min_reads = int(self.params['min_reads'])
            return num_reads <= self.min_reads

        print("%d reads in %s" % (num_reads, self.bam_file))
        if tooFewReads():
            print(("\tDoes not have more than %d reads. "
                   "Skipping this file") % (self.min_reads,))
            return

        if self.fraction_of_reads < 1:
            self.runDownsampling()

#        if self.params.has_key('runSpades'):
        self.runSPAdes()

    def runSPAdes(self):
        
        if self.params.has_key('spadesversion'):
            version = self.params['spadesversion']
        else:
            version = "3.1.0"
                
        assert(version >= "3.0.0")

        rel_path = os.path.join("SPAdes-%s-Linux" % version, "bin", "spades.py")
        spades_path = os.path.join(self.assembler_path, rel_path)

        output_dir = os.path.join(self.sample_name, "spades")
        contigs_fn = os.path.join(output_dir, "contigs.fasta")
        scaffolds_fn = os.path.join(output_dir, "scaffolds.fasta")
        log_fn = os.path.join(output_dir, "spades.log")
        skip_assembly = self.quast_only and fileExistsAndNonEmpty(contigs_fn)
        if self.params.has_key('spadesOptions'):
             user_options = self.params['spadesOptions']
        else:
             user_options = "-k 21,33,55,77,99"     

        spades_info = {'contigs' : contigs_fn,
                       'scaffolds' : scaffolds_fn,
                       'log' : log_fn,
                       'userOptions' : user_options,
                       'version' : version }

        pid = os.getpid()
        if not skip_assembly:
            cmd = ("{spades_path} --iontorrent --tmp-dir /tmp/{pid} "
                   "-s {self.bam_to_assemble} -o {output_dir} "
                   "{user_options} > /dev/null").format(**locals())
            print("Running AssemblerSPAdes - SPAdes %s" % version)
            self.runCommand(cmd)

        report_dir = self.createQuastReport(contigs_fn, output_dir)
        spades_info['quastReportDir'] = report_dir
        self.info['spades'] = spades_info

    def createQuastReport(self, contigs_fn, output_dir):
        version = "2.3"
        rel_path = os.path.join("quast-%s" % version, "quast.py")
        quast_path = os.path.join(self.assembler_path, rel_path)

#       quast_reference = self.params['bgenome']
        quast_reference = "None"
        quast_results_dir = os.path.join(output_dir, "quast_results")

        print("Running QUAST %s" % version)
        reference_param = ("-R " + quast_reference) if quast_reference!="None" else " "
        cmd = ("{quast_path} -o {quast_results_dir} "
               "{reference_param} {contigs_fn}").format(**locals())
        self.runCommand(cmd)

        try:
            if os.path.isfile(os.path.join(quast_results_dir, "report.html")):
                return os.path.abspath(quast_results_dir)
            else:
                return None
        except:
            return None

import sys
if __name__ == "__main__":
    if len(sys.argv) == 4:
        sample_id = sys.argv[1]
        sample_seq = sys.argv[2]
        bam_file = sys.argv[3]
        runner = AssemblerRunner(sample_id, sample_seq, bam_file)
        runner.execute()
        runner.writeInfo("info_%s.%s.json" % (sample_id, sample_seq))
    else:
        assert(len(sys.argv) == 2) # not a barcode run
        bam_file = sys.argv[1]

        # HACK: sample_name = '.' => essentially vanishes from all paths
        runner = AssemblerRunner('', '', bam_file)
        runner.execute()
        runner.writeInfo("info.json")
