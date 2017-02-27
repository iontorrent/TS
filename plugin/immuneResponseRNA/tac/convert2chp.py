# pylint: disable=line-too-long
""" pileup.py """
import sys
import os
import uuid
from run import Run

class Tac(Run):
    """ Pileup """
    def add_options(self):
        """ Define options """
        self.add_option("-i", "--input-file", "string", "Input file")
        self.add_option("-o", "--output-dir", "string", "Directory for storing output from the run")
        self.add_option("-m", "--method", "string", "Normalization methond")

    def override_options(self):
        """ Override json parameter values with command line arguments """
        self.parameters['input_file'] = self.options.input_file
        self.parameters['output_dir'] = self.options.output_dir
        self.parameters['method'] = self.options.method

    def validate_options(self):
        """ Parameter validation """
        if (self.options.input_file == None):
            self.fatal_error("Please specify an input-file.")
        if (self.options.output_dir == None):
            self.fatal_error("Please specify an output-dir.")
        if (self.options.method == None):
            self.options.method = 'RPM'

    def process(self):
        """ Process """
        tac_script_path = os.path.dirname(os.path.realpath(__file__))
        chp_bin = os.path.join(tac_script_path, 'apt2-dset-util')
        try:
            os.mkdir(self.parameters['output_dir'])
        except:
            pass
        headers = []
        data = []
        try:
            fin = open(self.parameters['input_file'], 'r')
        except IOError:
            self.fatal_error("Cannot open input file:\t" + self.parameters['input_file'])
        for line in fin:
            line = line.rstrip()
            if line.startswith("#"):
                continue
            if line.startswith("Target\t") or line.startswith("\"Target\"\t"):
                headers = line.split("\t")
                i = 0
                for header in headers:
                    if header.startswith('"') and header.endswith('"'):
                        headers[i] = header[1:-1]
                    i += 1
                continue
            cols = line.split("\t")
            if cols[0].startswith('"') and cols[0].endswith('"'):
                cols[0] = cols[0][1:-1]
            data.append(cols)
        fin.close()
        
        index = 0
        for header in headers:
            if index > 0:
                try:
                    filename = self.parameters["output_dir"] + "/" + header
                    print(filename)
                    fout = open(filename, 'w')
                except IOError:
                    self.fatal_error("Cannot open output file:\t" + filename)
                fout.write("#%%BEGIN-FILE=/\n")
                fout.write("#%gdh:0:data_source=affymetrix-quantification-analysis\n")
                fout.write("#%gdh:0:uuid=" + str(uuid.uuid1()) + "\n")
                fout.write("#%gdh:0:locale=\n")
                fout.write("#%gdh:0:datetime=en-US\n")
                fout.write("#%gdh:0:affymetrix-algorithm-name=" + self.parameters['method'] + "\n")
                fout.write("#%gdh:0:affymetrix-algorithm-version=1.0\n")
                fout.write("#%gdh:0:affymetrix-array-type=Immune-response\n")
                fout.write("#%gdh:0:program-name=ImmuneResponse_plugin\n")
                fout.write("#%gdh:0:program-version=v1.0\n")
                fout.write("#%gdh:0:program-company=ThermoFisherScientific\n")
                fout.write("#%gdh:0:affymetrix-algorithm-param-exec-guid=\n")
                fout.write('#%gdh:0:affymetrix-algorithm-param-quantification-name=' + self.parameters['method'] + "\n")
                fout.write('#%gdh:0:affymetrix-algorithm-param-quantification-version="1.0"\n')
                fout.write('#%gdh:0:affymetrix-algorithm-param-quantification-scale=log2\n')
                fout.write('#%gdh:0:affymetrix-algorithm-param-quantification-type=scaled-RPM\n')
                fout.write("#%%BEGIN-GROUP=/Quantification\n")
                fout.write("#%%BEGIN-DATASET=/Quantification/Quantification\n")
                fout.write("#\n")
                fout.write("#%%field-000=ProbeSetName_&size,int32\n")
                fout.write("#%%field-001=ProbeSetName,string8,17\n")
                fout.write("#%%field-002=Quantification,float32,-1\n")
                fout.write("#\n")
                fout.write("#%%dims=0:\n")
                fout.write("#\n")
                fout.write("#%%row-cnt=" + str(len(data)) + "\n")
                fout.write("#\n")
                fout.write("ProbeSetName_&size	ProbeSetName	Quantification\n")
                for cols in data:
                    fout.write(str(len(cols[0])) + "\t" + cols[0] + "\t" + cols[index] + "\n")
                fout.close()
                cmd = chp_bin + " "
                cmd += "-i " + filename + " "
                cmd += "-o " + filename + ".gene.chp "
                cmd += "-log-file " + filename + ".log"
                self.run_command(cmd, "apt2-dset-util")
                os.remove(filename)
                os.remove(filename + ".log")
            index += 1

if __name__ == '__main__':
    TAC = Tac("1.0", sys.argv[1:])
