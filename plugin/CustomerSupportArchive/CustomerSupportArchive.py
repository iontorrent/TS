#!/usr/bin/python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import textwrap
import shutil
import subprocess
import json
import re
import fnmatch
import zipfile
import shlex
import tarfile

from subprocess import *
from ion.plugin import *
import makeCSA

class CustomerSupportArchive(IonPlugin):
    """Generate an enhanced FSA"""
    version = "0.4.4"
    allow_autorun = True # if true, no additional user input
    runtypes = [ RunType.FULLCHIP]
    depends = [] 

    # the rndplugins run with the FSA
    plugin_options = {
        # # TODO, bubblePlots on fullchip
        # # "bubblePlots": {
        # #     "files": [
        # #         "*outlier-by-flow.png",
        # #         "*outlier-all.png",
        # #         "*.csv",
        # #         "*.gif",
        # #         "*.html",
        # #         "*.log"
        # #     ]
        # # },
        # tested
        "rawTrace": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # # TODO, rawplots on fullchip
        # # "rawPlots": {
        # #     "files":[
        # #         "*png",
        # #         "*log",
        # #         "*html"
        # #     ]
        # # },
        # # TODO, separator on fullchip
        # # "separator": {
        # #     "files":[
        # #         "*png",
        # #         "*log",
        # #         "*html"
        # #     ]
        # # },
        # # # tested, separator_spatial on fullchip
        # # "separator_spatial": {
        # #     "files":[
        # #         "*png",
        # #         "*log",
        # #         "*html"
        # #     ]
        # # },
        # Tested
        "flowErr": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # tested
        "flowRate": {
            "files":[
                "*png",
                "*log",
                "*html",
                "*csv"
            ]
        },
        # tested
        "True_loading": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # tested
        "Lane_Diagnostics": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # tested
        "autoCal": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # tested
        "chipDiagnostics": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        #tested
        "ValkyrieWorkflow": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # tested
        "libPrepLog": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
        # tested
        "NucStepSpatialV2": {
            "files":[
                "*png",
                "*log",
                "*html"
            ]
        },
    }


    def launch(self):
        """ main """
        version = "0.4.4"
        print "Running the FieldSupport plugin."
        print "Reading startpluginjson"
        start_json = getattr(self, 'startpluginjson', None)
        if not start_json:
            try:
                with open('startplugin.json', 'r') as fh:
                    start_json = json.load(fh)
            except:
                self.log.error("Error reading start plugin json")


        print "Reading barcodesjson"
        barcodes_json = getattr(self, 'barcodesjson', None)
        if not barcodes_json:
            try:
                with open('barcodes.json', 'r') as fh:
                    barcodes_json = json.load(fh)
            except:
                self.log.error("Error reading barcodes json")

        # sampleNames used to remove redundant files from sample logs
        sampleNames = []
        sampleNamePattern = []
        if barcodes_json:
            try:
                sampleNames = self.parseSampleNames(barcodes_json)
                for sn in sampleNames:
                    sampleNamePattern.append(sn + '/outputs/SigProcActor-00/')
                    sampleNamePattern.append(sn + '/outputs/BaseCallingActor-00/')
            except:
                pass
        else:
            pass

        sampleResultsDirs = []
        if start_json:
            try: 
                for sample, sampledict in start_json["plan"]["barcodedSamples"].iteritems():
                    sampleResultsDirs.append(sampledict["sampleResultsDir"])
            except:
                pass


        # Load directory information        
        self.results_dir = start_json["runinfo"]["results_dir"]
        self.raw_data_dir = start_json["runinfo"]["raw_data_dir"]
        self.analysis_dir = start_json["runinfo"]["analysis_dir"] 
        self.sigproc_dir = start_json["runinfo"]["sigproc_dir"]
        self.plugin_dir = start_json["runinfo"]["plugin_dir"]

        # Load run information
        self.runType = start_json["runplugin"]["run_type"]
        self.chipType = start_json["expmeta"]["chiptype"]
        self.analysis_name = start_json["expmeta"]["results_name"]
        self.nflow = start_json["expmeta"]["run_flows"]
        self.flowOrder = start_json["expmeta"]["flowOrder"]
        self.library_key = start_json["runinfo"]["library_key"]
        self.efuse = start_json['expmeta']['chipBarcode' ]
        self.runid = start_json['expmeta']['runid' ]

        zip_name = start_json["expmeta"]["results_name"]+ '.' + os.path.basename(self.analysis_dir) + ".CSA.tar"
        zip_path = os.path.join(self.results_dir, zip_name)

        if os.path.isfile(zip_path):
            cmd = 'mv '+ zip_path + ' ' + os.path.join(os.path.dirname(zip_path), 'backup.tar') 
            print 'backing up previous tar file - ' + cmd
            os.system(cmd)

        # Make CSA zip using plugin makeCSA
        print "making zip file : " + zip_name
        makeCSA.makeCSA(
            self.analysis_dir,
            start_json["runinfo"]["raw_data_dir"],
            sampleResultsDirs,
            zip_path,
            sampleNamePattern
        )


        # Run rndplugin
        for name, options in self.plugin_options.items():
            try:
                self.run_rndplugin(name)
            except:
                pass

        # Modify zip archive to include rndplugin files
        with tarfile.TarFile.open(zip_path, dereference=True, mode='a') as f:
            # Add rndplugin files
            for name, options in self.plugin_options.items():
                if name != 'Lane_Diagnostics':
                    f.add(os.path.join(self.results_dir, name), arcname = name, recursive=True)
                else:
                    for root, _, file_names in os.walk(os.path.join(self.results_dir, name)):
                        for pattern in options["files"]:
                            for file_name in fnmatch.filter(file_names, pattern):
                                f.add(os.path.join(root, file_name), os.path.join(name, file_name))
        try:
            with tarfile.TarFile.open(zip_path, dereference=True, mode='a') as f:
                f.add(os.path.join(self.results_dir, "CustomerSupportArchive_" + version + ".log"))
        except:
            print "cannot include CSA log - " + os.path.join(self.results_dir, "CustomerSupportArchive_" + version + ".log")
            pass
        
        # cmd = 'XZ_OPT="-9e" tar cJf '+ zip_path[:-4] + '.tar.xz -C ' + os.path.dirname(zip_path) + ' ' + os.path.basename(zip_path)
        cmd = 'xz -9vf '+ zip_path
        print 'compressing tar file - ' + cmd
        os.system(cmd)
        cmd = 'mv ' + zip_path + '.xz ' + zip_path[:-4] + '.txz'
        try:
            os.system(cmd)
        except:
            print "failed at moving tar file to txz"
            sys.exit(1)
       
        print "Done"
        sys.exit(0)

    def run_rndplugin(self, plugin_name):
        print "Running plugin " + plugin_name
        plugin_dir = os.path.join(self.plugin_dir, plugin_name)
        print "plugin_dir " + plugin_dir
        output_dir = os.path.join(self.results_dir, plugin_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        env = {
            "DIRNAME": plugin_dir,
            "SIGPROC_DIR": self.sigproc_dir,
            "ANALYSIS_DIR": self.analysis_dir,
            "TSP_CHIPTYPE": self.chipType,
            "RAW_DATA_DIR": self.raw_data_dir,
            "TSP_ANALYSIS_NAME": self.analysis_name,
            "TSP_FILEPATH_PLUGIN_DIR": output_dir,
            "TSP_NUM_FLOWS": str(self.nflow),
            "TSP_FLOWORDER": self.flowOrder,
            "TSP_LIBRARY_KEY": self.library_key,
            "TSP_RUNTYPE": self.runType,
            "TSP_CHIPBARCODE": self.efuse.encode('utf-8'),
            "TSP_RUNID": self.runid,
            "explog_path": os.path.join(self.analysis_dir, 'explog.txt'),
            "TSP_LIMIT_OUTPUT": "1",  # Tells plugins they are being run by FieldSupport instead of the pipeline
            "CSA": "True"
        }
        
    
        # some plugins need barcodes.json
        if plugin_name in  ('flowErr', 'Lane_Diagnostics', 'chipDiagnostics', 'ValkyrieWorkflow', 'libPrepLog', 'autoCal', 'NucStepSpatialV2'):
            print "coping barcodes.json" # and startplugin.json"
            p = Popen(["cp", os.path.join(self.results_dir, "barcodes.json"), output_dir])
            output = p.communicate()[0]
            p = Popen(["cp", os.path.join(self.results_dir, "startplugin.json"), output_dir])
            output = p.communicate()[0]

        # call plugins:
        if plugin_name in ('True_loading', 'flowRate', 'Lane_Diagnostics', 'chipDiagnostics', 'ValkyrieWorkflow', 'libPrepLog', 'autoCal', 'NucStepSpatialV2'):  
            cmd = ["python", "-u", plugin_dir + "/" + plugin_name + ".py", "-vv"]
            p = Popen(cmd, stdout = subprocess.PIPE, cwd = plugin_dir, env = env)
            while True:
                nextline = p.stdout.readline()
                if nextline == '' and p.poll() is not None:
                    break
                sys.stdout.write(nextline)
                sys.stdout.flush()
            output = p.communicate()[0]
            if p.returncode != 0:
                print("launch failed %s" %output)
            
        else:
            p = Popen(["bash", "launch.sh"], stdout = subprocess.PIPE, cwd = plugin_dir, env = env)
            while True:
                nextline = p.stdout.readline()
                if nextline == '' and p.poll() is not None:
                    break
                sys.stdout.write(nextline)
                sys.stdout.flush()
            output = p.communicate()[0]
            if p.returncode != 0:
                print("launch failed %s" %output)
        print plugin_name + ' execution completed'
        
    def parseSampleNames(self, barcodesjson):
        sampleN = set()
        for barcode, item  in barcodesjson.items():
            sample_name = item["bam_filepath"].split('/')[-4]
            if sample_name not in sampleN:
                sampleN.add(sample_name)
        return list(sampleN)


if __name__ == "__main__":
    PluginCLI()