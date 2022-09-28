#!/usr/bin/python

import os
import sys
from subprocess import *
from ion.plugin import *

class GBU_HBU_Analysis(IonPlugin):
    '''Calculates Gene-Based Uniformity (GBU), Hotspot-Based Uniformity (HBU), and Panel-Based Uniformity (PBU) for panels analyzed by coverageAnalysis plugin.'''
    version = '5.10.0.0'
    major_block = False
    runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
    runlevels = [ RunLevel.DEFAULT ]
    depends = ['coverageAnalysis']
    
    def launch(self):
        plugin = Popen([
            '%s/GBU_HBU_Analysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version,
            'startplugin.json', 'barcodes.json' ], stdout=PIPE, shell=False )
        plugin.communicate()
        sys.exit(plugin.poll())

if __name__ == "__main__":
    PluginCLI()

