#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Ion Plugin Hello World

from ion.plugin import *

class Plugin_02_HelloWorld(IonPlugin):
    """  Demo Plugin to show the new plugin definition and features """
    @classmethod
    def version(cls):
        return "1.0." + filter(str.isdigit, "$Revision: 39102$")

    features = [ Feature.EXPORT, ]
    runtypes = [ RunType.THUMB, RunType.FULLCHIP]
    runlevel = [ RunLevel.BLOCK, RunLevel.DEFAULT ]

    def initialize(self, **kwargs):
        self.blockcount = 0

    def block(self, block):
        self.blockcount += 1
        return True

    def launch(self, data=None):
        return True

    ## These you probably do not need to override
    def report(self):
        output = {
            'sections': {
                'title': 'Hello, World!',
                'type': 'html',
                'content': '<p>Welcome to Ion Torrent Plugin Framework 3.0</p><br/><p>There were %s blocks seen.</p>' % self.blockcount,
            }
        }
        return output

    def metrics(self):
        """ Write result.json metrics """
        return { 'blocks': self.blockcount }

    def getUserInput(self):
        return {
            'columns': [
                {"Name": "Workflow", "Order" : "1", "Type" : "list", "ValueType" : "String", "Values" : ['A', 'B', 'C']},
            ],
            'restrictionRules': [],
        }

# Devel use - running directly
if __name__ == "__main__": PluginCLI()

