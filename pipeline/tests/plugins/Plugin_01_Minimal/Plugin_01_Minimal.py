#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.plugin import IonPlugin, PluginCLI

class Plugin_01_Minimal(IonPlugin):
    def launch(self, data=None):
        self.log.info("Hello, Ion World!")
        return True

if __name__ == '__main__': PluginCLI()
