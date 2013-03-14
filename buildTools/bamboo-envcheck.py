#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys
import os.path
import subprocess
from subprocess import call


if __name__ == "__main__":
    print " Verifying test environment..."
    try:
        import xmlrunner
        print "    xml test output will be generated"
    except ImportError:
        print "    python unittest package not found, no xml test output will be generated"
        return_code = call("sudo easy_install unittest-xml-reporting", shell=True)

