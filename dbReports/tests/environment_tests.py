#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys

class Django(unittest.TestCase):

    def test_django_init(self):
        """djangoinit performs some basic django settings import tasks, such as
        disabling logging, which are needed by non-web django code.
        It uses a form of lazy evaluation and needs to be proactively
        inspected in order to affirm that it has loaded some actual settings.
        """
        from iondb.bin import djangoinit
        self.failIf(djangoinit.settings.DATABASES is None)


if __name__ == "__main__":
    try:
        # Write JUnit compatible XML test reports if possible
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,
                                              output='test-reports')
    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)
