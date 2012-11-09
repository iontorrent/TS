#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys
import os.path


# All of this is a little too fancy, but it's function is to populate PYTHONPATH
# with the correct paths for reaching various dependencies irrespective of
# the path from which these tests are run.
test_dir = os.path.abspath(os.path.dirname(__file__))

def is_relative(path):
    return os.path.normpath(os.path.join(test_dir, path))

sys.path.extend(map(is_relative, [
    "../bin",
    "../python/",
]))


class Pipeline(unittest.TestCase):

    def test_get_versions(self):
        from ion.utils.TSversion import findVersions
        self.failIf(findVersions() is None)


if __name__ == "__main__":
    try:
        # Write JUnit compatible XML test reports if possible
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,
                                              output='test-reports')
    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)

