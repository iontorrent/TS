# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django_nose import NoseTestSuiteRunner
from djcelery.contrib.test_runner import CeleryTestSuiteRunner

USAGE = """\
Custom test runner to allow testing of celery delayed tasks.
But inherit from django_nose NoseTestSuiteRunner, but run celery inline.
"""


class IonTestSuiteRunner(CeleryTestSuiteRunner, NoseTestSuiteRunner):
    """ Use Both CeleryTestSuiteRunner, which forces celery to run inline,
        And the NoseTestSuiteRunner
    """
    pass
