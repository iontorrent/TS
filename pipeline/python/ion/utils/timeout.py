# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# From: http://pguides.net/python-tutorial/python-timeout-a-function/
"""Decorator function to provide a timeout for a function"""
import sys
import signal


class TimeoutException(Exception):
    pass


def timeout(timeout_time, default):
    """timeout_time in seconds.  default is return value when timeout occurs"""

    def timeout_function(f):
        def f2(*args):
            def timeout_handler(signum, frame):
                raise TimeoutException()

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_time)  # triger alarm in timeout_time seconds
            try:
                retval = f(*args)
            except TimeoutException:
                return default
            finally:
                signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)
            return retval

        return f2

    return timeout_function
