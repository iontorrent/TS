import os
def touch(fname, times=None):
    '''touch a file (fname)'''
    with file(fname, 'a'):
        os.utime(fname, times)

def toBoolean(val, default=True):
    """convert strings from CSV to Python bool
    if they have an empty string - default to true unless specified otherwise
    """
    if default:
        trueItems = ["true", "t", "yes", "y", "1", "" ]
        falseItems = ["false", "f", "no", "n", "none", "0" ]
    else:
        trueItems = ["true", "t", "yes", "y", "1", "on"]
        falseItems = ["false", "f", "no", "n", "none", "0", "" ]

    if str(val).strip().lower() in trueItems:
        return True
    if str(val).strip().lower() in falseItems:
        return False