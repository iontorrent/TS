''' 
This is purly a hook to attempt to have a version of deinerlace in this folder.
We should not maintain the compiled deinterlace.so object in the git repo.

As part of any build process, this file should removed and replaced with the compiled object
'''
import sys, os
_pathlen = len(sys.path)
_moduleDir = os.path.abspath( os.path.dirname( __file__ ) )

# Add as much as you want here
sys.path.append( '/software/testing' )
sys.path.append( '/software/p2' )
sys.path.append('/home/scott/python/deinterlace')
sys.path.append('/home/brennan/0_repos/pyCextensions/')
sys.path.append('/home/brennan/repos/pyCextensions/')
sys.path.append('/rnd/brennan/pyCextensions/')
sys.path.append('%s/pydeint' % _moduleDir)

try:
    if sys.version_info[0] == 3:
        import deinterlace3 as deinterlace
    else:
        import deinterlace
finally:
    # restore the path
    sys.path = sys.path[:_pathlen]
    # clean up variables
    del _pathlen
    del _moduleDir
    del sys, os

