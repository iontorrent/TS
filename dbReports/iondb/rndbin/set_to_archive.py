#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

# Purpose: marks given experiments to be Archived.

import sys
import argparse
import traceback

from iondb.bin.djangoinit import *
from iondb.rundb import models
from iondb.rundb.data import dmactions_types

def main(experiment_list,archive,export):
    '''list of experiment names'''
    for name in experiment_list:
        try:
            print "Experiment: %s" % name

            # Get Experiment object with matching name; should be only one object
            exp = models.Experiment.objects.get(expName=name)

            # Get first Results object
            result = exp.results_set.all()[0]
            print "Result: %s" % result.resultsName

            # Get first DMFileStat object of type 'Signal Processing Input'
            dmfilestat = result.get_filestat(dmactions_types.SIG)
            print "DMFileStat: %s" % dmfilestat.action_state

            if export:
                # Set action_state to 'SE'
                dmfilestat.setactionstate('SE')
                print "    Set to: Export"
            else:
                # Set action_state to 'SA'
                dmfilestat.setactionstate('SA')
                print "    Set to: Archive"
        except:
            print traceback.format_exc()
        finally:
            print ""

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method to set a batch of runs to be Archived.  Make sure media is setup prior to use')
    parser.add_argument('experiment',nargs='*',help="one or more experiment names")
    parser.add_argument('--archive',action='store_true',default=True,help='Default.  Sets Archive Pending flag')
    parser.add_argument('--export',action='store_true',default=False,help='Sets Export Pending flag (instead of archive)')
    args = vars(parser.parse_args())
    if args['export']:
        export = True
        archive = False
    else:
        export = False
        archive = True

    sys.exit(main(args['experiment'],archive,export))
