# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
import argparse
import ConfigParser


def main_merge(stats_list, stats_file, verbose):

    process_parameter_file = 'processParameters.txt'

    # Output file
    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')

    for i,maskstats in enumerate(stats_list):
 
        if verbose:
            print "Reading", maskstats

        config = ConfigParser.RawConfigParser()
        config.read(maskstats)

        keys = ['Excluded Wells',
                'Empty Wells',
                'Pinned Wells',
                'Ignored Wells',
                'Bead Wells',
                'Dud Beads',
                'Reference Beads',
                'Live Beads',
                'Test Fragment Beads',
                'Library Beads',
                'TF Filtered Beads (read too short)',
                'TF Filtered Beads (fail keypass)',
                'TF Filtered Beads (too many positive flows)',
                'TF Filtered Beads (poor signal fit)',
                'TF Validated Beads',
                'Lib Filtered Beads (read too short)',
                'Lib Filtered Beads (fail keypass)',
                'Lib Filtered Beads (too many positive flows)',
                'Lib Filtered Beads (poor signal fit)',
                'Lib Validated Beads']

        if i==0:
            head, tail = os.path.split(maskstats)
            config_pp = ConfigParser.RawConfigParser()
            config_pp.read(os.path.join(head, process_parameter_file))
            chip = config_pp.get('global', 'Chip')
            size = chip.split(',')
            config_out.set('global','Start Row', '0')
            config_out.set('global','Start Column', '0')
            config_out.set('global','Width', int(size[0]))
            config_out.set('global','Height', int(size[1]))
            config_out.set('global','Total Wells', int(size[0])*int(size[1]))
            config_out.set('global','Percent Template-Positive Library Beads', '0') # TODO

            for key in keys:
                config_out.set('global', key, '0')

        for key in keys:
            try:
                value_in = config.get('global', key)
                value_out = config_out.get('global', key)
                config_out.set('global', key, int(value_in) + int(value_out))
            except:
                print "ERROR: StatsMerge: key %s doesn't exist" % key

    sum_wells = 0
    sum_wells += config_out.getint('global', 'Empty Wells')
    sum_wells += config_out.getint('global', 'Pinned Wells')
    sum_wells += config_out.getint('global', 'Ignored Wells')
    sum_wells += config_out.getint('global', 'Bead Wells')
    sum_wells += config_out.getint('global', 'Excluded Wells')

    if config_out.get('global','Total Wells') != sum_wells:
        print "ERROR: StatsMerge: Total Wells: %s (sum) != %s (expected)" % (sum_wells, config_out.get('global','Total Wells'))

    with open(stats_file, 'wb') as configfile:
        if verbose:
            print "Writing", stats_file

        config_out.write(configfile)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('statslist', nargs='+')
    parser.add_argument('outputstats', help='e.g. analysis.bfmask.stats')
    args = parser.parse_args()

    main_merge(args.statslist, args.outputstats, args.verbose)
