# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
import argparse
import ConfigParser


def main_merge(blockfolder, verbose):

    process_parameter_file = 'processParameters.txt'
    stats_file = 'bfmask.stats'

    # Output file
    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')

    for i,folder in enumerate(blockfolder):

        infile = os.path.join(folder, stats_file)

        if verbose:
            print "Reading", infile

        config = ConfigParser.RawConfigParser()
        config.read(infile)

        keys = ['Total Wells', 'Excluded Wells', 'Empty Wells',
                'Pinned Wells', 'Ignored Wells', 'Bead Wells', 'Dud Beads',
                'Ambiguous Beads', 'Live Beads',
                'Test Fragment Beads', 'Library Beads',
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

            config_pp = ConfigParser.RawConfigParser()
            config_pp.read(os.path.join(folder, process_parameter_file))
            chip = config_pp.get('global', 'Chip')
            size = chip.split(',')
            config_out.set('global','Start Row', '0')
            config_out.set('global','Start Column', '0')
            config_out.set('global','Width', int(size[0]))
            config_out.set('global','Height', int(size[1]))

            config_out.set('global','Percent Template-Positive Library Beads', '0') # TODO

            for key in keys:
                config_out.set('global', key, '0')

        for key in keys:
            value_in = config.get('global', key)
            value_out = config_out.get('global', key)
            config_out.set('global', key, int(value_in) + int(value_out))

    with open(stats_file, 'wb') as configfile:
        if verbose:
            print "Writing", stats_file

        config_out.write(configfile)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('blockfolder', nargs='+')
    args = parser.parse_args()

    main_merge(args.blockfolder, args.verbose)
