#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser

class AlignTable:
    def __init__(self, fn):
        fh = open(fn, 'r')
        self.header = fh.readline().rstrip('\r\n').split('\t')
        self.data = []
        for line in fh.readlines():
            line = line.rstrip('\r\n')
            tokens = line.split('\t')
            for i in xrange(len(tokens)):
                tokens[i] = int(tokens[i])
            self.data.append(tokens)
        fh.close()

    def merge(self, t):
        # check assumptions
        if len(self.header) != len(t.header):
            return False
        for i in xrange(len(self.header)):
            if self.header[i] != t.header[i]:
                return False
        if len(self.data) != len(t.data):
            return False
        for i in xrange(len(self.data)):
            if self.data[i][0] != t.data[i][0]:
                return False
        # merge
        for i in xrange(len(self.data)):
            for j in xrange(1, len(self.data[i])):
                self.data[i][j] += t.data[i][j]
        return True

    def dump(self):
        # header
        for i in xrange(len(self.header)):
            if 0 < i:
                sys.stdout.write('\t')
            sys.stdout.write(self.header[i])
        sys.stdout.write('\n')
        # data
        for i in xrange(len(self.data)):
            for j in xrange(1, len(self.data[i])):
                if 1 < j:
                    sys.stdout.write('\t')
                sys.stdout.write(str(self.data[i][j]))
            sys.stdout.write('\n')

def print_json( alignTables ):
    #init final_json
    final_json = {}
    for x in xrange(1, len(alignTables[0].header)):
        final_json[ alignTables[0].header[x] ] = {}

    for alignTable in alignTables:
        for tableRow in alignTable.data:
            #tableRow[0] should be the read length
            for x in xrange(1, len(tableRow) ):
                try:
                    final_json[ alignTable.header[x] ][ tableRow[0] ] += tableRow[x]
                except:
                    final_json[ alignTable.header[x] ][ tableRow[0] ] = tableRow[x]

    import json
    print json.dumps( final_json, sort_keys=True, indent=4 )




def check_option(parser, value, name):
    if None == value:
        print 'Option ' + name + ' required.\n'
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    parser = OptionParser()
    #parser.add_option('-1', '--fn-read-one', dest='fn_read_one', default=None, help="Read one")
    #parser.add_option('-2', '--fn-read-two', dest='fn_read_two', default=None, help="Read two")
    #parser.add_option('-p', '--fn-prefix', dest='fn_prefix', default=None, help="Output prefix")

    (options, args) = parser.parse_args()

    #check_option(parser, options.fn_read_one, '-1')
    #check_option(parser, options.fn_read_two, '-2')
    #check_option(parser, options.fn_prefix, '-p')

    if len(args) < 1:
        parser.print_help()
        sys.exit(1)

    data = None
    i = 0
    for fn in args:
        x = AlignTable(fn)
        if 0 == i:
            data = x
        else:
            if not data.merge(x):
                sys.stderr.write("Merge failed\n")
                sys.exit(1)
        i += 1

    print_json( [data] )
