#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import json
from optparse import OptionParser

class AlignStatsL:

    def __get_val_int(self, line):
        (key, val) = line.split(' = ')
        return int(val);
    
    def __get_val_float(self, line):
        (key, val) = line.split(' = ')
        return float( val )

    def __get_length(self, line):
        l = line.split(" ")
        return l[1]

    def __get_num_reads(self, line):
        l = line.split(" ")
        m = l[1].split("Q")
        return [int(m[0]), int(l[4])]

    def __init__(self, lines):
        # assumes 14 lines
        self.length = self.__get_length(lines[0])
        self.cp = self.__get_val_float(lines[0])
        self.mcd = self.__get_val_float(lines[1])
        self.a = self.__get_val_int(lines[2])
        self.mal = self.__get_val_int(lines[3])
        self.mpi = self.__get_val_int(lines[4])
        self.la = self.__get_val_int(lines[5])
        self.used = 0
        self.x = []
        self.y = []
        self.finalized = False

        i = 6
        while i < len(lines):
            if not str(self.length) in lines[i]:
                self.used = i
                break
            (x, y) = self.__get_num_reads(lines[i])
            self.x.append(int(x))
            self.y.append(int(y))
            i = i + 1
        self.used = i
        self.finalized = True

    def dump(self):
        if not self.finalized:
            self.finalize()
        print "Filtered %s Coverage Percentage = %d" % (self.length, self.cp)
        print "Filtered %s Mean Coverage Depth = %d" % (self.length, self.mcd)
        print "Filtered %s Alignments = %d" % (self.length, self.a)
        print "Filtered %s Mean Alignment Length = %d" % (self.length, self.mal)
        print "Filtered Mapped Bases in %s Alignments = %d" % (self.length, self.mpi)
        print "Filtered %s Longest Alignment = %d" % (self.length, self.la)
        for i in xrange(len(self.x)):
            print "Filtered %d%s Reads = %d" % (self.x[i], self.length, self.y[i])

    def merge(self, a):
        if self.cp != a.cp or self.mcd != a.mcd:
            return False
        self.a += a.a
        if self.finalized:
            self.mal = (self.a * self.mal)
        else:
            self.mal += (a.a * a.mal)
        self.mpi += a.mpi
        if self.la < a.la:
            self.la = a.la
        for i in xrange(len(self.x)):
            if self.x[i] != a.x[i]:
                return False
            self.y[i] += a.y[i]
        self.finalized = False
        return True

    def finalize(self):
        self.mal = int(self.mal / float(self.a))
        self.finalized = True


class AlignStatsHeader:

    def __init__(self, lines):
        self.header = {}
        self.keys = []
        for i in xrange(5):
            line = lines[i]
            line = line.rstrip('\r\n')
            (key, val) = line.split(" = ")
            self.header[key] = val
            self.keys.append(key)

    def dump(self):
        for key in self.keys:
            print "%s = %s" % (key, self.header[key])

    def eq(self, header):
        if len(self.keys) != len(header.keys):
            return False
        for key in self.keys:
            if not key in header.keys:
                return False
        return True

class AlignStats:
    def __init__(self, lines):
        i = 0
        self.header = AlignStatsHeader(lines)
        i += 5
        self.z = []
        while i < len(lines):
            stats = AlignStatsL(lines[i:])
            self.z.append(stats)
            i += stats.used

    def dump(self):
        self.header.dump()
        for a in self.z:
            a.dump()

    def merge(self, a):
        # check headers
        if not self.header.eq(a.header):
            return False
        # check lengths
        for i in xrange(len(self.z)):
            if self.z[i].length != a.z[i].length:
                return False
        # merge
        for i in xrange(len(self.z)):
            if not self.z[i].merge(a.z[i]):
                return False
        return True



sectionmap = {
        'other':['Coverage Percentage'],
        'to_average':['Mean Coverage Depth', 'Mean Alignment Length' ],
        'to_sum':['Alignments','alignment_sums','Mapped Bases in Alignments' ],
        'to_maximize':['Longest Alignment'],
        }

varmap= {
        'Coverage Percentage':'cp',
        'Mean Coverage Depth':'mcd',
        'Alignments':'a',
        'Mean Alignment Length':'mal',
        'Mapped Bases in Alignments':'mpi',
        'Longest Alignment':'la',
        #'alignment_sums':'to_sum'
        }


def make_json( alignstat ):
    
    json_dump = {}
    json_dump['header'] = alignstat.header.header
    genome_size = int(json_dump['header']['Genomesize'])
    total_num_reads = int( json_dump['header']['Total number of Reads'] )
    for section in sectionmap:
        try:
            tmp = json_dump[ section ]
        except:
            json_dump[ section ] = {}
        for stat in alignstat.z:
            category = str( stat.length )
            try:
                tmp = json_dump[ section ][ category ]
            except:
                json_dump[ section ][ category ] = {}
            for subcategory in sectionmap[ section ]:
                try:
                    tmp = json_dump[ section ][ category ][ subcategory ]
                except:
                    json_dump[ section ][ category ][ subcategory ] = {}
                if subcategory != 'alignment_sums':
                    if section == 'to_average':
                        l = []
                        l.append( stat.__dict__[ varmap[ subcategory ]  ] )
                        if subcategory.find('Mean Alignment') >= 0:
                            l.append( total_num_reads )
                        elif subcategory.find('Mean Coverage') >= 0:
                            l.append(genome_size)
                        json_dump[ section ][ category ][ subcategory ] = l
                    else:
                        json_dump[ section ][ category ][ subcategory ] = stat.__dict__[ varmap[ subcategory ] ]
                else:
                    sums = {}
                    for i in xrange( len(stat.x) ):
                        sums[ stat.x[i] ] = stat.y[ i ] 
                    json_dump[ section ][ category ][ subcategory ] = sums
                        
    
    return json_dump

def merge_json( jsons ):
    import copy as dp
    final_json = dp.deepcopy( jsons[1] ) #just to init structure
    #zero out fields:

    for category in final_json[ 'to_sum' ]:
        for readlen in final_json[ 'to_sum' ][ category ][ 'alignment_sums' ]:
            final_json[ 'to_sum' ][ category ][ 'alignment_sums' ][ readlen ] = 0


    #merge headers:
    total_reads = 0
    for thejson in jsons:
        total_reads += int( thejson['header']['Total number of Reads'] )

    final_json['header']['Total number of Reads'] = str(total_reads)

    for section in sectionmap:
        
        if section == 'other':
            #TODO add merge process for read totals
            pass
        elif section == 'to_average':
            #jsons is a list of json dictionaries in a json-like format
            #to_average: { "Q10": { 
            
            for thejson in jsons:
                for category in thejson[ section ]:
                    for subcategory in thejson[ section ][ category ]:
                        #( a1*d1 + a2*d2 )/ ( d1 + d2 )
                        numerator = sum( [ ajson[ section ][ category ][ subcategory ][0]*ajson[ section ][ category ][ subcategory ][1] for ajson in jsons ] )
                        denom = sum( [ ajson[ section ][ category ][ subcategory ][1] for ajson in jsons ] ) 
                        #print category,subcategory, ( numerator / denom )
                        final_json[ section ][ category ][ subcategory ] = [ float( numerator ) / float( denom ), denom ]
        elif section == 'to_sum':
            #section  cat   subcat           
            #to_sum { Q10 { alignment_sums { len:sum } } }
            for thejson in jsons:
                for category in thejson[ section ]:
                    for subcategory in thejson[ section ][ category ]:
                        if subcategory == 'alignment_sums':
                            alignment_sums = {}
                            for readlen in thejson[ 'to_sum' ][ category ][ 'alignment_sums' ]:
                                #print section,category,subcategory,readlen
                                final_json[ section ][ category ][ 'alignment_sums' ][ readlen ] += thejson[ section ][ category ][ subcategory ][ readlen ]

                            #final_json[ section ][ category ][ subcategory ] = alignment_sums
                        else:
                            thesum = sum( [ ajson[ section ][ category ][ subcategory ] for ajson in jsons ] )
                            #sys.exit()
                            final_json[ section ][ category ][ subcategory ] = thesum
        elif section == 'to_maximize':
            for category in thejson[ section ]:
                for subcategory in thejson[ section ][ category ]:
                    thejson_val = thejson[ section ][ category ][ subcategory ]
                    final_json_val = final_json[ section ][ category ][ subcategory ]
                    if thejson_val > final_json_val:
                        final_json[ section ][ category ][ subcategory ] = thejson_val

    return final_json

def print_json( thejson ):
    print json.dumps( thejson, sort_keys=True, indent=4 )


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

    a = []
    for fn in args:
        fh = open(fn, 'r')
        a.append(AlignStats(fh.readlines()))
        fh.close()
        fh = None
    
    data = None
    i = 0
    thejsons = []
    for i,x in enumerate(a):
        #x.dump()
        thejsons.append( make_json( x ) )
        #write_json( thejsons[-1], outfile="alignStats.%d.json" % ( i ) )
        if 0 == i:
            data = x
        i += 1

    #data.dump()

    print_json( merge_json(thejsons) ) 
