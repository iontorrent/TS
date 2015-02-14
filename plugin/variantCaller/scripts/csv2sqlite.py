#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

# A simple Python script to convert csv files to sqlite (with type guessing)
#
# @author: Rufus Pollock
# Placed in the Public Domain
# from https://github.com/rgrp/csv2sqlite

import csv
import sqlite3
import json
import os

def convert(filepath_or_fileobj, dbpath, table='data'):
    if isinstance(filepath_or_fileobj, basestring):
        fo = open(filepath_or_fileobj)
    else:
        fo = filepath_or_fileobj
    reader = csv.reader(fo,delimiter='\t')

    types = _guess_types(fo)
    fo.seek(0)
    headers = reader.next()

    for i, h in enumerate(headers):
        if h == "Filter":
            headers[i] = headers[i-1] + " " + h

    _columns = ','.join(
        ['"%s" %s' % (header, _type) for (header,_type) in zip(headers, types)]
        )

    conn = sqlite3.connect(dbpath)
    c = conn.cursor()

    #NOTE: Chrom is expected to be the first col in the csv file
    columns = '"id" INTEGER PRIMARY KEY, "ChromSort" UNSIGNED BIG INT,' + _columns
    c.execute('CREATE table %s (%s)' % (table, columns))

    #try to get the correct chromosome order
    chromosome_order = {}
    if os.path.exists(os.path.join(os.path.split(dbpath)[0],"variant_summary.json")):
        print "the variant_summary file is there"
        variant_summary = json.load(open(os.path.join(os.path.split(dbpath)[0],"variant_summary.json")))
        for i, variant in enumerate(variant_summary["variants_by_chromosome"]):
            chromosome_order[variant["chromosome"]] = i
    else:
        print "there isn't a variant_summary file, the ordering will be random"

    #add one more ? to be for the PK
    _insert_tmpl = 'insert into %s values (%s)' % (table, ','.join(['?']*(2+len(headers))))
    for i, row in enumerate(reader):
        # we need to take out commas from int and floats for sqlite to
        # recognize them properly ...
        row = [ x.replace(',', '') if y in ['real', 'integer'] else x
                for (x,y) in zip(row, types) ]

        #convert the chrom name into an int so it can be sorted
        row.insert(0, chromosome_order.get(row[0], 0))
        #insert the PK, starting at 0
        row.insert(0,i)
        c.execute(_insert_tmpl, row)

    conn.commit()
    c.close()

def _guess_types(fileobj, max_sample_size=100):
    '''Guess column types (as for SQLite) of CSV.

    :param fileobj: read-only file object for a CSV file.
    '''
    reader = csv.reader(fileobj,delimiter="\t")
    # skip header
    _headers = reader.next()
    # we default to text for each field
    types = ['text'] * len(_headers)
    # order matters
    # (order in form of type you want used in case of tie to be last)
    options = [
        ('text', unicode),
        ('real', float),
        ('integer', int)
        # 'date',
        ]
    # for each column a set of bins for each type counting successful casts
    perresult = {
        'integer': 0,
        'real': 0,
        'text': 0
        }
    results = [ dict(perresult) for x in range(len(_headers)) ]
    count = -1
    for count,row in enumerate(reader):
        for idx,cell in enumerate(row):
            cell = cell.strip()
            # replace ',' with '' to improve cast accuracy for ints and floats
            cell = cell.replace(',', '')
            for key,cast in options:
                try:
                    # for null cells we can assume success
                    if cell:
                        cast(cell)
                    results[idx][key] += 1
                except (ValueError), inst:
                    pass
        if count >= max_sample_size:
            break
    for idx,colresult in enumerate(results):
        for _type, dontcare in options:
            if colresult[_type] == count + 1:
                types[idx] = _type
    return types


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('''csv2sqlite.py {csv-file-path} {sqlite-db-path} [{table-name}]

Convert a csv file to a table in an sqlite database (which need not yet exist).

* table-name is optional and defaults to 'data'
''')
        sys.exit(1)
    convert(*sys.argv[1:])
