# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Simple filter/query operations on CSV files

import csv
import datetime

try:
  from itertools import compress_THIS_DOES_NOT_WORK
except:
  # compress is new in 2.7
  def compress(a, b):
    return [va for va, vb in zip(a,b) if vb == 1]


class CSVDB:
  def __init__(self, csvfile, types = None, delimiter = None):
    """
    csvfile is a path to a file or an open file-like object.

    types is a dict of types for each column: col_id: type
    Note that 'type' can be any callable object that takes a string
    value and returns a transformed value.  Objects are converted
    using type at load time.  If no headers are present, col_id is the
    integer index for the column.
    """

    # Open the CSV, guess it's content, and parse it
    if type(csvfile) is str:
      csvfile = open(csvfile)

    sample = csvfile.read(4096)

    if delimiter is None:
      dialect = csv.Sniffer().sniff(sample)
      reader = csv.reader(csvfile, dialect)
    else:
      reader = csv.reader(csvfile, delimiter=delimiter)

    csvfile.seek(0)
    try:
      if csv.Sniffer().has_header(sample):
        self.header = reader.next()
      else:
        self.header = None
    except:
      self.header = None

    _convert =  lambda row: row # identity

    if types is not None:

      if self.header is not None:
        type_map = [(self.header.index(col), mapper) for col, mapper in types.iteritems()]

      def type_convert(row, types=type_map):
        for idx, mapper in types:
          row[idx] = mapper(row[idx])
        return row
      _convert = type_convert

    self.data = [_convert(row) for row in reader]
    self.csvpath = csv
    
    # Create a header with column numbers instead of ids
    if self.header is None:
      self.header = [i for i in range(len(self.data[0]))]

    csvfile.close()
    
    return

  def __iter__(self): return iter(self.data)
  def __len__(self): return len(self.data)

  def filter(self, rows = [], cols = [], order_by = [], col_dict = False):
    """
    Filter the data.
    
    Each row filter is a list of ((col, [values,]), ...).  A row is
    returned if the value in col matches one of the values.

    cols is a list of columns to return.

    order_by is a list of ['col', (col, ascending)] used to 
    compare rows for ordering.  If ascending is -1, values are
    returned in reverse order.  
    
    If col_dict is true, return the results as a dict: {col, [values, ...]}
    """

    if len(cols) == 0:
      cols = self.header
    elif type(cols) is str:
      cols = [cols]

    # Create the column mask for each row
    col_mask = [ ((col in cols) and 1 or 0) for col in self.header]
    
    filters = [(self.header.index(col), ((type(values) is str) and [values] or values))
                for col, values in rows]
    
    # Filter the rows
    filtered = []
    for row in self.data:
      passed = 0
      for idx, values in filters:
        if row[idx] in values:
          passed += 1
      
      if passed == len(filters):
        filtered.append(compress(row, col_mask))
    
    # Sort the results
    if len(order_by) != 0:
      # Get the compressed id for each columns
      col_ids = compress(self.header, col_mask)
      
      cmps = []

      for order in order_by:
        if type(order) is str:
          asc = (col_ids.index(order), 1)
        else:
          asc = (col_ids.index(order[0]), order[1])

        cmps.append(asc)

      def filter_cmp(a, b, cmps = cmps):
        """
        Return the non-zero first comparison result
        """
        c_val = 0

        for idx, asc in cmps:
          c_val = cmp(a[idx], b[idx]) * asc
          if c_val != 0:
            break

        return c_val
      
      filtered.sort(filter_cmp)

    # Pivot into a dict
    if col_dict:
      d = {}
      col_ids = compress(self.header, col_mask)

      for col in col_ids:
        d[col] = []
      
      for row in filtered:
        for idx, col in enumerate(col_ids):
          d[col].append(row[idx])
      
      filtered = d

    return filtered

  def uniq(self, col):
    """
    Return the unique values in a column
    """
    
    idx = self.header.index(col)
    s = set()
    for row in self.data:
      s.add(row[idx])
    return s


if __name__=='__main__':
  import urllib

  def str2date(s): return datetime.datetime.strptime(s, '%m/%d/%Y')
  def nullint(s): 
    if s == 'NA': return 0 
    else:         return int(s)

  f = urllib.urlretrieve('http://shortstack:9876/data/analysis/run_table.txt')
  db = CSVDB(
    f[0], 
    types = {
      'RunDate': str2date,
      'total_reads': nullint, 
      'post_filter_reads': nullint, 
      'mRNA_matches': nullint},
    )

  projects = db.uniq('Project')

  for project in projects:
    rows = db.filter(
      rows=[('Project', project)], 
      cols=['RunDate', 'Type', 'total_reads', 'post_filter_reads', 'mRNA_matches'],
      order_by = [
        'Type', 
        ('total_reads', -1),
        ]
      )
    
    print project, '-' * (60 - len(project))
    for row in rows:
      print '    ', '\t'.join((str(v) for v in row))
    print

  print list(db.uniq('Project'))

  pivot = db.filter([('RunDate', '01/28/2011')], ['Run', 'Project', 'total_reads'], col_dict=True)
  for col, values in pivot.iteritems():
    print col, values


  print 'Iter Count:', len([row for row in db])
   # db = CSVDB(open('ercc.coverage'), delimiter='\t')
   # for row in db: print row[0]
