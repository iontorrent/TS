# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

def colorify(s):
   """
   Deprecated.
   """
   ret = ""
   map = {"T": '<font color="red">T</font>',
          "C": '<font color="blue">C</font>',
          "A": '<font color="green">A</font>',
          "G": '<font color="gray">G</font>',
          " ": '&nbsp'}
   for c in s:
       if map.has_key(c):
           ret += map[c]
       else:
           ret += c
   return ret 

