#!/usr/bin/env python
'''Parses explog.txt and explog_final.txt'''
from ion.utils.explogparser import load_log_path
from ion.utils.explogparser import parse_log

explog_path = "./explog.txt"
# Parse the explog.txt file
text = load_log_path(explog_path)
dict = parse_log(text)

#NOTE: keywords will have spaces replaced with underscores and all letters lowercased.
keywords = [
    "experimenterrorlog",
    "experimentinfolog",
    "explog_done",
    "experiment_name",
    "f",
]
for key,value in dict.items():
    if any(key.startswith(item) for item in keywords):
        print "(%s) ==> %s" % (key, value)