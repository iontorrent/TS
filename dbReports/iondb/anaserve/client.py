# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import xmlrpclib


PORT = 10000


def connect(host, port):
    return xmlrpclib.ServerProxy("http://%s:%d" % (host, port), allow_none=True)

def get_running_jobs(host, port):
    conn = connect(host, port)
    return conn.running()

def load(fname):
    infile = open(fname)
    ret = infile.read()
    infile.close()
    return ret

if __name__ == '__main__':
    # do some testing
    server = connect('localhost', PORT)
    tscript = load("testscript.py")
    name = server.startanalysis("test1",
                                tscript,
                                load("testparams.json"))
    print "Run name:", name
    name2 = server.startanalysis("test2",
                                 tscript,
                                 {"name": "test script 2",
                                  "numbers": [1, 2, 3, 4, 5]})
