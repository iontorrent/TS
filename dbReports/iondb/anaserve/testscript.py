# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

def cat(fname):
    infile = open(fname)
    print(infile.read())
    infile.close()

if __name__ == '__main__':
    import sys
    print "Command line:", ' '.join(sys.argv)
    if len(sys.argv) > 1:
        print "Parameters received:", cat(sys.argv[1])
        sys.exit(0)
    else:
        print "Received insufficient arguments"
        sys.exit(1)
