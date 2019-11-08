# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

import torrentPy
import unittest
import os

# execfile('../example/examples.py')


class TestTorrentPy(unittest.TestCase):
    def setUp(self):
        # trick to load example function that we'll use for testing
        execfile("../examples/examples.py", globals())

    def test_read_Wells(self):
        r = read_Wells(os.path.join(".", "1.wells"))
        self.assertTrue(r["max_amplitude"] > 0)

        with self.assertRaises(Exception):
            print("Expected Error:")
            r = read_Wells(os.path.join(".", "blah.wells"))

    def test_read_Bam(self):
        r = read_Bam(os.path.join(".", "rawlib.bam"))
        self.assertEqual(r["numRecs"], r["numRecs1"])
        self.assertEqual(r["numRecs2"], 10)
        self.assertGreater(r["bamlist_dnareg"], 1)

        with self.assertRaises(Exception):
            print("Expected Error:")
            r = read_Bam(os.path.join(".", "blah.bam"))

    def test_read_Dat(self):
        r = read_Dat(os.path.join(".", "acq_0000.dat"))
        v = r["d"].flatten()
        self.assertGreater(v.max(), 0)
        v1 = r["d1"].flatten()
        self.assertGreater(v1.max(), 0)

        with self.assertRaises(Exception):
            print("Expected Error:")
            r = read_Dat(os.path.join(".", "blah.dat"))

    def test_read_BfMask(self):
        r = read_BfMask(os.path.join(".", "bfmask.bin"))
        v = r["bfmask"].flatten().sum()
        self.assertGreater(v, 0)
        with self.assertRaises(Exception):
            print("Expected Error:")
            r = read_BfMask(os.path.join(".", "blah.bin"))

    def test_read_Debug(self):
        r = read_Debug(".")
        with self.assertRaises(Exception):
            print("Expected Error:")
            r = read_Debug("blah")

    def test_treephaser(self):
        r = treephaser(".")


if __name__ == "__main__":

    unittest.main()

##
##    print r
##    try:
##        r=read_Wells(os.path.join(args.data_dir,'2.wells'))
##    except Exception, e: print e
##
##    r=read_Wells_Flow(os.path.join(args.data_dir,'1.wells'))
##    print r
##
#    ret=read_Bam( os.path.join(args.data_dir,'rawlib.bam') )
#    #ret=read_Bam( os.path.join(args.data_dir,'blah.bam') )
#
#    ret = read_Dat( os.path.join(args.data_dir,'acq_0000.dat') )
#    #ret = read_Dat( os.path.join(args.data_dir,'blah.dat') )
#
#    ret = read_BfMask( os.path.join(args.data_dir,'bfmask.bin') )
# ret = read_BfMask( os.path.join(args.data_dir,'blah.bin') )
