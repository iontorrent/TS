# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import csv
import StringIO


def table2csv(table):
    buf = StringIO.StringIO()
    wrtr = csv.writer(buf)
    wrtr.writerows(table)
    ret = buf.getvalue()
    buf.close()
    return ret
