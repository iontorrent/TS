# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import datetime
import re
import os
import fnmatch
#import logging

    
def getSavedSpace(num_days):
    '''Reports disk space saved as recorded in reportsLog.log.
    Format of the line is: "2012-10-19 11:09:03,551 [INFO] Auto_user_CB1-42-r9723-314wfa-tl_26: Size saved:* 2495.00 KB"
                            ^^^^^^^^^^^^^^^^^^^                                                              ^^^^^^^ ^^
                            group 1                                                                          grp 2   grp 3
    '''
    #logger = logging.getLogger(__name__)

    re1='((?:2|1)\\d{3}(?:-|\\/)(?:(?:0[1-9])|(?:1[0-2]))(?:-|\\/)(?:(?:0[1-9])|(?:[1-2][0-9])|(?:3[0-1]))(?:T|\\s)(?:(?:[0-1][0-9])|(?:2[0-3])):(?:[0-5][0-9]):(?:[0-5][0-9]))'	# Time Stamp 1
    re2 = '.*?'  # Non-greedy match on filler
    re3 = 'Size saved'  # Word 1
    re4 = '.*?'  # Non-greedy match on filler
    re5 = '([+-]?\\d*\\.\\d+)(?![-+0-9\\.])'  # Float 1
    re6 = '(.*?$)'  # Non-greedy match on filler
    saved_filter = re.compile(re1 + re2 + re3 + re4 + re5 + re6, re.IGNORECASE)

    saves = []
    dates = []
    
    # Read all the log files of the form: /var/log/in/reportsLog.log*
    directory = '/var/log/ion'
    for filename in os.listdir(directory):
        
        if fnmatch.fnmatch(filename, 'reportsLog.log*'):
        
            with open(os.path.join(directory, filename)) as filehandle:
                
                for line in filehandle.readlines():
                    test = saved_filter.match(line)
                    if test:
                        dates.append(datetime.datetime.strptime(test.group(1),"%Y-%m-%d %H:%M:%S"))
                        if 'KB' in test.group(3):
                            saves.append(float(test.group(2)) / 1024)
                        else:
                            saves.append(float(test.group(2)))

    total = 0
    if num_days == 0:
        for st in saves:
            total += st
    else:
        # Get space saved in previous num_days
        # time_threshold is a date num_days in the past
        time_threshold = datetime.datetime.now() - datetime.timedelta(num_days)
        for i, dat in enumerate(dates):
            # find all dates in list that are younger than time_threshold date
            if dat >= time_threshold:
                # sum the space saved for these dates
                total += saves[i]

    return round(total, 1)
