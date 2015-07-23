# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
"""
Query localhost via REST API for list of attached file servers

Author: Mel Davey
"""

import json
import httplib2


def main():
    """Query localhost via REST API for list of attached file servers"""
    #add our authentication first
    http = httplib2.Http(".cache")
    http.add_credentials('ionadmin', 'ionadmin')

    # ask for list of file servers from Torrent Browser running on localhost
    url = "http://localhost/rundb/api/v1/fileserver?format=json"
    resp, content = http.request(url, "GET")

    # content is a json object, we make a python dict out of it
    contentdict = json.loads(content)

    # and grab the objects dict from the content
    objects = contentdict['objects']

    # loop through each object in objects list and extract field of interest
    for obj in objects:
        print obj['filesPrefix']

if __name__ == "__main__":
    main()
