#(c) 2011 Ion Torrent Systems, Inc.
#(c) 2011 Life Technologies

# simple script to query the localhost via the REST API for a list of attached file servers in the Torrent Browser
# Author: Mel Davey

import json
import httplib2

#add our authentication first
h = httplib2.Http(".cache")
h.add_credentials('ionadmin', 'ionadmin')

#ask for the list of file servers from the Torrent Browser running on localhost
resp, content = h.request("http://localhost/rundb/api/v1/fileserver?format=json", "GET")

#content is a json object, we make a python dict out of it
contentdict = json.loads(content)

#and grab the objects dict from the content
objects = contentdict['objects']

#and loop through each object in our objects list and extract the field of interest
for obj in objects:
    print obj['filesPrefix']

