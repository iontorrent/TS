#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import requests
import simplejson as json

resp = requests.get('http://localhost/rundb/api/v1/rig/PGM?format=json',
auth=('ionadmin', 'ionadmin'))

rdata = json.loads(resp.content)

rdata.update(name='PGM2')
rdata.pop('resource_uri')
rdata["location"].pop('resource_uri')

pdata = json.dumps(rdata)

status = requests.put('http://localhost/rundb/api/v1/rig/PGM2/',
data=pdata,
headers={'content-type':'application/json'},
auth=('ionadmin', 'ionadmin'))
