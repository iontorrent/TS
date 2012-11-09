#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import traceback
from djangoinit import *
from django import db
from django.db import transaction
import sys
import os

import iondb.bin.djangoinit
from django.db import models
from iondb.rundb import models
from socket import gethostname
from django.contrib.auth.models import User

def add_barcode_set():
	kwargs = {'name' : 'single_nuke',
		'type' : 'text',
		'sequence' : 'T',
		'length':1,
		'floworder':'TACG'
	}
	ret = models.dnaBarcode(**kwargs)
	ret.save()
	kwargs = {
		'name' : 'single_nuke',
		'type' :'text',
		'sequence' : 'A',
		'length':1,
		'floworder':'TACG'
	}
	ret = models.dnaBarcode(**kwargs)
	ret.save()
	kwargs = {
		'name' : 'single_nuke',
		'type' : 'text',
		'sequence' : 'C',
		'length':1,
		'floworder':'TACG'
	}
	ret = models.dnaBarcode(**kwargs)
	ret.save()
	kwargs = {
		'name' :'single_nuke',
		'type' : 'text',
		'sequence' : 'G',
		'length':1,
		'floworder':'TACG'
	}
	ret = models.dnaBarcode(**kwargs)
	ret.save()

if __name__=="__main__":
    add_barcode_set()
