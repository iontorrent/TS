#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os

def USBinstallercheck():
	root_dir='/media'
	directories = os.listdir(root_dir)
	for eachDir in directories:
		if eachDir.startswith("TS"):
			dirPath = os.path.join(root_dir, eachDir)
			inDir =os.listdir(dirPath)
			for eachFile in inDir:
				if eachFile == 'runme':
					return True

	return False
