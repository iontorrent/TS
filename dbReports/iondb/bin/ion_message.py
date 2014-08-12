#!/usr/bin python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
from djangoinit import *
from iondb.rundb.models import Message


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Post a site message to the Torrent browser.")
	parser.add_argument("message", type=str, help="The message string to be displayed.")
	parser.add_argument("-d", "--duplicate", action="store_true", help="Create a duplicate post if necessary.")
	args = parser.parse_args()

	msg = Message.objects.filter(body=args.message).count()
	if not msg or args.duplicate:
		Message.info(args.message)
		print("Posted message")
	else:
		print("Didn't post duplicate message")
