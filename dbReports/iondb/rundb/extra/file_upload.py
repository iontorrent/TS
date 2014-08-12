# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.utils import timezone
from StringIO import StringIO
import subprocess
import select
import os.path


class BasicFileUpload(object):

	def __init__(self, local_path, remote_path, config=None):
		self.local_path = local_path
		self.remote_path = remote_path
		self.progress = 0
		self.size = os.path.getsize(local_path)
		self.config = config
		self.start_time = None
		self.end_time = None
		self.status = "Initialized"

	def authenticate(self):
		pass

	def start(self):
		self.start_time = timezone.now()
		self.status = "Started"

	def poll(self):
		return self.status == "Complete"

	def get_progress(self):
		return float(self.size) and (self.progress / float(self.size))

	def finished(self):
		self.status = "Complete"
		self.end_time = timezone.now()

	def abort(self):
		self.status = "Aborted"


class RsyncFileUpload(BasicFileUpload):

	def __init__(self, local_path, remote_path, config=None):
		self.proc = None
		self.cmd = None
		self.output = StringIO()
		self.rsync_started = None
		self.user = None
		self.password = None
		super(RsyncFileUpload, self).__init__(local_path, remote_path, config)

	def authenticate(self, **kwargs):
		self.user = kwargs['user']
		self.password = kwargs['password']

	def start(self):
		self.cmd = [
			"rsync",
			"--progress",
			self.local_path,
			self.remote_path
		]
		self.proc = subprocess.Popen(self.cmd, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
			env={"USER": self.user, "RSYNC_PASSWORD": self.password})
		return super(RsyncFileUpload, self).start()

	def get_progress(self):
		returncode = self.proc.poll()
		if returncode == 0:
			self.finished()
			self.progress = self.size
			return 1.0
		elif returncode is None:
			fileno = self.proc.stdout.fileno()
			value = None
			reads = True
			while reads:
				reads, _, _ = select.select([fileno], [], [], 0)
				if reads:
					value = reads[0].readline()
					progress = value.split('\r')[-1]
					bytes = progress.split()[0]
					try:
						self.progress = int(bytes)
						return float(self.size) and (self.progress / float(self.size))
					except ValueError as err:
						print(err)
		else:
			self.status = "Failed"

	def abort(self):
		self.proc.kill()
		super(RsyncFileUpload, self).abort()


