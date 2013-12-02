#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# vim: tabstop=4 shiftwidth=4 softtabstop=4 noexpandtab
# Ion Plugin - Ion Reporter Uploader

import getopt
import os
import sys
import json
import datetime
import time
import urllib
import urllib2
import subprocess
import ast
import shutil
from ion.plugin import *

import extend

#######
###  global variables      
pluginName = ""
plugin_dir = ""

class IonReporterUploader(IonPlugin):
	version = "4.0-r%s" % filter(str.isdigit,"$Revision: 77705 $")
	runtypes = [ RunType.THUMB, RunType.FULLCHIP, RunType.COMPOSITE ]
	runlevels = [ RunLevel.PRE, RunLevel.BLOCK, RunLevel.POST ]
	features = [ Feature.EXPORT ]

	#JIRA [TS-7563]
	allow_autorun = False

	global pluginName , plugin_dir, launchOption, commonScratchDir
	pluginName = "IonReporterUploader"
	plugin_dir = os.getenv("RUNINFO__PLUGIN_DIR") or os.path.dirname(__file__)
	launchOption = "upload_and_launch"
	extend.setPluginName(pluginName)
	extend.setPluginDir(plugin_dir)

	#print "plugin dir is " + os.getenv("RUNINFO__PLUGIN_DIR") 
	#print "plugin name is " + os.getenv("RUNINFO__PLUGIN_NAME") 
	#print "plugin runtime key is " + os.getenv("RUNINFO__PK") 

	def pre_launch(self):
		return True

	#Launches script with parameter jsonfilename. Assumes data is a dict
	def launch(self,data=None):
		global launchOption, commonScratchDir
		print pluginName + ".py launch()"
		startpluginjsonfile = os.getenv("RESULTS_DIR") + "/startplugin.json"
		print "input file is " + startpluginjsonfile
		data = open(startpluginjsonfile).read()
		commonScratchDir = self.get_commonScratchDir(data)
		runtype = self.get_runinfo("run_type", data)
		runlevel = self.get_runinfo("runlevel", data)
		print "RUN TYPE " + runtype
		print "RUN LEVEL " + runlevel
		dt = json.loads(data)
		pluginconfig = dt["pluginconfig"]
		if 'launchoption' in pluginconfig:
			launchOption = pluginconfig["launchoption"]
		self.write_classpath()

		if runtype == "composite" and runlevel == "pre":
			self.pre(data)
		elif runtype == "composite" and runlevel == "block":
			self.block(data)
		elif runtype == "composite" and runlevel == "post":
			print "POST IS CALLED" 
			self.post(data)
		elif runtype == "wholechip" and runlevel == "default": #PGM
			self.default(data)
		elif runtype == "wholechip" and runlevel == "post": #PGM
			self.default(data)
		else:
			print "IonReporterUploader : ignoring the above combination of runtype and runlevel .. exit .."
		return True

	#Run Mode: Pre - clear old JSON, set initial run timestamp, log version and start time	
	def pre(self, data):
		global pluginName, launchOption
		self.clear_JSON()
		self.set_serial_number()

		timestamp = self.get_timestamp()
		file = open(commonScratchDir + "/timestamp.txt", "w+")
		file.write(timestamp)	
		file.close()
		self.inc_submissionCounts()
		self.write_log("VERSION=1.2", data)
		self.write_log(timestamp + " " + pluginName, data)
		self.write_classpath()
		self.test_report(data)
		self.get_plugin_parameters(data)
		log_text = self.get_timestamp() + pluginName + " : executing the IonReporter Uploader Client -- - pre"
		print "LAUNCH OPTION " + launchOption
		if launchOption == "upload_and_launch":
			os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.Launcher -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o pre ||true")
		elif launchOption == "upload_only":
			os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.LauncherForUploadOnly -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o pre ||true")
		os.system("sleep 2")
		return True

	#Run Mode: Block (Proton) - copy status_block.html, test report exists, log, initialize classpath and objects.json, run java code
	def block(self, data):
		global pluginName, launchOption
		self.copy_status_block()
		#self.test_report(data)
		log_text = self.get_timestamp() + pluginName + " : executing the IonReporter Uploader Client -- block"
		self.write_classpath()
		self.get_plugin_parameters(data)
		print "LAUNCH OPTION " + launchOption
		if launchOption == "upload_and_launch":
			os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.Launcher -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o block ||true")
		elif launchOption == "upload_only":
			os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.LauncherForUploadOnly -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o block ||true")
		os.system("sleep 2")
		self.write_log(pluginName + " : executed the IonReporter Client ... Exit Code = " + `os.getenv("LAUNCHERCLIENTEXITCODE")`, data)
		print "Returning from Block"
		return True

	#Run Mode: Default (PGM)- copy status_block.html, test report exists, log, initialize classpath and objects.json, run java code
	def default(self, data):
								global pluginName, launchOption
								self.clear_JSON()
								self.set_serial_number()
								timestamp = self.get_timestamp()
								file = open(commonScratchDir + "/timestamp.txt", "w+")
								file.write(timestamp)
								file.close()
								self.inc_submissionCounts()
								self.write_log("VERSION=1.2", data)
								self.write_log(timestamp + " IonReporterUploader", data)
								self.copy_status_block()
								self.test_report(data)
								log_text = self.get_timestamp() + pluginName + " : executing the IonReporter Uploader Client -- default"
								#print "before calling classpath"
								self.write_classpath()
								self.get_plugin_parameters(data)
								print "LAUNCH OPTION " + launchOption
								if launchOption == "upload_and_launch":
									os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.Launcher -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o default")
								elif launchOption == "upload_only":
									os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.LauncherForUploadOnly -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o default")
								os.system("sleep 2")
								return True

	# Run Mode: Post
	def post(self, data):
		global pluginName, launchOption
		self.write_classpath()
		log_text = self.get_timestamp() + pluginName + ": executing the IonReporter Uploader Client -- post"
		self.get_plugin_parameters(data)
		print "LAUNCH OPTION " + launchOption
		if launchOption == "upload_and_launch":
			os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.Launcher -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o post  ||true")
		elif launchOption == "upload_only":
			os.system("java -Xms3g -Xmx3g -XX:MaxPermSize=256m -Dlog.home=${RESULTS_DIR} com.lifetechnologies.ionreporter.clients.irutorrentplugin.LauncherForUploadOnly -j ${RESULTS_DIR}/startplugin.json -l " + self.write_log(log_text, data) + " -o post  ||true")
		os.system("sleep 2")
		return True 

	# Versions
	def get_versions(self):
		global pluginName
		self.write_classpath()
		api_url = os.getenv('RUNINFO__API_URL', 'http://localhost/rundb/api/') + "/v1/plugin/?format=json&name=" + pluginName + "&active=true"
		f = urllib2.urlopen(api_url)
		d = json.loads(f.read())
		objects = d["objects"]
		config = objects[0]["config"]
		return extend.get_versions({"irAccount":config})


	# Returns Sample Relationship Fields used in TS 3.0. First invokes java program to save workflows to JSON file. Then reads.
	def getUserInput(self):
		global pluginName
		self.write_classpath()
		api_url = os.getenv('RUNINFO__API_URL', 'http://localhost/rundb/api/') + "/v1/plugin/?format=json&name=" + pluginName + "&active=true"
		f = urllib2.urlopen(api_url)
		d = json.loads(f.read())
		objects = d["objects"]
		if not objects : 
			return None
		config = objects[0]["config"]
		return {"status": "false", "error": ""}

	    #return extend.getUserInput({"irAccount":config})
		#return extend.get_versions({"irAccount":config})
		#return extend.authCheck({"irAccount":config})
		#return extend.getWorkflowList({"irAccount":config})
		#return extend.getUserDataUploadPath({"irAccount":config})
		#return extend.sampleExistsOnIR({"sampleName":"Sample_100","irAccount":config})        # always returning false?
		#return extend.getUserDetails({"userid":"vipinchandran_n@persistent.co.in","password":"123456","irAccount":config})
		#return extend.validateUserInput({"userInput":{},"irAccount":config})
		#return extend.getWorkflowCreationLandingPageURL({"irAccount":config})


	def get_commonScratchDir(self,data):
		d =json.loads(data)
		runinfo = d["runinfo"]
		runinfoPlugin = runinfo["plugin"]
		return runinfoPlugin["results_dir"]

	# increment the submission counts   # not thread safe, TBD.
	def inc_submissionCounts(self):
		newCount = 1
		line=""
		#if os.path.exists(os.getenv("RESULTS_DIR") + "/submissionCount.txt"):
		if os.path.exists(commonScratchDir + "/submissionCount.txt"):
			submissionfile = open(commonScratchDir + "/submissionCount.txt")
                	line = submissionfile.readline()
                	submissionfile.close()
		if line != "":
			newCount = newCount + int(line)
		submissionfileWriter = open(commonScratchDir + "/submissionCount.txt","w")
		submissionfileWriter.write(str(newCount))
		submissionfileWriter.close()
                return newCount
	
	# Returns timestamp from system
	def get_timestamp(self):
		now = datetime.datetime.now()
		timeStamp = `now.year` + "-" + `now.month` + "-" + `now.day` + "_" + `now.hour` + "_" + `now.minute` + "_" + `now.second`
		return timeStamp

	# Returns values (thru keys) from run_info json file
	def get_runinfo(self, key, data):
		d = json.loads(data)
		runinfo = d["runinfo"]
		if (key == "runlevel"):
			plugin = d["runplugin"]
			value = plugin[key]
			return value
		elif (key == "run_type"):
			plugin = d["runplugin"]
			value = plugin[key]
			return value
		value = runinfo[key]
		return value

	# Defines classpath 
	def write_classpath(self):
		global pluginName
		# do not print anything to the standard out here .. the api calls use this, and gets messed up.
		#sub1 = subprocess.Popen("find /results/plugins/" + pluginName + "/ |grep \"\.jar$\" |xargs |sed 's/ /:/g'", shell=True, stdout=subprocess.PIPE)
		sub1 = subprocess.Popen("find " + plugin_dir + "/ |grep \"\.jar$\" |xargs |sed 's/ /:/g'", shell=True, stdout=subprocess.PIPE)
		#classpath_str = os.getenv("RUNINFO__PLUGIN_DIR") + "/lib/java/shared:" + sub1.stdout.read().strip()
		classpath_str = plugin_dir + "/lib/java/shared:" + sub1.stdout.read().strip()
		os.environ["CLASSPATH"] = classpath_str
		if (os.getenv("LD_LIBRARY_PATH")):
			ld_str = plugin_dir + "/lib:" + os.getenv("LD_LIBRARY_PATH")
		else :  
			ld_str = plugin_dir + "/lib"
			os.environ["LD_LIBRARY_PATH"] =  ld_str

	# Tests if report exists
	def test_report(self, data):
		global pluginName
		net = self.get_runinfo("net_location", data)
		pk = self.get_runinfo("pk", data)
		report = "/Default_Report.php?do_print=true"
		try1 = net + "/report/latex/" + str(pk) + ".pdf"
		try:
			urllib2.urlopen(try1)
		except urllib2.URLError, e:
			self.write_log("Report Generation (report.pdf) failed", data) 


	# Create objects.json file (plugin parameters) thru RESTful
	def get_plugin_parameters(self, data):
		global pluginName
		api_url = self.get_runinfo("api_url", data) + "/v1/plugin/?format=json&name=" + pluginName + "&active=true"
		results_dir = self.get_runinfo("results_dir", data) + "/objects.json"
		urllib.urlretrieve(api_url, results_dir)

		#Check if new file exists
		if not os.path.isfile(results_dir):
			api_url = os.getenv('RUNINFO__API_URL', 'http://localhost/rundb/api/') + "/v1/plugin/?format=json&name=" + pluginName + "&active=true"
			urllib.urlretrieve(api_url, results_dir)
			if not os.path.isfile(results_dir):
				self.write_log("ERROR getting objects from database", data)
				sys.exit()

	# Writes to directory log file
	def write_log(self, text, data):
		log_file = self.get_runinfo("results_dir", data) + "/log.txt"
		file = open(log_file, "a")
		file.write(text)
		file.write("\n")
		return log_file
	
	#Clear JSON file to initial state (0%)
	def clear_JSON(self):
		if os.path.exists(os.getenv("RESULTS_DIR") + "./progress.json"):
			prog = open("progress.json", "w")
			prog.write("{ \"progress\": \"0\", \"status\": \"Started\", \"channels\" :[]  }")
		return True
	
	def set_serial_number(self):
		sub1 = subprocess.Popen("cat "+ os.getenv("ANALYSIS_DIR") + "/ion_params_00.json", shell=True, stdout=subprocess.PIPE)
                word = sub1.stdout.read().strip()
                first_index = word.find("serial_number") + 22
                word2 = word[first_index:]
                end_index = word2.find("\\") 
                serial_number = word2[:end_index]
		#block = open(os.getenv("RESULTS_DIR") + "/serial.txt", "w")
		block = open(commonScratchDir + "/serial.txt", "w")
		block.write(serial_number)
		
	def copy_status_block(self):
		shutil.copyfile(os.getenv("RUNINFO__PLUGIN_DIR") + "/status_block.html", os.getenv("RESULTS_DIR") + "/progress.html" )

if __name__ == "__main__": PluginCLI()
'''
if __name__=="__main__":
  ## test code for plugin api calls
	iru = IonReporterUploader();
	print ""
	print ""
	print "getversions output = " + iru.get_versions()
	print ""
	print ""
	print ""
	print "getUserInput output = " 
	print iru.getUserInput()
	print ""
	print ""
	print "inc_submissionCount output = " 
	print iru.inc_submissionCounts()
	print ""
	print ""
'''




