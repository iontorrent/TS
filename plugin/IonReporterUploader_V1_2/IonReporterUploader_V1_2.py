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


####### 
###  global variables 
pluginName = ""
plugin_dir = ""

class IonReporterUploader_V1_2(IonPlugin):
	version = "3.2.0-r%s" % filter(str.isdigit,"$Revision: 51281 $")
	runtypes = [ RunType.THUMB, RunType.FULLCHIP, RunType.COMPOSITE ]
	runlevels = [ RunLevel.PRE, RunLevel.BLOCK, RunLevel.POST ]
	features = [ Feature.EXPORT ]

	global pluginName , plugin_dir, launchOption, commonScratchDir
	pluginName = "IonReporterUploader_V1_2"
	plugin_dir = os.getenv("RUNINFO__PLUGIN_DIR") or os.path.dirname(__file__)
	launchOption = "upload_and_launch"

	#print "plugin dir is " + os.getenv("RUNINFO__PLUGIN_DIR") 
	#print "plugin name is " + os.getenv("RUNINFO__PLUGIN_NAME") 
	#print "plugin runtime key is " + os.getenv("RUNINFO__PK") 

	def pre_launch(self):
		return True
	
	#Launches script with parameter jsonfilename. Assumes data is a dict
	def launch(self,data=None):
		global launchOption, commonScratchDir
		print pluginName + ".py launch()"
		data = open(os.getenv("RESULTS_DIR") + "/startplugin.json").read()
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
								self.write_log(timestamp + " IonReporterUploader_V1_2", data)
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
		server = config["server"]
		token = config["token"]
		protocol = config["protocol"]
		port = config["port"]
		# Srikanth Jandhyala 
		try:
			url = protocol + "://"+server+":"+port+"/grws_1_2/data/versionList/"
			versions_list_req = urllib2.Request(url)
			versions_list_req.add_header("Authorization",token)
			versions_list_fp = urllib2.urlopen(versions_list_req, None,90)
			versions_list_data = versions_list_fp.read().strip()
		except urllib2.HTTPError, e:
			raise Exception("Error Code " + str(e.code))
		except urllib2.URLError, e:
			raise Exception("Error Reason " + e.reason.args[1])

		return versions_list_data


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
		server = config["server"]
		token = config["token"]
		protocol = config["protocol"]
		port = config["port"]
		version = config["version"]
		version = version.split("IR")[1]

		try:
			url = protocol + "://"+server+":"+port+"/grws_1_2/data/workflowList/"
			sample_relationship_req = urllib2.Request(url)
			sample_relationship_req.add_header("Authorization",token)
			sample_relationship_req.add_header("Version", version)

			sample_relationship_fp = urllib2.urlopen(sample_relationship_req,None,90)
			sample_relationship_data = sample_relationship_fp.read().strip()
		except urllib2.HTTPError, e:
				raise Exception("Error Code " + str(e.code))
		except urllib2.URLError, e:
			raise Exception(" Error Reason " + e.reason.args[1])

		# Srikanth Jandhyala
		columnsMapStr = sample_relationship_data.strip()
		columnsMapList = eval(columnsMapStr)

		sampleRelationshipDict = {}
		sampleRelationshipDict["column-map"] = columnsMapList
		sampleRelationshipDict["columns"] = []

		relationshipTypeDict = {"Name":"RelationshipType", "Order":"3", "Type":"list","ValueType": "String", "Values":["Self","Tumor_Normal","Trio"]}
		setIDDict = {"Name":"SetID", "Order":"4", "Type":"input", "ValueType":"Integer"}
		relationDict = {"Name": "Relation", "Order":"5","Type":"list","ValueType":"String","Values":["Tumor","Normal","Father","Mother","Self"]}

		workflowDict = {"Name":"Workflow", "order":"2", "Type":"list", "ValueType":"String"}
		genderDict = {"Name":"Gender", "order":"1", "Type":"list", "ValueType":"String","Values":["Male","Female","Unknown"]}
		workflowDictValues = []
		for entry in columnsMapList:
			workflowName = entry["Workflow"]
			workflowDictValues.append(workflowName)
		workflowDict["Values"] = workflowDictValues

		sampleRelationshipDict["columns"].append(genderDict)
		sampleRelationshipDict["columns"].append(workflowDict)
		sampleRelationshipDict["columns"].append(relationshipTypeDict)
		sampleRelationshipDict["columns"].append(setIDDict)
		sampleRelationshipDict["columns"].append(relationDict)

		restrictionRulesList = []
		restrictionRulesList.append({"For":{"Name": "ApplicationType", "Value":"Tumor Normal Sequencing"}, "Valid":{"Name":"RelationshipType", "Values":["Tumor_Normal"]}})
		restrictionRulesList.append({"For":{"Name": "RelationshipType", "Value":"Tumor_Normal"}, "Valid":{"Name":"Relation", "Values":["Tumor", "Normal"]}})
		restrictionRulesList.append({"For":{"Name": "RelationshipType", "Value":"Trio"}, "Valid":{"Name":"Relation", "Values":["Father", "Mother", "Self"]}})
		restrictionRulesList.append({"For":{"Name": "RelationShipType", "Value":"Self"}, "Disabled":{"Name":"SetID"}})
		restrictionRulesList.append({"For":{"Name": "RelationShipType", "Value":"Self"}, "Disabled":{"Name":"Relation"}})
		sampleRelationshipDict["restrictionRules"] = restrictionRulesList
		#print sampleRelationshipDict
		return sampleRelationshipDict


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
		url = self.get_runinfo("url_root", data)
		report = "/Default_Report.php?do_print=true"
		
		try1 = "http://127.0.0.1/" + url + report
		try2 = net + url + report
	
		try:
			urllib2.urlopen(try1)
		except urllib2.URLError, e:
			self.write_log("Report Generation (report.pdf) failed", data) 
		#urllib2.urlopen(try2) #This query asks for a system username/password... so skip it


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
		file.write("\n")
		file.write(text)
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
	iru = IonReporterUploader_V1_2();
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




