# pylint: disable=line-too-long, no-self-use, broad-except, bare-except
"""
    pileup.py
    """
import sys
#import resource
import json
import time
import datetime
import subprocess
import os
from optparse import OptionParser

class Run(object):
    """ Run """
    def __init__(self, version, argv):
        start_time = time.time()
        if len(argv) == 0:
            sys.argv.append("--help")
        self.version = version
        self.parser = OptionParser("usage: %prog [options]", version="%prog " + version)
        self.options = None
        self.parameters = {}
        self.add_option("-p", "--parameters-file", "string", "Parameter input in json format")
        self.add_options()
        self.parse_args()

        self.override_options()
        self.validate_options()
        if 'output_dir' in self.parameters:
            if not os.path.isdir(self.get_parameter('output_dir')):
                os.makedirs(self.get_parameter('output_dir'))

        self.printtime(sys.argv[0] + "\tversion " + self.version + "\t" + str(datetime.datetime.now()))
        self.printtime("")
        cmd = sys.argv[0]
        for arg in argv:
            cmd += " " + arg
        self.printtime(cmd)
        self.printtime("")
        for key in self.parameters:
            self.printtime
            (key + " " + str(self.parameters[key]))
        self.printtime("")
        try:
            self.process()
        except Exception as exception:
            try:
                self.fatal_error(exception.message)
            except:
                self.fatal_error("An unknown excpetion has occurred")

        end_time = time.time()
        self.printtime("")
        self.printtime("run time = " + str(int(end_time - start_time)) + " seconds")

    def add_options(self):
        """ add_option """
        raise NotImplementedError("Subclass must implement abstract method: add_options")

    def override_options(self):
        """ override_options """
        raise NotImplementedError("Subclass must implement abstract method: override_options")

    def validate_options(self):
        """ validate_options """
        raise NotImplementedError("Subclass must implement abstract method: validate_options")

    def process(self):
        """ process """
        raise NotImplementedError("Subclass must implement abstract method: process")

    def add_option(self, short_name, long_name, type_in, help_in):
        """ add_option """
        dest_in = long_name
        if dest_in.startswith("--"):
            dest_in = dest_in[2:]
        dest_in = dest_in.replace("-", "_")
        self.parser.add_option(short_name, long_name, action="store", type=type_in, dest=dest_in, help=help_in)

    def mem_usage(self):
        """ Reports memory usage """
        usage = resource.getrusage(resource.RUSAGE_SELF)
        byte_count = usage[2]*resource.getpagesize()
        self.printtime("Memory Usage " + str(round((byte_count/1000000.0), 2)) + " MB")

    def printtime(self, message, *args):
        """ print messages """
        if args:
            message = message % args
        print "[ " + time.strftime('%X') + " ] " + message
        sys.stdout.flush()
        sys.stderr.flush()

    def error(self, message):
        """ Logs errors """
        self.printtime("Error: " + message + "\n")

    def fatal_error(self, message):
        """ Logs fatal errors """
        self.printtime("Fatal Error: " + message + "\n")
        sys.exit(1)

    def get_parameter(self, key):
        """ Get a parameter value """
        if key in self.parameters:
            return self.parameters[key]
        else:
            self.fatal_error("Cannot find '" + key + "' in parameters.")
        return ""

    def run_command(self, command, description):
        """ Runs a command """
        self.printtime(' ')
        self.printtime('Task    : ' + description)
        self.printtime('Command : ' + command)
        self.printtime(' ')
        cmd = command.split(" ")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, err) = proc.communicate()
        returncode = proc.returncode
        return (returncode, out, err)

    def parse_args(self):
        """ parse_args """
        self.options = self.parser.parse_args()[0]

        if self.options.parameters_file:
            try:
                json_file_in = open(self.options.parameters_file, 'r')
                json_in = json.load(json_file_in, parse_float=str)
                json_file_in.close()
                self.parameters['parameters-file'] = self.options.parameters_file
                if 'parameters' in json_in:
                    for parameter in json_in['parameters']:
                        self.parameters[parameter['parameter_name']] = parameter['value']
                for key in json_in:
                    if key == 'parameters':
                        for parameter in json_in[key]:
                            self.parameters[parameter['parameter_name']] = parameter['value']
                    elif key == 'input_bams':
                        self.parameters[key] = json_in[key]
                    else:
                        self.parameters[key] = json_in[key]
            except IOError:
                self.fatal_error('Failed to load and parse ' + os.path.basename(self.options.parameters_file))

