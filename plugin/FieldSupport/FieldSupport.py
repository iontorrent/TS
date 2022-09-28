#!/usr/bin/python
# Copyright 2017 Thermo Fisher Scientific. All Rights Reserved.

import os
import json
import zipfile
import requests
import urlparse
import string
import subprocess
import fnmatch
import shutil

from subprocess import *
from ion.plugin import PluginCLI, IonPlugin, RunLevel, RunType
import makeCSA


class FieldSupport(IonPlugin):
    """Generate an enhanced CSA"""
    version = '5.18.1.0'
    runtypes = [RunType.THUMB, RunType.FULLCHIP]
    runlevels = [RunLevel.LAST]

    plugin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    start_plugin = dict()
    template = None
    state = {
        "warning": None,
        "progress": 0,
        "download_link": None
    }

    plugin_options = {
        "bubblePlots": {
            "files": [
                "*outlier-by-flow.png",
                "*outlier-all.png",
                "*.csv",
                "*.gif",
                "*.html",
                "*.log"
            ]
        }
    }

    plugin_options_fc = {
        "GBU_HBU_Analysis": {
            "files": [
                "*.png",
                "*.xls",
                "*.json",
                "*.csv",
                "*.txt",
                "*.html",
                "*.css",
                "*.js",
                "*flot",
                "*lifechart",
                "*slickgrid",
                "*.log"
            ]
        }
    }

    def write_status(self):
        if not self.template:
            with open(os.path.join(self.plugin_dir, "templates", "status_block.html")) as status_template:
                self.template = string.Template(status_template.read())
        with open(os.path.join(self.start_plugin['runinfo']['results_dir'], "status_block.html"), "w") as status_file:
            # Map state to template values here
            context = {
                "bar_width": self.state["progress"],
                "progress_class": "",
                "button_text": "",
                "warning_text": "",
                "button_class": "btn-primary",
                "download_url": "#",
                "enable_refresh": "false",
                "warning_row_class": "hide",
                "progress_row_class": "show"
            }
            if self.state["warning"]:
                context["warning_text"] = self.state["warning"]
                context["warning_row_class"] = "show"
                context["progress_row_class"] = "hide"
            else:
                # In progress
                if not self.state["download_link"]:
                    context["progress_class"] = "progress-striped active"
                    context["button_text"] = "Generating Support Archive"
                    context["button_class"] = "disabled"
                    context["enable_refresh"] = "true"
                # Done
                else:
                    context["button_text"] = "Download Support Archive"
                    context["download_url"] = self.state["download_link"]
            status_file.write(self.template.substitute(**context))

    def fetch_api_resource(self, url, params={}):
        url = urlparse.urljoin(self.start_plugin['runinfo']['api_url'], url)
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_pdf(self, pk):
        url = urlparse.urljoin(self.start_plugin['runinfo']['api_url'], '/rundb/api/v1/results/%s/report/' % pk)
        params = {
            "api_key": self.start_plugin['runinfo']['api_key'],
            "pluginresult": self.start_plugin['runinfo']['pluginresult']
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.content

    def fetch_fullchip_report_pdf(self):
        # We need a pk of a result that's not a thumbnail
        results_resource = self.fetch_api_resource(
            "/rundb/api/v1/results/%i/" % int(self.start_plugin['runinfo']['pk'])
        )
        experiment_pk = int(results_resource["experiment"].strip("/").split("/")[-1])  # Extract url from pk
        related_results_resources = self.fetch_api_resource("/rundb/api/v1/results/", params={
            "experiment": experiment_pk,
            "order_by": "-timeStamp",
            "limit": "0"
        })
        for resource in related_results_resources["objects"]:
            if "thumb" not in resource["metaData"]:
                return self.fetch_pdf(resource["id"])
        return None

    def fetch_thumbnail_report_pdf(self):
        return self.fetch_pdf(self.start_plugin['runinfo']['pk'])

    def edit_startpluginjson(self, starpluginfile, dependent, plugin_name):
        with open(starpluginfile, 'r+') as start_plugin_file:
            start_plugin = json.load(start_plugin_file)
            plugin_result_base =  os.path.dirname(start_plugin["runinfo"]["results_dir"])
            all_subdirs = []
            for d in os.listdir(plugin_result_base):
                if os.path.isdir(os.path.join(plugin_result_base, d)) and dependent in d:
                    all_subdirs.append(os.path.join(plugin_result_base, d))
            latest_subdir = max(all_subdirs, key=os.path.getmtime)
            start_plugin["pluginconfig"]["coverage_analysis_path"] = str(latest_subdir) + '/'
            start_plugin["pluginconfig"]["dup_resolve"] = "Mean"
            start_plugin["pluginconfig"]["launch_mode"] = "Manual"
            results_dir = self.start_plugin['runinfo']['results_dir']
            rdir = os.path.join(results_dir, 'FieldSupport',plugin_name)
            pdir = os.path.join(self.plugin_dir, "rndplugins", plugin_name)
            start_plugin['runinfo']['results_dir'] = rdir
            start_plugin['runinfo']['plugin_dir'] = pdir
            start_plugin['runinfo']['plugin_name'] = plugin_name
            start_plugin['runinfo']['plugin']['depends'] = dependent
            start_plugin['runinfo']['plugin']['name'] = plugin_name
            start_plugin['runinfo']['plugin']['path'] = pdir
            start_plugin['depends'] = {}
            start_plugin['depends']['coverageAnalysis'] = {}
            start_plugin['depends']['coverageAnalysis']['pluginresult_path'] = start_plugin["pluginconfig"]["coverage_analysis_path"] 
            start_plugin_file.seek(0)
            json.dump(start_plugin, start_plugin_file, indent=2,sort_keys=True)
            

    def run_rndplugin(self, plugin_name):
        plugin_dir = os.path.join(self.plugin_dir, "rndplugins", plugin_name)
        output_dir = os.path.join(self.start_plugin["runinfo"]["results_dir"], "FieldSupport", plugin_name)
        os.mkdir(output_dir)
        print(output_dir)
        env = {
            "DIRNAME": plugin_dir,
            "SIGPROC_DIR": self.start_plugin["runinfo"]["sigproc_dir"],
            "TSP_CHIPTYPE": self.start_plugin["runinfo"]["chipType"],
            "RAW_DATA_DIR": self.start_plugin["runinfo"]["raw_data_dir"],
            "TSP_ANALYSIS_NAME": self.start_plugin["expmeta"]["results_name"],
            "TSP_FILEPATH_PLUGIN_DIR": output_dir,
            "TSP_LIMIT_OUTPUT": "1"  # Tells plugins they are being run by FieldSupport instead of the pipeline
        }

        if plugin_name in ('GBU_HBU_Analysis') and not self.thumbnail:
            print("coping barcodes.json and startplugin.json")
            p = Popen(["cp", os.path.join(self.start_plugin["runinfo"]["results_dir"], "barcodes.json"), output_dir])
            output = p.communicate()[0]
            p = Popen(["cp", os.path.join(self.start_plugin["runinfo"]["results_dir"], "startplugin.json"), output_dir])
            output = p.communicate()[0]
            self.edit_startpluginjson(os.path.join(output_dir, "startplugin.json"), "coverageAnalysis", "GBU_HBU_Analysis")

            version = '5.10.0.0'
            plugin = Popen([
            '%s/GBU_HBU_Analysis_plugin.py' % plugin_dir, '-V', version,
            os.path.join(output_dir, 'startplugin.json'), os.path.join(output_dir,'barcodes.json') ], stdout=PIPE, shell=False )
            print(plugin.communicate()[0])

        else:
            subprocess.check_output(["bash", "launch.sh"], cwd=plugin_dir, env=env)

    def launch(self, data=None):
        self.log.info("Launching Field Support.")
        self.thumbnail = True
        with open('startplugin.json', 'r') as start_plugin_file:
            self.start_plugin = json.load(start_plugin_file)

        # Exit early if this is not a thumbnail run
        if self.start_plugin["runplugin"]["run_type"] != "thumbnail" and self.start_plugin['runinfo']['platform'] != "pgm":
            # self.state["warning"] = "This plugin can only be run on thumbnail or PGM reports. " \
            #                        "Please rerun this plugin on this run's thumbnail report."
            # self.write_status()
            # self.log.info("Field Support Aborted.")
            # return False
            self.thumbnail = False

        self.state["progress"] = 10
        self.write_status()

        results_dir = self.start_plugin['runinfo']['results_dir']
        zip_name = self.start_plugin["expmeta"]["results_name"] + ".FieldSupport.zip"
        zip_path = os.path.join(results_dir, zip_name)

        # Make CSA zip using pipeline utils makeCSA
        try:
            makeCSA.makeCSA(
                self.start_plugin["runinfo"]["report_root_dir"],
                self.start_plugin["runinfo"]["raw_data_dir"],
                zip_path,
                self.start_plugin.get('chefSummary', dict()).get('chefLogPath', '')
            )
        except IOError as e:
            self.log.info("I/O error({0}): {1}".format(e.errno, e.strerror))
        except Exception as e:
            self.log.info("Unknown Exception:", e)

        self.state["progress"] = 30
        self.write_status()

        os.mkdir(os.path.join(results_dir, "FieldSupport"))
        print(os.path.join(results_dir, "FieldSupport"))

        # Now run each rndplugin
        if self.thumbnail:
            for name, options in self.plugin_options.items():
                self.run_rndplugin(name)
        else:
            for name, options in self.plugin_options_fc.items():
                self.run_rndplugin(name)

        self.state["progress"] = 70
        self.write_status()

        # Modify zip archive to include extra files
        with zipfile.ZipFile(zip_path, mode='a', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as f:
            # Write indicator
            f.writestr('FieldSupport/version', self.version)

            # Write the thumbnail report pdf if it is not present
            if "report.pdf" not in f.namelist():
                pdf_content = self.fetch_thumbnail_report_pdf()
                if pdf_content:
                    f.writestr('report.pdf', pdf_content)

            # Now we need to find the pk of a non thumbnail report
            if "full_report.pdf" not in f.namelist():
                try:
                    pdf_content = self.fetch_fullchip_report_pdf()
                    if pdf_content:
                        f.writestr('full_report.pdf', pdf_content)
                except Exception as e:
                    self.log.info("Failed to fetch full chip report. This will always fail on clusters.")
                    self.log.exception(e)

            # Add rndplugin files
            if self.thumbnail:
                for name, options in self.plugin_options.items():
                    for root, _, file_names in os.walk(os.path.join(results_dir, "FieldSupport", name)):
                        for pattern in options["files"]:
                            for file_name in fnmatch.filter(file_names, pattern):
                                f.write(os.path.join(root, file_name), os.path.join("FieldSupport", name, file_name))
            else: # GBU
                relroot = os.path.abspath(os.path.join(os.path.join(results_dir, "FieldSupport"), os.pardir))
                for name, options in self.plugin_options_fc.items():
                    for root, _, file_names in os.walk(os.path.join(results_dir, "FieldSupport", name), followlinks=True):
                        f.write(root, os.path.relpath(root, relroot))
                        for pattern in options["files"]:
                            for file_name in fnmatch.filter(file_names, pattern):
                                arcname = os.path.join(os.path.relpath(root, relroot), file_name)
                                f.write(os.path.join(root, file_name), arcname)
                                # if os.path.basename(root) not in name:
                                #     f.write(os.path.join(root, file_name), os.path.join("FieldSupport", name, os.path.basename(root), file_name))
                                # else:
                                #     f.write(os.path.join(root, file_name), os.path.join("FieldSupport", name, file_name))

        # Remove rndplugins output
        shutil.rmtree(os.path.join(results_dir, "FieldSupport"))

        self.state["progress"] = 90
        self.write_status()

        # Convert zip archive to tar.xz
        tar_name = self.start_plugin["expmeta"]["results_name"][0:(128 - 20)] + ".FieldSupport.tar.xz"
        temp_dir = os.path.join(results_dir, "temp")

        subprocess.check_call(["unzip", "-q", zip_path, "-d", temp_dir])
        subprocess.check_call(["tar", "cfJ", tar_name, "-C", temp_dir, "."], env={"XZ_OPT": "-9"})

        # Remove temp dir and zip archive
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        os.unlink(zip_path)

        # Link up the zip
        self.state["download_link"] = tar_name

        self.state["progress"] = 100
        self.write_status()

        self.log.info("Field Support Complete.")
        return True


if __name__ == "__main__":
    PluginCLI(FieldSupport())
