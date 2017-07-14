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

from ion.plugin import PluginCLI, IonPlugin, RunLevel, RunType
from ion.utils import makeCSA


class FieldSupport(IonPlugin):
    """Generate an enhanced CSA"""
    version = '5.4.0.3'
    runtypes = [RunType.FULLCHIP]
    runlevels = [RunLevel.LAST]

    plugin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

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
        url = urlparse.urljoin(self.start_plugin['runinfo']['api_url'], '/report/latex/%s.pdf/' % pk)
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

    def run_rndplugin(self, plugin_name):
        plugin_dir = os.path.join(self.plugin_dir, "rndplugins", plugin_name)
        output_dir = os.path.join(self.start_plugin["runinfo"]["results_dir"], "FieldSupport", plugin_name)
        os.mkdir(output_dir)
        env = {
            "DIRNAME": plugin_dir,
            "SIGPROC_DIR": self.start_plugin["runinfo"]["sigproc_dir"],
            "TSP_CHIPTYPE": self.start_plugin["runinfo"]["chipType"],
            "RAW_DATA_DIR": self.start_plugin["runinfo"]["raw_data_dir"],
            "TSP_ANALYSIS_NAME": self.start_plugin["expmeta"]["results_name"],
            "TSP_FILEPATH_PLUGIN_DIR": output_dir,
            "TSP_LIMIT_OUTPUT": "1"  # Tells plugins they are being run by FieldSupport instead of the pipeline
        }
        subprocess.check_output(["bash", "launch.sh"], cwd=plugin_dir, env=env)

    def launch(self):
        self.log.info("Launching Field Support.")

        with open('startplugin.json', 'r') as start_plugin_file:
            self.start_plugin = json.load(start_plugin_file)

        # Exit early if this is not a thumbnail run
        if self.start_plugin["runplugin"]["run_type"] != "thumbnail":
            self.state["warning"] = "This plugin can only be run on thumbnail reports. " \
                                    "Please rerun this plugin on this run's thumbnail report."
            self.write_status()
            self.log.info("Field Support Aborted.")
            return False

        self.state["progress"] = 10
        self.write_status()

        results_dir = self.start_plugin['runinfo']['results_dir']
        zip_name = self.start_plugin["expmeta"]["results_name"] + ".FieldSupport.zip"
        zip_path = os.path.join(results_dir, zip_name)

        # Make CSA zip using pipeline utils makeCSA
        makeCSA.makeCSA(
            self.start_plugin["runinfo"]["report_root_dir"],
            self.start_plugin["runinfo"]["raw_data_dir"],
            zip_path
        )

        self.state["progress"] = 30
        self.write_status()

        os.mkdir(os.path.join(results_dir, "FieldSupport"))

        # Now run each rndplugin
        for name, options in self.plugin_options.items():
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
            for name, options in self.plugin_options.items():
                for root, _, file_names in os.walk(os.path.join(results_dir, "FieldSupport", name)):
                    for pattern in options["files"]:
                        for file_name in fnmatch.filter(file_names, pattern):
                            f.write(os.path.join(root, file_name), os.path.join("FieldSupport", name, file_name))

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
        shutil.rmtree(temp_dir)
        os.unlink(zip_path)

        # Link up the zip
        self.state["download_link"] = tar_name

        self.state["progress"] = 100
        self.write_status()

        self.log.info("Field Support Complete.")
        return True


if __name__ == "__main__":
    PluginCLI(FieldSupport())
