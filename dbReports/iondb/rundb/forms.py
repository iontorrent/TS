# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import subprocess
import socket
import logging
import sys
from django import forms
from django.forms import widgets as djangoWidget
from django.forms.extras import widgets
from django.forms.widgets import PasswordInput
from django import shortcuts
from iondb.rundb import models
import datetime
from iondb.utils import devices
from iondb.rundb import tasks
from django.forms.util import flatatt
from django.utils.safestring import mark_safe
from django.utils.encoding import StrAndUnicode, force_unicode
from itertools import chain
from django.utils.html import escape, conditional_escape
import string
from django.conf import settings

logger = logging.getLogger(__name__)

from django.utils.safestring import mark_safe


class Plugins_SelectMultiple(forms.CheckboxSelectMultiple):

    def render(self, name, value, attrs=None):
        csm = super(Plugins_SelectMultiple, self).render(name, value, attrs=None)
        csm = csm.replace(u'<ul>', u'').replace(u'</ul>', u'').replace(u'<label>', u'').replace(u'</label>', u'')
        # add 'configure' button for plugins
        btn_html = '&nbsp; <button type="button" class="configure_plugin" id="configure_plugin_XXX" data-plugin_pk=XXX style="display: none;"> Configure </button>'
        output = ''
        columns = 3
        for line in csm.split('<li>'):
            if line.find('value="') > 0:
                columns -= 1
                if columns == 2:
                    output += '<div class="row-fluid">'
                output += '<div class="span4">' + line.split('</li>')[0]
                #output += line.split('</li>')[0]
                if columns == 0:
                    output += '</div>'
                    columns = 3

                pk = line.split('value="')[1].split('"')[0]
                plugin = models.Plugin.objects.get(pk=pk)
                if plugin.isPlanConfig:
                    output += btn_html.replace('XXX', pk)
                output += '</div>'
                # disable IRU if not configured
                if 'IonReporterUploader' == plugin.name:
                    if 'checked' in line:
                        output = output.replace('/> IonReporterUploader', '/><span rel="tooltip" title="Edit run to configure IonReporterUploader"> IonReporterUploader</span>')
                    else:
                        output = output.replace('/> IonReporterUploader', 'disabled /><span rel="tooltip" title="Edit run to configure IonReporterUploader"> IonReporterUploader not configured</span>')

        if columns != 3: output += '</div>'

        return mark_safe(output)


class DataSelect(djangoWidget.Widget):

    """this is added to be able to have data attribs to the options"""

    allow_multiple_selected = False

    def __init__(self, attrs=None, choices=()):
        super(DataSelect, self).__init__(attrs)
        # choices can be any iterable, but we may need to render this widget
        # multiple times. Thus, collapse it into a list so it can be consumed
        # more than once.
        self.choices = list(choices)

    def render(self, name, value, attrs=None, choices=()):
        if value is None: value = ''
        final_attrs = self.build_attrs(attrs, name=name)
        output = [u'<select%s>' % flatatt(final_attrs)]
        options = self.render_options(choices, [value])
        if options:
            output.append(options)
        output.append(u'</select>')
        return mark_safe(u'\n'.join(output))

    def render_option(self, selected_choices, option_value, option_label, option_pk, option_version):
        option_value = force_unicode(option_value)
        if option_value in selected_choices:
            selected_html = u' selected="selected"'
            if not self.allow_multiple_selected:
                # Only allow for a single selection.
                selected_choices.remove(option_value)
        else:
            selected_html = ''
        return u'<option data-pk="%s" data-version="%s" value="%s"%s>%s</option>' % (
            option_pk, option_version, escape(option_value), selected_html,
            conditional_escape(force_unicode(option_label)))

    def render_options(self, choices, selected_choices):
        # Normalize to strings.
        selected_choices = set(force_unicode(v) for v in selected_choices)
        output = []
        for option_value, option_label, option_pk, option_version in chain(self.choices, choices):
                output.append(self.render_option(selected_choices, option_value, option_label, option_pk, option_version))
        return u'\n'.join(output)


class CmdlineArgsField(forms.CharField):

    def __init__(self):
        super(CmdlineArgsField, self).__init__(
            max_length=1024,
            required=False,
            widget=forms.Textarea(attrs={'class': 'span12 args', 'rows': 4})
        )

    def clean(self, value):
        value = super(CmdlineArgsField, self).clean(value)
        if not set(value).issubset(string.printable):
            raise forms.ValidationError(("Command contains non ascii characters."))
        return value


class RunParamsForm(forms.Form):

    report_name = forms.CharField(max_length=128,
                                widget=forms.TextInput(attrs={'size': '60', 'class': 'textInput input-xlarge'}))

    beadfindArgs = CmdlineArgsField()
    analysisArgs = CmdlineArgsField()
    prebasecallerArgs = CmdlineArgsField()
    calibrateArgs = CmdlineArgsField()
    basecallerArgs = CmdlineArgsField()
    alignmentArgs = CmdlineArgsField()
    ionstatsArgs = CmdlineArgsField()

    thumbnailBeadfindArgs = CmdlineArgsField()
    thumbnailAnalysisArgs = CmdlineArgsField()
    prethumbnailBasecallerArgs = CmdlineArgsField()
    thumbnailCalibrateArgs = CmdlineArgsField()
    thumbnailBasecallerArgs = CmdlineArgsField()
    thumbnailAlignmentArgs = CmdlineArgsField()
    thumbnailIonstatsArgs = CmdlineArgsField()

    custom_args = forms.BooleanField(required=False, widget=forms.HiddenInput)

    blockArgs = forms.CharField(max_length=128, required=False, widget=forms.HiddenInput)

    libraryKey = forms.CharField(max_length=128, required=False, initial="TCAG", widget=forms.TextInput(attrs={'size': '60', 'class': 'input-xlarge'}))
    tfKey = forms.CharField(max_length=128, required=False, initial="ATCG", widget=forms.TextInput(attrs={'size': '60', 'class': 'input-xlarge'}))

    # unused?
    align_full = forms.BooleanField(required=False, initial=False)

    do_thumbnail = forms.BooleanField(required=False, initial=False, label="Thumbnail only")
    do_base_recal = forms.CharField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}), label="Base Recalibration Mode")

    realign = forms.BooleanField(required=False)
    mark_duplicates = forms.BooleanField(required=False, initial=False)

    previousReport = forms.CharField(required=False, widget=DataSelect(attrs={'class': 'input-xlarge'}))
    previousThumbReport = forms.CharField(required=False, widget=DataSelect(attrs={'class': 'input-xlarge'}))

    project_names = forms.CharField(max_length=1024,
                           required=False,
                           widget=forms.TextInput(attrs={'size': '60', 'class': 'textInput input-xlarge'}))

    def clean_report_name(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        reportName = self.cleaned_data.get("report_name")
        errors = []
        if reportName[0] == "-":
            errors.append(("The Report name can not begin with '-'"))
        if len(reportName) > 60:
            errors.append(("Report Name needs to be less than 60 characters long"))
        if not set(reportName).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
            errors.append(("That Report name has invalid characters. The valid values are letters, numbers, underscore and period."))

        if errors:
            raise forms.ValidationError(errors)
        else:
            return reportName

    def clean_libraryKey(self):
        key = self.cleaned_data.get('libraryKey')
        if not set(key).issubset("ATCG"):
            raise forms.ValidationError(("This key has invalid characters. The valid values are TACG."))
        else:
            return key

    def clean_tfKey(self):
        key = self.cleaned_data.get('tfKey')
        if not set(key).issubset("ATCG"):
            raise forms.ValidationError(("This key has invalid characters. The valid values are TACG."))
        else:
            return key

    def clean_project_names(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        projectNames = self.cleaned_data.get("project_names")
        names = []
        for name in projectNames.split(','):
            if name:
              names.append(name)
              if len(name) > 64:
                  raise forms.ValidationError(("Project Name needs to be less than 64 characters long. Please separate different projects with a comma."))
              if not set(name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
                  raise forms.ValidationError(("Project name has invalid characters. The valid values are letters, numbers, underscore and period."))
        return ','.join(names)



from iondb.rundb.plan.views_helper import dict_bed_hotspot


class AnalysisSettingsForm(forms.ModelForm):

    reference = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    targetRegionBedFile = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    hotSpotRegionBedFile = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    plugins = forms.ModelMultipleChoiceField(required=False, widget=Plugins_SelectMultiple(),
    queryset=models.Plugin.objects.filter(selected=True, active=True).order_by('name', '-version'))
    pluginsUserInput = forms.CharField(required=False, widget=forms.HiddenInput())
    #barcodeKitName = forms.CharField(required=False, max_length=128, widget=forms.TextInput(attrs={'class': 'input-xlarge', 'readonly':'true'}) )
    barcodeKitName = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))

    threePrimeAdapter = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    barcodedReferences = forms.CharField(required=False, widget=forms.HiddenInput())

    def __init__(self, *args, **kwargs):
        super(AnalysisSettingsForm, self).__init__(*args, **kwargs)
        # initialize choices when form instance created
        references = models.ReferenceGenome.objects.filter(index_version=settings.TMAP_VERSION, enabled=True)
        self.fields['reference'].choices = [('', 'none')] + [(v[0], "%s (%s)" % (v[0], v[1])) for v in references.values_list('short_name', 'name')]
        bedfiles = dict_bed_hotspot()
        self.fields['targetRegionBedFile'].choices = [('', '')] + [(v.file, v.path.split("/")[-1].replace(".bed", "")) for v in bedfiles.get('bedFiles', [])]
        self.fields['hotSpotRegionBedFile'].choices = [('', '')] + [(v.file, v.path.split("/")[-1].replace(".bed", "")) for v in bedfiles.get('hotspotFiles', [])]
        self.fields['barcodeKitName'].choices = [('', '')]+list(models.dnaBarcode.objects.order_by('name').distinct('name').values_list('name', 'name'))
        adapters = models.ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'chemistryType', 'name')
        self.fields['threePrimeAdapter'].choices = [(v[1], "%s (%s)" % (v[0], v[1])) for v in adapters.values_list('name', 'sequence')]

    class Meta:
        model = models.ExperimentAnalysisSettings
        fields = ('reference', 'targetRegionBedFile', 'hotSpotRegionBedFile', 'barcodeKitName', 'threePrimeAdapter')


class ExperimentSettingsForm(forms.ModelForm):

    sample = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    barcodedSamples = forms.CharField(required=False, widget=forms.HiddenInput())
    runtype = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    libraryKitname = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    sequencekitname = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    chipBarcode = forms.CharField(required=False, max_length=64, widget=forms.TextInput(attrs={'class': 'textInput input-xlarge validateAlphaNumNoSpace'}))
    libraryKey = forms.ChoiceField(required=False, widget=forms.Select(attrs={'class': 'input-xlarge'}))
    notes = forms.CharField(required=False, max_length=128, widget=forms.TextInput(attrs={'class': 'textInput input-xlarge'}))
    mark_duplicates = forms.BooleanField(required=False, initial=False)
    sampleTubeLabel = forms.CharField(required=False, max_length=512, widget=forms.TextInput(attrs={'class': 'textInput input-xlarge'}))

    def __init__(self, *args, **kwargs):
        super(ExperimentSettingsForm, self).__init__(*args, **kwargs)
        # initialize sample and key choices when form instance created
        self.fields['sample'].choices = [('', '')] + list(models.Sample.objects.filter().values_list('id', 'displayedName'))
        self.fields['libraryKey'].choices = [(v[0], "%s (%s)" % (v[0], v[1])) for v in models.LibraryKey.objects.filter().values_list('sequence', 'description')]
        self.fields['runtype'].choices = [(v[0], "%s (%s)" % (v[1], v[0])) for v in models.RunType.objects.all().order_by("id").values_list('runType', 'description')]
        self.fields['libraryKitname'].choices = [('', '')] + list(models.KitInfo.objects.filter(kitType='LibraryKit').values_list('name', 'name'))
        self.fields['sequencekitname'].choices = [('', '')]+list(models.KitInfo.objects.filter(kitType='SequencingKit').values_list('name', 'name'))

    class Meta:
        model = models.Experiment
        fields = ('sample', 'sequencekitname', 'chipBarcode', 'notes')


class EmailAddress(forms.ModelForm):

    "Made to have full symetry with the EmailAddress model fields"
    class Meta:
        model = models.EmailAddress


class EditReferenceGenome(forms.Form):
    name = forms.CharField(max_length=512, required=True, label="Short Name")
    version = forms.CharField(max_length=100, required=False, label="Version")
    NCBI_name = forms.CharField(max_length=512, required=True, label="Description")
    notes = forms.CharField(max_length=1048, required=False, widget=forms.Textarea(attrs={'cols': 50, 'rows': 4}))
    enabled = forms.BooleanField(required=False)
    genome_key = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    index_version = forms.CharField(widget=forms.HiddenInput(), required=True)

    def clean_name(self):
        """don't allow duplicate names
        make an exception for the genome we are working with
        """

        index_version = self.data.getlist("index_version")[0]
        genome_key = self.data.getlist("genome_key")[0]
        get_name = self.cleaned_data.get('name')
        genomes = models.ReferenceGenome.objects.filter(short_name=get_name).exclude(pk=genome_key)

        if not set(get_name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"):
            raise forms.ValidationError(("The short name has invalid characters. The valid values are letters, numbers, and underscores."))

        for g in genomes:
            if get_name == g.name and g.index_version == index_version:
                error_str = "A reference with the name" + get_name + " and index version" + index_version + " already exists"
                raise forms.ValidationError(error_str)
        return get_name


class UserProfileForm(forms.ModelForm):
    # validate name length against the longest possible name
    #  Source: http://www.independent.co.uk/news/uk/this-britain/captain-fantastic-claims-worlds-longest-name-993957.html
    # Captain Fantastic Faster Than Superman Spiderman Batman Wolverine Hulk And The Flash Combined
    name = forms.CharField(max_length=93)
    email = forms.EmailField()

    def __init__(self, *args, **kw):
        super(UserProfileForm, self).__init__(*args, **kw)
        self.fields['email'].initial = self.instance.user.email

        self.fields.keyOrder = ['name', 'email', 'phone_number']

    def save(self, *args, **kw):
        super(UserProfileForm, self).save(*args, **kw)
        self.instance.user.email = self.cleaned_data.get('email')
        self.instance.user.save()

    class Meta:
        model = models.UserProfile


class NetworkConfigForm(forms.Form):
    modes = (("dhcp", "DHCP",), ("static", "Static"))
    mode = forms.ChoiceField(widget=forms.widgets.RadioSelect, choices=modes)
    address = forms.IPAddressField(label="IP Address", required=False)
    subnet = forms.IPAddressField(required=False, label="Subnet")
    gateway = forms.IPAddressField(required=False)
    nameservers = forms.CharField(required=False, max_length=256)
    dnssearch = forms.CharField(required=False, max_length=256, label="Search Domain")
    proxy_address = forms.CharField(required=False, max_length=256)
    proxy_port = forms.CharField(required=False)
    proxy_username = forms.CharField(required=False)
    proxy_password = forms.CharField(required=False)
    no_proxy = forms.CharField(required=False, max_length=256, label="Set no_proxy")
    default_no_proxy = settings.DEFAULT_NO_PROXY
    
    def get_network_settings(self):
        """Usage: /usr/sbin/TSquery [option]...
        --eth-dev                 Specify eth device to query
        --debug, -d               Prints script commands when executing (set -x)
        --help, -h                Prints command line args
        --version, -v             Prints version

        Outputs:
            proxy_address:
            proxy_port:
            proxy_username:
            proxy_password:
            no_proxy:
            network_device:
            network_mode:
            network_address:10.25.3.211
            network_subnet:255.255.254.0
            network_gateway:10.25.3.1
            network_nameservers:10.25.3.2,10.45.16.11
            network_dnssearch:ite,itw,cbd
        """
        settings = {
                "mode": "",
                "address": "",
                "subnet": "",
                "gateway": "",
                "nameservers": "",
                "dnssearch": "",
                "proxy_address": "",
                "proxy_port": "",
                "proxy_username": "",
                "proxy_password": "",
                "no_proxy": "",
                }
        cmd = ["/usr/sbin/TSquery"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            out_lines = stdout.split("\n")
            formatted_output = "\n".join("    %s" % l for l in out_lines)
            logger.info("TSquery output:\n%s" % formatted_output)
            for line in out_lines:
                if line:
                    key, value = line.split(":", 1)
                    if key.startswith("network_"):
                        key = key.replace("network_", '', 1)
                    settings[key] = value
        except Exception as error:
            logger.error("When attempting to run TSquery:\n%s" % error)
        return settings

    def new_config(self, one, two, keys):
        return not all(k in one and k in two and one[k] == two[k] for k in keys)

    def set_to_current_values(self):
        settings = self.get_network_settings()
        self.fields['mode'].initial = settings["mode"]
        self.fields['address'].initial = settings["address"]
        self.fields['subnet'].initial = settings["subnet"]
        self.fields['gateway'].initial = settings["gateway"]
        self.fields['nameservers'].initial = settings["nameservers"]
        self.fields['dnssearch'].initial = settings["dnssearch"]
        self.fields['proxy_address'].initial = settings["proxy_address"]
        self.fields['proxy_port'].initial = settings["proxy_port"]
        self.fields['proxy_username'].initial = settings["proxy_username"]
        self.fields['proxy_password'].initial = settings["proxy_password"]
        self.fields['no_proxy'].initial = settings["no_proxy"]

    def __init__(self, *args, **kw):
        super(NetworkConfigForm, self).__init__(*args, **kw)
        self.set_to_current_values()

    def save(self, *args, **kw):

        def ax_proxy():
            """
            Helper method for TSsetproxy script which will automatically set the "--remove" argument
            """
            cmd = ["sudo", "/usr/sbin/TSsetproxy", "--remove"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if stderr:
                logger.warning("Network error: %s" % stderr)

        def proxyconf(address, port, username, password):
            """
            Helper method for TSsetproxy script
            :param address:  --address     Proxy address (http://proxy.net)
            :param port:     --port        Proxy port number
            :param username: --username    Username for authentication
            :param password: --password    Password for authentication
            """
            cmd = ["sudo", "/usr/sbin/TSsetproxy", "--address", address, "--port", port, "--username", username, "--password", password]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if stderr:
                logger.warning("Network error: %s" % stderr)

        def dhcp():
            """
            Helper method to call into the TSstaticip script with the "remove" option to revert back to dhcp
            """
            cmd = ["sudo", "/usr/sbin/TSstaticip", "--remove"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if stderr:
                logger.warning("Network error: %s" % stderr)

        def static_ip(address, subnet, gateway, nameserver=None, search=None):
            """
            Helper method to call into the TSstaticip script
            :param address:    --ip         Define host IP address
            :param subnet:     --nm         Define subnet mask (netmask)
            :param gateway:    --gw         Define gateway/router IP address
            :param nameserver: --nameserver Specify one or more nameserver IP addresses
            :param search:     --search     Specify one or more search domains
            """
            cmd = ["sudo", "/usr/sbin/TSstaticip", "--ip", address, "--nm", subnet,"--gw", gateway]
            if nameserver:
                cmd += ["--nameserver", nameserver]
            if search:
                cmd += ["--search", search]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if stderr:
                logger.warning("Network error: %s" % stderr)

        network_settings = self.get_network_settings()
        host_config = ["mode", "address", "subnet", "gateway"]
        if self.new_config(self.cleaned_data, network_settings, host_config):
            if self.cleaned_data['mode'] == "dhcp":
                dhcp()
            elif self.cleaned_data['mode'] == "static":
                address = self.cleaned_data['address']
                subnet = self.cleaned_data['subnet']
                gateway = self.cleaned_data['gateway']
                nameservers = None
                dnssearch = None

                if self.new_config(self.cleaned_data, network_settings, ["nameservers", "dnssearch"]):
                    logger.info("User changed the DNS and host network settings.")
                    if self.cleaned_data['nameservers']:
                        nameservers = self.cleaned_data['nameservers']

                    if self.cleaned_data['dnssearch']:
                        dnssearch = self.cleaned_data['dnssearch']

                logger.info("User changed the host network settings.")
                static_ip(address, subnet, gateway, nameserver=nameservers, search=dnssearch)
        else:
            logger.info("new_config failed to pass")

        proxy_config = ["proxy_address", "proxy_port", "proxy_username", "proxy_password"]
        if self.new_config(self.cleaned_data, network_settings, proxy_config):
            logger.info("User changed the proxy settings.")
            if self.cleaned_data['proxy_address'] and self.cleaned_data['proxy_port']:
                address = self.cleaned_data['proxy_address']
                port = self.cleaned_data['proxy_port']
                user = self.cleaned_data['proxy_username']
                password = self.cleaned_data['proxy_password']
                proxyconf(address, port, user, password)
            else:
                ax_proxy()

        self.set_to_current_values()


class AmpliseqLogin(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=PasswordInput)
