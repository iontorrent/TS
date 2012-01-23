# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import subprocess
import socket
import logging
from django import forms
from django.forms.extras import widgets
from iondb.rundb import models
import datetime
from iondb.backup import devices
from iondb.rundb import tasks

logger = logging.getLogger(__file__)


class RunParamsForm(forms.Form):
    report_name = forms.CharField(max_length=128)
    path = forms.CharField(max_length=512)
    args = forms.CharField(max_length=1024, 
                           initial=models.GlobalConfig.objects.all()[0].get_default_command(), 
                           required=False,
                           widget=forms.TextInput(attrs={'size':'60'}))
    blockArgs = forms.CharField(max_length=64,
                           required=False,
                           widget=forms.TextInput(attrs={'size':'60'}))
    libraryKey = forms.CharField(max_length=128,
                           required=False,
                           widget=forms.TextInput(attrs={'size':'60'}))
    tf_config = forms.FileField(required=False,
                                widget=forms.FileInput(attrs={'size':'60'}))
    takeover_node = forms.BooleanField(required=False, initial=False)
    align_full = forms.BooleanField(required=False, initial=False)
    qname = forms.CharField(max_length=128,widget=forms.HiddenInput(), required=True)
    aligner_opts_extra = forms.CharField(max_length=100000,required=False,widget=forms.Textarea(attrs={'cols': 50, 'rows': 4}))

    def clean_report_name(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        reportName = self.cleaned_data.get("report_name")
        if not set(reportName).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
            raise forms.ValidationError(("That Report name has invalid characters. The valid values are letters, numbers, underscore and period."))
        else:
            return reportName

    def clean_libraryKey(self):
        #TODO: note that because this is a hidden advanced field it will not be clear if it fails
        key = self.cleaned_data.get('libraryKey')
        if not set(key).issubset("ATCG"):
            raise forms.ValidationError(("This key has invalid characters. The valid values are TACG."))
        else:
            return key


class BugReportForm(forms.Form):
    CATEGORY_CHOICES = (
        ("LINK", "Broken Link"),
        ("UI", "Clunky/Confusing Interface"),
        ("MISSING", "Missing Data"),
        ("OTHER", "Other"),)
    email = forms.EmailField(min_length=3, max_length=256,
                             label="Your Email")
    category = forms.ChoiceField(choices=CATEGORY_CHOICES)
    description = forms.CharField(widget=forms.Textarea)
    src_url = forms.CharField(widget=forms.HiddenInput)
    
class ExperimentFilter(forms.Form):
    CHOICES=()
    YEARS = range(2008, datetime.date.today().year+2)
    SELECT_DATE = widgets.SelectDateWidget(years=YEARS,
                                           attrs={"class":"date_select"})
    date_start = forms.DateField(required=False,
                                 widget=SELECT_DATE,
                                 initial=datetime.date(2008,11,5))
    date_end = forms.DateField(required=False,
                               widget=SELECT_DATE,
                               initial=datetime.date.today)
    pgm = forms.ChoiceField(choices=CHOICES, required=False, initial="None",
                            label="PGM")
    project = forms.ChoiceField(choices=CHOICES, required=False, initial="None")
    sample = forms.ChoiceField(choices=CHOICES, required=False, initial="None")
    library = forms.ChoiceField(choices=CHOICES, required=False,
                                initial="None", label="Reference")
    storage = forms.ChoiceField(choices=CHOICES, required=False,initial="None")
    starred = forms.BooleanField(required=False, initial=False)

    def __init__(self, *args, **kwargs):
        super(ExperimentFilter, self).__init__(*args, **kwargs)       
        choice_model = models.Experiment
        def choicify(fieldname):
            dbl = lambda x: (x,x)
            rawvals = choice_model.objects.values(fieldname).distinct()
            vals = [v[fieldname] for v in rawvals]
            try:
                if type(vals[0])==type(0):
                    vals.sort()
                    vals.reverse()
                else:
                    vals.sort(key=lambda x: x.lower())
            except:
                pass
            choices = [("None","None")]
            choices.extend(map(dbl,vals))
            return choices

        test = ['pgm','project','sample','library']
        fields = ["pgmName","project","sample","library"]
        for field,field_name in zip(test,fields):
            self.fields[field].choices = choicify(field_name)
        choices = [("None","None")]
        s_choices = list(choice_model.STORAGE_CHOICES)
        s_choices.sort(key=lambda x: x[1])
        choices.extend(i for i in s_choices)
        self.fields['storage'].choices = choices

class SearchForm(forms.Form):
    SEARCHBOX_WIDGET = forms.TextInput(attrs={"class":"searchbox"})
    searchterms = forms.CharField(widget=SEARCHBOX_WIDGET)

class SortForm(forms.Form):
    SORT_WIDGET=forms.HiddenInput(attrs={"class":"sortfield"})
    sortfield = forms.CharField(widget=SORT_WIDGET)
       
class ReportFilter(forms.Form):
    CHOICES=()
    YEARS = range(2008, datetime.date.today().year+2)
    SELECT_DATE = widgets.SelectDateWidget(years=YEARS,
                                           attrs={"class":"date_select"})
    date_start = forms.DateField(required=False,
                                 widget=SELECT_DATE,
                                 initial=datetime.date(2008,11,5))
    date_end = forms.DateField(required=False,
                               widget=SELECT_DATE,
                               initial=datetime.date.today)
    status = forms.ChoiceField(choices=CHOICES, required=False, initial="None")
    template = forms.ChoiceField(choices=CHOICES, required=False, initial="None")
    cycles = forms.ChoiceField(choices=CHOICES, required=False, initial="None", label="Flows")
    project = forms.ChoiceField(choices=CHOICES, required=False, initial="None")
    sample = forms.ChoiceField(choices=CHOICES, required=False, initial="None")
    library = forms.ChoiceField(choices=CHOICES, required=False,
                                initial="None", label="Reference")

    def __init__(self, *args, **kwargs):
        super(ReportFilter, self).__init__(*args, **kwargs)
        def choicify(fieldname,choiceModel):
            dbl = lambda x: (x,x)
            rawvals = choice_model.objects.values(fieldname).distinct()
            vals = [v[fieldname] for v in rawvals]
            try:
                if type(vals[0])==type(0):
                    vals.sort()
                    vals.reverse()
                else:
                    vals.sort(key=lambda x: x.lower())
            except:
                pass
            choices = [("None","None")]
            choices.extend(map(dbl,vals))
            return choices

        test = ['status','cycles']
        fields = ['status','processedCycles']
        choice_model = models.Results
        for field,field_name in zip(test,fields):
            self.fields[field].choices = choicify(field_name, choice_model)
        # TODO: Cycles -> flows hack, very temporary.
        self.fields['cycles'].choices[1:] = [(value, label * 4) for value, label in self.fields['cycles'].choices[1:]]
        test = ['template']
        fields = ['name']
        choice_model = models.TFMetrics
        for field,field_name in zip(test,fields):
            self.fields[field].choices = choicify(field_name, choice_model)
        test = ['project', 'sample', 'library']
        fields = ['project', 'sample', 'library']
        choice_model = models.Experiment
        for field,field_name in zip(test,fields):
            self.fields[field].choices = choicify(field_name, choice_model)

class AddTemplate(forms.Form):
    def clean_name(self):
        """
        Verify that the user input doesn't have chars that we don't want
        """
        name = self.cleaned_data.get("name")
        templates = models.Template.objects.all()

        #only exclude pk if it is a new reference
        pk = self.data.getlist("pk")[0]
        if pk:
            templates = models.Template.objects.all().exclude(pk=pk)

        if not set(name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_. "):
            raise forms.ValidationError(("This name has invalid characters. The valid values are letters, numbers, underscore and period."))
        
        for sequence in templates:
            if name == sequence.name:
                raise forms.ValidationError(("A template with the name %s already exists" % name))
        return name

    def clean_sequence(self):
        seq = self.cleaned_data.get('sequence')
        if not set(seq).issubset("ATCG"):
            raise forms.ValidationError(("This sequence has invalid characters. The valid values are TACG."))

        templates = models.Template.objects.all()

        #only exclude pk if it is a new reference
        pk = self.data.getlist("pk")[0]
        if pk:
            templates = models.Template.objects.all().exclude(pk=pk)

        for sequence in templates:
            if seq == sequence.sequence:
                raise forms.ValidationError(("This sequence already exists with name %s" % sequence.name))
        return seq
    
    def clean_key(self):
        key = self.cleaned_data.get('key')
        if not set(key).issubset("ATCG"):
            raise forms.ValidationError(("This key has invalid characters. The valid values are TACG."))
        else:
            return key

    name = forms.CharField(max_length=64)
    sequence = forms.CharField(max_length=2048, required=True)
    key = forms.CharField(max_length=64, required=True)
    isofficial = forms.BooleanField(required=False, initial=True)
    comments = forms.CharField(required=False)
    pk = forms.IntegerField(widget=forms.HiddenInput(), required=True)

class EditBackup(forms.Form):
    def get_dir_choices():
        basicChoice = [(None, 'None')]
        for choice in devices.to_media(devices.disk_report()):
            basicChoice.append(choice)
        return tuple(basicChoice)
    def get_loc_choices():
        basicChoice = []
        for loc in models.Location.objects.all():
            basicChoice.append((loc,loc))
        return tuple(basicChoice)
    def make_throttle_choices():
        choice = [(0,'Unlimited'), (10000,'10MB/Sec')]
        return tuple(choice)

    archive_directory = forms.ChoiceField(choices=())
    number_to_archive = forms.IntegerField()
    timeout = forms.IntegerField()
    percent_full_before_archive = forms.IntegerField()
    bandwidth_limit = forms.ChoiceField(make_throttle_choices())
    email = forms.EmailField(required=False)
    enabled = forms.BooleanField(required=False)
    grace_period = forms.IntegerField()

    def __init__(self,*args,**kwargs):
        super(EditBackup,self).__init__(*args,**kwargs)
        def get_dir_choices():
            basicChoice = [(None, 'None')]
            for choice in devices.to_media(devices.disk_report()):
                basicChoice.append(choice)
            return tuple(basicChoice)
        self.fields['archive_directory'].choices = get_dir_choices()

class StorageOptions(forms.Form):
    ex = models.Experiment()
    storage_options = forms.ChoiceField(choices=ex.get_storage_choices(), required=False)

class EditEmail(forms.Form):
    email_address = forms.EmailField(required=True)
    selected = forms.BooleanField(required=False, initial=False)

class BestRunsSort(forms.Form):
    library_metrics = forms.ChoiceField(choices=(),initial='i100Q17_reads')

    def __init__(self,*args,**kwargs):
        super(BestRunsSort,self).__init__(*args,**kwargs)
        met = models.LibMetrics._meta.get_all_field_names()
        self.fields['library_metrics'].choices = ((m,m) for m in met)
    
class EditReferenceGenome(forms.Form):
    name = forms.CharField(max_length=512,required=True)
    NCBI_name = forms.CharField(max_length=512,required=True)
    read_sample_size = forms.CharField(max_length=512,required=False)
    notes = forms.CharField(max_length=1048,required=False,widget=forms.Textarea(attrs={'cols': 50, 'rows': 4}))
    enabled = forms.BooleanField(required=False)
    genome_key = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    index_version = forms.CharField(widget=forms.HiddenInput(), required=True)

    def clean_name(self):
        """don't allow duplicate names
        make an exception for the genome we are working with
        """

        index_version = self.data.getlist("index_version")[0]
        genome_key = self.data.getlist("genome_key")[0]
        sample_size = self.data.getlist("read_sample_size")[0]
        get_name = self.cleaned_data.get('name')
        genomes = models.ReferenceGenome.objects.filter(short_name=get_name).exclude(pk=genome_key)

        if not set(get_name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"):
            raise forms.ValidationError(("The short name has invalid characters. The valid values are letters, numbers, and underscores."))

        if not sample_size.isdigit():
            raise forms.ValidationError(("The read sample size must be a positive number."))

        for g in genomes:
            if get_name == g.name and g.index_version == index_version :
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
    modes = (("dhcp","DHCP",), ("static", "Static"))
    mode = forms.ChoiceField(widget=forms.widgets.RadioSelect, choices=modes)
    address = forms.IPAddressField(label="IP Address", required=False)
    subnet = forms.IPAddressField(required=False, label="Subnet")
    gateway = forms.IPAddressField(required=False)
    nameservers = forms.CharField(required=False, max_length=256)
    proxy_address = forms.CharField(required=False, max_length=256)
    proxy_port = forms.CharField(required=False)
    proxy_username = forms.CharField(required=False)
    proxy_password = forms.CharField(required=False)
    collab_ip = forms.IPAddressField(required=False)

    def get_network_settings(self):
        """Usage: ./TSquery [options]
             --all, -a           Output all information (default)
             --proxy_info        Output just proxy http info
             --eth_info          Output just ethernet info
             --eth-dev           Specify eth device to query (default = eth0)
             --json              Output json format
             --outputfile, -o    Write output to file
        Outputs:
        http_proxy:not set
        network_device:eth0
        network_mode:dhcp
        network_address:10.0.2.15
        network_subnet:255.255.255.0
        network_gateway:10.0.2.2
        """
        settings = {
                "mode":"",
                "address":"",
                "subnet":"",
                "gateway":"",
                "nameservers":"",
                "proxy_address": "",
                "proxy_port": "",
                "proxy_username": "",
                "proxy_password": "",
                "collab_ip": "",
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
        self.fields['proxy_address'].initial = settings["proxy_address"]
        self.fields['proxy_port'].initial = settings["proxy_port"]
        self.fields['proxy_username'].initial = settings["proxy_username"]
        self.fields['proxy_password'].initial = settings["proxy_password"]
        self.fields['collab_ip'].initial = settings["collab_ip"]

    def __init__(self, *args, **kw):
        super(NetworkConfigForm, self).__init__(*args, **kw)
        self.set_to_current_values()

    def save(self, *args, **kw):
        host_task, proxy_task, dns_task = None, None, None
        settings = self.get_network_settings()
        host_config = ["mode", "address", "subnet", "gateway"]
        if self.new_config(self.cleaned_data, settings, host_config):
            logger.info("User changed the host network settings.")
            if self.cleaned_data['mode'] == "dhcp":
                host_task = tasks.dhcp.delay()
            elif self.cleaned_data['mode'] == "static":
                address = self.cleaned_data['address']
                subnet = self.cleaned_data['subnet']
                gateway = self.cleaned_data['gateway']
                host_task = tasks.static_ip.delay(address, subnet, gateway)
        proxy_config = ["proxy_address", "proxy_port", "proxy_username", "proxy_password"]
        if self.new_config(self.cleaned_data, settings, proxy_config):
            logger.info("User changed the proxy settings.")
            if self.cleaned_data['proxy_address'] and self.cleaned_data['proxy_port']:
                address = self.cleaned_data['proxy_address']
                port = self.cleaned_data['proxy_port']
                user = self.cleaned_data['proxy_username']
                password = self.cleaned_data['proxy_password']
                proxy_task = tasks.proxyconf.delay(address, port, user, password)
            else:
                proxy_task = tasks.ax_proxy.delay()
        if self.new_config(self.cleaned_data, settings, ["nameservers"]):
            logger.info("User changed the DNS settings.")
            if self.cleaned_data['nameservers']:
                dns_task = tasks.dnsconf.delay(self.cleaned_data['nameservers'])
        if host_task:
            host_task.get()
        if proxy_task:
            proxy_task.get()
        if dns_task:
            dns_task.get()
        self.set_to_current_values()
