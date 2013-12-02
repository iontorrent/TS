#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import json
import fnmatch
import os
import sys
import dateutil.parser
import zipfile
import stat
import traceback
from glob import glob


class Alignment(object):

  def __init__(self):
    self.genome = "something.fasta"
    self.sampling = False
    self.output_format = {}
    self.output_filters = { 'filter':False } #need to define these
    self.output_dir = None
    self.log_file = None
    self.igv_link = "http://igv.com"
    self.program = "alignmentQC.pl"
    self.program_params = None
    self.fastq = None
    self.fastq_link = None
    self.unmapped_bam = None
    self.analysis_dir = None
    self.basecaller_dir = None
    self.analysis_params = None
    self.cmd = None
    
    
  def check_igv(self):
    pass

  
  def get_sam_meta(self):
    # collect all the meta data for the SAM file
    SAM_META = {}
    # sm - name for reads - project name
    SAM_META['SM'] = self.analysis_params['project']
    # lb - library name 
    SAM_META['LB'] = self.analysis_params['libraryName']
    SAM_META['PL'] = "IONTORRENT"
    
    #TODO: do not assume localhost.  Find the name of the masternode
  
    # get the exp data from the database
    exp_json = json.loads(self.analysis_params['exp_json'])

    # ds - the "notes", only the alphanumeric and space characters.
    SAM_META['DS'] = ''.join(ch for ch in exp_json['notes'] if ch.isalnum() or ch == " ")
    # pu - the platform unit 
    SAM_META['PU'] = "PGM/" + exp_json['chipType'].replace('"',"")  
    
    # dt - the run date
    exp_log_json = json.loads(exp_json['log'])
    iso_exp_time = exp_log_json['start_time']
    
    # convert to ISO time
    iso_exp_time = dateutil.parser.parse(iso_exp_time)
    SAM_META['DT'] = iso_exp_time.isoformat()
    
    # site name should be here
    site_name = self.analysis_params['site_name']
    site_name = ''.join(ch for ch in site_name if ch.isalnum() )
    SAM_META['CN'] = site_name
      
    # Build the SAM meta data arg string
    sam_meta_args = ""
    if SAM_META:
      sam_meta_args= '--aligner-opts-rg "'
      first = True
      for key, value in SAM_META.items():
        if value:
          sam_arg =  r'-R \"'
          end =  r'\"'
          
          sam_arg = sam_arg + key + ":" + value + end
          
          if first:
            sam_meta_args = sam_meta_args + sam_arg
            first = False
          else:
            sam_meta_args = sam_meta_args + " " + sam_arg
          
    if sam_meta_args:
      sam_meta_args = sam_meta_args + '"'
    return sam_meta_args


  def construct_command(self):
    self.file_prefix = "%s/%s_%s" % (alignment.output_dir, alignment.analysis_params['expName'], alignment.analysis_params['resultsName'])
    #sam_meta_args = self.get_sam_meta()
    sam_meta_args = ""
    if self.sampling:
      self.program_params = "--out-base-name %s --genome %s --input %s %s >> %s 2>&1" % (self.file_prefix, self.genome, self.unmapped_bam, sam_meta_args, self.log_file)
    else:
      self.program_params = "--align-all-reads --out-base-name %s --genome %s --input %s %s >> %s 2>&1" % (self.file_prefix, self.genome, self.unmapped_bam, sam_meta_args, self.log_file)
    self.cmd = "%s %s" % (self.program, self.program_params)


def check_output(alignment):
  files = os.listdir(alignment.output_dir)
  for file_format in alignment.output_format:
    #file_format is just raw extension, no punctuation marks.  should just be [A-Za-z0-9]
    filter_str = "*.%s" % (file_format)
    if not fnmatch.filter(files, filter_str):
      f_h = open(alignment.log_file, "a") #append to file
      if f_h:
        f_h.write("[alignment.py] file of format: %s was not produced\n" % file_format)
      else:
        os.system("echo [alignment.py] file of format: %s was not produced >> %s" % file_format, alignment.log_file)
  if "bam" not in alignment.output_format:
    bam_file = alignment.file_prefix + ".bam"
    bam_file_index = bam_file + ".bai"
    if os.path.exists(bam_file):
      os.remove(bam_file)
    if os.path.exists(bam_file_index):
      os.remove(bam_file_index)


def enforce_output_file_formats(alignment):
  for file_format in alignment.output_format:
    if file_format != "sam" and file_format != "bam":
      if alignment.fastq.find(file_format) > 0:
        if not os.path.exists(alignment.fastq_link):
          os.symlink(alignment.fastq, alignment.fastq_link)
    elif file_format == "sam":
      make_sam_file(alignment)
    

def make_sam_file(alignment):
  bam = "%s.bam" % alignment.file_prefix
  sam = "%s.sam" % alignment.file_prefix
  samfile_cmd = "samtools view -h %s > %s" % (bam, sam)
  print "[alignment.py]", samfile_cmd
  if os.system(samfile_cmd) != 0:
    print "[alignment.py] Failed extracting header from '%s' using samtools." % bam
    os.remove(sam)
    sys.exit(1)


def make_simple_html(alignment):
  dirlist = os.listdir(alignment.output_dir)
  align_sum = fnmatch.filter(dirlist, "*alignment.summary")[0]
  align_sum = open(align_sum, "r")
  html_file = open(alignment.output_dir + "/Alignment.html", "w")
  html_file.write("<html>\n<body>\n")
  html_file.write("<h1>Alignment Summary</h1>")
  for line in align_sum:
    html_file.write(line)
    html_file.write("<br>")
  align_sum.close()
  html_file.write("</body>\n</html>\n")
  html_file.close()


def make_zipfile(alignment):
  zip_path = "%s.zip" % (alignment.file_prefix)
  zip_name = zip_path
  myzip = zipfile.ZipFile(zip_name, "w", allowZip64=True)
  def exist( f ):
    if os.path.exists( f ) or os.path.isfile( f ):
      print "[alignment.py] %s exists " % ( f )
    else:
      print "[alignment.py] %s does not exist" % ( f )
  try:
    for suffix in alignment.output_format:
      file_name = "%s.%s" % (alignment.file_prefix, suffix)
      if os.path.isabs(file_name):
        head, tail = os.path.split(file_name)
        file_name = tail
      myzip.write(file_name)
  except zipfile.LargeZipFile:
    print "[alignment.py] The zip file was too large, ZIP64 extensions could not be enabled"
  except:
    print "[alignment.py] Unexpected error writing %s to %s" % ( file_name, zip_name ) 
    exist( file_name )
    exist( zip_name )
    traceback.print_exc()
  finally:
    myzip.close()
  myzip.close()


def create_igv_session(alignment, base_url, plugin_url):
  session_xml = open(alignment.output_dir + "/igv_session.xml", "w")
  session_xml.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
  session_xml.write('<Global genome="' + genome_name + '" version="3">\n')
  session_xml.write('\t<Resources>\n')
  session_xml.write('\t\t<Resource name="'+ options.bam_file + '" path="' + bam_url + '"/>\n')
  session_xml.write('\t</Resources>\n')
  session_xml.write('</Global>\n')


def check_stale_files(dir):
  flist = os.listdir(dir)
  files = ["*.fastq", "*alignment.summary", "alignStats_err.txt", "*.sam", "*.bam", "*.bami", "*.zip", "*.dat", "*.php"]
  for f in files:
    match = fnmatch.filter(flist, f)
    for m in match:
      path = os.path.join(dir,m)
      print "[alignment.py] removing %s" % (path)
      try:
        os.remove(path)
      except:
        print "[alignment.py] failed to remove %s/%s" % (path)
        traceback.print_exc()


if __name__ == '__main__':
  json_file = open(sys.argv[1], "r")
  genome_library = sys.argv[2]
  params = json.load(json_file)
  pluginconfig = params['pluginconfig']
  
  alignment = Alignment()
  if 'genome' in pluginconfig:
    alignment.genome = pluginconfig['genome']
  else:
    alignment.genome = genome_library
  
  # Sampling and SAM output options no longer offered in 4.0
  if 'choice' in pluginconfig:
    if pluginconfig['choice'] == "true":
      alignment.sampling = True
    else:
      alignment.sampling = False
  else:
    alignment.sampling = False
  
  if "file" in pluginconfig:
    if isinstance(pluginconfig['file'], unicode):
      tmp_str = str(pluginconfig['file'])
      alignment.output_format[tmp_str] = True
    else:
      for format in pluginconfig['file']:
        alignment.output_format[format] = True
  else:
    alignment.output_format['bam'] = True
  
  alignment.analysis_dir = params['runinfo']['analysis_dir']
  alignment.basecaller_dir = params['runinfo']['basecaller_dir']
  alignment.analysis_params = json.load(open(alignment.analysis_dir + "/ion_params_00.json", "r"))
  alignment.output_dir = params['runinfo']['results_dir']

  # remove old output files
  check_stale_files(alignment.output_dir)

  # set alignment output dir
  alignment.unmapped_bam = sys.argv[3]
  if not os.path.exists(alignment.unmapped_bam):
    print "[alignment.py] unmapped bam doesn't exist. Looking for mapped bam.."
    #sys.exit(1)
    alignment.unmapped_bam = sys.argv[4]
    if not os.path.exists(alignment.unmapped_bam):
      print "[alignment.py] mapped bam doesn't exist. Exiting.."
    sys.exit(1)

  alignment.log_file = alignment.output_dir + "/alignment_log.txt"
  
  # setup igv input
  #base_url = params['runinfo']['url_root']
  #plugin_url = base_url + "/plugin_out/Alignment_out/"
  #create_igv_session(alignment, base_url)
  
  # run cmd
  alignment.construct_command()
  print "[alignment.py] command: %s\n" % (alignment.cmd)
  if os.system(alignment.cmd) != 0:
    print "[alignment.py] ERROR - Failed running alignment command. See '%s'" % alignment.log_file
    sys.exit(1)
  enforce_output_file_formats(alignment)
  check_output(alignment)

  # replaced with BAM/BAI (BAI file was missing SAM no longer supported)
  #make_zipfile(alignment)

