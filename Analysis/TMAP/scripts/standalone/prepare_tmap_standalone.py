#!/usr/bin/env python

"""
Utility for creating the TMAP standalone source based on the TorrentSuite source tree
"""

VERSION="1.0.a"

import sys, os, os.path, re, shutil, argparse, logging

# status logger
status_logger = logging.getLogger ('status_logger')
status_logger.addHandler (logging.StreamHandler ())
status_logger.setLevel (logging.INFO)
# error logger
error_logger = logging.getLogger ('error_logger')
error_logger.addHandler (logging.StreamHandler ())
error_logger.setLevel (logging.ERROR)
# debug logger
debug_logger = logging.getLogger ('debug_logger')
debug_logger.addHandler (logging.StreamHandler ())
debug_logger.setLevel (logging.DEBUG)

class FSO:
    "File system object types"
    DIR = "directory",
    FILE = "file"

copylist = (
    ("Analysis/TMAP/src", "src", FSO.DIR),
    ("external/samtools_tmap_e783ea9/samtools", "src/samtools", FSO.DIR),
    ("Analysis/TMAP/config.h.in", "config.h.in"),
    ("Analysis/TMAP/LICENSE", "LICENSE"),
    ("Analysis/version", "version"),
    "buildTools/IonVersion.cpp.in",
    "buildTools/IonVersion.env.in",
    "buildTools/IonVersion.h.in" )

CMAKELISTS_RPATH = "Analysis/TMAP"
CMAKELISTS_FNAME = "CMakeLists.txt"
CMAKELISTS_TEMPLATE_FNAME = "CMakeLists.txt.template"

checklist = (
    CMAKELISTS_TEMPLATE_FNAME,
    "buildTools/cmake/CMakeLists.compiler.txt",
    "buildTools/cmake/CMakeLists.version.txt" )

class Copier:
    def __init__ (self, source_dir, dest_dir):
        self.src_dir = source_dir
        self.dst_dir = dest_dir
    def __call__ (self, spec):
        """
        copies objects from src_dir into dst_dir
        Spec is either tuple or string
        If tuple of 3 items: src_name, dest_name, filesystem_obj_type
        If tuple of 2 items: file_src_name, file_dest_name
        If string: file subpath, destination same as source (created under dst directory)
        """
        debug_logger.debug ("spec = {}".format (spec))
        if isinstance (spec, basestring):
            src, dst, t = spec, spec, FSO.FILE
        else:
            try:
                spec_iter = iter (spec)
                src = spec_iter.next ()
                dst = spec_iter.next ()
                try:
                    t = spec_iter.next ()
                except StopIteration:
                    t = FSO.FILE
            except TypeError:
                raise Exception ("Internal: Wrong spec passed to copier: neither string nor iterable ({})".format (spec))
            except StopIteration:
                raise Exception ("Internal: Wrong spec passed to copier: not enough items in the 'spec' iterable ({}), at least 2 items, src and dest, are required".format (spec))
        debug_logger.debug ("src = {}, dst = {}".format (src, dst))
        src_full = os.path.join (self.src_dir, src)
        dst_full = os.path.join (self.dst_dir, dst)
        # check that source exists
        if t == FSO.FILE and not os.path.isfile (src_full):
            raise Exception ("Source file not found: {}".format (src_full))
        if t == FSO.DIR and not os.path.isdir (src_full):
            raise Exception ("Source directory not found: {}".format (src_full))
        # make the destination dir if needed
        dst_path = os.path.dirname (dst_full)
        if not os.path.isdir (dst_path):
            os.makedirs (dst_path)
        if t == FSO.FILE:
            status_logger.info ("Copying file {} to {}".format (src_full, dst_full))
            shutil.copy (src_full, dst_full)
        else:
            if os.path.isdir (dst_full):
                status_logger.info ("Removing already existing directory {} that is in the way".format (dst_full))
                shutil.rmtree (dst_full)
            status_logger.info ("Copying directory {} to {}".format (src_full, dst_full))
            shutil.copytree (src_full, dst_full) # NB: symlinks are False (default) , so content is copied

def prepare_cmakelists (template_fname, source_fname, dest_fname):
    """
    reads in the template and fills in sections setting following variables:
      GLOBAL_SOURCES
      samtools_SOURCES
      tmap_SOURCES
    to match these variables found in the source file
    writes output (with the sections filled in) to dest_fname
    """
    # NB copying is done line-by-line, to keep logical grouping of sources

    var_names = ("GLOBAL_SOURCES", "samtools_SOURCES", "tmap_SOURCES")

    var_names_re = r"set\s*\(\s*(?P<varname>" + "|".join ("(" + var_name +  ")" for var_name in var_names) + ")\s*(?P<tail>.*)$"
    var_names_matcher = re.compile (var_names_re)
    var_end_re = r"((?P<head>.*)\s*\))|$"
    var_end_matcher = re.compile (var_end_re)

    # read the needed variable definition sections
    value_lists = {} # var_name->[value lines]
    cur_section = None
    with file (source_fname) as source:
        for line in source:
            line = line.strip ()
            if cur_section is None:
                m = var_names_matcher.match (line)
                if m is None:
                    continue
                cur_section = m.group ("varname")
                var_values = m.group ("tail")
                if len (var_values):
                    value_lists.setdefault (cur_section, []).append (var_values)
            else:
                m = var_end_matcher.match (line)
                if m is not None:
                    var_values = m.group ("head")
                    if len (var_values):
                        value_lists.setdefault (cur_section, []).append (var_values)
                    cur_section = None
                else:
                    if len (line) and not line.startswith ("#"):
                        value_lists.setdefault (cur_section, []).append (line)

    if cur_section is not None:
        raise Exception ("Source CMakeLlists file ({}) parsing error: unterminated assignment section for '{}'".format (source_fname, cur_section))
    seen_vars = set (value_lists)
    expected_vars = set (var_names)
    if seen_vars != expected_vars:
        dd = expected_vars - seen_vars
        raise Exception ("Source CMakeLlists file ({}) parsing error: missing assignment{} for: {}".format (source_fname, ("" if len (dd) == 1 else "s"), ", ".join (dd)))

    # now read the template and output to destination
    cur_section = None
    written_sections = set ()
    with file (template_fname) as template:
        with file (dest_fname, "w") as dest:
            for line in template:
                sline = line.strip ()
                if cur_section is None:
                    m = var_names_matcher.match (sline)
                    if m is None:
                        dest.write (line)
                    else:
                        cur_section = m.group ("varname")
                        if cur_section in written_sections:
                            error_logger.error ("Warning: duplicate section '{}' found in template '{}'".format (cur_section, template_fname))
                        written_sections.add (cur_section)
                        dest.write ("set ( {}\n".format (cur_section))
                        for content_line in value_lists [cur_section]:
                            dest.write ("    {}\n".format (content_line))
                        dest.write (")\n")
                else:
                    m = var_end_matcher.match (sline)
                    if m is not None:
                        cur_section = None

    if cur_section is not None:
        raise Exception ("Template CMakeLlists file ({}) parsing error: unterminated assignment section for '{}'".format (template_fname, cur_section))
    if written_sections != expected_vars:
        dd = expected_vars - written_sections
        raise Exception ("Template CMakeLlists file ({}) parsing error: missing assignment section{} for '{}'".format (template_fname, ("" if len (dd) == 1 else "s"), ", ".join (dd)))
    return 1

def parse_cmdline (args):
    epilog = "Destination directory should be pre-populated with the following directories/files:\n    {}".format ("\n    ".join (fsobj for fsobj in checklist))
    parser = argparse.ArgumentParser (description = __doc__, epilog = epilog, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument (
        "--verbose",
        dest = "verbose",
        action = "store_true",
        default = False,
        help="Print out the actions performed"
    )
    parser.add_argument (
        "-D",
        dest = "debug",
        action = "store_true",
        default = False,
        help="Enable debug output"
    )
    parser.add_argument (
        "--dest",
        dest = "dest",
        default = "./",
        help="Destination directory, defaults to current"
    )
    parser.add_argument (
        "source",
        help="TorrentSuite source code directory"
    )
    params = parser.parse_args (args [1:])
    return params

def check_ts_dir (dirname):
    if not os.path.isdir (dirname):
        raise Exception ("TorrentSuite source directory ({}) not found".format (dirname))

def check_presence (path, fname):
    tt = os.path.join (path, fname)
    if not os.path.exists (tt):
        raise Exception ("File or directory {} not found".format (tt))

def process (par):

    status_logger.info ("Checking pre-requisites for TMAP standalone preparation from TorrentSuite in {} ".format (par.source))

    check_ts_dir (par.source)
    for fname in checklist:
        check_presence (par.dest, fname)
    check_presence (os.path.join (par.source, CMAKELISTS_RPATH), CMAKELISTS_FNAME)

    status_logger.info ("Preparing TMAP standalone from torrent-suite in %s".format (par.source))

    cp = Copier (par.source, par.dest)
    for spec in copylist:
        cp (spec)

    status_logger.info ("Preparing TMAP standalone CMakeLists.txt in %s".format (par.dest))

    cmake_templ_fname = os.path.join (par.dest, CMAKELISTS_TEMPLATE_FNAME)
    cmake_src_fname = os.path.join (par.source, CMAKELISTS_RPATH, CMAKELISTS_FNAME)
    cmake_dest_fname = os.path.join (par.dest, CMAKELISTS_FNAME)
    prepare_cmakelists (cmake_templ_fname, cmake_src_fname, cmake_dest_fname)

    status_logger.info ("Done: all files are successfully prepared in %s\n" % par.dest)

    return 0

def main (args):
    par = parse_cmdline (args)
    if not par.verbose:
        status_logger.setLevel (logging.ERROR)
    if not par.debug:
        debug_logger.setLevel (logging.ERROR)
    return process (par)
    #except Exception as e:
    #    error_logger.error ("Error: {}".format (e))
    #    return 1

if __name__ == "__main__":
    sys.exit (main (sys.argv))
