#!/usr/bin/python
# Copyright (C) 2017 Thermo Fisher Scientific All Rights Reserved
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 27 09:53:37 2017

Bed file reader and writer classers
Format definitions from https://genome.ucsc.edu/FAQ/FAQformat
"""



# ================================================================

class IonBedHeader:
    
    def __init__(self):
        self.Clear()
        
    def Clear(self):
        self.track=[]
        self.browser=[]
        self.type='bed'
        self.columns = []
        
    def AddHeaderLine(self, line):
        sline = line.rstrip().split()
        
        if sline[0]=='track':
            self.track.append(self._getHeaderLineDict(sline))
            if self.track[-1].get('type', 'bed') != 'bed':
                self.type = self.track[-1]['type']
            return True
        elif sline[0]=='browser':
            self.browser.append(self._getHeaderLineDict(sline))
            return True
        else:
            return False
            
    def AddTrackDict(self, track):
        self.track.append(track)
        
    def AddBrowserDict(self, browser):
        self.browser.append()
        
    # We do not check whether we have a valid format & all values are strings
    def _getHeaderLineDict(self, sline):
        idict = {}
        for item in sline[1:]:
            sitem = item.split('=')
            if len(sitem) > 1:
                idict[sitem[0]] = sitem[1]
            else:
                idict[sitem[0]] = ''
        return idict
        
    def _dictToHeaderStr(self, htype, hdict):
        mystr = htype
        for k in hdict:
            mystr += (' ' + k)
            if len(hdict[k])>0:
                mystr += ('='+hdict[k])
        return mystr
        
    def WriteHeader(self, fhandle):
        try:
            # Write browser lines first
            for bline in self.browser:
                fhandle.write(self._dictToHeaderStr('browser',bline)+'\n')
            for tline in self.track:
                fhandle.write(self._dictToHeaderStr('track',tline)+'\n')
            return True
        except:
            print('IonBedHeader ERROR unable to write header.')
            return False

# ================================================================
# Reader class that creates a dictionary from a bed file

class IonBedReader:
    
    # All 12 possible columns defined in the bed format in order
    # First 3 columns are mandatory
    bedColumns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', \
    'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    
    # Two extra columns at the end if type=bedDetail
    bedDetailCols = ['id', 'info']
    
    
    # A list specifying the chromosome order in the bed file
    def __init__(self, file=None, filename=None):
        
        self.chromOrder = []
        self.columns = []
        self.header = IonBedHeader()
        self.nBedCols = 0
        self.bline = ''
        self.filename = filename
        
        
        try:
            if file==None:            
                self.bed = open(filename, 'r')
            else:
                self.bed = file
            self.bline = self.bed.readline().rstrip()
        except:
            print ('IonBedReader ERROR cannot open/read bedfile %s' % filename)
            return
            
        # Read header lines
        if self.bline == '':
            print ('IonBedReader ERROR bedfile %s is empty!' % filename)
            return
        while self.header.AddHeaderLine(self.bline):
            self.bline = self.bed.readline().rstrip()
            
        # We are now at the first values line and can determine the number/type of columns used
        bsplit = self.bline.split('\t')
        nCols = len(bsplit)
        if self.header.type == 'bedDetail':
            nCols -= 2
        if nCols < 3:
            print ('IonBedReader ERROR bedfile %s needs at least chrom,chromStart,chromEnd fields.' % self.filename)
            return
        elif nCols > 12:
            print ('IonBedReader ERROR bedfile %s has too many columns.' % self.filename)
            return
        self.columns = self.bedColumns[0:nCols]
        if self.header.type == 'bedDetail':
            for col in self.bedDetailCols:
                self.columns.append(col)
        # Propagate columns used in file to header object (for writer init) 
        self.header.columns = self.columns
        
# ----------------------------------------         
        
    def close(self):
        if not self.bed.closed:
            self.bed.close()
        
# ----------------------------------------       
# Load all of the bed file into a single dictionary
        
    def load(self):
            
        beddict = {}
        if self.bed.closed:
            print ('IonBedReader ERROR bedfile %s is not open.' % self.filename)
            return beddict
        
        for k in self.columns:
            beddict[k] = []                
        for k,v in zip(self.columns, self.bline.split('\t')):
            beddict[k].append(self._convertBedType(k, v))

        lnum=1
        for line in self.bed:
            lnum += 1
            bsplit = line.rstrip().split('\t')
            if len(bsplit)!= len(self.columns):
                print ('IonBedReader ERROR unexpected number of columns in line %d of file %s' % (lnum, self.filename))
                return {}
                
            for k,v in zip(self.columns, bsplit):
                beddict[k].append(self._convertBedType(k, v))
            
        return beddict
        print('Nothing to see yet!')
        # for line in self.bed
        
# ----------------------------------------
# This function is meant to be evaluates while reading a target file line by line
# It evaluates to : -1 iff chrom/pos is before the target
#                    0 iff chrom/pos falls within the target
#                    1 iff chrom/pos is after the target
        
    def InTarget(self, target, chrom, pos):
        
        if chrom not in self.chromOrder:
            return 1
        if not target:
            return -1
        # This case should never happen if the function is used properly
        if target['chrom'] not in self.chromOrder:
            return -1
            
        t_idx = self.chromOrder.index(target['chrom'])
        c_idx = self.chromOrder.index(target['chrom'])
        if c_idx < t_idx:
            return -1
        elif c_idx > t_idx:
            return 1
        elif pos < target['chromStart']:
            return -1
        elif pos >= target['chromEnd']:
            return 1
        else:
            return 0

# ----------------------------------------
# Returns a dictionary of a single bed line
        
    def readline(self):
        
        if self.bline == '':
            return {}
        else:
            ldict = self.bedLine2Dict(self.bline)
            self.bline = self.bed.readline().rstrip()
            
        if ldict['chrom'] not in self.chromOrder:
            self.chromOrder.append(ldict['chrom'])
        return ldict
            
# ----------------------------------------      
    def bedLine2Dict(self,bline):
            
        ldict = {}
        bsplit = self.bline.split('\t')
        if len(bsplit)!= len(self.columns):
            print ('IonBedReader ERROR bedfile %s needs chrom,chromStart,chromEnd fields' % self.filename)
            return {}
                
        for k,v in zip(self.columns, bsplit):
            ldict[k]=self._convertBedType(k, v)
                
        return ldict
            
# ----------------------------------------
# Ordered by frequency of use in Ion bed files
            
    def _convertBedType(self, key, val):
            
        if   key in ['chrom','name','strand','id']:
            return str(val)
        elif key in ['chromStart','chromEnd','score']:
            if val=='.':
                return None
            else:
                return int(val)
        elif key == 'info':
            return self._parseInfoField(val)
        elif key in ['thickStart','thickEnd','blockCount']:
            if val=='.':
                return None
            else:
                return int(val)
        elif key in ['itemRgb','blockSizes','blockStarts']:
            return val.split(',')
        else:
            return str(val)
                
# ---------------------------------------- 
# A 1) semicolon separated list of
#   2) potentially '=' separated key/ value pairs, the value of which
#   3) can be comma separated lists
# Example: GENE_ID=EGFR;COSMIC_ID=COSM6252,COSM13427;Pool=1;HOTSPOTS_ONLY
        
    def _parseInfoField(self, info):
            
        level1 = info.split(';')
        idict ={}
        for item1 in level1:
            level2 = item1.split('=')
            if len(level2) == 1:
                idict[item1] = []
            else:
                idict[level2[0]] = level2[1].split(',')
                    
        return idict

# ================================================================
# Writer class that takes a dictionary of the type produced by the bed reader class 
# and transforms it into a bed file
# We do not do a lot of format checks yet and rely on the user to provide sensible input

class IonBedWriter:
    
    # The superset of possible bed columns in order
    mandatoryBedCols = ['chrom', 'chromStart', 'chromEnd']
    
    optBedCols = ['name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 
    'blockCount', 'blockSizes', 'blockStarts', 'id', 'info']
    
    
    def __init__(self, file=None, filename=None, header=None):
        
        try:
            if file==None:            
                self.bed = open(filename, 'w')
            else:
                self.bed = file
        except:
            print ('IonBedReader ERROR cannot open/read bedfile %s' % filename)
            
        if header == None:
            self.header = IonBedHeader()
        else:
            self.header = header
        self.columns = []
        self.wroteHeader = False
        
# ----------------------------------------         
        
    def close(self):
        if not self.bed.closed:
            self.bed.close()
            
# ----------------------------------------  
            
    def WriteHeader(self):
        self.header.WriteHeader(self.bed)
        self.wroteHeader = True

# ---------------------------------------- 

    def write(self, fdict):
        
        if not type(fdict['chrom'])==list:
            print('IonBedWriter ERROR expected type(chrom)==list for write function.')
            return
        if not self.wroteHeader:    
            self.WriteHeader()
        
        for line in range(0,len(fdict['chrom'])):
            myline = str(fdict['chrom'][line]) + '\t' + str(fdict['chromStart'][line]) + '\t' + str(fdict['chromEnd'][line])
            for col in self.optBedCols:
                if col in fdict:
                    myline += ('\t' + self._convertColToStr(col, fdict[col][line]))
            self.bed.write(myline+'\n')
            

# ---------------------------------------- 

    def writeline(self, ldict):
        
        if not type(ldict['chrom'])==str:
            print('IonBedWriter ERROR expected type(chrom)==str for writeline function.')
            return
            
        # Write header before first line, if we haven't done so already
        if not self.wroteHeader:
            self.WriteHeader()
        # First ever line to be written defines the columns and subseqeunt lines check that we are consistent
        # and only use valid column names
        #if len(self.columns)==0:
            
        # Force error if one of the mandatory columns is missing
        myline = ldict['chrom'] + '\t' + str(ldict['chromStart']) + '\t' + str(ldict['chromEnd'])
        for col in self.optBedCols:
            if col in ldict:
                myline += ('\t' + self._convertColToStr(col, ldict[col]))
        self.bed.write(myline+'\n')
            

# ---------------------------------------- 
        
    def _convertColToStr(self, col, val):
        
        # nothing to do for string fields
        
        if   col in ['chrom','name','strand','id']:
            return str(val)
        elif col in ['chromStart','chromEnd','score']:
            if val==None:
                return '.'
            else:
                return str(val)
        elif col == 'info':
            return self._mergeInfoField(val)
        elif col in ['thickStart','thickEnd','blockCount']:
            if val==None:
                return '.'
            else:
                return str(val)
        elif col in ['itemRgb','blockSizes','blockStarts']:
            # We try to be accomodating if someone specified a non-str list
            return ','.join(str(e) for e in col)

# ---------------------------------------- 
    
    def _mergeInfoField(self, info):
        
        infostr = ''
        for l1 in info:
            if len(infostr)>0:
                infostr += ';'
            infostr += l1
            if len(info[l1]) > 0:
                infostr += ('=' + ','.join(str(e) for e in info[l1]))
        
        return infostr
    