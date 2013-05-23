#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import math
import os
import os.path
import getopt

def usage():
    global progName
    sys.stderr.write("Usage: %s [--min-bayesian-score=<float> --downsampled-vcf=<out filename>] <input vcf> <output vcf>\n" % progName)

def chompLastSemi(input):
    output = input;
    if output[-1] == ';':
        output = output[:-1]
    return output

def main(argv):
    # arg processing
    global progName
    min_bayesian_score = 2.5
    min_var_freq = 0.2
    bay_score_minlen = 11
    strand_bias = 0.95
    
    out_ds_name="#"
    try:
        opts, args = getopt.getopt( argv, "hm:", ["help", "min-bayesian-score=","downsampled-vcf=","min-var-freq=","bay-score-minlen=","variant-strand-bias="] )
    except getopt.GetoptError, msg:
        sys.stderr.write(str(msg))
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-mbs", "--min-bayesian-score"):
            min_bayesian_score = float(arg)
        elif opt in ("-mvf", "--min-var-freq"):
            min_var_freq = float(arg)       
        elif opt in ("-bml", "--bay-score-minlen"):
            bay_score_minlen = float(arg)       
        elif opt in ("-dsvcf","--downsampled-vcf"):
            out_ds_name=str(arg)
        elif opt in ("-vsb", "--variant-strand-bias"):
            strand_bias = float(arg)              
 
    if(len(args) != 2):
        sys.stderr.write("Error: Invalid number of arguments %i \n" % len(args))
        usage()
        sys.exit(1)
    invcf = args[0]
    if not os.path.exists(invcf):
        sys.stderr.write("No input vcf file found at: %s\n" % invcf)
        sys.exit(1)

    inf = open(invcf,'r')
    out = open(args[1],'w')

    global out_ds
    if out_ds_name[0] != '#':
       out_ds=open(out_ds_name,'w')
       
    header=""
    records=""
    records_ds=""
    badformat = 0
      
    for lines in inf:
        if lines[0]=='#':
            header = header + lines
        else:
            is_ds=0
            attr={}
            fields = lines.split('\t')
            ref = fields[3]
            alt = fields[4]
            gatk_score=float(fields[5])
            info = fields[7]
            info = chompLastSemi(info).split(';')
            str_bias=0
            
            #removing SNPs
            if len(ref) == len(alt):
                continue
            
            var_cov = 0 
            for items in info:
               if "=" in items:
                  key,val = items.split('=')
                  attr[key]=val
            
            # calculate strand-bias if info is available
                  if key == "Strand_Counts":
                     varstr = val.split(':')[1]
                     variants = varstr.split(',')
                     indel=""
                     if len(fields[3]) < len(fields[4]):
                        indel="+%s" % fields[4][1:len(fields[4])]
                     else:
                        indel="%iD" % (len(fields[3]) - len(fields[4]))
                        
                     tot_cov_plus = 0
                     tot_cov_minus = 0
                     var_cov_plus = 0
                     var_cov_minus = 0
                     ii=0
                     for v in variants:
                        if ii % 3==0:
                           if ii+2>=len(variants):
                              badformat = 1
                           else:   
                              tot_cov_plus = tot_cov_plus + int(variants[ii+1])
                              tot_cov_minus = tot_cov_minus + int(variants[ii+2])                        
                              if indel==v:
                                 var_cov_plus = int(variants[ii+1])
                                 var_cov_minus = int(variants[ii+2])
                                 var_cov = var_cov_plus+var_cov_minus
                        ii=ii+1 
                     if var_cov_plus*tot_cov_minus + var_cov_minus*tot_cov_plus > 0:
                        if var_cov_plus*tot_cov_minus > var_cov_minus*tot_cov_plus:
                           str_bias = float(var_cov_plus*tot_cov_minus)/float(var_cov_plus*tot_cov_minus + var_cov_minus*tot_cov_plus) 
                        else:  
                           str_bias = float(var_cov_minus*tot_cov_plus)/float(var_cov_plus*tot_cov_minus + var_cov_minus*tot_cov_plus)
               elif "DS" in items:
                  is_ds=1
            
            var_freq = 0
            if len(fields) > 8:
               info_def = fields[8].split(':')
               info_val = fields[9].split(':')
               
               # report first non-empty genotype field for multi-sample cases
               jj=0
               if info_val[0] == './.':
                  for clmn in fields:
                     if jj>9:
                        badformat = 1
                        info_val = clmn.split(':')                             
                        if info_val[0] != './.':
                           break
                     jj=jj+1
                  
               # split out the GENOTYPE attributes
               ii=-1
               for items in info_def:
                  ii=ii+1
                  if len(items)>0:
                     attr[items]=info_val[ii]
             
               var_freq = float(var_cov)/float(attr['DP'])
               
               if var_freq == 0:
                  var_freq = float(attr['FA'].split(',')[0])

     #FILTERS FOR 1st and 2nd stages            

            #removing low-score indels if no-downsampling calculaion (applied only in second step)
            if out_ds_name[0]=='#':
            
            #filter based on frequency          
               if var_freq < float(min_var_freq):
                  continue
                  
               if len(ref) < bay_score_minlen and len(alt) < bay_score_minlen:
                  # filter out strand biased positions
                  if str_bias > float(strand_bias):
                     continue
                  if float(attr['Bayesian_Score']) < float(min_bayesian_score) :
                     continue
               elif float(attr['Bayesian_Score']) == 2 :
                   continue


# Change back the condition after Multi-Threading GATK<->BayAPI issue is resolved
#            if is_ds==0:
            if is_ds==-1:
               records=records+lines
            else:
               records_ds=records_ds+lines 
    
    if badformat == 1:
       print("WARNING: Unsupported VCF file format, multi-sample columns are present. Analysis is done sample by sample.")
                   
    inf.close() 
    
    if out_ds_name[0]!='#':    
       out_ds.write("%s%s" %(header,records_ds))
       out.write("%s" %(records))
       out_ds.close()
       out.close()
       if records_ds!="":
          sys.exit(3)
       else:
          sys.exit(0)       
    else:
       out.write("%s%s%s" %(header,records,records_ds))
    
  
if __name__ == '__main__':
    global progName
    progName = sys.argv[0]
    main(sys.argv[1:])

