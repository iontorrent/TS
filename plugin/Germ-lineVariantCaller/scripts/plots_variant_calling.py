#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
os.environ['HOME']='/tmp'
from matplotlib import use
use("Agg") #Needed for some odd linux dependency
import sys
from optparse import OptionParser
import string
import re
import gzip
import numpy as np
import matplotlib.pyplot as plt

def check_option(parser, value, name):
    if None == value:
        print 'Option ' + name + ' required.\n'
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-v', '--vcf-file', help='the bgzip compressed VCF file', dest='vcf_file')
    parser.add_option('-o', '--output-dir', help='the output directory', dest='output_dir')
    parser.add_option('-c', '--max-coverage', type="int", help='the maximum coverage to consider', dest='max_coverage', default=100)
    parser.add_option('-l', '--max-indel-length', type="int", help='the maximum indel length to consider', dest='max_indel_length', default=10)
    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        exit(1)
    check_option(parser, options.vcf_file, '-v')
    check_option(parser, options.output_dir, '-r')
    check_option(parser, options.max_coverage, '-c')
    check_option(parser, options.max_indel_length, '-l')

def float_list_divide_sum(l, sum):
	if 0 < sum:
		for i in range(len(l)):
			l[i] = float(l[i]) / float(sum)
	return l

def float_list_divide(l):
	return float_list_divide_sum(l, sum(l))

def plot_indel_length(heights, width, varType='Insertions', ploidy='Het', color='b', maxVals=10):
	if maxVals < len(heights):
		heights = heights[1:maxVals]
	p1 = plt.bar(left=np.arange(len(heights)), height=heights, color=color)
	plt.title('Number of ' + varType.lower() + ' (' + ploidy.lower() + ') at a given length'); 
	plt.ylabel('Number of ' + varType.lower() + ' (' + ploidy + ')')
	plt.xlabel(varType + ' length')
	plt.xticks(np.arange(len(heights)) + width, 1+np.arange(len(heights)))
	#plt.show()

def plot_coverages(coverages, maxCoverage):
	plt.plot(range(maxCoverage), float_list_divide(coverages[0][0]),
			range(maxCoverage), float_list_divide(coverages[1][0]),
			range(maxCoverage), float_list_divide(coverages[2][0]),
			range(maxCoverage), float_list_divide(coverages[0][1]),
			range(maxCoverage), float_list_divide(coverages[1][1]),
			range(maxCoverage), float_list_divide(coverages[2][1]))
	plt.xlabel('Total coverage')
	plt.ylabel('Fraction of variant positions')
	plt.legend(('Ins (Het)', 'Del (Het)', 'SNP (Het)', 'Ins (Hom)', 'Del (Hom)', 'SNP (Hom)'))
	#plt.show()

def plot_variant_coverages(coverages, maxCoverage):
	plt.plot(range(maxCoverage), float_list_divide(coverages[0][0]),
			range(maxCoverage), float_list_divide(coverages[1][0]),
			range(maxCoverage), float_list_divide(coverages[2][0]),
			range(maxCoverage), float_list_divide(coverages[0][1]),
			range(maxCoverage), float_list_divide(coverages[1][1]),
			range(maxCoverage), float_list_divide(coverages[2][1]))
	plt.xlabel('Variant coverage')
	plt.ylabel('Fraction of variant positions')
	plt.legend(('Ins (Het)', 'Del (Het)', 'SNP (Het)', 'Ins (Hom)', 'Del (Hom)', 'SNP (Hom)'))
	#plt.show()

def plot_variant_fractions(frequencies):
	plt.plot(float_list_divide_sum(range(len(frequencies[0][0])), 100.), float_list_divide(frequencies[0][0]),
		float_list_divide_sum(range(len(frequencies[1][0])), 100.), float_list_divide(frequencies[1][0]),
		float_list_divide_sum(range(len(frequencies[2][0])), 100.), float_list_divide(frequencies[2][0]),
		float_list_divide_sum(range(len(frequencies[0][1])), 100.), float_list_divide(frequencies[0][1]),
		float_list_divide_sum(range(len(frequencies[1][1])), 100.), float_list_divide(frequencies[1][1]),
		float_list_divide_sum(range(len(frequencies[2][1])), 100.), float_list_divide(frequencies[2][1]))
	plt.xlabel('Fraction of reads observing the variant')
	plt.ylabel('Fraction of variant positions')
	plt.legend(('Ins (Het)', 'Del (Het)', 'SNP (Het)', 'Ins (Hom)', 'Del (Hom)', 'SNP (Hom)'))
	#plt.show()

def plot_strand_frequencies(frequencies):
	plt.plot(float_list_divide_sum(range(len(frequencies[0][0])), 100.), float_list_divide(frequencies[0][0]),
		float_list_divide_sum(range(len(frequencies[1][0])), 100.), float_list_divide(frequencies[1][0]),
		float_list_divide_sum(range(len(frequencies[2][0])), 100.), float_list_divide(frequencies[2][0]),
		float_list_divide_sum(range(len(frequencies[0][1])), 100.), float_list_divide(frequencies[0][1]),
		float_list_divide_sum(range(len(frequencies[1][1])), 100.), float_list_divide(frequencies[1][1]),
		float_list_divide_sum(range(len(frequencies[2][1])), 100.), float_list_divide(frequencies[2][1]))
	plt.xlabel('Forward strand frequency')
	plt.ylabel('Fraction of variants')
	plt.legend(('Ins (Het)', 'Del (Het)', 'SNP (Het)', 'Ins (Hom)', 'Del (Hom)', 'SNP (Hom)'))
	#plt.show()

def plot_snps(snps):
	float_list_divide(snps)
	plt.barh(bottom=range(0, -12, -1), width=snps, left=0)
	plt.title('SNP Mutation Profile')
	plt.xlabel('SNP frequency within all SNPs')
	plt.yticks([(i + 0.5) for i in range(0, -12, -1)], ('A->C', 'A->G', 'A->T', 'C->A', 'C->G', 'C->T', 'G->A', 'G->C', 'G->T', 'T->A', 'T->C', 'T->G'))
	#plt.show()

class VariantInfo:
	nt2int = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3, 'a' : 0, 'c' : 1, 'g' : 2, 't' : 3}

	def __init__(self, maxIndelLen, maxCov):
		self.maxIndelLength = maxIndelLen 
		self.maxCoverage = maxCov
		# the distribution of indel lengths for het/hom ins/del
		self.indelLengths = [[[0 for k in range(self.maxIndelLength)] for j in range(2)] for i in range(2)]
		# the coverages for het/hom SNPs/ins/del
		self.coverages = [[[0 for k in range(self.maxCoverage)] for j in range(2)] for i in range(3)]
		# the variant coverages for het/hom SNPs/ins/del
		self.varCoverages = [[[0 for k in range(self.maxCoverage)] for j in range(2)] for i in range(3)]
		# the allele frequencies for het/hom SNPs/ins/del
		self.afs = [[[0 for k in range(100)] for j in range(2)] for i in range(3)] 
		# the strand frequencies for het/hom SNPs/ins/del
		self.strands = [[[0 for k in range(100)] for j in range(2)] for i in range(3)] 
		# the snp mutation table (A->C, A->G, ...)
		self.snps = [[0 for j in range(12)] for i in range(2)] # SNPs and het/hom
	
	# indelType: 0 ins, 1 del
	# ploidy: 0 het, 1 hom
	# length: indel length
	def addIndelLength(self, indelType, ploidy, length):
		if self.maxIndelLength < length:
			length = self.maxIndelLength
		self.indelLengths[indelType][ploidy][length-1]+=1
	
	# varType: 0 ins, 1 del, 2 snp
	# ploidy: 0 het, 1 hom
	# coverage: coverage at the variant position
	def addCoverage(self, varType, ploidy, coverage):
		if self.maxCoverage < coverage:
			coverage = self.maxCoverage
		self.coverages[varType][ploidy][coverage-1]+=1
	
	# varType: 0 ins, 1 del, 2 snp
	# ploidy: 0 het, 1 hom
	# coverage: coverage of that variant
	def addVariantCoverage(self, varType, ploidy, coverage):
		if self.maxCoverage < coverage:
			coverage = self.maxCoverage
		self.varCoverages[varType][ploidy][coverage-1]+=1
	
	# varType: 0 ins, 1 del, 2 snp
	# ploidy: 0 het, 1 hom
	# af: allele frequency
	def addAF(self, varType, ploidy, af):
		if af < 1:
			index = int(round(af * 100))
		else:
			index = 99
		if 100 <= index:
			index = 99
		self.afs[varType][ploidy][index]+=1
	
	# varType: 0 ins, 1 del, 2 snp
	# ploidy: 0 het, 1 hom
	# strand: forward strand frequency of that variant
	def addStrand(self, varType, ploidy, strand):
		if strand < 1:
			index = int(round(strand * 100))
		else:
			index = 99
		if 100 <= index:
			index = 99
		self.strands[varType][ploidy][index]+=1
	
	# ploidy: 0 het, 1 hom
	def addSNP(self, ploidy, baseFrom, baseTo):
		self.snps[ploidy][(self.nt2int[baseFrom] * 3) + self.nt2int[baseTo]]+=1

# NB: hard coded file path and coverage
f = gzip.open(options.vcf_file, 'r')
varInfo = VariantInfo(int(options.max_indel_length), int(options.max_coverage))
for line in f:
	# process
	if not re.search('^#', line):
		# tokenize
		tokens = line.split('\t')
		# create a dictionary of the info
		info = {} # empty dictionary
		infoTokens = tokens[7].split(';');
		for infoToken in infoTokens:
			index = string.find(infoToken, '=')
			if 0 <= index:
				info[infoToken[0:index]] = infoToken[(index+1):]
		# ref/alt alleles
		ref = tokens[3]
		alt = tokens[4]
        # pick the first one if multiple variant calls exist
        	if re.search(',', alt):
            		alt = alt[0:string.find(alt, ',')]
		# ploidy
		gt = tokens[9][0:string.find(tokens[9], ':')]
		if re.search('0', gt):
			ploidy = 0 # het
		else:
			ploidy = 1 # hom
		# coverage
		coverage = int(info['DP'])
		# variant coverage, allele frequence, and strand frequency
		dp4 = info['DP4']
		dp4Tokens = dp4.split(',')
		varCoverage = int(dp4Tokens[2]) + int(dp4Tokens[3])
		alleleFrequency = float(varCoverage) / float(coverage)
		strandFrequency = float(dp4Tokens[2]) / (float(dp4Tokens[2]) + float(dp4Tokens[3]))
		# variant type
		varType = 2 # SNP by default
		# add
		if re.search('^INDEL', tokens[7]):
			# get the type and indel length
			if len(alt) < len(ref):
				varType = 1 # deletion
				indelLength = len(ref) - len(alt)
			else:
				varType = 0 # insertion
				indelLength = len(alt) - len(ref)
			varInfo.addIndelLength(varType, ploidy, indelLength)
		else:
			varInfo.addSNP(ploidy, ref, alt)
		varInfo.addCoverage(varType, ploidy, coverage)
		varInfo.addVariantCoverage(varType, ploidy, varCoverage)
		varInfo.addAF(varType, ploidy, alleleFrequency)
		varInfo.addStrand(varType, ploidy, strandFrequency)

f.close()

varTypeDict = {0 : 'Insertions', 1 : 'Deletions', 2 : 'SNPs'}
ploidyDict = {0 : 'Het', 1 : 'Hom'} 

# indel length
for varType in range(3):
	for ploidy in range(2):
		# indel length
		if 0 == varType or 1 == varType:
			plot_indel_length(varInfo.indelLengths[varType][ploidy], 0.35, varTypeDict[varType], ploidyDict[ploidy])
			plt.savefig(options.output_dir + "/indelLength.%s.%s.png" % (varTypeDict[varType], ploidyDict[ploidy]))
			plt.close()

# coverage
plot_coverages(varInfo.coverages, varInfo.maxCoverage)
plt.savefig(options.output_dir + "/coverage.png")
plt.close()

# variant coverage
plot_variant_coverages(varInfo.varCoverages, varInfo.maxCoverage)
plt.savefig(options.output_dir + "/variantCoverage.png")
plt.close()

# allele frequency
plot_variant_fractions(varInfo.afs)
plt.savefig(options.output_dir + "/variantFractions.png")
plt.close()

# strand frequency
plot_strand_frequencies(varInfo.strands)
plt.savefig(options.output_dir + "/strandFrequency.png")
plt.close()

# snps
plot_snps(varInfo.snps[0])
plt.savefig(options.output_dir + "/snpInfo.png")
plt.close()
