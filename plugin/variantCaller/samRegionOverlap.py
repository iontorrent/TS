# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import pysam
import sys
import os

class VCFReader(object):
    
    def __init__(self, vcfFile):
        
        self.vcfFile = open(vcfFile, "r")
    
    def NextVCF(self):
        
        return VCF( self.vcfFile.readline().rstrip() )
    
    
    
class VCF(object):
    """
        line of a VCF file
    """
    def __init__(self, vcfLine):
        
        tokens = line.split()
        self.chromosome = tokens[0]
        self.position = int(tokens[1])
        self.variant = tokens[2]
        self.excess = tokens[3]
 
def cigarToString (cigarTuples):
    numToCigarDic = ['M', 'I', 'D', 'N', 'S', 'H', 'P']
    cigarString = ""
    for element in cigarTuples:
        cigarString += str(element[1])
        cigarString += numToCigarDic[element[0]]
    return cigarString

def PileupSam( samfile, chromosome, startPos, stopPos = -1 ):
    
    """
        pass
    """
    
    if stopPos == -1:
        stopPos = startPos
    pileups = []
    for pileupColumn in samfile.pileup(chromosome, int(startPos), int(stopPos)):
            pileups.append( pileupColumn )
    return pileups

def SamFetch( samfile, chromosome, startPos, stopPos = -1):
    """
        fetches reads from a sam region.  only returns each read once
    """
    if stopPos == -1:
        stopPos = startPos
        
    alignedReads =[]
    for alignedRead in samfile.fetch( chromosome, int( startPos ), int( stopPos ) ):
        yield alignedRead
        

def PrintReadName( read ):
    print read.qname

# This is a streaming version, but really have to be careful about
# all the buffers (including ones in the calling program.)
def GetPositionsFromStdin():
    positions = []
    curTag = ""
    for char in sys.stdin.readline():
        if char == "\n":
            if(curTag != ""):
                positions.append( curTag )
            return positions
        if char == ",":
            positions.append( curTag )
            curTag = ""
        else:
            curTag += char
    sys.stdin.flush()
    if curTag:
        positions.append( curTag )
    return positions

# Standard kind of reader, but places all positions into memory
#  Ok, if list isn't really huge.
def GetAllPositionsFromStdin():
    positions = []
    for line in sys.stdin:
        positions.append(line.strip().split(","))
    return positions

# Test of this function:
# /data/antioch/projects/ion_variant_calling/mixed-NA12878-NA19099/samRegionOverlap/tests
# chr1,4038332,4038332  # Complete set of reads
# chr1,4038331,4038332  # No reads
# chr1,4038332,4038351  # No reads, just beyond 19M
# chr1,4038332,4038350  # Just the  19M reads, no 18M
# chr1,4038332,4038349  # All the reads (18M and  19M)

def DoesReadOverLapPositions(read, firstPos, secondPos):
    
    #print "firstPos[%s] >= read.pos[%s]: %s " % (str(firstPos), str(read.pos), str(firstPos >= read.pos) )
    #print "secondPos[%s] <= read.aend[%s]: %s " % (str(secondPos), str(read.aend), str(secondPos <= read.aend) )
    #print "%s %d %d" % (cigarToString(read.cigar), read.pos, read.aend)
    if ( firstPos >= read.pos+1 ) and ( secondPos <= read.aend):
        return True

def PrintSamRecord( read, samStrm ):
    readStr = ""
    strand = ""
    if read.is_reverse:
        strand = "-"
    else:
        strand = "+"
    bamFields = [strand, read.qname] #, read.qname, read.flag, read.rname,read.pos+1,read.mapq,
                 #cigarToString(read.cigar), read.mrnm, read.mpos+1, read.isize, 
                 #read.seq, read.qual]
    #[read.qname, read.flag. read.mapq, cigarToString(read.cigar), read.pos+1, read.aend]
    for field in bamFields:
        readStr += str(field) + "\t"
    print readStr[:-1]
    sys.stdout.flush()

def ReadStdin_OutputSpanningReads(samfile):
    samStrm = pysam.Samfile( samfile, "rb")
    #positions = GetPositionsFromStdin()
    #while positions:
    for position in GetAllPositionsFromStdin():
        # -1 of postion 1 b/c of 0-base, -1 then +1 for pos 2 b/c of 0-base and closed interval
        #print "positions:",
        #print positions
        if len(position) < 3:
            continue
        print "RECORDS-START:%s:%s:%s" % (position[0],position[1],position[2])
        for read in SamFetch(samStrm, position[0], int(position[1])-1, int(position[2])):
            if DoesReadOverLapPositions(read, int(position[1]), int(position[2]) ):
                PrintSamRecord( read, samStrm )
        print "RECORDS-END"
        #std.stdout.flush()
        #positions = GetPositionsFromStdin()
    print "DONE"

if __name__ == '__main__':
    samfile = sys.argv[1]
    ReadStdin_OutputSpanningReads(samfile)
