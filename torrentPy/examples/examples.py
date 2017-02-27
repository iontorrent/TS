# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

import torrentPy
import argparse
import numpy
import os

#WARNING: This file used in unit tests (see ../tests/test_torrentPy.py)
#If you modify examples, please verify/modify unit tests

def read_Wells( file_name ):
    wells=torrentPy.WellsReader(file_name)
    w=wells.LoadWells(0,0,25,25)    #Read a 25x25 block of wells, starting at (row,col)=(0,0)
    max_amplitude = w[20,21,:].max()    #find max amplitude over all flows for well at (20,21)
    print "Max Amplitude: ", max_amplitude
    return locals()


def read_Wells_Flow( file_name ):
    wells=torrentPy.WellsReader(file_name)
    w=wells.LoadWellsFlow(0,0,25,25,0)    #Read a 25x25 block of wells, starting at (row,col)=(0,0), for flow=0
    amplitude = w[20,21]   
    print "Wells amplitude: ", amplitude        
    return locals()


def read_Bam( file_name ):
    bam=torrentPy.BamReader(file_name) #open bam file
    numRecs = bam.GetNumRecords()  #find the number of records. Expensive, so don't do it unless you really need to know.
    bamlist = list(bam)  #bam object is an iterator. read everything into a list.
    numRecs1 = len(bamlist)

    #Move iterator to genomic position
    bam=torrentPy.BamReader(file_name) #open bam file
    bam.Jump(0,10000)
    b=bam.next()    #iterator returns a "dict" with 
    print "All available fields: ", b.keys()
    print "Read position: ", (b['row'],b['col'])
    
    #Restrict to reads from a chip region
    bam.Rewind()    #return iterator to the initial position
    bam.SetChipRegion(10,20,30,40) #Region specified as ( minRow, maxRow, minCol, maxCol )
    bamlist_reg = list(bam)
    
    #Restrict to reads to genomic coordinates
    #Note that ChipRegion restriction is still in effect!
    bam.Rewind()    #return iterator to the initial position
    bam.SetDNARegion(0,0,0,100000) #Region specified as ( leftRefId, leftPosition, rightRefId, rightPosition )
    bamlist_dnareg = list(bam)
    
    #Load a random sample of 10 reads.
    bam.Rewind()
    bam.SetSampleSize(10)
    bamlist_sample = list(bam)
    numRecs2 = len(bamlist_sample)
    
    #Read Bam Header
    bamHeader=bam.header
    print bamHeader
    
    #Obtain phase corrected signal
    keyFlows = numpy.array([1,0,1,0,0,1,0,1])
    bam.PhaseCorrect(bamlist[0],keyFlows)    
    print bamlist[0]['predicted']    
    print bamlist[0]['residual']    
    
    return locals()

def read_Dat( file_name ):
    dat = torrentPy.RawDatReader(file_name)
    d = dat.LoadSlice(10,20,25,20) #Load traces from a region on a chip format (start_row, start_col, height, width )

    dat = torrentPy.RawDatReader(file_name,normalize=False) #Load unnormalized data
    d1 = dat.LoadSlice(10,20,25,20) #Load traces from a region on a chip format (start_row, start_col, height, width )
    
    return locals()

def read_BfMask( file_name ):
    #Loading beadfind mask, Library reads
    bfmask = torrentPy.LoadBfMaskByType(file_name,torrentPy.BfMask.MaskLib)
    return locals()

def read_Debug( dir_name ):
    db = torrentPy.IonDebugData.DebugParams( dir_name )
    db.LoadData()
    pos = ((10,11),(10,12),(22,33)) #specify position as tuple of (row,col) positions
    Flow = 7
    Nuc="G"
    #to get regional parameters we need to know position, flow and nuc
    regParams=db.getBgRegionParams(pos, Flow, Nuc) #regParams are returned as a list of dicts - one entry per position
    beadParams=db.getBeadParams(pos,Flow) #beadParams are returned as dict of numpy arrays - one array entry by position
    print "Regional Params: ", regParams
    print "Bead Params: ", beadParams
    return locals()

def treephaser( dir_name ):
    bamreader = torrentPy.BamReader(dir_name+"/rawlib.bam")
    tp=torrentPy.TreePhaser(bamreader.header['FlowOrder'][0])

    with open(dir_name+"/Calibration.json") as f:
        calib_model = f.read()
        tp.setCalibFromJson(calib_model,4)

    tp.setCAFIEParams(.01,.01,.01)
    
    keySeq = bamreader.header['KeySequence'][0]
    flowOrder = bamreader.header['FlowOrder'][0]
    key=torrentPy.seqToFlow(keySeq,flowOrder[0:8])

    bamreader.SetSampleSize(3)
    bamlist_sample = list(bamreader)
    #apply different basecaller solvers to a list of reads
    bamlist_sample[0].pop('predicted',None)
    tp.treephaserSolveMulti('treephaser',bamlist_sample,key)
    print(bamlist_sample[0]['predicted'][0:10])
    tp.treephaserSolveMulti('treephaserSWAN',bamlist_sample,key)
    print(bamlist_sample[0]['predicted'][0:10])
    tp.treephaserSolveMulti('treephaserDP',bamlist_sample,key)
    print(bamlist_sample[0]['predicted'][0:10])
    tp.treephaserSolveMulti('treephaserAdaptive',bamlist_sample,key)
    print(bamlist_sample[0]['predicted'][0:10])
    tp.treephaserSolveMulti('treephaserDP',bamlist_sample,key)
    print(bamlist_sample[0]['predicted'][0:10])
    
    
    #process single read
    b=bamlist_sample[0]
    tp.treephaserDPSolve(b['meas'],key)    
    print(b['predicted'][0:10])
    tp.treephaserSolve(b['meas'],key)    
    print(b['predicted'][0:10])
    tp.treephaserAdaptiveSolve(b['meas'],key)    
    print(b['predicted'][0:10])
    tp.treephaserSWANSolve(b['meas'],key)    
    print(b['predicted'][0:10])

    
    #Simulate a sequence
    tp.setStateProgression(True)
    res=tp.Simulate("TCAGGTTTACG",60)
    print(res)

    rr=torrentPy.LightFlowAlignment(b['keySeq']+b['qseq_bases'],b['keySeq']+b['tseq_bases'],b['flowOrder'],False,0.1);
    print(rr)
    return locals()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data-dir',help='path to debug data directory',default='.')
    args=parser.parse_args()
    
    r=read_Wells(os.path.join(args.data_dir,'1.wells'))

    ret=read_Bam( os.path.join(args.data_dir,'rawlib.bam') )
    
    ret = read_Dat( os.path.join(args.data_dir,'acq_0000.dat') )
    
    ret = read_BfMask( os.path.join(args.data_dir,'bfmask.bin') )
    
    ret = read_Debug( args.data_dir )
    
    ret = treephaser(args.data_dir)